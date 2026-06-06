"""Qualitative sample grid generator for a trained colorizer.

Loads a checkpoint, picks N deterministic images from a chosen split,
generates a side-by-side grid `[grayscale L | predicted RGB | ground-truth RGB]`,
and writes a PNG. Used both as a CLI and imported from scripts/train.py.

Supports both regression checkpoints (ResNetUNet — direct ab output) and
classification checkpoints (ResNetUNetClassifier — logits converted via
annealed-mean to ab).

Usage:
    uv run python scripts/generate_samples.py \\
        --checkpoint checkpoints/resnet_unet_run01/best.pth \\
        --output outputs/samples/resnet_unet_run01/epoch_05.png

    uv run python scripts/generate_samples.py \\
        --checkpoint checkpoints/cls_run01/best.pth \\
        --output outputs/samples/cls_run01/best.png \\
        --model-kind classification
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import numpy as np
import torch
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.dataset import AB_MAX, DEFAULT_PROCESSED_DIR, ColorizationDataset, lab_to_rgb  # noqa: E402
from src.data.quantize import annealed_mean, bilateral_smooth_ab  # noqa: E402
from src.models.resnet_unet import ResNetUNet  # noqa: E402
from src.models.resnet_unet_cls import ResNetUNetClassifier  # noqa: E402


def pick_device(requested: str | None = None) -> torch.device:
    if requested:
        return torch.device(requested)
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def _load_regression_model(checkpoint_path: Path, device: torch.device) -> ResNetUNet:
    model = ResNetUNet(freeze_encoder=True)
    state = torch.load(checkpoint_path, map_location=device, weights_only=False)
    if isinstance(state, dict):
        if "model_state_dict" in state:
            state = state["model_state_dict"]
        elif "generator_state_dict" in state:
            state = state["generator_state_dict"]
    model.load_state_dict(state)
    return model.to(device).eval()


def _load_classification_model(
    checkpoint_path: Path, device: torch.device
) -> tuple[ResNetUNetClassifier, int]:
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    num_classes = int(ckpt.get("num_classes")) if "num_classes" in ckpt else None
    if num_classes is None:
        # Infer from state dict shape (out_conv.weight shape (in_ch, Q, k, k)).
        state = ckpt.get("model_state_dict", ckpt)
        weight = state["out_conv.weight"]
        num_classes = int(weight.shape[1])
    model = ResNetUNetClassifier(num_classes=num_classes, freeze_encoder=True)
    state = ckpt.get("model_state_dict", ckpt)
    model.load_state_dict(state)
    return model.to(device).eval(), num_classes


def _load_bin_centers(path: Path, device: torch.device) -> torch.Tensor:
    return torch.from_numpy(np.load(path)).to(torch.float32).to(device)


def _grayscale_rgb_from_l(l_tensor: torch.Tensor) -> np.ndarray:
    """Render the L channel as a 3-channel grayscale image (uint8 H,W,3)."""
    gray = ((l_tensor.squeeze(0) + 1.0) / 2.0 * 255.0).clamp(0, 255).cpu().numpy()
    gray_u8 = gray.astype(np.uint8)
    return np.stack([gray_u8, gray_u8, gray_u8], axis=-1)


def _hstack_with_separator(panels: list[np.ndarray], sep: int = 4) -> np.ndarray:
    h = panels[0].shape[0]
    separator = np.full((h, sep, 3), 255, dtype=np.uint8)
    out = panels[0]
    for p in panels[1:]:
        out = np.concatenate([out, separator, p], axis=1)
    return out


def _vstack_with_separator(rows: list[np.ndarray], sep: int = 4) -> np.ndarray:
    w = rows[0].shape[1]
    separator = np.full((sep, w, 3), 255, dtype=np.uint8)
    out = rows[0]
    for r in rows[1:]:
        out = np.concatenate([out, separator, r], axis=0)
    return out


@torch.no_grad()
def build_sample_grid(
    model: ResNetUNet,
    dataset: ColorizationDataset,
    indices: list[int],
    device: torch.device,
    *,
    bin_centers: torch.Tensor | None = None,
    temperature: float = 0.38,
    bilateral: bool = False,
    bilateral_d: int = 9,
    bilateral_sigma_color: float = 15.0,
    bilateral_sigma_space: float = 9.0,
) -> np.ndarray:
    """Build a 3-column grid (gray | predicted | truth).

    If `bin_centers` is provided, the model is treated as a classifier and the
    logits are converted to ab via annealed-mean. `bilateral=True` additionally
    smooths the per-image ab output with `bilateral_smooth_ab` — useful when
    using low temperature (T ≲ 0.15) where discrete-bin patchiness shows up.
    """
    model.eval()
    is_classifier = bin_centers is not None
    rows: list[np.ndarray] = []
    for idx in indices:
        l_tensor, ab_true = dataset[idx]
        l_in = l_tensor.unsqueeze(0).to(device)
        out = model(l_in)
        if is_classifier:
            ab_lab = annealed_mean(out, bin_centers, temperature)  # (1, 2, H, W) in LAB units
            if bilateral:
                ab_lab = bilateral_smooth_ab(
                    ab_lab,
                    diameter=bilateral_d,
                    sigma_color=bilateral_sigma_color,
                    sigma_space=bilateral_sigma_space,
                )
            ab_pred = (ab_lab / AB_MAX).cpu().squeeze(0).clamp(-1, 1)
        else:
            ab_pred = out.cpu().squeeze(0).clamp(-1, 1)

        gray_panel = _grayscale_rgb_from_l(l_tensor)
        pred_panel = lab_to_rgb(l_tensor, ab_pred)
        true_panel = lab_to_rgb(l_tensor, ab_true)

        row = _hstack_with_separator([gray_panel, pred_panel, true_panel])
        rows.append(row)

    return _vstack_with_separator(rows)


def deterministic_indices(dataset_size: int, n: int, seed: int = 0) -> list[int]:
    if dataset_size <= n:
        return list(range(dataset_size))
    rng = np.random.default_rng(seed)
    return sorted(rng.choice(dataset_size, size=n, replace=False).tolist())


def generate(
    checkpoint: Path,
    output_path: Path,
    split: str = "val",
    num_samples: int = 8,
    seed: int = 0,
    device_str: str | None = None,
    model_kind: str = "regression",
    bin_centers_path: Path | None = None,
    temperature: float = 0.38,
    bilateral: bool = False,
    bilateral_d: int = 9,
    bilateral_sigma_color: float = 15.0,
    bilateral_sigma_space: float = 9.0,
) -> Path:
    device = pick_device(device_str)
    dataset = ColorizationDataset(split=split, horizontal_flip=False)
    idxs = deterministic_indices(len(dataset), num_samples, seed=seed)

    if model_kind == "classification":
        model, _ = _load_classification_model(checkpoint, device)
        bin_centers = _load_bin_centers(
            bin_centers_path or DEFAULT_PROCESSED_DIR / "ab_bin_centers.npy", device
        )
        grid = build_sample_grid(
            model, dataset, idxs, device,
            bin_centers=bin_centers,
            temperature=temperature,
            bilateral=bilateral,
            bilateral_d=bilateral_d,
            bilateral_sigma_color=bilateral_sigma_color,
            bilateral_sigma_space=bilateral_sigma_space,
        )
    else:
        model = _load_regression_model(checkpoint, device)
        grid = build_sample_grid(model, dataset, idxs, device)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(grid).save(output_path, format="PNG", optimize=True)
    return output_path


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate qualitative sample grid")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--split", default="val", choices=["train", "val", "test"])
    parser.add_argument("--num-samples", type=int, default=8)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default=None, help="cpu | mps | cuda")
    parser.add_argument("--model-kind", default="regression",
                        choices=["regression", "classification"])
    parser.add_argument("--bin-centers", type=Path,
                        default=DEFAULT_PROCESSED_DIR / "ab_bin_centers.npy")
    parser.add_argument("--temperature", type=float, default=0.38)
    parser.add_argument("--bilateral", action="store_true",
                        help="Apply cv2.bilateralFilter on ab channels after annealed-mean.")
    parser.add_argument("--bilateral-d", type=int, default=9)
    parser.add_argument("--bilateral-sigma-color", type=float, default=15.0)
    parser.add_argument("--bilateral-sigma-space", type=float, default=9.0)
    args = parser.parse_args()

    out = generate(
        checkpoint=args.checkpoint,
        output_path=args.output,
        split=args.split,
        num_samples=args.num_samples,
        seed=args.seed,
        device_str=args.device,
        model_kind=args.model_kind,
        bin_centers_path=args.bin_centers,
        temperature=args.temperature,
        bilateral=args.bilateral,
        bilateral_d=args.bilateral_d,
        bilateral_sigma_color=args.bilateral_sigma_color,
        bilateral_sigma_space=args.bilateral_sigma_space,
    )
    print(f"[samples] wrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
