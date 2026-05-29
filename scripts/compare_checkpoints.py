"""Side-by-side comparison of two checkpoints on the same val images.

Columns:  [grayscale L | Phase A pred | Phase B pred | ground-truth RGB]

Usage:
    uv run python scripts/compare_checkpoints.py \\
        --checkpoint-a checkpoints/resnet_unet_run03/best.pth \\
        --checkpoint-b checkpoints/cgan_run01/best_generator.pth \\
        --output outputs/samples/final_comparison.png
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

from src.data.dataset import ColorizationDataset, lab_to_rgb  # noqa: E402
from src.models.resnet_unet import ResNetUNet  # noqa: E402
from scripts.generate_samples import (  # noqa: E402
    _grayscale_rgb_from_l,
    _hstack_with_separator,
    _vstack_with_separator,
    deterministic_indices,
    pick_device,
)


def _load_generator(path: Path, device: torch.device) -> ResNetUNet:
    model = ResNetUNet(freeze_encoder=True)
    state = torch.load(path, map_location=device, weights_only=False)
    if isinstance(state, dict) and "model_state_dict" in state:
        model.load_state_dict(state["model_state_dict"])
    elif isinstance(state, dict) and "generator_state_dict" in state:
        model.load_state_dict(state["generator_state_dict"])
    else:
        model.load_state_dict(state)
    return model.to(device).eval()


@torch.no_grad()
def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint-a", type=Path, required=True)
    parser.add_argument("--checkpoint-b", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--split", default="val")
    parser.add_argument("--num-samples", type=int, default=8)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    device = pick_device()
    model_a = _load_generator(args.checkpoint_a, device)
    model_b = _load_generator(args.checkpoint_b, device)
    dataset = ColorizationDataset(split=args.split, horizontal_flip=False)
    indices = deterministic_indices(len(dataset), args.num_samples, seed=args.seed)

    rows: list[np.ndarray] = []
    for idx in indices:
        l_tensor, ab_true = dataset[idx]
        l_in = l_tensor.unsqueeze(0).to(device)
        ab_a = model_a(l_in).cpu().squeeze(0).clamp(-1, 1)
        ab_b = model_b(l_in).cpu().squeeze(0).clamp(-1, 1)

        gray = _grayscale_rgb_from_l(l_tensor)
        pred_a = lab_to_rgb(l_tensor, ab_a)
        pred_b = lab_to_rgb(l_tensor, ab_b)
        truth = lab_to_rgb(l_tensor, ab_true)

        rows.append(_hstack_with_separator([gray, pred_a, pred_b, truth]))

    grid = _vstack_with_separator(rows)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(grid).save(args.output, format="PNG", optimize=True)
    print(f"[compare] wrote {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
