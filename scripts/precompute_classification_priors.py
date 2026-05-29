"""Compute the ab gamut and class rebalance weights for Zhang-style training.

Samples N random training images, accumulates the empirical distribution of ab
pixel values into a 10×10 LAB-unit grid, drops bins below a minimum population
threshold, and writes:

  - data/processed/ab_bin_centers.npy       (Q, 2)  float32, LAB units
  - data/processed/ab_rebalance_weights.npy (Q,)    float32
  - data/processed/bin_priors.png                   gamut + density heatmap

Usage:
    uv run python scripts/precompute_classification_priors.py
    uv run python scripts/precompute_classification_priors.py --sample-size 500
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.dataset import DEFAULT_PROCESSED_DIR, AB_MAX, ColorizationDataset  # noqa: E402
from src.data.quantize import compute_bin_centers, compute_rebalance_weights  # noqa: E402


def sample_train_filenames(n: int, seed: int) -> list[str]:
    import pandas as pd

    manifest = pd.read_csv(DEFAULT_PROCESSED_DIR / "manifest.csv")
    train = manifest[manifest["split"] == "train"]
    if len(train) <= n:
        return train["filename"].tolist()
    rng = np.random.default_rng(seed)
    chosen = rng.choice(train["filename"].to_numpy(), size=n, replace=False)
    return sorted(chosen.tolist())


def save_bin_heatmap(
    centers: np.ndarray,
    counts: np.ndarray,
    grid_step: float,
    ab_range: tuple[float, float],
    out_path: Path,
) -> None:
    a_min, a_max = ab_range
    n_cells = int(round((a_max - a_min) / grid_step))
    density = np.full((n_cells, n_cells), np.nan, dtype=np.float64)
    a_idx = ((centers[:, 0] - a_min - grid_step / 2) / grid_step).round().astype(np.int64)
    b_idx = ((centers[:, 1] - a_min - grid_step / 2) / grid_step).round().astype(np.int64)
    density[a_idx, b_idx] = counts
    # Log scale so we can see both common and rare bins.
    log_density = np.log10(np.where(np.isnan(density), np.nan, np.maximum(density, 1)))

    fig, ax = plt.subplots(figsize=(7, 7))
    im = ax.imshow(
        log_density.T,
        origin="lower",
        extent=(a_min, a_max, a_min, a_max),
        cmap="magma",
        interpolation="nearest",
    )
    ax.scatter(centers[:, 0], centers[:, 1], s=4, c="cyan", alpha=0.4, label="bin center")
    ax.set_xlabel("a (green ↔ red)")
    ax.set_ylabel("b (blue ↔ yellow)")
    ax.set_title(f"ab gamut, {len(centers)} bins kept (log10 pixel count)")
    plt.colorbar(im, ax=ax, shrink=0.8)
    ax.legend(loc="lower right")
    plt.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample-size", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--grid-step", type=float, default=10.0)
    parser.add_argument("--min-fraction", type=float, default=5e-4,
                        help="Drop bins with population below this fraction of total pixels.")
    parser.add_argument("--lambda", type=float, default=0.5, dest="lambda_",
                        help="Rebalancing smoothing (Zhang's λ).")
    parser.add_argument("--batch-size", type=int, default=16)
    args = parser.parse_args()

    out_dir = DEFAULT_PROCESSED_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[priors] sampling {args.sample_size} train filenames (seed {args.seed})")
    files = sample_train_filenames(args.sample_size, args.seed)
    dataset = ColorizationDataset(split="train", filenames=files, horizontal_flip=False)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)

    print(f"[priors] streaming ab pixels from {len(dataset)} images at batch={args.batch_size}")
    # Accumulate raw pixels in LAB units. (N_images * 256 * 256 * 2 floats; at
    # 5000 images that's 2.5 GB of float32. Stream-bin instead of keeping all.)
    ab_range = (-110.0, 110.0)
    grid_step = args.grid_step
    n_cells = int(round((ab_range[1] - ab_range[0]) / grid_step))
    cell_counts = torch.zeros(n_cells * n_cells, dtype=torch.int64)
    t0 = time.perf_counter()
    seen_pixels = 0
    for batch_idx, (_, ab) in enumerate(loader):
        ab_lab = (ab * AB_MAX).reshape(-1, 2)
        in_range = (
            (ab_lab[:, 0] >= ab_range[0]) & (ab_lab[:, 0] < ab_range[1]) &
            (ab_lab[:, 1] >= ab_range[0]) & (ab_lab[:, 1] < ab_range[1])
        )
        ab_lab = ab_lab[in_range]
        a_idx = ((ab_lab[:, 0] - ab_range[0]) / grid_step).to(torch.long).clamp_(0, n_cells - 1)
        b_idx = ((ab_lab[:, 1] - ab_range[0]) / grid_step).to(torch.long).clamp_(0, n_cells - 1)
        cell_id = a_idx * n_cells + b_idx
        cell_counts.scatter_add_(0, cell_id, torch.ones_like(cell_id))
        seen_pixels += ab_lab.shape[0]
        if (batch_idx + 1) % 20 == 0:
            print(
                f"[priors]   batch {batch_idx + 1}/{len(loader)}  "
                f"pixels={seen_pixels:,}  "
                f"elapsed={time.perf_counter() - t0:.1f}s",
                flush=True,
            )

    total = cell_counts.sum().item()
    threshold = max(1, int(total * args.min_fraction))
    keep = cell_counts >= threshold
    n_kept = int(keep.sum().item())
    print(f"[priors] total pixels = {total:,}; keeping {n_kept}/{n_cells * n_cells} bins "
          f"(≥ {threshold:,} pixels each, {100 * args.min_fraction:.3f}% threshold)")

    cell_ids = torch.arange(n_cells * n_cells)
    kept_ids = cell_ids[keep]
    a_grid = (kept_ids // n_cells).to(torch.float32) * grid_step + ab_range[0] + grid_step / 2
    b_grid = (kept_ids %  n_cells).to(torch.float32) * grid_step + ab_range[0] + grid_step / 2
    centers = torch.stack([a_grid, b_grid], dim=1).numpy().astype(np.float32)
    counts = cell_counts[keep].to(torch.float32)
    weights = compute_rebalance_weights(counts, lambda_=args.lambda_).numpy().astype(np.float32)

    centers_path = out_dir / "ab_bin_centers.npy"
    weights_path = out_dir / "ab_rebalance_weights.npy"
    heatmap_path = out_dir / "bin_priors.png"
    np.save(centers_path, centers)
    np.save(weights_path, weights)
    save_bin_heatmap(centers, counts.numpy(), grid_step, ab_range, heatmap_path)

    print(f"[priors] wrote {centers_path.relative_to(PROJECT_ROOT)}  shape={centers.shape}")
    print(f"[priors] wrote {weights_path.relative_to(PROJECT_ROOT)}  shape={weights.shape}  "
          f"min={weights.min():.3f}  max={weights.max():.3f}  mean={weights.mean():.3f}")
    print(f"[priors] wrote {heatmap_path.relative_to(PROJECT_ROOT)}")
    print(f"[priors] done in {time.perf_counter() - t0:.1f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
