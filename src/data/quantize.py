"""ab-space quantization helpers for Zhang-style classification colorization.

The colorization task is reframed as per-pixel classification over a small set
of ab-color bins (Q ≈ 313). This file provides the bin lookup, rebalancing
weight computation, and annealed-mean inference used by the classification
training mode.

Conventions
-----------
- `bin_centers` is shape (Q, 2) in LAB units (a, b ∈ [-110, 110]).
- `ab` tensors fed in here are in the dataset's **normalized** range [-1, 1].
  We multiply by AB_MAX (=128) before nearest-bin lookup so we work in real
  LAB units; outputs of `annealed_mean` are in LAB units and the caller
  divides by 128 to get back to [-1, 1].
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F

AB_MAX = 128.0


def compute_bin_centers(
    ab_samples: torch.Tensor,
    grid_step: float = 10.0,
    min_fraction: float = 5e-4,
    ab_range: tuple[float, float] = (-110.0, 110.0),
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute the valid ab-gamut bin centers and per-bin counts.

    Parameters
    ----------
    ab_samples : Tensor (..., 2) in LAB units (NOT normalized).
    grid_step  : width of each grid cell in LAB units (Zhang uses 10).
    min_fraction : drop bins whose population is below this fraction of total pixels.
    ab_range   : inclusive lower / exclusive upper bound on a and b values.

    Returns
    -------
    centers : Tensor (Q, 2) of bin centers (LAB units).
    counts  : Tensor (Q,) of per-bin pixel counts (matches centers row-by-row).
    """
    flat = ab_samples.reshape(-1, 2).to(torch.float32)
    a_min, a_max = ab_range
    in_range = (
        (flat[:, 0] >= a_min) & (flat[:, 0] < a_max) &
        (flat[:, 1] >= a_min) & (flat[:, 1] < a_max)
    )
    flat = flat[in_range]

    n_cells_per_axis = int(round((a_max - a_min) / grid_step))
    # Integer cell indices in [0, n_cells_per_axis)
    a_idx = ((flat[:, 0] - a_min) / grid_step).to(torch.long).clamp_(0, n_cells_per_axis - 1)
    b_idx = ((flat[:, 1] - a_min) / grid_step).to(torch.long).clamp_(0, n_cells_per_axis - 1)
    cell_id = a_idx * n_cells_per_axis + b_idx

    counts_full = torch.bincount(cell_id, minlength=n_cells_per_axis * n_cells_per_axis)
    total = counts_full.sum().item()
    if total == 0:
        raise ValueError("No ab samples fell inside the configured ab_range.")

    threshold = max(1, int(total * min_fraction))
    keep = counts_full >= threshold

    cell_ids = torch.arange(n_cells_per_axis * n_cells_per_axis)
    kept_ids = cell_ids[keep]
    a_grid = (kept_ids // n_cells_per_axis).to(torch.float32) * grid_step + a_min + grid_step / 2
    b_grid = (kept_ids %  n_cells_per_axis).to(torch.float32) * grid_step + a_min + grid_step / 2
    centers = torch.stack([a_grid, b_grid], dim=1)
    counts = counts_full[keep].to(torch.float32)
    return centers, counts


def compute_rebalance_weights(
    counts: torch.Tensor,
    lambda_: float = 0.5,
) -> torch.Tensor:
    """Zhang et al. class-rebalancing weights.

    w(q) ∝ 1 / ((1 - λ) * p̂(q) + λ / Q)
    Normalized so E_q[w(q)] = 1 under the empirical prior p̂.

    Parameters
    ----------
    counts  : (Q,) per-bin pixel counts.
    lambda_ : smoothing toward uniform (Zhang's default 0.5).

    Returns
    -------
    weights : (Q,) float32 weights, mean-1 under the empirical prior.
    """
    counts = counts.to(torch.float32)
    p_hat = counts / counts.sum()
    Q = counts.numel()
    w = 1.0 / ((1.0 - lambda_) * p_hat + lambda_ / Q)
    # Normalize so weighted prior sums to 1.
    w = w / (w * p_hat).sum()
    return w


def ab_to_class(
    ab_normalized: torch.Tensor,
    bin_centers: torch.Tensor,
) -> torch.Tensor:
    """Vectorized nearest-bin lookup for a batch of ab pixels.

    Parameters
    ----------
    ab_normalized : Tensor (N, 2, H, W) in normalized [-1, 1] range.
    bin_centers   : Tensor (Q, 2) in LAB units.

    Returns
    -------
    class_idx : LongTensor (N, H, W) of nearest-bin indices.
    """
    if ab_normalized.dim() != 4 or ab_normalized.shape[1] != 2:
        raise ValueError(f"ab_normalized must be (N, 2, H, W); got {tuple(ab_normalized.shape)}")
    ab_lab = ab_normalized * AB_MAX  # → LAB units
    bin_centers = bin_centers.to(ab_normalized.device, ab_normalized.dtype)

    # Flatten spatial dims to (N*H*W, 2) for the per-pixel argmin.
    n, _, h, w = ab_lab.shape
    pix = ab_lab.permute(0, 2, 3, 1).reshape(-1, 2)            # (M, 2)
    # Squared L2 distance: ||x||² + ||c||² - 2 x · c. Compute the last two
    # since ||x||² is constant across bins for a given pixel.
    cross = pix @ bin_centers.t()                              # (M, Q)
    centers_sq = (bin_centers * bin_centers).sum(dim=1)        # (Q,)
    neg_sq_dist = 2.0 * cross - centers_sq                      # higher = closer
    class_idx = neg_sq_dist.argmax(dim=1).to(torch.long)        # (M,)
    return class_idx.reshape(n, h, w)


def annealed_mean(
    logits: torch.Tensor,
    bin_centers: torch.Tensor,
    temperature: float = 0.38,
) -> torch.Tensor:
    """Convert per-pixel class logits to ab predictions via Zhang's annealed-mean.

    A temperature-scaled softmax over the class probabilities is weighted by
    bin centers. T → 0 approaches argmax (vivid but patchy); T → ∞ approaches
    the mean (gray). Zhang's tuned default is 0.38.

    Parameters
    ----------
    logits      : Tensor (N, Q, H, W).
    bin_centers : Tensor (Q, 2) in LAB units.
    temperature : softmax temperature.

    Returns
    -------
    ab_lab : Tensor (N, 2, H, W) in LAB units.
    """
    log_probs = F.log_softmax(logits, dim=1)
    probs = F.softmax(log_probs / temperature, dim=1)
    bin_centers = bin_centers.to(logits.device, logits.dtype)
    # einsum: probs (n, q, h, w) × centers (q, c) → (n, c, h, w)
    return torch.einsum("nqhw,qc->nchw", probs, bin_centers)


def bilateral_smooth_ab(
    ab_lab: torch.Tensor,
    diameter: int = 9,
    sigma_color: float = 15.0,
    sigma_space: float = 9.0,
) -> torch.Tensor:
    """Bilateral post-processing on classification-derived ab channels.

    At low annealed-mean temperatures Phase C produces vivid but spatially
    "patchy" output because adjacent pixels jump between discrete bins. A
    bilateral filter smooths within similar-color regions while preserving
    edges, which is exactly the right tool for this discrete-bin artifact.

    Runs on CPU via OpenCV. For our batch sizes (8-16) and image size
    (256×256) this is fast (<100 ms per image). The filter is applied per
    channel and per image; we do NOT use the L channel as a guide because
    cv2.bilateralFilter is single-image. For a joint-bilateral variant
    that uses L as the guide we'd need opencv-contrib's ximgproc, which
    isn't bundled with the cv2 wheel we have.

    Parameters
    ----------
    ab_lab      : Tensor (N, 2, H, W) in LAB units (any device).
    diameter    : Pixel neighborhood diameter. 9 covers ~4 pixel radius.
    sigma_color : Filter sigma in LAB color space. Bin width is 5, so 15
                  averages roughly across 3 adjacent bins — enough to kill
                  inter-bin jitter without bleeding into different colors.
    sigma_space : Spatial sigma in pixels. Smaller = tighter smoothing.

    Returns
    -------
    ab_lab_smoothed : Tensor (N, 2, H, W) on the same device/dtype as input.
    """
    import cv2  # local import: optional dep; only required for this path

    if ab_lab.dim() != 4 or ab_lab.shape[1] != 2:
        raise ValueError(f"ab_lab must be (N, 2, H, W); got {tuple(ab_lab.shape)}")
    device, dtype = ab_lab.device, ab_lab.dtype
    ab_np = ab_lab.detach().cpu().to(torch.float32).numpy()  # (N, 2, H, W)

    smoothed = np.empty_like(ab_np)
    for n in range(ab_np.shape[0]):
        for c in range(2):
            smoothed[n, c] = cv2.bilateralFilter(
                ab_np[n, c], diameter, sigma_color, sigma_space
            )
    return torch.from_numpy(smoothed).to(device=device, dtype=dtype)
