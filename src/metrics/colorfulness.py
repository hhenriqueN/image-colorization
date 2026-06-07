"""Hasler-Süsstrunk colorfulness metric.

Reference: Hasler & Süsstrunk, "Measuring colourfulness in natural images" (2003).

The L1-in-LAB regression objective has its minimum at the per-pixel conditional
mean, which for natural images sits near gray/brown — so tracking only val_l1
during training systematically steers toward desaturated output. Logging
colorfulness alongside L1 surfaces the sepia-attractor problem as a number on
epoch 1 instead of as something noticed by eye after a full run.
"""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np


def hasler_susstrunk(rgb_uint8: np.ndarray) -> float:
    """Compute the Hasler-Süsstrunk colorfulness metric.

    Parameters
    ----------
    rgb_uint8 : np.ndarray, shape (H, W, 3), dtype uint8, values 0-255.

    Returns
    -------
    float : colorfulness M; higher = more colorful. Typical natural-image range
        is roughly 0 (pure gray) to ~100 (very saturated).
    """
    if rgb_uint8.ndim != 3 or rgb_uint8.shape[2] != 3:
        raise ValueError(f"rgb_uint8 must be (H, W, 3); got {rgb_uint8.shape}")

    rgb = rgb_uint8.astype(np.float64)
    R, G, B = rgb[..., 0], rgb[..., 1], rgb[..., 2]
    rg = R - G
    yb = 0.5 * (R + G) - B

    std_root = float(np.sqrt(rg.std() ** 2 + yb.std() ** 2))
    mean_root = float(np.sqrt(rg.mean() ** 2 + yb.mean() ** 2))
    return std_root + 0.3 * mean_root


def batch_colorfulness(rgbs: Iterable[np.ndarray]) -> float:
    """Mean colorfulness across an iterable of HWC uint8 RGB images."""
    values = [hasler_susstrunk(rgb) for rgb in rgbs]
    if not values:
        return 0.0
    return float(np.mean(values))
