"""PyTorch Dataset for colorization training.

Reads a manifest produced by scripts/preprocess_images.py and yields
(L, ab) tensor pairs in LAB color space, both normalized to [-1, 1].
"""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd
import torch
from PIL import Image
from skimage import color
from torch.utils.data import Dataset

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

L_MAX = 100.0
AB_MAX = 128.0

Split = Literal["train", "val", "test"]


class ColorizationDataset(Dataset):
    """Returns (L, ab) tensor pairs for grayscale-to-color training.

    L:  shape (1, H, W), normalized to [-1, 1]
    ab: shape (2, H, W), normalized to [-1, 1]
    """

    def __init__(
        self,
        split: Split,
        processed_dir: Path = DEFAULT_PROCESSED_DIR,
        horizontal_flip: bool = True,
        filenames: Iterable[str] | None = None,
    ) -> None:
        """
        filenames: optional iterable of manifest filenames to restrict this
        dataset to a subset (e.g. when training on a fraction of the data).
        Filenames not present in the manifest for the given split are ignored.
        """
        manifest = pd.read_csv(processed_dir / "manifest.csv")
        entries = manifest[manifest["split"] == split]
        if filenames is not None:
            wanted = set(filenames)
            entries = entries[entries["filename"].isin(wanted)]
        self.entries = entries.reset_index(drop=True)
        self.split_dir = processed_dir / split
        self.horizontal_flip = horizontal_flip and split == "train"

        if len(self.entries) == 0:
            raise ValueError(
                f"No entries for split '{split}' in {processed_dir}/manifest.csv"
                + (" matching the given filenames" if filenames is not None else "")
            )

    def __len__(self) -> int:
        return len(self.entries)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        filename = self.entries.iloc[index]["filename"]
        image_path = self.split_dir / filename

        with Image.open(image_path) as img:
            rgb = np.asarray(img.convert("RGB"), dtype=np.float32) / 255.0

        if self.horizontal_flip and torch.rand(1).item() < 0.5:
            rgb = np.ascontiguousarray(rgb[:, ::-1, :])

        lab = color.rgb2lab(rgb).astype(np.float32)

        l_channel = lab[..., 0:1] / L_MAX * 2.0 - 1.0
        ab_channels = lab[..., 1:3] / AB_MAX

        l_tensor = torch.from_numpy(l_channel).permute(2, 0, 1)
        ab_tensor = torch.from_numpy(ab_channels).permute(2, 0, 1)
        return l_tensor, ab_tensor


def lab_to_rgb(l_tensor: torch.Tensor, ab_tensor: torch.Tensor) -> np.ndarray:
    """Inverse of the dataset normalization: take a (L, ab) pair back to an RGB array.

    Accepts tensors of shape (1, H, W) and (2, H, W), or batched (N, C, H, W).
    Returns a uint8 numpy array of shape (H, W, 3) or (N, H, W, 3).
    """
    if l_tensor.dim() == 3:
        l_tensor = l_tensor.unsqueeze(0)
        ab_tensor = ab_tensor.unsqueeze(0)
        squeeze = True
    else:
        squeeze = False

    l = (l_tensor + 1.0) / 2.0 * L_MAX
    ab = ab_tensor * AB_MAX
    lab = torch.cat([l, ab], dim=1).permute(0, 2, 3, 1).cpu().numpy()

    rgb_batch = np.stack([color.lab2rgb(image) for image in lab])
    rgb_batch = np.clip(rgb_batch * 255.0, 0, 255).astype(np.uint8)
    return rgb_batch[0] if squeeze else rgb_batch
