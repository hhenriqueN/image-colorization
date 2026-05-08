"""Preprocess raw Open Images downloads for colorization training.

Steps per image:
    1. Skip images that are effectively grayscale.
    2. Center-crop to square, resize to 256x256, re-encode as JPEG.
    3. Assign to train/val/test (80/10/10) using a deterministic shuffle.

Outputs:
    data/processed/train/<id>.jpg
    data/processed/val/<id>.jpg
    data/processed/test/<id>.jpg
    data/processed/manifest.csv
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import random
from pathlib import Path

import numpy as np
from PIL import Image
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_RAW_DIR = PROJECT_ROOT / "data" / "raw"
DEFAULT_OUT_DIR = PROJECT_ROOT / "data" / "processed"

IMAGE_SIZE = 256
JPEG_QUALITY = 90
GRAYSCALE_THRESHOLD = 5.0  # mean abs channel diff below this => treat as B&W
SPLIT_RATIOS = {"train": 0.8, "val": 0.1, "test": 0.1}
DEFAULT_SEED = 42


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", type=Path, default=DEFAULT_RAW_DIR)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    return parser.parse_args()


def is_grayscale(img: Image.Image) -> bool:
    if img.mode in ("L", "1"):
        return True
    arr = np.asarray(img.convert("RGB"), dtype=np.int16)
    r, g, b = arr[..., 0], arr[..., 1], arr[..., 2]
    channel_diff = (np.abs(r - g) + np.abs(g - b) + np.abs(r - b)).mean() / 3.0
    return channel_diff < GRAYSCALE_THRESHOLD


def square_resize(img: Image.Image, size: int) -> Image.Image:
    w, h = img.size
    side = min(w, h)
    left = (w - side) // 2
    top = (h - side) // 2
    cropped = img.crop((left, top, left + side, top + side))
    return cropped.resize((size, size), Image.Resampling.LANCZOS)


SPLIT_BUCKETS = 100


def assign_split(image_id: str) -> str:
    """Deterministically assign a split based on a hash of the image ID.

    Independent of processing order or grayscale-rejection rate, so the
    final 80/10/10 ratios hold over all kept images, not over the raw count.
    """
    bucket = int(hashlib.sha256(image_id.encode()).hexdigest()[:8], 16) % SPLIT_BUCKETS
    train_cutoff = int(SPLIT_RATIOS["train"] * SPLIT_BUCKETS)
    val_cutoff = train_cutoff + int(SPLIT_RATIOS["val"] * SPLIT_BUCKETS)
    if bucket < train_cutoff:
        return "train"
    if bucket < val_cutoff:
        return "val"
    return "test"


def collect_image_paths(raw_dir: Path) -> list[Path]:
    extensions = ("*.jpg", "*.jpeg", "*.png")
    paths: list[Path] = []
    for ext in extensions:
        paths.extend(raw_dir.rglob(ext))
    return sorted(paths)


def main() -> None:
    args = parse_args()

    raw_paths = collect_image_paths(args.raw_dir)
    if not raw_paths:
        raise SystemExit(f"No images found under {args.raw_dir}. Run download first.")

    rng = random.Random(args.seed)
    rng.shuffle(raw_paths)

    for split in SPLIT_RATIOS:
        (args.out_dir / split).mkdir(parents=True, exist_ok=True)

    manifest_path = args.out_dir / "manifest.csv"
    kept = 0
    skipped_gray = 0
    skipped_error = 0

    with manifest_path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["filename", "split", "original_path", "original_width", "original_height"])

        for src_path in tqdm(raw_paths, desc="Preprocessing"):
            try:
                with Image.open(src_path) as img:
                    img.load()
                    if is_grayscale(img):
                        skipped_gray += 1
                        continue
                    rgb = img.convert("RGB")
                    original_size = rgb.size
                    processed = square_resize(rgb, IMAGE_SIZE)
            except (OSError, ValueError):
                skipped_error += 1
                continue

            out_name = f"{src_path.stem}.jpg"
            split = assign_split(src_path.stem)
            out_path = args.out_dir / split / out_name
            processed.save(out_path, format="JPEG", quality=JPEG_QUALITY)
            writer.writerow([out_name, split, str(src_path), original_size[0], original_size[1]])
            kept += 1

    print(
        f"Kept {kept:,} images, skipped {skipped_gray:,} grayscale, "
        f"{skipped_error:,} unreadable. Manifest at {manifest_path}."
    )


if __name__ == "__main__":
    main()
