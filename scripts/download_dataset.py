"""Download Open Images V7 train images via FiftyOne.

Two modes:
    1. Random sampling (default) — pulls --num-images images using --seed.
    2. Reproducibility mode (--manifest) — downloads exactly the image IDs
       listed in a manifest.csv produced by a prior run of preprocess_images.py.

Usage:
    # First teammate: random sample
    python scripts/download_dataset.py

    # Other teammates: download the exact same set
    python scripts/download_dataset.py --manifest data/processed/manifest.csv
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import fiftyone.zoo as foz

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "data" / "raw"
DEFAULT_NUM_IMAGES = 100_000
DEFAULT_SEED = 42


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--num-images",
        type=int,
        default=DEFAULT_NUM_IMAGES,
        help=f"Number of images to sample (default: {DEFAULT_NUM_IMAGES:,}). Ignored if --manifest is set.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_SEED,
        help=f"Random seed for sampling (default: {DEFAULT_SEED}). Ignored if --manifest is set.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f"Where to store raw images (default: {DEFAULT_OUTPUT_DIR.relative_to(PROJECT_ROOT)})",
    )
    parser.add_argument(
        "--split",
        choices=["train", "validation", "test"],
        default="train",
        help="Open Images split to sample from (default: train)",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=None,
        help="Path to a manifest.csv from a prior preprocess run. When set, "
             "downloads exactly those image IDs for bit-exact reproducibility.",
    )
    return parser.parse_args()


def read_image_ids(manifest_path: Path) -> list[str]:
    with manifest_path.open() as f:
        reader = csv.DictReader(f)
        return [Path(row["filename"]).stem for row in reader]


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    common_kwargs = {
        "split": args.split,
        "label_types": [],
        "dataset_dir": str(args.output_dir),
    }

    if args.manifest is not None:
        image_ids = read_image_ids(args.manifest)
        print(
            f"Reproducibility mode: downloading {len(image_ids):,} images by ID "
            f"from {args.manifest} to {args.output_dir}"
        )
        foz.load_zoo_dataset(
            "open-images-v7",
            image_ids=image_ids,
            **common_kwargs,
        )
    else:
        print(
            f"Random sample mode: downloading {args.num_images:,} images "
            f"(split={args.split}, seed={args.seed}) to {args.output_dir}"
        )
        foz.load_zoo_dataset(
            "open-images-v7",
            max_samples=args.num_images,
            shuffle=True,
            seed=args.seed,
            **common_kwargs,
        )

    print("Done. Run scripts/preprocess_images.py next.")


if __name__ == "__main__":
    main()
