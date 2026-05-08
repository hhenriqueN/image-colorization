"""Download Open Images V7 train images from the CVDF S3 mirror.

The mirror is hosted by the Common Visual Data Foundation as a free, public,
non-Requester-Pays S3 bucket. No AWS account or credentials needed.

Two modes:
    1. Random sampling (default) — pulls --num-images images using --seed.
       Requires the master image_ids.csv to be present locally (FiftyOne
       caches it under ~/fiftyone/open-images-v7/<split>/metadata/).
    2. Reproducibility (--manifest) — downloads exactly the image IDs in a
       manifest.csv produced by a prior preprocess run. No metadata CSV needed.

Already-downloaded files in the output dir or in the FiftyOne cache are
detected and migrated/skipped, so this is safe to resume after interruption.

Usage:
    # Canonical (first person), random sample
    python scripts/download_dataset.py

    # Group members reproducing the exact same set
    python scripts/download_dataset.py --manifest data/processed/manifest.csv
"""

from __future__ import annotations

import argparse
import asyncio
import csv
import random
import shutil
import sys
from pathlib import Path

import httpx
from tqdm.asyncio import tqdm_asyncio

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "data" / "raw"
DEFAULT_NUM_IMAGES = 100_000
DEFAULT_SEED = 42
DEFAULT_CONCURRENCY = 32
S3_BASE = "https://open-images-dataset.s3.amazonaws.com"
FIFTYONE_CACHE = Path.home() / "fiftyone" / "open-images-v7"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--num-images", type=int, default=DEFAULT_NUM_IMAGES)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--concurrency", type=int, default=DEFAULT_CONCURRENCY)
    parser.add_argument("--split", choices=["train", "validation", "test"], default="train")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--manifest",
        type=Path,
        default=None,
        help="Path to a manifest.csv from a prior preprocess run; downloads "
             "exactly those image IDs for bit-exact reproducibility.",
    )
    return parser.parse_args()


def migrate_fiftyone_cache(split: str, target_dir: Path) -> int:
    cache_dir = FIFTYONE_CACHE / split / "data"
    if not cache_dir.exists():
        return 0
    target_dir.mkdir(parents=True, exist_ok=True)
    moved = 0
    for src in cache_dir.glob("*.jpg"):
        dst = target_dir / src.name
        if dst.exists():
            continue
        try:
            shutil.move(str(src), str(dst))
            moved += 1
        except OSError:
            pass
    return moved


def load_master_image_ids(split: str) -> list[str]:
    csv_path = FIFTYONE_CACHE / split / "metadata" / "image_ids.csv"
    if not csv_path.exists():
        sys.exit(
            f"Master metadata CSV not found at {csv_path}.\n"
            "Either run the FiftyOne downloader once (which fetches it), or "
            "use --manifest mode if a teammate has already committed manifest.csv."
        )
    with csv_path.open() as f:
        reader = csv.DictReader(f)
        return [row["ImageID"] for row in reader]


def read_ids_from_manifest(manifest_path: Path) -> list[str]:
    with manifest_path.open() as f:
        reader = csv.DictReader(f)
        return [Path(row["filename"]).stem for row in reader]


def determine_target_ids(
    args: argparse.Namespace, existing_ids: set[str]
) -> list[str]:
    if args.manifest is not None:
        return read_ids_from_manifest(args.manifest)

    all_ids = load_master_image_ids(args.split)
    print(f"Loaded {len(all_ids):,} image IDs from master metadata CSV.")

    pool = [iid for iid in all_ids if iid not in existing_ids]
    rng = random.Random(args.seed)
    rng.shuffle(pool)
    needed = max(args.num_images - len(existing_ids), 0)
    new_picks = pool[:needed]
    return list(existing_ids) + new_picks


async def download_one(
    client: httpx.AsyncClient,
    semaphore: asyncio.Semaphore,
    image_id: str,
    split: str,
    output_dir: Path,
) -> bool:
    out_path = output_dir / f"{image_id}.jpg"
    if out_path.exists() and out_path.stat().st_size > 0:
        return True
    url = f"{S3_BASE}/{split}/{image_id}.jpg"
    async with semaphore:
        try:
            response = await client.get(url)
            response.raise_for_status()
            out_path.write_bytes(response.content)
            return True
        except (httpx.HTTPError, OSError):
            return False


async def run_downloads(
    image_ids: list[str], split: str, output_dir: Path, concurrency: int
) -> int:
    timeout = httpx.Timeout(60.0, connect=10.0)
    limits = httpx.Limits(
        max_connections=concurrency * 2,
        max_keepalive_connections=concurrency,
    )
    semaphore = asyncio.Semaphore(concurrency)
    successes = 0
    async with httpx.AsyncClient(timeout=timeout, limits=limits) as client:
        coros = [
            download_one(client, semaphore, iid, split, output_dir)
            for iid in image_ids
        ]
        for fut in tqdm_asyncio.as_completed(coros, total=len(coros), desc="S3 download"):
            if await fut:
                successes += 1
    return successes


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir / args.split
    output_dir.mkdir(parents=True, exist_ok=True)

    moved = migrate_fiftyone_cache(args.split, output_dir)
    if moved:
        print(f"Migrated {moved:,} images from FiftyOne cache to {output_dir}.")

    existing_ids = {p.stem for p in output_dir.glob("*.jpg")}
    target_ids = determine_target_ids(args, existing_ids)
    to_download = [iid for iid in target_ids if iid not in existing_ids]

    print(
        f"Target set: {len(target_ids):,}. Already on disk: "
        f"{len(target_ids) - len(to_download):,}. To fetch: {len(to_download):,}."
    )

    if not to_download:
        print("Nothing to download.")
        return

    successes = asyncio.run(
        run_downloads(to_download, args.split, output_dir, args.concurrency)
    )
    on_disk_after = len(list(output_dir.glob("*.jpg")))
    print(
        f"Downloaded {successes:,} of {len(to_download):,} new images. "
        f"Total in {output_dir}: {on_disk_after:,}."
    )
    print("Done. Run scripts/preprocess_images.py next.")


if __name__ == "__main__":
    main()
