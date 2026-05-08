# image-colorization

Deep learning project: train a neural network to colorize black and white images. Predict the `a`, `b` chrominance channels (LAB color space) from the `L` luminance channel using a convolutional encoder–decoder, working at 256×256 resolution.

For a deeper overview (color space, model family, layout) see [`CLAUDE.md`](./CLAUDE.md).

## Quick start (teammates)

If you're cloning this repo to get the dataset and start training, run these four commands:

```bash
# 1. Install uv (if you don't have it)
curl -LsSf https://astral.sh/uv/install.sh | sh

# 2. Install Python deps into a local .venv
uv sync

# 3. Download the exact images everyone uses (~30 min, 29 GB)
uv run python scripts/download_dataset.py --manifest data/processed/manifest.csv

# 4. Resize to 256x256 and split into train/val/test (~10–15 min, +2 GB)
uv run python scripts/preprocess_images.py
```

That's it. After step 4 you'll have:

- `data/processed/train/` — 74,919 JPEGs (80%)
- `data/processed/val/` — 9,222 JPEGs (10%)
- `data/processed/test/` — 9,321 JPEGs (10%)

The split assignment is hash-based on the image ID, so every teammate ends up with **the exact same images in the exact same splits** — bit-exact reproducibility.

> ⚠️ **Don't commit your local `data/processed/manifest.csv`.** Running preprocess locally rewrites it with absolute paths from your machine. Just leave those changes uncommitted (`git restore data/processed/manifest.csv` if it ever gets staged).

## Disk space you'll need

| Folder | Size | What it is |
|---|---|---|
| `data/raw/train/` | ~29 GB | original-resolution downloads (you can `rm -rf` this after step 4) |
| `data/processed/` | ~2 GB | 256×256 JPEGs ready for training |

Once preprocessing succeeds, you can delete `data/raw/` to reclaim 29 GB:

```bash
rm -rf data/raw/
```

If you ever need it back, just re-run step 3.

## Project layout

```
image-colorization/
├── data/                          # gitignored (except manifest.csv)
│   ├── raw/                       # original-resolution Open Images
│   └── processed/
│       ├── train/  val/  test/    # 256x256 JPEGs
│       └── manifest.csv           # tracked in git for reproducibility
├── scripts/
│   ├── download_dataset.py        # async S3 downloader (manifest-pinned or random)
│   └── preprocess_images.py       # filter B&W, resize, split
├── src/
│   └── data/
│       └── dataset.py             # ColorizationDataset (RGB→LAB)
├── pyproject.toml                 # deps + project config (uv-managed)
└── .python-version                # pinned to 3.12
```

## How the dataset is sourced

100K random sample from [Open Images V7](https://storage.googleapis.com/openimages/web/index.html) (CC BY 2.0, free for ML use). Images are pulled from the [CVDF S3 mirror](https://github.com/cvdfoundation/open-images-dataset) — a free, public, no-account-required bucket hosted by a non-profit. The downloader uses async httpx with concurrency 32 (configurable via `--concurrency`).

After download, ~6% are filtered out as grayscale (we want only color training data), leaving ~93K images that get resized and split.

## Setting a fresh canonical dataset (rare)

You only need this if `data/processed/manifest.csv` doesn't exist in the repo, e.g. when bootstrapping the project for a new course/semester:

```bash
uv run python scripts/download_dataset.py        # random sample, seed 42
uv run python scripts/preprocess_images.py
git add data/processed/manifest.csv
git commit -m "chore: pin dataset manifest"
git push
```

`download_dataset.py` accepts `--num-images`, `--seed`, `--concurrency`, and `--output-dir` — run with `--help` for details.
