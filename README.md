# image-colorization

Deep learning project: train a neural network to colorize black and white images. Predict the `a`, `b` chrominance channels (LAB color space) from the `L` luminance channel using a convolutional encoder–decoder, working at 256×256 resolution.

For a deeper overview of the approach (color space, model family, layout) see [`CLAUDE.md`](./CLAUDE.md).

## Setup

You need [`uv`](https://docs.astral.sh/uv/) for dependency management. If you don't have it:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Then, from the project root:

```bash
uv sync
```

This creates a `.venv/` and installs all dependencies pinned by `uv.lock`. Run scripts with `uv run python ...` and they'll use the project's environment automatically.

## Downloading the dataset

We train on a 100K-image subset of [Open Images V7](https://storage.googleapis.com/openimages/web/index.html) (CC BY 2.0, free for ML use). Images come from the [CVDF S3 mirror](https://github.com/cvdfoundation/open-images-dataset) — a free, public, no-account-required bucket hosted by a non-profit. The downloader uses async httpx with high concurrency, so 100K images typically finishes in under an hour.

The download script has two modes — pick the one that matches your situation.

### Group members: download the exact pinned dataset

If `data/processed/manifest.csv` is already committed in the repo, use it to download the **same images** as everyone else (bit-exact reproducibility):

```bash
uv run python scripts/download_dataset.py --manifest data/processed/manifest.csv
uv run python scripts/preprocess_images.py
```

This will:
1. Download ~100K specific Open Images by ID into `data/raw/` (~1.5 GB).
2. Filter out grayscale photos, center-crop and resize to 256×256, write JPEGs into `data/processed/{train,val,test}/` and rewrite `manifest.csv`.

After preprocessing your local `manifest.csv` should match the committed one — same train/val/test assignments for every teammate.

### Setting the canonical dataset (first person only)

If `data/processed/manifest.csv` doesn't exist yet, sample a fresh 100K and commit the manifest so the rest of the team can reproduce it:

```bash
uv run python scripts/download_dataset.py
uv run python scripts/preprocess_images.py
git add data/processed/manifest.csv
git commit -m "chore: pin dataset manifest"
git push
```

The download script accepts `--num-images`, `--seed`, and `--output-dir` — see `--help` for details.

## Project layout

```
image-colorization/
├── data/                          # gitignored (except manifest.csv)
│   ├── raw/                       # FiftyOne-downloaded Open Images
│   └── processed/
│       ├── train/  val/  test/    # 256x256 JPEGs
│       └── manifest.csv           # tracked in git for reproducibility
├── scripts/
│   ├── download_dataset.py        # FiftyOne downloader (random or manifest-pinned)
│   └── preprocess_images.py       # filter B&W, resize, split
├── src/
│   └── data/
│       └── dataset.py             # ColorizationDataset (RGB→LAB)
├── pyproject.toml                 # deps + project config (uv-managed)
└── .python-version                # pinned to 3.12
```

## Disk usage

- `data/raw/` — ~1.5 GB (original-resolution JPEGs from Flickr)
- `data/processed/` — ~3 GB (256×256 JPEGs across train/val/test)
- Total: ~4.5 GB. Make sure you have headroom before running the download.
