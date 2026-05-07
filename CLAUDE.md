# Image Colorization

Deep learning project that trains a neural network to colorize black and white images.

## Goal

Learn a mapping from grayscale input images to plausible colored outputs by training a deep learning model on pairs of (grayscale, color) images. Given a single-channel black and white photo, the model should produce a realistic three-channel colored version.

## Approach

- **Task type:** image-to-image translation (grayscale → color).
- **Training signal:** supervised learning on color images, where the grayscale version is derived from the color ground truth.
- **Color space:** typically operate in LAB color space — predict the `a` and `b` chrominance channels from the `L` (luminance) channel, then recombine.
- **Model family:** convolutional encoder–decoder architectures (e.g. U-Net style), with optional adversarial or perceptual losses to improve color realism.

## Project Status

Early stage. Data pipeline is scaffolded; model, training loop, and evaluation are still to be built.

## Data Pipeline

**Source:** [Open Images V7](https://storage.googleapis.com/openimages/web/index.html), 100K random sample from the train split (CC BY 2.0 images, free for ML use).

**Working resolution:** 256×256. Train at this size; at inference, predict `ab` at 256×256 then upsample chrominance and recombine with the original full-resolution `L` channel.

**Pipeline:**

1. `scripts/download_dataset.py` — uses FiftyOne to pull a random 100K subset (no labels) into `data/raw/`.
2. `scripts/preprocess_images.py` — filters out grayscale photos, center-crops to square, resizes to 256×256 JPEG (q=90), splits 80/10/10 into `data/processed/{train,val,test}/`, writes `manifest.csv`.
3. `src/data/dataset.py` — `ColorizationDataset` reads the manifest, converts RGB→LAB at runtime, returns `(L, ab)` tensors normalized to `[-1, 1]`.

**Run end-to-end** (uses [uv](https://docs.astral.sh/uv/) for dependency management):

```bash
uv sync                                            # creates .venv, installs deps from pyproject.toml
uv run python scripts/download_dataset.py          # ~1.5 GB raw download
uv run python scripts/preprocess_images.py         # ~3 GB processed (256x256 JPEG)
```

Both scripts accept `--num-images`, `--seed`, and path overrides — see `--help`.

### Reproducibility (group workflow)

Open Images points to Flickr URLs that can disappear over time, so a fresh random sample isn't bit-exact across teammates. We pin the dataset by committing the post-preprocess `manifest.csv` (allowed through `.gitignore` while everything else under `data/` stays ignored).

**One person sets the canonical dataset:**

```bash
uv run python scripts/download_dataset.py
uv run python scripts/preprocess_images.py
git add data/processed/manifest.csv
git commit -m "chore: pin dataset manifest"
```

**Everyone else reproduces it exactly:**

```bash
uv sync
uv run python scripts/download_dataset.py --manifest data/processed/manifest.csv
uv run python scripts/preprocess_images.py
```

In `--manifest` mode the downloader fetches only those specific Open Images IDs, so the resulting splits are identical for every teammate.

## Stack (planned)

- Python
- PyTorch (or TensorFlow) for model definition and training
- NumPy / Pillow / OpenCV for image I/O and color-space conversion
- Jupyter notebooks for exploration and result visualization

## Layout

```
image-colorization/
├── data/                          # gitignored
│   ├── raw/                       # FiftyOne-downloaded Open Images
│   └── processed/                 # 256x256 JPEGs + manifest.csv
│       ├── train/  val/  test/
│       └── manifest.csv
├── scripts/
│   ├── download_dataset.py        # FiftyOne subset downloader
│   └── preprocess_images.py       # resize, filter B&W, split
├── src/
│   └── data/
│       └── dataset.py             # ColorizationDataset (RGB→LAB)
├── tests/                         # to be added
├── docs/                          # design notes, experiment logs
├── pyproject.toml                 # deps + project config (uv-managed)
└── .python-version                # pinned Python (3.12)
```
