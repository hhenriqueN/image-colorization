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

## Training pipeline

Three phases of increasing quality. Each has its own notebook under `notebooks/`.

| Phase | Notebook | Architecture | Loss | Dependency |
|---|---|---|---|---|
| 1 | `01_unet.ipynb` | U-Net from scratch | L1 | None — independent baseline |
| 2 | `02_resnet_unet.ipynb` | ResNet-34 encoder + U-Net decoder | L1 + Perceptual | None — uses ImageNet weights |
| 3 | `03_cgan.ipynb` | Phase 2 generator + PatchGAN discriminator | L1 + Perceptual + Adversarial | Requires Phase 2 `best.pth` |

**Phase 1** trains everything from scratch with a pixel-level L1 loss. Learns correct spatial structure (where colors go) but produces desaturated outputs — L1 pushes the model toward the mean color rather than a vivid prediction. Useful as a baseline.

**Phase 2** replaces the encoder with a frozen ResNet-34 pretrained on ImageNet, giving the decoder rich semantic features from epoch 1. A VGG perceptual loss adds a high-level structural signal on top of L1. Convergence is 3–5× faster; colors are better saturated and more accurately localized.

**Phase 3** adds a PatchGAN discriminator that scores each 70×70 patch as real or fake. The adversarial loss forces the generator to produce locally convincing colors, eliminating the muddy averaged outputs from L1 alone. The generator is initialized from Phase 2 weights for stable GAN training.

Checkpoints are saved to `checkpoints/<phase>/` (gitignored). To run Phase 3, point `PHASE2_CHECKPOINT` in `03_cgan.ipynb` at the Phase 2 `best.pth` before starting.

## Project layout

```
image-colorization/
├── data/                          # gitignored (except manifest.csv)
│   ├── raw/                       # original-resolution Open Images
│   └── processed/
│       ├── train/  val/  test/    # 256x256 JPEGs
│       └── manifest.csv           # tracked in git for reproducibility
├── notebooks/
│   ├── 01_unet.ipynb              # Phase 1 training & evaluation
│   ├── 02_resnet_unet.ipynb       # Phase 2 training & evaluation
│   └── 03_cgan.ipynb              # Phase 3 training & evaluation
├── scripts/
│   ├── download_dataset.py        # async S3 downloader (manifest-pinned or random)
│   └── preprocess_images.py       # filter B&W, resize, split
├── src/
│   ├── data/
│   │   └── dataset.py             # ColorizationDataset (RGB→LAB)
│   ├── losses/
│   │   └── perceptual.py          # VGG-16 perceptual loss (differentiable LAB→RGB)
│   └── models/
│       ├── unet.py                # Phase 1: vanilla U-Net
│       ├── resnet_unet.py         # Phase 2: ResNet-34 encoder + U-Net decoder
│       └── discriminator.py       # Phase 3: 70×70 PatchGAN discriminator
├── checkpoints/                   # gitignored — saved during training
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
