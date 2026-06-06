# image-colorization

Deep learning project: train a neural network to colorize black and white images. Predict the `a`, `b` chrominance channels (LAB color space) from the `L` luminance channel using a convolutional encoder–decoder, working at 256×256 resolution.

**Current best model:** Phase C — Zhang-style classification colorizer over 214 ab bins, warm-started from Phase B's cGAN generator. Final `val_l1 = 0.0775` on the annealed-mean output. At inference temperature **T=0.10 + bilateral filter** on the ab channels (`src/data/quantize.py::bilateral_smooth_ab`) it produces visibly more saturated results than any of the regression-loss phases without the discrete-bin patchiness you get from low T alone — see [`notebooks/04_compare_results.ipynb`](./notebooks/04_compare_results.ipynb).

For a deeper overview (color space, model family) see [`CLAUDE.md`](./CLAUDE.md). For the full trajectory of what we tried, why, the bugs we hit, and what still doesn't work, see [`docs/local_training_log.md`](./docs/local_training_log.md).

## Where the deliverable is

| Artifact | Path | What's in it |
|---|---|---|
| **Comparison notebook** | [`notebooks/04_compare_results.ipynb`](./notebooks/04_compare_results.ipynb) | Fully executed (~25 MB). Metrics tables for all runs, training curves, cGAN balance plot, 5-column visual grid (gray ∣ Phase A ∣ Phase B ∣ Phase C ∣ truth) on held-out val images, per-image L1 deltas, and the rendered training-time sample grids. |
| **Run log** | [`docs/local_training_log.md`](./docs/local_training_log.md) | Documented journey: Kaggle → local M4, the NaN-on-MPS bug, Phase A/B/C recipes, the T=0.20 finding, what still doesn't work, prioritized next steps. |
| **Best generator weights** | `checkpoints/cls_run01/best.pth` *(gitignored)* | Phase C classifier. 214 logits/pixel; convert to ab via `src/data/quantize.py::annealed_mean(..., T=0.20)`. |
| **Phase C priors** | `data/processed/ab_bin_centers.npy`, `ab_rebalance_weights.npy`, `bin_priors.png` *(gitignored)* | The 214-bin ab gamut computed from our own training data. Re-generate with `scripts/precompute_classification_priors.py`. |

## Models trained (chronological)

The original three-Kaggle-notebook plan didn't work end-to-end on Kaggle (kernel restarts, timeouts), so we pivoted to local M4 (24 GB) MPS training. The Kaggle notebooks (`01_unet_kaggle.ipynb`, `02_resnet_unet_kaggle.ipynb`, `03_cgan_kaggle.ipynb`) are kept in `notebooks/` for reference.

| Phase | Run | Architecture | Loss | Train size | Epochs | Best val_l1 | Wall time |
|---|---|---|---|---|---|---|---|
| A1 | `resnet_unet_run02` | Frozen ResNet-34 + U-Net decoder | L1 + 0.1·Perceptual | 3,000 | 5 | 0.1057 | 12 min |
| A2 | `resnet_unet_run03` | Same | Same | 10,000 | 8 | 0.0834 | 70 min |
| B  | `cgan_run01` | Phase A generator + PatchGAN | L1 + 0.1·Perceptual + 0.01·LSGAN | 5,000 | 8 | 0.0800 | 70 min |
| **C** | **`cls_run01`** | **ResNet-34 + decoder + 214-way classifier head** | **Rebalanced cross-entropy (Zhang 2016)** | **25,000** | **10** | **0.0775** | **~6h 50** |

- **Phase A** = ResNet-UNet transfer learning. The `01_unet.ipynb` from-scratch U-Net is **dropped** — too slow and produces washed output. Phase A1 was a smoke test on 3K images; A2 was the real run with 10K images.
- **Phase B** = cGAN polish over Phase A. The discriminator collapsed by epoch 4 (`d_loss → 0`), so most of the L1 improvement in this phase comes from extra training, not adversarial pressure. Some saturation gain was real.
- **Phase C** = classification reformulation. Per-pixel softmax over 214 ab bins instead of regressing ab. Rebalanced cross-entropy pushes against the L1 sepia attractor. At T=0.20 the annealed-mean inference produces clearly more vivid output than any earlier phase.

Each phase's `log.jsonl` (one JSON line per epoch) and per-epoch sample grids live under `checkpoints/<run>/` and `outputs/samples/<run>/` respectively.

## Reproducing the trained models

Phase A2:
```bash
uv run python scripts/train.py \
  --model resnet_unet \
  --train-subset 10000 --val-subset 500 \
  --batch-size 24 --epochs 8 --lr 2e-4 \
  --lambda-perceptual 0.1 --freeze-encoder \
  --checkpoint-dir checkpoints/resnet_unet_run03 --device mps
```

Phase B (warm-started from Phase A):
```bash
uv run python scripts/train.py \
  --model cgan \
  --train-subset 5000 --val-subset 500 \
  --batch-size 16 --epochs 8 \
  --lr 2e-4 --lr-d 1e-4 \
  --lambda-perceptual 0.1 --lambda-adv 0.01 \
  --warm-start checkpoints/resnet_unet_run03/best.pth \
  --checkpoint-dir checkpoints/cgan_run01 --device mps
```

Phase C (warm-started from Phase B):
```bash
# One-shot prior computation — produces data/processed/ab_bin_centers.npy etc.
uv run python scripts/precompute_classification_priors.py \
  --sample-size 5000 --grid-step 5 --min-fraction 1e-5

# Training
uv run python scripts/train.py \
  --model resnet_unet_cls \
  --train-subset 25000 --val-subset 500 \
  --batch-size 16 --epochs 10 --lr 3e-4 --freeze-encoder \
  --warm-start checkpoints/cgan_run01/best_generator.pth \
  --checkpoint-dir checkpoints/cls_run01 --device mps
```

Wrap any of these in `caffeinate -dimsu` on macOS to keep the system from sleeping during long runs.

Refresh the comparison notebook after training:
```bash
uv run python scripts/build_compare_notebook.py
uv run jupyter nbconvert --to notebook --execute --inplace \
  notebooks/04_compare_results.ipynb
```

## Quick start (teammates / dataset only)

If you're cloning this repo to get the dataset and reproduce, run these four commands:

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

After step 4 you'll have:

- `data/processed/train/` — 74,919 JPEGs (80%)
- `data/processed/val/` — 9,222 JPEGs (10%)
- `data/processed/test/` — 9,321 JPEGs (10%)

The split assignment is hash-based on the image ID, so every teammate ends up with **the exact same images in the exact same splits** — bit-exact reproducibility.

> ⚠️ **Don't commit your local `data/processed/manifest.csv`.** Running preprocess locally rewrites it with absolute paths from your machine. Just leave those changes uncommitted (`git restore data/processed/manifest.csv` if it ever gets staged).

### Disk space you'll need

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
├── data/                              # gitignored (except manifest.csv)
│   ├── raw/                           # original-resolution Open Images
│   └── processed/
│       ├── train/  val/  test/        # 256×256 JPEGs
│       ├── manifest.csv               # tracked — bit-exact split reproducibility
│       ├── ab_bin_centers.npy         # Phase C: 214 ab bin centers (regen via priors script)
│       ├── ab_rebalance_weights.npy   # Phase C: Zhang-style class weights
│       └── bin_priors.png             # Phase C: gamut heatmap (sanity check)
├── notebooks/
│   ├── 01_unet.ipynb                  # legacy: original Kaggle phase 1 (dropped)
│   ├── 02_resnet_unet.ipynb           # legacy: original Kaggle phase 2
│   ├── 03_cgan.ipynb                  # legacy: original Kaggle phase 3
│   ├── 0[1-3]_*_kaggle.ipynb          # Kaggle-adapted versions of the same
│   └── 04_compare_results.ipynb       # MAIN deliverable — A vs B vs C side-by-side
├── scripts/
│   ├── download_dataset.py            # async S3 downloader (manifest-pinned or random)
│   ├── preprocess_images.py           # filter B&W, resize, split
│   ├── make_kaggle_notebooks.py       # adapts the local notebooks for Kaggle
│   ├── check_mps.py                   # one-shot MPS sanity check
│   ├── train.py                       # headless CLI: --model {resnet_unet, cgan, resnet_unet_cls}
│   ├── generate_samples.py            # sample grid generator (both model kinds)
│   ├── compare_checkpoints.py         # 4-column standalone A-vs-B grid
│   ├── precompute_classification_priors.py  # one-shot: compute Phase C ab gamut + weights
│   └── build_compare_notebook.py      # builds notebooks/04_compare_results.ipynb
├── src/
│   ├── data/
│   │   ├── dataset.py                 # ColorizationDataset (RGB→LAB, [-1, 1])
│   │   └── quantize.py                # Phase C: ab-bin quantization + annealed-mean
│   ├── losses/
│   │   └── perceptual.py              # VGG-16 perceptual loss (NaN-safe on MPS)
│   └── models/
│       ├── unet.py                    # legacy: vanilla Phase 1 U-Net (unused)
│       ├── resnet_unet.py             # Phase A/B: ResNet-34 encoder + U-Net decoder
│       ├── resnet_unet_cls.py         # Phase C: same backbone, 214-way classifier head
│       └── discriminator.py           # Phase B: 70×70 PatchGAN discriminator
├── docs/
│   └── local_training_log.md          # full journey: Kaggle → local → A → B → C, with metrics
├── checkpoints/                       # gitignored — saved during training
├── outputs/                           # gitignored — sample PNGs and visualizations
├── pyproject.toml                     # deps + project config (uv-managed)
└── .python-version                    # pinned to 3.12
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
