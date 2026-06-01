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

Three trained colorizers exist after the local-training pivot (the original Kaggle plan
didn't work end-to-end on Kaggle):

- **Phase A** (`checkpoints/resnet_unet_run03/best.pth`): ResNet-34-encoder + U-Net-decoder,
  L1 + 0.1·perceptual, 10K subset, 8 epochs. Best `val_l1=0.0834`.
- **Phase B** (`checkpoints/cgan_run01/best_generator.pth`): same generator warm-started
  from A, paired with a PatchGAN discriminator. LSGAN + L1 + perceptual, 5K subset, 8
  epochs. Best `val_l1=0.0800`. D collapsed by epoch 4, so most of the L1 gain came
  from continued L1/perceptual training rather than from adversarial pressure.
- **Phase C** (`checkpoints/cls_run01/best.pth`): Zhang-style classification head over
  214 ab bins, rebalanced cross-entropy, annealed-mean at inference. 25K subset, 10
  epochs (~6.9 h on M4 MPS). Best `val_l1=0.0775`. **T=0.20 at inference is the
  saturation breakthrough** — same weights produce visibly more vivid output than
  Phase A/B (Capitol gets blue sky, apple goes vivid red, boats become cyan).

The main deliverable is `notebooks/04_compare_results.ipynb`, executed in-place with a
5-column visual grid comparing all three phases against ground truth on held-out val
images. The full journey log (Kaggle pivot, NaN-on-MPS bug, per-cycle metrics, honest
assessment of failures) lives in `docs/local_training_log.md`.

## Stack (actual)

- Python 3.12 (pinned via `.python-version`), `uv` for env + dep management
- PyTorch 2.11 + torchvision, MPS backend on M4 (`PYTORCH_ENABLE_MPS_FALLBACK=1`)
- scikit-image (RGB↔LAB), Pillow (image I/O), pandas + numpy, matplotlib for plots
- nbformat + jupyter nbconvert to programmatically build and execute the comparison notebook

## Data Pipeline

**Source:** [Open Images V7](https://storage.googleapis.com/openimages/web/index.html), 100K random sample from the train split (CC BY 2.0 images, free for ML use). Images are downloaded from the [CVDF S3 mirror](https://github.com/cvdfoundation/open-images-dataset) (free, public, no AWS account required) — much faster and more reliable than fetching individual Flickr URLs.

**Working resolution:** 256×256. Train at this size; at inference, predict `ab` at 256×256 then upsample chrominance and recombine with the original full-resolution `L` channel.

**Pipeline:**

1. `scripts/download_dataset.py` — async httpx downloader pulling from the S3 mirror with high concurrency. Skips images already on disk and migrates any leftovers from a previous FiftyOne run. Uses FiftyOne's locally cached `image_ids.csv` for the master ID list (sample seed-42, 100K).
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

## Layout (actual)

```
image-colorization/
├── data/processed/
│   ├── train/  val/  test/        # 256×256 JPEGs (gitignored)
│   ├── manifest.csv               # tracked — bit-exact split reproducibility
│   ├── ab_bin_centers.npy         # Phase C: 214 ab bin centers (gitignored)
│   ├── ab_rebalance_weights.npy   # Phase C: class weights (gitignored)
│   └── bin_priors.png             # Phase C: gamut heatmap (gitignored)
├── notebooks/
│   ├── 01_unet.ipynb / 01_unet_kaggle.ipynb              # legacy, dropped
│   ├── 02_resnet_unet.ipynb / 02_resnet_unet_kaggle.ipynb  # legacy Kaggle source
│   ├── 03_cgan.ipynb / 03_cgan_kaggle.ipynb              # legacy Kaggle source
│   └── 04_compare_results.ipynb                          # MAIN deliverable, executed
├── scripts/
│   ├── download_dataset.py preprocess_images.py          # data pipeline
│   ├── make_kaggle_notebooks.py                          # legacy Kaggle adapter
│   ├── check_mps.py                                      # MPS sanity check
│   ├── train.py                                          # headless CLI, all 3 model kinds
│   ├── generate_samples.py compare_checkpoints.py        # visual eval
│   ├── precompute_classification_priors.py               # Phase C: build ab gamut + weights
│   └── build_compare_notebook.py                         # rebuild notebook 04
├── src/
│   ├── data/dataset.py                                   # ColorizationDataset (RGB→LAB)
│   ├── data/quantize.py                                  # Phase C: bins + annealed-mean
│   ├── losses/perceptual.py                              # VGG perceptual loss (NaN-safe MPS)
│   └── models/
│       ├── unet.py                                       # legacy Phase 1, unused
│       ├── resnet_unet.py                                # Phase A/B model
│       ├── resnet_unet_cls.py                            # Phase C classifier head
│       └── discriminator.py                              # Phase B PatchGAN
├── docs/local_training_log.md                            # full journey + metrics
├── checkpoints/  outputs/                                # gitignored training artifacts
├── pyproject.toml  .python-version                       # uv-managed, Python 3.12
└── README.md  CLAUDE.md                                  # public + Claude-facing docs
```
