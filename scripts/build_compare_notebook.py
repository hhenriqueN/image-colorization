"""Build notebooks/04_compare_results.ipynb programmatically.

Run via: uv run python scripts/build_compare_notebook.py
Then execute via: uv run jupyter nbconvert --to notebook --execute --inplace notebooks/04_compare_results.ipynb
"""

from __future__ import annotations

from pathlib import Path

import nbformat as nbf

PROJECT_ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK_PATH = PROJECT_ROOT / "notebooks" / "04_compare_results.ipynb"


def md(source: str) -> nbf.NotebookNode:
    return nbf.v4.new_markdown_cell(source.strip("\n"))


def code(source: str) -> nbf.NotebookNode:
    return nbf.v4.new_code_cell(source.strip("\n"))


def build() -> nbf.NotebookNode:
    nb = nbf.v4.new_notebook()
    nb.metadata = {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3",
        },
        "language_info": {"name": "python"},
    }

    cells: list[nbf.NotebookNode] = []

    cells.append(md("""
# Image Colorization — model comparison

Loads all three trained checkpoints (Phase A regression, Phase B cGAN-polished, Phase C
Zhang-style classification), plots training curves across all runs, and renders side-by-side
visual comparisons on validation images.

**Phase A**: frozen ImageNet ResNet-34 encoder + trainable U-Net decoder. Loss = L1 + 0.1·perceptual.
**Phase B**: same generator warm-started from Phase A, paired with a PatchGAN discriminator
under LSGAN + L1 + perceptual.
**Phase C**: same encoder/decoder warm-started from Phase B, but the final layer outputs
*Q* per-pixel logits instead of regressing ab. Loss = weighted cross-entropy over precomputed
ab bins (Zhang et al. 2016). Inference reconstructs ab via temperature-scaled annealed-mean.
The point of Phase C is to break the sepia attractor that L1-regression suffers from.
"""))

    cells.append(code("""
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from IPython.display import Image as IPImage, display
from PIL import Image

PROJECT_ROOT = Path.cwd().resolve()
if PROJECT_ROOT.name == "notebooks":
    PROJECT_ROOT = PROJECT_ROOT.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.dataset import AB_MAX, ColorizationDataset, lab_to_rgb
from src.data.quantize import annealed_mean
from src.models.resnet_unet import ResNetUNet
from src.models.resnet_unet_cls import ResNetUNetClassifier

device = (
    torch.device("mps") if torch.backends.mps.is_available()
    else torch.device("cuda") if torch.cuda.is_available()
    else torch.device("cpu")
)
print(f"PROJECT_ROOT = {PROJECT_ROOT}")
print(f"device       = {device}")
print(f"torch        = {torch.__version__}")
"""))

    cells.append(md("## 1. Training metrics across all runs"))

    cells.append(code("""
RUNS = {
    "Phase A run02 (3K, 5 ep)":       PROJECT_ROOT / "checkpoints" / "resnet_unet_run02" / "log.jsonl",
    "Phase A run03 (10K, 8 ep)":      PROJECT_ROOT / "checkpoints" / "resnet_unet_run03" / "log.jsonl",
    "Phase B cgan_run01 (5K, 8 ep)":  PROJECT_ROOT / "checkpoints" / "cgan_run01"        / "log.jsonl",
    "Phase C cls_run01 (25K, 10 ep)": PROJECT_ROOT / "checkpoints" / "cls_run01"         / "log.jsonl",
}

def load_log(p: Path) -> pd.DataFrame:
    if not p.exists():
        return pd.DataFrame()
    rows = [json.loads(line) for line in p.read_text().splitlines() if line.strip()]
    return pd.DataFrame(rows)

logs = {name: load_log(p) for name, p in RUNS.items()}

summary_rows = []
for name, df in logs.items():
    if df.empty:
        continue
    summary_rows.append({
        "run": name,
        "epochs": int(df["epoch"].max()),
        "best_val_l1": float(df["val_l1"].min()),
        "final_val_l1": float(df.iloc[-1]["val_l1"]),
        "wall_min": round(float(df.iloc[-1]["wall_time_s"]) / 60, 1),
    })
summary = pd.DataFrame(summary_rows)
summary
"""))

    cells.append(md("## 2. Validation L1 curves"))

    cells.append(code("""
fig, ax = plt.subplots(figsize=(8, 5))
for name, df in logs.items():
    if df.empty:
        continue
    ax.plot(df["epoch"], df["val_l1"], marker="o", label=name)
ax.set_xlabel("epoch")
ax.set_ylabel("val_l1 (lower = better)")
ax.set_title("Validation L1 on ab channels across runs")
ax.grid(True, alpha=0.3)
ax.legend()
plt.tight_layout()
plt.show()
"""))

    cells.append(md("""
## 3. cGAN training dynamics

Watching the discriminator loss (`d`) and generator adversarial loss (`g_adv`) tells us
whether GAN training is balanced. Healthy LSGAN sits around `d ≈ 0.25`; if `d → 0` the
discriminator is winning and `g_adv` will climb.
"""))

    cells.append(code("""
cgan_df = logs.get("Phase B cgan_run01 (5K, 8 ep)", pd.DataFrame())
if not cgan_df.empty:
    fig, axes = plt.subplots(1, 2, figsize=(13, 4))
    axes[0].plot(cgan_df["epoch"], cgan_df["train_d"], marker="o", color="tab:red",   label="D loss")
    axes[0].plot(cgan_df["epoch"], cgan_df["train_g_adv"], marker="o", color="tab:blue", label="G_adv loss")
    axes[0].axhline(0.25, linestyle="--", color="gray", label="LSGAN balanced (0.25)")
    axes[0].set_xlabel("epoch"); axes[0].set_ylabel("loss"); axes[0].set_title("cGAN balance")
    axes[0].legend(); axes[0].grid(True, alpha=0.3)

    axes[1].plot(cgan_df["epoch"], cgan_df["train_g_l1"], marker="o", color="tab:green", label="train G_L1")
    axes[1].plot(cgan_df["epoch"], cgan_df["val_l1"],     marker="o", color="tab:purple", label="val L1")
    axes[1].set_xlabel("epoch"); axes[1].set_ylabel("L1"); axes[1].set_title("cGAN L1 components")
    axes[1].legend(); axes[1].grid(True, alpha=0.3)
    plt.tight_layout(); plt.show()
else:
    print("No cGAN log available.")
"""))

    cells.append(md("## 4. Load all three models"))

    cells.append(code("""
def load_regression(path: Path) -> ResNetUNet:
    model = ResNetUNet(freeze_encoder=True)
    state = torch.load(path, map_location=device, weights_only=False)
    if isinstance(state, dict):
        if "model_state_dict" in state:
            state = state["model_state_dict"]
        elif "generator_state_dict" in state:
            state = state["generator_state_dict"]
    model.load_state_dict(state)
    return model.to(device).eval()

def load_classifier(path: Path):
    ckpt = torch.load(path, map_location=device, weights_only=False)
    num_classes = int(ckpt.get("num_classes", ckpt["model_state_dict"]["out_conv.weight"].shape[1]))
    temperature = float(ckpt.get("temperature", 0.38))
    model = ResNetUNetClassifier(num_classes=num_classes, freeze_encoder=True)
    model.load_state_dict(ckpt["model_state_dict"])
    model = model.to(device).eval()
    return model, temperature

phase_a_ckpt = PROJECT_ROOT / "checkpoints" / "resnet_unet_run03" / "best.pth"
phase_b_ckpt = PROJECT_ROOT / "checkpoints" / "cgan_run01"        / "best_generator.pth"
phase_c_ckpt = PROJECT_ROOT / "checkpoints" / "cls_run01"         / "best.pth"

# Inference temperature for Phase C. The checkpoint was trained reporting
# val_l1 at Zhang's default T=0.38, but at inference we found T=0.20 gives
# a much better saturation tradeoff on our 214-bin gamut. T → 0 approaches
# argmax (vivid but patchy); T → ∞ approaches the mean (sepia again).
PHASE_C_TEMPERATURE = 0.20

phase_a = load_regression(phase_a_ckpt)
phase_b = load_regression(phase_b_ckpt)
phase_c, phase_c_T = (None, None)
bin_centers = None
if phase_c_ckpt.exists():
    phase_c, _trained_T = load_classifier(phase_c_ckpt)
    phase_c_T = PHASE_C_TEMPERATURE
    bin_centers = torch.from_numpy(np.load(
        PROJECT_ROOT / "data" / "processed" / "ab_bin_centers.npy"
    )).to(torch.float32).to(device)
    print(f"loaded Phase C from {phase_c_ckpt.relative_to(PROJECT_ROOT)}  "
          f"T_train={_trained_T}  T_inference={phase_c_T}  Q={bin_centers.shape[0]}")
else:
    print(f"Phase C checkpoint NOT FOUND at {phase_c_ckpt} — Phase C columns will be skipped")
print(f"loaded Phase A from {phase_a_ckpt.relative_to(PROJECT_ROOT)}")
print(f"loaded Phase B from {phase_b_ckpt.relative_to(PROJECT_ROOT)}")
"""))

    cells.append(md("""
## 5. Visual comparison on held-out validation images

Each row is one image. The columns are:

| Col 1 | Col 2 | Col 3 | Col 4 | Col 5 |
|---|---|---|---|---|
| **INPUT** | **GENERATED — Phase A** (regression, no GAN) | **GENERATED — Phase B** (cGAN-polished) | **GENERATED — Phase C** (Zhang classification) | **REAL** (ground-truth photo) |

So the leftmost column is what the model sees; the three middle columns are what the model
*invented*; the rightmost column is what we wish the model had produced.

Phase C is the new column — the Zhang-style classification model. Its whole reason to exist is
to break the sepia attractor of L1 regression, so look for **stronger saturation, especially
on inputs Phase A/B handle badly** (bright artificial paint, vivid water, graphic content).

Indices are picked deterministically from the full val split (9,222 images) — all out-of-sample.
If Phase C is unavailable, the column is omitted.
"""))

    cells.append(code("""
val_ds = ColorizationDataset(split="val", horizontal_flip=False)
print(f"full val split: {len(val_ds)} images")

rng = np.random.default_rng(0)
N = 8
indices = sorted(rng.choice(len(val_ds), size=N, replace=False).tolist())
print(f"sampled indices: {indices}")
"""))

    cells.append(code("""
@torch.no_grad()
def predict_rgb_regression(model, l_tensor):
    l_in = l_tensor.unsqueeze(0).to(device)
    ab_pred = model(l_in).cpu().squeeze(0).clamp(-1, 1)
    return lab_to_rgb(l_tensor, ab_pred)

@torch.no_grad()
def predict_rgb_classifier(model, l_tensor, bin_centers, T):
    l_in = l_tensor.unsqueeze(0).to(device)
    logits = model(l_in)
    ab_lab = annealed_mean(logits, bin_centers, temperature=T)
    ab_pred = (ab_lab / AB_MAX).cpu().squeeze(0).clamp(-1, 1)
    return lab_to_rgb(l_tensor, ab_pred)

def gray_rgb(l_tensor):
    gray = ((l_tensor.squeeze(0) + 1.0) / 2.0 * 255.0).clamp(0, 255).numpy().astype(np.uint8)
    return np.stack([gray, gray, gray], axis=-1)

has_phase_c = phase_c is not None
column_titles = [
    "INPUT\\n(B&W, what the model sees)",
    "GENERATED — Phase A\\n(ResNet-UNet, no GAN)",
    "GENERATED — Phase B\\n(cGAN-polished)",
]
column_colors = ["#444444", "#1f77b4", "#d62728"]
if has_phase_c:
    column_titles.append("GENERATED — Phase C\\n(Zhang classification)")
    column_colors.append("#ff7f0e")
column_titles.append("REAL\\n(ground truth photo)")
column_colors.append("#2ca02c")

n_cols = len(column_titles)
fig, axes = plt.subplots(N, n_cols, figsize=(2.8 * n_cols, 3.4 * N))

for row, idx in enumerate(indices):
    l_tensor, ab_true = val_ds[idx]
    panels = [
        gray_rgb(l_tensor),
        predict_rgb_regression(phase_a, l_tensor),
        predict_rgb_regression(phase_b, l_tensor),
    ]
    if has_phase_c:
        panels.append(predict_rgb_classifier(phase_c, l_tensor, bin_centers, phase_c_T))
    panels.append(lab_to_rgb(l_tensor, ab_true))

    for col, panel in enumerate(panels):
        ax = axes[row, col]
        ax.imshow(panel)
        ax.set_xticks([]); ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_edgecolor(column_colors[col])
            spine.set_linewidth(2.5)
        ax.set_title(column_titles[col], fontsize=10, weight="bold",
                     color=column_colors[col], pad=4)
    axes[row, 0].set_ylabel(f"val image #{idx}", rotation=90, fontsize=10,
                            weight="bold", labelpad=8)

plt.tight_layout()
plt.show()
"""))

    cells.append(md("""
## 6. Per-image L1 gap (Phase A vs Phase B)

A lower L1 on a given image means the prediction is numerically closer to the ground truth's ab channels.
Negative `delta = B - A` means cGAN improved on that image.
"""))

    cells.append(code("""
import torch.nn.functional as F

@torch.no_grad()
def l1_regression(model, l_tensor, ab_true):
    l_in = l_tensor.unsqueeze(0).to(device)
    ab_pred = model(l_in).cpu().squeeze(0).clamp(-1, 1)
    return F.l1_loss(ab_pred, ab_true).item()

@torch.no_grad()
def l1_classifier(model, l_tensor, ab_true, bin_centers, T):
    l_in = l_tensor.unsqueeze(0).to(device)
    ab_lab = annealed_mean(model(l_in), bin_centers, temperature=T)
    ab_pred = (ab_lab / AB_MAX).cpu().squeeze(0).clamp(-1, 1)
    return F.l1_loss(ab_pred, ab_true).item()

rows = []
for idx in indices:
    l_tensor, ab_true = val_ds[idx]
    la = l1_regression(phase_a, l_tensor, ab_true)
    lb = l1_regression(phase_b, l_tensor, ab_true)
    row = {"idx": idx, "phase_a_l1": la, "phase_b_l1": lb, "delta_b_minus_a": lb - la}
    if has_phase_c:
        lc = l1_classifier(phase_c, l_tensor, ab_true, bin_centers, phase_c_T)
        row["phase_c_l1"] = lc
        row["delta_c_minus_a"] = lc - la
        row["delta_c_minus_b"] = lc - lb
    rows.append(row)
delta_df = pd.DataFrame(rows).sort_values("delta_b_minus_a")
delta_df
"""))

    cells.append(md("""
## 7. Pre-rendered training-time sample grids

Below are the per-epoch sample grids saved during training (8 fixed images from the 500-image val
subset that `train.py` used). These are the same images we watched evolve during each cycle.

**Each grid has 3 columns:**

| Col 1 | Col 2 | Col 3 |
|---|---|---|
| **INPUT** (B&W input image) | **GENERATED** (what the model produced) | **REAL** (ground truth color photo) |

Compare the three grids to see how the predictions evolved: vanilla Phase A early → Phase A fully
trained → Phase B (cGAN polished).
"""))

    cells.append(code("""
def show_grid_with_labels(path: Path, title: str) -> None:
    if not path.exists():
        print(f\"=== {title}: NOT FOUND at {path} ===\")
        return
    img = np.array(Image.open(path))
    h, w = img.shape[:2]
    # The grid has 3 columns separated by 4-pixel white strips
    col_w = (w - 8) / 3
    col_centers = [col_w / 2, col_w + 4 + col_w / 2, 2 * col_w + 8 + col_w / 2]
    labels = [
        ("INPUT\\n(B&W)",            "#444444"),
        ("GENERATED\\n(model out)",  "#1f77b4"),
        ("REAL\\n(ground truth)",    "#2ca02c"),
    ]

    fig_h = 12 * h / w  # keep aspect ratio
    fig, ax = plt.subplots(figsize=(9, fig_h))
    ax.imshow(img)
    ax.set_xticks([]); ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)
    for cx, (lbl, color) in zip(col_centers, labels):
        ax.text(cx, -h * 0.012, lbl, ha=\"center\", va=\"bottom\",
                fontsize=11, weight=\"bold\", color=color)
    ax.set_title(title, fontsize=12, weight=\"bold\", pad=42)
    plt.tight_layout()
    plt.show()

sample_dirs = [
    (\"Phase A run02 — epoch 5  (3K subset, baseline)\",
        PROJECT_ROOT / \"outputs\" / \"samples\" / \"resnet_unet_run02\" / \"epoch_005.png\"),
    (\"Phase A run03 — epoch 8  (10K subset, fully trained)\",
        PROJECT_ROOT / \"outputs\" / \"samples\" / \"resnet_unet_run03\" / \"epoch_008.png\"),
    (\"Phase B cgan_run01 — epoch 8  (cGAN polish, warm-started from Phase A)\",
        PROJECT_ROOT / \"outputs\" / \"samples\" / \"cgan_run01\"        / \"epoch_008.png\"),
    (\"Phase C cls_run01 — final epoch  (Zhang classification, warm-started from Phase B)\",
        PROJECT_ROOT / \"outputs\" / \"samples\" / \"cls_run01\"         / \"epoch_010.png\"),
]
for label, path in sample_dirs:
    show_grid_with_labels(path, label)
"""))

    cells.append(md("""
## Notes

- `val_l1` is on the **normalized** ab channels (range `[-1, 1]`), so 0.08 corresponds to an
  average per-channel error of about 10 LAB units.
- The cGAN's *raw L1* sometimes ticks up while *visual saturation* improves — the adversarial
  signal trades a tiny bit of pixel-wise fidelity for more believable color distributions.
- The biggest remaining gap is on graphics / bright artificial colors (logos, painted walls):
  both phases tend toward muted naturalistic colors there.
"""))

    nb["cells"] = cells
    return nb


def main() -> int:
    nb = build()
    NOTEBOOK_PATH.parent.mkdir(parents=True, exist_ok=True)
    nbf.write(nb, NOTEBOOK_PATH)
    print(f"wrote {NOTEBOOK_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
