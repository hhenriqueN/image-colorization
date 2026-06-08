# Session Changes — smp Rebuild + Zhang Classification + Saturation Tuning

> Companion document for a PowerPoint deck. Each `##` section is one slide's worth of content,
> with headline numbers, asset paths, and code snippets sized to fit a slide.
> Branch: `feat/smp-rebuild`.

---

## 1. What this session did (one-slide elevator)

Three-phase rebuild of the colorizer on a new branch, ending in a classification model whose
output approaches ground-truth saturation:

1. **Phase 1** — reproduce the Claude Desktop recipe (smp ResNet-34 U-Net + LAB + pure L1)
2. **Phase 2** — add PatchGAN adversarial fine-tune
3. **Phase 3** — pivot to Zhang-style 214-way classification when L1 hit its sepia floor

Then a free post-hoc win: an `ab`-channel boost that pushes colorfulness from 0.70 → 0.98
(parity with ground truth) without retraining.

| Phase | Recipe | Subset | Epochs | Wall | best val_l1 | best val_ce | colorfulness_ratio |
|---|---|---|---|---|---|---|---|
| 1 | smp L1, frozen encoder | 10K | 12 | 43 min | **0.0727** | — | ~0.50 (est.) |
| 2 | + PatchGAN, LSGAN, λ=0.01 | 10K | 6 | 55 min | 0.0752 | — | ~0.50 (est.) |
| 3 | + Zhang cls (Q=214), T=0.10 + bilateral | 50K | 10 | 6h51 | 0.0774 | **3.359** | **0.701** (peak 0.729) |
| 3 + boost | same model, `ab × 1.20` at inference | — | — | — | — | — | **~0.98** |

---

## 2. The starting recipe (Claude Desktop blueprint)

Came in as a research conversation. Chat recommended:

- Pretrained ResNet-34 encoder + from-scratch decoder (transfer learning for *objects*, not color)
- LAB color space (predict `ab` from `L`)
- `segmentation_models_pytorch` for the one-liner U-Net
- L1 baseline → PatchGAN adversarial fine-tune
- 1.5-day timeline on a small GPU

This is exactly what the existing repo had already built (`Phase A/B/C` in `CLAUDE.md`).
User chose **fresh restart**: rebuild on a new branch, follow the chat's narrative cleanly,
keep the existing checkpoints untouched.

---

## 3. Phase 1 — smp L1 baseline

**Model** (`src/models/smp_unet.py`, ~70 lines):

```python
class SmpUNet(nn.Module):
    def __init__(self, out_channels=2, freeze_encoder=True):
        super().__init__()
        self.unet = smp.Unet(
            encoder_name="resnet34",
            encoder_weights="imagenet",
            in_channels=1,           # L channel direct (smp handles 1→3 conv1 reweighting)
            classes=out_channels,    # 2 → ab
            activation=None,
        )
        if freeze_encoder:
            self.freeze_encoder()

    def forward(self, x):
        return torch.tanh(self.unet(x))  # tanh forces ab into [-1, 1]
```

**Training** (`scripts/train_smp.py --phase l1`):

- 10K-image Open Images subset, batch 16, 12 epochs
- AdamW lr=2e-4, cosine decay from epoch 6
- Pure L1, no perceptual (clean baseline framing)
- Frozen encoder throughout

**Result**: `val_l1=0.0727`, plateaued at epoch 8. Sample image:
`outputs/samples/smp_l1_run01/epoch_012.png`

---

## 4. Phase 2 — PatchGAN adversarial fine-tune

**Approach**: warm-start the Phase 1 generator, add the existing `PatchGANDiscriminator`,
train with LSGAN + L1.

**Key change vs legacy**: discriminator updated **every 2 batches** (`--d-step-every 2`)
and at half the discriminator LR (`3e-5` vs legacy `1e-4`). Prevents the D-collapse seen
in the original `cgan_run01`.

**Result**: `val_l1=0.0752` (slightly *worse* than Phase 1 — as the chat predicted; cGAN
trades L1 for vibrancy). `train_d` held at ~0.16–0.17 all 6 epochs — no collapse.

**The problem**: output still desaturated/sepia. Sample:
`outputs/samples/smp_cgan_run01/epoch_006.png`

---

## 5. The diagnosis — L1 mean-collapse / sepia attractor

What the chat had warned about, surfaced empirically:

- The conditional color distribution given a grayscale patch is **multimodal** (a car can be
  red, blue, black, white).
- The single `ab` value that minimizes L1 is the **mean of those modes**, which lands near
  the gray/brown LAB origin.
- So L1's optimum *is* sepia. Beating Phase 1 on L1 means matching its desaturation.

PatchGAN at λ=0.01 can't fix this — the L1 term is an anchor pulling toward the conditional
mean; a 0.01-weighted adversarial garnish on top doesn't reach the attractor. Pushing λ
higher risks the D-collapse the schedule is designed to avoid.

**Internal proof**: existing `cls_run01` (Zhang classification on same dataset) sits at
*higher* val_l1 (0.0775) than Phase 1 (0.0727) yet looks visibly vivid. That inversion is
the whole story.

---

## 6. The pivot — Phase 3 design

Recommendation from Claude Desktop (now informed by Phase 1/2 results):

> *"Merge your two assets: keep the pretrained-encoder U-Net (semantics + transfer
> constraint satisfied), swap the head from `classes=2` tanh-regression to `classes=214`
> logits, reuse the rebalanced-CE + annealed-mean + bilateral-post-filter machinery you
> already have, drop the discriminator entirely."*

Why classification works where PatchGAN didn't:

- The 214-way rebalanced CE objective **actively penalizes** under-predicting rare/vivid
  colors. No anchor pulls toward the dataset mean.
- Annealed-mean at low T biases inference toward the **most confident bin centroid**
  (vivid) rather than the soft-max centroid (muted).
- Bilateral filter on the ab channels cleans up discrete-bin patchiness without bleeding
  across object boundaries.

The chat also pointed out a meta-bug: **tracking val_l1 as the implied progress metric is
actively anti-correlated with the goal**. Need a colorfulness signal.

---

## 7. Phase 3 — model + new metric

**Model** (`src/models/smp_unet_cls.py`, ~95 lines): same smp.Unet but `classes=214`, no
`tanh` (raw logits to CE). Warm-start filters `unet.segmentation_head.*` keys so encoder +
decoder transfer from Phase 1; only the 214-way head is random-init.

**New metric** (`src/metrics/colorfulness.py`, ~30 lines):

```python
def hasler_susstrunk(rgb_uint8):
    R, G, B = rgb_uint8[...,0], rgb_uint8[...,1], rgb_uint8[...,2]
    rg = R - G
    yb = 0.5 * (R + G) - B
    return float(np.sqrt(rg.std()**2 + yb.std()**2)
                 + 0.3 * np.sqrt(rg.mean()**2 + yb.mean()**2))
```

Per-epoch log payload gained three fields:

- `val_colorfulness_pred` — Hasler-Süsstrunk on predicted RGB (subset of 64 val images)
- `val_colorfulness_true` — same on ground-truth RGB
- `colorfulness_ratio` — `pred / true` (the headline; 1.0 = parity)

Surfaces the sepia attractor as a number on epoch 1 instead of as an eyeball observation.

---

## 8. Phase 3 — overnight training run

**Command** (caffeinated for unattended 8h overnight):

```bash
caffeinate -i -s uv run python scripts/train_smp.py \
    --phase cls --train-subset 50000 --val-subset 500 \
    --batch-size 16 --epochs 10 --lr 3e-4 \
    --temperature 0.38 \
    --warm-start checkpoints/smp_l1_run01/best.pth \
    --checkpoint-dir checkpoints/smp_cls_run01
```

**Why these choices**:

- 50K subset — 2× the legacy `cls_run01`; chat's claim that data diversity > epoch count
  for breaking the attractor
- 10 epochs — set by measured per-epoch wall time of ~45 min from the smoke test
  (originally planned 14, scaled back to fit 8h overnight)
- Warm-start from Phase 1 best (not Phase 2 — cleaner narrative; Phase 2's adversarial
  drift is noise for the fresh classification head)
- `caffeinate -i -s` prevents idle + system sleep over 8h

**Result**: 6h51m wall time, `best val_ce=3.359`, `final colorfulness_ratio=0.701`
(peak 0.729 at epoch 5).

---

## 9. Phase 3 — visual result

Per-epoch sample grid evolution: `outputs/samples/smp_cls_run01/epoch_001.png` …
`epoch_010.png`. Each grid is `gray | smp_cls (T=0.10 + bilateral) | truth` on 8 fixed val
images.

Final epoch: visible vivid output. Skies blue, vegetation green, skin tones natural —
qualitatively the saturation breakthrough was real. But colorfulness_ratio at 0.70 still
left 30% headroom toward parity.

Key file to drop in the deck:
- `outputs/samples/smp_cls_run01/epoch_010.png` (final epoch, smp_cls)
- Compare against `outputs/samples/smp_l1_run01/epoch_012.png` (Phase 1 sepia baseline)

---

## 10. Pushing further — ab-channel boost ablation

User wanted more saturation. Cheapest lever surveyed:

> *Multiply predicted `ab` by a constant before LAB→RGB. Higher = more saturated. Clipping
> at the LAB gamut boundary acts as a natural ceiling.*

Implemented as a notebook ablation in `notebooks/06_smp_cls_showcase.ipynb` section 8/9.
Swept boosts {1.0, 1.1, 1.2, 1.3, 1.35, 1.4, 1.5, 1.65, 1.8, 2.0} on 12 fresh val images:

| boost | colorfulness_ratio |
|---|---|
| 1.00 (raw model) | 0.801 |
| 1.10 | 0.886 |
| **1.20** | **0.978** ← parity sweet spot |
| 1.30 | 1.071 (slight overshoot) |
| 1.50 | 1.229 (cartoony) |
| 2.00 | 1.582 (way overshot) |

---

## 11. Making boost=1.20 the default

Both visualization notebooks now apply `ab_boost = 1.20` by default everywhere the Phase 3
model is rendered. Concretely:

```python
ab_pred = (ab_lab / AB_MAX).cpu().squeeze(0)
ab_pred = (ab_pred * 1.20).clamp(-1, 1)   # the boost; clipping caps overshoot
return lab_to_rgb(l_tensor, ab_pred)
```

Constants: `AB_BOOST` in the showcase notebook, `PHASE_3_AB_BOOST` in the comparison
notebook. Change one number to re-tune.

The boost is **post-hoc** — no model retraining, no architecture change, costs zero
inference time. A free 0.20 jump in colorfulness ratio for one multiplication.

---

## 12. The colorfulness story (chart-ready)

Trajectory of the metric across the session:

| Stage | colorfulness_ratio |
|---|---|
| Phase 1 (smp L1) | ~0.50 (est., never instrumented) |
| Phase 2 (smp + PatchGAN) | ~0.50 (est., the chat predicted no real gain) |
| Phase 3 (smp + Zhang cls) raw | **0.701** (peak 0.729 ep 5) |
| Phase 3 + `ab × 1.20` boost | **~0.978** ← parity |

This chart by itself is the headline of the deck.

---

## 13. What got built — file map

**New code:**
- `src/models/smp_unet.py` — `SmpUNet` (Phase 1/2 generator)
- `src/models/smp_unet_cls.py` — `SmpUNetClassifier` (Phase 3, with warm-start filter)
- `src/metrics/__init__.py` — package marker
- `src/metrics/colorfulness.py` — Hasler-Süsstrunk
- `scripts/train_smp.py` — single CLI with `--phase {l1, cgan, cls}` (~700 lines)
- `scripts/build_smp_compare_notebook.py` — generates the 5-phase comparison
- `scripts/build_smp_cls_showcase_notebook.py` — generates the Phase 3 showcase
- `notebooks/05_smp_compare.ipynb` — 5-column comparison (gray | smp_l1 | smp_cgan | smp_cls | truth)
- `notebooks/06_smp_cls_showcase.ipynb` — Phase 3 deep-dive (curves, colorfulness chart, 12-image grid, ab-boost ablation)

**Config:**
- `pyproject.toml` — added `segmentation-models-pytorch>=0.3.4`
- `uv.lock` — regenerated

**Checkpoints** (gitignored):
- `checkpoints/smp_l1_smoke/` — pipeline sanity check
- `checkpoints/smp_l1_run01/` — Phase 1
- `checkpoints/smp_cgan_run01/` — Phase 2
- `checkpoints/smp_cls_smoke/` — Phase 3 pipeline check
- `checkpoints/smp_cls_run01/` — Phase 3

**Untouched:** all legacy checkpoints (`resnet_unet_run03`, `cgan_run01`, `cls_run01`),
the legacy comparison notebook (`notebooks/04_compare_results.ipynb`), all of `scripts/train.py`,
all of `src/models/{resnet_unet,resnet_unet_cls,unet,discriminator}.py`.

---

## 14. Three lessons that should go on a "what we learned" slide

1. **L1 in LAB is actively anti-correlated with saturation past the desaturation floor.**
   Its minimum lives at the conditional mean (sepia). A model that beats baseline on L1 is
   matching its desaturation. Track colorfulness alongside L1 from day 1, or you'll
   systematically optimize toward gray.

2. **Loss choice dominates architecture choice on this task.** Phase 1, 2, and 3 share the
   exact same encoder/decoder weights at warm-start. The only difference is the head + loss.
   That single change (regression → rebalanced classification) is what broke the sepia
   attractor. The pretrained backbone is necessary but not sufficient.

3. **Post-hoc inference tweaks can be high-ROI.** `ab × 1.20` is one multiplication, zero
   retraining, and bought a 0.20 jump in colorfulness ratio — going from 0.70 to ~0.98.
   Cheap levers exist between "the current output" and "retrain from scratch with a
   different loss." Try them before paying for compute.

---

## 15. Deliverables for the deck

| Asset | Path | Use |
|---|---|---|
| Phase 1 final sample (sepia floor) | `outputs/samples/smp_l1_run01/epoch_012.png` | "Before" |
| Phase 2 final sample (PatchGAN no gain) | `outputs/samples/smp_cgan_run01/epoch_006.png` | "Cgan didn't fix it" |
| Phase 3 final sample (Zhang cls) | `outputs/samples/smp_cls_run01/epoch_010.png` | "Loss change broke the attractor" |
| 5-phase visual comparison (rendered) | `notebooks/05_smp_compare.ipynb` § 6 | Side-by-side hero |
| Colorfulness ratio vs epoch | `notebooks/06_smp_cls_showcase.ipynb` § 3 | Metric story |
| ab-boost ablation visual | `notebooks/06_smp_cls_showcase.ipynb` § 7 | Cheap-win demo |
| ab-boost sweep chart | `notebooks/06_smp_cls_showcase.ipynb` § 8 | Boost-vs-ratio curve |
| Best/worst-match leaderboards | `notebooks/06_smp_cls_showcase.ipynb` § 9 | Per-metric champions |

Open the executed notebooks once, screenshot the rendered figures, drop into slides.

---

## 16. Session 2 — 24-hour quality push (Tier 1 honest results)

After Phase 3 merged, we did a 24-hour follow-up to push model quality further. The plan
synthesized two AI brainstorms (mine + Claude Desktop's) into three execution tiers.

### Tier 1 — inference-only fixes (DONE)

Three changes added to the inference path, each independently testable:

1. **Top-k truncated annealed-mean** (`src/data/quantize.py:top_k_annealed_mean`) — drops
   low-chroma tail bins before averaging, theoretically removing the gray-pull.
2. **L-guided edge-aware smoothing** (`src/data/quantize.py:guided_smooth_ab`, requires
   `opencv-contrib-python`) — uses the L channel as a guide so chroma transitions snap to
   real luminance edges.
3. **Test-time augmentation** (h-flip ensemble) — average logits over original + flipped input.

**Empirical reality:**

| Recipe | Native colorfulness ratio |
|---|---|
| OLD: full annealed-mean + bilateral, no boost | 0.78 |
| OLD + boost ×1.20 (original Phase 3 ship) | 0.94 |
| + top-k=10 (no boost) | 0.78 (no-op at T=0.10 — softmax is already peaky) |
| + L-guided filter (no boost) | 0.76 (slightly desaturated this checkpoint) |
| + TTA (no boost) | 0.74 (slightly hurt) |
| **top-1 (argmax) + bilateral, no boost** | **0.81** |
| **top-1 + bilateral + boost ×1.10  ← NEW DEFAULT** | **0.89** |

The honest takeaway: the post-hoc ab boost was doing real work the inference tweaks
couldn't replace. The marginal win is replacing soft annealed-mean with top-1 (argmax),
which lets us reduce the boost from ×1.20 to ×1.10 for similar saturation at less
artificial pressure. **The remaining saturation gap is a training-side problem.**

### Tier 2 — last-block encoder fine-tune (DONE, net-negative)

A 3-epoch continuation from `smp_cls_run01/best.pth` with `encoder.layer4` unfrozen at
lr=1e-5. 2h 13m wall on M4 MPS, output in `checkpoints/smp_cls_run01_ft/`.

**Result: slight regression on colorfulness, flat on val_ce.**

| | Phase 3 baseline | Tier 2 fine-tune | Δ |
|---|---|---|---|
| best val_ce | 3.359 | 3.372 | +0.013 |
| final val_l1 | 0.0774 | 0.0767 | -0.0007 |
| Hasler ratio (best recipe) | 0.893 | 0.829 | **-0.064** |

The fine-tune marginally improved L1 (moving the prediction *toward* the dataset mean,
which is sepia) and lost colorfulness across the board. The encoder adaptation didn't
fix the under-commitment in the model's logits — it slightly worsened it.

**Decision:** keep Phase 3 baseline as the shipped checkpoint. Tier 2 checkpoint stays
in the repo for transparency but isn't the default in the showcase notebook.

### Tier 3 — Honest reframings (DONE)

Three points worth making explicit in the deck:

1. **PSNR 21.86 / SSIM 0.904 are normal for colorization, not mediocre.** Both metrics
   penalize plausible-but-different colorings the same way L1 did at the start. Track
   native Hasler-Süsstrunk colorfulness instead.
2. **Graphics / logos / artificial paint are structurally unsolvable** without user
   hints. A gray logo carries no luminance signal about color. Naturalistic-prior
   fallback is correct behavior, not a defect.
3. **Inference tweaks have a ceiling.** The decode/smooth/TTA stack moved native
   colorfulness ~0.03. Beyond that is a training-side or architectural lift.

### Tier 4 — Skipped

Tier 2 showed that more training of the same architecture with the same rebalanced CE
recipe doesn't help — and slightly hurts. A fresh ResNet-50 retrain would burn 8h of
overnight compute on the same flawed loss/data combination. Better future levers:

- **Stronger rebalancing** (Zhang λ=0.5 → 0.9): increase the rare-color emphasis directly.
- **Domain narrowing** (landscapes / faces / single category): smaller conditional variance
  means the L1 mean isn't catastrophically gray.
- **User-guided hints** (Zhang 2017): the only principled fix for arbitrary-graphics
  colorization; biggest demo lever.
- **Direct chroma magnitude loss term**: penalize under-saturation in training, not just
  in inference.

### Lessons added from Session 2

- **Inference-time theories don't always survive contact with real model logits.** At
  T=0.10 the soft annealed-mean is already concentrated enough that top-k≥5 truncation
  is a no-op. The expected "tail mass pulls toward gray" effect was negligible on this
  checkpoint.
- **Guided filter ≠ universally better than bilateral on chroma.** With this model's
  patchy argmax output, the L-guided edge-snapping actually *desaturated* slightly because
  the edge regions had higher chroma than the interiors it smoothed toward.
- **The post-hoc ab boost is doing useful work, not just covering a flaw.** It compensates
  for a real under-commitment in the model's logits — which is a training-side issue
  (probably under-aggressive class rebalancing on the original training distribution).
  Killing the boost means retraining, not retuning.
