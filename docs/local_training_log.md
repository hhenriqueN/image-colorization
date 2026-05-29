# Local training log — 2026-05-27 → 2026-05-28

A run-by-run log of the first end-to-end local-training pass after we abandoned
Kaggle. Documents what we tried, what we learned, the bugs we hit, the metrics,
and an honest assessment of where the model still fails so the next iteration
has a clear target.

---

## Why we switched

Kaggle training kept failing (kernel restarts mid-epoch, dataset-mount
friction, session timeouts before convergence). With the deadline close,
we moved everything to local M4 Mac (24 GB unified memory) using PyTorch
2.11's MPS backend.

The original 3-phase plan (vanilla U-Net → ResNet-UNet → cGAN) was too long
end-to-end. We dropped Phase 1 (from-scratch U-Net, 16.6 M params, L1 only)
because it produces washed-out results even when fully trained and would have
eaten hours we didn't have. The Phase 2 model
(`src/models/resnet_unet.py`) already does exactly what we needed — frozen
ImageNet-pretrained ResNet-34 encoder + trainable 3.5 M-param U-Net decoder —
so we made that our **only** baseline and used it as the warm-start for the
cGAN polish phase.

---

## What we added

| File | Purpose |
|---|---|
| `scripts/check_mps.py` | One-shot forward+backward sanity check on MPS (throughput, peak memory, NaN detection) |
| `scripts/train.py` | Headless CLI training for both `resnet_unet` and `cgan`. JSONL metrics, best.pth tracking, per-epoch sample grids, NaN guard, gradient clipping, cosine LR decay, warm-start support |
| `scripts/generate_samples.py` | Standalone qualitative sample-grid generator |
| `scripts/compare_checkpoints.py` | 4-column side-by-side grid (gray \| Phase A \| Phase B \| ground truth) |
| `scripts/build_compare_notebook.py` | Generates the comparison notebook programmatically |
| `notebooks/04_compare_results.ipynb` | Executed notebook with metrics, training curves, visual comparison, and per-image L1 deltas |

Modified:
- `src/losses/perceptual.py` — NaN fix (see below)
- `.gitignore` — added `outputs/` and `logs/`

---

## Bug fixed mid-run: NaN gradient in the perceptual loss on MPS

Symptoms: cycle 1 hit `loss=nan` between step 0 and step 25 of epoch 1, then
NaN cascaded for the rest of training.

Root cause: `lab_to_rgb_tensor` in `src/losses/perceptual.py` had

```python
rgb = torch.where(
    rgb_lin <= 0.0031308,
    12.92 * rgb_lin,
    1.055 * rgb_lin.pow(1.0 / 2.4) - 0.055,
)
```

`rgb_lin.pow(1/2.4)` has gradient proportional to `rgb_lin ** (-0.583)`, which is
**+∞ at `rgb_lin == 0`**. `torch.where` computes gradients through *both*
branches; the false-branch gradient is then multiplied by the (0/1) mask. So
`0 * inf = NaN` leaked through even when the linear branch was the one
selected. CUDA happened to tolerate this in the original 02 notebook; MPS does
not.

Fix: clamp the pow input to a tiny positive epsilon (`safe = rgb_lin.clamp(min=1e-7)`).
The forward output is unchanged because the mask only selects this branch when
`rgb_lin > 0.0031308`. Also added an `isfinite(loss)` guard in `train.py` to
defensively skip any future NaN batch.

---

## Training cycles

Hardware: M4 Mac, 24 GB unified memory. PyTorch 2.11.0 + MPS. All runs wrapped
in `caffeinate -dimsu` to prevent system sleep.

Throughput on synthetic data was ~25 images/sec at batch 16. With real data
(disk read + skimage `rgb2lab` in a single dataloader thread because
`num_workers > 0` hangs on MPS), it dropped to ~22-24 img/sec.

### Cycle 1 — `resnet_unet_run02`

Quick smoke test of the full pipeline.

| Setting | Value |
|---|---|
| Train subset | 3,000 / 74,919 |
| Val subset | 500 / 9,222 |
| Batch | 16 |
| Epochs | 5 |
| LR | 2e-4 (cosine decay from epoch 3) |
| λ_perceptual | 0.1 |
| Wall time | ~12 min |

```
ep 1: val_l1=0.1450
ep 2: val_l1=0.1253
ep 3: val_l1=0.1123
ep 4: val_l1=0.1071
ep 5: val_l1=0.1057   ← best
```

Qualitative: skin tones, sky, foliage, and white walls in the right place from
epoch 1. Everything else muted. Convergence flat by epoch 4 on the 3K subset.

### Cycle 2 — `resnet_unet_run03`

Same recipe, more data + more epochs.

| Setting | Value |
|---|---|
| Train subset | 10,000 |
| Val subset | 500 |
| Batch | 24 |
| Epochs | 8 |
| LR | 2e-4 (cosine decay from epoch 4) |
| λ_perceptual | 0.1 |
| Wall time | ~70 min |

```
ep 1: val_l1=0.1213
ep 2: val_l1=0.1036
ep 3: val_l1=0.0932
ep 4: val_l1=0.0898
ep 5: val_l1=0.0880
ep 6: val_l1=0.0853
ep 7: val_l1=0.0840
ep 8: val_l1=0.0834   ← best, used as Phase B warm start
```

21 % lower val_l1 than cycle 1. Saturation modestly improved on the watched
sample images but still clearly muted vs ground truth.

### Cycle 3 — `cgan_run01` (Phase B polish)

Warm-started from `resnet_unet_run03/best.pth`. cGAN adds PatchGAN discriminator
under LSGAN.

| Setting | Value |
|---|---|
| Train subset | 5,000 |
| Val subset | 500 |
| Batch | 16 |
| Epochs | 8 |
| LR_g / LR_d | 2e-4 / 1e-4 (cosine decay from epoch 4) |
| λ_L1 / λ_perc / λ_adv | 1.0 / 0.1 / 0.01 |
| Wall time | ~70 min |

```
ep 1: val_l1=0.0839 g_adv=0.64 d=0.169
ep 2: val_l1=0.0848 g_adv=0.91 d=0.076   D winning
ep 3: val_l1=0.0852 g_adv=0.92 d=0.069   "
ep 4: val_l1=0.0858 g_adv=0.94 d=0.056   "
ep 5: val_l1=0.0818 g_adv=     d=0.037   LR decay kicks in, val_l1 recovers
ep 6: val_l1=0.0811 g_adv=     d=0.018
ep 7: val_l1=0.0815 g_adv=1.00 d=0.005   D saturated, G trains on L1+perc only
ep 8: val_l1=0.0800 g_adv=     d=0.003   ← best
```

D dynamics were broken — `d_loss` collapsed to ~0 by epoch 4, meaning the
discriminator was solving the task too easily and the adversarial signal to
the generator effectively vanished. From that point on the cGAN was just
"Phase A continuation on a 5K subset", which is why `val_l1` kept improving
slightly. So **the val_l1 gain in Phase B is not really from the adversarial
loss** — it's from the extra training. Saturation did increase modestly on
the watched samples, which is a genuine GAN effect from the first few epochs
where `d_loss` was still nonzero.

---

## Honest assessment of model quality

`val_l1` went from 0.106 → 0.083 → 0.080 over the three cycles, a 25 %
quantitative improvement. But L1 on ab channels is a misleading metric:
predicting the mean color (sepia) is a strong L1 floor and that's roughly
what the model gravitates to on hard inputs.

### What works (out-of-sample val images)

- **Skin / faces** — believable warm tones, lips faintly visible
- **Sky** — pale blue in the right region most of the time
- **Foliage** — green-ish, sometimes correct
- **Indoor walls / neutral scenes** — left alone, which is correct
- **Sunsets / warm lighting** — captures the warm cast

### What doesn't work

This is where the current model still fails badly:

- **Bright artificial paint** — the screenshot of the Capitol-style building
  (val image #377) should be vivid red dome + yellow detailing + blue sky.
  The model produces uniform sepia. Same for the boat (#2486) — should be
  cyan hull and red paint, model gives grayish-beige.
- **Saturated water** — turquoise / vivid blue / green water → muted gray-blue
- **Graphic / illustrated content** — abstract patterns, painted murals, logos
  are not in distribution and the model defaults to brown
- **Specific people-wear colors** — red shirts, blue uniforms come out muted
  because the model can't confidently decide and L1 punishes wrong guesses
  more than muted-but-close guesses

### Why this happens

1. **L1 loss has a sepia attractor.** When the model is uncertain (most of
   the time on unusual subjects), the L1-minimizing answer is the dataset
   mean color, which is brownish-gray.
2. **Perceptual loss is weak (λ=0.1).** VGG features at relu2_2 / relu3_3
   are mostly texture/structure, not chrominance, so they don't push hard
   on color saturation.
3. **GAN training was broken** (D collapsed). The adversarial signal that
   would have pushed against the sepia attractor disappeared after a few
   epochs. The cGAN's small saturation gain comes from epochs 1-3 only.
4. **Only 10K training images** for the most successful Phase A run. The
   full 75K hasn't been used yet on a converged model.

---

## What to try next (concrete next steps)

Ordered by expected impact / cost:

1. **Run Phase A on the full 74,919 training images** for 10-15 epochs.
   Same recipe as cycle 2, just more data. Estimated wall time: ~10-12 hours
   on this M4. Should meaningfully reduce the sepia bias on rare categories.

2. **Re-do the cGAN with a healthier D balance**:
   - Lower `lr_d` from 1e-4 → 3e-5 (slow D learning)
   - Raise `λ_adv` from 0.01 → 0.05 (give G a stronger adversarial signal
     before D saturates)
   - Add a `D_step_every_n=2` schedule (train D every other batch)
   - Restart from Phase A's full-data `best.pth`, not the 10K one.

3. **Unfreeze `enc_layer4`** of the ResNet-34 encoder mid-training (e.g.
   from epoch 5) with a much smaller LR (5e-5). Gives the model some
   freedom to specialize the deepest features for chrominance prediction
   without destroying the ImageNet semantic prior.

4. **Stronger backbone** via `timm` — EfficientNet-B3 or ConvNeXt-Tiny as
   the encoder. Likely the single biggest quality jump available, at the
   cost of adding a new dependency and rewriting the encoder hook-up.

5. **Color-aware reweighted loss** — a per-pixel weight that upweights
   rare colors (high a/b magnitudes) and downweights gray pixels. Standard
   trick in colorization literature; directly attacks the sepia attractor.

6. **Classification reformulation** — instead of regressing ab continuously,
   discretize the ab plane into ~313 bins (Zhang et al. 2016) and predict a
   distribution. Inherently produces saturated samples because the argmax
   is a real color from the dataset, not a regressed mean.

Items 1 and 2 use only the existing code — no architecture changes — so
they're the cheapest first attempts. Items 5 and 6 are the most likely to
break the sepia ceiling permanently but require new loss code.

---

## Files / artifacts to look at

- Checkpoints (gitignored):
  `checkpoints/resnet_unet_run03/best.pth` (Phase A best, val_l1=0.0834)
  `checkpoints/cgan_run01/best_generator.pth` (Phase B best, val_l1=0.0800)
- Comparison notebook: `notebooks/04_compare_results.ipynb` (already
  executed with embedded plots and images, ~21 MB)
- Per-epoch training sample grids: `outputs/samples/{run_name}/epoch_NNN.png`
- Per-run metrics: `checkpoints/{run_name}/log.jsonl` (one JSON object per
  epoch)
