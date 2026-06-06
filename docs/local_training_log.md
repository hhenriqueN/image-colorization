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

# Phase C — classification reformulation (Zhang et al. 2016)

We implemented next-step #6 from the "what to try next" list. Inspired by
the Medium post the user shared, which is a tutorial on running Zhang et al.'s
pretrained Caffe weights via OpenCV — we didn't use those weights; we
reimplemented the *technique* in our PyTorch code so we'd have a model we
trained ourselves.

The core idea: don't regress ab — *classify* it. Discretize the ab plane into
a small set of bins, have the model output a per-pixel probability over
those bins, train with cross-entropy weighted to upweight rare colors, and
at inference reconstruct ab via an annealed-mean over the bin centers. This
sidesteps the L1 attractor (which always wants the mean color, ≈ sepia) by
removing L1 from the loss entirely.

## Bin quantization

Computed our own ab gamut from a 5,000-image sample of our training data —
not Zhang's `pts_in_hull.npy` — so the bins reflect Open Images' actual
distribution. With a 5-LAB-unit grid over `[-110, 110]` and a 0.001 % minimum
population threshold, we kept **214 bins out of 1,936 cells** (`scripts/
precompute_classification_priors.py`). The gamut is a long diagonal stripe in
ab space (`data/processed/bin_priors.png`) — most natural images live on the
warm/cool color axis, so the model only needs to discriminate among ~214
realistic colors rather than the full 1,936 grid.

Rebalance weights computed via Zhang's formula
`w(q) ∝ 1 / ((1-λ) p̂(q) + λ/Q)` with λ=0.5, normalized so the weighted
prior sums to 1. Final weights spanned 0.13 (most common bins, near gray) to
6.45 (rarest saturated bins) — about a 50× boost for rare colors during
training.

## Architecture & training

| Item | Value |
|---|---|
| Encoder | Frozen ImageNet ResNet-34 (same as Phase A/B) |
| Decoder | U-Net (same as Phase A/B) |
| Output head | `ConvTranspose2d(128, 214)` with no activation (vs Phase A's 2 + Tanh) |
| Loss | `nn.CrossEntropyLoss(weight=rebalance_weights)` over (N, 214, H, W) logits |
| Warm start | `checkpoints/cgan_run01/best_generator.pth` (Phase B best); only the new 214-channel output head was random-init |
| Train subset | 25,000 / 74,919 |
| Val subset | 500 |
| Batch | 16 |
| Epochs | 10 |
| LR | 3e-4 with cosine decay from epoch 5 |
| Grad clip | 1.0 |
| Wall time | **~6 h 50 min** (~25 min/epoch on M4, vs ~9 min/epoch for Phase A) |

Inference: `annealed_mean(logits, bin_centers, T)` — log-softmax, divide by
`T`, softmax, weighted average over bin centers. **T was the saturation knob.**

## Metrics

```
ep  1: train_ce=3.6318  val_ce=3.5136  val_l1=0.0791  *
ep  2: train_ce=3.4976  val_ce=3.4553  val_l1=0.0790  *
ep  3: train_ce=3.4707  val_ce=3.4430  val_l1=0.0784  *
ep  4: train_ce=3.4546  val_ce=3.4396  val_l1=0.0775  *
ep  5: train_ce=3.4383  val_ce=3.4232  val_l1=0.0779  *  ← LR decay starts
ep  6: train_ce=3.4272  val_ce=3.4150  val_l1=0.0781  *
ep  7: train_ce=3.4120  val_ce=3.3975  val_l1=0.0770  *
ep  8: train_ce=3.3953  val_ce=3.4054  val_l1=0.0777
ep  9: train_ce=3.3741  val_ce=3.3868  val_l1=0.0780  *
ep 10: train_ce=3.3611  val_ce=3.3830  val_l1=0.0775  *  ← best.pth
```

Final `val_l1=0.0775` on the annealed-mean output at training-time T=0.38,
slightly better than Phase B's `0.0800` — a 3 % L1 win, plus the
qualitative win below.

## The temperature finding

T=0.38 (Zhang's default for his 313-bin gamut) gave Phase C samples that
looked almost identical to Phase B's — still muted. The training metrics
hadn't proven that classification was a real upgrade.

**T=0.20 changed everything.** Same trained weights, just a lower softmax
temperature at inference → the annealed-mean concentrates on the
highest-probability bins instead of spreading over many → the output stops
being a weighted-average-of-colors (sepia) and starts being a confident
single-color guess (vivid).

Visible on the 8 held-out comparison images:
- **Capitol-style building** — sepia at T=0.38 → blue sky and warm building
  tones at T=0.20. Phase A and Phase B both miss this.
- **Apple close-up** — washed brown at T=0.38 → vivid red at T=0.20.
- **Zebra-print chair** — muted at T=0.38 → yellow + red tones at T=0.20.
- **Boats** — gray-tan at T=0.38 → cyan water at T=0.20.

T=0.10 is even more vivid but starts to show patchiness (discrete-bin
artifacts where adjacent pixels jump between bins). T=0.20 is the
right tradeoff and is what `04_compare_results.ipynb` now uses.

This is precisely the Zhang-attractor-breaker effect we couldn't get from
adversarial training in Phase B.

## What still doesn't work

- **214 bins is coarse** — bin width is 5 LAB units, so the model can't
  represent fine color gradations within a single bin. With 313+ bins (a
  finer grid) we'd get smoother gradients.
- **Patchiness at low T** — discrete-bin artifacts. Zhang addressed this in
  his original paper with a bilateral filter post-processing step; we
  haven't implemented that.
- **The metric story is muddled** — `val_l1` improved only 3 %, which is
  not what an undergraduate course rubric would call a dramatic gain.
  Visual quality is genuinely better but if you're being scored on a single
  number, Phase C looks marginal. The right framing is: "Phase C *finally
  gets specific colors right* on inputs where A/B unfailingly produced
  sepia." Pure L1 numbers don't capture that.

## What worked well

- **Warm-starting from Phase B's weights** kept the spatial-structure quality
  from L1+perceptual training while the new classification head learned to
  emit distributions instead of point estimates. Convergence was reasonable
  in 10 epochs.
- **Computing our own gamut** (not using Zhang's ImageNet pts_in_hull) gave
  a tighter, dataset-matched bin set. The Open Images distribution is
  mostly nature/people, not Zhang's broader ImageNet, so a smaller
  data-fitted gamut was the right call.
- **The MPS NaN fix from Phase A** carried over — no new numerical issues
  in cross-entropy training.

## Update — Bilateral post-processing (2026-06-06)

Implemented step 1 of the "next steps" list: bilateral filter on the
annealed-mean ab output. This is the standard fix for discrete-bin
patchiness at low temperatures.

`src/data/quantize.py::bilateral_smooth_ab` runs `cv2.bilateralFilter`
on each ab channel of each image independently (no joint-bilateral
because `cv2.ximgproc.jointBilateralFilter` isn't in the wheel we have;
vanilla bilateral on ab was enough). Wired through `generate_samples.py`
as `--bilateral` and into the notebook's predict path.

**Parameter sweep at T=0.10:**
| diameter | sigma_color | sigma_space | result |
|---|---|---|---|
| 9 | 15 | 9 | mild smoothing — patchiness still visible on apple, zebra chair |
| **15** | **25** | **10** | **clean — chosen recipe** |
| 21 | 40 | 12 | nearly identical to d=15; diminishing returns |

Also tested **T=0.05 + bilateral d=15** — even more vivid, still clean,
but starting to over-saturate naturally-muted scenes (sky becomes too
purple, foliage too yellow). Kept T=0.10 as the safer default.

**Winning recipe**: `annealed_mean(T=0.10)` → `bilateral_smooth_ab(d=15,
sigma_color=25, sigma_space=10)`. This is what
`notebooks/04_compare_results.ipynb` now displays in the Phase C column.

Compared to T=0.20 (the previous notebook default), the new recipe:
- preserves the cyan boats, vivid red apple, blue Capitol sky etc. that
  T=0.20 already produced
- pushes saturation noticeably further on the zebra-print chair (yellow
  + red separation is more confident), the rollercoaster clothing, and
  rare-color regions in general
- has no visible patchiness — the bilateral kills the bin-jump artifacts
  cleanly without bleeding across edges

The improvement is real but incremental — Phase C with bilateral is
better than Phase C without, but the much bigger jumps in the project
were Phase A → A2 (more data) and the temperature finding itself
(T=0.38 → T=0.20).

## Next steps if we keep iterating

In order (1 done, 2-4 remaining):
1. ~~Bilateral filter post-processing~~ ✅ done 2026-06-06.
2. **Finer bin grid (step 4, ~350 bins)** for smoother color gradations.
3. **Drop the warm-start** and train Phase C from a Phase-A-warm start
   instead — the cGAN's bias may be pulling the classification head toward
   conservative outputs.
4. **Train on the full 75 K images** for ≥ 20 epochs. Phase C with a
   bigger dataset should generalize better to the rare-color tail that
   rebalancing tries to upweight.
5. *(new)* **Joint bilateral filter** with the L channel as guide. Our
   vanilla bilateral works without L guidance because the ab channels
   already encode chrominance edges, but JBF would preserve subtler
   luminance-only edges (specular highlights, shadow boundaries). Would
   need `opencv-contrib-python` or a PyTorch implementation.

---

## Files / artifacts to look at

- Checkpoints (gitignored):
  `checkpoints/resnet_unet_run03/best.pth` (Phase A best, val_l1=0.0834)
  `checkpoints/cgan_run01/best_generator.pth` (Phase B best, val_l1=0.0800)
  `checkpoints/cls_run01/best.pth` (Phase C best, val_l1=0.0775 @ T=0.38)
- Phase C priors (gitignored):
  `data/processed/ab_bin_centers.npy` (214 × 2)
  `data/processed/ab_rebalance_weights.npy` (214,)
  `data/processed/bin_priors.png` (gamut heatmap, sanity check)
- Comparison notebook: `notebooks/04_compare_results.ipynb`
  (now includes Phase C as a 4th model column, executed in-place)
- Per-epoch training sample grids: `outputs/samples/{run_name}/epoch_NNN.png`
- Phase C temperature comparisons:
  `outputs/samples/cls_run01/best_T038.png` (default, muted)
  `outputs/samples/cls_run01/best_T020.png` (previous notebook recipe)
  `outputs/samples/cls_run01/best_T010.png` (vivid, patchy without filter)
  `outputs/samples/cls_run01/T010_bil_d15_sc25_ss10.png` (current winner: vivid + smooth)
  `outputs/samples/cls_run01/T005_bil_d15_sc30_ss12.png` (over-saturates muted scenes)
  `outputs/samples/cls_run01/T005_nobil.png` (max patchiness reference)
- Per-run metrics: `checkpoints/{run_name}/log.jsonl` (one JSON object per
  epoch)
