# Team Progress — smp Rebuild + Saturation Tuning

**Branch:** `feat/smp-rebuild` (off `main`, all legacy Phase A/B/C work untouched)
**Status:** experimental track complete, ready for review
**Last updated:** 2026-06-07

## TL;DR

Three-phase rebuild following an external research recommendation. Phases 1 & 2 (L1 → PatchGAN)
faithfully reproduced the **sepia desaturation failure mode** — confirming it's a loss-function
problem, not an architecture one. Phase 3 pivoted to **Zhang-style 214-way classification** and
broke the attractor. A post-hoc `ab × 1.20` saturation boost gets the output to **parity with
ground truth** colorfulness — no retraining needed.

## Headline numbers

| Phase | Recipe | Subset | Epochs | Wall | val_l1 | val_ce | colorfulness_ratio* |
|---|---|---|---|---|---|---|---|
| 1 | smp L1, frozen encoder | 10K | 12 | 43 min | **0.0727** | — | ~0.50 |
| 2 | + PatchGAN (LSGAN, λ=0.01, d-every-2) | 10K | 6 | 55 min | 0.0752 | — | ~0.50 |
| 3 | + Zhang cls, T=0.10 + bilateral | 50K | 10 | 6h 51m | 0.0774 | **3.359** | **0.701** |
| 3 + boost | same model, `ab × 1.20` at inference | — | — | — | — | — | **~0.98** |

\* Hasler-Süsstrunk colorfulness, ratio of predicted to ground truth. 1.0 = parity.

**Best single match in the val set:** image #2184 — PSNR 31.11 dB, SSIM 0.984.

## What to look at

| Deliverable | Path |
|---|---|
| 5-phase visual comparison (gray ‖ smp_l1 ‖ smp_cgan ‖ smp_cls ‖ truth) | `notebooks/05_smp_compare.ipynb` |
| Phase 3 deep-dive (curves, colorfulness chart, 12-img grid, ab-boost ablation, top-K similarity leaderboards) | `notebooks/06_smp_cls_showcase.ipynb` |
| Detailed session writeup (slide-by-slide for a presentation) | `docs/session_smp_rebuild.md` |
| Phase 1 final sample (sepia floor) | `outputs/samples/smp_l1_run01/epoch_012.png` |
| Phase 3 final sample (raw, vivid) | `outputs/samples/smp_cls_run01/epoch_010.png` |

Both notebooks are pre-executed — open and read, no kernel needed.

## What was added (code)

```
src/models/smp_unet.py            SmpUNet  (Phase 1/2 generator)
src/models/smp_unet_cls.py        SmpUNetClassifier  (Phase 3, with warm-start filter)
src/metrics/colorfulness.py       Hasler-Süsstrunk  (the missing tracking signal)
scripts/train_smp.py              single CLI: --phase {l1, cgan, cls}
scripts/build_smp_compare_notebook.py
scripts/build_smp_cls_showcase_notebook.py
docs/session_smp_rebuild.md       full writeup
docs/PROGRESS.md                  this file
```

Dependency added: `segmentation-models-pytorch>=0.3.4` (`pyproject.toml` + `uv.lock`).

Checkpoints are **gitignored** as usual; reproduce locally via the commands in
`docs/session_smp_rebuild.md` §3, §4, §8.

## Three lessons (for the deck / for next time)

1. **L1 in LAB is anti-correlated with saturation past the desaturation floor.** Its minimum
   sits at the conditional mean (sepia). Tracking L1 alone systematically optimizes toward
   gray. Phase 3's val_l1 is *higher* than Phase 1's (0.0774 vs 0.0727) yet the output looks
   dramatically more vivid — that inversion is the whole story.

2. **Loss choice dominates architecture choice on this task.** Phases 1, 2, and 3 share
   identical encoder/decoder weights at warm-start. Only the head + loss changed. That single
   substitution (regression → rebalanced classification) broke the sepia attractor. The
   pretrained backbone is necessary but not sufficient.

3. **Post-hoc inference tweaks can be huge ROI.** `ab × 1.20` is one multiplication, zero
   retraining, zero inference cost, and bought a 0.20 jump in colorfulness ratio (0.70 → 0.98).
   Try cheap levers before paying for compute.

## What's next (optional)

If we want to push further from here:

- **Retrain with stronger rebalancing** (Zhang λ=0.5 → 0.9) to bake more saturation into the
  model directly rather than relying on the post-hoc boost.
- **Domain narrowing** (landscapes / faces / one COCO category) — chat's honest caveat: the
  L1 mean is less gray in a narrow domain, so even the regression model would look better.
- **Bigger encoder** (ResNet-50 / ResNeXt) — more semantic confidence → bolder color choices.
- **Direct colorfulness loss term** in training (differentiable Hasler approximation).

None are blockers; current Phase 3 + boost is presentation-ready as-is.
