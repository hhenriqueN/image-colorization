# Team Progress — smp Rebuild + Saturation Tuning

**Branch:** `feat/smp-rebuild` (off `main`, all legacy Phase A/B/C work untouched)
**Status:** Phase 3 merged to `main`; Tier-1 inference fixes + Tier-2 fine-tune in progress
**Last updated:** 2026-06-07

## TL;DR

Three-phase rebuild following an external research recommendation. Phases 1 & 2 (L1 → PatchGAN)
faithfully reproduced the **sepia desaturation failure mode** — confirming it's a loss-function
problem, not an architecture one. Phase 3 pivoted to **Zhang-style 214-way classification** and
broke the attractor.

A 24-hour follow-up push (decode tweaks + L-guided smoothing + TTA + encoder fine-tune)
**didn't change the picture meaningfully**. The encoder fine-tune (Tier 2) was net-negative
for our goal — colorfulness dropped ~0.06 across all recipes while val_ce stayed flat. The
inference tweaks (Tier 1) gave a marginal win: **top-1 (argmax) decode + bilateral +
`ab × 1.10`** matches the original ×1.20 pipeline's saturation at less artificial post-hoc
pressure. Further gains require different methodological levers (stronger rebalancing,
domain narrowing, user hints), not more iterations of the same architecture.

## Headline numbers — native colorfulness ratio (no boost)

Hasler-Süsstrunk colorfulness, ratio of predicted to ground truth. **1.0 = parity.** Native =
no post-hoc `ab` boost.

| Phase / Recipe | Native ratio | With original ×1.20 boost |
|---|---|---|
| Phase 3 OLD: full annealed-mean + bilateral | ~0.78 | ~0.94 |
| Tier-1: top-k=10 + bilateral | ~0.78 | — (no-op vs full-Q at T=0.10) |
| Tier-1: top-k=10 + guided filter | ~0.76 | — (slightly desaturated this checkpoint) |
| Tier-1: + TTA | ~0.74 | — (slightly hurt) |
| Tier-1 **final pick**: top-1 + bilateral | **~0.81** | — |
| Tier-1 final + `ab × 1.10` | **~0.89** | — |

\* Measured on a 30–100 image deterministic val sample. The original Phase 3 sample reported
0.701; the spread reflects different val subsets.

**Best single match:** image #2184 — PSNR 31.11 dB, SSIM 0.984.

## Important reframings (read before treating the numbers as "good/bad")

1. **PSNR / SSIM are not the right metrics for colorization.** They penalize plausible-but-
   different colorings: a perfect *blue* car scored against a *red* ground truth tanks both
   metrics. Automatic colorizers routinely sit at PSNR 20–25 dB. Our 21.86 dB is normal.
   **Track native colorfulness instead.**

2. **L1 in LAB is anti-correlated with saturation past the desaturation floor** because
   L1's minimum lives at the conditional mean (sepia). A model that beats baseline on L1
   may just be matching its desaturation. Phase 3's val_l1 is *higher* than Phase 1's
   (0.0774 vs 0.0727) yet the output looks dramatically more vivid — that inversion is the
   whole story.

3. **Graphics / logos / artificial paint jobs are structurally unsolvable** without user
   hints. A gray logo carries no color information in luminance — the model's
   naturalistic-prior fallback is *correct behavior*, not a defect. The only principled
   fix is allowing color hints (Zhang's 2017 user-guided paper), which we haven't built.

## What's running / done in the 24-hour push

| Tier | What | Status | Honest result |
|---|---|---|---|
| 1 | Inference-only fixes: top-k decode, L-guided filter, TTA | ✅ done | Mostly no-op or slightly negative on this model. The boost is doing most of the work. Marginal win: top-1 decode lets us reduce boost from 1.20 to 1.10 for similar saturation. |
| 2 | Continuation fine-tune: 3 epochs, `encoder.layer4` unfrozen, lr=1e-5 | ✅ done | **Net-negative for our goal.** val_ce flat (3.359 → 3.372), colorfulness ratio dropped ~0.06 across all recipes. The fine-tune pulled colors marginally toward dataset mean. Checkpoint kept for transparency; not shipped. |
| 3 | Docs/deck reframes (this file + session writeup) | ✅ done | — |
| 4 (gated) | Fresh ResNet-50 retrain overnight | ❌ skipped | Tier-2 result suggests more of the same architecture won't help. Defer to a different methodological lever (rebalancing λ, domain narrowing, user hints). |

## What to look at

| Deliverable | Path |
|---|---|
| 5-phase visual comparison (gray ‖ smp_l1 ‖ smp_cgan ‖ smp_cls ‖ truth) | `notebooks/05_smp_compare.ipynb` |
| Phase 3 deep-dive + Tier-1 ablation + boost sweep + best/worst matches | `notebooks/06_smp_cls_showcase.ipynb` (§4b is the new ablation) |
| Detailed session writeup (slide-by-slide for a presentation) | `docs/session_smp_rebuild.md` |
| Phase 1 final sample (sepia floor) | `outputs/samples/smp_l1_run01/epoch_012.png` |
| Phase 3 final sample (raw, vivid) | `outputs/samples/smp_cls_run01/epoch_010.png` |

Both notebooks are pre-executed — open and read, no kernel needed.

## What was added (code)

```
src/models/smp_unet.py            SmpUNet  (Phase 1/2 generator)
src/models/smp_unet_cls.py        SmpUNetClassifier  (Phase 3, + auto head transfer)
src/metrics/colorfulness.py       Hasler-Süsstrunk  (the missing tracking signal)
src/data/quantize.py              + top_k_annealed_mean, + guided_smooth_ab (Tier-1)
scripts/train_smp.py              single CLI: --phase {l1, cgan, cls}, + --unfreeze-after
scripts/generate_samples.py       build_sample_grid: + top_k, use_guided, tta, ab_boost
scripts/build_smp_compare_notebook.py
scripts/build_smp_cls_showcase_notebook.py
docs/session_smp_rebuild.md       full writeup
docs/PROGRESS.md                  this file
```

Dependencies: `segmentation-models-pytorch>=0.3.4`, `opencv-contrib-python>=4.8`.

Checkpoints are **gitignored** as usual; reproduce locally via the commands in
`docs/session_smp_rebuild.md` §3, §4, §8.

## Three lessons (for the deck / for next time)

1. **L1 in LAB is anti-correlated with saturation past the desaturation floor.** Tracking
   it as a quality signal systematically steers toward gray. Use Hasler-Süsstrunk
   colorfulness alongside L1 from day 1.

2. **Loss choice dominates architecture choice on this task.** Phases 1, 2, and 3 share
   identical encoder/decoder weights at warm-start. Only the head + loss changed. That
   single substitution (regression → rebalanced classification) broke the sepia attractor.

3. **Theory-vs-empirics gap: not every clever inference tweak ships.** Tier 1 of the
   24-hour push expected top-k + guided + TTA to remove the post-hoc boost. Empirically,
   the boost was doing real work the inference tweaks couldn't replace. The honest
   takeaway: top-1 decode lets us *reduce* the boost (1.20 → 1.10), not eliminate it.
   The remaining saturation gap is a training-side issue, not a decode one.

## What's next (optional)

If we want to push further from here after Tier 2:

- **Stronger rebalancing** (Zhang λ=0.5 → 0.9) for a fresh ResNet-50 retrain (~8h overnight).
- **Domain narrowing** (landscapes / faces / one COCO category) — would dramatically
  improve vividness in that domain, at the cost of generality.
- **Direct chroma magnitude loss term** in training — penalize under-saturation directly.
- **User-guided hints (Zhang 2017)** — biggest demo lever; ~2 days of work.

None are blockers; current Phase 3 + Tier-1 recipe is presentation-ready as-is.
