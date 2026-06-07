"""Headless training script for the smp-based colorizer rebuild.

Replays the Claude Desktop chat's recommended recipe:
  - Phase 1 (--phase l1):   smp ResNet-34 U-Net, frozen encoder, pure L1 loss
  - Phase 2 (--phase cgan): same generator warm-started from Phase 1 best.pth,
                            paired with the existing PatchGAN. LSGAN + L1.

Reuses helpers from scripts.train (sample_filenames, cosine_lr, etc.) and
scripts.generate_samples (deterministic_indices, build_sample_grid).

Examples:
    uv run python scripts/train_smp.py \\
        --phase l1 --train-subset 10000 --val-subset 500 \\
        --batch-size 16 --epochs 12 --lr 2e-4 \\
        --checkpoint-dir checkpoints/smp_l1_run01

    uv run python scripts/train_smp.py \\
        --phase cgan --train-subset 10000 --val-subset 500 \\
        --batch-size 16 --epochs 6 \\
        --lr 1e-4 --lr-d 3e-5 --lambda-adv 0.01 --d-step-every 2 \\
        --warm-start checkpoints/smp_l1_run01/best.pth \\
        --checkpoint-dir checkpoints/smp_cgan_run01
"""

from __future__ import annotations

import argparse
import math
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.dataset import AB_MAX, ColorizationDataset, lab_to_rgb  # noqa: E402
from src.data.quantize import ab_to_class, annealed_mean  # noqa: E402
from src.metrics.colorfulness import hasler_susstrunk  # noqa: E402
from src.models.discriminator import PatchGANDiscriminator  # noqa: E402
from src.models.smp_unet import SmpUNet  # noqa: E402
from src.models.smp_unet_cls import SmpUNetClassifier  # noqa: E402

import scripts.generate_samples as samples_module  # noqa: E402
from scripts.train import (  # noqa: E402
    cosine_lr,
    pick_device,
    sample_filenames,
    set_lr,
    set_seed,
    synchronize,
    write_log_line,
    _load_generator_state,
    _trainable_params,
)


@dataclass(frozen=True)
class SmpTrainConfig:
    phase: str
    train_subset: int | None
    val_subset: int | None
    batch_size: int
    epochs: int
    lr: float
    lr_d: float
    lambda_perceptual: float
    lambda_adv: float
    d_step_every: int
    grad_clip: float
    freeze_encoder: bool
    seed: int
    num_workers: int
    sample_every: int
    num_samples: int
    checkpoint_dir: Path
    resume: Path | None
    warm_start: Path | None
    device_str: str | None
    temperature: float = 0.38
    bin_centers_path: Path | None = None
    rebalance_weights_path: Path | None = None
    colorfulness_subset: int = 64


def parse_args() -> SmpTrainConfig:
    p = argparse.ArgumentParser(description="Train smp-based colorizer (L1 → cGAN → cls)")
    p.add_argument("--phase", choices=["l1", "cgan", "cls"], required=True)
    p.add_argument("--train-subset", type=int, default=10000)
    p.add_argument("--val-subset", type=int, default=500)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--epochs", type=int, default=12)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--lr-d", type=float, default=3e-5, help="Discriminator LR (cgan phase)")
    p.add_argument(
        "--lambda-perceptual",
        type=float,
        default=0.0,
        help="VGG-perceptual weight. 0 = pure L1 baseline (chat-recommended).",
    )
    p.add_argument("--lambda-adv", type=float, default=0.01)
    p.add_argument(
        "--d-step-every",
        type=int,
        default=2,
        help="Train discriminator every N batches (>=1). 2 mitigates D collapse.",
    )
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--freeze-encoder", action="store_true", default=True)
    p.add_argument("--unfreeze-encoder", dest="freeze_encoder", action="store_false")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--sample-every", type=int, default=1)
    p.add_argument("--num-samples", type=int, default=8)
    p.add_argument("--checkpoint-dir", type=Path, required=True)
    p.add_argument("--resume", type=Path, default=None)
    p.add_argument("--warm-start", type=Path, default=None)
    p.add_argument("--device", default=None, help="cpu | mps | cuda (default: auto)")
    p.add_argument("--temperature", type=float, default=0.38,
                   help="Annealed-mean softmax T for val_l1 logging in cls phase (Zhang's 0.38).")
    p.add_argument("--bin-centers", type=Path,
                   default=PROJECT_ROOT / "data" / "processed" / "ab_bin_centers.npy",
                   help="Path to precomputed ab bin centers (Q,2 npy) — cls phase.")
    p.add_argument("--rebalance-weights", type=Path,
                   default=PROJECT_ROOT / "data" / "processed" / "ab_rebalance_weights.npy",
                   help="Path to precomputed class rebalance weights (Q,) npy — cls phase.")
    p.add_argument("--colorfulness-subset", type=int, default=64,
                   help="N val images for Hasler-Süsstrunk colorfulness logging — cls phase.")
    args = p.parse_args()

    return SmpTrainConfig(
        phase=args.phase,
        train_subset=args.train_subset,
        val_subset=args.val_subset,
        batch_size=args.batch_size,
        epochs=args.epochs,
        lr=args.lr,
        lr_d=args.lr_d,
        lambda_perceptual=args.lambda_perceptual,
        lambda_adv=args.lambda_adv,
        d_step_every=max(1, args.d_step_every),
        grad_clip=args.grad_clip,
        freeze_encoder=args.freeze_encoder,
        seed=args.seed,
        num_workers=args.num_workers,
        sample_every=args.sample_every,
        num_samples=args.num_samples,
        checkpoint_dir=args.checkpoint_dir,
        resume=args.resume,
        warm_start=args.warm_start,
        device_str=args.device,
        temperature=args.temperature,
        bin_centers_path=args.bin_centers,
        rebalance_weights_path=args.rebalance_weights,
        colorfulness_subset=args.colorfulness_subset,
    )


def build_dataloaders(cfg: SmpTrainConfig) -> tuple[DataLoader, DataLoader]:
    train_files = sample_filenames("train", cfg.train_subset, cfg.seed)
    val_files = sample_filenames("val", cfg.val_subset, cfg.seed)
    train_ds = ColorizationDataset(split="train", filenames=train_files, horizontal_flip=True)
    val_ds = ColorizationDataset(split="val", filenames=val_files, horizontal_flip=False)
    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        pin_memory=False,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=False,
        drop_last=False,
    )
    return train_loader, val_loader


def _save_sample_grid(
    model: nn.Module,
    val_loader: DataLoader,
    cfg: SmpTrainConfig,
    device: torch.device,
    epoch: int,
) -> str | None:
    if cfg.sample_every <= 0:
        return None
    if epoch % cfg.sample_every != 0 and epoch != cfg.epochs:
        return None
    try:
        out_dir = PROJECT_ROOT / "outputs" / "samples" / cfg.checkpoint_dir.name
        out_dir.mkdir(parents=True, exist_ok=True)
        out_png = out_dir / f"epoch_{epoch:03d}.png"
        ds_val = val_loader.dataset
        idxs = samples_module.deterministic_indices(len(ds_val), cfg.num_samples, seed=0)
        grid = samples_module.build_sample_grid(model, ds_val, idxs, device)
        from PIL import Image as _Image

        _Image.fromarray(grid).save(out_png, format="PNG", optimize=True)
        return str(out_png)
    except Exception as exc:  # don't kill training over a sample failure
        print(f"[train-smp] sample generation failed: {exc}", flush=True)
        return None


# ----------------------------------------------------------------------------
# Phase 1: SmpUNet L1 training
# ----------------------------------------------------------------------------


def train_smp_l1(cfg: SmpTrainConfig, device: torch.device) -> None:
    cfg.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    log_path = cfg.checkpoint_dir / "log.jsonl"

    model = SmpUNet(freeze_encoder=cfg.freeze_encoder).to(device)
    l1 = nn.L1Loss()

    if cfg.warm_start is not None:
        _load_generator_state(model, cfg.warm_start, device)
        print(f"[train-smp-l1] warm-started from {cfg.warm_start}")

    optimizer = torch.optim.AdamW(_trainable_params(model), lr=cfg.lr)

    start_epoch = 1
    best_val_l1 = math.inf
    if cfg.resume is not None and cfg.resume.exists():
        ckpt = torch.load(cfg.resume, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])
        if "optimizer_state_dict" in ckpt:
            optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        start_epoch = ckpt.get("epoch", 0) + 1
        best_val_l1 = ckpt.get("best_val_l1", math.inf)
        print(f"[train-smp-l1] resumed from {cfg.resume} at epoch {start_epoch}")

    train_loader, val_loader = build_dataloaders(cfg)
    print(
        f"[train-smp-l1] train={len(train_loader.dataset)} val={len(val_loader.dataset)} "
        f"batches/epoch={len(train_loader)} device={device}"
    )

    decay_start = max(1, cfg.epochs // 2)
    run_start = time.perf_counter()

    for epoch in range(start_epoch, cfg.epochs + 1):
        lr_now = cosine_lr(cfg.lr, epoch - 1, cfg.epochs, decay_start)
        set_lr(optimizer, lr_now)

        model.train()
        epoch_start = time.perf_counter()
        running_train_l1 = 0.0
        n_batches = 0

        for step, (l_in, ab_true) in enumerate(train_loader):
            l_in = l_in.to(device)
            ab_true = ab_true.to(device)

            ab_pred = model(l_in)
            loss = l1(ab_pred, ab_true)

            if not torch.isfinite(loss):
                print(
                    f"[train-smp-l1] WARN non-finite loss at epoch {epoch} step {step}; "
                    f"skipping update (loss={loss.item()})",
                    flush=True,
                )
                optimizer.zero_grad(set_to_none=True)
                continue

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            if cfg.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(_trainable_params(model), cfg.grad_clip)
            optimizer.step()

            running_train_l1 += loss.item()
            n_batches += 1

            if step % 25 == 0:
                print(
                    f"[train-smp-l1] epoch {epoch}/{cfg.epochs} step {step}/{len(train_loader)} "
                    f"l1={loss.item():.4f}",
                    flush=True,
                )

        synchronize(device)
        epoch_time = time.perf_counter() - epoch_start
        train_l1 = running_train_l1 / max(1, n_batches)

        model.eval()
        val_l1_sum = 0.0
        n_val = 0
        with torch.no_grad():
            for l_in, ab_true in val_loader:
                l_in = l_in.to(device)
                ab_true = ab_true.to(device)
                ab_pred = model(l_in)
                val_l1_sum += l1(ab_pred, ab_true).item() * l_in.size(0)
                n_val += l_in.size(0)
        val_l1 = val_l1_sum / max(1, n_val)

        is_best = val_l1 < best_val_l1
        if is_best:
            best_val_l1 = val_l1

        state = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "best_val_l1": best_val_l1,
            "phase": "l1",
        }
        torch.save(state, cfg.checkpoint_dir / "last.pth")
        if is_best:
            torch.save(state, cfg.checkpoint_dir / "best.pth")

        sample_path = _save_sample_grid(model, val_loader, cfg, device, epoch)

        payload = {
            "epoch": epoch,
            "train_l1": train_l1,
            "val_l1": val_l1,
            "best_val_l1": best_val_l1,
            "is_best": is_best,
            "lr": lr_now,
            "epoch_time_s": epoch_time,
            "wall_time_s": time.perf_counter() - run_start,
            "sample_path": sample_path,
        }
        write_log_line(log_path, payload)
        print(
            f"[train-smp-l1] epoch {epoch}/{cfg.epochs} done in {epoch_time:.1f}s — "
            f"train_l1={train_l1:.4f} val_l1={val_l1:.4f} best={best_val_l1:.4f}"
            f"{' [BEST]' if is_best else ''}",
            flush=True,
        )

    print(f"[train-smp-l1] finished. Best val_l1={best_val_l1:.4f}. Checkpoints at {cfg.checkpoint_dir}")


# ----------------------------------------------------------------------------
# Phase 2: SmpUNet + PatchGAN (cGAN) training
# ----------------------------------------------------------------------------


def train_smp_cgan(cfg: SmpTrainConfig, device: torch.device) -> None:
    cfg.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    log_path = cfg.checkpoint_dir / "log.jsonl"

    G = SmpUNet(freeze_encoder=cfg.freeze_encoder).to(device)
    D = PatchGANDiscriminator().to(device)
    l1 = nn.L1Loss()
    mse = nn.MSELoss()  # LSGAN

    if cfg.warm_start is not None:
        _load_generator_state(G, cfg.warm_start, device)
        print(f"[train-smp-cgan] warm-started generator from {cfg.warm_start}")

    opt_g = torch.optim.AdamW(_trainable_params(G), lr=cfg.lr, betas=(0.5, 0.999))
    opt_d = torch.optim.AdamW(D.parameters(), lr=cfg.lr_d, betas=(0.5, 0.999))

    start_epoch = 1
    best_val_l1 = math.inf
    if cfg.resume is not None and cfg.resume.exists():
        ckpt = torch.load(cfg.resume, map_location=device, weights_only=False)
        G.load_state_dict(ckpt["generator_state_dict"])
        D.load_state_dict(ckpt["discriminator_state_dict"])
        opt_g.load_state_dict(ckpt["opt_g_state_dict"])
        opt_d.load_state_dict(ckpt["opt_d_state_dict"])
        start_epoch = ckpt.get("epoch", 0) + 1
        best_val_l1 = ckpt.get("best_val_l1", math.inf)
        print(f"[train-smp-cgan] resumed from {cfg.resume} at epoch {start_epoch}")

    train_loader, val_loader = build_dataloaders(cfg)
    print(
        f"[train-smp-cgan] train={len(train_loader.dataset)} val={len(val_loader.dataset)} "
        f"batches/epoch={len(train_loader)} device={device} d_step_every={cfg.d_step_every}"
    )

    decay_start = max(1, cfg.epochs // 2)
    run_start = time.perf_counter()
    low_d_streak = 0

    for epoch in range(start_epoch, cfg.epochs + 1):
        lr_g_now = cosine_lr(cfg.lr, epoch - 1, cfg.epochs, decay_start)
        lr_d_now = cosine_lr(cfg.lr_d, epoch - 1, cfg.epochs, decay_start)
        set_lr(opt_g, lr_g_now)
        set_lr(opt_d, lr_d_now)

        G.train()
        D.train()
        epoch_start = time.perf_counter()
        run_g_total = run_g_l1 = run_g_adv = run_d = 0.0
        n_d_updates = 0
        n_batches = 0

        for step, (l_in, ab_true) in enumerate(train_loader):
            l_in = l_in.to(device)
            ab_true = ab_true.to(device)

            # ----- Discriminator step (every d_step_every batches) -----
            if step % cfg.d_step_every == 0:
                with torch.no_grad():
                    ab_fake = G(l_in)
                d_real = D(l_in, ab_true)
                d_fake = D(l_in, ab_fake)
                real_target = torch.ones_like(d_real)
                fake_target = torch.zeros_like(d_fake)
                loss_d = 0.5 * (mse(d_real, real_target) + mse(d_fake, fake_target))
                opt_d.zero_grad(set_to_none=True)
                loss_d.backward()
                if cfg.grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(D.parameters(), cfg.grad_clip)
                opt_d.step()
                run_d += loss_d.item()
                n_d_updates += 1

            # ----- Generator step (every batch) -----
            ab_pred = G(l_in)
            d_pred = D(l_in, ab_pred)
            loss_g_adv = mse(d_pred, torch.ones_like(d_pred))
            loss_g_l1 = l1(ab_pred, ab_true)
            loss_g = loss_g_l1 + cfg.lambda_adv * loss_g_adv

            opt_g.zero_grad(set_to_none=True)
            loss_g.backward()
            if cfg.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(_trainable_params(G), cfg.grad_clip)
            opt_g.step()

            run_g_total += loss_g.item()
            run_g_l1 += loss_g_l1.item()
            run_g_adv += loss_g_adv.item()
            n_batches += 1

            if step % 25 == 0:
                d_last = run_d / max(1, n_d_updates)
                print(
                    f"[train-smp-cgan] epoch {epoch}/{cfg.epochs} step {step}/{len(train_loader)} "
                    f"g={loss_g.item():.4f} l1={loss_g_l1.item():.4f} "
                    f"adv={loss_g_adv.item():.4f} d_avg={d_last:.4f}",
                    flush=True,
                )

        synchronize(device)
        epoch_time = time.perf_counter() - epoch_start

        # Validation: L1 on ab
        G.eval()
        val_l1_sum = 0.0
        n_val = 0
        with torch.no_grad():
            for l_in, ab_true in val_loader:
                l_in = l_in.to(device)
                ab_true = ab_true.to(device)
                ab_pred = G(l_in)
                val_l1_sum += l1(ab_pred, ab_true).item() * l_in.size(0)
                n_val += l_in.size(0)
        val_l1 = val_l1_sum / max(1, n_val)

        is_best = val_l1 < best_val_l1
        if is_best:
            best_val_l1 = val_l1

        state = {
            "epoch": epoch,
            "generator_state_dict": G.state_dict(),
            "discriminator_state_dict": D.state_dict(),
            "opt_g_state_dict": opt_g.state_dict(),
            "opt_d_state_dict": opt_d.state_dict(),
            "best_val_l1": best_val_l1,
            "phase": "cgan",
        }
        torch.save(state, cfg.checkpoint_dir / "last.pth")
        if is_best:
            torch.save(
                {"model_state_dict": G.state_dict(), "epoch": epoch},
                cfg.checkpoint_dir / "best_generator.pth",
            )
            torch.save(state, cfg.checkpoint_dir / "best.pth")

        sample_path = _save_sample_grid(G, val_loader, cfg, device, epoch)

        train_d = run_d / max(1, n_d_updates)
        if train_d < 0.05:
            low_d_streak += 1
            if low_d_streak >= 2:
                print(
                    f"[train-smp-cgan] WARN train_d={train_d:.4f} below 0.05 for "
                    f"{low_d_streak} consecutive epochs — possible D collapse.",
                    flush=True,
                )
        else:
            low_d_streak = 0

        payload = {
            "epoch": epoch,
            "train_g": run_g_total / max(1, n_batches),
            "train_g_l1": run_g_l1 / max(1, n_batches),
            "train_g_adv": run_g_adv / max(1, n_batches),
            "train_d": train_d,
            "val_l1": val_l1,
            "best_val_l1": best_val_l1,
            "is_best": is_best,
            "lr_g": lr_g_now,
            "lr_d": lr_d_now,
            "epoch_time_s": epoch_time,
            "wall_time_s": time.perf_counter() - run_start,
            "sample_path": sample_path,
            "d_step_every": cfg.d_step_every,
        }
        write_log_line(log_path, payload)
        print(
            f"[train-smp-cgan] epoch {epoch}/{cfg.epochs} done in {epoch_time:.1f}s — "
            f"val_l1={val_l1:.4f} best={best_val_l1:.4f} d={train_d:.3f}"
            f"{' [BEST]' if is_best else ''}",
            flush=True,
        )

    print(f"[train-smp-cgan] finished. Best val_l1={best_val_l1:.4f}. Checkpoints at {cfg.checkpoint_dir}")


# ----------------------------------------------------------------------------
# Phase 3: SmpUNet classifier (Zhang-style 214-way ab classification)
# ----------------------------------------------------------------------------


def _colorfulness_subset_metrics(
    model: nn.Module,
    val_ds: ColorizationDataset,
    bin_centers_lab: torch.Tensor,
    temperature: float,
    n_images: int,
    device: torch.device,
) -> tuple[float, float, float]:
    """Compute mean Hasler-Süsstrunk colorfulness on a deterministic val subset.

    Returns (pred_colorfulness, true_colorfulness, ratio). The ratio is the
    headline number: 1.0 = pred matches truth, <1.0 = desaturated.
    """
    idxs = samples_module.deterministic_indices(len(val_ds), n_images, seed=0)
    pred_sum = 0.0
    true_sum = 0.0
    model.eval()
    with torch.no_grad():
        for idx in idxs:
            l_t, ab_t = val_ds[idx]
            l_in = l_t.unsqueeze(0).to(device)
            logits = model(l_in)
            ab_lab = annealed_mean(logits, bin_centers_lab, temperature=temperature)
            ab_pred = (ab_lab / AB_MAX).cpu().squeeze(0).clamp(-1, 1)
            rgb_pred = lab_to_rgb(l_t, ab_pred)
            rgb_true = lab_to_rgb(l_t, ab_t)
            pred_sum += hasler_susstrunk(rgb_pred)
            true_sum += hasler_susstrunk(rgb_true)
    pred_mean = pred_sum / max(1, len(idxs))
    true_mean = true_sum / max(1, len(idxs))
    ratio = pred_mean / max(true_mean, 1e-8)
    return pred_mean, true_mean, ratio


def _save_cls_sample_grid(
    model: SmpUNetClassifier,
    val_loader: DataLoader,
    cfg: SmpTrainConfig,
    device: torch.device,
    bin_centers_lab: torch.Tensor,
    epoch: int,
) -> str | None:
    """Render sample grid at the inference recipe (T=0.10 + bilateral) so the
    per-epoch progress shows the actual saturation breakthrough."""
    if cfg.sample_every <= 0:
        return None
    if epoch % cfg.sample_every != 0 and epoch != cfg.epochs:
        return None
    try:
        out_dir = PROJECT_ROOT / "outputs" / "samples" / cfg.checkpoint_dir.name
        out_dir.mkdir(parents=True, exist_ok=True)
        out_png = out_dir / f"epoch_{epoch:03d}.png"
        ds_val = val_loader.dataset
        idxs = samples_module.deterministic_indices(len(ds_val), cfg.num_samples, seed=0)
        grid = samples_module.build_sample_grid(
            model, ds_val, idxs, device,
            bin_centers=bin_centers_lab,
            temperature=0.10,
            bilateral=True,
            bilateral_d=15,
            bilateral_sigma_color=25.0,
            bilateral_sigma_space=10.0,
        )
        from PIL import Image as _Image
        _Image.fromarray(grid).save(out_png, format="PNG", optimize=True)
        return str(out_png)
    except Exception as exc:
        print(f"[train-smp-cls] sample generation failed: {exc}", flush=True)
        return None


def train_smp_cls(cfg: SmpTrainConfig, device: torch.device) -> None:
    cfg.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    log_path = cfg.checkpoint_dir / "log.jsonl"

    if cfg.bin_centers_path is None or not cfg.bin_centers_path.exists():
        raise FileNotFoundError(
            f"Bin centers not found at {cfg.bin_centers_path}. "
            f"Run scripts/precompute_classification_priors.py first."
        )
    if cfg.rebalance_weights_path is None or not cfg.rebalance_weights_path.exists():
        raise FileNotFoundError(
            f"Rebalance weights not found at {cfg.rebalance_weights_path}."
        )
    bin_centers_lab = torch.from_numpy(np.load(cfg.bin_centers_path)).to(torch.float32).to(device)
    rebalance_weights = torch.from_numpy(np.load(cfg.rebalance_weights_path)).to(torch.float32).to(device)
    num_classes = bin_centers_lab.shape[0]
    print(
        f"[train-smp-cls] num_classes={num_classes} bin_centers={cfg.bin_centers_path.name} "
        f"weights mean={rebalance_weights.mean().item():.3f}"
    )

    if cfg.warm_start is not None:
        model = SmpUNetClassifier.from_smp_regression_checkpoint(
            cfg.warm_start,
            num_classes=num_classes,
            freeze_encoder=cfg.freeze_encoder,
            map_location=device,
        ).to(device)
        print(f"[train-smp-cls] warm-started encoder/decoder from {cfg.warm_start} "
              f"(new {num_classes}-way head is random-init)")
    else:
        model = SmpUNetClassifier(
            num_classes=num_classes, freeze_encoder=cfg.freeze_encoder
        ).to(device)

    ce_loss = nn.CrossEntropyLoss(weight=rebalance_weights)
    l1_loss = nn.L1Loss()  # logging only; not optimized
    optimizer = torch.optim.AdamW(_trainable_params(model), lr=cfg.lr)

    start_epoch = 1
    best_val_ce = math.inf
    if cfg.resume is not None and cfg.resume.exists():
        ckpt = torch.load(cfg.resume, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])
        if "optimizer_state_dict" in ckpt:
            optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        start_epoch = ckpt.get("epoch", 0) + 1
        best_val_ce = ckpt.get("best_val_ce", math.inf)
        print(f"[train-smp-cls] resumed from {cfg.resume} at epoch {start_epoch}")

    train_loader, val_loader = build_dataloaders(cfg)
    print(
        f"[train-smp-cls] train={len(train_loader.dataset)} val={len(val_loader.dataset)} "
        f"batches/epoch={len(train_loader)} device={device}"
    )

    decay_start = max(1, cfg.epochs // 2)
    run_start = time.perf_counter()

    for epoch in range(start_epoch, cfg.epochs + 1):
        lr_now = cosine_lr(cfg.lr, epoch - 1, cfg.epochs, decay_start)
        set_lr(optimizer, lr_now)

        model.train()
        epoch_start = time.perf_counter()
        running_train_ce = 0.0
        n_batches = 0

        for step, (l_in, ab_true) in enumerate(train_loader):
            l_in = l_in.to(device)
            ab_true = ab_true.to(device)

            with torch.no_grad():
                target_cls = ab_to_class(ab_true, bin_centers_lab)

            logits = model(l_in)
            loss = ce_loss(logits, target_cls)

            if not torch.isfinite(loss):
                print(
                    f"[train-smp-cls] WARN non-finite loss at epoch {epoch} step {step}; "
                    f"skipping update (loss={loss.item()})",
                    flush=True,
                )
                optimizer.zero_grad(set_to_none=True)
                continue

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            if cfg.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(_trainable_params(model), cfg.grad_clip)
            optimizer.step()

            running_train_ce += loss.item()
            n_batches += 1

            if step % 25 == 0:
                print(
                    f"[train-smp-cls] epoch {epoch}/{cfg.epochs} step {step}/{len(train_loader)} "
                    f"ce={loss.item():.4f}",
                    flush=True,
                )

        synchronize(device)
        epoch_time = time.perf_counter() - epoch_start
        train_ce = running_train_ce / max(1, n_batches)

        # Validation: CE on logits + L1 on annealed-mean ab (cross-phase comparable).
        model.eval()
        val_ce_sum = 0.0
        val_l1_sum = 0.0
        n_val = 0
        with torch.no_grad():
            for l_in, ab_true in val_loader:
                l_in = l_in.to(device)
                ab_true = ab_true.to(device)
                logits = model(l_in)
                target_cls = ab_to_class(ab_true, bin_centers_lab)
                v_ce = ce_loss(logits, target_cls)
                ab_pred = annealed_mean(logits, bin_centers_lab, temperature=cfg.temperature) / AB_MAX
                v_l1 = l1_loss(ab_pred, ab_true)
                val_ce_sum += v_ce.item() * l_in.size(0)
                val_l1_sum += v_l1.item() * l_in.size(0)
                n_val += l_in.size(0)
        val_ce = val_ce_sum / max(1, n_val)
        val_l1 = val_l1_sum / max(1, n_val)

        # Colorfulness on a deterministic val subset (~5 s).
        val_color_pred, val_color_true, color_ratio = _colorfulness_subset_metrics(
            model, val_loader.dataset, bin_centers_lab,
            temperature=cfg.temperature, n_images=cfg.colorfulness_subset, device=device,
        )

        # MPS memory creep insurance for the unattended long run.
        if device.type == "mps":
            torch.mps.empty_cache()

        is_best = val_ce < best_val_ce
        if is_best:
            best_val_ce = val_ce

        state = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "best_val_ce": best_val_ce,
            "num_classes": num_classes,
            "temperature": cfg.temperature,
            "phase": "cls",
        }
        torch.save(state, cfg.checkpoint_dir / "last.pth")
        if is_best:
            torch.save(state, cfg.checkpoint_dir / "best.pth")

        sample_path = _save_cls_sample_grid(
            model, val_loader, cfg, device, bin_centers_lab, epoch,
        )

        payload = {
            "epoch": epoch,
            "train_ce": train_ce,
            "val_ce": val_ce,
            "val_l1": val_l1,
            "val_colorfulness_pred": val_color_pred,
            "val_colorfulness_true": val_color_true,
            "colorfulness_ratio": color_ratio,
            "best_val_ce": best_val_ce,
            "is_best": is_best,
            "lr": lr_now,
            "epoch_time_s": epoch_time,
            "wall_time_s": time.perf_counter() - run_start,
            "sample_path": sample_path,
            "temperature": cfg.temperature,
            "num_classes": num_classes,
        }
        write_log_line(log_path, payload)
        print(
            f"[train-smp-cls] epoch {epoch}/{cfg.epochs} done in {epoch_time:.1f}s — "
            f"train_ce={train_ce:.4f} val_ce={val_ce:.4f} val_l1={val_l1:.4f} "
            f"color_ratio={color_ratio:.3f} best_ce={best_val_ce:.4f}"
            f"{' [BEST]' if is_best else ''}",
            flush=True,
        )

    print(f"[train-smp-cls] finished. Best val_ce={best_val_ce:.4f}. Checkpoints at {cfg.checkpoint_dir}")


def main() -> int:
    cfg = parse_args()
    set_seed(cfg.seed)
    device = pick_device(cfg.device_str)
    print(f"[train-smp] device={device} phase={cfg.phase}")

    if cfg.phase == "l1":
        train_smp_l1(cfg, device)
    elif cfg.phase == "cgan":
        train_smp_cgan(cfg, device)
    elif cfg.phase == "cls":
        train_smp_cls(cfg, device)
    else:
        raise ValueError(f"Unknown phase: {cfg.phase}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
