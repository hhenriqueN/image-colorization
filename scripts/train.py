"""Headless training script for image colorization.

Supports three model paths:
  - resnet_unet     : ResNet-34 encoder (frozen by default) + U-Net decoder
                      Loss = L1 + lambda_perceptual * VGG-perceptual
  - cgan            : Same generator + PatchGAN discriminator
                      Loss = L1 + lambda_perceptual * VGG + lambda_adv * LSGAN
  - resnet_unet_cls : Same encoder/decoder, but the final layer outputs Q logits
                      per pixel (Zhang-style classification). Loss = weighted
                      cross-entropy over precomputed ab bins. Best.pth is
                      tracked by val_ce; val_l1 (annealed-mean) is also logged.

Designed to run in the background. Writes progress to a JSONL log file
in the checkpoint dir, prints sparse human-readable lines to stdout,
and saves best.pth and last.pth every epoch.

Examples:
    uv run python scripts/train.py \\
        --model resnet_unet --train-subset 3000 --val-subset 500 \\
        --batch-size 16 --epochs 5 --lr 2e-4 \\
        --checkpoint-dir checkpoints/resnet_unet_run01

    uv run python scripts/train.py \\
        --model cgan --train-subset 5000 --val-subset 500 \\
        --batch-size 8 --epochs 8 \\
        --warm-start checkpoints/resnet_unet_run01/best.pth \\
        --checkpoint-dir checkpoints/cgan_run01

    uv run python scripts/train.py \\
        --model resnet_unet_cls --train-subset 25000 --val-subset 500 \\
        --batch-size 16 --epochs 10 --lr 3e-4 \\
        --warm-start checkpoints/cgan_run01/best_generator.pth \\
        --checkpoint-dir checkpoints/cls_run01
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
import time
from dataclasses import dataclass
from pathlib import Path

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.dataset import (  # noqa: E402
    AB_MAX,
    DEFAULT_PROCESSED_DIR,
    ColorizationDataset,
    lab_to_rgb,
)
from src.data.quantize import ab_to_class, annealed_mean  # noqa: E402
from src.losses.perceptual import PerceptualLoss  # noqa: E402
from src.models.discriminator import PatchGANDiscriminator  # noqa: E402
from src.models.resnet_unet import ResNetUNet  # noqa: E402
from src.models.resnet_unet_cls import ResNetUNetClassifier  # noqa: E402

import scripts.generate_samples as samples_module  # noqa: E402


@dataclass(frozen=True)
class TrainConfig:
    model: str
    train_subset: int | None
    val_subset: int | None
    batch_size: int
    epochs: int
    lr: float
    lr_d: float
    lambda_perceptual: float
    lambda_adv: float
    grad_clip: float
    freeze_encoder: bool
    unfreeze_after: int | None
    seed: int
    num_workers: int
    sample_every: int
    num_samples: int
    checkpoint_dir: Path
    resume: Path | None
    warm_start: Path | None
    device_str: str | None
    temperature: float
    bin_centers_path: Path
    rebalance_weights_path: Path


def parse_args() -> TrainConfig:
    p = argparse.ArgumentParser(description="Train colorizer")
    p.add_argument("--model", choices=["resnet_unet", "cgan", "resnet_unet_cls"], default="resnet_unet")
    p.add_argument("--train-subset", type=int, default=3000)
    p.add_argument("--val-subset", type=int, default=500)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--lr-d", type=float, default=1e-4, help="Discriminator LR for cgan")
    p.add_argument("--lambda-perceptual", type=float, default=0.1)
    p.add_argument("--lambda-adv", type=float, default=0.01)
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--freeze-encoder", action="store_true", default=True)
    p.add_argument("--unfreeze-encoder", dest="freeze_encoder", action="store_false")
    p.add_argument(
        "--unfreeze-after",
        type=int,
        default=None,
        help="Epoch number (1-indexed) after which to unfreeze encoder",
    )
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--sample-every", type=int, default=1, help="Generate samples every N epochs")
    p.add_argument("--num-samples", type=int, default=8)
    p.add_argument("--checkpoint-dir", type=Path, required=True)
    p.add_argument("--resume", type=Path, default=None)
    p.add_argument("--warm-start", type=Path, default=None)
    p.add_argument("--device", default=None, help="cpu | mps | cuda (default: auto)")
    p.add_argument("--temperature", type=float, default=0.38,
                   help="Annealed-mean softmax temperature for resnet_unet_cls (Zhang's 0.38).")
    p.add_argument("--bin-centers", type=Path,
                   default=DEFAULT_PROCESSED_DIR / "ab_bin_centers.npy",
                   help="Path to precomputed ab bin centers (Q,2 npy).")
    p.add_argument("--rebalance-weights", type=Path,
                   default=DEFAULT_PROCESSED_DIR / "ab_rebalance_weights.npy",
                   help="Path to precomputed class rebalance weights (Q,) npy.")
    args = p.parse_args()

    return TrainConfig(
        model=args.model,
        train_subset=args.train_subset,
        val_subset=args.val_subset,
        batch_size=args.batch_size,
        epochs=args.epochs,
        lr=args.lr,
        lr_d=args.lr_d,
        lambda_perceptual=args.lambda_perceptual,
        lambda_adv=args.lambda_adv,
        grad_clip=args.grad_clip,
        freeze_encoder=args.freeze_encoder,
        unfreeze_after=args.unfreeze_after,
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
    )


def pick_device(requested: str | None) -> torch.device:
    if requested:
        return torch.device(requested)
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def sample_filenames(
    split: str,
    n: int | None,
    seed: int,
    processed_dir: Path = DEFAULT_PROCESSED_DIR,
) -> list[str] | None:
    if n is None:
        return None
    manifest = pd.read_csv(processed_dir / "manifest.csv")
    in_split = manifest[manifest["split"] == split]
    if len(in_split) <= n:
        return in_split["filename"].tolist()
    rng = np.random.default_rng(seed)
    chosen = rng.choice(in_split["filename"].to_numpy(), size=n, replace=False)
    return sorted(chosen.tolist())


def build_dataloaders(
    cfg: TrainConfig,
) -> tuple[DataLoader, DataLoader]:
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


def write_log_line(log_path: Path, payload: dict) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a") as f:
        f.write(json.dumps(payload) + "\n")


def synchronize(device: torch.device) -> None:
    if device.type == "mps":
        torch.mps.synchronize()
    elif device.type == "cuda":
        torch.cuda.synchronize()


def _trainable_params(model: nn.Module):
    return (p for p in model.parameters() if p.requires_grad)


def _load_generator_state(model: ResNetUNet, path: Path, device: torch.device) -> None:
    state = torch.load(path, map_location=device, weights_only=False)
    if isinstance(state, dict) and "model_state_dict" in state:
        model.load_state_dict(state["model_state_dict"])
    else:
        model.load_state_dict(state)


def cosine_lr(initial_lr: float, epoch: int, total_epochs: int, decay_start: int) -> float:
    if epoch < decay_start:
        return initial_lr
    progress = (epoch - decay_start) / max(1, total_epochs - decay_start)
    progress = min(max(progress, 0.0), 1.0)
    return initial_lr * 0.5 * (1.0 + math.cos(math.pi * progress))


def set_lr(optimizer: torch.optim.Optimizer, lr: float) -> None:
    for pg in optimizer.param_groups:
        pg["lr"] = lr


# ----------------------------------------------------------------------------
# Phase A: ResNetUNet training
# ----------------------------------------------------------------------------


def train_resnet_unet(cfg: TrainConfig, device: torch.device) -> None:
    cfg.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    log_path = cfg.checkpoint_dir / "log.jsonl"

    model = ResNetUNet(freeze_encoder=cfg.freeze_encoder).to(device)
    perceptual = PerceptualLoss().to(device) if cfg.lambda_perceptual > 0 else None
    l1 = nn.L1Loss()

    if cfg.warm_start is not None:
        _load_generator_state(model, cfg.warm_start, device)
        print(f"[train] warm-started from {cfg.warm_start}")

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
        print(f"[train] resumed from {cfg.resume} at epoch {start_epoch}")

    train_loader, val_loader = build_dataloaders(cfg)
    print(
        f"[train] train={len(train_loader.dataset)} val={len(val_loader.dataset)} "
        f"batches/epoch={len(train_loader)} device={device}"
    )

    decay_start = max(1, cfg.epochs // 2)
    run_start = time.perf_counter()

    for epoch in range(start_epoch, cfg.epochs + 1):
        if cfg.unfreeze_after is not None and epoch == cfg.unfreeze_after + 1:
            model.unfreeze_encoder()
            optimizer = torch.optim.AdamW(_trainable_params(model), lr=cfg.lr * 0.5)
            print(f"[train] unfroze encoder at epoch {epoch}; rebuilt optimizer with lr={cfg.lr * 0.5}")

        lr_now = cosine_lr(cfg.lr, epoch - 1, cfg.epochs, decay_start)
        set_lr(optimizer, lr_now)

        model.train()
        epoch_start = time.perf_counter()
        running_train_loss = 0.0
        running_train_l1 = 0.0
        n_batches = 0

        for step, (l_in, ab_true) in enumerate(train_loader):
            l_in = l_in.to(device, non_blocking=False)
            ab_true = ab_true.to(device, non_blocking=False)

            ab_pred = model(l_in)
            loss_l1 = l1(ab_pred, ab_true)
            loss = loss_l1
            if perceptual is not None:
                loss = loss + cfg.lambda_perceptual * perceptual(l_in, ab_pred, l_in, ab_true)

            if not torch.isfinite(loss):
                print(
                    f"[train] WARN non-finite loss at epoch {epoch} step {step}; "
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

            running_train_loss += loss.item()
            running_train_l1 += loss_l1.item()
            n_batches += 1

            if step % 25 == 0:
                print(
                    f"[train] epoch {epoch}/{cfg.epochs} step {step}/{len(train_loader)} "
                    f"loss={loss.item():.4f} l1={loss_l1.item():.4f}",
                    flush=True,
                )

        synchronize(device)
        epoch_time = time.perf_counter() - epoch_start
        train_loss = running_train_loss / max(1, n_batches)
        train_l1 = running_train_l1 / max(1, n_batches)

        # Validation
        model.eval()
        val_l1_sum = 0.0
        val_loss_sum = 0.0
        n_val = 0
        with torch.no_grad():
            for l_in, ab_true in val_loader:
                l_in = l_in.to(device)
                ab_true = ab_true.to(device)
                ab_pred = model(l_in)
                v_l1 = l1(ab_pred, ab_true)
                v_total = v_l1
                if perceptual is not None:
                    v_total = v_total + cfg.lambda_perceptual * perceptual(l_in, ab_pred, l_in, ab_true)
                val_l1_sum += v_l1.item() * l_in.size(0)
                val_loss_sum += v_total.item() * l_in.size(0)
                n_val += l_in.size(0)

        val_l1 = val_l1_sum / max(1, n_val)
        val_loss = val_loss_sum / max(1, n_val)

        is_best = val_l1 < best_val_l1
        if is_best:
            best_val_l1 = val_l1

        state = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "best_val_l1": best_val_l1,
            "cfg": cfg.__dict__,
        }
        torch.save(state, cfg.checkpoint_dir / "last.pth")
        if is_best:
            torch.save(state, cfg.checkpoint_dir / "best.pth")

        # Sample grid
        sample_path: str | None = None
        if cfg.sample_every > 0 and (epoch % cfg.sample_every == 0 or epoch == cfg.epochs):
            try:
                out_dir = PROJECT_ROOT / "outputs" / "samples" / cfg.checkpoint_dir.name
                out_dir.mkdir(parents=True, exist_ok=True)
                out_png = out_dir / f"epoch_{epoch:03d}.png"
                ds_val = val_loader.dataset
                idxs = samples_module.deterministic_indices(len(ds_val), cfg.num_samples, seed=0)
                grid = samples_module.build_sample_grid(model, ds_val, idxs, device)
                from PIL import Image as _Image
                _Image.fromarray(grid).save(out_png, format="PNG", optimize=True)
                sample_path = str(out_png)
            except Exception as exc:  # don't kill training over a sample failure
                print(f"[train] sample generation failed: {exc}", flush=True)

        payload = {
            "epoch": epoch,
            "train_loss": train_loss,
            "train_l1": train_l1,
            "val_loss": val_loss,
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
            f"[train] epoch {epoch}/{cfg.epochs} done in {epoch_time:.1f}s — "
            f"train_l1={train_l1:.4f} val_l1={val_l1:.4f} best={best_val_l1:.4f}"
            f"{' [BEST]' if is_best else ''}",
            flush=True,
        )

    print(f"[train] finished. Best val_l1={best_val_l1:.4f}. Checkpoints at {cfg.checkpoint_dir}")


# ----------------------------------------------------------------------------
# Phase B: cGAN training
# ----------------------------------------------------------------------------


def train_cgan(cfg: TrainConfig, device: torch.device) -> None:
    cfg.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    log_path = cfg.checkpoint_dir / "log.jsonl"

    G = ResNetUNet(freeze_encoder=cfg.freeze_encoder).to(device)
    D = PatchGANDiscriminator().to(device)
    perceptual = PerceptualLoss().to(device) if cfg.lambda_perceptual > 0 else None
    l1 = nn.L1Loss()
    mse = nn.MSELoss()  # LSGAN

    if cfg.warm_start is not None:
        _load_generator_state(G, cfg.warm_start, device)
        print(f"[train-cgan] warm-started generator from {cfg.warm_start}")

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
        print(f"[train-cgan] resumed from {cfg.resume} at epoch {start_epoch}")

    train_loader, val_loader = build_dataloaders(cfg)
    print(
        f"[train-cgan] train={len(train_loader.dataset)} val={len(val_loader.dataset)} "
        f"batches/epoch={len(train_loader)} device={device}"
    )

    decay_start = max(1, cfg.epochs // 2)
    run_start = time.perf_counter()

    for epoch in range(start_epoch, cfg.epochs + 1):
        lr_g_now = cosine_lr(cfg.lr, epoch - 1, cfg.epochs, decay_start)
        lr_d_now = cosine_lr(cfg.lr_d, epoch - 1, cfg.epochs, decay_start)
        set_lr(opt_g, lr_g_now)
        set_lr(opt_d, lr_d_now)

        G.train()
        D.train()
        epoch_start = time.perf_counter()
        run_g_total = run_g_l1 = run_g_perc = run_g_adv = run_d = 0.0
        n_batches = 0

        for step, (l_in, ab_true) in enumerate(train_loader):
            l_in = l_in.to(device)
            ab_true = ab_true.to(device)

            # ----- Discriminator step -----
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

            # ----- Generator step -----
            ab_pred = G(l_in)
            d_pred = D(l_in, ab_pred)
            loss_g_adv = mse(d_pred, torch.ones_like(d_pred))
            loss_g_l1 = l1(ab_pred, ab_true)
            loss_g_perc = (
                perceptual(l_in, ab_pred, l_in, ab_true) if perceptual is not None else torch.tensor(0.0, device=device)
            )
            loss_g = (
                loss_g_l1
                + cfg.lambda_perceptual * loss_g_perc
                + cfg.lambda_adv * loss_g_adv
            )
            opt_g.zero_grad(set_to_none=True)
            loss_g.backward()
            if cfg.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(_trainable_params(G), cfg.grad_clip)
            opt_g.step()

            run_g_total += loss_g.item()
            run_g_l1 += loss_g_l1.item()
            run_g_perc += loss_g_perc.item()
            run_g_adv += loss_g_adv.item()
            run_d += loss_d.item()
            n_batches += 1

            if step % 25 == 0:
                print(
                    f"[train-cgan] epoch {epoch}/{cfg.epochs} step {step}/{len(train_loader)} "
                    f"g={loss_g.item():.4f} l1={loss_g_l1.item():.4f} "
                    f"adv={loss_g_adv.item():.4f} d={loss_d.item():.4f}",
                    flush=True,
                )

        synchronize(device)
        epoch_time = time.perf_counter() - epoch_start

        # Validation: L1 on ab (qualitative target for "best" still uses L1)
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
            "cfg": cfg.__dict__,
        }
        torch.save(state, cfg.checkpoint_dir / "last.pth")
        if is_best:
            # also save generator-only for downstream use
            torch.save({"model_state_dict": G.state_dict(), "epoch": epoch}, cfg.checkpoint_dir / "best_generator.pth")
            torch.save(state, cfg.checkpoint_dir / "best.pth")

        sample_path: str | None = None
        if cfg.sample_every > 0 and (epoch % cfg.sample_every == 0 or epoch == cfg.epochs):
            try:
                out_dir = PROJECT_ROOT / "outputs" / "samples" / cfg.checkpoint_dir.name
                out_dir.mkdir(parents=True, exist_ok=True)
                out_png = out_dir / f"epoch_{epoch:03d}.png"
                ds_val = val_loader.dataset
                idxs = samples_module.deterministic_indices(len(ds_val), cfg.num_samples, seed=0)
                grid = samples_module.build_sample_grid(G, ds_val, idxs, device)
                from PIL import Image as _Image
                _Image.fromarray(grid).save(out_png, format="PNG", optimize=True)
                sample_path = str(out_png)
            except Exception as exc:
                print(f"[train-cgan] sample generation failed: {exc}", flush=True)

        payload = {
            "epoch": epoch,
            "train_g": run_g_total / max(1, n_batches),
            "train_g_l1": run_g_l1 / max(1, n_batches),
            "train_g_perc": run_g_perc / max(1, n_batches),
            "train_g_adv": run_g_adv / max(1, n_batches),
            "train_d": run_d / max(1, n_batches),
            "val_l1": val_l1,
            "best_val_l1": best_val_l1,
            "is_best": is_best,
            "lr_g": lr_g_now,
            "lr_d": lr_d_now,
            "epoch_time_s": epoch_time,
            "wall_time_s": time.perf_counter() - run_start,
            "sample_path": sample_path,
        }
        write_log_line(log_path, payload)
        print(
            f"[train-cgan] epoch {epoch}/{cfg.epochs} done in {epoch_time:.1f}s — "
            f"val_l1={val_l1:.4f} best={best_val_l1:.4f} d={payload['train_d']:.3f}"
            f"{' [BEST]' if is_best else ''}",
            flush=True,
        )

    print(f"[train-cgan] finished. Best val_l1={best_val_l1:.4f}. Checkpoints at {cfg.checkpoint_dir}")


# ----------------------------------------------------------------------------
# Phase C: ResNetUNet classifier training (Zhang-style)
# ----------------------------------------------------------------------------


def _annealed_mean_normalized(
    logits: torch.Tensor,
    bin_centers_lab: torch.Tensor,
    temperature: float,
) -> torch.Tensor:
    """Annealed-mean (LAB units) → renormalized to [-1, 1] for downstream LAB→RGB."""
    ab_lab = annealed_mean(logits, bin_centers_lab, temperature=temperature)
    return ab_lab / AB_MAX


@torch.no_grad()
def _cls_sample_grid(
    model: ResNetUNetClassifier,
    dataset: ColorizationDataset,
    indices: list[int],
    device: torch.device,
    bin_centers_lab: torch.Tensor,
    temperature: float,
) -> "np.ndarray":
    """Reuse generate_samples.build_sample_grid logic but route through annealed-mean."""
    rows = []
    model.eval()
    for idx in indices:
        l_tensor, ab_true = dataset[idx]
        l_in = l_tensor.unsqueeze(0).to(device)
        logits = model(l_in)
        ab_pred = _annealed_mean_normalized(logits, bin_centers_lab, temperature).cpu().squeeze(0).clamp(-1, 1)

        gray_panel = samples_module._grayscale_rgb_from_l(l_tensor)
        pred_panel = lab_to_rgb(l_tensor, ab_pred)
        true_panel = lab_to_rgb(l_tensor, ab_true)
        rows.append(samples_module._hstack_with_separator([gray_panel, pred_panel, true_panel]))
    return samples_module._vstack_with_separator(rows)


def train_resnet_unet_cls(cfg: TrainConfig, device: torch.device) -> None:
    cfg.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    log_path = cfg.checkpoint_dir / "log.jsonl"

    if not cfg.bin_centers_path.exists():
        raise FileNotFoundError(
            f"Bin centers not found at {cfg.bin_centers_path}. "
            f"Run scripts/precompute_classification_priors.py first."
        )
    bin_centers_lab = torch.from_numpy(np.load(cfg.bin_centers_path)).to(torch.float32).to(device)
    rebalance_weights = torch.from_numpy(np.load(cfg.rebalance_weights_path)).to(torch.float32).to(device)
    num_classes = bin_centers_lab.shape[0]
    print(f"[train-cls] num_classes={num_classes} bin_centers={cfg.bin_centers_path.name} "
          f"weights mean={rebalance_weights.mean().item():.3f}")

    if cfg.warm_start is not None:
        model = ResNetUNetClassifier.from_regression_checkpoint(
            cfg.warm_start, num_classes=num_classes,
            freeze_encoder=cfg.freeze_encoder, map_location=device,
        ).to(device)
        print(f"[train-cls] warm-started encoder/decoder from {cfg.warm_start} "
              f"(new {num_classes}-way head is random-init)")
    else:
        model = ResNetUNetClassifier(num_classes=num_classes, freeze_encoder=cfg.freeze_encoder).to(device)

    ce_loss = nn.CrossEntropyLoss(weight=rebalance_weights)
    l1_loss = nn.L1Loss()  # only for logging/comparison; not optimized
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
        print(f"[train-cls] resumed from {cfg.resume} at epoch {start_epoch}")

    train_loader, val_loader = build_dataloaders(cfg)
    print(
        f"[train-cls] train={len(train_loader.dataset)} val={len(val_loader.dataset)} "
        f"batches/epoch={len(train_loader)} device={device}"
    )

    decay_start = max(1, cfg.epochs // 2)
    run_start = time.perf_counter()

    for epoch in range(start_epoch, cfg.epochs + 1):
        if cfg.unfreeze_after is not None and epoch == cfg.unfreeze_after + 1:
            model.unfreeze_encoder()
            optimizer = torch.optim.AdamW(_trainable_params(model), lr=cfg.lr * 0.5)
            print(f"[train-cls] unfroze encoder at epoch {epoch}; lr={cfg.lr * 0.5}")

        lr_now = cosine_lr(cfg.lr, epoch - 1, cfg.epochs, decay_start)
        set_lr(optimizer, lr_now)

        model.train()
        epoch_start = time.perf_counter()
        running_train_ce = 0.0
        n_batches = 0

        for step, (l_in, ab_true) in enumerate(train_loader):
            l_in = l_in.to(device)
            ab_true = ab_true.to(device)

            # Vectorized nearest-bin lookup per pixel; (N, H, W) class indices.
            with torch.no_grad():
                target_cls = ab_to_class(ab_true, bin_centers_lab)

            logits = model(l_in)
            loss = ce_loss(logits, target_cls)

            if not torch.isfinite(loss):
                print(
                    f"[train-cls] WARN non-finite loss at epoch {epoch} step {step}; "
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
                    f"[train-cls] epoch {epoch}/{cfg.epochs} step {step}/{len(train_loader)} "
                    f"ce={loss.item():.4f}",
                    flush=True,
                )

        synchronize(device)
        epoch_time = time.perf_counter() - epoch_start
        train_ce = running_train_ce / max(1, n_batches)

        # Validation: CE on logits + L1 on annealed-mean ab (for cross-phase comparison).
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
                ab_pred = _annealed_mean_normalized(logits, bin_centers_lab, cfg.temperature)
                v_l1 = l1_loss(ab_pred, ab_true)
                val_ce_sum += v_ce.item() * l_in.size(0)
                val_l1_sum += v_l1.item() * l_in.size(0)
                n_val += l_in.size(0)
        val_ce = val_ce_sum / max(1, n_val)
        val_l1 = val_l1_sum / max(1, n_val)

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
            "cfg": {k: (str(v) if isinstance(v, Path) else v) for k, v in cfg.__dict__.items()},
        }
        torch.save(state, cfg.checkpoint_dir / "last.pth")
        if is_best:
            torch.save(state, cfg.checkpoint_dir / "best.pth")

        sample_path: str | None = None
        if cfg.sample_every > 0 and (epoch % cfg.sample_every == 0 or epoch == cfg.epochs):
            try:
                out_dir = PROJECT_ROOT / "outputs" / "samples" / cfg.checkpoint_dir.name
                out_dir.mkdir(parents=True, exist_ok=True)
                out_png = out_dir / f"epoch_{epoch:03d}.png"
                ds_val = val_loader.dataset
                idxs = samples_module.deterministic_indices(len(ds_val), cfg.num_samples, seed=0)
                grid = _cls_sample_grid(model, ds_val, idxs, device, bin_centers_lab, cfg.temperature)
                from PIL import Image as _Image
                _Image.fromarray(grid).save(out_png, format="PNG", optimize=True)
                sample_path = str(out_png)
            except Exception as exc:
                print(f"[train-cls] sample generation failed: {exc}", flush=True)

        payload = {
            "epoch": epoch,
            "train_ce": train_ce,
            "val_ce": val_ce,
            "val_l1": val_l1,
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
            f"[train-cls] epoch {epoch}/{cfg.epochs} done in {epoch_time:.1f}s — "
            f"train_ce={train_ce:.4f} val_ce={val_ce:.4f} val_l1={val_l1:.4f} "
            f"best_ce={best_val_ce:.4f}"
            f"{' [BEST]' if is_best else ''}",
            flush=True,
        )

    print(f"[train-cls] finished. Best val_ce={best_val_ce:.4f}. Checkpoints at {cfg.checkpoint_dir}")


def main() -> int:
    cfg = parse_args()
    set_seed(cfg.seed)
    device = pick_device(cfg.device_str)
    print(f"[train] device={device} model={cfg.model}")

    if cfg.model == "resnet_unet":
        train_resnet_unet(cfg, device)
    elif cfg.model == "cgan":
        train_cgan(cfg, device)
    elif cfg.model == "resnet_unet_cls":
        train_resnet_unet_cls(cfg, device)
    else:
        raise ValueError(f"Unknown model: {cfg.model}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
