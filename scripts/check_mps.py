"""One-shot MPS sanity check for the colorization stack.

Runs a forward + backward pass of ResNetUNet (and optionally PerceptualLoss)
on a small synthetic batch under MPS, prints throughput (images/sec) and
peak memory. Exits non-zero on failure so the ralph loop can detect it.

Usage:
    uv run python scripts/check_mps.py
    uv run python scripts/check_mps.py --batch-size 8 --skip-perceptual
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import torch  # noqa: E402

from src.losses.perceptual import PerceptualLoss  # noqa: E402
from src.models.resnet_unet import ResNetUNet  # noqa: E402


def pick_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def fmt_bytes(num_bytes: int) -> str:
    gb = num_bytes / (1024**3)
    return f"{gb:.2f} GiB"


def main() -> int:
    parser = argparse.ArgumentParser(description="MPS sanity check for ResNetUNet")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--iters", type=int, default=5)
    parser.add_argument("--skip-perceptual", action="store_true")
    args = parser.parse_args()

    device = pick_device()
    print(f"[check_mps] device: {device}")
    print(f"[check_mps] torch: {torch.__version__}")

    model = ResNetUNet(freeze_encoder=True).to(device)
    perceptual = None if args.skip_perceptual else PerceptualLoss().to(device)
    l1 = torch.nn.L1Loss()

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen = sum(p.numel() for p in model.parameters() if not p.requires_grad)
    print(f"[check_mps] trainable params: {trainable:,}")
    print(f"[check_mps] frozen params:    {frozen:,}")

    bsz = args.batch_size
    sz = args.image_size
    l = torch.randn(bsz, 1, sz, sz, device=device)
    ab = torch.randn(bsz, 2, sz, sz, device=device).clamp_(-1, 1)

    optimizer = torch.optim.AdamW(
        (p for p in model.parameters() if p.requires_grad),
        lr=2e-4,
    )

    # Warmup
    print("[check_mps] warmup forward+backward...")
    pred_ab = model(l)
    loss = l1(pred_ab, ab)
    if perceptual is not None:
        loss = loss + 0.1 * perceptual(l, pred_ab, l, ab)
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
    if device.type == "mps":
        torch.mps.synchronize()

    print("[check_mps] timing...")
    start = time.perf_counter()
    for _ in range(args.iters):
        pred_ab = model(l)
        loss = l1(pred_ab, ab)
        if perceptual is not None:
            loss = loss + 0.1 * perceptual(l, pred_ab, l, ab)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
    if device.type == "mps":
        torch.mps.synchronize()
    elapsed = time.perf_counter() - start

    images = args.iters * bsz
    ips = images / elapsed
    print(
        f"[check_mps] {args.iters} steps, batch={bsz}, "
        f"{elapsed:.2f}s total → {ips:.2f} images/sec"
    )

    if device.type == "mps":
        peak = torch.mps.driver_allocated_memory()
        print(f"[check_mps] peak MPS memory: {fmt_bytes(peak)}")

    print(f"[check_mps] final loss value: {loss.item():.4f}")
    print("[check_mps] OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
