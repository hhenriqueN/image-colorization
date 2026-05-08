"""PatchGAN discriminator for conditional image colorization (Phase 3).

Judges whether (L, ab) pairs are real or generated, patch-by-patch rather
than globally — this forces the generator to be realistic at a local level.

Input:  concat(L, ab) → (N, 3, 256, 256)
        Real pair:  (L, ground-truth ab)
        Fake pair:  (L, predicted ab)
Output: (N, 1, 30, 30)  — each value covers a 70×70 receptive field

Architecture: C64 → C128 → C256 → C512 → 1  (pix2pix standard)
No BN on the first layer; no sigmoid on output (use with LSGAN / BCEWithLogits).
"""

from __future__ import annotations

import torch
import torch.nn as nn


class _DiscBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, batchnorm: bool = True, stride: int = 2) -> None:
        super().__init__()
        layers: list[nn.Module] = [
            nn.Conv2d(in_ch, out_ch, kernel_size=4, stride=stride, padding=1, bias=not batchnorm)
        ]
        if batchnorm:
            layers.append(nn.BatchNorm2d(out_ch))
        layers.append(nn.LeakyReLU(0.2, inplace=True))
        self.block = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class PatchGANDiscriminator(nn.Module):
    """70×70 PatchGAN — takes a conditional (L, ab) pair and returns a patch score map."""

    def __init__(self, in_channels: int = 3) -> None:
        super().__init__()
        self.model = nn.Sequential(
            _DiscBlock(in_channels, 64,  batchnorm=False, stride=2),  # (64, 128, 128)
            _DiscBlock(64,  128, stride=2),                            # (128, 64, 64)
            _DiscBlock(128, 256, stride=2),                            # (256, 32, 32)
            _DiscBlock(256, 512, stride=1),                            # (512, 31, 31)
            nn.Conv2d(512, 1, kernel_size=4, stride=1, padding=1),    # (1,  30, 30)
        )

    def forward(self, l: torch.Tensor, ab: torch.Tensor) -> torch.Tensor:
        return self.model(torch.cat([l, ab], dim=1))
