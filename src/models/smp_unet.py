"""segmentation_models_pytorch ResNet-34 U-Net wrapper for LAB colorization.

Input:  (N, 1, 256, 256)  normalized L channel in [-1, 1]
Output: (N, 2, 256, 256)  predicted ab channels in [-1, 1]

This is the smp-based companion to src/models/resnet_unet.py. The smp library
handles the 1-channel input by averaging the pretrained conv1 weights along
the input dim, so no x.repeat(1,3,1,1) trick is needed. A `tanh` is applied
to the raw smp output so the ab range matches ColorizationDataset's
normalization — required for downstream lab_to_rgb and PatchGAN BN to behave.
"""

from __future__ import annotations

import segmentation_models_pytorch as smp
import torch
import torch.nn as nn


class SmpUNet(nn.Module):
    """smp.Unet(resnet34, imagenet) with frozen-encoder helpers and tanh output."""

    def __init__(self, out_channels: int = 2, freeze_encoder: bool = True) -> None:
        super().__init__()

        self.unet = smp.Unet(
            encoder_name="resnet34",
            encoder_weights="imagenet",
            in_channels=1,
            classes=out_channels,
            activation=None,
        )
        self._encoder_frozen = False
        if freeze_encoder:
            self.freeze_encoder()

    # ------------------------------------------------------------------
    # Encoder freeze / unfreeze helpers
    # ------------------------------------------------------------------

    def freeze_encoder(self) -> None:
        for p in self.unet.encoder.parameters():
            p.requires_grad_(False)
        self._encoder_frozen = True
        self.unet.encoder.eval()

    def unfreeze_encoder(self) -> None:
        for p in self.unet.encoder.parameters():
            p.requires_grad_(True)
        self._encoder_frozen = False

    def train(self, mode: bool = True) -> "SmpUNet":
        super().train(mode)
        if self._encoder_frozen:
            self.unet.encoder.eval()
        return self

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.tanh(self.unet(x))
