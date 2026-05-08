"""ResNet-34 encoder + U-Net decoder for image colorization (Phase 2).

Input:  (N, 1, 256, 256)  normalized L channel in [-1, 1]
Output: (N, 2, 256, 256)  predicted ab channels in [-1, 1]

The L channel is repeated 3× before the encoder so pretrained conv1 weights
are reused without modification.

ResNet-34 feature map sizes for 256×256 input:
  enc_conv1  (stride 2): (N,  64, 128, 128)  ← s1
  enc_layer1 (pool+l1):  (N,  64,  64,  64)  ← s2
  enc_layer2 (stride 2): (N, 128,  32,  32)  ← s3
  enc_layer3 (stride 2): (N, 256,  16,  16)  ← s4
  enc_layer4 (stride 2): (N, 512,   8,   8)  ← bottleneck

Decoder mirrors the encoder with ConvTranspose2d; skips are cat'd channel-wise.
The encoder is frozen by default — call .unfreeze_encoder() to fine-tune.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torchvision.models as models


class _DecoderBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, dropout: bool = False) -> None:
        super().__init__()
        layers: list[nn.Module] = [
            nn.ConvTranspose2d(in_ch, out_ch, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
        ]
        if dropout:
            layers.append(nn.Dropout(0.5))
        layers.append(nn.ReLU(inplace=True))
        self.block = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class ResNetUNet(nn.Module):
    """ResNet-34 encoder + U-Net decoder for LAB colorization."""

    def __init__(self, out_channels: int = 2, freeze_encoder: bool = True) -> None:
        super().__init__()

        backbone = models.resnet34(weights=models.ResNet34_Weights.IMAGENET1K_V1)

        self.enc_conv1  = nn.Sequential(backbone.conv1, backbone.bn1, backbone.relu)
        self.enc_pool   = backbone.maxpool
        self.enc_layer1 = backbone.layer1
        self.enc_layer2 = backbone.layer2
        self.enc_layer3 = backbone.layer3
        self.enc_layer4 = backbone.layer4

        if freeze_encoder:
            self.freeze_encoder()

        # Decoder channel sizes account for skip concatenation before each block
        self.dec4 = _DecoderBlock(512,           256, dropout=True)  # b          → (256, 16, 16)
        self.dec3 = _DecoderBlock(256 + 256,     128)                # +s4(256)   → (128, 32, 32)
        self.dec2 = _DecoderBlock(128 + 128,      64)                # +s3(128)   → (64,  64, 64)
        self.dec1 = _DecoderBlock( 64 +  64,      64)                # +s2(64)    → (64, 128,128)

        self.out_conv = nn.Sequential(
            nn.ConvTranspose2d(64 + 64, out_channels, kernel_size=4, stride=2, padding=1),
            nn.Tanh(),
        )

    # ------------------------------------------------------------------
    # Encoder freeze / unfreeze helpers
    # ------------------------------------------------------------------

    def freeze_encoder(self) -> None:
        for p in self._encoder_parameters():
            p.requires_grad_(False)

    def unfreeze_encoder(self) -> None:
        for p in self._encoder_parameters():
            p.requires_grad_(True)

    def _encoder_parameters(self):
        for m in (self.enc_conv1, self.enc_pool, self.enc_layer1,
                  self.enc_layer2, self.enc_layer3, self.enc_layer4):
            yield from m.parameters()

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x3 = x.repeat(1, 3, 1, 1)                       # (N, 1, H, W) → (N, 3, H, W)

        s1 = self.enc_conv1(x3)                          # (N,  64, 128, 128)
        s2 = self.enc_layer1(self.enc_pool(s1))          # (N,  64,  64,  64)
        s3 = self.enc_layer2(s2)                         # (N, 128,  32,  32)
        s4 = self.enc_layer3(s3)                         # (N, 256,  16,  16)
        b  = self.enc_layer4(s4)                         # (N, 512,   8,   8)

        d = self.dec4(b)                                 # (N, 256,  16,  16)
        d = self.dec3(torch.cat([d, s4], dim=1))         # (N, 128,  32,  32)
        d = self.dec2(torch.cat([d, s3], dim=1))         # (N,  64,  64,  64)
        d = self.dec1(torch.cat([d, s2], dim=1))         # (N,  64, 128, 128)
        return self.out_conv(torch.cat([d, s1], dim=1))  # (N,   2, 256, 256)
