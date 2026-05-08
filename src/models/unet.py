"""U-Net encoder-decoder for image colorization.

Input:  (N, 1, 256, 256)  normalized L channel in [-1, 1]
Output: (N, 2, 256, 256)  predicted ab channels in [-1, 1]

Five-level encoder-decoder. Bottleneck at 8x8 for 256x256 inputs.

Encoder: Conv2d(stride=2) + BatchNorm + LeakyReLU(0.2)
Decoder: ConvTranspose2d(stride=2) + BatchNorm + (Dropout) + ReLU
Skip connections are concatenated channel-wise before each decoder stage.
Final output passes through Tanh to stay in [-1, 1].

Channel progression:
  Encoder: 1 -> 64 -> 128 -> 256 -> 512 -> 512  (bottleneck)
  Decoder: 512 -> cat(512,512)=1024 -> 256 -> cat(256,256)=512
              -> 128 -> cat(128,128)=256 -> 64 -> cat(64,64)=128 -> 2
"""

from __future__ import annotations

import torch
import torch.nn as nn


class _EncoderBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, batchnorm: bool = True) -> None:
        super().__init__()
        layers: list[nn.Module] = [
            nn.Conv2d(in_ch, out_ch, kernel_size=4, stride=2, padding=1, bias=not batchnorm)
        ]
        if batchnorm:
            layers.append(nn.BatchNorm2d(out_ch))
        layers.append(nn.LeakyReLU(0.2, inplace=True))
        self.block = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


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


class UNet(nn.Module):
    """U-Net for colorization: L channel -> ab channels."""

    def __init__(self, in_channels: int = 1, out_channels: int = 2) -> None:
        super().__init__()

        # Encoder — each block halves spatial resolution
        self.enc1 = _EncoderBlock(in_channels, 64, batchnorm=False)  # 256 -> 128
        self.enc2 = _EncoderBlock(64, 128)                            # 128 -> 64
        self.enc3 = _EncoderBlock(128, 256)                           # 64  -> 32
        self.enc4 = _EncoderBlock(256, 512)                           # 32  -> 16
        self.enc5 = _EncoderBlock(512, 512)                           # 16  -> 8  (bottleneck)

        # Decoder — ConvTranspose2d doubles spatial resolution
        # in_ch accounts for the skip concatenated just before each block
        self.dec1 = _DecoderBlock(512, 512, dropout=True)   # bottleneck: 8  -> 16
        self.dec2 = _DecoderBlock(1024, 256, dropout=True)  # +skip enc4: 16 -> 32
        self.dec3 = _DecoderBlock(512, 128)                 # +skip enc3: 32 -> 64
        self.dec4 = _DecoderBlock(256, 64)                  # +skip enc2: 64 -> 128

        # Final upsample — no BN, Tanh to bound output to [-1, 1]
        self.out_conv = nn.Sequential(
            nn.ConvTranspose2d(128, out_channels, kernel_size=4, stride=2, padding=1),  # +skip enc1: 128 -> 256
            nn.Tanh(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Encode
        s1 = self.enc1(x)   # (64,  128, 128)
        s2 = self.enc2(s1)  # (128,  64,  64)
        s3 = self.enc3(s2)  # (256,  32,  32)
        s4 = self.enc4(s3)  # (512,  16,  16)
        b  = self.enc5(s4)  # (512,   8,   8)

        # Decode with skip connections
        d = self.dec1(b)                            # (512,  16, 16)
        d = self.dec2(torch.cat([d, s4], dim=1))    # (256,  32, 32)
        d = self.dec3(torch.cat([d, s3], dim=1))    # (128,  64, 64)
        d = self.dec4(torch.cat([d, s2], dim=1))    # (64,  128, 128)
        return self.out_conv(torch.cat([d, s1], dim=1))  # (2, 256, 256)
