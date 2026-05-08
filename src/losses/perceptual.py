"""VGG-16 perceptual loss for colorization training.

Compares relu2_2 and relu3_3 features between predicted and target RGB.
Includes a differentiable LAB→RGB conversion so gradients flow all the way
back through the colorization model — no numpy detour needed.

Usage:
    criterion = PerceptualLoss().to(device)
    loss = criterion(pred_l, pred_ab, true_l, true_ab)
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torchvision.models as models


def lab_to_rgb_tensor(l: torch.Tensor, ab: torch.Tensor) -> torch.Tensor:
    """Differentiable LAB → sRGB conversion.

    l:  (N, 1, H, W) normalized to [-1, 1]  (L channel)
    ab: (N, 2, H, W) normalized to [-1, 1]  (ab channels)
    Returns: (N, 3, H, W) in [0, 1]
    """
    L = (l + 1.0) * 50.0           # [-1,1] → [0, 100]
    ab_denorm = ab * 128.0          # [-1,1] → [-128, 128]
    a_ch = ab_denorm[:, 0:1]
    b_ch = ab_denorm[:, 1:2]

    # LAB → XYZ  (D65 illuminant)
    fy = (L + 16.0) / 116.0
    fx = a_ch / 500.0 + fy
    fz = fy - b_ch / 200.0

    delta = 6.0 / 29.0

    def f_inv(t: torch.Tensor) -> torch.Tensor:
        return torch.where(t > delta, t.pow(3), (t - 16.0 / 116.0) / 7.787)

    X = 0.95047 * f_inv(fx)
    Y = 1.00000 * f_inv(fy)
    Z = 1.08883 * f_inv(fz)

    # XYZ → linear sRGB
    r =  3.2406 * X - 1.5372 * Y - 0.4986 * Z
    g = -0.9689 * X + 1.8758 * Y + 0.0415 * Z
    b =  0.0557 * X - 0.2040 * Y + 1.0570 * Z

    rgb_lin = torch.cat([r, g, b], dim=1).clamp(0.0, 1.0)

    # Linear → gamma-corrected sRGB
    rgb = torch.where(
        rgb_lin <= 0.0031308,
        12.92 * rgb_lin,
        1.055 * rgb_lin.pow(1.0 / 2.4) - 0.055,
    )
    return rgb.clamp(0.0, 1.0)


class PerceptualLoss(nn.Module):
    """L1 distance on VGG-16 relu2_2 and relu3_3 feature maps.

    Both feature layers are averaged and summed, giving the model a signal at
    two scales of abstraction (texture at relu2_2, structure at relu3_3).
    The VGG weights are permanently frozen.
    """

    _MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
    _STD  = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)

    def __init__(self) -> None:
        super().__init__()
        vgg = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1)
        feats = vgg.features

        # relu2_2 ends at index 9; relu3_3 ends at index 16
        self.slice1 = nn.Sequential(*list(feats.children())[:10])
        self.slice2 = nn.Sequential(*list(feats.children())[10:17])

        for p in self.parameters():
            p.requires_grad_(False)

    def forward(
        self,
        pred_l: torch.Tensor,
        pred_ab: torch.Tensor,
        true_l: torch.Tensor,
        true_ab: torch.Tensor,
    ) -> torch.Tensor:
        pred_rgb = self._vgg_normalize(lab_to_rgb_tensor(pred_l, pred_ab))
        true_rgb = self._vgg_normalize(lab_to_rgb_tensor(true_l, true_ab))

        p1 = self.slice1(pred_rgb)
        t1 = self.slice1(true_rgb)
        loss = (p1 - t1).abs().mean()

        p2 = self.slice2(p1)
        t2 = self.slice2(t1)
        loss = loss + (p2 - t2).abs().mean()

        return loss

    def _vgg_normalize(self, rgb: torch.Tensor) -> torch.Tensor:
        mean = self._MEAN.to(rgb.device)
        std  = self._STD.to(rgb.device)
        return (rgb - mean) / std
