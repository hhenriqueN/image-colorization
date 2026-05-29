"""ResNet-34 encoder + U-Net decoder with a Q-way classification head (Phase C).

Identical to `ResNetUNet` except the final layer outputs `num_classes` logits
per pixel instead of 2 regressed ab values + Tanh. Used with weighted
cross-entropy training and annealed-mean inference (see
`src/data/quantize.py`).
"""

from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn

from src.models.resnet_unet import ResNetUNet


class ResNetUNetClassifier(ResNetUNet):
    """ResNetUNet with the regression head replaced by a Q-way classifier head.

    Forward output is (N, num_classes, H, W) — raw logits, no activation.
    """

    def __init__(self, num_classes: int, freeze_encoder: bool = True) -> None:
        super().__init__(out_channels=2, freeze_encoder=freeze_encoder)
        # Replace the regression+Tanh head with a plain Conv-Transpose to logits.
        self.num_classes = num_classes
        self.out_conv = nn.ConvTranspose2d(
            64 + 64, num_classes, kernel_size=4, stride=2, padding=1
        )

    @classmethod
    def from_regression_checkpoint(
        cls,
        ckpt_path: Path,
        num_classes: int,
        freeze_encoder: bool = True,
        map_location: str | torch.device = "cpu",
    ) -> "ResNetUNetClassifier":
        """Warm-start by loading a regression ResNetUNet checkpoint.

        Loads all matching parameters from the regression checkpoint and silently
        skips the final-layer mismatch (`out_conv.*`). The new classifier head is
        left at its random init.
        """
        model = cls(num_classes=num_classes, freeze_encoder=freeze_encoder)
        state = torch.load(ckpt_path, map_location=map_location, weights_only=False)
        if isinstance(state, dict):
            if "model_state_dict" in state:
                state = state["model_state_dict"]
            elif "generator_state_dict" in state:
                state = state["generator_state_dict"]
        filtered = {k: v for k, v in state.items() if not k.startswith("out_conv.")}
        missing, unexpected = model.load_state_dict(filtered, strict=False)
        # `out_conv.*` will show up under `missing` because the filter removed
        # them from the source state; that's intentional.
        unexpected_real = [k for k in unexpected if not k.startswith("out_conv.")]
        if unexpected_real:
            raise RuntimeError(f"Unexpected keys in warm-start checkpoint: {unexpected_real}")
        return model
