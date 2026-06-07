"""smp.Unet ResNet-34 encoder + 214-way ab classification head (Phase 3).

Same encoder/decoder as src/models/smp_unet.py:SmpUNet, but the segmentation
head outputs `num_classes` logits per pixel instead of 2 regressed ab values.
Used with weighted cross-entropy training and annealed-mean inference (see
src/data/quantize.py).

Designed to warm-start from a regression SmpUNet checkpoint by filtering out
the head weights (`unet.segmentation_head.*`) so encoder + decoder transfer
cleanly to the new task.
"""

from __future__ import annotations

from pathlib import Path

import segmentation_models_pytorch as smp
import torch
import torch.nn as nn


class SmpUNetClassifier(nn.Module):
    """smp.Unet(resnet34, imagenet) with a Q-way classification head.

    Forward output is (N, num_classes, H, W) — raw logits, no activation.
    """

    def __init__(self, num_classes: int = 214, freeze_encoder: bool = True) -> None:
        super().__init__()

        self.num_classes = num_classes
        self.unet = smp.Unet(
            encoder_name="resnet34",
            encoder_weights="imagenet",
            in_channels=1,
            classes=num_classes,
            activation=None,
        )
        self._encoder_frozen = False
        if freeze_encoder:
            self.freeze_encoder()

    # ------------------------------------------------------------------
    # Encoder freeze / unfreeze helpers (same contract as SmpUNet)
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

    def train(self, mode: bool = True) -> "SmpUNetClassifier":
        super().train(mode)
        if self._encoder_frozen:
            self.unet.encoder.eval()
        return self

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.unet(x)

    # ------------------------------------------------------------------
    # Warm-start from a regression SmpUNet checkpoint
    # ------------------------------------------------------------------

    @classmethod
    def from_smp_regression_checkpoint(
        cls,
        ckpt_path: Path,
        num_classes: int = 214,
        freeze_encoder: bool = True,
        map_location: str | torch.device = "cpu",
    ) -> "SmpUNetClassifier":
        """Warm-start by loading an SmpUNet regression checkpoint.

        Filters out the segmentation head (`unet.segmentation_head.*`) and
        loads everything else. The new num_classes-way head stays random-init.
        Raises if the checkpoint contains keys outside the expected prefixes.
        """
        model = cls(num_classes=num_classes, freeze_encoder=freeze_encoder)
        state = torch.load(ckpt_path, map_location=map_location, weights_only=False)
        if isinstance(state, dict):
            if "model_state_dict" in state:
                state = state["model_state_dict"]
            elif "generator_state_dict" in state:
                state = state["generator_state_dict"]

        head_prefix = "unet.segmentation_head."
        filtered = {k: v for k, v in state.items() if not k.startswith(head_prefix)}
        missing, unexpected = model.load_state_dict(filtered, strict=False)

        unexpected_real = [k for k in unexpected if not k.startswith(head_prefix)]
        if unexpected_real:
            raise RuntimeError(
                f"Unexpected keys in warm-start checkpoint: {unexpected_real}"
            )
        missing_real = [k for k in missing if not k.startswith(head_prefix)]
        if missing_real:
            raise RuntimeError(
                f"Missing non-head keys in warm-start checkpoint: {missing_real}"
            )
        return model
