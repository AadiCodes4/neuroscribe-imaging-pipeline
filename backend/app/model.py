"""
A small 2D U-Net-style segmentation model, implemented in plain PyTorch
(no MONAI dependency, to keep install size/time small and portable).

Design note: the task allows either a full 3D volumetric model or a 2D
slice-based one. A 2D model was chosen here so the whole pipeline (data
generation -> training -> FastAPI inference -> React frontend) can train and
run in seconds on CPU, while still being a real encoder/decoder U-Net with
skip connections rather than a toy single-layer network.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class DoubleConv(nn.Module):
    """(Conv2d -> BatchNorm -> ReLU) x 2"""

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            # inplace=False: needed so autograd's full-backward hooks (used by
            # the Grad-CAM interpretability method) can safely read/replace
            # gradients without clobbering a view created by ReLU.
            nn.ReLU(inplace=False),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=False),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class SmallUNet(nn.Module):
    """A compact U-Net for binary segmentation of single-channel 2D images.

    Architecture:
        in (1, H, W)
          -> DoubleConv(1, 16)   -- skip1
          -> MaxPool -> DoubleConv(16, 32)  -- skip2
          -> MaxPool -> DoubleConv(32, 64)  -- bottleneck
          -> Upsample -> concat(skip2) -> DoubleConv(96, 32)
          -> Upsample -> concat(skip1) -> DoubleConv(48, 16)
          -> Conv2d(16, 1)  -- output logits (1, H, W)

    Output is raw logits; apply sigmoid to get a per-pixel lesion probability.
    """

    def __init__(self, in_channels: int = 1, out_channels: int = 1, base_channels: int = 16):
        super().__init__()
        c = base_channels

        self.enc1 = DoubleConv(in_channels, c)
        self.pool1 = nn.MaxPool2d(2)
        self.enc2 = DoubleConv(c, c * 2)
        self.pool2 = nn.MaxPool2d(2)

        self.bottleneck = DoubleConv(c * 2, c * 4)
        # Named explicitly so interpretability.py can hook this layer for Grad-CAM.
        self.bottleneck_conv = self.bottleneck.net[-2]  # last Conv2d in bottleneck

        self.up2 = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
        self.dec2 = DoubleConv(c * 4 + c * 2, c * 2)

        self.up1 = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
        self.dec1 = DoubleConv(c * 2 + c, c)

        self.out_conv = nn.Conv2d(c, out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        s1 = self.enc1(x)          # (B, c, H, W)
        p1 = self.pool1(s1)        # (B, c, H/2, W/2)

        s2 = self.enc2(p1)         # (B, 2c, H/2, W/2)
        p2 = self.pool2(s2)        # (B, 2c, H/4, W/4)

        b = self.bottleneck(p2)    # (B, 4c, H/4, W/4)

        u2 = self.up2(b)           # (B, 4c, H/2, W/2)
        u2 = torch.cat([u2, s2], dim=1)  # (B, 4c+2c, H/2, W/2)
        d2 = self.dec2(u2)         # (B, 2c, H/2, W/2)

        u1 = self.up1(d2)          # (B, 2c, H, W)
        u1 = torch.cat([u1, s1], dim=1)  # (B, 2c+c, H, W)
        d1 = self.dec1(u1)         # (B, c, H, W)

        logits = self.out_conv(d1)  # (B, 1, H, W)
        return logits


def dice_loss(logits: torch.Tensor, targets: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Soft Dice loss for binary segmentation (used alongside BCE)."""
    probs = torch.sigmoid(logits)
    probs_flat = probs.view(probs.size(0), -1)
    targets_flat = targets.view(targets.size(0), -1)
    intersection = (probs_flat * targets_flat).sum(dim=1)
    union = probs_flat.sum(dim=1) + targets_flat.sum(dim=1)
    dice = (2 * intersection + eps) / (union + eps)
    return 1 - dice.mean()


def segmentation_loss(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """Combined BCE + Dice loss, a standard, real choice for segmentation."""
    bce = nn.functional.binary_cross_entropy_with_logits(logits, targets)
    dsc = dice_loss(logits, targets)
    return bce + dsc


def build_model() -> SmallUNet:
    return SmallUNet(in_channels=1, out_channels=1, base_channels=16)
