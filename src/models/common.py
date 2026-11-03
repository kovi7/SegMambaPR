from __future__ import annotations

from typing import Optional

import torch
from torch import nn


class ImageOnlyWrapper(nn.Module):
    """Wrap a MONAI segmentation network so that it returns a dict like the other models."""

    def __init__(self, net: nn.Module):
        super().__init__()
        self.net = net

    def forward(self, image: torch.Tensor, tabular: Optional[torch.Tensor] = None, tabular_mask: Optional[torch.Tensor] = None):
        return {"seg_logits": self.net(image)}


class ConvNormAct3d(nn.Module):
    """Conv -> norm -> LeakyReLU; ``act=False`` drops the activation (for residual branches)."""

    def __init__(self, in_ch: int, out_ch: int, kernel_size: int = 3, stride: int = 1, norm: str = "instance",
                 act: bool = True):
        super().__init__()
        padding = kernel_size // 2
        if norm.lower() == "batch":
            norm_layer = nn.BatchNorm3d(out_ch)
        elif norm.lower() == "group":
            groups = min(8, out_ch)
            while out_ch % groups != 0 and groups > 1:
                groups -= 1
            norm_layer = nn.GroupNorm(groups, out_ch)
        else:
            # instance norm as GroupNorm(C, C): same statistics, no copy out of channels_last_3d
            norm_layer = nn.GroupNorm(out_ch, out_ch, affine=True)
        layers = [
            nn.Conv3d(in_ch, out_ch, kernel_size=kernel_size, stride=stride, padding=padding, bias=False),
            norm_layer,
        ]
        if act:
            layers.append(nn.LeakyReLU(0.01, inplace=True))
        self.block = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class ResidualConvBlock3d(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, norm: str = "instance"):
        super().__init__()
        self.conv1 = ConvNormAct3d(in_ch, out_ch, norm=norm)
        self.conv2 = ConvNormAct3d(out_ch, out_ch, norm=norm)
        self.skip = nn.Identity() if in_ch == out_ch else nn.Conv3d(in_ch, out_ch, kernel_size=1, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv2(self.conv1(x)) + self.skip(x)
