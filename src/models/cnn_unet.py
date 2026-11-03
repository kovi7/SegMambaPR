from __future__ import annotations

from typing import Optional, Sequence

import torch
import torch.nn.functional as F
from torch import nn

from .common import ResidualConvBlock3d
from .tab_gating import MultiScaleTabularAggregator, TabGate3D, build_tabular_embedding


class CNNUNet3D(nn.Module):
    """Compact 3D residual U-Net baseline with optional TabGating."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        channels: Sequence[int] = (32, 64, 128, 256, 320),
        norm: str = "instance",
        tab_dim: int = 0,
        tab_embed_dim: int = 64,
        tab_dropout: float = 0.0,
        gated_stages: Sequence[int] = (),
        tabular_mode: str = "numeric",
        gamma_init: float = 0.0,
        use_feature_aggregator: bool = False,
    ):
        super().__init__()
        self.use_tabular = tab_dim > 0 and len(gated_stages) > 0
        self.gated_stages = {int(s) for s in gated_stages}
        if self.use_tabular:
            self.tab_encoder = build_tabular_embedding(tab_dim, tab_embed_dim, tab_dropout, mode=tabular_mode)
        else:
            self.tab_encoder = None

        self.enc = nn.ModuleList()
        self.down = nn.ModuleList()
        prev = in_channels
        for ch in channels:
            self.enc.append(ResidualConvBlock3d(prev, ch, norm=norm))
            self.down.append(nn.MaxPool3d(2))
            prev = ch
        self.enc_gates = nn.ModuleDict()
        if self.use_tabular:
            for i in self.gated_stages:
                self.enc_gates[str(i)] = TabGate3D(int(channels[i]), tab_embed_dim, gamma_init=gamma_init)

        self.aggregator = (
            MultiScaleTabularAggregator(list(channels), tab_embed_dim) if (self.use_tabular and use_feature_aggregator) else None
        )

        rev = list(channels[::-1])
        self.up = nn.ModuleList()
        self.dec = nn.ModuleList()
        for i in range(len(rev) - 1):
            self.up.append(nn.ConvTranspose3d(rev[i], rev[i + 1], kernel_size=2, stride=2))
            self.dec.append(ResidualConvBlock3d(rev[i + 1] * 2, rev[i + 1], norm=norm))
        self.out = nn.Conv3d(channels[0], out_channels, kernel_size=1)

    def _tab_emb(self, tabular: Optional[torch.Tensor], tabular_mask: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
        if not self.use_tabular:
            return None
        return self.tab_encoder(tabular, tabular_mask)

    def forward(self, image: torch.Tensor, tabular: Optional[torch.Tensor] = None, tabular_mask: Optional[torch.Tensor] = None):
        tab_emb = self._tab_emb(tabular, tabular_mask)
        skips = []
        x = image
        for i, block in enumerate(self.enc):
            x = block(x)
            if tab_emb is not None and str(i) in self.enc_gates:
                x = self.enc_gates[str(i)](x, tab_emb)
            skips.append(x)
            if i < len(self.enc) - 1:
                x = self.down[i](x)

        if self.aggregator is not None and tab_emb is not None:
            scale, shift = self.aggregator(skips, tab_emb)
            x = MultiScaleTabularAggregator.apply_film(x, scale, shift)
            skips[-1] = x

        for i, (up, dec) in enumerate(zip(self.up, self.dec)):
            x = up(x)
            skip = skips[-(i + 2)]
            if x.shape[2:] != skip.shape[2:]:
                x = F.interpolate(x, size=skip.shape[2:], mode="trilinear", align_corners=False)
            x = torch.cat([x, skip], dim=1)
            x = dec(x)
        return {"seg_logits": self.out(x)}


class PlainCNNUNet3D(CNNUNet3D):
    """CNN U-Net without tabular input."""

    def __init__(self, in_channels: int, out_channels: int, channels=(24, 48, 96, 192, 256), norm: str = "instance"):
        super().__init__(in_channels=in_channels, out_channels=out_channels, channels=channels, norm=norm)
