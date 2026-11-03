from __future__ import annotations

from typing import Optional, Sequence, Tuple

import torch
from torch import nn


class TabularEmbedding(nn.Module):
    """Embed normalised tabular values together with their availability mask.

    Missing values are value 0 and mask 0. A learned all-missing embedding is added when
    the whole vector is unavailable, as for every test case. ``dropout`` drops the whole
    vector for a random subset of samples during training.
    """

    def __init__(self, tab_dim: int, embed_dim: int = 64, dropout: float = 0.0, mlp_dropout: float = 0.1):
        super().__init__()
        self.tab_dim = int(tab_dim)
        self.dropout = float(dropout)
        self.value_proj = nn.Sequential(
            nn.Linear(tab_dim * 2, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(mlp_dropout),
            nn.Linear(embed_dim, embed_dim),
            nn.ReLU(inplace=True),
        )
        self.all_missing = nn.Parameter(torch.zeros(embed_dim))
        nn.init.normal_(self.all_missing, std=0.02)

    def forward(self, tab: Optional[torch.Tensor], mask: Optional[torch.Tensor]) -> torch.Tensor:
        if tab is None or mask is None:
            raise ValueError("TabularEmbedding expects tab and mask. Use zeros and mask=0 for missing data.")
        if tab.ndim > 2:
            tab = tab.view(tab.shape[0], -1)
        if mask.ndim > 2:
            mask = mask.view(mask.shape[0], -1)
        tab = tab.float()
        mask = mask.float()
        if tab.shape[1] != self.tab_dim:
            raise ValueError(f"Expected tab_dim={self.tab_dim}, got {tab.shape[1]}")
        if self.training and self.dropout > 0:
            # drop the whole vector, which trains the missing-data path used at test time
            keep = (torch.rand(tab.shape[0], 1, device=tab.device) > self.dropout).float()
            mask = mask * keep
            tab = tab * mask
        x = torch.cat([tab * mask, mask], dim=1)
        emb = self.value_proj(x)
        missing_all = (mask.sum(dim=1, keepdim=True) == 0).float()
        return emb + missing_all * self.all_missing.view(1, -1)


class TabularTextEmbedding(nn.Module):
    """Project a precomputed sentence embedding of the metadata to ``embed_dim``.

    The embeddings come from ``tools/precompute_tabular_text.py``. Missing metadata use
    one scalar mask and the learned ``all_missing`` embedding.
    """

    def __init__(self, text_dim: int, embed_dim: int = 64, dropout: float = 0.0, mlp_dropout: float = 0.1):
        super().__init__()
        self.text_dim = int(text_dim)
        self.dropout = float(dropout)
        self.value_proj = nn.Sequential(
            nn.Linear(text_dim, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(mlp_dropout),
            nn.Linear(embed_dim, embed_dim),
            nn.ReLU(inplace=True),
        )
        self.all_missing = nn.Parameter(torch.zeros(embed_dim))
        nn.init.normal_(self.all_missing, std=0.02)

    def forward(self, text_emb: Optional[torch.Tensor], mask: Optional[torch.Tensor]) -> torch.Tensor:
        if text_emb is None or mask is None:
            raise ValueError("TabularTextEmbedding expects text_emb and mask. Use the cached 'missing' vector and mask=0 for missing data.")
        if text_emb.ndim > 2:
            text_emb = text_emb.view(text_emb.shape[0], -1)
        mask = mask.float().view(mask.shape[0], -1)[:, :1]
        if self.training and self.dropout > 0:
            keep = (torch.rand_like(mask) > self.dropout).float()
            mask = mask * keep
        proj = self.value_proj(text_emb.float())
        missing_all = 1.0 - mask
        return proj * mask + missing_all * self.all_missing.view(1, -1)


def set_tabular_dropout(model: nn.Module, p: float) -> int:
    """Set the whole-vector tabular dropout of every tabular embedding; return their count."""
    n = 0
    for module in model.modules():
        if isinstance(module, (TabularEmbedding, TabularTextEmbedding)):
            module.dropout = float(p)
            n += 1
    return n


def build_tabular_embedding(tab_dim: int, embed_dim: int, dropout: float, mode: str = "numeric") -> nn.Module:
    """Build the numeric or the text tabular embedding."""
    mode = str(mode).lower()
    if mode == "text":
        return TabularTextEmbedding(text_dim=tab_dim, embed_dim=embed_dim, dropout=dropout)
    if mode == "numeric":
        return TabularEmbedding(tab_dim=tab_dim, embed_dim=embed_dim, dropout=dropout)
    raise ValueError(f"Unsupported tabular.mode={mode!r}. Use 'numeric' or 'text'.")


class ChannelTabGate3D(nn.Module):
    """Channel gate of TabAttention: sigmoid(image_mlp(avg) + image_mlp(max) + tab_mlp(tab_emb)).

    Initialised near identity. The bias enters twice, so the initial gate is sigmoid(4) ~ 0.98.
    """

    IDENTITY_BIAS = 2.0

    def __init__(self, channels: int, tab_embed_dim: int, reduction: int = 16):
        super().__init__()
        hidden = max(channels // reduction, 4)
        self.image_mlp = nn.Sequential(
            nn.Linear(channels, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, channels),
        )
        self.tab_mlp = nn.Sequential(
            nn.Linear(tab_embed_dim, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, channels),
        )
        self._init_near_identity()

    def _init_near_identity(self) -> None:
        nn.init.zeros_(self.image_mlp[-1].weight)
        nn.init.constant_(self.image_mlp[-1].bias, self.IDENTITY_BIAS)
        nn.init.zeros_(self.tab_mlp[-1].weight)
        nn.init.zeros_(self.tab_mlp[-1].bias)

    def forward(self, x: torch.Tensor, tab_emb: torch.Tensor) -> torch.Tensor:
        avg = x.mean(dim=(2, 3, 4))
        mx = x.amax(dim=(2, 3, 4))
        logits = self.image_mlp(avg) + self.image_mlp(mx) + self.tab_mlp(tab_emb)
        gate = torch.sigmoid(logits)
        return x * gate.view(x.shape[0], x.shape[1], 1, 1, 1)


class SpatialTabGate3D(nn.Module):
    """Spatial gate of TabAttention: sigmoid(conv([mean_pool, max_pool, tab_map])).

    The tabular embedding is projected to a few channels and broadcast over space, so the
    gate works at any resolution. Initialised near identity.
    """

    IDENTITY_BIAS = 4.0

    def __init__(self, tab_embed_dim: int, tab_channels: int = 4, kernel_size: int = 7):
        super().__init__()
        self.tab_proj = nn.Linear(tab_embed_dim, tab_channels)
        padding = kernel_size // 2
        self.conv = nn.Conv3d(2 + tab_channels, 1, kernel_size=kernel_size, padding=padding)
        nn.init.zeros_(self.tab_proj.weight)
        nn.init.zeros_(self.tab_proj.bias)
        nn.init.zeros_(self.conv.weight)
        nn.init.constant_(self.conv.bias, self.IDENTITY_BIAS)

    def forward(self, x: torch.Tensor, tab_emb: torch.Tensor) -> torch.Tensor:
        avg = x.mean(dim=1, keepdim=True)
        mx = x.amax(dim=1, keepdim=True)
        spatial_shape = x.shape[2:]
        tab_map = self.tab_proj(tab_emb)
        tab_map = tab_map.view(*tab_map.shape, 1, 1, 1).expand(-1, -1, *spatial_shape)
        logits = self.conv(torch.cat([avg, mx, tab_map], dim=1))
        gate = torch.sigmoid(logits)
        return x * gate


class TabGate3D(nn.Module):
    """TabGating: channel and spatial gate inside a learnable residual, ``x + gamma * (y - x)``.

    The temporal attention of TabAttention is left out, since the input is a single CT phase.
    """

    def __init__(
        self,
        channels: int,
        tab_embed_dim: int,
        reduction: int = 16,
        spatial_tab_channels: int = 4,
        spatial_kernel_size: int = 7,
        gamma_init: float = 0.0,
    ):
        super().__init__()
        self.channel_gate = ChannelTabGate3D(channels, tab_embed_dim, reduction=reduction)
        self.spatial_gate = SpatialTabGate3D(tab_embed_dim, tab_channels=spatial_tab_channels, kernel_size=spatial_kernel_size)
        self.gamma = nn.Parameter(torch.tensor(float(gamma_init)))

    def forward(self, x: torch.Tensor, tab_emb: torch.Tensor) -> torch.Tensor:
        y = self.channel_gate(x, tab_emb)
        y = self.spatial_gate(y, tab_emb)
        return x + self.gamma * (y - x)


class MultiScaleTabularAggregator(nn.Module):
    """Fuse pooled features of all encoder stages with the tabular embedding, then FiLM the bottleneck.

    The FiLM output is scaled by a learnable gamma, initialised small and nonzero so that
    gradient reaches every layer from the first step.
    """

    def __init__(self, stage_channels: Sequence[int], tab_embed_dim: int, fused_dim: int = 128, gamma_init: float = 0.05):
        super().__init__()
        self.stage_proj = nn.ModuleList([nn.Linear(int(c), fused_dim) for c in stage_channels])
        self.tab_proj = nn.Linear(tab_embed_dim, fused_dim)
        n_inputs = len(stage_channels) + 1
        self.fuse = nn.Sequential(
            nn.Linear(fused_dim * n_inputs, fused_dim),
            nn.LayerNorm(fused_dim),
            nn.ReLU(inplace=True),
            nn.Linear(fused_dim, fused_dim),
            nn.ReLU(inplace=True),
        )
        bottleneck_channels = int(stage_channels[-1])
        self.film_head = nn.Linear(fused_dim, 2 * bottleneck_channels)
        self.gamma = nn.Parameter(torch.tensor(float(gamma_init)))

    def forward(self, stage_features: Sequence[torch.Tensor], tab_emb: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        pooled = [f.mean(dim=(2, 3, 4)) for f in stage_features]
        projected = [proj(p) for proj, p in zip(self.stage_proj, pooled)]
        combined = torch.cat(projected + [self.tab_proj(tab_emb)], dim=1)
        aggregated = self.fuse(combined)
        scale, shift = self.film_head(aggregated).chunk(2, dim=1)
        return self.gamma * scale, self.gamma * shift

    @staticmethod
    def apply_film(x: torch.Tensor, scale: torch.Tensor, shift: torch.Tensor) -> torch.Tensor:
        b, c = x.shape[0], x.shape[1]
        return x * (1.0 + scale.view(b, c, 1, 1, 1)) + shift.view(b, c, 1, 1, 1)
