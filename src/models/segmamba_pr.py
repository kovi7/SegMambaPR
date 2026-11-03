# MambaLayer, MlpChannel, MambaEncoder and the encoder-decoder layout of SegMambaPR are adapted from
# SegMamba (https://github.com/ge-xing/SegMamba), whose model file carries the notice
# "Copyright (c) MONAI Consortium, Licensed under the Apache License, Version 2.0".
from __future__ import annotations

from typing import Optional, Sequence

import torch
import torch.nn.functional as F
from torch import nn
from einops import rearrange
from monai.networks.blocks.dynunet_block import UnetOutBlock
from monai.networks.blocks.unetr_block import UnetrBasicBlock, UnetrUpBlock

from .common import ConvNormAct3d
from .tab_gating import MultiScaleTabularAggregator, TabGate3D, build_tabular_embedding

try:
    from mamba_ssm import Mamba
except Exception:  # MambaLayer raises if mamba-ssm is required
    Mamba = None


class AdaptiveDirectionalFusion(nn.Module):
    """Weights of the Mamba scan directions predicted from globally pooled features.

    The last projection is zero-initialised and the sum is rescaled by ``n_directions``,
    so the module starts as the equal-weight sum. ``direction_usage`` keeps a running
    mean of the weights in the checkpoint.
    """

    def __init__(self, channels: int, n_directions: int, per_channel: bool = True, momentum: float = 0.01):
        super().__init__()
        self.n_directions = int(n_directions)
        self.per_channel = bool(per_channel)
        self.momentum = float(momentum)
        hidden = max(int(channels) // 4, 8)
        out_dim = self.n_directions * int(channels) if self.per_channel else self.n_directions
        self.mlp = nn.Sequential(
            nn.Linear(int(channels), hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, out_dim),
        )
        nn.init.zeros_(self.mlp[-1].weight)
        nn.init.zeros_(self.mlp[-1].bias)
        self.register_buffer("direction_usage", torch.full((self.n_directions,), 1.0 / self.n_directions))

    def forward(self, x: torch.Tensor, outs: Sequence[torch.Tensor]) -> torch.Tensor:
        if len(outs) != self.n_directions:
            raise ValueError(f"expected {self.n_directions} direction outputs, got {len(outs)}")
        b, c = x.shape[:2]
        logits = self.mlp(x.mean(dim=(2, 3, 4)))
        logits = logits.view(b, self.n_directions, c if self.per_channel else 1)
        w = torch.softmax(logits, dim=1)

        if self.training:
            with torch.no_grad():
                mean_w = w.detach().float().mean(dim=(0, 2))
                self.direction_usage.mul_(1 - self.momentum).add_(self.momentum * mean_w)

        stacked = torch.stack(list(outs), dim=1)                       # B, n_dir, C, D, H, W
        return self.n_directions * (w.view(b, self.n_directions, -1, 1, 1, 1) * stacked).sum(dim=1)


class SpatialDirectionalFusion(nn.Module):
    """Per-voxel weights of the Mamba scan directions, shared within ``groups`` of channels.

    A pointwise projection and a 3^3 convolution predict the weights, with a softmax over
    directions at every voxel. Initialised as the equal-weight sum, like
    ``AdaptiveDirectionalFusion``. ``store_weights`` keeps the last weight field.
    """

    def __init__(self, channels: int, n_directions: int, groups: int = 4, momentum: float = 0.01):
        super().__init__()
        self.n_directions = int(n_directions)
        self.groups = max(1, min(int(groups), int(channels)))
        while int(channels) % self.groups != 0 and self.groups > 1:
            self.groups -= 1
        self.momentum = float(momentum)
        self.store_weights = False
        self.last_weights: Optional[torch.Tensor] = None
        hidden = max(int(channels) // 8, 8)
        self.head = nn.Sequential(
            nn.Conv3d(int(channels), hidden, kernel_size=1, bias=False),
            nn.GroupNorm(hidden, hidden, affine=False),
            nn.ReLU(inplace=True),
            nn.Conv3d(hidden, self.n_directions * self.groups, kernel_size=3, padding=1),
        )
        nn.init.zeros_(self.head[-1].weight)
        nn.init.zeros_(self.head[-1].bias)
        self.register_buffer("direction_usage", torch.full((self.n_directions,), 1.0 / self.n_directions))

    def forward(self, x: torch.Tensor, outs: Sequence[torch.Tensor]) -> torch.Tensor:
        if len(outs) != self.n_directions:
            raise ValueError(f"expected {self.n_directions} direction outputs, got {len(outs)}")
        b, c = x.shape[:2]
        spatial = x.shape[2:]
        logits = self.head(x).view(b, self.n_directions, self.groups, *spatial)
        w = torch.softmax(logits, dim=1)

        if self.training:
            with torch.no_grad():
                mean_w = w.detach().float().mean(dim=(0, 2, 3, 4, 5))
                self.direction_usage.mul_(1 - self.momentum).add_(self.momentum * mean_w)
        if self.store_weights and not self.training:
            self.last_weights = w.detach()

        w = w.repeat_interleave(c // self.groups, dim=2)
        stacked = torch.stack(list(outs), dim=1)                       # B, n_dir, C, D, H, W
        return self.n_directions * (w * stacked).sum(dim=1)


class MultiScaleDWConv3d(nn.Module):
    """Multi-scale depthwise stem before the Mamba scan, as a residual branch.

    Parallel depthwise branches (kernel sizes in-plane, through-plane capped at 3) are summed
    and mixed by a pointwise convolution. The repaired form (``merge_act=False``,
    ``gamma_init=0.05``) is zero-mean and starts close to identity. ``gamma_init=None`` with
    ``merge_act=True`` gives the first form.
    """

    def __init__(self, channels: int, kernel_sizes: Sequence[int] = (1, 3, 5), norm: str = "instance",
                 gamma_init: Optional[float] = 0.05, merge_act: bool = False):
        super().__init__()
        ks = [int(k) for k in kernel_sizes if int(k) >= 1]
        if not ks:
            raise ValueError("ms_dwconv_kernels must contain at least one kernel size")
        self.branches = nn.ModuleList()
        for k in ks:
            kz = 1 if k == 1 else 3
            self.branches.append(
                nn.Conv3d(channels, channels, kernel_size=(k, k, kz),
                          padding=(k // 2, k // 2, kz // 2), groups=channels, bias=False)
            )
        self.merge = ConvNormAct3d(channels, channels, kernel_size=1, norm=norm, act=merge_act)
        self.gamma = nn.Parameter(torch.tensor(float(gamma_init))) if gamma_init is not None else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.branches[0](x)
        for branch in self.branches[1:]:
            y = y + branch(x)
        y = self.merge(y)
        if self.gamma is not None:
            y = self.gamma * y
        return x + y


class MambaLayer(nn.Module):
    """Mamba layer that scans the volume in one to three flattening orders."""

    def __init__(self, dim: int, d_state: int = 16, d_conv: int = 4, expand: int = 2, require_mamba_ssm: bool = True,
                 scan_directions: int = 3, adaptive_fusion: bool = False, fusion_per_channel: bool = True,
                 fusion_spatial: bool = False, fusion_spatial_groups: int = 4):
        super().__init__()
        if Mamba is None and require_mamba_ssm:
            raise ImportError("mamba-ssm is required for SegMambaV2. Install mamba-ssm or disable this model.")
        self.dim = int(dim)
        self.scan_directions = int(scan_directions)
        if self.scan_directions not in (1, 2, 3):
            raise ValueError(f"scan_directions must be 1, 2 or 3, got {scan_directions}")
        self.norm = nn.LayerNorm(dim)
        # without mamba-ssm an MLP keeps the shapes, for testing only; it is not a scan
        if Mamba is not None:
            self.mamba = Mamba(d_model=dim, d_state=d_state, d_conv=d_conv, expand=expand)
        else:
            self.mamba = nn.Sequential(nn.Linear(dim, dim), nn.GELU(), nn.Linear(dim, dim))
        if adaptive_fusion and self.scan_directions > 1:
            self.fusion = (
                SpatialDirectionalFusion(dim, self.scan_directions, groups=fusion_spatial_groups)
                if fusion_spatial
                else AdaptiveDirectionalFusion(dim, self.scan_directions, per_channel=fusion_per_channel)
            )
        else:
            self.fusion = None

    def mamba_forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c = x.shape[:2]
        n_tokens = x.shape[2:].numel()
        img_dims = x.shape[2:]
        x = x.reshape(b, c, n_tokens).transpose(-1, -2)
        x = self.norm(x)
        x = self.mamba(x)
        x = x.transpose(-1, -2).reshape(b, c, *img_dims)
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_skip = x
        outs = [self.mamba_forward(x)]
        if self.scan_directions >= 2:
            x_2 = rearrange(x, "b c d w h -> b c w d h")
            outs.append(rearrange(self.mamba_forward(x_2), "b c w d h -> b c d w h"))
        if self.scan_directions >= 3:
            x_3 = rearrange(x, "b c d w h -> b c h w d")
            outs.append(rearrange(self.mamba_forward(x_3), "b c h w d -> b c d w h"))
        if self.fusion is not None:
            out = self.fusion(x, outs) + x_skip
        else:
            out = torch.stack(outs, dim=0).sum(dim=0) + x_skip
        # the flattening for the scan loses channels_last_3d; restore the input's memory format
        if x_skip.is_contiguous(memory_format=torch.channels_last_3d) and not out.is_contiguous(
            memory_format=torch.channels_last_3d
        ):
            out = out.contiguous(memory_format=torch.channels_last_3d)
        return out


def _kernel(k: int, anisotropic: bool):
    """Kernel size and padding; with ``anisotropic`` the through-plane extent is capped at 3."""
    if not anisotropic:
        return (k, k, k), (k // 2, k // 2, k // 2)
    kz = min(k, 3)
    return (k, k, kz), (k // 2, k // 2, kz // 2)


class SeparableConvBlock(nn.Module):
    """Depthwise k^3 convolution followed by a pointwise one, in place of a dense k^3.

    ``legacy=True`` builds the variant of the reported ``segmamba_pr_tabgated_sep`` run
    (no pointwise bias, affine GroupNorm, LeakyReLU).
    """

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, dilation: int = 1,
                 anisotropic: bool = False, legacy: bool = False):
        super().__init__()
        k, pad = _kernel(kernel_size, anisotropic)
        pad = tuple(p * dilation for p in pad)
        if legacy:
            self.block = nn.Sequential(
                nn.Conv3d(in_channels, in_channels, kernel_size=k, padding=pad, dilation=dilation,
                          groups=in_channels, bias=False),
                nn.Conv3d(in_channels, out_channels, kernel_size=1, bias=False),
                nn.GroupNorm(out_channels, out_channels, affine=True),
                nn.LeakyReLU(0.01, inplace=True),
            )
            return
        self.block = nn.Sequential(
            nn.Conv3d(in_channels, in_channels, kernel_size=k, padding=pad, dilation=dilation,
                      groups=in_channels, bias=False),
            nn.Conv3d(in_channels, out_channels, kernel_size=1, bias=True),
            nn.GroupNorm(out_channels, out_channels, affine=False),  # instance norm, keeps channels_last_3d
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class ConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, padding: int, dilation: int = 1):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size, padding=padding, dilation=dilation),
            nn.GroupNorm(out_channels, out_channels, affine=False),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class SEBlock(nn.Module):
    def __init__(self, channels: int, reduction: int = 8):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool3d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, max(channels // reduction, 4)),
            nn.ReLU(inplace=True),
            nn.Linear(max(channels // reduction, 4), channels),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, _, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1, 1)
        return x * y


class LargeKernelConv(nn.Module):
    """Large-kernel convolution block of SegMamba-V2; ``separable`` and ``anisotropic`` are off by default."""

    def __init__(self, channels: int, separable: bool = False, anisotropic: bool = False,
                 separable_legacy: bool = False):
        super().__init__()
        if separable:
            def blk(k, dilation=1):
                return SeparableConvBlock(channels, channels, k, dilation=dilation, anisotropic=anisotropic,
                                          legacy=separable_legacy)
        else:
            def blk(k, dilation=1):
                pad = (k // 2) * dilation
                return ConvBlock(channels, channels, k, padding=pad, dilation=dilation)
        self.deep_path = nn.Sequential(blk(5), blk(3), blk(3, dilation=2))
        self.shortcut_path = nn.Sequential(blk(3), blk(1))
        self.se = SEBlock(channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.se(self.deep_path(x) + self.shortcut_path(x) + x)


class MlpChannel(nn.Module):
    def __init__(self, hidden_size: int, mlp_dim: int):
        super().__init__()
        self.fc1 = nn.Conv3d(hidden_size, mlp_dim, 1)
        self.act = nn.GELU()
        self.fc2 = nn.Conv3d(mlp_dim, hidden_size, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(self.act(self.fc1(x)))


class MambaEncoder(nn.Module):
    def __init__(
        self,
        in_chans: int = 1,
        depths: Sequence[int] = (2, 2, 2, 2),
        dims: Sequence[int] = (48, 96, 192, 384),
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
        require_mamba_ssm: bool = True,
        out_indices: Sequence[int] = (0, 1, 2, 3),
        scan_directions: int = 3,
        adaptive_fusion: bool = False,
        fusion_per_channel: bool = True,
        fusion_spatial: bool = False,
        fusion_spatial_groups: int = 4,
        lk_separable: bool = False,
        lk_anisotropic: bool = False,
        lk_separable_legacy: bool = False,
        ms_dwconv: bool = False,
        ms_dwconv_kernels: Sequence[int] = (1, 3, 5),
        ms_dwconv_gamma_init: Optional[float] = 0.05,
        ms_dwconv_merge_act: bool = False,
    ):
        super().__init__()
        self.downsample_layers = nn.ModuleList()
        self.downsample_layers.append(nn.Sequential(nn.Conv3d(dims[0], dims[0], kernel_size=3, stride=2, padding=1)))
        for i in range(3):
            self.downsample_layers.append(nn.Sequential(nn.Conv3d(dims[i], dims[i + 1], kernel_size=3, stride=2, padding=1)))

        self.stages = nn.ModuleList()
        self.gscs = nn.ModuleList()
        self.ms_dwconvs = nn.ModuleList()
        for i in range(4):
            self.gscs.append(nn.Sequential(*[LargeKernelConv(dims[i], separable=lk_separable, anisotropic=lk_anisotropic,
                                                             separable_legacy=lk_separable_legacy)
                                             for _ in range(depths[i])]))
            if i < 2:
                self.stages.append(None)
                self.ms_dwconvs.append(None)
            else:
                self.ms_dwconvs.append(
                    MultiScaleDWConv3d(dims[i], ms_dwconv_kernels,
                                       gamma_init=ms_dwconv_gamma_init,
                                       merge_act=ms_dwconv_merge_act)
                    if ms_dwconv else None)
                self.stages.append(nn.Sequential(*[
                    MambaLayer(dim=dims[i], d_state=d_state, d_conv=d_conv, expand=expand, require_mamba_ssm=require_mamba_ssm,
                               scan_directions=scan_directions, adaptive_fusion=adaptive_fusion, fusion_per_channel=fusion_per_channel,
                               fusion_spatial=fusion_spatial, fusion_spatial_groups=fusion_spatial_groups)
                    for _ in range(depths[i])
                ]))
        self.out_indices = tuple(out_indices)
        for i_layer in range(4):
            self.add_module(f"norm{i_layer}", nn.GroupNorm(dims[i_layer], dims[i_layer], affine=False))
        self.mlps = nn.ModuleList([MlpChannel(dims[i], 2 * dims[i]) for i in range(4)])

    def forward_features(self, x: torch.Tensor):
        outs = []
        for i in range(4):
            x = self.downsample_layers[i](x)
            x = self.gscs[i](x)
            if self.ms_dwconvs[i] is not None:
                x = self.ms_dwconvs[i](x)
            if self.stages[i] is not None:
                x = self.stages[i](x)
            if i in self.out_indices:
                norm_layer = getattr(self, f"norm{i}")
                outs.append(self.mlps[i](norm_layer(x)))
        return tuple(outs)

    def forward(self, x: torch.Tensor):
        return self.forward_features(x)


class InterfaceMarginHead(nn.Module):
    """Interface margin: a learned per-voxel correction added to the tumour logit.

    A small convolutional branch on the last decoder features predicts one scalar per voxel.
    It is masked by a soft band built from the model's own predicted probabilities and dilated
    by ``radius_mm`` (spacing-aware max pool). ``mask_mode="interface"`` multiplies the vessel
    band by the dilated tumour probability, so the correction acts only where predicted tumour
    and predicted vessel meet; ``"vessel"`` uses the vessel band alone. The probabilities are
    detached, and ``gamma`` starts at 0.05.
    """

    def __init__(
        self,
        channels: int,
        tumour_class: int = 1,
        vessel_classes: Sequence[int] = (2, 3),
        radius_mm: float = 5.0,
        spacing: Sequence[float] = (0.8, 0.8, 3.0),
        hidden: int = 16,
        gamma_init: float = 0.05,
        mask_mode: str = "vessel",
    ):
        super().__init__()
        if mask_mode not in ("vessel", "interface"):
            raise ValueError(f"interface_margin_mask must be 'vessel' or 'interface', got {mask_mode!r}")
        self.mask_mode = mask_mode
        self.tumour_class = int(tumour_class)
        self.vessel_classes = tuple(int(c) for c in vessel_classes)
        radii = [max(int(round(float(radius_mm) / float(sp))), 0) for sp in spacing]
        self.kernel = tuple(2 * r + 1 for r in radii)
        self.padding = tuple(radii)
        self.branch = nn.Sequential(
            nn.Conv3d(int(channels), int(hidden), kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(int(hidden), int(hidden), affine=False),
            nn.ReLU(inplace=True),
            nn.Conv3d(int(hidden), 1, kernel_size=1),
        )
        nn.init.zeros_(self.branch[-1].bias)
        self.gamma = nn.Parameter(torch.tensor(float(gamma_init)))

    def vessel_band(self, logits: torch.Tensor) -> torch.Tensor:
        """Soft near-vessel mask from the model's own prediction, dilated by radius_mm."""
        prob = torch.softmax(logits.detach().float(), dim=1)
        vessel = prob[:, self.vessel_classes, ...].sum(dim=1, keepdim=True)
        return F.max_pool3d(vessel, kernel_size=self.kernel, stride=1, padding=self.padding)

    def tumour_band(self, logits: torch.Tensor) -> torch.Tensor:
        """Soft near-tumour mask from the model's own prediction, dilated by radius_mm (detached)."""
        prob = torch.softmax(logits.detach().float(), dim=1)
        tumour = prob[:, self.tumour_class:self.tumour_class + 1, ...]
        return F.max_pool3d(tumour, kernel_size=self.kernel, stride=1, padding=self.padding)

    def forward(self, feats: torch.Tensor, logits: torch.Tensor) -> torch.Tensor:
        band = self.vessel_band(logits)
        if self.mask_mode == "interface":
            band = band * self.tumour_band(logits)
        band = band.to(logits.dtype)
        delta = self.branch(feats) * band
        bump = torch.zeros_like(logits)
        bump[:, self.tumour_class:self.tumour_class + 1] = self.gamma * delta
        return logits + bump


class SegMambaPR(nn.Module):
    """SegMambaPR: SegMamba-V2 backbone with optional TabGating and the added modules.

    With every flag at its default the model is SegMamba-V2. Gated stage indices refer to
    the skip features: 0-3 = encoder1-4, 4 = bottleneck.
    """

    def __init__(
        self,
        in_chans: int = 1,
        out_chans: int = 6,
        depths: Sequence[int] = (2, 2, 2, 2),
        feat_size: Sequence[int] = (48, 96, 192, 384),
        hidden_size: int = 768,
        norm_name: str = "instance",
        spatial_dims: int = 3,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
        require_mamba_ssm: bool = True,
        scan_directions: int = 3,
        adaptive_fusion: bool = False,
        fusion_per_channel: bool = True,
        fusion_spatial: bool = False,
        fusion_spatial_groups: int = 4,
        lk_separable: bool = False,
        lk_anisotropic: bool = False,
        lk_separable_legacy: bool = False,
        ms_dwconv: bool = False,
        ms_dwconv_kernels: Sequence[int] = (1, 3, 5),
        ms_dwconv_gamma_init: Optional[float] = 0.05,
        ms_dwconv_merge_act: bool = False,
        interface_margin: bool = False,
        interface_margin_radius_mm: float = 5.0,
        interface_margin_spacing: Sequence[float] = (0.8, 0.8, 3.0),
        interface_margin_hidden: int = 16,
        interface_margin_gamma_init: float = 0.05,
        interface_margin_mask: str = "vessel",
        tab_dim: int = 0,
        tab_embed_dim: int = 64,
        tab_dropout: float = 0.0,
        gated_stages: Sequence[int] = (),
        tabular_mode: str = "numeric",
        gamma_init: float = 0.0,
        use_feature_aggregator: bool = False,
    ):
        super().__init__()
        self.in_chans = int(in_chans)
        self.out_chans = int(out_chans)
        self.feat_size = tuple(int(v) for v in feat_size)
        self.hidden_size = int(hidden_size)
        self.use_tabular = tab_dim > 0 and len(gated_stages) > 0
        self.gated_stages = {int(s) for s in gated_stages}
        self.tab_encoder = build_tabular_embedding(tab_dim, tab_embed_dim, tab_dropout, mode=tabular_mode) if self.use_tabular else None

        self.vit = MambaEncoder(
            in_chans=in_chans,
            depths=depths,
            dims=self.feat_size,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand,
            require_mamba_ssm=require_mamba_ssm,
            scan_directions=scan_directions,
            adaptive_fusion=adaptive_fusion,
            fusion_per_channel=fusion_per_channel,
            fusion_spatial=fusion_spatial,
            fusion_spatial_groups=fusion_spatial_groups,
            lk_separable=lk_separable,
            lk_anisotropic=lk_anisotropic,
            lk_separable_legacy=lk_separable_legacy,
            ms_dwconv=ms_dwconv,
            ms_dwconv_kernels=ms_dwconv_kernels,
            ms_dwconv_gamma_init=ms_dwconv_gamma_init,
            ms_dwconv_merge_act=ms_dwconv_merge_act,
        )
        def _cl_safe_norm(channels: int):
            # instance norm as GroupNorm(C, C), which keeps channels_last_3d
            return ("group", {"num_groups": channels, "affine": False}) if norm_name == "instance" else norm_name

        self.encoder1 = UnetrBasicBlock(spatial_dims=spatial_dims, in_channels=in_chans, out_channels=self.feat_size[0], kernel_size=3, stride=1, norm_name=_cl_safe_norm(self.feat_size[0]), res_block=True)
        self.encoder2 = UnetrBasicBlock(spatial_dims=spatial_dims, in_channels=self.feat_size[0], out_channels=self.feat_size[1], kernel_size=3, stride=1, norm_name=_cl_safe_norm(self.feat_size[1]), res_block=True)
        self.encoder3 = UnetrBasicBlock(spatial_dims=spatial_dims, in_channels=self.feat_size[1], out_channels=self.feat_size[2], kernel_size=3, stride=1, norm_name=_cl_safe_norm(self.feat_size[2]), res_block=True)
        self.encoder4 = UnetrBasicBlock(spatial_dims=spatial_dims, in_channels=self.feat_size[2], out_channels=self.feat_size[3], kernel_size=3, stride=1, norm_name=_cl_safe_norm(self.feat_size[3]), res_block=True)
        self.encoder5 = UnetrBasicBlock(spatial_dims=spatial_dims, in_channels=self.feat_size[3], out_channels=self.hidden_size, kernel_size=3, stride=1, norm_name=_cl_safe_norm(self.hidden_size), res_block=True)

        self.decoder5 = UnetrUpBlock(spatial_dims=spatial_dims, in_channels=self.hidden_size, out_channels=self.feat_size[3], kernel_size=3, upsample_kernel_size=2, norm_name=_cl_safe_norm(self.feat_size[3]), res_block=True)
        self.decoder4 = UnetrUpBlock(spatial_dims=spatial_dims, in_channels=self.feat_size[3], out_channels=self.feat_size[2], kernel_size=3, upsample_kernel_size=2, norm_name=_cl_safe_norm(self.feat_size[2]), res_block=True)
        self.decoder3 = UnetrUpBlock(spatial_dims=spatial_dims, in_channels=self.feat_size[2], out_channels=self.feat_size[1], kernel_size=3, upsample_kernel_size=2, norm_name=_cl_safe_norm(self.feat_size[1]), res_block=True)
        self.decoder2 = UnetrUpBlock(spatial_dims=spatial_dims, in_channels=self.feat_size[1], out_channels=self.feat_size[0], kernel_size=3, upsample_kernel_size=2, norm_name=_cl_safe_norm(self.feat_size[0]), res_block=True)
        self.decoder1 = UnetrBasicBlock(spatial_dims=spatial_dims, in_channels=self.feat_size[0], out_channels=self.feat_size[0], kernel_size=3, stride=1, norm_name=_cl_safe_norm(self.feat_size[0]), res_block=True)
        self.out = UnetOutBlock(spatial_dims=spatial_dims, in_channels=self.feat_size[0], out_channels=self.out_chans)

        self.enc_gates = nn.ModuleDict()
        gate_channels = {0: self.feat_size[0], 1: self.feat_size[1], 2: self.feat_size[2], 3: self.feat_size[3], 4: self.hidden_size}
        if self.use_tabular:
            for i in self.gated_stages:
                if i in gate_channels:
                    self.enc_gates[str(i)] = TabGate3D(gate_channels[i], tab_embed_dim, gamma_init=gamma_init)

        self.aggregator = (
            MultiScaleTabularAggregator([gate_channels[i] for i in range(5)], tab_embed_dim)
            if (self.use_tabular and use_feature_aggregator)
            else None
        )

        self.interface_margin = (
            InterfaceMarginHead(
                self.feat_size[0],
                radius_mm=interface_margin_radius_mm,
                spacing=interface_margin_spacing,
                hidden=interface_margin_hidden,
                gamma_init=interface_margin_gamma_init,
                mask_mode=interface_margin_mask,
            )
            if interface_margin
            else None
        )

    def _tab_emb(self, tabular: Optional[torch.Tensor], tabular_mask: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
        if not self.use_tabular:
            return None
        return self.tab_encoder(tabular, tabular_mask)

    def _gate(self, x: torch.Tensor, stage: int, tab_emb: Optional[torch.Tensor]) -> torch.Tensor:
        if tab_emb is not None and str(stage) in self.enc_gates:
            return self.enc_gates[str(stage)](x, tab_emb)
        return x

    def forward(self, image: torch.Tensor, tabular: Optional[torch.Tensor] = None, tabular_mask: Optional[torch.Tensor] = None):
        tab_emb = self._tab_emb(tabular, tabular_mask)
        enc1 = self._gate(self.encoder1(image), 0, tab_emb)
        outs = self.vit(enc1)
        enc2 = self._gate(self.encoder2(outs[0]), 1, tab_emb)
        enc3 = self._gate(self.encoder3(outs[1]), 2, tab_emb)
        enc4 = self._gate(self.encoder4(outs[2]), 3, tab_emb)
        enc_hidden = self._gate(self.encoder5(outs[3]), 4, tab_emb)
        if self.aggregator is not None and tab_emb is not None:
            scale, shift = self.aggregator([enc1, enc2, enc3, enc4, enc_hidden], tab_emb)
            enc_hidden = MultiScaleTabularAggregator.apply_film(enc_hidden, scale, shift)
        dec3 = self.decoder5(enc_hidden, enc4)
        dec2 = self.decoder4(dec3, enc3)
        dec1 = self.decoder3(dec2, enc2)
        dec0 = self.decoder2(dec1, enc1)
        out = self.decoder1(dec0)
        seg_logits = self.out(out)
        if self.interface_margin is not None:
            seg_logits = self.interface_margin(out, seg_logits)
        return {"seg_logits": seg_logits}
