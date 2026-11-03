from __future__ import annotations

from typing import Any, Dict

from torch import nn

from .cnn_unet import PlainCNNUNet3D
from .common import ImageOnlyWrapper
from .segmamba_pr import SegMambaPR


def build_model(cfg: Dict[str, Any], tab_dim: int) -> nn.Module:
    model_cfg = cfg["model"]
    name = str(model_cfg.get("name", "monai_unet")).lower()
    in_channels = int(model_cfg.get("in_channels", 1))
    out_channels = int(model_cfg.get("out_channels", cfg.get("num_classes", 6)))
    tab_cfg = cfg.get("tabular", {})
    tabular_mode = str(tab_cfg.get("mode", "numeric"))

    if name == "cnn_unet":
        return PlainCNNUNet3D(
            in_channels=in_channels,
            out_channels=out_channels,
            channels=tuple(model_cfg.get("channels", [24, 48, 96, 192, 256])),
            norm=model_cfg.get("norm", "instance"),
        )

    if name == "monai_unet":
        from monai.networks.nets import UNet
        channels = tuple(model_cfg.get("channels", [32, 64, 128, 256, 320]))
        net = UNet(
            spatial_dims=3,
            in_channels=in_channels,
            out_channels=out_channels,
            channels=channels,
            strides=tuple(model_cfg.get("strides", [2, 2, 2, 2])),
            num_res_units=int(model_cfg.get("num_res_units", 2)),
            norm=model_cfg.get("norm", "INSTANCE"),
        )
        return ImageOnlyWrapper(net)

    if name == "swinunetr":
        from monai.networks.nets import SwinUNETR
        roi_size = tuple(cfg["data"].get("roi_size", [128, 128, 128]))
        kwargs = dict(
            in_channels=in_channels,
            out_channels=out_channels,
            feature_size=int(model_cfg.get("feature_size", 24)),
            use_checkpoint=bool(model_cfg.get("use_checkpoint", True)),
        )
        try:
            net = SwinUNETR(img_size=roi_size, **kwargs)
        except TypeError:
            net = SwinUNETR(**kwargs)
        return ImageOnlyWrapper(net)

    if name in {"segmamba_pr", "segmamba_pr_tabgated"}:
        gated_default = [] if name == "segmamba_pr" else [1, 2, 3, 4]
        require_mamba = bool(model_cfg.get("require_mamba_ssm", True))
        if bool(model_cfg.get("debug_allow_mamba_fallback", False)):
            require_mamba = False
        return SegMambaPR(
            in_chans=in_channels,
            out_chans=out_channels,
            depths=tuple(model_cfg.get("depths", [2, 2, 2, 2])),
            feat_size=tuple(model_cfg.get("feat_size", [48, 96, 192, 384])),
            hidden_size=int(model_cfg.get("hidden_size", 768)),
            norm_name=model_cfg.get("norm_name", "instance"),
            spatial_dims=int(model_cfg.get("spatial_dims", 3)),
            d_state=int(model_cfg.get("d_state", 16)),
            d_conv=int(model_cfg.get("d_conv", 4)),
            expand=int(model_cfg.get("expand", 2)),
            require_mamba_ssm=require_mamba,
            scan_directions=int(model_cfg.get("scan_directions", 3)),
            adaptive_fusion=bool(model_cfg.get("adaptive_fusion", False)),
            fusion_per_channel=bool(model_cfg.get("fusion_per_channel", True)),
            fusion_spatial=bool(model_cfg.get("fusion_spatial", False)),
            fusion_spatial_groups=int(model_cfg.get("fusion_spatial_groups", 4)),
            lk_separable=bool(model_cfg.get("lk_separable", False)),
            lk_anisotropic=bool(model_cfg.get("lk_anisotropic", False)),
            lk_separable_legacy=bool(model_cfg.get("lk_separable_legacy", False)),
            ms_dwconv=bool(model_cfg.get("ms_dwconv", False)),
            ms_dwconv_kernels=tuple(model_cfg.get("ms_dwconv_kernels", [1, 3, 5])),
            ms_dwconv_gamma_init=(None if model_cfg.get("ms_dwconv_gamma_init", 0.05) is None
                                  else float(model_cfg.get("ms_dwconv_gamma_init", 0.05))),
            ms_dwconv_merge_act=bool(model_cfg.get("ms_dwconv_merge_act", False)),
            interface_margin=bool(model_cfg.get("interface_margin", False)),
            interface_margin_radius_mm=float(model_cfg.get("interface_margin_radius_mm", 5.0)),
            interface_margin_spacing=tuple(model_cfg.get(
                "interface_margin_spacing", (cfg.get("infer") or {}).get("spacing", [0.8, 0.8, 3.0]))),
            interface_margin_hidden=int(model_cfg.get("interface_margin_hidden", 16)),
            interface_margin_gamma_init=float(model_cfg.get("interface_margin_gamma_init", 0.05)),
            interface_margin_mask=str(model_cfg.get("interface_margin_mask", "vessel")),
            tab_dim=tab_dim,
            tab_embed_dim=int(model_cfg.get("tab_embed_dim", 64)),
            tab_dropout=float(tab_cfg.get("dropout", 0.25)),
            gated_stages=tuple(model_cfg.get("gated_stages", gated_default)),
            tabular_mode=tabular_mode,
            gamma_init=float(model_cfg.get("gamma_init", 0.0)),
            use_feature_aggregator=bool(model_cfg.get("use_feature_aggregator", False)),
        )

    raise ValueError(f"Unknown model.name={name}")
