from __future__ import annotations

from typing import Dict, Optional, Sequence

import torch
import torch.nn.functional as F
from torch import nn
from monai.losses import DiceCELoss


class VesselBandWeight(nn.Module):
    """Band of ``radius_mm`` around the ground-truth vessels, where the CE weight is raised.

    The band is a max-pool dilation with a kernel derived from the voxel spacing, so the
    radius is a physical distance. It is used only inside the loss.
    """

    def __init__(
        self,
        classes: Sequence[int] = (2, 3),
        radius_mm: float = 5.0,
        spacing: Sequence[float] = (0.8, 0.8, 3.0),
        weight: float = 2.5,
    ):
        super().__init__()
        self.classes = tuple(int(c) for c in classes)
        self.weight = float(weight)
        radii = [max(int(round(float(radius_mm) / float(s))), 0) for s in spacing]
        self.kernel = tuple(2 * r + 1 for r in radii)
        self.padding = tuple(radii)

    @torch.no_grad()
    def forward(self, label: torch.Tensor) -> torch.Tensor:
        mask = torch.zeros_like(label, dtype=torch.float32)
        for c in self.classes:
            mask = torch.maximum(mask, (label == c).float())
        return F.max_pool3d(mask, kernel_size=self.kernel, stride=1, padding=self.padding)


def build_vessel_band(cfg: Optional[Dict]) -> Optional[VesselBandWeight]:
    if not cfg or not bool(cfg.get("enabled", False)):
        return None
    return VesselBandWeight(
        classes=cfg.get("classes", [2, 3]),
        radius_mm=float(cfg.get("radius_mm", 5.0)),
        spacing=cfg.get("spacing", [0.8, 0.8, 3.0]),
        weight=float(cfg.get("weight", 2.5)),
    )


class SegmentationLoss(nn.Module):
    """Combined Dice and cross-entropy loss over the foreground classes."""

    def __init__(self, num_classes: int, include_background: bool = False,
                 vessel_band: Optional[VesselBandWeight] = None):
        super().__init__()
        self.loss = DiceCELoss(
            to_onehot_y=True,
            softmax=True,
            include_background=include_background,
            squared_pred=True,
            smooth_nr=1e-5,
            smooth_dr=1e-5,
        )
        self.vessel_band = vessel_band

    def _band_term(self, logits: torch.Tensor, label: torch.Tensor) -> torch.Tensor:
        # mean(w * ce) = mean(ce) + (w - 1) * sum(ce * band) / numel; mean(ce) is already in DiceCELoss
        band = self.vessel_band(label)
        ce = F.cross_entropy(logits, label[:, 0], reduction="none")
        surplus = (self.vessel_band.weight - 1.0) * (ce * band[:, 0]).sum() / ce.numel()
        return surplus

    def forward(self, logits: torch.Tensor, label: torch.Tensor) -> torch.Tensor:
        if label.ndim == 4:
            label = label.unsqueeze(1)
        label = label.long()
        max_label = int(label.max().detach().cpu()) if label.numel() else 0
        if max_label >= logits.shape[1]:
            raise ValueError(
                f"Label value {max_label} is outside model output range 0..{logits.shape[1]-1}. "
                "For PANORAMA, use the converted 6-label schema: 0 background, 1 PDAC, "
                "2 veins, 3 arteries, 4 pancreas+pancreatic duct, 5 CBD. "
                "Raw masks with CBD=6 must be converted before training."
            )
        total = self.loss(logits, label)
        if self.vessel_band is not None:
            total = total + self._band_term(logits, label)
        return total


def pdac_presence_from_label(label: torch.Tensor, pdac_class: int = 1) -> torch.Tensor:
    if label.ndim == 5:
        label = label[:, 0]
    return (label == int(pdac_class)).flatten(1).any(dim=1).float()


def pdac_score_from_logits(seg_logits: torch.Tensor, pdac_class: int = 1) -> torch.Tensor:
    prob = torch.softmax(seg_logits, dim=1)[:, int(pdac_class)]
    return prob.flatten(1).amax(dim=1)
