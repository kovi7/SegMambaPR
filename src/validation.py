from __future__ import annotations

from typing import Dict

import torch
from monai.inferers import sliding_window_inference


def sliding_window_kwargs(cfg) -> dict:
    """Sliding-window settings shared by validation and inference.

    Gaussian blending of the windows, and replicate padding for volumes shorter than the ROI.
    """
    val = cfg.get("val", {}) or {}
    return {
        "overlap": float(val.get("overlap", 0.5)),
        "mode": str(val.get("sw_mode", "gaussian")),
        "padding_mode": str(val.get("sw_padding_mode", "replicate")),
    }

from .losses import pdac_presence_from_label, pdac_score_from_logits
from .metrics import binary_metrics, dice_per_class_from_logits, finalize_mean_dict, update_mean_dict


@torch.no_grad()
def validate(
    model,
    loader,
    loss_fn,
    cfg: Dict,
    device: torch.device,
    prefix: str = "val",
    force_missing_tabular: bool = False,
) -> Dict[str, float]:
    """Validate segmentation and PDAC detection from the segmentation.

    ``force_missing_tabular`` zeros the tabular values and masks, as at test time.
    """
    model.eval()
    num_classes = int(cfg.get("num_classes", cfg["model"].get("out_channels", 6)))
    roi_size = tuple(cfg["data"].get("roi_size", [128, 128, 128]))
    sw_batch_size = int(cfg.get("val", {}).get("sw_batch_size", 2))
    use_channels_last = bool(cfg.get("train", {}).get("channels_last", False)) and device.type == "cuda"
    pdac_class = int(cfg.get("labels", {}).get("pdac", 1))
    total_loss = 0.0
    n = 0
    compute_loss = bool(cfg.get("val", {}).get("compute_loss", True))
    dice_acc = {}
    y_true = []
    y_score = []

    for batch in loader:
        image = batch["image"].to(device)
        label = batch["label"].to(device)
        tab = batch["tabular"].float().to(device)
        tab_mask = batch["tabular_mask"].float().to(device)
        if force_missing_tabular:
            tab = torch.zeros_like(tab)
            tab_mask = torch.zeros_like(tab_mask)

        def predictor(x):
            if use_channels_last:
                x = x.contiguous(memory_format=torch.channels_last_3d)
            local_tab = tab.expand(x.shape[0], -1) if tab.shape[0] == 1 and x.shape[0] > 1 else tab[: x.shape[0]]
            local_mask = tab_mask.expand(x.shape[0], -1) if tab_mask.shape[0] == 1 and x.shape[0] > 1 else tab_mask[: x.shape[0]]
            return model(x, tabular=local_tab, tabular_mask=local_mask)["seg_logits"]

        logits = sliding_window_inference(
            image,
            roi_size=roi_size,
            sw_batch_size=sw_batch_size,
            predictor=predictor,
            **sliding_window_kwargs(cfg),
        )
        if compute_loss:
            try:
                loss = loss_fn(logits, label)
                total_loss += float(loss.detach().cpu())
                n += 1
            except torch.cuda.OutOfMemoryError:
                # the full-volume one-hot label can exhaust memory; the loss never selects a checkpoint
                compute_loss = False
                torch.cuda.empty_cache()
                print(f"Validation loss disabled after CUDA OOM at {num_classes} classes; "
                      "Dice and detection metrics are unaffected.", flush=True)
        update_mean_dict(dice_acc, dice_per_class_from_logits(logits, label, num_classes=num_classes))
        y_true.extend(pdac_presence_from_label(label, pdac_class=pdac_class).detach().cpu().numpy().astype(int).tolist())
        y_score.extend(pdac_score_from_logits(logits, pdac_class=pdac_class).detach().cpu().numpy().tolist())

    out = {f"{prefix}_loss": total_loss / n} if n else {}
    out.update(finalize_mean_dict(dice_acc, prefix=f"{prefix}_"))
    out.update({f"{prefix}_pdac_" + k: v for k, v in binary_metrics(y_true, y_score, threshold=float(cfg.get("val", {}).get("pdac_threshold", 0.5))).items()})
    return out
