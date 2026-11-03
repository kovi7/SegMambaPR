from __future__ import annotations

from typing import Any, Dict, List, Sequence

import numpy as np
import torch
from scipy import ndimage as ndi


def dice_per_class_from_logits(logits: torch.Tensor, label: torch.Tensor, num_classes: int) -> Dict[str, float]:
    pred = torch.argmax(logits, dim=1)
    if label.ndim == 5:
        label = label[:, 0]
    out = {}
    for c in range(1, num_classes):
        p = pred == c
        y = label == c
        inter = (p & y).sum().float()
        denom = p.sum().float() + y.sum().float()
        out[f"dice_c{c}"] = float("nan") if denom == 0 else float((2 * inter / denom).detach().cpu())
    valid = [v for v in out.values() if not np.isnan(v)]
    out["dice_mean_fg"] = float(np.mean(valid)) if valid else float("nan")
    return out


def binary_metrics(
    y_true: Sequence[int],
    y_score: Sequence[float],
    threshold: float = 0.5,
    y_pred: Optional[Sequence[int]] = None,
) -> Dict[str, float]:
    """Threshold metrics and ranking metrics (AUROC, AP) of a continuous score.

    ``y_pred`` overrides the decision ``y_score >= threshold`` when given.
    """
    y_true = np.asarray(y_true).astype(int)
    y_score = np.asarray(y_score).astype(float)
    y_pred = (y_score >= threshold).astype(int) if y_pred is None else np.asarray(y_pred).astype(int)
    tp = int(((y_true == 1) & (y_pred == 1)).sum())
    tn = int(((y_true == 0) & (y_pred == 0)).sum())
    fp = int(((y_true == 0) & (y_pred == 1)).sum())
    fn = int(((y_true == 1) & (y_pred == 0)).sum())
    sens = tp / (tp + fn + 1e-8)
    spec = tn / (tn + fp + 1e-8)
    prec = tp / (tp + fp + 1e-8)
    f1 = 2 * prec * sens / (prec + sens + 1e-8)
    out = {
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "accuracy": float((tp + tn) / (tp + tn + fp + fn + 1e-8)),
        "sensitivity": float(sens),
        "specificity": float(spec),
        "precision": float(prec),
        "f1": float(f1),
        "balanced_accuracy": float(0.5 * (sens + spec)),
    }
    try:
        from sklearn.metrics import average_precision_score, roc_auc_score
        if len(np.unique(y_true)) == 2:
            out["auroc"] = float(roc_auc_score(y_true, y_score))
            out["ap"] = float(average_precision_score(y_true, y_score))
    except Exception:
        pass
    return out


def multiclass_metrics(y_true: Sequence[str], y_pred: Sequence[str], categories: Sequence[str]) -> Dict[str, Any]:
    """One-vs-rest precision, recall and F1 per category, macro averages and accuracy.

    Categories without support are left out of the macro averages.
    """
    y_true = list(y_true)
    y_pred = list(y_pred)
    n = len(y_true)
    out: Dict[str, Any] = {"n": n, "accuracy": float(sum(t == p for t, p in zip(y_true, y_pred)) / n) if n else float("nan")}

    per_class: Dict[str, Dict[str, float]] = {}
    recalls, precisions, f1s = [], [], []
    for cat in categories:
        tp = sum(1 for t, p in zip(y_true, y_pred) if t == cat and p == cat)
        fn = sum(1 for t, p in zip(y_true, y_pred) if t == cat and p != cat)
        fp = sum(1 for t, p in zip(y_true, y_pred) if t != cat and p == cat)
        tn = sum(1 for t, p in zip(y_true, y_pred) if t != cat and p != cat)
        support = tp + fn
        recall = tp / (tp + fn + 1e-8)
        precision = tp / (tp + fp + 1e-8)
        f1 = 2 * precision * recall / (precision + recall + 1e-8)
        per_class[cat] = {
            "support": support, "tp": tp, "fp": fp, "fn": fn, "tn": tn,
            "recall": float(recall), "precision": float(precision), "f1": float(f1),
        }
        if support > 0:
            recalls.append(recall)
            precisions.append(precision)
            f1s.append(f1)

    out["per_class"] = per_class
    out["macro_recall"] = float(np.mean(recalls)) if recalls else float("nan")
    out["macro_precision"] = float(np.mean(precisions)) if precisions else float("nan")
    out["macro_f1"] = float(np.mean(f1s)) if f1s else float("nan")
    # multiclass balanced accuracy = macro-averaged recall
    out["balanced_accuracy"] = out["macro_recall"]
    return out


def update_mean_dict(acc: Dict[str, List[float]], values: Dict[str, float]) -> None:
    for k, v in values.items():
        if isinstance(v, float) and np.isnan(v):
            continue
        acc.setdefault(k, []).append(float(v))


def finalize_mean_dict(acc: Dict[str, List[float]], prefix: str = "") -> Dict[str, float]:
    return {prefix + k: float(np.mean(v)) for k, v in acc.items() if len(v) > 0}


def surface(mask: np.ndarray) -> np.ndarray:
    if not mask.any():
        return mask.astype(bool)
    er = ndi.binary_erosion(mask, iterations=1, border_value=0)
    return np.logical_and(mask, np.logical_not(er))


def _bbox_slices(mask: np.ndarray, margin: int = 2) -> tuple:
    idx = np.argwhere(mask)
    lo = np.maximum(idx.min(axis=0) - margin, 0)
    hi = np.minimum(idx.max(axis=0) + margin + 1, mask.shape)
    return tuple(slice(int(l), int(h)) for l, h in zip(lo, hi))


def hd95_binary(pred: np.ndarray, gt: np.ndarray, spacing=(1.0, 1.0, 1.0)) -> float:
    pred = pred.astype(bool)
    gt = gt.astype(bool)
    if not pred.any() and not gt.any():
        return float("nan")
    if pred.any() != gt.any():
        return float("inf")
    box = _bbox_slices(pred | gt)
    pred, gt = pred[box], gt[box]
    sp = surface(pred)
    sg = surface(gt)
    if not sp.any() or not sg.any():
        return float("inf")
    dt_g = ndi.distance_transform_edt(~sg, sampling=spacing)
    dt_p = ndi.distance_transform_edt(~sp, sampling=spacing)
    dists = np.concatenate([dt_g[sp], dt_p[sg]])
    return float(np.percentile(dists, 95))
