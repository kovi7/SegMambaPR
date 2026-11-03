from __future__ import annotations

import argparse
import csv
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import nibabel as nib
import numpy as np

from .config import load_config
from .metrics import binary_metrics, hd95_binary
from .plot_metrics import label_names_from_cfg, plot_evaluation
from .utils import load_json, save_json
from .data import prepare_rows, TabularEncoderPreprocess


def dice(pred: np.ndarray, gt: np.ndarray, c: int) -> float:
    p = pred == c
    y = gt == c
    denom = p.sum() + y.sum()
    if denom == 0:
        return float("nan")
    return float(2 * np.logical_and(p, y).sum() / denom)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--manifest", default=None, help="Optional manifest JSON. If omitted, rows are loaded from --config data sources.")
    p.add_argument("--pred_dir", required=True)
    p.add_argument("--split", default="test")
    p.add_argument("--out_csv", default="evaluation.csv")
    p.add_argument("--config", default=None, help="Optional config. If given, rows are loaded from the config manifest/data sources, and num_classes/pdac_class/pdac_threshold/spacing default to the config's values.")
    p.add_argument("--num_classes", type=int, default=None, help="Defaults to config num_classes, else 6.")
    p.add_argument("--pdac_class", type=int, default=None, help="Defaults to config labels.pdac, else 1.")
    p.add_argument("--pdac_threshold", type=float, default=None, help="Defaults to config val.pdac_threshold, else 0.5.")
    p.add_argument("--spacing", nargs=3, type=float, default=None, help="Defaults to config infer.spacing/data.spacing, else 0.8 0.8 3.0.")
    p.add_argument("--dice_review_threshold", type=float, default=0.5, help="Cases with dice_mean_fg below this are flagged for review.")
    p.add_argument("--hd95_review_threshold", type=float, default=20.0, help="Cases with hd95_mean_fg (mm) above this are flagged for review.")
    p.add_argument("--jobs", type=int, default=1, help="Parallel worker processes for per-case dice/HD95 (CPU-only, no GPU needed). 1 = sequential.")
    return p.parse_args()


def load_label(path: str, label_remap: Optional[Dict[str, Any]] = None) -> np.ndarray:
    # masks on disk keep the stored label ids, so the training label remap is applied here too
    arr = nib.load(path).get_fdata().astype(np.int16)
    if label_remap:
        orig_labels = label_remap.get("orig_labels", label_remap.get("from", []))
        target_labels = label_remap.get("target_labels", label_remap.get("to", []))
        remapped = arr.copy()
        for orig, target in zip(orig_labels, target_labels):
            remapped[arr == int(orig)] = int(target)
        arr = remapped
    return arr


def evaluate_case(
    r: Dict[str, Any],
    pred_dir: str,
    num_classes: int,
    pdac_class: int,
    spacing: Sequence[float],
    label_remap: Optional[Dict[str, Any]] = None,
) -> Optional[Dict[str, Any]]:
    """Per-case Dice, HD95 and PDAC presence (module-level so that --jobs can use processes)."""
    case_id = str(r["case_id"])
    pred_path = Path(pred_dir) / f"{case_id}.nii.gz"
    if not pred_path.exists():
        return None
    pred = nib.load(str(pred_path)).get_fdata().astype(np.int16)
    gt = load_label(r["label"], label_remap)
    if pred.shape != gt.shape:
        raise ValueError(f"Shape mismatch for {case_id}: pred {pred.shape}, gt {gt.shape}. Use --config matching inference orientation or fix preprocessing.")
    row: Dict[str, Any] = {"case_id": case_id}
    dice_values = []
    hd_values = []
    for c in range(1, num_classes):
        d = dice(pred, gt, c)
        h = hd95_binary(pred == c, gt == c, spacing=tuple(spacing))
        row[f"dice_c{c}"] = d
        row[f"hd95_c{c}"] = h
        if not np.isnan(d):
            dice_values.append(d)
        if not np.isnan(h) and np.isfinite(h):
            hd_values.append(h)
    row["dice_mean_fg"] = float(np.mean(dice_values)) if dice_values else float("nan")
    row["hd95_mean_fg"] = float(np.mean(hd_values)) if hd_values else float("nan")
    row["pred_has_pdac"] = int((pred == pdac_class).any())
    row["true_has_pdac"] = int((gt == pdac_class).any())
    row["pdac_score"] = float((pred == pdac_class).mean())
    return row


def case_verdict(row: Dict[str, Any], dice_threshold: float, hd95_threshold: float) -> tuple[str, str]:
    """Flag a case for review with a short rule-based reason."""
    dice_val = row["dice_mean_fg"]
    hd_val = row["hd95_mean_fg"]
    if row["true_has_pdac"] == 1 and row["pred_has_pdac"] == 0:
        return "review", "PDAC missed (false negative): no tumor voxels predicted despite ground-truth PDAC"
    if row["true_has_pdac"] == 0 and row["pred_has_pdac"] == 1:
        return "review", "PDAC false positive: tumor predicted where ground truth has none"
    if np.isnan(dice_val):
        return "review", "No foreground overlap for any class (empty prediction or empty ground truth)"
    if dice_val < dice_threshold:
        return "review", f"Low Dice ({dice_val:.2f} < {dice_threshold}): segmentation likely inaccurate"
    if not np.isnan(hd_val) and hd_val > hd95_threshold:
        return "review", f"High HD95 ({hd_val:.1f}mm > {hd95_threshold}mm): fragmented/mislocalized prediction despite acceptable Dice"
    return "ok", "Good agreement with ground truth"


def main():
    args = parse_args()
    cfg = load_config(args.config) if args.config else None
    num_classes = args.num_classes if args.num_classes is not None else int((cfg or {}).get("num_classes", 6))
    pdac_class = args.pdac_class if args.pdac_class is not None else int((cfg or {}).get("labels", {}).get("pdac", 1))
    pdac_threshold = args.pdac_threshold if args.pdac_threshold is not None else float((cfg or {}).get("val", {}).get("pdac_threshold", 0.5))
    spacing = args.spacing if args.spacing is not None else (cfg or {}).get("infer", {}).get("spacing", (cfg or {}).get("data", {}).get("spacing", [0.8, 0.8, 3.0]))
    label_remap = (cfg or {}).get("data", {}).get("label_remap")

    if args.manifest:
        rows = [r for r in load_json(args.manifest) if r.get("split") == args.split]
    elif cfg is not None:
        rows, _ = prepare_rows(cfg, args.split, TabularEncoderPreprocess(scanner_min_count=cfg.get("tabular", {}).get("scanner_min_count", 2)).fit([]), force_missing=True)
    else:
        raise ValueError("Provide either --manifest or --config so evaluation can find labels.")
    pred_dir = str(Path(args.pred_dir))
    out_rows: List[Dict[str, Any]] = []
    if args.jobs > 1:
        with ProcessPoolExecutor(max_workers=args.jobs) as ex:
            futures = [ex.submit(evaluate_case, r, pred_dir, num_classes, pdac_class, spacing, label_remap) for r in rows]
            computed = [fut.result() for fut in futures]
    else:
        computed = [evaluate_case(r, pred_dir, num_classes, pdac_class, spacing, label_remap) for r in rows]
    y_true: List[int] = []
    y_score: List[float] = []
    y_pred: List[int] = []
    for row in computed:
        if row is None:
            continue
        row["status"], row["likely_reason"] = case_verdict(row, args.dice_review_threshold, args.hd95_review_threshold)
        y_true.append(row["true_has_pdac"])
        # AUROC/AP rank the PDAC voxel fraction; the decision is "any PDAC voxel predicted"
        y_score.append(row["pdac_score"])
        y_pred.append(row["pred_has_pdac"])
        out_rows.append(row)
    with open(args.out_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(out_rows[0].keys()) if out_rows else ["case_id"])
        writer.writeheader()
        writer.writerows(out_rows)
    print(f"Saved {args.out_csv} with {len(out_rows)} rows")
    if out_rows:
        pdac_metrics = binary_metrics(y_true, y_score, threshold=pdac_threshold, y_pred=y_pred)
        n_review = sum(1 for r in out_rows if r["status"] == "review")
        summary = {
            "n_cases": len(out_rows),
            "n_ok": len(out_rows) - n_review,
            "n_review": n_review,
            "dice_mean_fg_mean": float(np.nanmean([r["dice_mean_fg"] for r in out_rows])),
            "dice_mean_fg_std": float(np.nanstd([r["dice_mean_fg"] for r in out_rows])),
            "hd95_mean_fg_mean": float(np.nanmean([r["hd95_mean_fg"] for r in out_rows])),
            "pdac_detection": pdac_metrics,
            "review_cases": [
                {"case_id": r["case_id"], "reason": r["likely_reason"]}
                for r in out_rows if r["status"] == "review"
            ],
        }
        summary_path = Path(args.out_csv).with_name(f"{Path(args.out_csv).stem}_summary.json")
        save_json(summary, summary_path)
        print(f"Saved {summary_path}: {summary['n_ok']}/{summary['n_cases']} ok, {n_review} flagged for review")
        print("Mean Dice foreground:", summary["dice_mean_fg_mean"])
        print("PDAC detection from hard mask:", pdac_metrics)
        try:
            plot_paths = plot_evaluation(args.out_csv, label_names=label_names_from_cfg((cfg or {}).get("labels")))
            print("Saved evaluation plots:", plot_paths)
        except Exception as exc:
            print(f"Plotting failed: {exc}", flush=True)


if __name__ == "__main__":
    main()
