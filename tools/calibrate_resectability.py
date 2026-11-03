"""Fit the distance criteria of the resectability rule on the validation split.

The angle criteria keep their reference values. The tumour-to-vessel distance criteria are
added with a logical OR, so they can only add high-risk calls to the angle test.
--tune_angles also searches the angle thresholds.

    python -m src.geometry.dpcg_rules --pred_dir <val preds> \
        --out_csv val_resect.csv --dpcg_csv /path/to/panorama/dpcg_labels.csv
    python tools/calibrate_resectability.py --csv val_resect.csv \
        --out_json resectability_thresholds.json

    python -m src.geometry.dpcg_rules --pred_dir <test preds> \
        --out_csv test_resect.csv --dpcg_csv /path/to/panorama/dpcg_labels.csv \
        --thresholds_json resectability_thresholds.json
"""
from __future__ import annotations

import argparse
import csv
import itertools
import json
from typing import Dict, List, Sequence, Tuple

import numpy as np

HIGH_RISK = {"Borderline/non-resectable risk", "Locally advanced risk"}
FAR_MM = 1e6  # stand-in for "no measurable contact", so a <= test never fires


def _clinical_thresholds() -> Dict[str, float]:
    """Default thresholds of the rule, imported by path so the tool runs from the repository root."""
    import sys
    from pathlib import Path
    root = Path(__file__).resolve().parent.parent
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    from src.geometry.dpcg_rules import CLINICAL_THRESHOLDS
    return CLINICAL_THRESHOLDS


def load_rows(path: str) -> List[Dict[str, str]]:
    with open(path, newline="", encoding="utf-8") as f:
        rows = [r for r in csv.DictReader(f) if r.get("true_category")]
    if not rows:
        raise SystemExit(f"{path} has no rows with a true_category - run dpcg_rules.py with --dpcg_csv first.")
    return rows


def num(row: Dict[str, str], key: str, missing: float) -> float:
    try:
        v = float(row[key])
    except (KeyError, TypeError, ValueError):
        return missing
    return v if np.isfinite(v) else missing


def _arrays(rows: Sequence[Dict[str, str]]):
    """Labels and features of the PDAC-positive cases, the only ones scored."""
    considered = [r for r in rows if r["true_category"] != "No PDAC"]
    if not considered:
        raise SystemExit("No PDAC-positive cases with ground-truth stage - cannot calibrate.")
    return (
        np.array([r["true_category"] in HIGH_RISK for r in considered]),
        np.array([num(r, "artery_min_dist_mm", FAR_MM) for r in considered]),
        np.array([num(r, "vein_min_dist_mm", FAR_MM) for r in considered]),
        np.array([num(r, "artery_contact_deg", 0.0) for r in considered]),
        np.array([num(r, "vein_contact_deg", 0.0) for r in considered]),
    )


def _predict(th: Dict[str, float], art_d, vein_d, art_a, vein_a) -> np.ndarray:
    """Disjunction of the enabled criteria, as in the borderline test of dpcg_like_category."""
    pred = np.zeros(len(art_d), dtype=bool)
    if th.get("artery_dist_borderline", -1) >= 0: pred |= art_d <= th["artery_dist_borderline"]
    if th.get("vein_dist_borderline", -1) >= 0: pred |= vein_d <= th["vein_dist_borderline"]
    if th.get("artery_deg_borderline", -1) >= 0: pred |= art_a > th["artery_deg_borderline"]
    if th.get("vein_deg_borderline", -1) >= 0: pred |= vein_a > th["vein_deg_borderline"]
    return pred


def binary_scores(y_true: np.ndarray, y_pred: np.ndarray) -> Tuple[float, float, int, int]:
    tp = int((y_pred & y_true).sum())
    fp = int((y_pred & ~y_true).sum())
    fn = int((~y_pred & y_true).sum())
    tn = int((~y_pred & ~y_true).sum())
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    spec = tn / (tn + fp) if tn + fp else 0.0
    return f1, 0.5 * (rec + spec), tp, fp


def calibrate(rows: Sequence[Dict[str, str]], grid_art_dist: Sequence[float], grid_vein_dist: Sequence[float],
              grid_art_deg: Sequence[float], grid_vein_deg: Sequence[float], beta: float) -> Dict:
    """Grid-search the borderline thresholds that maximise F-beta for high risk vs resectable.

    Only PDAC-positive cases are scored. beta > 1 weights recall above precision.
    """
    y, art_d, vein_d, art_a, vein_a = _arrays(rows)

    def fbeta(y_true, y_pred, b):
        tp = int((y_pred & y_true).sum()); fp = int((y_pred & ~y_true).sum()); fn = int((~y_pred & y_true).sum())
        prec = tp / (tp + fp) if tp + fp else 0.0
        rec = tp / (tp + fn) if tp + fn else 0.0
        if prec + rec == 0:
            return 0.0
        b2 = b * b
        return (1 + b2) * prec * rec / (b2 * prec + rec)

    best = None
    for ad, vd, aa, va in itertools.product(grid_art_dist, grid_vein_dist, grid_art_deg, grid_vein_deg):
        pred = np.zeros(len(y), dtype=bool)
        if ad >= 0:
            pred |= art_d <= ad
        if vd >= 0:
            pred |= vein_d <= vd
        if aa >= 0:
            pred |= art_a > aa
        if va >= 0:
            pred |= vein_a > va
        score = fbeta(y, pred, beta)
        if best is None or score > best[0]:
            best = (score, dict(artery_dist_borderline=ad, vein_dist_borderline=vd,
                                artery_deg_borderline=aa, vein_deg_borderline=va))
    score, th = best
    f1, bal, tp, fp = binary_scores(y, _predict(th, art_d, vein_d, art_a, vein_a))
    return {
        "thresholds": th,
        "n_pdac_positive": int(len(y)),
        "n_high_risk": int(y.sum()),
        "fbeta": float(score),
        "beta": beta,
        "f1": float(f1),
        "balanced_accuracy": float(bal),
        "detected_high_risk": tp,
        "false_alarms": fp,
    }


def feature_report(rows: Sequence[Dict[str, str]]) -> None:
    """Rank single features by how well each separates high-risk from resectable cases."""
    considered = [r for r in rows if r["true_category"] != "No PDAC"
                  and r.get("dpcg_like_category") != "No PDAC"]
    if len(considered) < 4:
        return
    y = np.array([r["true_category"] in HIGH_RISK for r in considered])
    if y.sum() == 0 or y.sum() == len(y):
        return

    print(f"\nSingle-feature separation, {int(y.sum())} high-risk vs {int((~y).sum())} resectable:")
    print(f"  {'feature':<24}{'AUC':>7}   direction")
    for key, missing in (("artery_min_dist_mm", FAR_MM), ("vein_min_dist_mm", FAR_MM),
                         ("artery_contact_deg", 0.0), ("vein_contact_deg", 0.0),
                         ("pdac_voxels", 0.0)):
        if key not in considered[0]:
            continue
        x = np.array([num(r, key, missing) for r in considered])
        a, b = x[y], x[~y]
        wins = (a[:, None] > b[None, :]).sum() + 0.5 * (a[:, None] == b[None, :]).sum()
        auc = float(wins / (len(a) * len(b)))
        direction = "higher = risk" if auc >= 0.5 else "LOWER = risk"
        print(f"  {key:<24}{max(auc, 1 - auc):>7.3f}   {direction}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--csv", required=True, help="resectability.csv of the validation split, with reference categories.")
    ap.add_argument("--out_json", default="configs/resectability_thresholds.json")
    ap.add_argument("--beta", type=float, default=2.0, help="F-beta weight on recall (>1 favours recall).")
    ap.add_argument("--tune_angles", action="store_true", help="Also search the angle thresholds (by default they keep their reference values).")
    args = ap.parse_args()

    rows = load_rows(args.csv)
    feature_report(rows)
    clinical = _clinical_thresholds()
    res = calibrate(
        rows,
        grid_art_dist=[-1, 1, 2, 3, 4, 5, 6, 8, 10],
        grid_vein_dist=[-1, 1, 2, 3, 5, 8],
        grid_art_deg=[-1, 5, 10, 20, 45, 90] if args.tune_angles else [clinical["artery_deg_borderline"]],
        grid_vein_deg=[-1, 15, 30, 60, 120, 180] if args.tune_angles else [clinical["vein_deg_borderline"]],
        beta=args.beta,
    )

    # the locally advanced thresholds stay at their reference values (too few cases to fit)
    full = dict(_clinical_thresholds())
    full.update(res["thresholds"])
    res["thresholds"] = full

    y, art_d, vein_d, art_a, vein_a = _arrays(rows)
    base = {k: clinical[k] for k in ("artery_deg_borderline", "vein_deg_borderline")}
    b_f1, b_bal, b_tp, b_fp = binary_scores(y, _predict(base, art_d, vein_d, art_a, vein_a))

    n_hi = int(y.sum())
    base_hits = f"{b_tp}/{n_hi}"
    cal_hits = f"{res['detected_high_risk']}/{n_hi}"
    label = "calibrated (angles tuned)" if args.tune_angles else "angle + distance"

    print(f"\nCalibrated on {res['n_pdac_positive']} PDAC-positive cases ({n_hi} high-risk)")
    print(f"  {'':<26}{'detected':>10}{'false':>8}{'F1':>8}{'bal.acc':>9}")
    print(f"  {'angle only (reference)':<26}{base_hits:>10}{b_fp:>8}{b_f1:>8.3f}{b_bal:>9.3f}")
    print(f"  {label:<26}{cal_hits:>10}{res['false_alarms']:>8}{res['f1']:>8.3f}{res['balanced_accuracy']:>9.3f}")
    if not args.tune_angles:
        print(f"  the distance criterion adds {res['detected_high_risk'] - b_tp} detection(s) "
              f"at {res['false_alarms'] - b_fp} extra false alarm(s); it can only add, never remove")
    print("  thresholds:")
    for k, v in full.items():
        state = "off" if v < 0 else f"{v:g}"
        print(f"    {k:26s} {state}")

    with open(args.out_json, "w", encoding="utf-8") as f:
        json.dump(res, f, indent=2)
    print(f"\nWrote {args.out_json}")
    print("Apply with: python -m src.geometry.dpcg_rules ... --thresholds_json " + args.out_json)


if __name__ == "__main__":
    main()
