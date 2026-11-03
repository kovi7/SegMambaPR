"""Mean predicted tumour probability in distance rings around the ground-truth arteries.

The rings come from the ground-truth vessels, so they are the same voxels for every model;
this is an evaluation measurement. A control ring at 10-15 mm separates probability placed
at the interface from probability spread everywhere.

    # 1. inference that keeps the tumour channel
    bash scripts/infer.sh configs/panorama/X.yaml best test output/panorama/X/inference_probs \
        --save_prob_classes 1

    # 2. the measurement
    python tools/band_mass.py --config configs/panorama/X.yaml --split test \
        --prob_dir output/panorama/X/inference_probs/probs \
        --dpcg_csv /path/to/panorama/dpcg_labels.csv \
        --out_csv output/panorama/X/inference_probs/band_mass.csv

    # 3. two runs, paired over cases
    python tools/band_mass.py --compare a/band_mass.csv b/band_mass.csv
"""
from __future__ import annotations

import argparse
import csv
import os
import random
import statistics as st
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import nibabel as nib
import numpy as np
from scipy import ndimage as ndi

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.config import load_config
from src.data import prepare_rows
from src.evaluate import load_label

RINGS = [(0.0, 1.0), (1.0, 2.0), (2.0, 3.0), (3.0, 5.0)]
FAR = (10.0, 15.0)
HIGH = {"borderline_resectable", "locally_advanced"}


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", help="Run config, for the manifest, the label remap and the class indices.")
    p.add_argument("--split", default="test")
    p.add_argument("--prob_dir", help="Directory of <case>_prob.nii.gz written by --save_prob_classes.")
    p.add_argument("--out_csv", help="Per-case output.")
    p.add_argument("--dpcg_csv", default=None, help="Ground-truth resectability stages, for the group summary.")
    p.add_argument("--band_mm", type=float, default=5.0, help="Outer edge of the near band. Default 5 mm.")
    p.add_argument("--roi_mm", type=float, default=30.0,
                   help="Both bands are restricted to this distance from the ground-truth pancreas, so the "
                        "near band and the control ring sit in the same neighbourhood. Default 30 mm.")
    p.add_argument("--vessel_classes", default="3", help="Ground-truth classes the band is built from. Default 3 (arteries).")
    p.add_argument("--pdac_class", type=int, default=1)
    p.add_argument("--pancreas_class", type=int, default=4)
    p.add_argument("--spacing", nargs=3, type=float, default=None,
                   help="Overrides the voxel size. The default reads it from the header of each "
                        "saved probability map, which inference writes in the resampled space of "
                        "the labels (the case is skipped if the two arrays differ in shape).")
    p.add_argument("--limit", type=int, default=0, help="Stop after N cases, for a smoke test.")
    p.add_argument("--compare", nargs=2, metavar=("A_CSV", "B_CSV"), default=None,
                   help="Paired comparison of two finished runs instead of a new measurement.")
    p.add_argument("--metric", default="mean_prob_band", help="Column compared by --compare.")
    p.add_argument("--draws", type=int, default=4000, help="Bootstrap resamples for --compare.")
    return p.parse_args()


CATEGORY_TO_STAGE = {
    "No PDAC": "no_pdac",
    "Resectable-like": "resectable",
    "Borderline/non-resectable risk": "borderline_resectable",
    "Locally advanced risk": "locally_advanced",
}


def stage_map(dpcg_csv: Optional[str]) -> Dict[str, str]:
    """case_id -> reference resectability category, read with the loader the rule uses."""
    if not dpcg_csv or not os.path.exists(dpcg_csv):
        return {}
    from src.manifest import load_dpcg_rows
    from src.geometry.dpcg_rules import dpcg_truth_category
    out = {}
    for cid, row in load_dpcg_rows(dpcg_csv).items():
        cat = dpcg_truth_category(row)
        if cat in CATEGORY_TO_STAGE:
            out[cid] = CATEGORY_TO_STAGE[cat]
    return out


def stages_from_run(band_csv: str) -> Dict[str, str]:
    """Fallback for an empty category column: read it from a run's resectability.csv."""
    run_dir = os.path.dirname(os.path.dirname(os.path.abspath(band_csv)))
    for sub in sorted(os.listdir(run_dir)) if os.path.isdir(run_dir) else []:
        p = os.path.join(run_dir, sub, "resectability.csv")
        if sub.startswith("inference_test") and os.path.exists(p):
            with open(p, newline="", encoding="utf-8") as f:
                return {r["case_id"]: CATEGORY_TO_STAGE.get(r.get("true_category", ""), "")
                        for r in csv.DictReader(f)}
    return {}


def case_rows(cfg, split) -> List[Dict]:
    rows, _ = prepare_rows(cfg, split, None, force_missing=True)
    return rows


def measure(prob: np.ndarray, gt: np.ndarray, spacing, vessel_classes, pdac_class, pancreas_class,
            band_mm: float, roi_mm: float) -> Dict[str, float]:
    """Mean predicted tumour probability per distance ring around the ground-truth vessels."""
    vessels = np.isin(gt, vessel_classes)
    out = {"n_vessel_voxels": int(vessels.sum())}
    if not vessels.any():
        return out
    dist = ndi.distance_transform_edt(~vessels, sampling=spacing)
    pancreas = np.isin(gt, [pancreas_class, pdac_class])
    if pancreas.any():
        roi = ndi.distance_transform_edt(~pancreas, sampling=spacing) <= roi_mm
    else:
        roi = np.ones_like(vessels, dtype=bool)

    voxel_ml = float(np.prod(spacing)) / 1000.0
    band = (dist > 0) & (dist <= band_mm) & roi
    out["n_band_voxels"] = int(band.sum())
    out["mean_prob_band"] = float(prob[band].mean()) if band.any() else float("nan")
    out["mass_ml_band"] = float(prob[band].sum() * voxel_ml) if band.any() else float("nan")
    for lo, hi in RINGS:
        ring = (dist > lo) & (dist <= hi) & roi
        out[f"mean_prob_{lo:g}_{hi:g}"] = float(prob[ring].mean()) if ring.any() else float("nan")
    far = (dist > FAR[0]) & (dist <= FAR[1]) & roi
    out["mean_prob_far"] = float(prob[far].mean()) if far.any() else float("nan")
    near, farv = out["mean_prob_band"], out["mean_prob_far"]
    # the ratio is defined only when the control ring holds some probability
    out["diff_near_far"] = float(near - farv) if near == near and farv == farv else float("nan")
    out["ratio_near_far"] = float(near / farv) if farv == farv and farv >= 1e-5 and near == near else float("nan")
    tumour = gt == pdac_class
    out["mean_prob_in_gt_tumour"] = float(prob[tumour].mean()) if tumour.any() else float("nan")
    out["gt_tumour_ml"] = float(tumour.sum() * voxel_ml)
    return out


def run_measurement(args):
    cfg = load_config(args.config)
    remap = (cfg.get("data") or {}).get("label_remap")
    vessel_classes = [int(c) for c in str(args.vessel_classes).split(",") if c.strip() != ""]
    stages = stage_map(args.dpcg_csv)
    rows = case_rows(cfg, args.split)
    prob_dir = Path(args.prob_dir)

    out_rows, missing = [], []
    for i, r in enumerate(rows):
        if args.limit and i >= args.limit:
            break
        cid = str(r["case_id"])
        pp = prob_dir / f"{cid}_prob.nii.gz"
        if not pp.exists():
            missing.append(cid)
            continue
        img = nib.load(str(pp))
        prob = np.asarray(img.dataobj).astype(np.float32)
        if prob.ndim == 4:                       # several channels saved, tumour first
            prob = prob[0]
        gt = load_label(r["label"], remap)
        if prob.shape != gt.shape:
            missing.append(f"{cid} (shape {prob.shape} vs {gt.shape})")
            continue
        spacing = tuple(args.spacing) if args.spacing else tuple(float(z) for z in img.header.get_zooms()[:3])
        m = measure(prob, gt, spacing, vessel_classes, args.pdac_class, args.pancreas_class,
                    args.band_mm, args.roi_mm)
        m.update(case_id=cid, stage=stages.get(cid, ""), spacing="x".join(f"{s:g}" for s in spacing))
        out_rows.append(m)
        if (i + 1) % 25 == 0:
            print(f"  {i + 1}/{len(rows)}", flush=True)

    if not out_rows:
        raise SystemExit("Nie zmierzono ani jednego przypadku - sprawdz --prob_dir i --split.")
    fields = ["case_id", "stage", "spacing", "n_vessel_voxels", "n_band_voxels", "mean_prob_band",
              "mass_ml_band"] + [f"mean_prob_{lo:g}_{hi:g}" for lo, hi in RINGS] + \
             ["mean_prob_far", "diff_near_far", "ratio_near_far", "mean_prob_in_gt_tumour", "gt_tumour_ml"]
    os.makedirs(os.path.dirname(os.path.abspath(args.out_csv)) or ".", exist_ok=True)
    with open(args.out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(out_rows)
    print(f"\nsaved {args.out_csv}  ({len(out_rows)} cases"
          + (f", skipped {len(missing)}" if missing else "") + ")")
    if missing:
        print("  no probability map for:", ", ".join(missing[:8]), "..." if len(missing) > 8 else "")
    summarise(out_rows)


def summarise(rows: Sequence[Dict]):
    groups = {"high risk": lambda r: r.get("stage") in HIGH,
              "resectable": lambda r: r.get("stage") == "resectable",
              "non-PDAC": lambda r: r.get("stage") == "no_pdac"}
    print(f"\n{'group':16s} {'n':>4s} {'median band':>14s} {'median far':>15s} {'median diff':>16s}")
    for name, pick in groups.items():
        sel = [r for r in rows if pick(r)]
        if not sel:
            continue
        def med(key):
            v = [r[key] for r in sel if isinstance(r.get(key), float) and r[key] == r[key]]
            return st.median(v) if v else float("nan")
        print(f"{name:16s} {len(sel):4d} {med('mean_prob_band'):14.5f} {med('mean_prob_far'):15.5f} {med('diff_near_far'):16.5f}")
    print("\nradial profile (median of the mean tumour probability):")
    for lo, hi in RINGS:
        key = f"mean_prob_{lo:g}_{hi:g}"
        line = []
        for name, pick in groups.items():
            sel = [r[key] for r in rows if pick(r) and isinstance(r.get(key), float) and r[key] == r[key]]
            line.append(f"{name} {st.median(sel):.5f}" if sel else f"{name} -")
        print(f"  {lo:g}-{hi:g} mm   " + "   ".join(line))


def compare(args):
    def read(path):
        with open(path, newline="", encoding="utf-8") as f:
            return {r["case_id"]: r for r in csv.DictReader(f)}
    A, B = read(args.compare[0]), read(args.compare[1])
    if not any(r.get("stage") for r in A.values()):
        fallback = stages_from_run(args.compare[0]) or stages_from_run(args.compare[1])
        for cid, r in A.items():
            r["stage"] = fallback.get(cid, "")
        if fallback:
            print("(categories taken from resectability.csv, the stage column of band_mass.csv is empty)")
    ids = sorted(set(A) & set(B))
    pairs = []
    for cid in ids:
        try:
            a, b = float(A[cid][args.metric]), float(B[cid][args.metric])
        except (KeyError, ValueError):
            continue
        if a == a and b == b:
            pairs.append((cid, a, b, A[cid].get("stage", "")))
    if not pairs:
        raise SystemExit("No common cases with this metric.")
    rng = random.Random(0)

    def report(label, sel):
        d = [a - b for _, a, b, _ in sel]
        if not d:
            return
        means = sorted(st.fmean(rng.choices(d, k=len(d))) for _ in range(args.draws))
        lo, hi = means[int(0.025 * args.draws)], means[int(0.975 * args.draws)]
        wins = sum(1 for x in d if x > 0)
        print(f"{label:18s} n={len(d):3d}  mean difference {st.fmean(d):+.5f}  "
              f"[{lo:+.5f}, {hi:+.5f}]  median {st.median(d):+.5f}  larger {wins}/{len(d)}")

    print(f"metric: {args.metric}\n{os.path.basename(os.path.dirname(args.compare[0]))} minus "
          f"{os.path.basename(os.path.dirname(args.compare[1]))}\n")
    report("all", pairs)
    report("high risk", [p for p in pairs if p[3] in HIGH])
    report("resectable", [p for p in pairs if p[3] == "resectable"])
    report("non-PDAC", [p for p in pairs if p[3] == "no_pdac"])


def main():
    args = parse_args()
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    if args.compare:
        compare(args)
        return
    for need in ("config", "prob_dir", "out_csv"):
        if not getattr(args, need):
            raise SystemExit(f"--{need} jest wymagane przy pomiarze")
    run_measurement(args)


if __name__ == "__main__":
    main()
