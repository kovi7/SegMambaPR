from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import nibabel as nib
import numpy as np
from scipy import ndimage as ndi

CATEGORIES = ["No PDAC", "Resectable-like", "Borderline/non-resectable risk", "Locally advanced risk"]


def surface(mask: np.ndarray) -> np.ndarray:
    er = ndi.binary_erosion(mask, iterations=1, border_value=0)
    return np.logical_and(mask, np.logical_not(er))


def min_surface_distance_mm(a: np.ndarray, b: np.ndarray, spacing=(0.8, 0.8, 3.0)) -> float:
    if not a.any() or not b.any():
        return float("nan")
    sa = surface(a)
    sb = surface(b)
    dt_b = ndi.distance_transform_edt(~sb, sampling=spacing)
    return float(dt_b[sa].min()) if sa.any() else float("nan")


def drop_largest_component(mask: np.ndarray) -> np.ndarray:
    """Remove the biggest connected component of an artery mask - the aorta."""
    lab, n = ndi.label(mask, structure=ndi.generate_binary_structure(3, 1))
    if n <= 1:
        return mask.astype(bool)
    sizes = np.bincount(lab.ravel(), minlength=n + 1)
    sizes[0] = 0
    return np.logical_and(mask, lab != int(sizes.argmax()))


def _tolerance_steps(tolerance_mm: float, spacing: Sequence[float]) -> Tuple[int, int]:
    """In-plane and through-plane dilation steps covering ``tolerance_mm``."""
    if tolerance_mm <= 0:
        return 0, 0
    sxy = float(min(spacing[0], spacing[1])) or 1.0
    sz = float(spacing[2]) or 1.0
    return int(round(tolerance_mm / sxy)), int(round(tolerance_mm / sz))


def axial_contact_degrees(
    pdac: np.ndarray,
    vessel: np.ndarray,
    drop_largest: bool = False,
    min_component_voxels: int = 0,
    tolerance_mm: float = 0.0,
    spacing: Sequence[float] = (0.8, 0.8, 3.0),
) -> float:
    """Axial contact angle: contact border / vessel border * 360, maximised over slices and components.

    This is the measurement behind the reference categories. ``min_component_voxels`` drops
    small vessel fragments and ``tolerance_mm`` counts vessel border within that distance of
    the tumour as contact. Both are for predicted masks only and stay 0 on ground truth.
    """
    if not pdac.any() or not vessel.any():
        return 0.0
    mask = drop_largest_component(vessel) if drop_largest else vessel.astype(bool)
    if not mask.any():
        return 0.0

    cross = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=bool)
    square = np.ones((3, 3), dtype=bool)

    pdac_t = pdac
    n_xy, n_z = _tolerance_steps(tolerance_mm, spacing)
    if n_xy or n_z:
        struct = np.zeros((3, 3, 3), dtype=bool)
        struct[1, 1, :] = True
        struct[:, 1, 1] = True
        struct[1, :, 1] = True
        if n_z:
            pdac_t = ndi.binary_dilation(pdac_t, structure=struct, iterations=n_z)
        if n_xy > n_z:
            flat = np.zeros((3, 3, 3), dtype=bool)
            flat[:, 1, 1] = True
            flat[1, :, 1] = True
            pdac_t = ndi.binary_dilation(pdac_t, structure=flat, iterations=n_xy - n_z)

    dilated = {int(z): ndi.binary_dilation(pdac_t[:, :, z], structure=cross)
               for z in np.flatnonzero(pdac_t.any(axis=(0, 1)))}
    if not dilated:
        return 0.0

    lab, n = ndi.label(mask, structure=ndi.generate_binary_structure(3, 1))
    sizes = np.bincount(lab.ravel(), minlength=n + 1)
    boxes = ndi.find_objects(lab)
    best = 0.0
    for comp in range(1, n + 1):
        if sizes[comp] < min_component_voxels:
            continue
        box = boxes[comp - 1]
        if box is None:
            continue
        sub = lab[box] == comp
        ys, xs, zs = box
        for z, p_dil in dilated.items():
            zi = z - zs.start
            if not 0 <= zi < sub.shape[2]:
                continue
            v2 = sub[:, :, zi]
            if not v2.any():
                continue
            border = np.logical_and(v2, ~ndi.binary_erosion(v2, structure=square, border_value=0))
            perimeter = int(border.sum())
            if perimeter == 0:
                continue
            contact = int(np.logical_and(border, p_dil[ys, xs]).sum())
            best = max(best, contact / perimeter * 360.0)
    return float(best)


def extract_geometry(
    mask: np.ndarray,
    labels: Dict[str, int],
    spacing=(0.8, 0.8, 3.0),
    min_component_voxels: int = 0,
    contact_tolerance_mm: float = 0.0,
) -> Dict[str, float]:
    pdac = mask == labels.get("pdac", 1)
    veins = mask == labels.get("veins", 2)
    arteries = mask == labels.get("arteries", 3)
    kw = dict(min_component_voxels=min_component_voxels, tolerance_mm=contact_tolerance_mm, spacing=spacing)
    out = {
        "pdac_voxels": int(pdac.sum()),
        "vein_contact_deg": axial_contact_degrees(pdac, veins, **kw),
        "artery_contact_deg": axial_contact_degrees(pdac, arteries, drop_largest=True, **kw),
        "vein_min_dist_mm": min_surface_distance_mm(pdac, veins, spacing),
        "artery_min_dist_mm": min_surface_distance_mm(pdac, arteries, spacing),
    }
    return out


# angle tests are strict ">", so 0.0 means any measurable contact
CLINICAL_THRESHOLDS: Dict[str, float] = {
    "artery_deg_advanced": 90.0,
    "vein_deg_advanced": 270.0,
    "artery_deg_borderline": 0.0,
    "vein_deg_borderline": 90.0,
    # distance criteria in mm, -1 = off; fitted by tools/calibrate_resectability.py
    "artery_dist_advanced": -1.0,
    "artery_dist_borderline": -1.0,
    "vein_dist_borderline": -1.0,
}


def dpcg_like_category(
    feat: Dict[str, float],
    min_pdac_voxels: int = 20,
    thresholds: Optional[Dict[str, float]] = None,
) -> str:
    """Map tumour-vessel geometry to a resectability category (exploratory, not a clinical tool).

    An angle criterion fires above its threshold, a distance criterion at or below it.
    """
    t = dict(CLINICAL_THRESHOLDS)
    if thresholds:
        t.update({k: float(v) for k, v in thresholds.items() if k in t})

    if feat["pdac_voxels"] < min_pdac_voxels:
        return "No PDAC"

    art_deg = feat["artery_contact_deg"]
    vein_deg = feat["vein_contact_deg"]

    def dist(key: str) -> float:
        # an unmeasurable distance counts as far away
        v = feat.get(key, float("nan"))
        return float("inf") if v is None or not np.isfinite(v) else float(v)

    art_dist = dist("artery_min_dist_mm")
    vein_dist = dist("vein_min_dist_mm")

    # a negative threshold disables its criterion
    def deg_hit(value: float, key: str) -> bool:
        return t[key] >= 0 and value > t[key]

    def dist_hit(value: float, key: str) -> bool:
        return t[key] >= 0 and value <= t[key]

    if (
        deg_hit(art_deg, "artery_deg_advanced")
        or deg_hit(vein_deg, "vein_deg_advanced")
        or dist_hit(art_dist, "artery_dist_advanced")
    ):
        return "Locally advanced risk"
    if (
        deg_hit(art_deg, "artery_deg_borderline")
        or deg_hit(vein_deg, "vein_deg_borderline")
        or dist_hit(art_dist, "artery_dist_borderline")
        or dist_hit(vein_dist, "vein_dist_borderline")
    ):
        return "Borderline/non-resectable risk"
    return "Resectable-like"


def dpcg_truth_category(dpcg_row: Dict[str, Any]) -> Optional[str]:
    """Reference category of a row of the DPCG csv, or None when no category is set."""
    if not dpcg_row.get("has_pdac"):
        return "No PDAC"
    if dpcg_row.get("is_locally_advanced"):
        return "Locally advanced risk"
    if dpcg_row.get("is_borderline"):
        return "Borderline/non-resectable risk"
    if dpcg_row.get("is_resectable"):
        return "Resectable-like"
    return None


def plot_resectability_confusion(pairs: list[Tuple[str, str]], out_png: str) -> None:
    """Confusion matrix of (true_category, pred_category) pairs."""
    import matplotlib.pyplot as plt

    n = len(CATEGORIES)
    idx = {c: i for i, c in enumerate(CATEGORIES)}
    conf = np.zeros((n, n), dtype=int)
    for true_cat, pred_cat in pairs:
        conf[idx[true_cat], idx[pred_cat]] += 1

    short = ["No PDAC", "Resectable", "Borderline", "Loc. advanced"]
    fig, ax = plt.subplots(figsize=(6.5, 6))
    im = ax.imshow(conf, cmap="Blues")
    for i in range(n):
        for j in range(n):
            ax.text(j, i, str(conf[i, j]), ha="center", va="center")
    ax.set(
        xticks=range(n), yticks=range(n),
        xticklabels=short, yticklabels=short,
        xlabel="Predicted (from segmentation, dpcg_rules.py)", ylabel="True (dpcg_labels.csv)",
        title="Resectability: predicted vs. true",
    )
    plt.setp(ax.get_xticklabels(), rotation=30, ha="right")
    fig.colorbar(im, ax=ax, fraction=0.046)
    fig.tight_layout()
    fig.savefig(out_png, dpi=160)
    plt.close(fig)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--pred_dir", default=None, help="Directory of predicted masks. Not needed with --use_ground_truth.")
    p.add_argument("--out_csv", required=True)
    p.add_argument("--pdac", type=int, default=1)
    p.add_argument("--veins", type=int, default=2)
    p.add_argument("--arteries", type=int, default=3)
    p.add_argument("--dpcg_csv", default=None, help="If given, evaluate predicted categories against ground truth from this DPCG csv and save a confusion matrix + accuracy.")
    p.add_argument("--out_confusion_png", default=None, help="Confusion matrix path (default: alongside --out_csv, resectability_confusion_matrix.png).")
    p.add_argument("--out_stats_json", default=None, help="Per-class precision/recall/F1 + macro averages path (default: alongside --out_csv, resectability_stats.json).")
    p.add_argument("--thresholds_json", default=None, help="Calibrated thresholds from tools/calibrate_resectability.py. Without this, the published clinical thresholds are used.")
    p.add_argument("--use_ground_truth", action="store_true", help="Run the rule on ground-truth masks instead of predictions. Requires --config and --split.")
    p.add_argument("--config", default=None, help="Config, needed by --use_ground_truth to locate ground-truth labels and the label remapping.")
    p.add_argument("--split", default="test", help="Split to read ground-truth labels for, with --use_ground_truth.")
    p.add_argument("--min_vessel_voxels", type=int, default=0, help="Ignore vessel components smaller than this (predicted masks only).")
    p.add_argument("--min_pdac_voxels", type=int, default=20, help="Below this many tumour voxels a case is called No PDAC.")
    p.add_argument("--contact_tolerance_mm", type=float, default=0.0, help="Count vessel border within this distance of the tumour as contact (predicted masks only).")
    return p.parse_args()


def _zooms(path: str) -> Tuple[float, float, float]:
    """Voxel size in mm from the NIfTI header."""
    try:
        z = nib.load(str(path)).header.get_zooms()[:3]
        if len(z) == 3 and all(np.isfinite(v) and v > 0 for v in z):
            return (float(z[0]), float(z[1]), float(z[2]))
    except Exception:
        pass
    return (0.8, 0.8, 3.0)


def _ground_truth_masks(config_path: str, split: str):
    """Yield (case_id, ground-truth mask with the training label remap, spacing) for one split."""
    from ..config import load_config
    from ..data import prepare_rows, build_tabproc
    from ..evaluate import load_label

    cfg = load_config(config_path)
    rows, _ = prepare_rows(cfg, split, build_tabproc(cfg, []), force_missing=True)
    label_remap = cfg.get("data", {}).get("label_remap")
    for r in rows:
        if not r.get("label"):
            continue
        yield str(r["case_id"]), load_label(r["label"], label_remap), _zooms(r["label"])


def main():
    args = parse_args()
    labels = {"pdac": args.pdac, "veins": args.veins, "arteries": args.arteries}

    thresholds = None
    if args.thresholds_json:
        import json as _json
        payload = _json.load(open(args.thresholds_json, encoding="utf-8"))
        thresholds = payload.get("thresholds", payload)
        print(f"Using calibrated thresholds from {args.thresholds_json}: {thresholds}")

    if args.use_ground_truth:
        if not args.config:
            raise SystemExit("--use_ground_truth requires --config")
        if args.contact_tolerance_mm > 0:
            raise SystemExit(
                "--contact_tolerance_mm must be 0 with --use_ground_truth: on ground-truth masks "
                "it would only inflate the angles."
            )
        print(f"Diagnostic mode: running the rule on GROUND-TRUTH masks ({args.split} split), not predictions.")
        source = _ground_truth_masks(args.config, args.split)
    else:
        if not args.pred_dir:
            raise SystemExit("--pred_dir is required unless --use_ground_truth is given")
        pred_dir = Path(args.pred_dir)
        source = ((p.name.replace(".nii.gz", ""), nib.load(str(p)).get_fdata().astype(np.int16), _zooms(p))
                  for p in sorted(pred_dir.glob("*.nii.gz")))

    rows = []
    for case_id, mask, spacing in source:
        feat = extract_geometry(mask, labels, spacing=spacing, min_component_voxels=args.min_vessel_voxels,
                                contact_tolerance_mm=args.contact_tolerance_mm)
        feat["case_id"] = case_id
        feat["dpcg_like_category"] = dpcg_like_category(feat, min_pdac_voxels=args.min_pdac_voxels, thresholds=thresholds)
        rows.append(feat)

    if args.dpcg_csv:
        import json

        from ..manifest import load_dpcg_rows, normalize_case_id
        from ..metrics import multiclass_metrics

        truth = load_dpcg_rows(args.dpcg_csv)
        pairs = []
        n_no_truth = 0
        for feat in rows:
            cid = normalize_case_id(feat["case_id"])
            dpcg_row = truth.get(cid)
            true_cat = dpcg_truth_category(dpcg_row) if dpcg_row is not None else None
            feat["true_category"] = true_cat if true_cat is not None else ""
            if true_cat is None:
                n_no_truth += 1
                continue
            pairs.append((true_cat, feat["dpcg_like_category"]))

        if pairs:
            out_png = args.out_confusion_png or str(Path(args.out_csv).with_name("resectability_confusion_matrix.png"))
            plot_resectability_confusion(pairs, out_png)

            stats = multiclass_metrics([t for t, _ in pairs], [p for _, p in pairs], CATEGORIES)
            stats["n_skipped_no_truth"] = n_no_truth
            out_json = args.out_stats_json or str(Path(args.out_csv).with_name("resectability_stats.json"))
            json.dump(stats, open(out_json, "w", encoding="utf-8"), indent=2)

            correct = sum(1 for t, p in pairs if t == p)
            print(f"Resectability accuracy: {correct}/{len(pairs)} = {stats['accuracy']:.3f} ({n_no_truth} cases skipped, no ground-truth stage)")
            print(f"Macro precision={stats['macro_precision']:.3f} recall={stats['macro_recall']:.3f} f1={stats['macro_f1']:.3f} balanced_accuracy={stats['balanced_accuracy']:.3f}")
            for cat, m in stats["per_class"].items():
                print(f"  {cat:32s} support={m['support']:<4d} precision={m['precision']:.3f} recall={m['recall']:.3f} f1={m['f1']:.3f}")
            print(f"Saved confusion matrix to {out_png}")
            print(f"Saved stats to {out_json}")
        else:
            print(f"No cases had a usable ground-truth resectability stage in {args.dpcg_csv} - skipping confusion matrix and stats.")

    fieldnames = ["case_id", "pdac_voxels", "vein_contact_deg", "artery_contact_deg", "vein_min_dist_mm", "artery_min_dist_mm", "dpcg_like_category"]
    if args.dpcg_csv:
        fieldnames.append("true_category")
    with open(args.out_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Saved {args.out_csv}")


if __name__ == "__main__":
    main()
