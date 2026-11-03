"""Location filter: remove predicted pancreas and PDAC components that lie away from the gland.

The union of predicted pancreas and PDAC is split into 26-connected components and the largest
one is taken as the gland. Every other component whose nearest voxel lies farther than
margin_mm from it is set to background as a whole; a tumour attached to the gland is always
kept. Distances use the voxel spacing of the NIfTI header, and vessels and bile duct are left
unchanged. The margin is chosen on validation (50 mm in the thesis).

    python tools/filter_outside_pancreas.py --pred_dir <run>/inference_val_best/preds \
        --out_dir <run>/inference_val_best_roi50/preds --margin_mm 50

The per-case log goes to <out_dir>/../roi_filter.csv.
"""
from __future__ import annotations

import argparse
import csv
import os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Dict

import nibabel as nib
import numpy as np
from scipy import ndimage as ndi

STRUCT = np.ones((3, 3, 3), dtype=bool)


def filter_mask(pred: np.ndarray, spacing, margin_mm: float, pancreas: int = 4, pdac: int = 1):
    """Return (filtered copy, removed pancreas voxels, removed PDAC voxels)."""
    organ = (pred == pancreas) | (pred == pdac)
    lab, n = ndi.label(organ, structure=STRUCT)
    if n <= 1:
        return pred.copy(), 0, 0
    sizes = np.bincount(lab.ravel())[1:]
    main = lab == (int(np.argmax(sizes)) + 1)
    # the distance transform is computed only inside the main component's box widened by the margin
    sp = [float(s) for s in spacing]
    pad = [int(np.ceil(margin_mm / s)) + 1 for s in sp]
    sl = ndi.find_objects(main.astype(np.int8))[0]
    box = tuple(slice(max(s.start - p, 0), min(s.stop + p, dim)) for s, p, dim in zip(sl, pad, pred.shape))
    dist = np.full(pred.shape, np.inf, dtype=np.float32)
    dist[box] = ndi.distance_transform_edt(~main[box], sampling=sp)
    # a component is kept whole if its nearest voxel lies within margin_mm of the main component
    nearest = ndi.minimum(dist, labels=lab, index=np.arange(1, n + 1))
    keep_ids = np.flatnonzero(np.asarray(nearest) <= margin_mm) + 1
    drop = organ & ~np.isin(lab, keep_ids)
    out = pred.copy()
    removed_panc = int((drop & (pred == pancreas)).sum())
    removed_pdac = int((drop & (pred == pdac)).sum())
    out[drop] = 0
    return out, removed_panc, removed_pdac


def process_case(path: str, out_dir: str, margin_mm: float) -> Dict:
    img = nib.load(path)
    pred = np.asarray(img.dataobj)
    spacing = img.header.get_zooms()[:3]
    out, rp, rt = filter_mask(pred, spacing, margin_mm)
    dst = Path(out_dir) / Path(path).name
    nib.save(nib.Nifti1Image(out.astype(pred.dtype), img.affine, img.header), str(dst))
    name = Path(path).name
    case_id = name[:-7] if name.endswith(".nii.gz") else Path(path).stem
    return {"case_id": case_id, "removed_pancreas_voxels": rp, "removed_pdac_voxels": rt,
            "pdac_voxels_before": int((pred == 1).sum()), "pdac_voxels_after": int((out == 1).sum())}


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--pred_dir", required=True)
    p.add_argument("--out_dir", required=True, help="Write filtered masks here. Must differ from pred_dir.")
    p.add_argument("--margin_mm", type=float, required=True)
    p.add_argument("--jobs", type=int, default=int(os.environ.get("SLURM_CPUS_PER_TASK", 4)))
    a = p.parse_args()
    if os.path.abspath(a.out_dir) == os.path.abspath(a.pred_dir):
        raise SystemExit("out_dir must differ from pred_dir")
    os.makedirs(a.out_dir, exist_ok=True)
    paths = sorted(str(x) for x in Path(a.pred_dir).glob("*.nii*"))
    with ProcessPoolExecutor(max_workers=a.jobs) as ex:
        rows = list(ex.map(process_case, paths, [a.out_dir] * len(paths), [a.margin_mm] * len(paths)))
    log = Path(a.out_dir).parent / "roi_filter.csv"
    with open(log, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    lost = sum(1 for r in rows if r["pdac_voxels_before"] > 0 and r["pdac_voxels_after"] == 0)
    print(f"{len(rows)} cases, margin {a.margin_mm} mm: PDAC removed entirely in {lost} cases, "
          f"pancreas voxels removed in {sum(1 for r in rows if r['removed_pancreas_voxels'])} cases -> {log}")


if __name__ == "__main__":
    main()
