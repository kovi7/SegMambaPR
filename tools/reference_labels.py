"""Reference resectability categories from the ground-truth masks (DPCG contact-angle criteria).

For every PDAC-positive case the maximum axial contact angle between the tumour and each
connected vessel component is measured (arteries without their largest component, the aorta),
and the angles are mapped to resectable, borderline resectable or locally advanced. Cases
without a tumour voxel are non-PDAC. The output is the dpcg_labels.csv the configs read.

    python tools/reference_labels.py --label_dir /path/to/panorama/converted/labels \
        --out_csv /path/to/panorama/dpcg_labels.csv
"""
from __future__ import annotations

import argparse
import csv
import os

import nibabel as nib
import numpy as np
import scipy.ndimage as ndi

PDAC_LABEL, VEIN_LABEL, ART_LABEL = 1, 2, 3
SLICE_AXIS = 2  # axial slices


def load_label(path: str) -> np.ndarray:
    data = nib.load(path).get_fdata().astype(np.int16)
    if data.ndim == 4:
        if data.shape[-1] == 1:
            data = data[..., 0]
        elif data.shape[0] == 1:
            data = data[0]
        else:
            raise ValueError(f"Unexpected 4D NIfTI shape {data.shape} in {path}")
    if data.ndim != 3:
        raise ValueError(f"Expected a 3D label map, got shape={data.shape} in {path}")
    return data


def remove_aorta(art_mask: np.ndarray) -> np.ndarray:
    """Remove the largest 6-connected component of the artery mask."""
    struct = ndi.generate_binary_structure(3, 1)
    labeled, ncomp = ndi.label(art_mask, structure=struct)
    if ncomp <= 1:
        return art_mask
    sizes = ndi.sum(art_mask, labeled, index=range(1, ncomp + 1))
    cleaned = art_mask.copy()
    cleaned[labeled == int(np.argmax(sizes)) + 1] = False
    return cleaned


def max_contact_angle_for_component(mask: np.ndarray, tumor: np.ndarray, slice_axis: int) -> float:
    """Maximum over axial slices of contact border / vessel border * 360 for one vessel component."""
    max_angle = 0.0
    cross = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=bool)
    for k in range(mask.shape[slice_axis]):
        v = np.take(mask, k, axis=slice_axis)
        if not v.any():
            continue
        t = np.take(tumor, k, axis=slice_axis)
        if not t.any():
            continue
        border = v & ~ndi.binary_erosion(v, structure=np.ones((3, 3), bool))
        if not border.any():
            continue
        perimeter = float(border.sum())
        if perimeter <= 0:
            continue
        # vessel border pixels in the 4-neighbourhood of the tumour
        contact = float((border & ndi.binary_dilation(t, structure=cross)).sum())
        max_angle = max(max_angle, contact / perimeter * 360.0)
    return float(max_angle)


def max_contact_angle_grouped(vessel_mask: np.ndarray, tumor: np.ndarray, slice_axis: int,
                              drop_largest: bool = False) -> float:
    """Maximum contact angle over the connected vessel components."""
    mask = remove_aorta(vessel_mask.copy()) if drop_largest else vessel_mask.copy()
    if not mask.any():
        return 0.0
    labeled, ncomp = ndi.label(mask, structure=ndi.generate_binary_structure(3, 1))
    best = 0.0
    for lab in range(1, ncomp + 1):
        best = max(best, max_contact_angle_for_component(labeled == lab, tumor, slice_axis))
    return float(best)


def classify_dpcg(art_angle: float, vein_angle: float) -> str:
    """Arteries > 90 deg or veins > 270 deg: locally advanced; any arterial contact or veins > 90 deg: borderline."""
    if art_angle > 90.0 or vein_angle > 270.0:
        return "Locally advanced"
    if art_angle > 0.0 or vein_angle > 90.0:
        return "Borderline resectable"
    return "Resectable"


def category_to_multilabel(category: str, pdac_present: int):
    """[has_pdac, is_resectable, is_borderline, is_locally_advanced]"""
    if pdac_present == 0:
        return [0, 0, 0, 0]
    return {"Resectable": [1, 1, 0, 0],
            "Borderline resectable": [1, 0, 1, 0],
            "Locally advanced": [1, 0, 0, 1]}[category]


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--label_dir", required=True, help="Ground-truth masks <case>_trans.nii.gz with the stored label ids.")
    p.add_argument("--out_csv", required=True)
    p.add_argument("--max_cases", type=int, default=None)
    args = p.parse_args()

    files = sorted(f for f in os.listdir(args.label_dir) if f.endswith("_trans.nii.gz"))
    if args.max_cases is not None:
        files = files[:args.max_cases]
    print(f"{len(files)} label files")

    rows = []
    for i, fname in enumerate(files, start=1):
        case_id = fname[:-len("_trans.nii.gz")]
        seg = load_label(os.path.join(args.label_dir, fname))
        tumor = seg == PDAC_LABEL
        if not tumor.any():
            art_angle = vein_angle = 0.0
            category, pdac_present = "No PDAC", 0
        else:
            pdac_present = 1
            art_angle = max_contact_angle_grouped(seg == ART_LABEL, tumor, SLICE_AXIS, drop_largest=True)
            vein_angle = max_contact_angle_grouped(seg == VEIN_LABEL, tumor, SLICE_AXIS, drop_largest=False)
            category = classify_dpcg(art_angle, vein_angle)
        print(f"[{i}/{len(files)}] {case_id}: arteries {art_angle:.1f} deg, veins {vein_angle:.1f} deg, {category}")
        rows.append([case_id, *category_to_multilabel(category, pdac_present)])

    os.makedirs(os.path.dirname(os.path.abspath(args.out_csv)), exist_ok=True)
    with open(args.out_csv, "w", newline="") as f:
        writer = csv.writer(f, delimiter=";")
        writer.writerow(["case_id", "has_pdac", "is_resectable", "is_borderline", "is_locally_advanced"])
        writer.writerows(rows)
    print(f"Saved {args.out_csv}")


if __name__ == "__main__":
    main()
