"""Resample PANORAMA images and labels to 0.8 x 0.8 x 3.0 mm in LPS orientation.

Input:  <input_dir>/images/<patient>_<study>_0000.nii.gz and <input_dir>/labels/<patient>_<study>.nii.gz
Output: <output_dir>/images/<case>_0000_trans.nii.gz and <output_dir>/labels/<case>_trans.nii.gz
Images are resampled bilinearly and labels by nearest neighbour. Failed cases are listed in
<output_dir>/_errors.csv.

    python tools/convert_panorama.py --input_dir /path/to/panorama --output_dir /path/to/panorama/converted
"""
from __future__ import annotations

import argparse
import csv
import glob
import os
from typing import Dict, List, Tuple

from monai.transforms import Compose, EnsureChannelFirstd, LoadImaged, Orientationd, SaveImaged, Spacingd

AXIS_LABELS = (("L", "R"), ("P", "A"), ("I", "S"))


def stem_tokens(path: str) -> Tuple[str, str, str]:
    """(patient_id, study_id, channel) from a file name, e.g. 100005_00001_0000.nii.gz."""
    base = os.path.basename(path)
    if base.endswith(".nii.gz"):
        base = base[:-7]
    parts = base.split("_")
    if len(parts) >= 3:
        return parts[0], parts[1], parts[2]
    if len(parts) == 2:
        return parts[0], parts[1], ""
    return parts[0], "", ""


class ConvertSpacing:
    def __init__(self, input_path: str, output_path: str, spacing=(0.8, 0.8, 3.0), skip_existing: bool = False):
        self.output_path = output_path
        self.skip_existing = skip_existing
        os.makedirs(output_path, exist_ok=True)

        images = sorted(glob.glob(os.path.join(input_path, "images/*.nii.gz")))
        labels = sorted(glob.glob(os.path.join(input_path, "labels/*.nii.gz")))
        labels_by_key: Dict[Tuple[str, str], str] = {}
        for lp in labels:
            pid, sid, _ = stem_tokens(lp)
            labels_by_key[(pid, sid)] = lp

        self.pairs: List[Tuple[str, str]] = []
        self.missing_labels: List[str] = []
        for ip in images:
            pid, sid, _ = stem_tokens(ip)
            lbl = labels_by_key.get((pid, sid))
            if lbl is None:
                self.missing_labels.append(ip)
            else:
                self.pairs.append((ip, lbl))

        self.out_images = os.path.join(output_path, "images")
        self.out_labels = os.path.join(output_path, "labels")
        os.makedirs(self.out_images, exist_ok=True)
        os.makedirs(self.out_labels, exist_ok=True)

        self.errors_csv = os.path.join(output_path, "_errors.csv")
        if not os.path.exists(self.errors_csv):
            with open(self.errors_csv, "w", newline="") as f:
                csv.writer(f).writerow(["image_path", "label_path", "reason", "image_shape", "label_shape"])

        # SaveImaged appends the postfix "_trans" to the input file name
        self.transform = Compose([
            LoadImaged(keys=["image", "label"]),
            EnsureChannelFirstd(keys=["image", "label"]),
            Orientationd(keys=["image", "label"], axcodes="LPS", labels=AXIS_LABELS),
            Spacingd(keys=["image", "label"], pixdim=tuple(spacing), mode=("bilinear", "nearest")),
            SaveImaged(keys=["image"], output_dir=self.out_images, separate_folder=False, resample=False),
            SaveImaged(keys=["label"], output_dir=self.out_labels, separate_folder=False, resample=False),
        ])

    def __len__(self) -> int:
        return len(self.pairs)

    def _log_error(self, image_path: str, label_path: str, reason: str, ishape: str = "", lshape: str = ""):
        with open(self.errors_csv, "a", newline="") as f:
            csv.writer(f).writerow([image_path, label_path, reason, ishape, lshape])

    def _done(self, label_path: str) -> bool:
        name = os.path.basename(label_path)[:-len(".nii.gz")] + "_trans.nii.gz"
        return os.path.exists(os.path.join(self.out_labels, name))

    def __getitem__(self, idx: int):
        image_path, label_path = self.pairs[idx]
        if self.skip_existing and self._done(label_path):
            return None
        try:
            data = self.transform({"image": image_path, "label": label_path})
            ishape, lshape = tuple(data["image"].shape), tuple(data["label"].shape)
            if ishape != lshape:
                self._log_error(image_path, label_path, "shape_mismatch_after_resample", str(ishape), str(lshape))
            return data
        except Exception as e:
            self._log_error(image_path, label_path, f"exception:{type(e).__name__}: {e}")
            return None

    def process_all(self) -> None:
        for ip in self.missing_labels:
            self._log_error(ip, "", "missing_label")
        for i in range(len(self)):
            self[i]


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--input_dir", required=True, help="Directory with images/ and labels/.")
    p.add_argument("--output_dir", required=True)
    p.add_argument("--spacing", nargs=3, type=float, default=[0.8, 0.8, 3.0])
    p.add_argument("--skip_existing", action="store_true", help="Skip cases whose converted label already exists.")
    args = p.parse_args()
    converter = ConvertSpacing(args.input_dir, args.output_dir, args.spacing, args.skip_existing)
    converter.process_all()
    print(f"Processed {len(converter)} pairs. Error log: {converter.errors_csv}")


if __name__ == "__main__":
    main()
