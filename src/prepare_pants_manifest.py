from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import nibabel as nib
import numpy as np


# Official PanTS class map from the PanTS data README.
PANTS_CLASS_MAP: Dict[int, str] = {
    1: "adrenal_gland_left",
    2: "adrenal_gland_right",
    3: "aorta",
    4: "bladder",
    5: "celiac_artery",
    6: "colon",
    7: "common_bile_duct",
    8: "duodenum",
    9: "femur_left",
    10: "femur_right",
    11: "gall_bladder",
    12: "kidney_left",
    13: "kidney_right",
    14: "liver",
    15: "lung_left",
    16: "lung_right",
    17: "pancreas",
    18: "pancreas_body",
    19: "pancreas_head",
    20: "pancreas_tail",
    21: "pancreatic_duct",
    22: "postcava",
    23: "prostate",
    24: "spleen",
    25: "stomach",
    26: "superior_mesenteric_artery",
    27: "veins",
    28: "pancreatic_lesion",
}

PANTS6_CLASS_MAP: Dict[int, str] = {
    0: "background",
    1: "pancreatic_lesion_proxy",
    2: "veins_postcava",
    3: "arteries_aorta_celiac_sma",
    4: "pancreas_plus_pancreatic_duct",
    5: "common_bile_duct",
}

MODE_TO_NUM_CLASSES = {
    "all": 29,
    "pants28": 29,
    "pants6": 6,
}

MODE_TO_LABEL_SUBDIR = {
    "all": "all",
    "pants28": "all",
    "pants6": "pants6",
}

MODE_TO_LESION_LABEL = {
    "all": 28,
    "pants28": 28,
    "pants6": 1,
}

MODE_TO_DEFAULT_MANIFEST = {
    "all": "data/pants_manifest_all.json",
    "pants28": "data/pants_manifest_all.json",
    "pants6": "data/pants_manifest_pants6.json",
}


def _case_id_from_image_name(path: Path, image_suffix: str) -> str:
    name = path.name
    if not name.endswith(image_suffix):
        raise ValueError(f"Image file {path} does not end with suffix {image_suffix}")
    return name[: -len(image_suffix)]


def _load_split_map(split_json: Optional[str | Path]) -> Dict[str, str]:
    if split_json is None:
        return {}
    p = Path(split_json)
    if not p.exists():
        return {}
    data = json.loads(p.read_text(encoding="utf-8"))
    return {str(k): str(v) for k, v in data.items()}


def _infer_split_from_case_id(case_id: str, test_split_name: str) -> str:
    # fallback without splits.json: PanTS ImageTe cases start at PanTS_00009001
    try:
        numeric = int(case_id.split("_")[-1])
        return test_split_name if numeric >= 9001 else "train"
    except Exception:
        return "train"



def _convert_all_label_to_pants6_array(arr: np.ndarray) -> np.ndarray:
    out = np.zeros(arr.shape, dtype=np.uint8)
    # veins first, arteries, pancreas, CBD, then lesion on top.
    out[np.isin(arr, [22, 27])] = 2
    out[np.isin(arr, [3, 5, 26])] = 3
    out[np.isin(arr, [17, 18, 19, 20, 21])] = 4
    out[arr == 7] = 5
    out[arr == 28] = 1
    return out


def _ensure_pants6_from_all(converted_root: Path, image_suffix: str, label_suffix: str) -> None:
    all_dir = converted_root / "labels" / "all"
    pants6_dir = converted_root / "labels" / "pants6"
    if pants6_dir.exists() and any(pants6_dir.glob(f"*{label_suffix}")):
        return
    if not all_dir.exists():
        return
    pants6_dir.mkdir(parents=True, exist_ok=True)
    all_paths = sorted(all_dir.glob(f"*{label_suffix}"))
    if not all_paths:
        return
    print(f"Creating pants6 labels from full PanTS labels: {all_dir} -> {pants6_dir}")
    for src in all_paths:
        dst = pants6_dir / src.name
        if dst.exists():
            continue
        img = nib.load(str(src))
        arr = np.asanyarray(img.dataobj).astype(np.uint8)
        out = _convert_all_label_to_pants6_array(arr)
        header = img.header.copy()
        header.set_data_dtype(np.uint8)
        nib.save(nib.Nifti1Image(out, img.affine, header), str(dst))

def _count_label_voxels(label_path: Path, label_id: int) -> int:
    arr = np.asanyarray(nib.load(str(label_path)).dataobj)
    return int(np.count_nonzero(arr == label_id))


def _verify_pair(image_path: Path, label_path: Path, max_label: int) -> Tuple[List[int], List[int]]:
    image = nib.load(str(image_path))
    label = nib.load(str(label_path))
    if tuple(image.shape) != tuple(label.shape):
        raise AssertionError(f"Image/label shape mismatch: {image_path} {image.shape} vs {label_path} {label.shape}")
    labels = np.unique(np.asanyarray(label.dataobj)).astype(int).tolist()
    if labels and (min(labels) < 0 or max(labels) > max_label):
        raise AssertionError(f"Invalid label IDs in {label_path}: {labels}; allowed 0..{max_label}")
    return list(image.shape), labels


def build_converted_pants_manifest_rows(
    converted_root: str | Path = "../data/PanTs",
    mode: str = "pants6",
    image_dir: Optional[str | Path] = None,
    label_dir: Optional[str | Path] = None,
    split_json: Optional[str | Path] = None,
    image_suffix: str = "_0000_trans.nii.gz",
    label_suffix: str = "_trans.nii.gz",
    test_split_name: str = "val",
    verify: bool = True,
    count_labels: bool = True,
) -> List[Dict[str, object]]:
    if mode not in MODE_TO_NUM_CLASSES:
        raise ValueError(f"Unsupported PanTS mode={mode}. Use one of {sorted(MODE_TO_NUM_CLASSES)}")

    converted_root = Path(converted_root)
    image_dir = Path(image_dir) if image_dir is not None else converted_root / "images"
    if mode == "pants6" and label_dir is None:
        _ensure_pants6_from_all(converted_root, image_suffix=image_suffix, label_suffix=label_suffix)
    label_dir = Path(label_dir) if label_dir is not None else converted_root / "labels" / MODE_TO_LABEL_SUBDIR[mode]
    split_json = Path(split_json) if split_json is not None else converted_root / "splits.json"

    if not image_dir.exists():
        raise FileNotFoundError(f"Converted PanTS image_dir does not exist: {image_dir}")
    if not label_dir.exists():
        raise FileNotFoundError(f"Converted PanTS label_dir does not exist: {label_dir}")

    split_map = _load_split_map(split_json)
    rows: List[Dict[str, object]] = []
    missing_labels: List[str] = []
    image_paths = sorted(image_dir.glob(f"*{image_suffix}"))
    if not image_paths:
        raise FileNotFoundError(f"No converted PanTS images matching *{image_suffix} in {image_dir}")

    lesion_label = MODE_TO_LESION_LABEL[mode]
    max_label = MODE_TO_NUM_CLASSES[mode] - 1

    for image_path in image_paths:
        case_id = _case_id_from_image_name(image_path, image_suffix)
        label_path = label_dir / f"{case_id}{label_suffix}"
        if not label_path.exists():
            missing_labels.append(case_id)
            continue
        split = split_map.get(case_id, _infer_split_from_case_id(case_id, test_split_name))
        if split == "test":
            split = test_split_name

        if verify:
            _verify_pair(image_path, label_path, max_label=max_label)
        lesion_voxels = _count_label_voxels(label_path, lesion_label) if count_labels else 0

        rows.append({
            "case_id": case_id,
            "patient_id": case_id,
            "source": "PanTS",
            "image": str(image_path),
            "label": str(label_path),
            "split": split,
            "has_pdac": int(lesion_voxels > 0),
            "tumor_type": "pancreatic_lesion" if lesion_voxels > 0 else "none",
            "tumor_group": "pancreatic_lesion" if lesion_voxels > 0 else "control_or_no_lesion",
            "hard_negative": 0,
            "is_non_pdac_lesion": 0,
            "diagnosis_uncertain": 0,
            "pants_mode": "all" if mode == "pants28" else mode,
            "lesion_voxels": int(lesion_voxels),
        })

    if missing_labels:
        preview = ", ".join(missing_labels[:10])
        raise FileNotFoundError(f"Missing converted PanTS labels for {len(missing_labels)} cases in {label_dir}. First cases: {preview}")
    return rows


def summarize_rows(rows: List[Dict[str, object]], converted_root: str | Path, mode: str, image_dir: str | Path, label_dir: str | Path) -> Dict[str, object]:
    return {
        "converted_root": str(converted_root),
        "mode": "all" if mode == "pants28" else mode,
        "num_classes": MODE_TO_NUM_CLASSES[mode],
        "image_dir": str(image_dir),
        "label_dir": str(label_dir),
        "num_rows": len(rows),
        "num_train": sum(1 for r in rows if r.get("split") == "train"),
        "num_val": sum(1 for r in rows if r.get("split") == "val"),
        "num_with_lesion": sum(1 for r in rows if int(r.get("has_pdac", 0)) == 1),
        "label_classes": ({"0": "background", **{str(k): v for k, v in PANTS_CLASS_MAP.items()}} if mode in {"all", "pants28"} else {str(k): v for k, v in PANTS6_CLASS_MAP.items()}),
    }


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build a PanTS manifest from already converted CTs and label maps.")
    p.add_argument("--converted_root", default="../data/PanTs", help="Directory with images/, labels/all, labels/pants6 and splits.json.")
    p.add_argument("--mode", choices=["all", "pants28", "pants6"], default="pants6")
    p.add_argument("--image_dir", default=None, help="Override converted image directory.")
    p.add_argument("--label_dir", default=None, help="Override converted label directory.")
    p.add_argument("--split_json", default=None, help="Path to splits.json of the converted PanTS data.")
    p.add_argument("--out_manifest", default=None)
    p.add_argument("--image_suffix", default="_0000_trans.nii.gz")
    p.add_argument("--label_suffix", default="_trans.nii.gz")
    p.add_argument("--test_split_name", default="val")
    p.add_argument("--no_verify", action="store_true")
    p.add_argument("--no_count_labels", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    mode = args.mode
    converted_root = Path(args.converted_root)
    image_dir = Path(args.image_dir) if args.image_dir is not None else converted_root / "images"
    label_dir = Path(args.label_dir) if args.label_dir is not None else converted_root / "labels" / MODE_TO_LABEL_SUBDIR[mode]
    out_manifest = Path(args.out_manifest or MODE_TO_DEFAULT_MANIFEST[mode])

    rows = build_converted_pants_manifest_rows(
        converted_root=converted_root,
        mode=mode,
        image_dir=image_dir,
        label_dir=label_dir,
        split_json=args.split_json,
        image_suffix=args.image_suffix,
        label_suffix=args.label_suffix,
        test_split_name=args.test_split_name,
        verify=not args.no_verify,
        count_labels=not args.no_count_labels,
    )
    out_manifest.parent.mkdir(parents=True, exist_ok=True)
    out_manifest.write_text(json.dumps(rows, indent=2, sort_keys=True), encoding="utf-8")
    summary = summarize_rows(rows, converted_root, mode, image_dir, label_dir)
    Path(str(out_manifest) + ".summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    print(f"Wrote {len(rows)} rows to {out_manifest}")
    print(f"Summary: {out_manifest}.summary.json")


if __name__ == "__main__":
    main()
