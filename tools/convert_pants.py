"""Convert PanTS to the PANORAMA grid: one 0..28 label map per case, 0.8 x 0.8 x 3.0 mm, LPS.

The 28 binary PanTS masks of a case are merged into one label map (later classes overwrite
earlier ones, the lesion last) and resampled together with the CT. ImageTr becomes the train
split and ImageTe the val split. Output under --out_root:
  images/<case>_0000_trans.nii.gz
  labels/all/<case>_trans.nii.gz
  splits.json
  conversion_reports/all_summary.json, all_case_stats.json

Existing outputs are skipped unless --overwrite is given. With --workers > 1 every case runs
in a fresh process, which releases the memory held by nibabel/MONAI/ITK.

    python tools/convert_pants.py --pants_root /path/to/PanTS/data --out_root /path/to/PanTS/converted
"""
from __future__ import annotations

import argparse
import gc
import json
import multiprocessing as mp
import os
import shutil
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import nibabel as nib
import numpy as np
from monai.transforms import (
    Compose,
    EnsureChannelFirstd,
    LoadImaged,
    Orientationd,
    ResampleToMatchd,
    Spacingd,
)

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

AXIS_LABELS = (("L", "R"), ("P", "A"), ("I", "S"))


@dataclass
class CaseTask:
    case_id: str
    split: str
    image_path: str
    seg_dir: str


@dataclass
class CaseStats:
    case_id: str
    split: str
    image: str
    label: str
    status: str
    lesion_voxels: int = 0
    overwritten_voxels: int = 0
    missing_masks: List[str] = field(default_factory=list)
    present_masks: List[str] = field(default_factory=list)
    unique_labels: List[int] = field(default_factory=list)
    image_shape: List[int] = field(default_factory=list)
    label_shape: List[int] = field(default_factory=list)
    reason: str = ""


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Safely convert PanTS binary masks to converted CTs and full 0..28 label maps.")
    p.add_argument("--pants_root", required=True, help="Directory containing ImageTr, ImageTe, LabelTr, LabelTe.")
    p.add_argument("--out_root", required=True, help="Output directory for converted images and labels/all.")
    p.add_argument("--spacing", nargs=3, type=float, default=[0.8, 0.8, 3.0], metavar=("SX", "SY", "SZ"))
    p.add_argument("--orientation", default="LPS")
    p.add_argument("--image_suffix", default="_0000_trans.nii.gz")
    p.add_argument("--label_suffix", default="_trans.nii.gz")
    p.add_argument("--overwrite", action="store_true", help="Recompute cases even if both output files already exist.")
    p.add_argument("--strict_masks", action="store_true", help="Fail if any official PanTS mask is missing for a case.")
    p.add_argument("--max_cases", type=int, default=None, help="Optional smoke-test limit.")
    p.add_argument("--case_id", default=None, help="Optional single case id, e.g. PanTS_00001234.")
    p.add_argument("--no_verify", action="store_true", help="Do not reload saved files for verification.")
    p.add_argument("--workers", type=int, default=1, help="Number of parallel case workers. Use 1 first; 2 only if memory is stable.")
    p.add_argument("--tmp_dir", default=None, help="Temporary directory. Default: <out_root>/conversion_tmp.")
    p.add_argument("--log_every", type=int, default=10)
    return p.parse_args()


def _case_dirs(root: Path, split_name: str) -> List[Path]:
    d = root / split_name
    if not d.exists():
        return []
    return sorted([p for p in d.iterdir() if p.is_dir() and p.name.startswith("PanTS_")])


def iter_cases(pants_root: Path, max_cases: Optional[int] = None, case_id: Optional[str] = None) -> Iterable[CaseTask]:
    yielded = 0
    for image_split, label_split, split in [("ImageTr", "LabelTr", "train"), ("ImageTe", "LabelTe", "val")]:
        for image_case_dir in _case_dirs(pants_root, image_split):
            cid = image_case_dir.name
            if case_id is not None and cid != case_id:
                continue
            image_path = image_case_dir / "ct.nii.gz"
            seg_dir = pants_root / label_split / cid / "segmentations"
            if not image_path.exists():
                print(f"[SKIP] {cid}: missing CT: {image_path}")
                continue
            if not seg_dir.exists():
                print(f"[SKIP] {cid}: missing segmentations: {seg_dir}")
                continue
            yield CaseTask(case_id=cid, split=split, image_path=str(image_path), seg_dir=str(seg_dir))
            yielded += 1
            if max_cases is not None and yielded >= max_cases:
                return


def find_mask(seg_dir: Path, class_name: str) -> Optional[Path]:
    for p in [seg_dir / f"{class_name}.nii.gz", seg_dir / f"{class_name}.nii"]:
        if p.exists():
            return p
    lower = class_name.lower()
    for p in seg_dir.glob("*.nii*"):
        stem = p.name.lower().replace(".nii.gz", "").replace(".nii", "")
        if stem == lower:
            return p
    return None


def load_binary_mask(seg_dir: Path, class_id: int, expected_shape: Tuple[int, int, int]) -> Tuple[np.ndarray, Optional[Path]]:
    name = PANTS_CLASS_MAP[class_id]
    p = find_mask(seg_dir, name)
    if p is None:
        return np.zeros(expected_shape, dtype=bool), None
    arr = np.asanyarray(nib.load(str(p)).dataobj)
    if tuple(arr.shape) != tuple(expected_shape):
        raise ValueError(f"Shape mismatch for {p}: got {arr.shape}, expected {expected_shape}")
    return arr > 0, p


def build_all_labelmap(seg_dir: Path, shape: Tuple[int, int, int], strict_masks: bool = False) -> Tuple[np.ndarray, Dict[str, object]]:
    out = np.zeros(shape, dtype=np.uint8)
    missing: List[str] = []
    present: List[str] = []
    overwritten = 0
    for class_id in range(1, 29):
        mask, path = load_binary_mask(seg_dir, class_id, shape)
        name = PANTS_CLASS_MAP[class_id]
        if path is None:
            missing.append(name)
            if strict_masks:
                raise FileNotFoundError(f"Missing PanTS mask for class {class_id}: {name} in {seg_dir}")
            continue
        present.append(name)
        overwritten += int(np.count_nonzero(mask & (out > 0) & (out != class_id)))
        out[mask] = class_id
        del mask
    return out, {"missing_masks": missing, "present_masks": present, "overwritten_voxels": overwritten}


def save_raw_labelmap(labelmap: np.ndarray, ref_img: nib.Nifti1Image, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    header = ref_img.header.copy()
    header.set_data_dtype(np.uint8)
    nib.save(nib.Nifti1Image(labelmap.astype(np.uint8), ref_img.affine, header), str(out_path))


def as_numpy(x) -> np.ndarray:
    if hasattr(x, "detach"):
        return x.detach().cpu().numpy()
    if hasattr(x, "cpu") and hasattr(x.cpu(), "numpy"):
        return x.cpu().numpy()
    return np.asarray(x)


def get_affine(x) -> np.ndarray:
    affine = getattr(x, "affine", None)
    if affine is not None:
        return as_numpy(affine)
    meta = getattr(x, "meta", None)
    if isinstance(meta, Mapping) and "affine" in meta:
        return as_numpy(meta["affine"])
    raise RuntimeError("Could not recover affine from MONAI transformed image. Check MONAI version.")


def transform_pair(image_path: Path, raw_label_path: Path, spacing: Sequence[float], orientation: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    transform = Compose([
        LoadImaged(keys=["image", "label"], image_only=False),
        EnsureChannelFirstd(keys=["image", "label"]),
        Orientationd(keys=["image", "label"], axcodes=orientation, labels=AXIS_LABELS),
        Spacingd(keys=["image", "label"], pixdim=tuple(spacing), mode=["bilinear", "nearest"]),
        ResampleToMatchd(keys="label", key_dst="image", mode="nearest"),
    ])
    data = transform({"image": str(image_path), "label": str(raw_label_path)})
    image = as_numpy(data["image"])
    label = as_numpy(data["label"])
    if image.shape != label.shape:
        raise AssertionError(f"Image/label shape mismatch after MONAI transform: image={image.shape}, label={label.shape}")
    if image.ndim != 4 or image.shape[0] != 1:
        raise AssertionError(f"Unexpected transformed image shape: {image.shape}")
    return image[0].astype(np.float32), np.rint(label[0]).astype(np.uint8), get_affine(data["image"])


def save_nifti(data: np.ndarray, affine: np.ndarray, out_path: Path, dtype) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    header = nib.Nifti1Header()
    header.set_data_dtype(dtype)
    nib.save(nib.Nifti1Image(data.astype(dtype), affine, header), str(out_path))


def verify_output(image_path: Path, label_path: Path) -> Tuple[List[int], List[int], List[int]]:
    img = nib.load(str(image_path))
    lbl = nib.load(str(label_path))
    img_shape = list(img.shape)
    lbl_shape = list(lbl.shape)
    if tuple(img_shape) != tuple(lbl_shape):
        raise AssertionError(f"Saved image/label shape mismatch: image={img_shape}, label={lbl_shape}, label_path={label_path}")
    labels = np.unique(np.asanyarray(lbl.dataobj)).astype(int).tolist()
    if labels and (min(labels) < 0 or max(labels) > 28):
        raise AssertionError(f"Invalid labels in {label_path}: {labels}; allowed 0..28")
    return img_shape, lbl_shape, labels


def process_case(payload: Tuple[CaseTask, Dict[str, object]]) -> Dict[str, object]:
    task, cfg = payload
    out_root = Path(str(cfg["out_root"]))
    image_out_dir = out_root / "images"
    label_out_dir = out_root / "labels" / "all"
    image_out_path = image_out_dir / f"{task.case_id}{cfg['image_suffix']}"
    label_out_path = label_out_dir / f"{task.case_id}{cfg['label_suffix']}"

    if (not bool(cfg["overwrite"])) and image_out_path.exists() and label_out_path.exists():
        return CaseStats(
            case_id=task.case_id,
            split=task.split,
            image=str(image_out_path),
            label=str(label_out_path),
            status="skipped_existing",
        ).__dict__

    case_tmp_root = Path(str(cfg["tmp_dir"]))
    case_tmp = Path(tempfile.mkdtemp(prefix=f"{task.case_id}_", dir=str(case_tmp_root)))
    try:
        ref_img = nib.load(task.image_path)
        shape = tuple(int(v) for v in ref_img.shape)
        if len(shape) != 3:
            raise ValueError(f"unexpected CT shape {shape}")

        labelmap, info = build_all_labelmap(Path(task.seg_dir), shape, strict_masks=bool(cfg["strict_masks"]))
        # uncompressed temporary label map, removed after the case
        raw_label_path = case_tmp / f"{task.case_id}_raw_all.nii"
        save_raw_labelmap(labelmap, ref_img, raw_label_path)
        del labelmap
        gc.collect()

        image_arr, label_arr, affine = transform_pair(
            Path(task.image_path), raw_label_path, cfg["spacing"], str(cfg["orientation"])
        )
        save_nifti(image_arr, affine, image_out_path, np.float32)
        save_nifti(label_arr, affine, label_out_path, np.uint8)

        if bool(cfg["no_verify"]):
            image_shape = list(image_arr.shape)
            label_shape = list(label_arr.shape)
            unique_labels = np.unique(label_arr).astype(int).tolist()
        else:
            image_shape, label_shape, unique_labels = verify_output(image_out_path, label_out_path)

        lesion_voxels = int(np.count_nonzero(label_arr == 28))
        result = CaseStats(
            case_id=task.case_id,
            split=task.split,
            image=str(image_out_path),
            label=str(label_out_path),
            status="converted",
            lesion_voxels=lesion_voxels,
            overwritten_voxels=int(info["overwritten_voxels"]),
            missing_masks=list(info["missing_masks"]),
            present_masks=list(info["present_masks"]),
            unique_labels=unique_labels,
            image_shape=image_shape,
            label_shape=label_shape,
        ).__dict__
        del image_arr, label_arr
        gc.collect()
        return result
    except Exception as e:
        return CaseStats(
            case_id=task.case_id,
            split=task.split,
            image=str(image_out_path),
            label=str(label_out_path),
            status="failed",
            reason=repr(e),
        ).__dict__
    finally:
        shutil.rmtree(case_tmp, ignore_errors=True)
        gc.collect()


def main() -> None:
    args = parse_args()
    pants_root = Path(args.pants_root).resolve()
    out_root = Path(args.out_root)
    image_out_dir = out_root / "images"
    label_out_dir = out_root / "labels" / "all"
    reports_dir = out_root / "conversion_reports"
    tmp_dir = Path(args.tmp_dir) if args.tmp_dir else out_root / "conversion_tmp"

    image_out_dir.mkdir(parents=True, exist_ok=True)
    label_out_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)
    tmp_dir.mkdir(parents=True, exist_ok=True)

    if not (pants_root / "ImageTr").exists() and not (pants_root / "ImageTe").exists():
        raise RuntimeError(f"Could not find ImageTr/ImageTe under pants_root={pants_root}")

    tasks = list(iter_cases(pants_root, args.max_cases, args.case_id))
    if not tasks:
        raise RuntimeError("No PanTS cases found. Check --pants_root and --case_id.")

    cfg = {
        "out_root": str(out_root),
        "tmp_dir": str(tmp_dir),
        "spacing": list(args.spacing),
        "orientation": args.orientation,
        "image_suffix": args.image_suffix,
        "label_suffix": args.label_suffix,
        "overwrite": bool(args.overwrite),
        "strict_masks": bool(args.strict_masks),
        "no_verify": bool(args.no_verify),
    }

    print(f"Found {len(tasks)} cases")
    print(f"Output root: {out_root}")
    print(f"Temporary dir: {tmp_dir}")
    print(f"Workers: {args.workers}")

    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    os.environ.setdefault("ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS", "1")
    os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

    results: List[Dict[str, object]] = []
    workers = max(1, int(args.workers))

    if workers == 1:
        for idx, task in enumerate(tasks, start=1):
            res = process_case((task, cfg))
            results.append(res)
            if idx % max(1, args.log_every) == 0 or res.get("status") == "failed":
                print(f"[{idx}/{len(tasks)}] {task.case_id}: {res.get('status')} {res.get('reason', '')}")
    else:
        ctx = mp.get_context("spawn")
        with ctx.Pool(processes=workers, maxtasksperchild=1) as pool:
            for idx, res in enumerate(pool.imap_unordered(process_case, [(t, cfg) for t in tasks]), start=1):
                results.append(res)
                if idx % max(1, args.log_every) == 0 or res.get("status") == "failed":
                    print(f"[{idx}/{len(tasks)}] {res.get('case_id')}: {res.get('status')} {res.get('reason', '')}")

    converted = [r for r in results if r.get("status") == "converted"]
    skipped_existing = [r for r in results if r.get("status") == "skipped_existing"]
    failed = [r for r in results if r.get("status") == "failed"]
    usable = converted + skipped_existing
    splits = {str(r["case_id"]): str(r["split"]) for r in usable}

    (out_root / "splits.json").write_text(json.dumps(splits, indent=2, sort_keys=True), encoding="utf-8")
    summary = {
        "pants_root": str(pants_root),
        "out_root": str(out_root),
        "image_dir": str(image_out_dir),
        "label_dir_all": str(label_out_dir),
        "tmp_dir": str(tmp_dir),
        "spacing": list(args.spacing),
        "orientation": args.orientation,
        "num_cases_total_seen": len(tasks),
        "num_converted": len(converted),
        "num_skipped_existing": len(skipped_existing),
        "num_failed": len(failed),
        "num_usable": len(usable),
        "num_train": sum(1 for v in splits.values() if v == "train"),
        "num_val": sum(1 for v in splits.values() if v == "val"),
        "num_with_lesion_converted": sum(1 for r in converted if int(r.get("lesion_voxels", 0)) > 0),
        "label_map": {"0": "background", **{str(k): v for k, v in PANTS_CLASS_MAP.items()}},
        "failed_first_100": failed[:100],
    }
    (reports_dir / "all_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    (reports_dir / "all_case_stats.json").write_text(json.dumps(results, indent=2, sort_keys=True), encoding="utf-8")

    print(f"Done. converted={len(converted)}, skipped_existing={len(skipped_existing)}, failed={len(failed)}")
    print(f"Images: {image_out_dir}")
    print(f"All-label masks: {label_out_dir}")
    print(f"Splits: {out_root / 'splits.json'}")
    if failed:
        raise SystemExit(f"Failed cases: {len(failed)}. See {reports_dir / 'all_summary.json'}")


if __name__ == "__main__":
    main()
