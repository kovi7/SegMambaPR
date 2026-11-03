from __future__ import annotations

import math
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
from torch.utils.data import WeightedRandomSampler

from monai.data import CacheDataset, DataLoader, Dataset, list_data_collate
from monai.transforms import (
    Compose,
    MapTransform,
    EnsureChannelFirstd,
    EnsureTyped,
    LoadImaged,
    MapLabelValued,
    RandCropByLabelClassesd,
    RandCropByPosNegLabeld,
    RandFlipd,
    RandGaussianNoised,
    RandScaleIntensityd,
    RandShiftIntensityd,
    ScaleIntensityRanged,
    SpatialPadd,
)
from monai.transforms.utils import map_classes_to_indices

from .utils import load_json, save_json
from .manifest import (
    HARD_NEGATIVE_TUMOR_TYPES,
    build_panorama_manifest_rows,
    split_train_val_by_patient,
    summarize_manifest,
)


SEX_MAP = {"M": 0.0, "MALE": 0.0, "F": 1.0, "FEMALE": 1.0}
BENIGN_TUMOR_TYPES = HARD_NEGATIVE_TUMOR_TYPES
NO_PDAC_LABELS = {"no_pdac", "none", "normal", "0", "negative"}

DATASET_ROW_KEYS = ("image", "label", "tabular", "tabular_mask", "has_pdac", "case_id")


class ClassIndicesNumpyd(MapTransform):
    """Per-class voxel indices for RandCropByLabelClassesd, computed in numpy (much faster)."""

    def __init__(self, label_key: str, num_classes: int, out_key: str = "label_cls_indices"):
        super().__init__(keys=[label_key])
        self.label_key = label_key
        self.num_classes = num_classes
        self.out_key = out_key

    def __call__(self, data):
        d = dict(data)
        d[self.out_key] = map_classes_to_indices(np.asarray(d[self.label_key]), self.num_classes)
        return d


def project_rows_for_dataset(rows: Sequence[Dict[str, Any]], keys: Sequence[str] = DATASET_ROW_KEYS) -> List[Dict[str, Any]]:
    """Keep only the keys the transforms use; None values cannot be collated."""
    return [{k: r[k] for k in keys if k in r and r[k] is not None} for r in rows]


class TabularEncoderPreprocess:
    """Preprocess age, sex and scanner type without leaking validation/test data."""

    def __init__(self, scanner_min_count: int = 2):
        self.scanner_min_count = int(scanner_min_count)
        self.age_mean = 0.0
        self.age_std = 1.0
        self.scanners: List[str] = []
        self.scanner_to_idx: Dict[str, int] = {}

    @staticmethod
    def _scanner_value(row: Dict[str, Any]) -> str:
        for key in ("scanner_type", "scanner", "scanner_model", "manufacturer", "scanner_name"):
            value = str(row.get(key, "")).strip().lower()
            if value and value not in {"nan", "none", "null"}:
                return value
        return ""

    @staticmethod
    def _is_valid_number(value: Any) -> bool:
        if value is None or value == "":
            return False
        try:
            return not math.isnan(float(value))
        except Exception:
            return False

    def fit(self, rows: Sequence[Dict[str, Any]]) -> "TabularEncoderPreprocess":
        ages: List[float] = []
        scanners: List[str] = []
        for r in rows:
            age = r.get("age", None)
            if self._is_valid_number(age):
                ages.append(float(age))
            sc = self._scanner_value(r)
            if sc:
                scanners.append(sc)
        if ages:
            self.age_mean = float(np.mean(ages))
            self.age_std = float(np.std(ages) + 1e-6)
        counts = Counter(scanners)
        self.scanners = sorted([s for s, c in counts.items() if c >= self.scanner_min_count])
        self.scanner_to_idx = {s: i for i, s in enumerate(self.scanners)}
        return self

    @property
    def dim(self) -> int:
        return 2 + len(self.scanners)

    def state_dict(self) -> Dict[str, Any]:
        return {
            "scanner_min_count": self.scanner_min_count,
            "age_mean": self.age_mean,
            "age_std": self.age_std,
            "scanners": self.scanners,
        }

    @classmethod
    def from_state_dict(cls, state: Dict[str, Any]) -> "TabularEncoderPreprocess":
        obj = cls(scanner_min_count=int(state.get("scanner_min_count", 2)))
        obj.age_mean = float(state.get("age_mean", 0.0))
        obj.age_std = float(state.get("age_std", 1.0))
        obj.scanners = list(state.get("scanners", []))
        obj.scanner_to_idx = {s: i for i, s in enumerate(obj.scanners)}
        return obj

    def transform_one(self, row: Dict[str, Any], force_missing: bool = False) -> Tuple[np.ndarray, np.ndarray]:
        x = np.zeros(self.dim, dtype=np.float32)
        m = np.zeros(self.dim, dtype=np.float32)
        if force_missing:
            return x, m
        age = row.get("age", None)
        if self._is_valid_number(age):
            x[0] = (float(age) - self.age_mean) / self.age_std
            m[0] = 1.0
        sex = str(row.get("sex", row.get("gender", ""))).strip().upper()
        if sex in SEX_MAP:
            x[1] = SEX_MAP[sex]
            m[1] = 1.0
        scanner = self._scanner_value(row)
        if scanner in self.scanner_to_idx:
            idx = 2 + self.scanner_to_idx[scanner]
            x[idx] = 1.0
            m[idx] = 1.0
        return x, m


class TabularTextPreprocess:
    """Tabular encoder backed by precomputed sentence embeddings of age, sex and scanner type.

    The cache comes from ``tools/precompute_tabular_text.py``. Same interface as
    ``TabularEncoderPreprocess``.
    """

    def __init__(self, cache_path: str):
        self.cache_path = str(cache_path)
        self.cache: Dict[str, List[float]] = {}
        self.missing_vec: List[float] = []
        self._dim = 0

    def fit(self, rows: Sequence[Dict[str, Any]]) -> "TabularTextPreprocess":
        data = load_json(self.cache_path)
        self.cache = dict(data["embeddings"])
        self.missing_vec = list(data["missing"])
        self._dim = int(data.get("dim", len(self.missing_vec)))
        case_ids = [r.get("case_id") for r in rows if r.get("case_id") is not None]
        missing = [cid for cid in case_ids if cid not in self.cache]
        if case_ids and missing:
            print(
                f"[tabular_text] Warning: {len(missing)}/{len(case_ids)} cases missing from "
                f"{self.cache_path} (e.g. {missing[:3]}). They will use the 'missing' embedding. "
                "Re-run tools/precompute_tabular_text.py if these are new cases.",
                flush=True,
            )
        return self

    @property
    def dim(self) -> int:
        return self._dim

    def state_dict(self) -> Dict[str, Any]:
        return {"cache_path": self.cache_path, "dim": self._dim}

    @classmethod
    def from_state_dict(cls, state: Dict[str, Any]) -> "TabularTextPreprocess":
        return cls(cache_path=state["cache_path"]).fit([])

    def transform_one(self, row: Dict[str, Any], force_missing: bool = False) -> Tuple[np.ndarray, np.ndarray]:
        if force_missing:
            return np.array(self.missing_vec, dtype=np.float32), np.zeros(1, dtype=np.float32)
        vec = self.cache.get(row.get("case_id"))
        if vec is None:
            return np.array(self.missing_vec, dtype=np.float32), np.zeros(1, dtype=np.float32)
        return np.array(vec, dtype=np.float32), np.ones(1, dtype=np.float32)


def build_tabproc(cfg: Dict[str, Any], train_rows: Sequence[Dict[str, Any]]):
    """Build the tabular encoder selected by ``tabular.mode`` (numeric or text)."""
    tab_cfg = cfg.get("tabular", {})
    mode = str(tab_cfg.get("mode", "numeric")).lower()
    if mode == "text":
        cache_path = tab_cfg.get("text_cache_path")
        if not cache_path:
            raise ValueError("tabular.mode=text requires tabular.text_cache_path (see tools/precompute_tabular_text.py)")
        return TabularTextPreprocess(cache_path=cache_path).fit(train_rows)
    return TabularEncoderPreprocess(scanner_min_count=tab_cfg.get("scanner_min_count", 2)).fit(train_rows)


def tabproc_from_state_dict(mode: str, state: Dict[str, Any]):
    if str(mode).lower() == "text":
        return TabularTextPreprocess.from_state_dict(state)
    return TabularEncoderPreprocess.from_state_dict(state)


def _load_or_build_manifest_rows(cfg: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Load the JSON manifest, or build it from the image/label directories and metadata files."""
    data_cfg = cfg.get("data", {})
    manifest_path = data_cfg.get("manifest")
    if manifest_path and Path(manifest_path).exists():
        rows = [dict(r) for r in load_json(manifest_path)]
    else:
        dataset_name = str(data_cfg.get("dataset", "panorama")).lower()
        if dataset_name in {"pants", "pants_converted", "pants6", "pants_all", "pants28"}:
            from .prepare_pants_manifest import build_converted_pants_manifest_rows

            default_mode = "all" if dataset_name in {"pants_all", "pants28"} else "pants6"
            rows = build_converted_pants_manifest_rows(
                converted_root=data_cfg.get("converted_root", "../data/PanTs"),
                mode=data_cfg.get("pants_mode", default_mode),
                image_dir=data_cfg.get("image_dir"),
                label_dir=data_cfg.get("label_dir"),
                split_json=data_cfg.get("split_json"),
                image_suffix=data_cfg.get("image_suffix", "_0000_trans.nii.gz"),
                label_suffix=data_cfg.get("label_suffix", "_trans.nii.gz"),
                test_split_name=data_cfg.get("test_split_name", "val"),
                verify=bool(data_cfg.get("verify_paths", True)),
                count_labels=bool(data_cfg.get("count_labels", True)),
            )
        else:
            image_dir = data_cfg.get("image_dir") or data_cfg.get("raw_image_dir")
            label_dir = data_cfg.get("label_dir") or data_cfg.get("gt_label_dir")
            if not image_dir or not label_dir:
                raise ValueError(
                    "No usable dataset source. Provide either data.manifest or "
                    "data.image_dir + data.label_dir in the config."
                )
            rows = build_panorama_manifest_rows(
                image_dir=image_dir,
                label_dir=label_dir,
                clinical_xlsx=data_cfg.get("clinical_xlsx"),
                test_list_path=data_cfg.get("test_list_path"),
                dpcg_csv=data_cfg.get("dpcg_csv"),
                verification_csv=data_cfg.get("verification_csv"),
                verification_label_column=data_cfg.get("verification_label_column", "Pancreas_label_v2"),
                image_suffix=data_cfg.get("image_suffix", "_0000_trans.nii.gz"),
                label_suffix=data_cfg.get("label_suffix", "_trans.nii.gz"),
                val_fraction=float(data_cfg.get("val_fraction", 0.15)),
                seed=int(cfg.get("seed", 42)),
                pdac_label=int(cfg.get("labels", {}).get("pdac", 1)),
                infer_pdac_from_mask=bool(data_cfg.get("infer_pdac_from_mask", False)),
                include_test=True,
                require_label=bool(data_cfg.get("require_label", True)),
                assign_split=True,
                exclude_unusable_verification=bool(data_cfg.get("exclude_unusable_verification", False)),
            )
        generated_path = data_cfg.get("generated_manifest") or data_cfg.get("manifest")
        if generated_path:
            save_json(rows, generated_path)
            save_json(summarize_manifest(rows), str(Path(generated_path).with_suffix(".summary.json")))

    # rows without a split get a deterministic patient-level train/val split
    if any("split" not in r or not r.get("split") for r in rows):
        for r in rows:
            if "split" not in r or not r.get("split"):
                r["split"] = "train"
        rows = split_train_val_by_patient(
            rows,
            val_fraction=float(data_cfg.get("val_fraction", 0.15)),
            seed=int(cfg.get("seed", 42)),
        )
    return rows


def _split_rows(cfg_or_manifest: Dict[str, Any] | str | Path, split: str) -> List[Dict[str, Any]]:
    if isinstance(cfg_or_manifest, dict):
        rows = _load_or_build_manifest_rows(cfg_or_manifest)
    else:
        rows = load_json(cfg_or_manifest)
    return [dict(r) for r in rows if r.get("split") == split]


def infer_has_pdac(row: Dict[str, Any]) -> int:
    if "has_pdac" in row and row.get("has_pdac") is not None:
        return int(row["has_pdac"])
    label = str(row.get("label", "")).lower()
    tumor_type = str(row.get("tumor_type", "")).lower()
    if tumor_type in BENIGN_TUMOR_TYPES or int(row.get("hard_negative", 0) or 0):
        return 0
    if label in NO_PDAC_LABELS:
        return 0
    return 0


def prepare_rows(
    cfg: Dict[str, Any],
    split: str,
    tabproc: Optional[Any] = None,
    force_missing: Optional[bool] = None,
) -> Tuple[List[Dict[str, Any]], Any]:
    rows = _split_rows(cfg, split)
    if tabproc is None:
        train_rows = _split_rows(cfg, "train")
        tabproc = build_tabproc(cfg, train_rows)
    if force_missing is None:
        force_missing = bool(cfg.get("tabular", {}).get("force_missing", False))
    for r in rows:
        tab, tab_mask = tabproc.transform_one(r, force_missing=bool(force_missing))
        r["tabular"] = tab
        r["tabular_mask"] = tab_mask
        r["has_pdac"] = infer_has_pdac(r)
    return rows, tabproc


def get_transforms(cfg: Dict[str, Any], train: bool) -> Compose:
    data_cfg = cfg["data"]
    keys = ["image", "label"]
    roi_size = tuple(data_cfg.get("roi_size", [128, 128, 128]))
    hu_min = float(data_cfg.get("hu_min", -175.0))
    hu_max = float(data_cfg.get("hu_max", 250.0))
    # rescale after the crop, so the cache holds the raw image and a uint8 label
    cache_raw = bool(data_cfg.get("cache_raw", True)) and train
    scale = ScaleIntensityRanged(
        keys=["image"], a_min=hu_min, a_max=hu_max, b_min=0.0, b_max=1.0, clip=True,
    )
    xforms = [
        LoadImaged(keys=keys, image_only=False),
        EnsureChannelFirstd(keys=keys),
    ]
    if not cache_raw:
        xforms.append(scale)
    label_remap = data_cfg.get("label_remap", None)
    if label_remap is not None:
        orig_labels = label_remap.get("orig_labels", label_remap.get("from", []))
        target_labels = label_remap.get("target_labels", label_remap.get("to", []))
        if len(orig_labels) != len(target_labels) or len(orig_labels) == 0:
            raise ValueError("data.label_remap requires non-empty orig_labels/from and target_labels/to with equal length.")
        xforms.append(
            MapLabelValued(
                keys="label",
                orig_labels=[int(x) for x in orig_labels],
                target_labels=[int(x) for x in target_labels],
                dtype=np.uint8 if cache_raw else np.float32,
            )
        )
    if train:
        if cache_raw:
            xforms.append(SpatialPadd(keys=["image"], spatial_size=roi_size, value=hu_min))
            xforms.append(SpatialPadd(keys=["label"], spatial_size=roi_size, value=0))
        else:
            xforms.append(SpatialPadd(keys=keys, spatial_size=roi_size))
        crop_mode = str(data_cfg.get("crop_mode", "posneg")).lower()
        if crop_mode == "label_classes":
            ratios = data_cfg.get("class_crop_ratios")
            if ratios is None or (isinstance(ratios, str) and ratios.strip().lower() in {"none", "null", "~", ""}):
                raise ValueError("data.crop_mode=label_classes requires data.class_crop_ratios. Use crop_mode=posneg for the default sampler.")
            n_cls = int(cfg.get("num_classes", cfg.get("model", {}).get("out_channels", 6)))
            xforms.append(ClassIndicesNumpyd(label_key="label", num_classes=n_cls))
            xforms.append(
                RandCropByLabelClassesd(
                    keys=keys,
                    label_key="label",
                    spatial_size=roi_size,
                    ratios=ratios,
                    num_classes=n_cls,
                    indices_key="label_cls_indices",
                    num_samples=int(data_cfg.get("num_samples", 2)),
                    allow_smaller=False,
                    warn=False,  # classes absent from a case are common on PanTS
                )
            )
        elif crop_mode == "posneg":
            xforms.append(
                RandCropByPosNegLabeld(
                    keys=keys,
                    label_key="label",
                    spatial_size=roi_size,
                    pos=float(data_cfg.get("crop_pos", 2.0)),
                    neg=float(data_cfg.get("crop_neg", 1.0)),
                    num_samples=int(data_cfg.get("num_samples", 2)),
                    image_key="image",
                    image_threshold=(hu_min if cache_raw
                                     else float(data_cfg.get("image_threshold", 0.0))),
                    allow_smaller=False,
                )
            )
        else:
            raise ValueError(f"Unknown data.crop_mode={crop_mode}. Use posneg or label_classes.")
        xforms += [
            RandFlipd(keys=keys, prob=0.5, spatial_axis=0),
            RandFlipd(keys=keys, prob=0.5, spatial_axis=1),
            RandFlipd(keys=keys, prob=0.5, spatial_axis=2),
        ]
        if cache_raw:
            xforms.append(scale)
        xforms += [
            RandGaussianNoised(keys=["image"], prob=0.10, mean=0.0, std=0.02),
            RandScaleIntensityd(keys=["image"], factors=0.10, prob=0.20),
            RandShiftIntensityd(keys=["image"], offsets=0.10, prob=0.20),
        ]
    xforms += [EnsureTyped(keys=keys + ["tabular", "tabular_mask", "has_pdac"])]
    return Compose(xforms)


def build_sampler(rows: Sequence[Dict[str, Any]], cfg: Dict[str, Any]) -> Optional[WeightedRandomSampler]:
    sampler_cfg = cfg.get("sampler", {})
    if not sampler_cfg.get("enabled", False):
        return None
    weights: List[float] = []
    w_no = float(sampler_cfg.get("no_pdac_weight", 1.0))
    w_pdac = float(sampler_cfg.get("pdac_weight", 1.0))
    w_benign = float(sampler_cfg.get("benign_weight", max(w_no, 1.0)))
    for r in rows:
        tumor_type = str(r.get("tumor_type", "")).lower()
        has_pdac = infer_has_pdac(r)
        is_hard_negative = int(r.get("is_hard_negative", r.get("hard_negative", 0)) or (tumor_type in BENIGN_TUMOR_TYPES and has_pdac == 0))
        if is_hard_negative:
            weights.append(w_benign)  # hard negative: a non-PDAC lesion
        elif has_pdac == 0:
            weights.append(w_no)
        else:
            weights.append(w_pdac)
    multiplier = float(sampler_cfg.get("epoch_multiplier", 1.0))
    n = max(1, int(math.ceil(len(rows) * multiplier)))
    return WeightedRandomSampler(weights=weights, num_samples=n, replacement=True)


def build_loaders(cfg: Dict[str, Any]):
    train_rows, tabproc = prepare_rows(cfg, "train", None, force_missing=cfg.get("tabular", {}).get("force_missing", False))
    val_rows, _ = prepare_rows(cfg, "val", tabproc, force_missing=cfg.get("val", {}).get("force_missing_tabular", False))
    train_tf = get_transforms(cfg, train=True)
    val_tf = get_transforms(cfg, train=False)
    train_cache_rate = float(cfg["data"].get("cache_rate", 0.0))
    val_cache_rate = float(cfg["data"].get("val_cache_rate", 0.0))
    train_ds_rows = project_rows_for_dataset(train_rows)
    val_ds_rows = project_rows_for_dataset(val_rows)
    cache_workers = int(cfg["data"].get("cache_num_workers", cfg["data"].get("num_workers", 4)))
    train_ds = (
        CacheDataset(data=train_ds_rows, transform=train_tf, cache_rate=train_cache_rate,
                     num_workers=cache_workers)
        if train_cache_rate > 0
        else Dataset(data=train_ds_rows, transform=train_tf)
    )
    val_ds = (
        CacheDataset(data=val_ds_rows, transform=val_tf, cache_rate=val_cache_rate,
                     num_workers=cache_workers)
        if val_cache_rate > 0
        else Dataset(data=val_ds_rows, transform=val_tf)
    )
    sampler = build_sampler(train_rows, cfg)
    pin = bool(cfg["data"].get("pin_memory", True))
    train_loader = DataLoader(
        train_ds,
        batch_size=int(cfg["data"].get("batch_size", 1)),
        shuffle=sampler is None,
        sampler=sampler,
        num_workers=int(cfg["data"].get("num_workers", 4)),
        pin_memory=pin,
        collate_fn=list_data_collate,
        persistent_workers=int(cfg["data"].get("num_workers", 4)) > 0,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=1,
        shuffle=False,
        num_workers=int(cfg["data"].get("num_workers", 4)),
        pin_memory=pin,
        collate_fn=list_data_collate,
        persistent_workers=int(cfg["data"].get("num_workers", 4)) > 0,
    )
    return train_loader, val_loader, tabproc
