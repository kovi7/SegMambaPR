from __future__ import annotations

import json
import os
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

try:
    import nibabel as nib
except Exception:
    nib = None

NO_PDAC_LABELS = {"no-pdac", "non-pdac", "no_pdac", "non_pdac", "normal", "none", "0", "negative", "control"}
PDAC_LABELS = {"pdac", "pancreatic_ductal_adenocarcinoma", "pancreatic ductal adenocarcinoma", "1", "positive"}
AMBIGUOUS_PDAC_LABEL_TOKENS = {"pdac"}
HARD_NEGATIVE_TUMOR_TYPES = {"ipmn", "pnet", "mcn", "scn", "cyst", "spt", "sppn", "aov", "other_tumor", "mixed_non_pdac", "pzt", "ozt", "cp", "tumor_or_cyst"}
UNUSABLE_TUMOR_TYPES = {"no_pancreas", "pancreas_not_visible"}
# verification labels allowed to override a PDAC label to non-PDAC (solid tumours excluded)
BENIGN_VERIFICATION_OVERRIDE_TYPES = {"none", "ipmn", "mcn", "scn", "cyst", "no_pancreas", "pancreas_not_visible"}


def _clean(value: Any) -> str:
    if value is None:
        return ""
    try:
        if pd.isna(value):
            return ""
    except Exception:
        pass
    return re.sub(r"\s+", " ", str(value).strip())


def normalize_case_id(raw: Any) -> str:
    s = _clean(raw)
    if not s:
        return ""
    s = os.path.basename(s)
    for suffix in [".nii.gz", ".nii", "_0000_trans", "_0001_trans", "_0000", "_0001", "_trans"]:
        if s.endswith(suffix):
            s = s[: -len(suffix)]
    return s


def default_patient_id(case_id: str) -> str:
    return str(case_id).split("_")[0] if "_" in str(case_id) else str(case_id)


def parse_age_years(value: Any) -> Optional[float]:
    txt = _clean(value)
    if not txt:
        return None
    m = re.search(r"(\d+(?:\.\d+)?)", txt)
    return float(m.group(1)) if m else None


def normalize_sex(value: Any) -> str:
    s = _clean(value).upper()
    if s in {"M", "MALE"}:
        return "M"
    if s in {"F", "FEMALE"}:
        return "F"
    return ""


def normalize_scanner(value: Any) -> str:
    s = _clean(value)
    return "" if not s or s.lower() in {"nan", "none", "null", "0"} else s


def _label_key(value: Any) -> str:
    s = _clean(value).lower().replace("(?)", "?").replace("-", "_")
    s = re.sub(r"\s+", "_", s)
    return s.strip("_")


def classify_verification_label(value: Any, ambiguous_pdac_as_pdac: bool = False) -> Dict[str, Any]:
    """Normalise a label of PANORAMA_verification.csv into PDAC status, subtype and hard-negative flag.

    Ambiguous labels containing PDAC (e.g. PNET/PDAC?) are not treated as PDAC by default.
    """
    raw = _clean(value)
    key = _label_key(raw)
    key_no_q = key.replace("?", "")
    out: Dict[str, Any] = {
        "verification_label": raw,
        "verification_has_pdac": None,
        "tumor_type": "unknown",
        "tumor_group": "unknown",
        "hard_negative": 0,
        "is_non_pdac_lesion": 0,
        "diagnosis_uncertain": int("?" in key or not key),
    }
    if not key:
        return out
    if key_no_q in {"control", "normal", "none", "no_pdac", "non_pdac"}:
        out.update({"verification_has_pdac": 0, "tumor_type": "none", "tumor_group": "control"})
        return out
    if key_no_q == "pdac":
        out.update({"verification_has_pdac": 1, "tumor_type": "pdac", "tumor_group": "pdac"})
        return out
    if "pdac" in key_no_q:
        out.update({
            "verification_has_pdac": 1 if ambiguous_pdac_as_pdac else None,
            "tumor_type": "pdac_ambiguous",
            "tumor_group": "pdac",
            "hard_negative": 0,
            "is_non_pdac_lesion": 0,
            "diagnosis_uncertain": 1,
        })
        return out

    patterns = [
        ("ipmn", "ipmn", "benign_or_low_grade"),
        ("pnet", "pnet", "non_pdac_tumor"),
        ("scn", "scn", "benign_or_low_grade"),
        ("mcn", "mcn", "benign_or_low_grade"),
        ("cyst", "cyst", "benign_or_low_grade"),
        ("aov", "aov", "other_non_pdac"),
        ("spt", "spt", "non_pdac_tumor"),
        ("sppn", "sppn", "non_pdac_tumor"),
        ("tumor/cyst", "tumor_or_cyst", "other_non_pdac"),
        ("tumor", "other_tumor", "other_non_pdac"),
        ("pzt", "pzt", "inflammatory"),
        ("ozt", "ozt", "inflammatory"),
        ("cp", "cp", "inflammatory"),
        ("postsurgery", "postsurgery", "special"),
        ("brak_trzustki", "no_pancreas", "special"),
        ("niewidoczny", "pancreas_not_visible", "special"),
    ]
    hits = [(typ, group) for token, typ, group in patterns if token in key_no_q]
    if hits:
        tumor_types = sorted({t for t, _ in hits})
        tumor_groups = sorted({g for _, g in hits})
        tumor_type = tumor_types[0] if len(tumor_types) == 1 else "mixed_non_pdac"
        tumor_group = tumor_groups[0] if len(tumor_groups) == 1 else "mixed_non_pdac"
        out.update({
            "verification_has_pdac": 0,
            "tumor_type": tumor_type,
            "tumor_group": tumor_group,
            "hard_negative": int(tumor_group != "special"),
            "is_non_pdac_lesion": int(tumor_group != "special"),
            "diagnosis_uncertain": int(out["diagnosis_uncertain"] or tumor_type in {"other_tumor", "tumor_or_cyst", "mixed_non_pdac"}),
        })
        return out

    out.update({
        "verification_has_pdac": 0,
        "tumor_type": key_no_q,
        "tumor_group": "unknown_or_other",
        "hard_negative": 1,
        "is_non_pdac_lesion": 1,
        "diagnosis_uncertain": 1,
    })
    return out


def _read_csv_auto(path: str | Path) -> pd.DataFrame:
    p = Path(path)
    return pd.read_csv(p, sep="\t") if p.suffix.lower() == ".tsv" else pd.read_csv(p, sep=None, engine="python")


def load_test_ids(path: Optional[str | Path]) -> set[str]:
    if not path:
        return set()
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"test_list_path does not exist: {p}")
    return {cid for cid in (normalize_case_id(line) for line in p.read_text(encoding="utf-8").splitlines()) if cid}


def load_clinical_rows(path: Optional[str | Path]) -> Dict[str, Dict[str, Any]]:
    if not path:
        return {}
    p = Path(path)
    if not p.exists():
        print(f"[manifest] Warning: clinical_xlsx not found: {p}")
        return {}
    df = pd.read_excel(p)
    id_col = next((c for c in ["PANORAMA_study_id", "case_id", "study_id", "Original filename"] if c in df.columns), None)
    if id_col is None:
        raise ValueError(f"Clinical file needs an id column. Columns: {list(df.columns)}")
    out: Dict[str, Dict[str, Any]] = {}
    for _, r in df.iterrows():
        row = r.to_dict()
        cid = normalize_case_id(row.get(id_col, ""))
        if not cid:
            continue
        pid = _clean(row.get("PANORAMA_patient_id", row.get("Patient ID", row.get("patient_id", default_patient_id(cid)))))
        out[cid] = {
            "patient_id": pid or default_patient_id(cid),
            "age": parse_age_years(row.get("patient_age", row.get("age", row.get("Age", None)))),
            "sex": normalize_sex(row.get("patient_sex", row.get("sex", row.get("gender", row.get("Sex", ""))))),
            "scanner_type": normalize_scanner(row.get("scanner", row.get("scanner_type", row.get("manufacturer", row.get("Scanner", ""))))),
            "clinical_label": _clean(row.get("label", row.get("diagnosis", row.get("Pancreas_label_v2", row.get("Pancreas_label", ""))))),
        }
    return out


def load_dpcg_rows(path: Optional[str | Path]) -> Dict[str, Dict[str, Any]]:
    if not path:
        return {}
    p = Path(path)
    if not p.exists():
        print(f"[manifest] Warning: dpcg_csv not found: {p}")
        return {}
    df = _read_csv_auto(p)
    if "case_id" not in df.columns:
        raise ValueError(f"DPCG file must contain case_id. Columns: {list(df.columns)}")
    out: Dict[str, Dict[str, Any]] = {}
    for _, r in df.iterrows():
        row = r.to_dict()
        cid = normalize_case_id(row.get("case_id", ""))
        if not cid:
            continue

        def to_int(name: str) -> Optional[int]:
            if name not in row or pd.isna(row.get(name)):
                return None
            return int(float(row.get(name)))

        is_res = to_int("is_resectable")
        is_br = to_int("is_borderline")
        is_la = to_int("is_locally_advanced")

        has_pdac = to_int("has_pdac")
        if has_pdac is None:
            # a positive resectability category implies PDAC
            has_pdac = int((is_res or 0) or (is_br or 0) or (is_la or 0))
        out[cid] = {
            "has_pdac": int(has_pdac),
            "is_resectable": is_res,
            "is_borderline": is_br,
            "is_locally_advanced": is_la,
        }
    return out


def load_verification_rows(path: Optional[str | Path], label_column: str = "Pancreas_label_v2", ambiguous_pdac_as_pdac: bool = False) -> Dict[str, Dict[str, Any]]:
    if not path:
        return {}
    p = Path(path)
    if not p.exists():
        print(f"[manifest] Warning: verification_csv not found: {p}")
        return {}
    df = _read_csv_auto(p)
    case_col = next((c for c in ["Original filename", "case_id", "study_id", "PANORAMA_study_id"] if c in df.columns), None)
    if case_col is None:
        raise ValueError(f"Verification CSV needs Original filename/case_id. Columns: {list(df.columns)}")
    if label_column in df.columns:
        label_col = label_column
    elif "Pancreas_label_v2" in df.columns:
        label_col = "Pancreas_label_v2"
    elif "Pancreas_label" in df.columns:
        label_col = "Pancreas_label"
    else:
        raise ValueError(f"Verification CSV needs Pancreas_label_v2 or Pancreas_label. Columns: {list(df.columns)}")
    out: Dict[str, Dict[str, Any]] = {}
    for _, r in df.iterrows():
        row = r.to_dict()
        cid = normalize_case_id(row.get(case_col, ""))
        if not cid:
            continue
        primary_label = row.get(label_col, "")
        original_label = row.get("Pancreas_label", "")
        # keep ambiguous PDAC information from the original label column
        combined_for_pdac = f"{primary_label} {original_label}"
        info = classify_verification_label(primary_label, ambiguous_pdac_as_pdac=ambiguous_pdac_as_pdac)
        if ambiguous_pdac_as_pdac and "pdac" in _label_key(combined_for_pdac) and info.get("verification_has_pdac") != 1:
            info.update({
                "verification_has_pdac": 1,
                "tumor_type": "pdac_ambiguous",
                "tumor_group": "pdac",
                "hard_negative": 0,
                "is_non_pdac_lesion": 0,
                "diagnosis_uncertain": 1,
            })
        original_clean = _clean(row.get("Pancreas_label", ""))
        original_key = _label_key(original_clean)
        primary_key = _label_key(primary_label)
        if original_key and original_key != primary_key and ("?" in original_key or "pdac" in original_key):
            info["diagnosis_uncertain"] = 1
        info["verification_label_source"] = label_col
        info["verification_label_original"] = original_clean
        info["verification_label_v2"] = _clean(row.get("Pancreas_label_v2", ""))
        info["verification_dataset"] = _clean(row.get("Dataset", ""))
        pid = _clean(row.get("Patient ID", ""))
        if pid:
            info["verification_patient_id"] = pid
        out[cid] = info
    return out


def discover_image_label_pairs(image_dir: str | Path, label_dir: Optional[str | Path], image_suffix: str = "_0000_trans.nii.gz", label_suffix: str = "_trans.nii.gz", require_label: bool = False) -> List[Dict[str, Any]]:
    img_dir = Path(image_dir)
    if not img_dir.exists():
        raise FileNotFoundError(f"image_dir does not exist: {img_dir}")
    lab_dir = Path(label_dir) if label_dir else None
    if require_label and (lab_dir is None or not lab_dir.exists()):
        raise FileNotFoundError(f"label_dir does not exist: {lab_dir}")
    files = sorted(img_dir.glob(f"*{image_suffix}"))
    if not files:
        files = sorted(img_dir.glob("*.nii.gz")) + sorted(img_dir.glob("*.nii"))
    out: List[Dict[str, Any]] = []
    for img in files:
        cid = normalize_case_id(img.name)
        label_path: Optional[Path] = None
        if lab_dir is not None:
            candidate = lab_dir / f"{cid}{label_suffix}"
            if candidate.exists():
                label_path = candidate
            else:
                alternatives = list(lab_dir.glob(f"{cid}*.nii*"))
                label_path = alternatives[0] if alternatives else None
        if require_label and label_path is None:
            continue
        out.append({"case_id": cid, "image": str(img), "label": str(label_path) if label_path else None})
    return out


def has_label_value(label_path: str | Path, value: int) -> Optional[bool]:
    if nib is None:
        raise RuntimeError("nibabel is required for --infer_pdac_from_mask")
    p = Path(label_path)
    if not p.exists():
        return None
    arr = np.asanyarray(nib.load(str(p)).dataobj)
    return bool(np.any(arr == int(value)))


def infer_has_pdac_from_sources(label_path: Optional[str | Path], clinical_row: Dict[str, Any], dpcg_row: Dict[str, Any], verification_row: Dict[str, Any], pdac_label: int = 1, infer_from_mask: bool = False) -> tuple[Optional[int], str, int]:
    """Patient-level PDAC label, its source and a conflict flag.

    1. clinical_information.xlsx is the primary source.
    2. dpcg_labels.csv is the fallback when the clinical label is missing.
    3. PANORAMA_verification.csv gives the subtype and hard negatives.
    4. A PDAC label is overridden to non-PDAC only when verification gives a clearly benign
       or no-tumour class; the case is then marked as a conflict.
    """
    ver_type = str(verification_row.get("tumor_type", "")).lower()
    ver_group = str(verification_row.get("tumor_group", "")).lower()

    clinical_label = _label_key(clinical_row.get("clinical_label", clinical_row.get("label", "")))
    clinical_has: Optional[int] = None
    if clinical_label in PDAC_LABELS or clinical_label == "pdac":
        clinical_has = 1
    elif clinical_label in NO_PDAC_LABELS:
        clinical_has = 0

    dpcg_has = dpcg_row.get("has_pdac", None)
    dpcg_has = None if dpcg_has is None else int(dpcg_has)

    primary_has = clinical_has if clinical_has is not None else dpcg_has
    primary_source = "clinical_information" if clinical_has is not None else ("dpcg_labels" if dpcg_has is not None else "")

    if primary_has == 1 and (ver_type in BENIGN_VERIFICATION_OVERRIDE_TYPES or ver_group in {"control", "benign_or_low_grade"} and ver_type in {"ipmn", "mcn", "scn", "cyst"}):
        return 0, f"{primary_source}_overridden_by_benign_verification", 1

    if primary_has is not None:
        return int(primary_has), primary_source, 0

    ver_has = verification_row.get("verification_has_pdac", None)
    if ver_has is not None:
        return int(ver_has), "verification_fallback", 0

    if infer_from_mask and label_path:
        present = has_label_value(label_path, pdac_label)
        if present is not None:
            return int(present), "mask", 0

    return None, "default_no_pdac", 0

def _strat_label(row: Dict[str, Any]) -> str:
    if str(row.get("split", "")) == "test":
        return "test"
    if int(row.get("has_pdac", 0) or 0) == 0:
        return "hard_negative" if int(row.get("hard_negative", 0) or 0) else "no_pdac"
    return "pdac"


def split_train_val_by_patient(rows: List[Dict[str, Any]], val_fraction: float = 0.15, seed: int = 42) -> List[Dict[str, Any]]:
    rng = np.random.default_rng(int(seed))
    test_rows = [r for r in rows if r.get("split") == "test"]
    pool = [r for r in rows if r.get("split") != "test"]
    patient_to_rows: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for row in pool:
        pid = str(row.get("patient_id") or default_patient_id(str(row["case_id"])))
        patient_to_rows[pid].append(row)
    patient_ids = sorted(patient_to_rows.keys())
    if len(patient_ids) < 2:
        raise ValueError("Cannot create train/val split from fewer than 2 patients/cases.")
    priority = ["pdac", "hard_negative", "no_pdac"]
    patient_to_label: Dict[str, str] = {}
    for pid, prs in patient_to_rows.items():
        labels = [_strat_label(r) for r in prs]
        patient_to_label[pid] = next((lab for lab in priority if lab in labels), "no_pdac")
    val_pids: set[str] = set()
    for lab in sorted(set(patient_to_label.values())):
        pids = [pid for pid in patient_ids if patient_to_label[pid] == lab]
        rng.shuffle(pids)
        if len(pids) > 1:
            n_val = max(1, min(int(round(float(val_fraction) * len(pids))), len(pids) - 1))
            val_pids.update(pids[:n_val])
    desired = max(1, min(int(round(float(val_fraction) * len(patient_ids))), len(patient_ids) - 1))
    if len(val_pids) < desired:
        remaining = [pid for pid in patient_ids if pid not in val_pids]
        rng.shuffle(remaining)
        val_pids.update(remaining[: desired - len(val_pids)])
    elif len(val_pids) > desired:
        val_list = sorted(val_pids)
        rng.shuffle(val_list)
        val_pids = set(val_list[:desired])
    for row in pool:
        row["split"] = "val" if str(row.get("patient_id")) in val_pids else "train"
    return pool + test_rows


def build_panorama_manifest_rows(image_dir: str | Path, label_dir: str | Path, clinical_xlsx: Optional[str | Path] = None, test_list_path: Optional[str | Path] = None, dpcg_csv: Optional[str | Path] = None, verification_csv: Optional[str | Path] = None, verification_label_column: str = "Pancreas_label_v2", image_suffix: str = "_0000_trans.nii.gz", label_suffix: str = "_trans.nii.gz", val_fraction: float = 0.15, seed: int = 42, pdac_label: int = 1, infer_pdac_from_mask: bool = False, include_test: bool = True, require_label: bool = True, assign_split: bool = True, exclude_unusable_verification: bool = False, ambiguous_pdac_as_pdac: bool = False) -> List[Dict[str, Any]]:
    clinical = load_clinical_rows(clinical_xlsx)
    dpcg = load_dpcg_rows(dpcg_csv)
    verification = load_verification_rows(verification_csv, label_column=verification_label_column, ambiguous_pdac_as_pdac=ambiguous_pdac_as_pdac)
    test_ids = load_test_ids(test_list_path)
    pairs = discover_image_label_pairs(image_dir, label_dir, image_suffix, label_suffix, require_label=require_label)
    rows: List[Dict[str, Any]] = []
    for pair in pairs:
        cid = str(pair["case_id"])
        clin = clinical.get(cid, {})
        dpcg_row = dpcg.get(cid, {})
        ver = verification.get(cid, {})
        if exclude_unusable_verification and ver.get("tumor_type") in UNUSABLE_TUMOR_TYPES:
            continue
        has_pdac, has_pdac_source, benign_override_conflict = infer_has_pdac_from_sources(pair.get("label"), clin, dpcg_row, ver, pdac_label=pdac_label, infer_from_mask=infer_pdac_from_mask)
        if has_pdac is None:
            has_pdac = 0
        split = "test" if cid in test_ids else "train"
        if split == "test" and not include_test:
            continue
        # subtype metadata must agree with the final has_pdac label
        ver_type = str(ver.get("tumor_type", ""))
        ver_group = str(ver.get("tumor_group", ""))
        if int(has_pdac):
            tumor_type = "pdac"
            tumor_group = "pdac"
            hard_negative = 0
        else:
            if ver_type in {"pdac", "pdac_ambiguous"} or ver_group == "pdac":
                tumor_type = "conflict_verification_pdac"
                tumor_group = "conflict"
                hard_negative = 0
            else:
                tumor_type = ver_type or "none"
                tumor_group = ver_group or "control"
                hard_negative = int(ver.get("hard_negative", 0) or tumor_type in HARD_NEGATIVE_TUMOR_TYPES)
        dpcg_has = dpcg_row.get("has_pdac", None)
        ver_has = ver.get("verification_has_pdac", None)
        clinical_has = 1 if _label_key(clin.get("clinical_label", "")) in PDAC_LABELS else (0 if _label_key(clin.get("clinical_label", "")) in NO_PDAC_LABELS else None)
        label_conflict = int(
            benign_override_conflict
            or (clinical_has is not None and ver_has is not None and int(clinical_has) != int(ver_has))
            or (clinical_has is not None and dpcg_has is not None and int(clinical_has) != int(dpcg_has))
        )
        row: Dict[str, Any] = {
            "case_id": cid,
            "patient_id": str(ver.get("verification_patient_id", clin.get("patient_id", default_patient_id(cid)))),
            "image": pair["image"],
            "label": pair.get("label"),
            "split": split,
            "has_pdac": int(has_pdac),
            "tumor_type": tumor_type,
            "tumor_group": tumor_group,
            "hard_negative": hard_negative,
            "is_non_pdac_lesion": int((not int(has_pdac)) and (hard_negative or tumor_group not in {"control", "special"})),
            "diagnosis_uncertain": int(ver.get("diagnosis_uncertain", 0) or 0),
            "label_conflict": label_conflict,
            "age": clin.get("age", None),
            "sex": clin.get("sex", ""),
            "scanner_type": clin.get("scanner_type", ""),
            "clinical_label": clin.get("clinical_label", ""),
            "has_pdac_source": has_pdac_source,
            "verification_tumor_type": ver.get("tumor_type", ""),
            "verification_tumor_group": ver.get("tumor_group", ""),
            "verification_label": ver.get("verification_label", ""),
            "verification_label_source": ver.get("verification_label_source", ""),
            "verification_label_original": ver.get("verification_label_original", ""),
            "verification_label_v2": ver.get("verification_label_v2", ""),
            "verification_dataset": ver.get("verification_dataset", ""),
        }
        rows.append(row)
    if assign_split:
        rows = split_train_val_by_patient(rows, val_fraction=val_fraction, seed=seed)
    return sorted(rows, key=lambda r: (str(r.get("split", "")), str(r.get("case_id", ""))))


def summarize_manifest(rows: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    by_split = Counter(str(r.get("split", "")) for r in rows)
    by_split_pdac: Dict[str, Any] = {}
    for split in sorted(by_split):
        split_rows = [r for r in rows if str(r.get("split", "")) == split]
        by_split_pdac[split] = {
            "n": len(split_rows),
            "has_pdac": int(sum(int(r.get("has_pdac", 0) or 0) for r in split_rows)),
            "no_pdac": int(sum(1 - int(r.get("has_pdac", 0) or 0) for r in split_rows)),
            "hard_negative": int(sum(int(r.get("hard_negative", 0) or 0) for r in split_rows)),
            "non_pdac_lesion": int(sum(int(r.get("is_non_pdac_lesion", 0) or 0) for r in split_rows)),
            "patients": len(set(str(r.get("patient_id", r.get("case_id"))) for r in split_rows)),
            "label_conflicts": int(sum(int(r.get("label_conflict", 0) or 0) for r in split_rows)),
            "diagnosis_uncertain": int(sum(int(r.get("diagnosis_uncertain", 0) or 0) for r in split_rows)),
            "tumor_type": dict(Counter(str(r.get("tumor_type", "unknown")) for r in split_rows)),
            "has_pdac_source": dict(Counter(str(r.get("has_pdac_source", "unknown")) for r in split_rows)),
            "tumor_group": dict(Counter(str(r.get("tumor_group", "unknown")) for r in split_rows)),
        }
    return {
        "total": len(rows),
        "by_split": dict(by_split),
        "by_split_pdac": by_split_pdac,
        "by_tumor_type": dict(Counter(str(r.get("tumor_type", "unknown")) for r in rows)),
        "by_tumor_group": dict(Counter(str(r.get("tumor_group", "unknown")) for r in rows)),
        "by_verification_label": dict(Counter(str(r.get("verification_label", "")) for r in rows if r.get("verification_label"))),
        "by_has_pdac_source": dict(Counter(str(r.get("has_pdac_source", "unknown")) for r in rows)),
    }


def save_manifest(rows: Sequence[Dict[str, Any]], out_path: str | Path) -> None:
    p = Path(out_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8") as f:
        json.dump(list(rows), f, indent=2, sort_keys=True)
