from __future__ import annotations

import argparse
import itertools
from pathlib import Path
from typing import Any, Dict

import nibabel as nib
import numpy as np
import torch
from monai.data import DataLoader, Dataset, list_data_collate
from monai.inferers import sliding_window_inference

from .validation import sliding_window_kwargs
from monai.transforms import Compose, EnsureChannelFirstd, EnsureTyped, LoadImaged, Orientationd, ScaleIntensityRanged, Spacingd

from .config import load_config
from .data import prepare_rows, project_rows_for_dataset, tabproc_from_state_dict
from .models.factory import build_model
from .utils import ensure_dir, load_checkpoint, save_json


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--split", default="test")
    p.add_argument("--output_dir", default=None)
    p.add_argument("--force_missing_tabular", action="store_true")
    p.add_argument("--save_probs", action="store_true")
    p.add_argument("--save_prob_classes", default=None,
                   help="Comma-separated class indices to save as probability maps, e.g. 1 for PDAC.")
    p.add_argument("--allow_cpu", action="store_true", help="Allow CPU inference (configs require CUDA).")
    p.add_argument("--image_dir", default=None, help="Optional raw/original image directory for inference. Overrides infer.image_dir/raw_image_dir.")
    p.add_argument("--image_suffix", default=None, help="Optional suffix for raw image files, e.g. _0000.nii.gz or .nii.gz.")
    p.add_argument("--converted_input", action="store_true", help="Treat input images as already converted; disables raw Orientationd+Spacingd preprocessing.")
    p.add_argument("--tta_flips", default=None,
                   help="Flip test-time augmentation over the given spatial axes, e.g. 0,1,2 "
                        "(all 2^k combinations, probabilities averaged). Off by default.")
    return p.parse_args()


def _flip_sets(spec):
    """Tensor dims to flip in each TTA pass; [()] when TTA is off."""
    if not spec:
        return [()]
    axes = sorted({int(a) for a in str(spec).split(",") if a.strip() != ""})
    if not axes or any(a not in (0, 1, 2) for a in axes):
        raise ValueError(f"--tta_flips takes spatial axes from 0,1,2, got {spec!r}")
    dims = [a + 2 for a in axes]
    return [tuple(c) for r in range(len(dims) + 1) for c in itertools.combinations(dims, r)]


def _affine_from_batch(batch: Dict[str, Any]) -> np.ndarray:
    image = batch["image"]
    if hasattr(image, "meta") and "affine" in image.meta:
        aff = image.meta["affine"]
        if torch.is_tensor(aff):
            aff = aff[0] if aff.ndim == 3 else aff
            return aff.detach().cpu().numpy()
    meta = batch.get("image_meta_dict", {})
    if isinstance(meta, dict) and "affine" in meta:
        aff = meta["affine"]
        if torch.is_tensor(aff):
            aff = aff[0]
        elif isinstance(aff, (list, tuple)):
            aff = aff[0]
        return np.asarray(aff)
    return np.eye(4)



AXIS_LABELS = (("L", "R"), ("P", "A"), ("I", "S"))


def _replace_with_raw_images(rows: list[Dict[str, Any]], cfg: Dict[str, Any]) -> list[Dict[str, Any]]:
    """Point each row to the original CT in ``infer.image_dir``, keeping the manifest metadata."""
    infer_cfg = cfg.get("infer", {})
    image_dir = infer_cfg.get("image_dir") or infer_cfg.get("raw_image_dir")
    if not image_dir:
        return rows
    image_dir = Path(str(image_dir))
    primary_suffix = infer_cfg.get("image_suffix") or infer_cfg.get("raw_image_suffix") or "_0000.nii.gz"
    fallback_suffixes = infer_cfg.get("fallback_image_suffixes", [primary_suffix, ".nii.gz", "_0000.nii.gz"])
    suffixes = []
    for suffix in [primary_suffix, *fallback_suffixes]:
        if suffix not in suffixes:
            suffixes.append(str(suffix))
    out = []
    missing = []
    for r in rows:
        rr = dict(r)
        case_id = str(rr.get("case_id"))
        found = None
        for suffix in suffixes:
            cand = image_dir / f"{case_id}{suffix}"
            if cand.exists():
                found = cand
                break
        if found is None:
            missing.append(case_id)
        else:
            rr["image"] = str(found)
            rr["image_source"] = "raw_inference"
        out.append(rr)
    if missing:
        raise FileNotFoundError(
            f"Could not find raw inference images for {len(missing)} cases in {image_dir}. "
            f"Tried suffixes={suffixes}. First missing cases={missing[:10]}"
        )
    return out

def main():
    args = parse_args()
    cfg = load_config(args.config)
    try:
        ckpt_meta = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    except TypeError:
        ckpt_meta = torch.load(args.checkpoint, map_location="cpu")
    if "config" in ckpt_meta:
        # data paths from --config, model and tabular settings from the checkpoint
        trained_cfg = ckpt_meta["config"]
        cfg["model"] = trained_cfg.get("model", cfg.get("model", {}))
        cfg["num_classes"] = trained_cfg.get("num_classes", cfg.get("num_classes", 6))
        cfg["labels"] = trained_cfg.get("labels", cfg.get("labels", {}))
        cfg.setdefault("tabular", {}).update(trained_cfg.get("tabular", {}))
    if args.force_missing_tabular:
        cfg.setdefault("tabular", {})["force_missing"] = True
    if args.image_dir is not None:
        cfg.setdefault("infer", {})["image_dir"] = args.image_dir
    if args.image_suffix is not None:
        cfg.setdefault("infer", {})["image_suffix"] = args.image_suffix
    if args.converted_input:
        cfg.setdefault("infer", {})["preprocess_raw"] = False
    if args.allow_cpu:
        cfg.setdefault("infer", {})["require_cuda"] = False
    out_dir = ensure_dir(args.output_dir or Path(cfg.get("output_dir", "output/run")) / "inference")
    pred_dir = ensure_dir(out_dir / "preds")
    prob_dir = ensure_dir(out_dir / "probs")

    tabular_mode = str(cfg.get("tabular", {}).get("mode", "numeric"))
    if "tabular_preprocess" in ckpt_meta:
        tabproc = tabproc_from_state_dict(tabular_mode, ckpt_meta["tabular_preprocess"])
    elif cfg.get("tabular", {}).get("preprocess_state"):
        tabproc = tabproc_from_state_dict(tabular_mode, cfg["tabular"]["preprocess_state"])
    else:
        train_rows, tabproc = prepare_rows(cfg, "train", None)
    rows, _ = prepare_rows(cfg, args.split, tabproc, force_missing=bool(cfg.get("tabular", {}).get("force_missing", False)))
    rows = _replace_with_raw_images(rows, cfg)
    cfg.setdefault("tabular", {})["tab_dim"] = tabproc.dim

    require_cuda = bool(cfg.get("infer", {}).get("require_cuda", cfg.get("train", {}).get("require_cuda", True)))
    if require_cuda and not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for inference by this config/job, but torch.cuda.is_available() is False.")
    if torch.cuda.is_available():
        torch.set_float32_matmul_precision("high")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(cfg, tab_dim=tabproc.dim).to(device)
    use_channels_last = bool(cfg.get("train", {}).get("channels_last", False)) and device.type == "cuda"
    if use_channels_last:
        model = model.to(memory_format=torch.channels_last_3d)
    load_checkpoint(
        args.checkpoint,
        model,
        map_location=device,
        strict=False,
        resume_mode="weights",
        ignore_mismatched_shapes=bool(cfg.get("checkpoint", {}).get("ignore_mismatched_shapes", False)),
    )
    model.eval()

    roi_size = tuple(cfg["data"].get("roi_size", [128, 128, 128]))
    sw_batch_size = int(cfg.get("val", {}).get("sw_batch_size", 2))
    infer_cfg = cfg.get("infer", {})
    preprocess_raw = bool(infer_cfg.get("preprocess_raw", bool(infer_cfg.get("image_dir") or infer_cfg.get("raw_image_dir"))))
    xforms = [
        LoadImaged(keys=["image"], image_only=False),
        EnsureChannelFirstd(keys=["image"]),
    ]
    if preprocess_raw:
        # original CT: reorient and resample to the training grid
        xforms += [
            Orientationd(keys=["image"], axcodes=infer_cfg.get("orientation", "LPS"), labels=AXIS_LABELS),
            Spacingd(keys=["image"], pixdim=tuple(infer_cfg.get("spacing", cfg["data"].get("spacing", [0.8, 0.8, 3.0]))), mode="bilinear"),
        ]
    xforms += [
        ScaleIntensityRanged(keys=["image"], a_min=float(cfg["data"].get("hu_min", -175)), a_max=float(cfg["data"].get("hu_max", 250)), b_min=0, b_max=1, clip=True),
        EnsureTyped(keys=["image", "tabular", "tabular_mask"]),
    ]
    xform = Compose(xforms)
    ds_rows = project_rows_for_dataset(rows, keys=("image", "tabular", "tabular_mask", "case_id"))
    ds = Dataset(data=ds_rows, transform=xform)
    loader = DataLoader(ds, batch_size=1, shuffle=False, num_workers=int(cfg["data"].get("num_workers", 2)), collate_fn=list_data_collate)
    summary = []
    flip_sets = _flip_sets(args.tta_flips)
    if len(flip_sets) > 1:
        print(f"TTA: {len(flip_sets)} passes per case, flipped dims {flip_sets}", flush=True)
    with torch.no_grad():
        for batch in loader:
            image = batch["image"].to(device)
            tab = batch["tabular"].float().to(device)
            tab_mask = batch["tabular_mask"].float().to(device)
            case_obj = batch.get("case_id", ["case"])
            case_id = str(case_obj[0] if isinstance(case_obj, (list, tuple)) else case_obj)

            def predictor(x):
                if use_channels_last:
                    x = x.contiguous(memory_format=torch.channels_last_3d)
                local_tab = tab.expand(x.shape[0], -1) if tab.shape[0] == 1 and x.shape[0] > 1 else tab[: x.shape[0]]
                local_mask = tab_mask.expand(x.shape[0], -1) if tab_mask.shape[0] == 1 and x.shape[0] > 1 else tab_mask[: x.shape[0]]
                return model(x, tabular=local_tab, tabular_mask=local_mask)["seg_logits"]

            if len(flip_sets) == 1:
                logits = sliding_window_inference(image, roi_size=roi_size, sw_batch_size=sw_batch_size, predictor=predictor, **sliding_window_kwargs(cfg))
                prob = torch.softmax(logits, dim=1)[0].detach().cpu().numpy()
            else:
                prob_sum = None
                for dims in flip_sets:
                    x = torch.flip(image, dims) if dims else image
                    lg = sliding_window_inference(x, roi_size=roi_size, sw_batch_size=sw_batch_size, predictor=predictor, **sliding_window_kwargs(cfg))
                    if dims:
                        lg = torch.flip(lg, dims)
                    pr = torch.softmax(lg, dim=1)
                    prob_sum = pr if prob_sum is None else prob_sum + pr
                prob = (prob_sum / len(flip_sets))[0].detach().cpu().numpy()
            pred = np.argmax(prob, axis=0).astype(np.uint8)
            affine = _affine_from_batch(batch)
            nib.save(nib.Nifti1Image(pred, affine), pred_dir / f"{case_id}.nii.gz")
            if args.save_probs:
                nib.save(nib.Nifti1Image(prob.astype(np.float32), affine), prob_dir / f"{case_id}_prob.nii.gz")
            elif args.save_prob_classes:
                # float32, since NIfTI-1 has no float16
                idx = [int(c) for c in str(args.save_prob_classes).split(",") if c.strip() != ""]
                sel = prob[idx[0]] if len(idx) == 1 else prob[idx]
                nib.save(nib.Nifti1Image(sel.astype(np.float32), affine), prob_dir / f"{case_id}_prob.nii.gz")
            pdac_score = float(prob[int(cfg.get("labels", {}).get("pdac", 1))].max())
            summary.append({"case_id": case_id, "pdac_score_from_segmentation": pdac_score, "pred_path": str(pred_dir / f"{case_id}.nii.gz"),
                            "tta_passes": len(flip_sets)})
    save_json(summary, out_dir / "inference_summary.json")


if __name__ == "__main__":
    main()
