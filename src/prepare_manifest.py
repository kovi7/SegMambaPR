from __future__ import annotations

import argparse
from pathlib import Path

from .manifest import build_panorama_manifest_rows, save_manifest, summarize_manifest
from .utils import save_json


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build a reproducible PANORAMA manifest for multimodal segmentation experiments.")
    p.add_argument("--image_dir", required=True)
    p.add_argument("--label_dir", required=True)
    p.add_argument("--clinical_xlsx", default=None)
    p.add_argument("--test_list_path", default=None)
    p.add_argument("--dpcg_csv", default=None)
    p.add_argument("--verification_csv", default=None, help="Optional PANORAMA_verification.csv with detailed pancreas labels.")
    p.add_argument("--verification_label_column", default="Pancreas_label_v2", help="Column used as primary tumor label in verification CSV.")
    p.add_argument("--exclude_unusable_verification", action="store_true", help="Skip verification labels such as no pancreas / pancreas not visible.")
    p.add_argument("--ambiguous_pdac_as_pdac", action="store_true", default=False, help="Treat verification labels such as PNET/PDAC? as PDAC. Disabled by default; clinical_information/dpcg remain primary.")
    p.add_argument("--no_ambiguous_pdac_as_pdac", dest="ambiguous_pdac_as_pdac", action="store_false", help="Do not override ambiguous PDAC verification labels.")
    p.add_argument("--out", default="data/panorama_manifest.json")
    p.add_argument("--summary_out", default=None)
    p.add_argument("--image_suffix", default="_0000_trans.nii.gz")
    p.add_argument("--label_suffix", default="_trans.nii.gz")
    p.add_argument("--val_fraction", type=float, default=0.15)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--pdac_label", type=int, default=1)
    p.add_argument("--infer_pdac_from_mask", action="store_true")
    p.add_argument("--no_test", action="store_true", help="Do not include cases from PANORAMA_test.txt in the manifest.")
    p.add_argument("--allow_missing_labels", action="store_true", help="Keep image rows even when a label file is missing. Use only for inference manifests.")
    p.add_argument("--no_split", action="store_true", help="Do not assign train/val split; rows not in test stay as train. Not recommended for experiments.")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    rows = build_panorama_manifest_rows(
        image_dir=args.image_dir,
        label_dir=args.label_dir,
        clinical_xlsx=args.clinical_xlsx,
        test_list_path=args.test_list_path,
        dpcg_csv=args.dpcg_csv,
        verification_csv=args.verification_csv,
        verification_label_column=args.verification_label_column,
        image_suffix=args.image_suffix,
        label_suffix=args.label_suffix,
        val_fraction=args.val_fraction,
        seed=args.seed,
        pdac_label=args.pdac_label,
        infer_pdac_from_mask=args.infer_pdac_from_mask,
        include_test=not args.no_test,
        require_label=not args.allow_missing_labels,
        assign_split=not args.no_split,
        exclude_unusable_verification=args.exclude_unusable_verification,
        ambiguous_pdac_as_pdac=args.ambiguous_pdac_as_pdac,
    )
    save_manifest(rows, args.out)
    summary = summarize_manifest(rows)
    summary_path = args.summary_out or str(Path(args.out).with_suffix(".summary.json"))
    save_json(summary, summary_path)
    print(f"Saved manifest: {args.out}")
    print(f"Saved summary : {summary_path}")
    print(summary)


if __name__ == "__main__":
    main()
