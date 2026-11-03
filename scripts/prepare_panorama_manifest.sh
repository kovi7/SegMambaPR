#!/usr/bin/env bash
set -euo pipefail
IMAGE_DIR=${IMAGE_DIR:-/path/to/panorama/converted/images}
LABEL_DIR=${LABEL_DIR:-/path/to/panorama/converted/labels}
CLINICAL_XLSX=${CLINICAL_XLSX:-/path/to/panorama/clinical_information.xlsx}
TEST_LIST=${TEST_LIST:-/path/to/panorama/PANORAMA_test.txt}
DPCG_CSV=${DPCG_CSV:-/path/to/panorama/dpcg_labels.csv}
VERIFICATION_CSV=${VERIFICATION_CSV:-/path/to/panorama/PANORAMA_verification.csv}
OUT=${OUT:-data/panorama_manifest_seed42.json}
SEED=${SEED:-42}
VAL_FRACTION=${VAL_FRACTION:-0.15}
python -m src.prepare_manifest \
  --image_dir "$IMAGE_DIR" \
  --label_dir "$LABEL_DIR" \
  --clinical_xlsx "$CLINICAL_XLSX" \
  --test_list_path "$TEST_LIST" \
  --dpcg_csv "$DPCG_CSV" \
  --verification_csv "$VERIFICATION_CSV" \
  --verification_label_column Pancreas_label_v2 \
  --out "$OUT" \
  --seed "$SEED" \
  --val_fraction "$VAL_FRACTION"
