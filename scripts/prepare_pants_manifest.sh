#!/usr/bin/env bash
set -euo pipefail

# Builds the PanTS manifest from data converted by tools/convert_pants.py:
#   <PANTS_CONVERTED_ROOT>/images/*_0000_trans.nii.gz
#   <PANTS_CONVERTED_ROOT>/labels/all/*_trans.nii.gz
#   <PANTS_CONVERTED_ROOT>/splits.json

PANTS_CONVERTED_ROOT=${PANTS_CONVERTED_ROOT:-/path/to/PanTS/converted}
PANTS_MODE=${PANTS_MODE:-pants28}  # pants6, all, or legacy alias pants28
VERIFY=${VERIFY:-1}
COUNT_LABELS=${COUNT_LABELS:-1}

if [[ "$PANTS_MODE" == "pants6" ]]; then
  OUT_MANIFEST=${OUT_MANIFEST:-data/pants_manifest_pants6.json}
elif [[ "$PANTS_MODE" == "all" || "$PANTS_MODE" == "pants28" ]]; then
  OUT_MANIFEST=${OUT_MANIFEST:-data/pants_manifest_all.json}
else
  echo "Unknown PANTS_MODE=${PANTS_MODE}; use pants6, all, or pants28" >&2
  exit 2
fi

ARGS=(python -m src.prepare_pants_manifest --converted_root "$PANTS_CONVERTED_ROOT" --mode "$PANTS_MODE" --out_manifest "$OUT_MANIFEST")
if [[ "$VERIFY" == "0" ]]; then
  ARGS+=(--no_verify)
fi
if [[ "$COUNT_LABELS" == "0" ]]; then
  ARGS+=(--no_count_labels)
fi

"${ARGS[@]}"
