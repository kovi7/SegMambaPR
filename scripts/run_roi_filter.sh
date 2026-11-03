#!/usr/bin/env bash
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --job-name=pm_roifilter
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
set -euo pipefail

# Location filter on predicted pancreas and PDAC (tools/filter_outside_pancreas.py). CPU only.
# Reads existing preds/ and writes to new directories with a _roi<D> suffix.
#
# Usage:
# 1. validation grid, all margins per run:
#   bash scripts/run_roi_filter.sh val "0 10 25 50" <run> [<run> ...]
#   -> output/panorama/<run>/inference_val_best_roi<D>/{preds,evaluation.csv,roi_filter.csv}
#
# 2. apply the chosen margin (50 mm in the thesis) to every run:
#   bash scripts/run_roi_filter.sh apply 50 <run> [<run> ...]
#   -> <test_dir>_roi<D>/{preds,evaluation.csv,resectability.csv}
#      <val_dir>_roi<D>/{preds,evaluation.csv}            (if validation inference exists)
#      <test_dir>_roi<D>/resectability_calibrated/        (if validation inference exists)

CONDA_ENV=${CONDA_ENV:-myenv}
MODE=${1:?usage: run_roi_filter.sh val "<D ...>" <run>... | apply <D> <run>...}
shift

EXCLUDE=${EXCLUDE-}   # optional node list passed to sbatch --exclude
if [[ -z "${SLURM_JOB_ID:-}" ]]; then
  mkdir -p logs
  ARG=${1:?missing margins}; shift
  for RUN in "$@"; do
    sbatch ${EXCLUDE:+--exclude=$EXCLUDE} --job-name="pm_roi_${MODE}_${RUN}" "$0" "$MODE" "$ARG" "$RUN"
  done
  exit 0
fi

mkdir -p logs
set +u
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$CONDA_ENV"
set -u
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}
PYTHON_BIN=${PYTHON_BIN:-python}
JOBS=${SLURM_CPUS_PER_TASK:-8}
DPCG_CSV=${DPCG_CSV:-/path/to/panorama/dpcg_labels.csv}

ARG=${1:?missing margins}
RUN=${2:?missing run}
CFG="configs/panorama/$RUN.yaml"
[[ -f "$CFG" ]] || { echo "[ERROR] no config $CFG" >&2; exit 1; }
VAL="output/panorama/$RUN/inference_val_best"

filter_and_eval() {
  local src=$1 dst=$2 split=$3 margin=$4
  echo "[$(date)] filter $src -> $dst (margin ${margin} mm)"
  "$PYTHON_BIN" tools/filter_outside_pancreas.py --pred_dir "$src/preds" --out_dir "$dst/preds" \
    --margin_mm "$margin" --jobs "$JOBS"
  "$PYTHON_BIN" -m src.evaluate --config "$CFG" --pred_dir "$dst/preds" --split "$split" \
    --out_csv "$dst/evaluation.csv" --jobs "$JOBS"
}

if [[ "$MODE" == "val" ]]; then
  [[ -d "$VAL/preds" ]] || { echo "[skip] $RUN: no $VAL/preds"; exit 0; }
  for D in $ARG; do
    filter_and_eval "$VAL" "${VAL}_roi${D}" val "$D"
  done
  echo "[$(date)] Done: $RUN"
  exit 0
fi

[[ "$MODE" == "apply" ]] || { echo "unknown mode: $MODE" >&2; exit 2; }
D=$ARG
if [[ -d "output/panorama/$RUN/inference_test_after_train3_forced_missing_best/preds" ]]; then
  TEST="output/panorama/$RUN/inference_test_after_train3_forced_missing_best"
else
  TEST="output/panorama/$RUN/inference_test_after_train3_best"
fi
[[ -d "$TEST/preds" ]] || { echo "[ERROR] $RUN: no finished test inference with preds/" >&2; exit 1; }
TEST_F="${TEST}_roi${D}"
VAL_F="${VAL}_roi${D}"

filter_and_eval "$TEST" "$TEST_F" test "$D"
echo "[$(date)] angle rule on filtered test"
"$PYTHON_BIN" -m src.geometry.dpcg_rules --pred_dir "$TEST_F/preds" \
  --out_csv "$TEST_F/resectability.csv" --dpcg_csv "$DPCG_CSV"

if [[ -d "$VAL/preds" ]]; then
  [[ -f "$VAL_F/evaluation.csv" ]] || filter_and_eval "$VAL" "$VAL_F" val "$D"
  echo "[$(date)] calibrated resectability on filtered val/test"
  bash scripts/run_resectability_calibrated.sh --env "$CONDA_ENV" \
    "$VAL_F/preds" "$TEST_F/preds" "$TEST_F/resectability_calibrated"
else
  echo "[info] $RUN has no validation inference, calibrated resectability skipped"
fi
echo "[$(date)] Done: $TEST_F"
