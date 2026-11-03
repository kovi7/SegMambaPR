#!/usr/bin/env bash
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=100G
#SBATCH --time=4:00:00
#SBATCH --job-name=pm_infer
#SBATCH --signal=USR1@300
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
set -euo pipefail

mkdir -p logs

usage() {
  cat <<'USAGE'
Usage:
  bash scripts/infer.sh [--env <conda-env>] <config.yaml> <checkpoint.ckpt|best|last|both> [split] [output_dir_or_prefix] [extra infer args]

--env selects the conda environment activated inside the job (default: myenv) and must
come first.

Examples:
  bash scripts/infer.sh configs/panorama/segmamba_pr.yaml output/panorama/segmamba_pr/best.ckpt test output/panorama/segmamba_pr/inference_test
  bash scripts/infer.sh configs/panorama/segmamba_pr.yaml best test
  bash scripts/infer.sh configs/panorama/segmamba_pr.yaml last test
  bash scripts/infer.sh configs/panorama/segmamba_pr.yaml both test output/panorama/segmamba_pr/inference_test_after_train2

For checkpoint selector:
  best -> uses <config output_dir>/best.ckpt
  last -> uses <config output_dir>/last.ckpt
  both -> submits/runs best and last, with output suffixes _best and _last

After inference, src.evaluate scores the predictions against the ground truth of the split
(evaluation.csv, evaluation_summary.json, eval_*.png). RUN_EVAL=0 skips it.
USAGE
}

CONDA_ENV=myenv
if [[ "${1:-}" == "--env" ]]; then
  CONDA_ENV=${2:?--env requires an environment name}
  shift 2
fi

if [[ $# -lt 2 ]]; then
  usage >&2
  exit 2
fi

CONFIG=$1
CHECKPOINT_ARG=$2
SPLIT=${3:-test}
OUT_DIR=""
EXTRA_ARGS=()

if [[ $# -ge 4 ]]; then
  if [[ "${4}" == --* ]]; then
    EXTRA_ARGS=("${@:4}")
  else
    OUT_DIR=${4}
    if [[ $# -ge 5 ]]; then
      EXTRA_ARGS=("${@:5}")
    fi
  fi
fi

PYTHON_BIN=${PYTHON_BIN:-python}

get_config_output_dir() {
  "$PYTHON_BIN" - "$CONFIG" <<'PY'
import sys
from pathlib import Path
try:
    import yaml
except Exception as e:
    raise SystemExit(f"Could not import yaml while reading config output_dir: {e}")
with open(sys.argv[1], "r", encoding="utf-8") as f:
    cfg = yaml.safe_load(f) or {}
print(cfg.get("output_dir", "output/run"))
PY
}

CONFIG_BASE=$(basename "$CONFIG" .yaml)
CONFIG_GROUP=$(basename "$(dirname "$CONFIG")")
AUTO_JOB_NAME=${JOB_NAME:-pm_inf_${CONFIG_GROUP}_${CONFIG_BASE}_${SPLIT}}

if [[ -z "${SLURM_JOB_ID:-}" ]]; then
  echo "Not inside a SLURM job. Submitting with sbatch as ${AUTO_JOB_NAME}..."
  SBATCH_ARGS=(
    --job-name="${AUTO_JOB_NAME}"
    --output="logs/${AUTO_JOB_NAME}-%j.out"
    --error="logs/${AUTO_JOB_NAME}-%j.err"
  )
  if [[ -n "${SBATCH_DEPENDENCY:-}" ]]; then
    SBATCH_ARGS+=(--dependency="${SBATCH_DEPENDENCY}")
  fi
  exec sbatch "${SBATCH_ARGS[@]}" "$0" --env "$CONDA_ENV" "$CONFIG" "$CHECKPOINT_ARG" "$SPLIT" ${OUT_DIR:+"$OUT_DIR"} "${EXTRA_ARGS[@]}"
fi

# conda.sh defines `conda activate` in the non-interactive job shell
set +u
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$CONDA_ENV"
set -u

export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-16}
export MONAI_DATA_STORE=${MONAI_DATA_STORE:-/tmp}

run_one() {
  local ckpt=$1
  local out=$2
  local tag=$3
  if [[ ! -f "$ckpt" ]]; then
    echo "[ERROR] Missing checkpoint for ${tag}: ${ckpt}" >&2
    exit 1
  fi
  if [[ -z "$out" ]]; then
    out="$CFG_OUT/inference"
  fi
  local cmd=("$PYTHON_BIN" -m src.infer --config "$CONFIG" --checkpoint "$ckpt" --split "$SPLIT" --output_dir "$out")
  cmd+=("${EXTRA_ARGS[@]}")

  echo "[$(date)] Inference config: ${CONFIG}"
  echo "checkpoint (${tag}): ${ckpt}; split: ${SPLIT}; output: ${out}"
  nvidia-smi || true
  srun --ntasks=1 --cpus-per-task=${SLURM_CPUS_PER_TASK:-16} "${cmd[@]}"

  if [[ "${RUN_EVAL:-1}" == "1" ]]; then
    echo "[$(date)] Evaluating (${tag}): predictions in ${out}/preds"
    "$PYTHON_BIN" -m src.evaluate \
      --config "$CONFIG" \
      --pred_dir "$out/preds" \
      --split "$SPLIT" \
      --out_csv "$out/evaluation.csv" \
      --jobs "${EVAL_JOBS:-${SLURM_CPUS_PER_TASK:-4}}" \
      || echo "[WARN] Evaluation failed for ${tag} (no local ground truth for split '${SPLIT}'?), skipping."
  fi
}

CFG_OUT=$(get_config_output_dir)

case "$CHECKPOINT_ARG" in
  best)
    CKPT="$CFG_OUT/best.ckpt"
    OUT="${OUT_DIR:-$CFG_OUT/inference_${SPLIT}_best}"
    run_one "$CKPT" "$OUT" "best"
    ;;
  last)
    CKPT="$CFG_OUT/last.ckpt"
    OUT="${OUT_DIR:-$CFG_OUT/inference_${SPLIT}_last}"
    run_one "$CKPT" "$OUT" "last"
    ;;
  both)
    BEST_CKPT="$CFG_OUT/best.ckpt"
    LAST_CKPT="$CFG_OUT/last.ckpt"
    if [[ -n "$OUT_DIR" ]]; then
      BEST_OUT="${OUT_DIR}_best"
      LAST_OUT="${OUT_DIR}_last"
    else
      BEST_OUT="$CFG_OUT/inference_${SPLIT}_best"
      LAST_OUT="$CFG_OUT/inference_${SPLIT}_last"
    fi
    run_one "$BEST_CKPT" "$BEST_OUT" "best"
    run_one "$LAST_CKPT" "$LAST_OUT" "last"
    ;;
  *)
    if [[ ! -f "$CHECKPOINT_ARG" ]]; then
      echo "[ERROR] Checkpoint does not exist and is not one of: best, last, both: ${CHECKPOINT_ARG}" >&2
      exit 1
    fi
    run_one "$CHECKPOINT_ARG" "$OUT_DIR" "explicit"
    ;;
esac
