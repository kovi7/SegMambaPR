#!/usr/bin/env bash
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=1:00:00
#SBATCH --job-name=pm_resect
#SBATCH --signal=USR1@300
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
set -euo pipefail

# CPU only: geometry on already predicted masks.

mkdir -p logs

usage() {
  cat <<'USAGE'
Usage:
  bash scripts/run_resectability.sh [--env <conda-env>] <pred_dir> [dpcg_csv] [out_csv]

--env selects the conda environment activated inside the job (default: myenv) and must
come first.

Runs src.geometry.dpcg_rules on an existing inference preds/ directory (predicted
segmentation masks must already exist there - run scripts/infer.sh first).

  pred_dir  Directory of predicted *.nii.gz masks, e.g.
            output/panorama/segmamba_pr/inference_test_after_train2_best/preds
  dpcg_csv  Optional. Reference resectability categories (is_resectable/is_borderline/
            is_locally_advanced). If given, a confusion matrix and resectability_stats.json
            are written next to out_csv.
  out_csv   Default: <pred_dir>/../resectability.csv

Examples:
  bash scripts/run_resectability.sh output/panorama/segmamba_pr/inference_test_after_train2_best/preds /path/to/panorama/dpcg_labels.csv
  bash scripts/run_resectability.sh output/panorama/segmamba_pr/inference_test_after_train2_best/preds
USAGE
}

CONDA_ENV=myenv
if [[ "${1:-}" == "--env" ]]; then
  CONDA_ENV=${2:?--env requires an environment name}
  shift 2
fi

if [[ $# -lt 1 ]]; then
  usage >&2
  exit 2
fi

PRED_DIR=$1
DPCG_CSV=${2:-}
OUT_CSV=${3:-"$(dirname "$PRED_DIR")/resectability.csv"}

if [[ -z "${SLURM_JOB_ID:-}" ]]; then
  echo "Not inside a SLURM job. Submitting with sbatch..."
  exec sbatch "$0" --env "$CONDA_ENV" "$PRED_DIR" "$DPCG_CSV" "$OUT_CSV"
fi

# conda.sh defines `conda activate` in the non-interactive job shell
set +u
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$CONDA_ENV"
set -u

export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-4}
PYTHON_BIN=${PYTHON_BIN:-python}

if [[ ! -d "$PRED_DIR" ]]; then
  echo "[ERROR] pred_dir does not exist: ${PRED_DIR}" >&2
  exit 1
fi

CMD=("$PYTHON_BIN" -m src.geometry.dpcg_rules --pred_dir "$PRED_DIR" --out_csv "$OUT_CSV")
if [[ -n "$DPCG_CSV" ]]; then
  CMD+=(--dpcg_csv "$DPCG_CSV")
fi

echo "[$(date)] Resectability: pred_dir=${PRED_DIR} dpcg_csv=${DPCG_CSV:-<none, no evaluation>} out_csv=${OUT_CSV}"
srun --ntasks=1 --cpus-per-task=${SLURM_CPUS_PER_TASK:-4} "${CMD[@]}"
