#!/usr/bin/env bash
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=4:00:00
#SBATCH --job-name=pm_band_mass
#SBATCH --signal=USR1@300
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
set -euo pipefail

# Mean predicted tumour probability in rings around the ground-truth arteries (tools/band_mass.py).
# CPU only. Needs probability maps from inference with --save_prob_classes 1:
#   bash scripts/infer.sh configs/panorama/X.yaml best test output/panorama/X/inference_probs \
#       --force_missing_tabular --save_prob_classes 1
#
# Usage:
#   bash scripts/run_band_mass.sh [--env <conda-env>] <config.yaml> <prob_dir> [out_csv] [split]
#
#   prob_dir   directory of <case>_prob.nii.gz, normally <inference dir>/probs
#   out_csv    default: <prob_dir>/../band_mass.csv
#   split      default: test
#
# DPCG_CSV sets the reference categories, BAND_MM the band width and ROI_MM the
# neighbourhood of the pancreas both rings are restricted to.

mkdir -p logs

CONDA_ENV=myenv
if [[ "${1:-}" == "--env" ]]; then
  CONDA_ENV=${2:?--env requires an environment name}
  shift 2
fi

if [[ $# -lt 2 ]]; then
  sed -n '/^# Usage:/,/^$/p' "$0" >&2
  exit 2
fi

CONFIG=$1
PROB_DIR=$2
OUT_CSV=${3:-"$(dirname "$PROB_DIR")/band_mass.csv"}
SPLIT=${4:-test}

DPCG_CSV=${DPCG_CSV:-/path/to/panorama/dpcg_labels.csv}
BAND_MM=${BAND_MM:-5.0}
ROI_MM=${ROI_MM:-30.0}
VESSEL_CLASSES=${VESSEL_CLASSES:-3}

if [[ -z "${SLURM_JOB_ID:-}" ]]; then
  echo "Not inside a SLURM job. Submitting with sbatch..."
  exec sbatch "$0" --env "$CONDA_ENV" "$CONFIG" "$PROB_DIR" "$OUT_CSV" "$SPLIT"
fi

set +u
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$CONDA_ENV"
set -u

export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}
PYTHON_BIN=${PYTHON_BIN:-python}

if [[ ! -d "$PROB_DIR" ]]; then
  echo "[ERROR] prob_dir does not exist: ${PROB_DIR}" >&2
  echo "        Run inference with --save_prob_classes 1 first." >&2
  exit 1
fi

CMD=("$PYTHON_BIN" tools/band_mass.py --config "$CONFIG" --split "$SPLIT"
     --prob_dir "$PROB_DIR" --out_csv "$OUT_CSV"
     --band_mm "$BAND_MM" --roi_mm "$ROI_MM" --vessel_classes "$VESSEL_CLASSES")
if [[ -f "$DPCG_CSV" ]]; then
  CMD+=(--dpcg_csv "$DPCG_CSV")
else
  echo "[WARN] dpcg_csv not found at ${DPCG_CSV}, the per-stage summary will be empty." >&2
fi

echo "[$(date)] Band mass: config=${CONFIG} prob_dir=${PROB_DIR} split=${SPLIT} out=${OUT_CSV}"
srun --ntasks=1 --cpus-per-task=${SLURM_CPUS_PER_TASK:-8} "${CMD[@]}"
