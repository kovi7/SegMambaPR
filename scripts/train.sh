#!/usr/bin/env bash
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=16
# the whole training set is cached in memory (peak ~443 GiB)
#SBATCH --mem=500G
#SBATCH --time=48:00:00
#SBATCH --job-name=pm_train
#SBATCH --signal=USR1@300
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
set -euo pipefail

# an optional leading --time <spec> overrides the --time header
SBATCH_EXTRA=()
if [[ "${1:-}" == "--time" ]]; then
  SBATCH_EXTRA+=(--time="${2:?--time requires a value, e.g. 7-00:00:00}")
  shift 2
fi

if [[ -z "${SLURM_JOB_ID:-}" ]]; then
  echo "Not inside a SLURM job. Submitting with sbatch..."
  exec sbatch "${SBATCH_EXTRA[@]}" "$0" "$@"
fi

mkdir -p logs

# an optional leading --env <name> selects the conda environment
CONDA_ENV=myenv
if [[ "${1:-}" == "--env" ]]; then
  CONDA_ENV=${2:?--env requires an environment name}
  shift 2
fi

CONFIG=${1:?Usage: sbatch scripts/train.sh [--env <conda-env>] <config.yaml> [extra train args]}
shift || true

# conda.sh defines `conda activate` in the non-interactive job shell
set +u
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$CONDA_ENV"
set -u

export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-16}
export MONAI_DATA_STORE=${MONAI_DATA_STORE:-/tmp}

PYTHON_BIN=${PYTHON_BIN:-python}
echo "[$(date)] Training config: ${CONFIG}"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
nvidia-smi || true

srun --ntasks=1 --cpus-per-task=${SLURM_CPUS_PER_TASK:-16} \
  "$PYTHON_BIN" -m src.train --config "$CONFIG" "$@"
