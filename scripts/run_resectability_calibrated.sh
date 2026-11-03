#!/usr/bin/env bash
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=4:00:00
#SBATCH --job-name=pm_resect_cal
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
set -euo pipefail

# Resectability rule fitted on validation and applied once to test.
#
#   1. sweep contact_tolerance_mm x min_vessel_voxels on the validation predictions
#   2. keep the combination with the best four-class balanced accuracy
#   3. grid-search the distance thresholds on it (tools/calibrate_resectability.py)
#   4. apply both to the test predictions; the angle-only rule is also run for reference
#
# Usage:
#   bash scripts/run_resectability_calibrated.sh [--env <conda-env>] <val_pred_dir> <test_pred_dir> [out_dir]
#
#   val_pred_dir   preds/ of the validation split
#   test_pred_dir  preds/ of the test split
#   out_dir        default: <test_pred_dir>/../resectability_calibrated
#
# TOLERANCES and MIN_VESSEL_VOXELS set the sweep, BETA the F-beta weight on recall:
#   TOLERANCES="0 1 2 3 4" MIN_VESSEL_VOXELS="0 50" BETA=2.0 bash scripts/run_resectability_calibrated.sh ...

usage() { sed -n '/^# Usage:/,/^$/p' "$0" >&2; }

CONDA_ENV=myenv
if [[ "${1:-}" == "--env" ]]; then
  CONDA_ENV=${2:?--env requires an environment name}
  shift 2
fi

if [[ $# -lt 2 ]]; then usage; exit 2; fi

VAL_PRED_DIR=$1
TEST_PRED_DIR=$2
OUT_DIR=${3:-"$(dirname "$TEST_PRED_DIR")/resectability_calibrated"}

DPCG_CSV=${DPCG_CSV:-/path/to/panorama/dpcg_labels.csv}
TOLERANCES=${TOLERANCES:-"0 1 2 3 4"}
MIN_VESSEL_VOXELS=${MIN_VESSEL_VOXELS:-"0 50"}
BETA=${BETA:-2.0}

if [[ -z "${SLURM_JOB_ID:-}" ]]; then
  echo "Not inside a SLURM job. Submitting with sbatch..."
  exec sbatch "$0" --env "$CONDA_ENV" "$VAL_PRED_DIR" "$TEST_PRED_DIR" "$OUT_DIR"
fi

mkdir -p logs

set +u
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$CONDA_ENV"
set -u

export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}
PYTHON_BIN=${PYTHON_BIN:-python}

for d in "$VAL_PRED_DIR" "$TEST_PRED_DIR"; do
  if [[ ! -d "$d" ]]; then echo "[ERROR] pred_dir does not exist: $d" >&2; exit 1; fi
done
if [[ ! -f "$DPCG_CSV" ]]; then echo "[ERROR] dpcg_csv not found: $DPCG_CSV" >&2; exit 1; fi

mkdir -p "$OUT_DIR"
echo "[$(date)] val=${VAL_PRED_DIR}"
echo "[$(date)] test=${TEST_PRED_DIR}"
echo "[$(date)] out=${OUT_DIR}"

# ---- 1. sweep the contact criteria on validation -------------------------------------
echo
echo "=== 1/4  sweep contact_tolerance_mm x min_vessel_voxels on validation ==="
for tol in $TOLERANCES; do
  for mvv in $MIN_VESSEL_VOXELS; do
    sub="$OUT_DIR/val_tol${tol}_mvv${mvv}"
    mkdir -p "$sub"
    echo "--- tolerance=${tol}mm min_vessel_voxels=${mvv}"
    "$PYTHON_BIN" -m src.geometry.dpcg_rules \
      --pred_dir "$VAL_PRED_DIR" \
      --out_csv "$sub/resectability.csv" \
      --dpcg_csv "$DPCG_CSV" \
      --contact_tolerance_mm "$tol" \
      --min_vessel_voxels "$mvv"
  done
done

# ---- 2. pick the best combination ------------------------------------------------------
echo
echo "=== 2/4  pick the best combination on validation ==="
BEST_JSON="$OUT_DIR/sweep_best.json"
"$PYTHON_BIN" - "$OUT_DIR" "$BEST_JSON" <<'PY'
import glob, json, os, re, sys
out_dir, best_path = sys.argv[1], sys.argv[2]
rows = []
for p in sorted(glob.glob(os.path.join(out_dir, "val_tol*_mvv*", "resectability_stats.json"))):
    m = re.search(r"val_tol([0-9.]+)_mvv(\d+)", p)
    s = json.load(open(p, encoding="utf-8"))
    rows.append({"tolerance_mm": float(m.group(1)), "min_vessel_voxels": int(m.group(2)),
                 "balanced_accuracy": s["balanced_accuracy"], "macro_f1": s["macro_f1"],
                 "accuracy": s["accuracy"], "csv": os.path.join(os.path.dirname(p), "resectability.csv")})
if not rows:
    raise SystemExit("No resectability_stats.json produced - did the sweep run?")
rows.sort(key=lambda r: (-r["balanced_accuracy"], -r["macro_f1"], r["tolerance_mm"]))
print(f"{'tol_mm':>7s} {'min_vox':>8s} {'bal.acc':>8s} {'macroF1':>8s} {'acc':>8s}")
for r in rows:
    print(f"{r['tolerance_mm']:7.1f} {r['min_vessel_voxels']:8d} {r['balanced_accuracy']:8.3f} "
          f"{r['macro_f1']:8.3f} {r['accuracy']:8.3f}")
best = rows[0]
json.dump({"best": best, "all": rows}, open(best_path, "w", encoding="utf-8"), indent=2)
print(f"\nbest: tolerance={best['tolerance_mm']}mm min_vessel_voxels={best['min_vessel_voxels']}")
PY

BEST_TOL=$("$PYTHON_BIN" -c "import json,sys;print(json.load(open(sys.argv[1]))['best']['tolerance_mm'])" "$BEST_JSON")
BEST_MVV=$("$PYTHON_BIN" -c "import json,sys;print(json.load(open(sys.argv[1]))['best']['min_vessel_voxels'])" "$BEST_JSON")
BEST_CSV=$("$PYTHON_BIN" -c "import json,sys;print(json.load(open(sys.argv[1]))['best']['csv'])" "$BEST_JSON")

# ---- 3. fit the distance thresholds ----------------------------------------------------
echo
echo "=== 3/4  grid-search the distance thresholds on validation (beta=${BETA}) ==="
THRESHOLDS_JSON="$OUT_DIR/thresholds.json"
"$PYTHON_BIN" tools/calibrate_resectability.py \
  --csv "$BEST_CSV" \
  --out_json "$THRESHOLDS_JSON" \
  --beta "$BETA"

# ---- 4. apply to test, once ------------------------------------------------------------
echo
echo "=== 4/4  apply to the test split (tolerance=${BEST_TOL}mm min_vessel_voxels=${BEST_MVV}) ==="
mkdir -p "$OUT_DIR/test"
"$PYTHON_BIN" -m src.geometry.dpcg_rules \
  --pred_dir "$TEST_PRED_DIR" \
  --out_csv "$OUT_DIR/test/resectability.csv" \
  --dpcg_csv "$DPCG_CSV" \
  --thresholds_json "$THRESHOLDS_JSON" \
  --contact_tolerance_mm "$BEST_TOL" \
  --min_vessel_voxels "$BEST_MVV"

# reference: the angle-only rule on the same predictions
echo
echo "=== reference: angle-only rule on the same test predictions ==="
mkdir -p "$OUT_DIR/test_angle_only"
"$PYTHON_BIN" -m src.geometry.dpcg_rules \
  --pred_dir "$TEST_PRED_DIR" \
  --out_csv "$OUT_DIR/test_angle_only/resectability.csv" \
  --dpcg_csv "$DPCG_CSV"

echo
echo "[$(date)] Done."
echo "  calibrated test : $OUT_DIR/test/resectability_stats.json"
echo "  angle-only test : $OUT_DIR/test_angle_only/resectability_stats.json"
echo "  thresholds      : $THRESHOLDS_JSON"
echo "  sweep           : $BEST_JSON"
