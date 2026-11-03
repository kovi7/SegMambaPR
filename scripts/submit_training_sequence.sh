#!/usr/bin/env bash
set -euo pipefail

# Submit a chained SLURM training sequence for one or more configurations:
#   train_1 -> train_2 resume -> infer best+last -> train_3 resume -> infer best+last
# Extra arguments after "--" are passed only to train.sh.

mkdir -p logs

# model name = config name with "-" for "_"; configs/panorama/<name>.yaml
ALL_MODELS=(
  cnn-unet
  monai-unet
  swinunetr
  segmamba-pr
  segmamba-pr-tabgated
  segmamba-pr-tabgated-adf
  segmamba-pr-tabgated-dwconv
  segmamba-pr-tabgated-text
  segmamba-pr-tabgated-sep
  segmamba-pr-tabgated-band
  segmamba-pr-tabgated-msdw
  segmamba-pr-tabgated-sdf
  segmamba-pr-tabgated-aggregator
  segmamba-pr-tabgated-iface
  segmamba-pr-tabgated-band-iface
  segmamba-pr-tabgated-band-iface-pants
  pants-all-pretrain
)

usage() {
  cat <<'USAGE'
Usage:
  bash scripts/submit_training_sequence.sh --model <name>[,<name2>] [options] [-- extra train args]
  bash scripts/submit_training_sequence.sh --all [options] [-- extra train args]
  bash scripts/submit_training_sequence.sh --list

Sequence per selected model:
  train_1
    -> train_2 with --resume --resume_mode full --checkpoint_path <output_dir>/last.ckpt
    -> inference from best.ckpt and last.ckpt in one infer.sh job
    -> train_3 with --resume --resume_mode full --checkpoint_path <output_dir>/last.ckpt
    -> inference from best.ckpt and last.ckpt in one infer.sh job

Model names: see --list. A name is the config name with "-" for "_"
(segmamba-pr-tabgated-band -> configs/panorama/segmamba_pr_tabgated_band.yaml);
pants-all-pretrain is configs/pants/segmamba_pr_pretrain_pants_all.yaml.

Options:
  --all, -all                 Run the sequence for all registered model variants.
  --model, -m <name[,name2]>  Run the sequence only for selected model(s). Can be repeated.
  --list                      Print registered model names and exit.
  --env <name>                Conda environment activated inside every submitted job.
                              Default: myenv. Forwarded as --env to train.sh, infer.sh
                              and run_resectability.sh.
  --dependency, -d <dep>      Add sbatch dependency to the first training job, e.g. afterany:123456.
  --serial                    When multiple models are selected, start the next model sequence after the previous one finishes.
  --stage-epochs E1,E2,E3     Optional max_epochs overrides for train_1, train_2, train_3.
                              If omitted, YAML max_epochs is used unchanged in every training job.
  --split <split>             Inference split. Default: test.
  --strict                    Use afterok instead of afterany for internal dependencies.
  --save-probs                Pass --save_probs to all inference jobs.
  --with-resectability         After each inference step (train_2 and train_3), also submit
                               scripts/run_resectability.sh on the best/last preds/ dirs,
                               depending on that inference job. Skipped per-model with a
                               warning if the config has no data.dpcg_csv set. Training-only
                               (the default, this flag omitted) is unaffected either way.
  --train-only N              Submit N chained training stages (train_1, then N-1 resumes from
                              last.ckpt) and nothing else: no inference, no resectability.
                              Meant for PanTS pretraining, whose 29-label model cannot be
                              evaluated on the PANORAMA test split. Stages after the last epoch
                              exit without training.
  --resume-first              train_1 resumes from last.ckpt when one exists instead of
                              starting from scratch. Models with no checkpoint still start
                              from zero, so a mix of part-trained and never-started models
                              can be brought to the same budget in one command.
  --dry-run                   Print sbatch commands instead of submitting them.
  --help, -h                  Show this help.

Examples:
  bash scripts/submit_training_sequence.sh --model segmamba-pr
  bash scripts/submit_training_sequence.sh --model segmamba-pr,monai-unet --stage-epochs 250,500,750
  bash scripts/submit_training_sequence.sh --model segmamba-pr-tabgated --with-resectability
  bash scripts/submit_training_sequence.sh --model pants-all-pretrain --train-only 6

Notes:
  - Extra arguments after "--" are passed to training jobs only, not to inference.
  - Resume always uses last.ckpt, which holds the optimizer, scheduler and scaler state.
  - TabGated variants run inference with --force_missing_tabular, since no test case has clinical data.
  - PanTS fine-tuning: see README.md.
USAGE
}

SELECTED_MODELS=()
RUN_ALL=0
INITIAL_DEP=""
SERIAL=0
STAGE_EPOCHS=""
SPLIT="test"
STRICT=0
DRY_RUN=0
SAVE_PROBS=0
WITH_RESECTABILITY=0
RESUME_FIRST=0
TRAIN_ONLY=0
TRAIN_EXTRA=()
ENV_ARGS=()

append_models_csv() {
  local csv=$1
  local old_ifs=${IFS}
  IFS=',' read -r -a parts <<< "$csv"
  IFS=${old_ifs}
  local m
  for m in "${parts[@]}"; do
    m=$(echo "$m" | xargs)
    [[ -n "$m" ]] && SELECTED_MODELS+=("$m")
  done
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --all|-all)
      RUN_ALL=1
      shift
      ;;
    --model|-m)
      [[ $# -ge 2 ]] || { echo "ERROR: --model requires a value" >&2; exit 2; }
      append_models_csv "$2"
      shift 2
      ;;
    --env)
      [[ $# -ge 2 ]] || { echo "ERROR: --env requires an environment name" >&2; exit 2; }
      ENV_ARGS=(--env "$2")
      shift 2
      ;;
    --dependency|-d)
      [[ $# -ge 2 ]] || { echo "ERROR: --dependency requires a value" >&2; exit 2; }
      INITIAL_DEP="$2"
      shift 2
      ;;
    --serial)
      SERIAL=1
      shift
      ;;
    --stage-epochs)
      [[ $# -ge 2 ]] || { echo "ERROR: --stage-epochs requires E1,E2,E3" >&2; exit 2; }
      STAGE_EPOCHS="$2"
      shift 2
      ;;
    --split)
      [[ $# -ge 2 ]] || { echo "ERROR: --split requires a value" >&2; exit 2; }
      SPLIT="$2"
      shift 2
      ;;
    --strict)
      STRICT=1
      shift
      ;;
    --dry-run)
      DRY_RUN=1
      shift
      ;;
    --save-probs)
      SAVE_PROBS=1
      shift
      ;;
    --with-resectability)
      WITH_RESECTABILITY=1
      shift
      ;;
    --resume-first)
      RESUME_FIRST=1
      shift
      ;;
    --train-only)
      [[ $# -ge 2 && "$2" =~ ^[1-9][0-9]*$ ]] || { echo "ERROR: --train-only requires a number of stages" >&2; exit 2; }
      TRAIN_ONLY="$2"
      shift 2
      ;;
    --list)
      printf '%s\n' "${ALL_MODELS[@]}"
      exit 0
      ;;
    --help|-h)
      usage
      exit 0
      ;;
    --)
      shift
      TRAIN_EXTRA=("$@")
      break
      ;;
    *)
      echo "ERROR: unknown option: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

if [[ "$RUN_ALL" == "1" ]]; then
  SELECTED_MODELS=("${ALL_MODELS[@]}")
fi

if [[ ${#SELECTED_MODELS[@]} -eq 0 ]]; then
  echo "ERROR: choose --model <name> or --all" >&2
  usage >&2
  exit 2
fi

DEP_KIND="afterany"
if [[ "$STRICT" == "1" ]]; then
  DEP_KIND="afterok"
fi

EPOCH_1=""
EPOCH_2=""
EPOCH_3=""
if [[ -n "$STAGE_EPOCHS" ]]; then
  IFS=',' read -r EPOCH_1 EPOCH_2 EPOCH_3 <<< "$STAGE_EPOCHS"
  if [[ -z "${EPOCH_1:-}" || -z "${EPOCH_2:-}" || -z "${EPOCH_3:-}" ]]; then
    echo "ERROR: --stage-epochs must have exactly three comma-separated values, e.g. 250,500,750" >&2
    exit 2
  fi
fi

sanitize_name() {
  echo "$1" | tr '/.-' '___'
}

resolve_model() {
  local name=${1//-/_}
  CFG=""
  FORCE_MISSING=0
  if [[ "$name" == "pants_all_pretrain" || "$name" == "segmamba_pr_pretrain_pants_all" ]]; then
    CFG="configs/pants/segmamba_pr_pretrain_pants_all.yaml"
  else
    CFG="configs/panorama/${name}.yaml"
    # gated runs are tested without clinical metadata
    if [[ "$name" == *tabgated* ]]; then
      FORCE_MISSING=1
    fi
  fi
  if [[ ! -f "$CFG" ]]; then
    echo "ERROR: unknown model '$1' (no $CFG). Use --list." >&2
    exit 2
  fi
}

get_output_dir() {
  local cfg=$1
  local out
  out=$(grep -m1 '^output_dir:' "$cfg" | sed 's/^output_dir:[[:space:]]*//' | tr -d "\"'\r")
  if [[ -z "$out" ]]; then
    echo "ERROR: could not read output_dir from $cfg" >&2
    exit 2
  fi
  echo "$out"
}

get_config_dpcg_csv() {
  local cfg=$1
  # `|| true`: configs without data.dpcg_csv (PanTS) must not abort under set -e
  grep -m1 '^  dpcg_csv:' "$cfg" | sed 's/^  dpcg_csv:[[:space:]]*//' | tr -d "\"'\r" || true
}

print_cmd() {
  local q=()
  local a
  for a in "$@"; do
    q+=("$(printf '%q' "$a")")
  done
  printf '%s\n' "${q[*]}"
}

submit_sbatch() {
  local name=$1
  local dep=$2
  shift 2

  local cmd=(sbatch --parsable --job-name="$name")
  if [[ -n "$dep" ]]; then
    cmd+=(--dependency="$dep")
  fi
  cmd+=("$@")

  if [[ "$DRY_RUN" == "1" ]]; then
    print_cmd "${cmd[@]}" >&2
    echo "DRYRUN_${name}"
    return 0
  fi

  local jid
  jid=$("${cmd[@]}")
  jid=${jid%%;*}
  echo "$jid"
}

join_deps() {
  local kind=$1
  shift
  local ids=()
  local id
  for id in "$@"; do
    [[ -n "$id" ]] && ids+=("$id")
  done
  if [[ ${#ids[@]} -eq 0 ]]; then
    echo "" >&2
    return 0
  fi
  local joined
  joined=$(IFS=:; echo "${ids[*]}")
  echo "${kind}:${joined}"
}

submit_resectability_for_infer() {
  local safe_model=$1
  local stage_tag=$2
  local infer_dep=$3
  local infer_out_prefix=$4
  local dpcg_csv=$5

  if [[ -z "$dpcg_csv" ]]; then
    echo "[WARN] ${safe_model}: config has no data.dpcg_csv set - skipping --with-resectability for ${stage_tag}." >&2
    return 0
  fi

  local variant jid
  for variant in best last; do
    jid=$(submit_sbatch "pm_${safe_model}_resect_${stage_tag}_${variant}" "$(join_deps "$DEP_KIND" "$infer_dep")" \
      scripts/run_resectability.sh "${ENV_ARGS[@]}" "${infer_out_prefix}_${variant}/preds" "$dpcg_csv")
    echo "resectability (${stage_tag}, ${variant}): ${jid}" >&2
  done
}

submit_sequence_for_model() {
  local model=$1
  local first_dep=$2

  resolve_model "$model"
  local cfg="$CFG"
  local out_dir
  out_dir=$(get_output_dir "$cfg")
  local dpcg_csv
  dpcg_csv=$(get_config_dpcg_csv "$cfg")
  local safe_model
  safe_model=$(sanitize_name "$model")
  local infer_suffix=""
  local infer_extra=()

  if [[ "$FORCE_MISSING" == "1" ]]; then
    infer_extra+=(--force_missing_tabular)
    infer_suffix="_forced_missing"
  fi
  if [[ "$SAVE_PROBS" == "1" ]]; then
    infer_extra+=(--save_probs)
  fi

  local train1_args=()
  if [[ "$RESUME_FIRST" == "1" && -f "${out_dir}/last.ckpt" ]]; then
    train1_args=(--resume --resume_mode full --checkpoint_path "${out_dir}/last.ckpt")
    echo "train_1: resuming from ${out_dir}/last.ckpt" >&2
  elif [[ "$RESUME_FIRST" == "1" ]]; then
    echo "train_1: no last.ckpt, starting from scratch" >&2
  fi
  local train2_args=(--resume --resume_mode full --checkpoint_path "${out_dir}/last.ckpt")
  local train3_args=(--resume --resume_mode full --checkpoint_path "${out_dir}/last.ckpt")

  if [[ -n "$EPOCH_1" ]]; then
    train1_args+=(--max_epochs "$EPOCH_1")
    train2_args+=(--max_epochs "$EPOCH_2")
    train3_args+=(--max_epochs "$EPOCH_3")
  fi

  train1_args+=("${TRAIN_EXTRA[@]}")
  train2_args+=("${TRAIN_EXTRA[@]}")
  train3_args+=("${TRAIN_EXTRA[@]}")

  echo "" >&2
  echo "=== ${model} ===" >&2
  echo "config: ${cfg}" >&2
  echo "output_dir: ${out_dir}" >&2

  local t1 t2 i2 t3 i3

  if [[ "$TRAIN_ONLY" -gt 0 ]]; then
    local resume_args=(--resume --resume_mode full --checkpoint_path "${out_dir}/last.ckpt" "${TRAIN_EXTRA[@]}")
    local prev stage
    prev=$(submit_sbatch "pm_${safe_model}_t1" "$first_dep" scripts/train.sh "${ENV_ARGS[@]}" "$cfg" "${train1_args[@]}")
    echo "train_1: ${prev}" >&2
    for ((stage = 2; stage <= TRAIN_ONLY; stage++)); do
      prev=$(submit_sbatch "pm_${safe_model}_t${stage}_resume" "$(join_deps "$DEP_KIND" "$prev")"         scripts/train.sh "${ENV_ARGS[@]}" "$cfg" "${resume_args[@]}")
      echo "train_${stage} resume: ${prev}" >&2
    done
    join_deps "$DEP_KIND" "$prev"
    return 0
  fi

  t1=$(submit_sbatch "pm_${safe_model}_t1" "$first_dep" scripts/train.sh "${ENV_ARGS[@]}" "$cfg" "${train1_args[@]}")
  echo "train_1: ${t1}" >&2

  t2=$(submit_sbatch "pm_${safe_model}_t2_resume" "$(join_deps "$DEP_KIND" "$t1")"     scripts/train.sh "${ENV_ARGS[@]}" "$cfg" "${train2_args[@]}")
  echo "train_2 resume: ${t2}" >&2

  i2=$(submit_sbatch "pm_${safe_model}_i2_both" "$(join_deps "$DEP_KIND" "$t2")"     scripts/infer.sh "${ENV_ARGS[@]}" "$cfg" both "$SPLIT" "${out_dir}/inference_${SPLIT}_after_train2${infer_suffix}" "${infer_extra[@]}")
  echo "infer after train_2 best+last: ${i2}" >&2

  if [[ "$WITH_RESECTABILITY" == "1" ]]; then
    submit_resectability_for_infer "$safe_model" "t2" "$i2" "${out_dir}/inference_${SPLIT}_after_train2${infer_suffix}" "$dpcg_csv"
  fi

  t3=$(submit_sbatch "pm_${safe_model}_t3_resume" "$(join_deps "$DEP_KIND" "$i2")"     scripts/train.sh "${ENV_ARGS[@]}" "$cfg" "${train3_args[@]}")
  echo "train_3 resume: ${t3}" >&2

  i3=$(submit_sbatch "pm_${safe_model}_i3_both" "$(join_deps "$DEP_KIND" "$t3")"     scripts/infer.sh "${ENV_ARGS[@]}" "$cfg" both "$SPLIT" "${out_dir}/inference_${SPLIT}_after_train3${infer_suffix}" "${infer_extra[@]}")
  echo "infer after train_3 best+last: ${i3}" >&2

  if [[ "$WITH_RESECTABILITY" == "1" ]]; then
    submit_resectability_for_infer "$safe_model" "t3" "$i3" "${out_dir}/inference_${SPLIT}_after_train3${infer_suffix}" "$dpcg_csv"
  fi

  # dependency string for chaining the next model with --serial
  join_deps "$DEP_KIND" "$i3"
}

chain_dep="$INITIAL_DEP"
for model in "${SELECTED_MODELS[@]}"; do
  if [[ "$SERIAL" == "1" ]]; then
    chain_dep=$(submit_sequence_for_model "$model" "$chain_dep")
  else
    submit_sequence_for_model "$model" "$INITIAL_DEP" >/dev/null
  fi
done
