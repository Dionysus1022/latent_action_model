#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  scripts/run_lgbs_pipeline.sh \
    --task cube|pusht|tworoom|reacher|all \
    [--stage all|extract|train|eval] \
    [--output-root /data/ykz/lgbs_repro] \
    [--dataset-h5 /path/to/dataset.h5] \
    [--device cuda:0] \
    [--seed 42] \
    [--epochs 50] \
    [--batch-size 1024] \
    [--extract-batch-size 256] \
    [--extract-prefetch 1] \
    [--data-parallel] \
    [--num-eval 200] \
    [--eval-budget 50] \
    [--goal-offset 25] \
    [--compare|--idm-only|--cem-only] \
    [--force-extract] \
    [--force-train] \
    [--dry-run]

Runs the Latent Geometry Beyond Search GC-IDM reproduction pipeline:
  1. train_idm.py extract
  2. train_idm.py train
  3. eval_idm.py

The default dataset paths are provided by
external/latent-geometry-beyond-search/local_data_paths.py.
Use --dataset-h5 to override the default path for a single task.
EOF
}

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PAPER_ROOT="${REPO_ROOT}/external/latent-geometry-beyond-search"
PYTHON_BIN="${REPO_ROOT}/.venv/bin/python"

TASK=""
STAGE="all"
OUTPUT_ROOT="/data/ykz/lgbs_repro"
DATASET_H5=""
DEVICE="cuda:0"
SEED="42"
EPOCHS="50"
BATCH_SIZE="1024"
EXTRACT_BATCH_SIZE="256"
EXTRACT_PREFETCH="1"
DATA_PARALLEL="0"
NUM_EVAL="200"
EVAL_BUDGET="50"
GOAL_OFFSET="25"
EVAL_MODE="compare"
FORCE_EXTRACT="0"
FORCE_TRAIN="0"
DRY_RUN="0"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --task)
      TASK="$2"
      shift 2
      ;;
    --stage)
      STAGE="$2"
      shift 2
      ;;
    --output-root)
      OUTPUT_ROOT="$2"
      shift 2
      ;;
    --dataset-h5)
      DATASET_H5="$2"
      shift 2
      ;;
    --device)
      DEVICE="$2"
      shift 2
      ;;
    --seed)
      SEED="$2"
      shift 2
      ;;
    --epochs)
      EPOCHS="$2"
      shift 2
      ;;
    --batch-size)
      BATCH_SIZE="$2"
      shift 2
      ;;
    --extract-batch-size)
      EXTRACT_BATCH_SIZE="$2"
      shift 2
      ;;
    --extract-prefetch)
      EXTRACT_PREFETCH="$2"
      shift 2
      ;;
    --data-parallel)
      DATA_PARALLEL="1"
      shift
      ;;
    --num-eval)
      NUM_EVAL="$2"
      shift 2
      ;;
    --eval-budget)
      EVAL_BUDGET="$2"
      shift 2
      ;;
    --goal-offset)
      GOAL_OFFSET="$2"
      shift 2
      ;;
    --compare)
      EVAL_MODE="compare"
      shift
      ;;
    --idm-only)
      EVAL_MODE="idm_only"
      shift
      ;;
    --cem-only)
      EVAL_MODE="cem_only"
      shift
      ;;
    --force-extract)
      FORCE_EXTRACT="1"
      shift
      ;;
    --force-train)
      FORCE_TRAIN="1"
      shift
      ;;
    --dry-run)
      DRY_RUN="1"
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      usage >&2
      exit 1
      ;;
  esac
done

if [[ -z "${TASK}" ]]; then
  usage >&2
  exit 1
fi

if [[ ! -x "${PYTHON_BIN}" ]]; then
  echo "Python executable not found: ${PYTHON_BIN}" >&2
  exit 1
fi

if [[ ! -d "${PAPER_ROOT}" ]]; then
  echo "Paper repo not found: ${PAPER_ROOT}" >&2
  exit 1
fi

canonicalize_task() {
  local raw
  raw="$(echo "$1" | tr '[:upper:]' '[:lower:]')"
  case "${raw}" in
    all)
      echo "all"
      ;;
    cube)
      echo "cube"
      ;;
    pusht)
      echo "pusht"
      ;;
    tworoom|two-room|two_room)
      echo "tworoom"
      ;;
    reacher|researcher)
      echo "reacher"
      ;;
    *)
      echo "Unsupported task: $1" >&2
      exit 1
      ;;
  esac
}

checkpoint_for_task() {
  case "$1" in
    cube)
      echo "/data/ykz/cube/lewm_epoch_27_object.ckpt"
      ;;
    pusht)
      echo "/data/ykz/pusht/lewm_epoch_100_object.ckpt"
      ;;
    reacher)
      echo "/data/ykz/reacher/lewm_epoch_29_object.ckpt"
      ;;
    tworoom)
      echo "/data/ykz/tworoom/lewm_epoch_67_object.ckpt"
      ;;
    *)
      echo "Unsupported task: $1" >&2
      exit 1
      ;;
  esac
}

action_dim_for_task() {
  case "$1" in
    cube)
      echo "5"
      ;;
    pusht|reacher|tworoom)
      echo "2"
      ;;
    *)
      echo "Unsupported task: $1" >&2
      exit 1
      ;;
  esac
}

run_cmd() {
  local label="$1"
  local log_path="$2"
  shift 2

  echo
  echo "[pipeline] ${label}"
  printf '[command]'
  printf ' %q' "$@"
  printf '\n'
  echo "[log] ${log_path}"

  if [[ "${DRY_RUN}" == "1" ]]; then
    return 0
  fi

  PYTHONPATH="${REPO_ROOT}:${PAPER_ROOT}:${PYTHONPATH:-}" \
    PYTHONUNBUFFERED=1 MPLCONFIGDIR=/tmp/matplotlib-cache "$@" 2>&1 | tee "${log_path}"
}

should_run_stage() {
  local wanted="$1"
  [[ "${STAGE}" == "all" || "${STAGE}" == "${wanted}" ]]
}

run_task() {
  local task="$1"
  local action_dim
  local checkpoint
  local task_root
  local log_dir
  local emb_path
  local idm_path

  action_dim="$(action_dim_for_task "${task}")"
  checkpoint="$(checkpoint_for_task "${task}")"
  task_root="${OUTPUT_ROOT}/${task}"
  log_dir="${task_root}/logs"
  emb_path="${task_root}/${task}_embeddings.npz"
  idm_path="${task_root}/${task}_gcidm.pt"

  if [[ "${DRY_RUN}" != "1" ]]; then
    mkdir -p "${task_root}" "${log_dir}" /tmp/matplotlib-cache
  fi

  echo
  echo "[task] ${task}"
  echo "[paths] checkpoint=${checkpoint}"
  if [[ -n "${DATASET_H5}" ]]; then
    echo "[paths] dataset_h5=${DATASET_H5}"
  else
    echo "[paths] dataset_h5=default-local-mapping"
  fi
  echo "[paths] embeddings=${emb_path}"
  echo "[paths] idm=${idm_path}"

  if should_run_stage extract; then
    if [[ -f "${emb_path}" && "${FORCE_EXTRACT}" != "1" ]]; then
      echo "[skip] extract existing=${emb_path}"
    else
      local extract_dataset_args=()
      local extract_parallel_args=()
      if [[ -n "${DATASET_H5}" ]]; then
        extract_dataset_args+=(--h5 "${DATASET_H5}")
      else
        extract_dataset_args+=(--dataset "${task}")
      fi
      if [[ "${DATA_PARALLEL}" == "1" ]]; then
        extract_parallel_args+=(--data-parallel)
      fi
      run_cmd "${task}:extract" "${log_dir}/extract.log" \
        "${PYTHON_BIN}" "${PAPER_ROOT}/train_idm.py" extract \
        --checkpoint "${checkpoint}" \
        "${extract_dataset_args[@]}" \
        --output "${emb_path}" \
        --batch-size "${EXTRACT_BATCH_SIZE}" \
        --num-prefetch "${EXTRACT_PREFETCH}" \
        "${extract_parallel_args[@]}" \
        --device "${DEVICE}"
    fi
  fi

  if should_run_stage train; then
    if [[ -f "${idm_path}" && "${FORCE_TRAIN}" != "1" ]]; then
      echo "[skip] train existing=${idm_path}"
    else
      run_cmd "${task}:train" "${log_dir}/train.log" \
        "${PYTHON_BIN}" "${PAPER_ROOT}/train_idm.py" train \
        --embeddings "${emb_path}" \
        --output "${idm_path}" \
        --embed-dim 192 \
        --action-dim "${action_dim}" \
        --frameskip 1 \
        --hidden-dim 512 \
        --n-layers 3 \
        --noise-sigma 0.0 \
        --max-goal-horizon 50 \
        --lr 1e-3 \
        --batch-size "${BATCH_SIZE}" \
        --epochs "${EPOCHS}" \
        --seed "${SEED}" \
        --device "${DEVICE}"
    fi
  fi

  if should_run_stage eval; then
    local eval_args=()
    case "${EVAL_MODE}" in
      compare)
        eval_args+=(--compare)
        ;;
      idm_only)
        ;;
      cem_only)
        eval_args+=(--cem-only)
        ;;
      *)
        echo "Unsupported eval mode: ${EVAL_MODE}" >&2
        exit 1
        ;;
    esac

    if [[ "${EVAL_MODE}" != "cem_only" ]]; then
      eval_args+=(--idm "${idm_path}")
    fi
    if [[ -n "${DATASET_H5}" ]]; then
      eval_args+=(--dataset-h5 "${DATASET_H5}")
    fi

    run_cmd "${task}:eval" "${log_dir}/eval.log" \
      "${PYTHON_BIN}" "${PAPER_ROOT}/eval_idm.py" \
      --dataset "${task}" \
      --checkpoint "${checkpoint}" \
      "${eval_args[@]}" \
      --num-eval "${NUM_EVAL}" \
      --eval-budget "${EVAL_BUDGET}" \
      --goal-offset "${GOAL_OFFSET}" \
      --seed "${SEED}" \
      --device "${DEVICE}"
  fi

  echo "[task-result] task=${task} output=${task_root} logs=${log_dir}"
}

TASK_CANONICAL="$(canonicalize_task "${TASK}")"
case "${STAGE}" in
  all|extract|train|eval)
    ;;
  *)
    echo "Unsupported stage: ${STAGE}" >&2
    usage >&2
    exit 1
    ;;
esac

if [[ "${TASK_CANONICAL}" == "all" ]]; then
  if [[ -n "${DATASET_H5}" ]]; then
    echo "--dataset-h5 can only be used with a single --task, not --task all." >&2
    exit 1
  fi
  TASKS=(tworoom pusht cube reacher)
else
  TASKS=("${TASK_CANONICAL}")
fi

total="${#TASKS[@]}"
index=0
for task in "${TASKS[@]}"; do
  index=$((index + 1))
  echo
  printf '[%02d/%02d] starting task=%s stage=%s\n' "${index}" "${total}" "${task}" "${STAGE}"
  run_task "${task}"
done

echo
echo "[pipeline-result] tasks=${TASKS[*]} output_root=${OUTPUT_ROOT}"
