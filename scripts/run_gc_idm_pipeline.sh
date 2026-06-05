#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  scripts/run_gc_idm_pipeline.sh \
    --task cube|pusht|tworoom|two-room|reacher \
    --input-h5 /path/to/dataset.h5 \
    --wm-policy /path/to/lewm_epoch_xxx \
    [--num-samples 200000] \
    [--max-horizon 50] \
    [--epochs 50] \
    [--build-batch-size 64] \
    [--train-batch-size 1024] \
    [--val-batch-size 1024] \
    [--num-workers 4] \
    [--eval-num-eval 50] \
    [--checkpoint-selection last|best] \
    [--device cuda] \
    [--seed 42] \
    [--dry-run]

Runs:
  1. scripts/build_gc_idm_dataset.py
  2. train_gc_idm.py
  3. eval.py eval_profile=gc_idm
EOF
}

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${REPO_ROOT}/.venv/bin/python"

TASK=""
INPUT_H5=""
WM_POLICY=""
NUM_SAMPLES="200000"
MAX_HORIZON="50"
EPOCHS="50"
BUILD_BATCH_SIZE="64"
TRAIN_BATCH_SIZE="1024"
VAL_BATCH_SIZE="1024"
NUM_WORKERS="4"
EVAL_NUM_EVAL="50"
CHECKPOINT_SELECTION="last"
DEVICE="cuda"
SEED="42"
DRY_RUN="0"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --task)
      TASK="$2"
      shift 2
      ;;
    --input-h5)
      INPUT_H5="$2"
      shift 2
      ;;
    --wm-policy)
      WM_POLICY="$2"
      shift 2
      ;;
    --num-samples)
      NUM_SAMPLES="$2"
      shift 2
      ;;
    --max-horizon)
      MAX_HORIZON="$2"
      shift 2
      ;;
    --epochs)
      EPOCHS="$2"
      shift 2
      ;;
    --build-batch-size)
      BUILD_BATCH_SIZE="$2"
      shift 2
      ;;
    --train-batch-size)
      TRAIN_BATCH_SIZE="$2"
      shift 2
      ;;
    --val-batch-size)
      VAL_BATCH_SIZE="$2"
      shift 2
      ;;
    --num-workers)
      NUM_WORKERS="$2"
      shift 2
      ;;
    --eval-num-eval)
      EVAL_NUM_EVAL="$2"
      shift 2
      ;;
    --checkpoint-selection)
      CHECKPOINT_SELECTION="$2"
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

if [[ -z "${TASK}" || -z "${INPUT_H5}" || -z "${WM_POLICY}" ]]; then
  usage >&2
  exit 1
fi

if [[ ! -x "${PYTHON_BIN}" ]]; then
  echo "Python executable not found: ${PYTHON_BIN}" >&2
  exit 1
fi

canonicalize_task() {
  local raw
  raw="$(echo "$1" | tr '[:upper:]' '[:lower:]')"
  case "${raw}" in
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

TASK_CANONICAL="$(canonicalize_task "${TASK}")"
TASK_ROOT="$(dirname -- "${WM_POLICY}")"
OUTPUT_ROOT="${TASK_ROOT}/gc_idm"
DATASET_PATH="${OUTPUT_ROOT}/${TASK_CANONICAL}_gc_idm_dataset_${NUM_SAMPLES}.pt"
BUNDLE_PATH="${OUTPUT_ROOT}/gc_idm_best_bundle.pt"
LOG_DIR="${OUTPUT_ROOT}/pipeline_logs"

if [[ "${DRY_RUN}" != "1" ]]; then
  mkdir -p "${OUTPUT_ROOT}" "${LOG_DIR}"
fi

run_cmd() {
  local label="$1"
  shift
  local log_path="${LOG_DIR}/${label}.log"

  echo
  echo "[pipeline] ${label}"
  printf '[command]'
  printf ' %q' "$@"
  printf '\n'
  echo "[log] ${log_path}"

  if [[ "${DRY_RUN}" == "1" ]]; then
    return 0
  fi

  GC_IDM_FORCE_PROGRESS=1 PYTHONUNBUFFERED=1 "$@" 2>&1 | tee "${log_path}"
}

run_cmd build_dataset \
  "${PYTHON_BIN}" "${REPO_ROOT}/scripts/build_gc_idm_dataset.py" \
  --task "${TASK_CANONICAL}" \
  --wm-policy "${WM_POLICY}" \
  --dataset-h5 "${INPUT_H5}" \
  --num-samples "${NUM_SAMPLES}" \
  --batch-size "${BUILD_BATCH_SIZE}" \
  --output-path "${DATASET_PATH}" \
  --device "${DEVICE}" \
  --max-horizon "${MAX_HORIZON}" \
  --seed "${SEED}"

run_cmd train \
  "${PYTHON_BIN}" "${REPO_ROOT}/train_gc_idm.py" \
  --dataset-path "${DATASET_PATH}" \
  --output-dir "${OUTPUT_ROOT}" \
  --device "${DEVICE}" \
  --epochs "${EPOCHS}" \
  --batch-size "${TRAIN_BATCH_SIZE}" \
  --val-batch-size "${VAL_BATCH_SIZE}" \
  --num-workers "${NUM_WORKERS}" \
  --seed "${SEED}" \
  --checkpoint-selection "${CHECKPOINT_SELECTION}" \
  --diagnose-horizon-buckets

run_cmd eval \
  "${PYTHON_BIN}" "${REPO_ROOT}/eval.py" \
  --config-name "${TASK_CANONICAL}" \
  eval_profile=gc_idm \
  "+dataset_h5=${INPUT_H5}" \
  "gc_idm_bundle=${BUNDLE_PATH}" \
  "eval.num_eval=${EVAL_NUM_EVAL}" \
  "trajectory_quality.enabled=true" \
  "trajectory_quality.save_video=false"

echo
echo "[pipeline-result] task=${TASK_CANONICAL} dataset=${DATASET_PATH} bundle=${BUNDLE_PATH} logs=${LOG_DIR}"
