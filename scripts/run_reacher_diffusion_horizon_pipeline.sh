#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  scripts/run_reacher_diffusion_horizon_pipeline.sh \
    --horizon 5|10|15|... \
    [--input-h5 /data/ykz/reacher/reacher.h5] \
    [--wm-policy /data/ykz/reacher/lewm_epoch_29] \
    [--output-root /data/ykz/reacher/diffusion_pipeline] \
    [--num-samples 200000] \
    [--num-anchors 128] \
    [--epochs 80] \
    [--device cuda] \
    [--dry-run]

Builds and trains a Reacher diffusion backbone for a flattened action horizon.
The user-facing structural knob is only --horizon. Internally, Reacher keeps
the tested action_block=5 rollout contract, so --horizon must be a multiple of 5.

This script runs:
  1. planners/build_single_peak_dataset.py with --plan-horizon
  2. planners/build_action_anchors.py
  3. train_diffusion_planner.py with the stable simple_bce diffusion backbone

It does not fine-tune the score head.
EOF
}

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${REPO_ROOT}/.venv/bin/python"

HORIZON=""
INPUT_H5="/data/ykz/reacher/reacher.h5"
WM_POLICY="/data/ykz/reacher/lewm_epoch_29"
OUTPUT_ROOT="/data/ykz/reacher/diffusion_pipeline"
NUM_SAMPLES="200000"
NUM_ANCHORS="128"
EPOCHS="80"
BUILD_BATCH_SIZE="128"
TRAIN_BATCH_SIZE="64"
VAL_BATCH_SIZE="128"
NUM_WORKERS="4"
SEED="42"
DEVICE="cuda"
DRY_RUN="0"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --horizon)
      HORIZON="$2"
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
    --output-root)
      OUTPUT_ROOT="$2"
      shift 2
      ;;
    --num-samples)
      NUM_SAMPLES="$2"
      shift 2
      ;;
    --num-anchors)
      NUM_ANCHORS="$2"
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
    --seed)
      SEED="$2"
      shift 2
      ;;
    --device)
      DEVICE="$2"
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

if [[ -z "${HORIZON}" ]]; then
  usage >&2
  exit 1
fi

if [[ ! "${HORIZON}" =~ ^[0-9]+$ || "${HORIZON}" -le 0 ]]; then
  echo "--horizon must be a positive integer, got '${HORIZON}'." >&2
  exit 1
fi

ACTION_BLOCK="5"
if (( HORIZON % ACTION_BLOCK != 0 )); then
  echo "--horizon must be a multiple of ${ACTION_BLOCK}, got ${HORIZON}." >&2
  exit 1
fi
RECEDING_HORIZON="$((HORIZON / ACTION_BLOCK))"

if [[ ! -x "${PYTHON_BIN}" ]]; then
  echo "Python executable not found: ${PYTHON_BIN}" >&2
  exit 1
fi

if (( NUM_SAMPLES % 1000 == 0 )); then
  SAMPLE_TAG="$((NUM_SAMPLES / 1000))k"
else
  SAMPLE_TAG="${NUM_SAMPLES}"
fi

HORIZON_TAG="h${HORIZON}"
RUN_NAME="reacher_${HORIZON_TAG}_diffusion_${SAMPLE_TAG}_simple_bce_k${NUM_ANCHORS}_raw"
PLANNER_DATASET_PATH="${OUTPUT_ROOT}/single_peak_reacher_traj_${HORIZON_TAG}_${SAMPLE_TAG}_raw.pt"
ANCHOR_BUNDLE_PATH="${OUTPUT_ROOT}/reacher_action_anchors_${HORIZON_TAG}_${SAMPLE_TAG}_k${NUM_ANCHORS}_raw.pt"
DIFFUSION_OUTPUT_DIR="${OUTPUT_ROOT}/${RUN_NAME}"
DIFFUSION_BUNDLE_PATH="${DIFFUSION_OUTPUT_DIR}/diffusion_planner_best_bundle.pt"
PIPELINE_LOG_DIR="${DIFFUSION_OUTPUT_DIR}/pipeline_logs"
PIPELINE_SUMMARY_PATH="${DIFFUSION_OUTPUT_DIR}/pipeline_summary.txt"

timestamp() {
  date '+%Y-%m-%d %H:%M:%S'
}

format_cmd() {
  local formatted=""
  for arg in "$@"; do
    if [[ -n "${formatted}" ]]; then
      formatted+=" "
    fi
    printf -v quoted '%q' "${arg}"
    formatted+="${quoted}"
  done
  printf '%s' "${formatted}"
}

log_line() {
  local line="$1"
  echo "${line}"
  if [[ "${DRY_RUN}" == "1" ]]; then
    return
  fi
  mkdir -p "${PIPELINE_LOG_DIR}"
  printf '%s\n' "${line}" >> "${PIPELINE_SUMMARY_PATH}"
}

run_stage() {
  local stage_name="$1"
  local log_path="$2"
  shift 2
  local cmd=( "$@" )
  local cmd_text
  cmd_text="$(format_cmd "${cmd[@]}")"

  log_line "[stage-start] time=$(timestamp) stage=${stage_name}"
  echo
  echo "[run:${stage_name}] ${cmd_text}"

  if [[ "${DRY_RUN}" == "1" ]]; then
    log_line "[stage-dry-run] time=$(timestamp) stage=${stage_name} command=${cmd_text}"
    return 0
  fi

  mkdir -p "${PIPELINE_LOG_DIR}"
  set +e
  "${cmd[@]}" 2>&1 | tee "${log_path}"
  local cmd_status=${PIPESTATUS[0]}
  set -e

  if [[ ${cmd_status} -ne 0 ]]; then
    log_line "[stage-fail] time=$(timestamp) stage=${stage_name} exit_code=${cmd_status} log=${log_path}"
    return "${cmd_status}"
  fi

  log_line "[stage-done] time=$(timestamp) stage=${stage_name} status=success log=${log_path}"
}

BUILD_DATASET_CMD=(
  "${PYTHON_BIN}" "${REPO_ROOT}/planners/build_single_peak_dataset.py"
  --mode build
  --task reacher
  --label-source trajectory
  --wm-policy "${WM_POLICY}"
  --dataset-name dmc/reacher_random
  --dataset-h5 "${INPUT_H5}"
  --cache-dir "$(dirname -- "${INPUT_H5}")"
  --num-samples "${NUM_SAMPLES}"
  --batch-size "${BUILD_BATCH_SIZE}"
  --device "${DEVICE}"
  --on-error skip
  --plan-horizon "${HORIZON}"
  --action-block "${ACTION_BLOCK}"
  --output-path "${PLANNER_DATASET_PATH}"
)

BUILD_ANCHORS_CMD=(
  "${PYTHON_BIN}" "${REPO_ROOT}/planners/build_action_anchors.py"
  --mode build
  --dataset-path "${PLANNER_DATASET_PATH}"
  --output-path "${ANCHOR_BUNDLE_PATH}"
  --num-anchors "${NUM_ANCHORS}"
  --seed "${SEED}"
  --max-samples "${NUM_SAMPLES}"
)

TRAIN_DIFFUSION_CMD=(
  "${PYTHON_BIN}" "${REPO_ROOT}/train_diffusion_planner.py"
  --dataset-path "${PLANNER_DATASET_PATH}"
  --anchor-bundle-path "${ANCHOR_BUNDLE_PATH}"
  --output-dir "${DIFFUSION_OUTPUT_DIR}"
  --seed "${SEED}"
  --device "${DEVICE}"
  --epochs "${EPOCHS}"
  --batch-size "${TRAIN_BATCH_SIZE}"
  --val-batch-size "${VAL_BATCH_SIZE}"
  --num-workers "${NUM_WORKERS}"
  --loss-preset simple_bce
  --cls-loss-type bce
  --goal-loss-weight 0.0
)

echo "[task] task=reacher horizon=${HORIZON} receding_horizon=${RECEDING_HORIZON} action_block=${ACTION_BLOCK}"
echo "[paths] input_h5=${INPUT_H5}"
echo "[paths] wm_policy=${WM_POLICY}"
echo "[paths] planner_dataset=${PLANNER_DATASET_PATH}"
echo "[paths] anchor_bundle=${ANCHOR_BUNDLE_PATH}"
echo "[paths] diffusion_output_dir=${DIFFUSION_OUTPUT_DIR}"
echo "[paths] diffusion_bundle=${DIFFUSION_BUNDLE_PATH}"
echo "[paths] pipeline_log_dir=${PIPELINE_LOG_DIR}"
echo "[config] num_samples=${NUM_SAMPLES} num_anchors=${NUM_ANCHORS} epochs=${EPOCHS} seed=${SEED} device=${DEVICE}"

if [[ "${DRY_RUN}" != "1" ]]; then
  mkdir -p "${PIPELINE_LOG_DIR}"
  : > "${PIPELINE_SUMMARY_PATH}"
fi

run_stage "build_dataset" "${PIPELINE_LOG_DIR}/01_build_dataset.log" "${BUILD_DATASET_CMD[@]}"
run_stage "build_anchors" "${PIPELINE_LOG_DIR}/02_build_anchors.log" "${BUILD_ANCHORS_CMD[@]}"
run_stage "train_diffusion" "${PIPELINE_LOG_DIR}/03_train_diffusion.log" "${TRAIN_DIFFUSION_CMD[@]}"

log_line "[pipeline-result] time=$(timestamp) task=reacher horizon=${HORIZON} receding_horizon=${RECEDING_HORIZON} action_block=${ACTION_BLOCK} planner_dataset=${PLANNER_DATASET_PATH} anchor_bundle=${ANCHOR_BUNDLE_PATH} diffusion_bundle=${DIFFUSION_BUNDLE_PATH}"

echo
echo "[done] horizon diffusion backbone pipeline prepared for reacher"
echo "[done] diffusion_bundle=${DIFFUSION_BUNDLE_PATH}"
