#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  scripts/run_diffusion_pipeline.sh \
    --task pusht|tworoom|two-room|reacher|researcher \
    --input-h5 /path/to/input.h5 \
    --wm-policy /path/to/lewm_epoch_xxx \
    [--episode-key auto|ep_idx|episode_idx|...] \
    [--train-ratio 0.8] \
    [--seed 42] \
    [--num-samples 200000] \
    [--num-anchors 128] \
    [--epochs 80] \
    [--build-batch-size 128] \
    [--train-batch-size 64] \
    [--val-batch-size 128] \
    [--num-workers 4] \
    [--eval-num-eval 50] \
    [--device cuda] \
    [--use-split-dataset] \
    [--dry-run]

This script runs:
  1. planners/build_single_peak_dataset.py
  2. planners/build_action_anchors.py
  3. train_diffusion_planner.py
  4. eval.py (diffusion, wm_only)

By default it reads --input-h5 directly and does not split the dataset.
Pass --use-split-dataset to restore the old split-first behavior.
EOF
}

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${REPO_ROOT}/.venv/bin/python"

TASK=""
INPUT_H5=""
WM_POLICY=""
EPISODE_KEY="auto"
TRAIN_RATIO="0.8"
SEED="42"
NUM_SAMPLES="200000"
NUM_ANCHORS="128"
EPOCHS="80"
BUILD_BATCH_SIZE="128"
TRAIN_BATCH_SIZE="64"
VAL_BATCH_SIZE="128"
NUM_WORKERS="4"
EVAL_NUM_EVAL="50"
DEVICE="cuda"
USE_SPLIT_DATASET="0"
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
    --episode-key)
      EPISODE_KEY="$2"
      shift 2
      ;;
    --train-ratio)
      TRAIN_RATIO="$2"
      shift 2
      ;;
    --seed)
      SEED="$2"
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
    --eval-num-eval)
      EVAL_NUM_EVAL="$2"
      shift 2
      ;;
    --device)
      DEVICE="$2"
      shift 2
      ;;
    --use-split-dataset)
      USE_SPLIT_DATASET="1"
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

case "${TASK_CANONICAL}" in
  pusht)
    CONFIG_NAME="pusht"
    DATASET_NAME="pusht_expert_train"
    ;;
  tworoom)
    CONFIG_NAME="tworoom"
    DATASET_NAME="tworoom"
    ;;
  reacher)
    CONFIG_NAME="reacher"
    DATASET_NAME="dmc/reacher_random"
    ;;
esac

TASK_ROOT="$(dirname -- "${WM_POLICY}")"
DATASET_BASENAME="$(basename "${DATASET_NAME}")"
DIFFUSION_RUNS_ROOT="${TASK_ROOT}/diffusion_pipeline"
PIPELINE_MASTER_LOG_ROOT="${DIFFUSION_RUNS_ROOT}/${TASK_CANONICAL}_logs"

if (( NUM_SAMPLES % 1000 == 0 )); then
  SAMPLE_TAG="$((NUM_SAMPLES / 1000))k"
else
  SAMPLE_TAG="${NUM_SAMPLES}"
fi

TRAIN_SPLIT_ROOT="${TASK_ROOT}/splits/${DATASET_BASENAME}_train"
TEST_SPLIT_ROOT="${TASK_ROOT}/splits/${DATASET_BASENAME}_test"
TRAIN_H5="${TRAIN_SPLIT_ROOT}/${DATASET_NAME}.h5"
TEST_H5="${TEST_SPLIT_ROOT}/${DATASET_NAME}.h5"
if [[ "${USE_SPLIT_DATASET}" == "1" ]]; then
  DATASET_H5="${TRAIN_H5}"
  DATASET_CACHE_DIR="${TRAIN_SPLIT_ROOT}"
  EVAL_DATASET_H5="${TEST_H5}"
  EVAL_CACHE_DIR="${TEST_SPLIT_ROOT}"
  DATASET_MODE_TAG="splittrain"
else
  DATASET_H5="${INPUT_H5}"
  DATASET_CACHE_DIR="$(dirname -- "${INPUT_H5}")"
  EVAL_DATASET_H5="${INPUT_H5}"
  EVAL_CACHE_DIR="$(dirname -- "${INPUT_H5}")"
  DATASET_MODE_TAG="raw"
fi

PLANNER_DATASET_PATH="${TASK_ROOT}/single_peak_${TASK_CANONICAL}_traj_${SAMPLE_TAG}_${DATASET_MODE_TAG}.pt"
ANCHOR_BUNDLE_PATH="${TASK_ROOT}/${TASK_CANONICAL}_action_anchors_${SAMPLE_TAG}_k${NUM_ANCHORS}_${DATASET_MODE_TAG}.pt"
DIFFUSION_OUTPUT_DIR="${DIFFUSION_RUNS_ROOT}/${TASK_CANONICAL}_diffusion_${SAMPLE_TAG}_simple_bce_k${NUM_ANCHORS}_${DATASET_MODE_TAG}"
DIFFUSION_BUNDLE_PATH="${DIFFUSION_OUTPUT_DIR}/diffusion_planner_best_bundle.pt"
PIPELINE_LOG_DIR="${DIFFUSION_OUTPUT_DIR}/pipeline_logs"
PIPELINE_SUMMARY_PATH="${DIFFUSION_OUTPUT_DIR}/pipeline_summary.txt"
PIPELINE_MASTER_LOG="${PIPELINE_MASTER_LOG_ROOT}/pipeline.log"

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
  printf '%s\n' "${line}" >> "${PIPELINE_MASTER_LOG}"
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
  "${cmd[@]}" 2>&1 | tee "${log_path}" "${PIPELINE_MASTER_LOG}"
  local cmd_status=${PIPESTATUS[0]}
  set -e

  if [[ ${cmd_status} -ne 0 ]]; then
    log_line "[stage-fail] time=$(timestamp) stage=${stage_name} exit_code=${cmd_status} log=${log_path}"
    return "${cmd_status}"
  fi

  log_line "[stage-done] time=$(timestamp) stage=${stage_name} status=success log=${log_path}"
  return 0
}

emit_next_command() {
  local completed_stage="$1"
  local next_stage="$2"
  shift 2
  local cmd_text
  cmd_text="$(format_cmd "$@")"
  log_line "[next-cmd] time=$(timestamp) completed_stage=${completed_stage} next_stage=${next_stage} command=${cmd_text}"
}

extract_metric_from_log() {
  local pattern="$1"
  local log_path="$2"
  if [[ ! -f "${log_path}" ]]; then
    return 0
  fi
  grep -E "${pattern}" "${log_path}" | tail -n 1 | sed -E "s/${pattern}/\\1/" || true
}

write_final_summary() {
  local eval_log_path="$1"
  local success_rate="unknown"
  local evaluation_time="unknown"

  if [[ "${DRY_RUN}" == "0" && -f "${eval_log_path}" ]]; then
    local parsed_success_rate parsed_evaluation_time
    parsed_success_rate="$(extract_metric_from_log '.*\[summary\] success_rate=([0-9eE+.-]+).*' "${eval_log_path}")"
    parsed_evaluation_time="$(extract_metric_from_log '.*\[summary\] evaluation_time=([0-9eE+.-]+s).*' "${eval_log_path}")"
    if [[ -n "${parsed_success_rate}" ]]; then
      success_rate="${parsed_success_rate}"
    fi
    if [[ -n "${parsed_evaluation_time}" ]]; then
      evaluation_time="${parsed_evaluation_time}"
    fi
  fi

  log_line "[pipeline-result] time=$(timestamp) task=${TASK_CANONICAL} dataset_mode=${DATASET_MODE_TAG} dataset_h5=${DATASET_H5} eval_dataset_h5=${EVAL_DATASET_H5} split_train_h5=${TRAIN_H5} split_test_h5=${TEST_H5} planner_dataset=${PLANNER_DATASET_PATH} anchor_bundle=${ANCHOR_BUNDLE_PATH} diffusion_bundle=${DIFFUSION_BUNDLE_PATH} success_rate=${success_rate} evaluation_time=${evaluation_time}"
}

SPLIT_CMD=(
  "${PYTHON_BIN}" "${REPO_ROOT}/scripts/split_hdf5_by_episode.py"
  --input-h5 "${INPUT_H5}"
  --output-train-h5 "${TRAIN_H5}"
  --output-test-h5 "${TEST_H5}"
  --train-ratio "${TRAIN_RATIO}"
  --seed "${SEED}"
)
if [[ "${EPISODE_KEY}" != "auto" ]]; then
  SPLIT_CMD+=(--episode-key "${EPISODE_KEY}")
fi

BUILD_DATASET_CMD=(
  "${PYTHON_BIN}" "${REPO_ROOT}/planners/build_single_peak_dataset.py"
  --mode build
  --task "${TASK_CANONICAL}"
  --label-source trajectory
  --wm-policy "${WM_POLICY}"
  --dataset-name "${DATASET_NAME}"
  --dataset-h5 "${DATASET_H5}"
  --cache-dir "${DATASET_CACHE_DIR}"
  --num-samples "${NUM_SAMPLES}"
  --batch-size "${BUILD_BATCH_SIZE}"
  --device "${DEVICE}"
  --on-error skip
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

EVAL_DIFFUSION_CMD=(
  "${PYTHON_BIN}" "${REPO_ROOT}/eval.py"
  --config-name "${CONFIG_NAME}"
  planner_type=diffusion
  "policy=${WM_POLICY}"
  "diffusion_bundle=${DIFFUSION_BUNDLE_PATH}"
  diffusion_selection_mode=wm_only
  "diffusion_num_candidates=${NUM_ANCHORS}"
  "dataset_h5=${EVAL_DATASET_H5}"
  "cache_dir=${EVAL_CACHE_DIR}"
  "eval.num_eval=${EVAL_NUM_EVAL}"
)

echo "[task] task=${TASK_CANONICAL} config=${CONFIG_NAME} dataset_name=${DATASET_NAME}"
echo "[paths] input_h5=${INPUT_H5}"
echo "[paths] dataset_h5=${DATASET_H5}"
echo "[paths] eval_dataset_h5=${EVAL_DATASET_H5}"
echo "[paths] train_h5=${TRAIN_H5}"
echo "[paths] test_h5=${TEST_H5}"
echo "[paths] planner_dataset=${PLANNER_DATASET_PATH}"
echo "[paths] anchor_bundle=${ANCHOR_BUNDLE_PATH}"
echo "[paths] diffusion_output_dir=${DIFFUSION_OUTPUT_DIR}"
echo "[paths] diffusion_bundle=${DIFFUSION_BUNDLE_PATH}"
echo "[paths] pipeline_log_dir=${PIPELINE_LOG_DIR}"
echo "[paths] pipeline_summary=${PIPELINE_SUMMARY_PATH}"
echo "[paths] pipeline_full_log=${PIPELINE_MASTER_LOG}"
echo "[config] train_ratio=${TRAIN_RATIO} num_samples=${NUM_SAMPLES} num_anchors=${NUM_ANCHORS} epochs=${EPOCHS} eval.num_eval=${EVAL_NUM_EVAL}"

if [[ "${DRY_RUN}" != "1" ]]; then
  mkdir -p "${DIFFUSION_RUNS_ROOT}"
  mkdir -p "${PIPELINE_LOG_DIR}"
  mkdir -p "${PIPELINE_MASTER_LOG_ROOT}"
  : > "${PIPELINE_SUMMARY_PATH}"
  : > "${PIPELINE_MASTER_LOG}"
fi

SPLIT_LOG="${PIPELINE_LOG_DIR}/01_split.log"
BUILD_DATASET_LOG="${PIPELINE_LOG_DIR}/02_build_dataset.log"
BUILD_ANCHORS_LOG="${PIPELINE_LOG_DIR}/03_build_anchors.log"
TRAIN_DIFFUSION_LOG="${PIPELINE_LOG_DIR}/04_train_diffusion.log"
EVAL_DIFFUSION_LOG="${PIPELINE_LOG_DIR}/05_eval_diffusion.log"

if [[ "${USE_SPLIT_DATASET}" == "1" ]]; then
  run_stage "split" "${SPLIT_LOG}" "${SPLIT_CMD[@]}"
  emit_next_command "split" "build_dataset" "${BUILD_DATASET_CMD[@]}"
else
  log_line "[stage-skip] time=$(timestamp) stage=split reason=use_raw_dataset"
fi

run_stage "build_dataset" "${BUILD_DATASET_LOG}" "${BUILD_DATASET_CMD[@]}"
emit_next_command "build_dataset" "build_anchors" "${BUILD_ANCHORS_CMD[@]}"

run_stage "build_anchors" "${BUILD_ANCHORS_LOG}" "${BUILD_ANCHORS_CMD[@]}"
emit_next_command "build_anchors" "train_diffusion" "${TRAIN_DIFFUSION_CMD[@]}"

run_stage "train_diffusion" "${TRAIN_DIFFUSION_LOG}" "${TRAIN_DIFFUSION_CMD[@]}"
emit_next_command "train_diffusion" "eval_diffusion" "${EVAL_DIFFUSION_CMD[@]}"

run_stage "eval_diffusion" "${EVAL_DIFFUSION_LOG}" "${EVAL_DIFFUSION_CMD[@]}"
write_final_summary "${EVAL_DIFFUSION_LOG}"

echo
echo "[done] pipeline completed for task=${TASK_CANONICAL}"
if [[ "${DRY_RUN}" == "0" ]]; then
  echo "[done] summary=${PIPELINE_SUMMARY_PATH}"
  echo "[done] full_log=${PIPELINE_MASTER_LOG}"
fi
