#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  scripts/convert_visual_ogbench_datasets.sh [--chunk-size 50000] [--target-mode index|repeated|none] [--next-observation-mode skip|write] [--npz-cache-root /data/ykz] [--only scene|puzzle|antmaze]

Converts the local visual OGBench NPZ files into HDF5 files with pixels and episode metadata.
Each dataset prints an outer stage line; the Python converter prints per-stage and per-chunk progress.
The default target mode stores target_index instead of duplicating one final target image per transition.
The default next-observation mode skips next_pixels because LeWM HDF5 datasets use pixels and episode metadata.
Large compressed NPZ members are extracted into per-dataset .cache directories first so conversion can mmap them.
EOF
}

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${REPO_ROOT}/.venv/bin/python"
CHUNK_SIZE="50000"
TARGET_MODE="index"
TARGET_CHUNK_SIZE="5000"
NEXT_OBSERVATION_MODE="skip"
NPZ_CACHE_ROOT="/data/ykz"
ONLY=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --chunk-size)
      CHUNK_SIZE="$2"
      shift 2
      ;;
    --target-mode)
      TARGET_MODE="$2"
      shift 2
      ;;
    --target-chunk-size)
      TARGET_CHUNK_SIZE="$2"
      shift 2
      ;;
    --next-observation-mode)
      NEXT_OBSERVATION_MODE="$2"
      shift 2
      ;;
    --npz-cache-root)
      NPZ_CACHE_ROOT="$2"
      shift 2
      ;;
    --only)
      ONLY="$2"
      shift 2
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

if [[ ! -x "${PYTHON_BIN}" ]]; then
  echo "Python executable not found: ${PYTHON_BIN}" >&2
  exit 1
fi

case "${TARGET_MODE}" in
  index|repeated|none)
    ;;
  *)
    echo "Unsupported --target-mode value: ${TARGET_MODE}" >&2
    usage >&2
    exit 1
    ;;
esac

case "${NEXT_OBSERVATION_MODE}" in
  skip|write)
    ;;
  *)
    echo "Unsupported --next-observation-mode value: ${NEXT_OBSERVATION_MODE}" >&2
    usage >&2
    exit 1
    ;;
esac

declare -a DATASETS=(
  "scene|/data/ykz/scene/visual-scene-play-v0.npz|/data/ykz/scene/visual-scene-play-v0.h5|visual-scene-play-v0"
  "puzzle|/data/ykz/puzzle/visual-puzzle-3x3-play-v0.npz|/data/ykz/puzzle/visual-puzzle-3x3-play-v0.h5|visual-puzzle-3x3-play-v0"
  "antmaze|/data/ykz/antmaze/visual-antmaze-large-navigate-v0.npz|/data/ykz/antmaze/visual-antmaze-large-navigate-v0.h5|visual-antmaze-large-navigate-v0"
)

if [[ -n "${ONLY}" ]]; then
  case "${ONLY}" in
    scene|puzzle|antmaze)
      ;;
    *)
      echo "Unsupported --only value: ${ONLY}" >&2
      usage >&2
      exit 1
      ;;
  esac
fi

selected=()
for item in "${DATASETS[@]}"; do
  IFS='|' read -r name input_npz output_h5 dataset_name <<<"${item}"
  if [[ -z "${ONLY}" || "${ONLY}" == "${name}" ]]; then
    selected+=("${item}")
  fi
done

total="${#selected[@]}"
index=0
for item in "${selected[@]}"; do
  IFS='|' read -r name input_npz output_h5 dataset_name <<<"${item}"
  index=$((index + 1))
  cache_dir="${NPZ_CACHE_ROOT}/${name}/${dataset_name}.cache"
  printf '\n[%02d/%02d] convert %s\n' "${index}" "${total}" "${name}"
  printf '[cache] %q\n' "${cache_dir}"
  printf '[command] %q %q --input-npz %q --output-h5 %q --dataset-name %q --observation-output-key pixels --chunk-size %q --target-mode %q --target-chunk-size %q --next-observation-mode %q --npz-cache-dir %q\n' \
    "${PYTHON_BIN}" "${REPO_ROOT}/scripts/convert_ogbench_npz_to_hdf5.py" \
    "${input_npz}" "${output_h5}" "${dataset_name}" "${CHUNK_SIZE}" "${TARGET_MODE}" "${TARGET_CHUNK_SIZE}" "${NEXT_OBSERVATION_MODE}" "${cache_dir}"
  "${PYTHON_BIN}" "${REPO_ROOT}/scripts/convert_ogbench_npz_to_hdf5.py" \
    --input-npz "${input_npz}" \
    --output-h5 "${output_h5}" \
    --dataset-name "${dataset_name}" \
    --observation-output-key pixels \
    --chunk-size "${CHUNK_SIZE}" \
    --target-mode "${TARGET_MODE}" \
    --target-chunk-size "${TARGET_CHUNK_SIZE}" \
    --next-observation-mode "${NEXT_OBSERVATION_MODE}" \
    --npz-cache-dir "${cache_dir}"
done

echo
echo "[convert-result] converted=${total}"
