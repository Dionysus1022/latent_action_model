#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${REPO_ROOT}/.venv/bin/python"

exec "${PYTHON_BIN}" "${REPO_ROOT}/scripts/run_comparison_experiments.py" "$@"
