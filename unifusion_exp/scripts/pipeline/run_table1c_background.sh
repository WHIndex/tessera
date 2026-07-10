#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-/home/anaconda/envs/graphrag-yyq/bin/python}"
RUN_TAG="${1:-}"

if [[ -z "${RUN_TAG}" ]]; then
  echo "Usage: bash scripts/pipeline/run_table1c_background.sh <run_tag> -- <run_e2e_table1c_args...>"
  echo "Example:"
  echo "  bash scripts/pipeline/run_table1c_background.sh q120_seed20260405 -- \\\n+    --model-dir /path/to/e5-large-v2 \\\n+    --split-file /path/to/split.json \\\n+    --corpus-file /path/to/corpus.json \\\n+    --out-dir artifacts/results/q120_demo"
  exit 1
fi
shift

if [[ "${1:-}" == "--" ]]; then
  shift
fi

if [[ "$#" -eq 0 ]]; then
  echo "Error: missing run_e2e_table1c arguments after run_tag."
  exit 1
fi

mkdir -p "${ROOT_DIR}/logs"
LOG_FILE="${ROOT_DIR}/logs/table1c_${RUN_TAG}.log"
PID_FILE="${ROOT_DIR}/logs/table1c_${RUN_TAG}.pid"

PROGRESS_EVERY="${PROGRESS_EVERY:-10}"
PROGRESS_MIN_SECONDS="${PROGRESS_MIN_SECONDS:-60}"

nohup env PYTHONPATH="${ROOT_DIR}/src" "${PYTHON_BIN}" "${ROOT_DIR}/scripts/eval/run_e2e_table1c.py" \
  "$@" \
  --progress-every "${PROGRESS_EVERY}" \
  --progress-min-seconds "${PROGRESS_MIN_SECONDS}" \
  > "${LOG_FILE}" 2>&1 &

PID=$!
echo "${PID}" > "${PID_FILE}"

echo "[started] pid=${PID}"
echo "[log] ${LOG_FILE}"
echo "[pid-file] ${PID_FILE}"
echo "[watch] tail -f ${LOG_FILE}"
echo "[progress-only] grep -E '\\[progress\\]|\\[done\\]' ${LOG_FILE}"
