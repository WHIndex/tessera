#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
RUN_TAG="${1:-}"

if [[ -f "${ROOT_DIR}/configs/paths.env" ]]; then
  # shellcheck disable=SC1090
  source "${ROOT_DIR}/configs/paths.env"
fi

if [[ -z "${RUN_TAG}" ]]; then
  echo "Usage: bash scripts/pipeline/run_i2m_lite_q120_screening_background.sh <run_tag>"
  echo "Example:"
  echo "  EVAL_PROGRESS_EVERY=10 EVAL_PROGRESS_MIN_SECONDS=90 RUN_BOOTSTRAP=1 \\"
  echo "  BOOTSTRAP_PROGRESS_EVERY=500 BOOTSTRAP_PROGRESS_MIN_SECONDS=30 \\"
  echo "  bash scripts/pipeline/run_i2m_lite_q120_screening_background.sh i2m_q120_$(date +%Y%m%d)"
  exit 1
fi

mkdir -p "${ROOT_DIR}/logs"
LOG_FILE="${ROOT_DIR}/logs/i2m_q120_screening_${RUN_TAG}.log"
PID_FILE="${ROOT_DIR}/logs/i2m_q120_screening_${RUN_TAG}.pid"

nohup env RUN_TAG="${RUN_TAG}" \
  bash "${ROOT_DIR}/scripts/pipeline/run_i2m_lite_q120_screening.sh" \
  > "${LOG_FILE}" 2>&1 &

PID=$!
echo "${PID}" > "${PID_FILE}"

echo "[started] pid=${PID}"
echo "[log] ${LOG_FILE}"
echo "[pid-file] ${PID_FILE}"
echo "[watch] tail -f ${LOG_FILE}"
echo "[progress-only] grep -E '\\[progress\\]|\\[bootstrap-progress\\]|\\[done\\]|\\[all_done\\]' ${LOG_FILE}"