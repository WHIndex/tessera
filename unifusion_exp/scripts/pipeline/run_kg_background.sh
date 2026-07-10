#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
RUN_TAG="${1:-$(date +%Y%m%d_%H%M%S)}"
LOG_DIR="${ROOT_DIR}/logs"
mkdir -p "${LOG_DIR}"
LOG_FILE="${LOG_DIR}/kg_${RUN_TAG}.log"

nohup bash "${ROOT_DIR}/scripts/pipeline/run_kg_experiments.sh" "${RUN_TAG}" >"${LOG_FILE}" 2>&1 &
PID=$!

echo "[OK] started background KG run"
echo "PID=${PID}"
echo "LOG=${LOG_FILE}"
echo "watch: tail -f ${LOG_FILE}"
