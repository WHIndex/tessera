#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
source "${ROOT_DIR}/configs/paths.env"

RUN_TAG="${1:-$(date +%Y%m%d_%H%M%S)}"
RUN_DIR="${ROOT_DIR}/runs/${RUN_TAG}"
mkdir -p "${RUN_DIR}"

export PYTHONPATH="${ROOT_DIR}/src:${PYTHONPATH:-}"
PYTHON_BIN="${PYTHON_BIN:-python}"

echo "[run] RUN_DIR=${RUN_DIR}"

echo "[1/4] Build router dataset"
"${PYTHON_BIN}" "${ROOT_DIR}/scripts/data/build_router_dataset.py" \
  --mmrag-root "${MMRAG_DATA_ROOT}" \
  --out-dir "${RUN_DIR}/router_data"

echo "[2/4] Quick smoke training (small sample)"
"${PYTHON_BIN}" "${ROOT_DIR}/scripts/train/train_router.py" \
  --train-file "${RUN_DIR}/router_data/router_train.json" \
  --val-file "${RUN_DIR}/router_data/router_val.json" \
  --test-file "${RUN_DIR}/router_data/router_test.json" \
  --max-train 512 \
  --max-val 256 \
  --max-test 256 \
  --out-dir "${RUN_DIR}/router_metrics" \
  --run-id "router_smoke"

echo "[3/4] Full-size validation sweep"
"${PYTHON_BIN}" "${ROOT_DIR}/scripts/eval/eval_router_thresholds.py" \
  --train-file "${RUN_DIR}/router_data/router_train.json" \
  --val-file "${RUN_DIR}/router_data/router_val.json" \
  --out-dir "${RUN_DIR}/router_metrics" \
  --thresholds "0.3,0.4,0.5,0.6,0.7"

echo "[4/4] Done: metrics at ${RUN_DIR}/router_metrics"
