#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
source "${ROOT_DIR}/configs/paths.env"

RUN_TAG="${1:-$(date +%Y%m%d_%H%M%S)}"
RUN_DIR="${ROOT_DIR}/runs/${RUN_TAG}_router_deberta"
mkdir -p "${RUN_DIR}"

export PYTHONPATH="${ROOT_DIR}/src:${PYTHONPATH:-}"
PYTHON_BIN="${PYTHON_BIN:-python}"

echo "[run] RUN_DIR=${RUN_DIR}"

echo "[1/3] build router dataset"
"${PYTHON_BIN}" "${ROOT_DIR}/scripts/data/build_router_dataset.py" \
  --mmrag-root "${MMRAG_DATA_ROOT}" \
  --out-dir "${RUN_DIR}/router_data"

echo "[2/3] smoke train (small sample)"
"${PYTHON_BIN}" "${ROOT_DIR}/scripts/train/train_router_deberta.py" \
  --model-dir "${DEBERTA_MODEL_DIR}" \
  --train-file "${RUN_DIR}/router_data/router_train.json" \
  --val-file "${RUN_DIR}/router_data/router_val.json" \
  --test-file "${RUN_DIR}/router_data/router_test.json" \
  --max-train "${ROUTER_SMOKE_TRAIN:-512}" \
  --max-val "${ROUTER_SMOKE_VAL:-256}" \
  --max-test "${ROUTER_SMOKE_TEST:-256}" \
  --epochs "${ROUTER_SMOKE_EPOCHS:-1}" \
  --batch-size "${ROUTER_SMOKE_BS:-8}" \
  --out-dir "${RUN_DIR}/router_metrics" \
  --run-id "router_deberta_smoke"

echo "[3/3] full train"
"${PYTHON_BIN}" "${ROOT_DIR}/scripts/train/train_router_deberta.py" \
  --model-dir "${DEBERTA_MODEL_DIR}" \
  --train-file "${RUN_DIR}/router_data/router_train.json" \
  --val-file "${RUN_DIR}/router_data/router_val.json" \
  --test-file "${RUN_DIR}/router_data/router_test.json" \
  --epochs "${ROUTER_EPOCHS:-3}" \
  --batch-size "${ROUTER_BS:-16}" \
  --lr "${ROUTER_LR:-2e-5}" \
  --threshold "${ROUTER_THRESHOLD:-0.5}" \
  --out-dir "${RUN_DIR}/router_metrics" \
  --run-id "router_deberta_full"

echo "[done] output at ${RUN_DIR}/router_metrics"
