#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
source "${ROOT_DIR}/configs/paths.env"

RUN_TAG="${1:-$(date +%Y%m%d_%H%M%S)}"
RUN_DIR="${ROOT_DIR}/runs/${RUN_TAG}_kg"
mkdir -p "${RUN_DIR}"

export PYTHONPATH="${ROOT_DIR}/src:${PYTHONPATH:-}"
PYTHON_BIN="${PYTHON_BIN:-python}"
TRIPLES_PATH="${TRIPLES_PATH:-${SIMKGC_ROOT}/data/FB15k237/train.txt}"

echo "[run] RUN_DIR=${RUN_DIR}"
echo "[run] TRIPLES_PATH=${TRIPLES_PATH}"

echo "[1/3] Extract mmRAG graph entity ids"
"${PYTHON_BIN}" "${ROOT_DIR}/scripts/data/extract_mmrag_graph_entities.py" \
  --mmrag-root "${MMRAG_DATA_ROOT}" \
  --out-file "${RUN_DIR}/mmrag_graph_entities.json"

echo "[2/3] Small-sample smoke test (dry-run subset build)"
"${PYTHON_BIN}" "${ROOT_DIR}/scripts/train/train_transe_subgraph.py" \
  --entity-id-file "${RUN_DIR}/mmrag_graph_entities.json" \
  --triples-tsv "${TRIPLES_PATH}" \
  --out-dir "${RUN_DIR}/smoke" \
  --max-triples 5000 \
  --dry-run

echo "[3/3] Main train (configurable)"
"${PYTHON_BIN}" "${ROOT_DIR}/scripts/train/train_transe_subgraph.py" \
  --entity-id-file "${RUN_DIR}/mmrag_graph_entities.json" \
  --triples-tsv "${TRIPLES_PATH}" \
  --out-dir "${RUN_DIR}/train" \
  --max-triples "${KG_MAX_TRIPLES:-120000}" \
  --embedding-dim "${KG_EMBEDDING_DIM:-100}" \
  --epochs "${KG_EPOCHS:-20}" \
  --batch-size "${KG_BATCH_SIZE:-512}" \
  --device "${KG_DEVICE:-gpu}"

echo "[done] KG artifacts: ${RUN_DIR}/train"
