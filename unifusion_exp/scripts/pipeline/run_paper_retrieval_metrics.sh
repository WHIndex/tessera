#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-${ROOT_DIR}/../.conda/unifusion-gpt4o/bin/python}"
MODEL_DIR="${MODEL_DIR:-${ROOT_DIR}/../downloaded_resource/bge-large-en-v1.5}"
MMRAG_ROOT="${MMRAG_ROOT:-${ROOT_DIR}/../downloaded_resource/mmRAG/data/mmRAG_ds}"
RUN_PROFILE="${RUN_PROFILE:-smoke}"

if [[ "${RUN_PROFILE}" == "full" ]]; then
  SPLIT_FILE="${SPLIT_FILE:-${MMRAG_ROOT}/mmrag_test.json}"
  CORPUS_FILE="${CORPUS_FILE:-${MMRAG_ROOT}/processed_documents.json}"
  OUT_DIR="${OUT_DIR:-${ROOT_DIR}/artifacts/results/paper_retrieval_metrics_test1286_full}"
  MAX_QUERIES="${MAX_QUERIES:-1286}"
else
  SPLIT_FILE="${SPLIT_FILE:-${MMRAG_ROOT}/mmrag_dev.json}"
  CORPUS_FILE="${CORPUS_FILE:-${ROOT_DIR}/artifacts/retrieval/corpus_subset_devpos_v2_5000.json}"
  OUT_DIR="${OUT_DIR:-${ROOT_DIR}/artifacts/results/paper_retrieval_metrics_smoke_dev10}"
  MAX_QUERIES="${MAX_QUERIES:-10}"
fi

SPARSE_BACKEND="${SPARSE_BACKEND:-bm25}"
METHODS="${METHODS:-}"
INCLUDE_UNIHGKR="${INCLUDE_UNIHGKR:-0}"
EXTRA_RETRIEVAL_ARGS="${EXTRA_RETRIEVAL_ARGS:-}"

if [[ "${CORPUS_FILE}" == *"processed_documents.json" ]] && [[ "${ALLOW_FULL_CORPUS:-0}" != "1" ]]; then
  echo "[block] You are about to run the full mmRAG corpus: ${CORPUS_FILE}"
  echo "[block] Set ALLOW_FULL_CORPUS=1 to continue."
  exit 2
fi

cd "${ROOT_DIR}"
mkdir -p "${OUT_DIR}"

CMD=(
  "${PYTHON_BIN}" scripts/eval/eval_paper_retrieval_metrics.py
  --model-dir "${MODEL_DIR}"
  --split-file "${SPLIT_FILE}"
  --corpus-file "${CORPUS_FILE}"
  --out-json "${OUT_DIR}/paper_retrieval_metrics.json"
  --out-csv "${OUT_DIR}/paper_retrieval_metrics.csv"
  --out-md "${OUT_DIR}/paper_retrieval_metrics.md"
  --detail-json "${OUT_DIR}/paper_retrieval_metrics_detail.json"
  --max-queries "${MAX_QUERIES}"
  --sparse-backend "${SPARSE_BACKEND}"
)

if [[ -n "${METHODS}" ]]; then
  CMD+=(--methods "${METHODS}")
fi

if [[ "${INCLUDE_UNIHGKR}" == "1" ]]; then
  CMD+=(--include-unihgkr)
fi

if [[ -n "${EXTRA_RETRIEVAL_ARGS}" ]]; then
  # shellcheck disable=SC2206
  EXTRA_ARGS=( ${EXTRA_RETRIEVAL_ARGS} )
  CMD+=("${EXTRA_ARGS[@]}")
fi

"${CMD[@]}"

echo "[DONE] paper retrieval metrics -> ${OUT_DIR}"
