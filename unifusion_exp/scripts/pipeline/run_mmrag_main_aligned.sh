#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-/home/anaconda/envs/graphrag-yyq/bin/python}"
MODEL_DIR="${MODEL_DIR:-${ROOT_DIR}/../downloaded_resource/e5-large-v2}"
MMRAG_ROOT="${MMRAG_ROOT:-${ROOT_DIR}/../downloaded_resource/mmRAG/data/mmRAG_ds}"
SPLIT_FILE="${SPLIT_FILE:-${MMRAG_ROOT}/mmrag_test.json}"
CORPUS_FILE="${CORPUS_FILE:-${MMRAG_ROOT}/processed_documents.json}"
OUT_DIR="${OUT_DIR:-${ROOT_DIR}/artifacts/results/table1c_e2e_mmrag_main_aligned}"
MAX_QUERIES="${MAX_QUERIES:-1286}"
METHODS="${METHODS:-dense_concat,unifusion_rag,oracle_gold}"
REUSE_METHODS="${REUSE_METHODS:-}"
INCLUDE_ORACLE_MEASURED="${INCLUDE_ORACLE_MEASURED:-1}"
EXTRA_E2E_ARGS="${EXTRA_E2E_ARGS:-}"
READER="${READER:-ollama}"
OLLAMA_HOST="${OLLAMA_HOST:-http://127.0.0.1:11434}"
OLLAMA_MODEL="${OLLAMA_MODEL:-qwen2.5-7B-Instruct:latest}"
OPENAI_MODEL="${OPENAI_MODEL:-gpt-4o-mini}"
OPENAI_API_KEY_ENV="${OPENAI_API_KEY_ENV:-OPENAI_API_KEY}"
OPENAI_BASE_URL="${OPENAI_BASE_URL:-}"
OPENAI_TIMEOUT="${OPENAI_TIMEOUT:-120}"
OPENAI_TEMPERATURE="${OPENAI_TEMPERATURE:-0}"
OPENAI_MAX_TOKENS="${OPENAI_MAX_TOKENS:-64}"
OPENAI_MAX_RETRIES="${OPENAI_MAX_RETRIES:-3}"
OPENAI_RETRY_BACKOFF="${OPENAI_RETRY_BACKOFF:-2}"
OPENAI_FAIL_SOFT="${OPENAI_FAIL_SOFT:-0}"
UNIHGKR_MODEL_DIR="${UNIHGKR_MODEL_DIR:-${ROOT_DIR}/../downloaded_resource/compmix-ir-benchmarks/ZhishanQ-UniHGKR-base}"

if [[ "${CORPUS_FILE}" == *"corpus_subset_v1.json" ]]; then
  echo "[warn] CORPUS_FILE is qrel-augmented subset: ${CORPUS_FILE}"
  echo "[warn] This is not mmRAG main-setting aligned for direct SOTA comparison."
fi

if [[ "${CORPUS_FILE}" == *"processed_documents.json" ]] && [[ "${ALLOW_HEAVY_ALL_CHUNKS:-0}" != "1" ]]; then
  echo "[block] You are about to run on full all-chunks corpus (processed_documents.json)."
  echo "[block] Set ALLOW_HEAVY_ALL_CHUNKS=1 to continue."
  exit 2
fi

cd "${ROOT_DIR}"

CMD=(
  "${PYTHON_BIN}" scripts/eval/run_e2e_table1c.py
  --model-dir "${MODEL_DIR}"
  --split-file "${SPLIT_FILE}"
  --corpus-file "${CORPUS_FILE}"
  --out-dir "${OUT_DIR}"
  --max-queries "${MAX_QUERIES}"
  --retrieve-topk 20
  --qa-context-k 3
  --reader "${READER}"
  --official-mmrag-mode
  --methods "${METHODS}"
  --reuse-methods "${REUSE_METHODS}"
  --unihgkr-model-dir "${UNIHGKR_MODEL_DIR}"
)

if [[ "${INCLUDE_ORACLE_MEASURED}" == "1" ]]; then
  CMD+=(--include-oracle-measured-row)
fi

if [[ "${READER}" == "ollama" ]]; then
  CMD+=(--ollama-host "${OLLAMA_HOST}" --ollama-model "${OLLAMA_MODEL}")
elif [[ "${READER}" == "openai" ]]; then
  CMD+=(
    --openai-model "${OPENAI_MODEL}"
    --openai-api-key-env "${OPENAI_API_KEY_ENV}"
    --openai-base-url "${OPENAI_BASE_URL}"
    --openai-timeout "${OPENAI_TIMEOUT}"
    --openai-temperature "${OPENAI_TEMPERATURE}"
    --openai-max-tokens "${OPENAI_MAX_TOKENS}"
    --openai-max-retries "${OPENAI_MAX_RETRIES}"
    --openai-retry-backoff "${OPENAI_RETRY_BACKOFF}"
  )
  if [[ "${OPENAI_FAIL_SOFT}" == "1" ]]; then
    CMD+=(--openai-fail-soft)
  fi
fi

if [[ -n "${EXTRA_E2E_ARGS}" ]]; then
  # shellcheck disable=SC2206
  EXTRA_ARGS=( ${EXTRA_E2E_ARGS} )
  CMD+=("${EXTRA_ARGS[@]}")
fi

"${CMD[@]}"

${PYTHON_BIN} scripts/pipeline/compare_mmrag_main_sota.py \
  --table1c-metrics "${OUT_DIR}/table1c_e2e_metrics.json" \
  --out-json "${OUT_DIR}/mmrag_main_sota_compare.json" \
  --out-md "${OUT_DIR}/mmrag_main_sota_compare.md"

echo "[DONE] mmRAG main-aligned evaluation finished -> ${OUT_DIR}"
