#!/usr/bin/env bash
# Run TESSERA v36 GPT-4o-mini reader experiments for context budgets 4 and 5.
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${ROOT_DIR}"

: "${OPENAI_API_KEY:?OPENAI_API_KEY is missing. Export it before running this script.}"

unset HTTP_PROXY HTTPS_PROXY http_proxy https_proxy ALL_PROXY all_proxy
export NO_PROXY="*"
export no_proxy="*"
export PYTHONUNBUFFERED=1

RUN_ID="${RUN_ID:-$(date +%Y%m%d_%H%M%S)}"
PYTHON_BIN="${PYTHON_BIN:-/home/wanghui/rag/multimodalrag/.conda/unifusion-gpt4o/bin/python}"
MMRAG_ROOT="${MMRAG_ROOT:-/home/wanghui/rag/multimodalrag/downloaded_resource/mmRAG/data/mmRAG_ds}"
SPLIT_FILE="${SPLIT_FILE:-${MMRAG_ROOT}/mmrag_test.json}"
CORPUS_FILE="${CORPUS_FILE:-${MMRAG_ROOT}/processed_documents.json}"
RANKINGS_DEBUG="${RANKINGS_DEBUG:-${ROOT_DIR}/artifacts/results/20260706_212912_paper_retrieval_metrics_unifusion_esr_v36/rankings_debug.jsonl}"

OPENAI_MODEL="${OPENAI_MODEL:-gpt-4o-mini}"
OPENAI_BASE_URL="${OPENAI_BASE_URL:-https://api.key77qiqi.com/v1}"
OPENAI_TIMEOUT="${OPENAI_TIMEOUT:-300}"
OPENAI_MAX_RETRIES="${OPENAI_MAX_RETRIES:-8}"
OPENAI_RETRY_BACKOFF="${OPENAI_RETRY_BACKOFF:-3}"
OPENAI_MAX_TOKENS="${OPENAI_MAX_TOKENS:-64}"

echo "[config] run_id=${RUN_ID}"
echo "[config] model=${OPENAI_MODEL}"
echo "[config] base_url=${OPENAI_BASE_URL}"
echo "[config] rankings=${RANKINGS_DEBUG}"
echo "[config] no_proxy=${no_proxy}"

for K in 4 5; do
  METHOD_LABEL="unifusion_esr_v36_ctx${K}"
  OUT_DIR="${ROOT_DIR}/artifacts/results/${RUN_ID}_reader_gpt4omini_${METHOD_LABEL}"
  echo "[RUN] TESSERA v36 ctx${K} -> ${OUT_DIR}"
  "${PYTHON_BIN}" scripts/eval/run_reader_from_rankings.py \
    --split-file "${SPLIT_FILE}" \
    --corpus-file "${CORPUS_FILE}" \
    --rankings-debug "${RANKINGS_DEBUG}" \
    --ranking-method unifusion_rag \
    --method-label "${METHOD_LABEL}" \
    --out-dir "${OUT_DIR}" \
    --max-queries 1286 \
    --qa-context-k "${K}" \
    --reader openai \
    --openai-model "${OPENAI_MODEL}" \
    --openai-api-key-env OPENAI_API_KEY \
    --openai-base-url "${OPENAI_BASE_URL}" \
    --openai-timeout "${OPENAI_TIMEOUT}" \
    --openai-temperature 0 \
    --openai-max-tokens "${OPENAI_MAX_TOKENS}" \
    --openai-max-retries "${OPENAI_MAX_RETRIES}" \
    --openai-retry-backoff "${OPENAI_RETRY_BACKOFF}" \
    --openai-fail-soft
  echo "[DONE] TESSERA v36 ctx${K}"
done

echo "[DONE] all TESSERA budget runs finished"
