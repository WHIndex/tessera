#!/usr/bin/env bash
# GPT-4o-mini reader evaluation from a saved TESSERA retrieval rankings_debug.jsonl.
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${ROOT_DIR}"

: "${OPENAI_API_KEY:?OPENAI_API_KEY is missing. Run: export OPENAI_API_KEY=...}"

export PYTHONUNBUFFERED=1

PYTHON_BIN="${PYTHON_BIN:-/home/wanghui/rag/multimodalrag/.conda/unifusion-gpt4o/bin/python}"
MMRAG_ROOT="${MMRAG_ROOT:-/home/wanghui/rag/multimodalrag/downloaded_resource/mmRAG/data/mmRAG_ds}"
SPLIT_FILE="${SPLIT_FILE:-${MMRAG_ROOT}/mmrag_test.json}"
CORPUS_FILE="${CORPUS_FILE:-${MMRAG_ROOT}/processed_documents.json}"

# Default to the paper-facing TESSERA ranking key. Old artifacts can still use
# RANKING_METHOD=unifusion_rag.
RANKINGS_DEBUG="${RANKINGS_DEBUG:-${ROOT_DIR}/artifacts/results/20260706_212912_paper_retrieval_metrics_unifusion_esr_v36/rankings_debug.jsonl}"
RANKING_METHOD="${RANKING_METHOD:-tessera}"
METHOD_LABEL="${METHOD_LABEL:-tessera}"
RUN_ID="${RUN_ID:-$(date +%Y%m%d_%H%M%S)}"
OUT_DIR="${OUT_DIR:-${ROOT_DIR}/artifacts/results/${RUN_ID}_reader_gpt4omini_${METHOD_LABEL}}"

MAX_QUERIES="${MAX_QUERIES:-1286}"
QA_CONTEXT_K="${QA_CONTEXT_K:-3}"

OPENAI_MODEL="${OPENAI_MODEL:-gpt-4o-mini}"
OPENAI_API_KEY_ENV="${OPENAI_API_KEY_ENV:-OPENAI_API_KEY}"
OPENAI_BASE_URL="${OPENAI_BASE_URL:-https://api.key77qiqi.com/v1}"
OPENAI_TEMPERATURE="${OPENAI_TEMPERATURE:-0}"
OPENAI_MAX_TOKENS="${OPENAI_MAX_TOKENS:-64}"
OPENAI_TIMEOUT="${OPENAI_TIMEOUT:-180}"
OPENAI_MAX_RETRIES="${OPENAI_MAX_RETRIES:-5}"
OPENAI_RETRY_BACKOFF="${OPENAI_RETRY_BACKOFF:-3}"
OPENAI_FAIL_SOFT="${OPENAI_FAIL_SOFT:-1}"

CMD=(
  "${PYTHON_BIN}" scripts/eval/run_reader_from_rankings.py
  --split-file "${SPLIT_FILE}"
  --corpus-file "${CORPUS_FILE}"
  --rankings-debug "${RANKINGS_DEBUG}"
  --ranking-method "${RANKING_METHOD}"
  --method-label "${METHOD_LABEL}"
  --out-dir "${OUT_DIR}"
  --max-queries "${MAX_QUERIES}"
  --qa-context-k "${QA_CONTEXT_K}"
  --reader openai
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

printf '[RUN] %q ' "${CMD[@]}"
printf '\n'
"${CMD[@]}"

echo "[DONE] reader metrics -> ${OUT_DIR}"
