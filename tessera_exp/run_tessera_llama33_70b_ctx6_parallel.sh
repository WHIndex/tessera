#!/usr/bin/env bash
# Llama-3.3-70B reader evaluation from saved TESSERA rankings.
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${ROOT_DIR}"

RUN_ID="${RUN_ID:-$(date +%Y%m%d_%H%M%S)}"
export PYTHONUNBUFFERED=1
export PYTHON_BIN="${PYTHON_BIN:-python}"

SPLIT_FILE="${SPLIT_FILE:-${ROOT_DIR}/data/mmrag_test.json}"
CORPUS_FILE="${CORPUS_FILE:-${ROOT_DIR}/artifacts/retrieval/corpus_subset_v1.json}"
RANKINGS_DEBUG="${RANKINGS_DEBUG:-${ROOT_DIR}/artifacts/results/tessera/rankings_debug.jsonl}"
OUT_DIR="${OUT_DIR:-${ROOT_DIR}/artifacts/results/${RUN_ID}_reader_llama33_70b_tessera}"

"${PYTHON_BIN}" scripts/eval/run_reader_from_rankings.py \
  --split-file "${SPLIT_FILE}" \
  --corpus-file "${CORPUS_FILE}" \
  --rankings-debug "${RANKINGS_DEBUG}" \
  --ranking-method "${RANKING_METHOD:-tessera}" \
  --method-label "${METHOD_LABEL:-tessera}" \
  --out-dir "${OUT_DIR}" \
  --qa-context-k "${QA_CONTEXT_K:-6}" \
  --reader openai \
  --openai-model "${OPENAI_MODEL:-llama-3.3-70b}" \
  --openai-base-url "${OPENAI_BASE_URL:-${OPENAI_API_BASE:-}}" \
  --openai-timeout "${OPENAI_TIMEOUT:-300}" \
  --openai-max-retries "${OPENAI_MAX_RETRIES:-8}" \
  --openai-max-tokens "${OPENAI_MAX_TOKENS:-64}" \
  ${OPENAI_FAIL_SOFT:+--openai-fail-soft}

echo "[DONE] TESSERA Llama-3.3-70B reader: ${OUT_DIR}"
