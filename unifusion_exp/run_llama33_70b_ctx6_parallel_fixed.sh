#!/usr/bin/env bash
# Parallel Llama-3.3-70B API reader evaluation from fixed ctx6 context docs.
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${ROOT_DIR}"

: "${OPENAI_API_KEY:?OPENAI_API_KEY is missing. Export it before running this script.}"

unset HTTP_PROXY HTTPS_PROXY http_proxy https_proxy ALL_PROXY all_proxy
export NO_PROXY="*"
export no_proxy="*"
export PYTHONUNBUFFERED=1

RUN_ID="${RUN_ID:-$(date +%Y%m%d_%H%M%S)}"
PARALLEL_JOBS="${PARALLEL_JOBS:-3}"

PYTHON_BIN="${PYTHON_BIN:-/home/wanghui/rag/multimodalrag/.conda/unifusion-gpt4o/bin/python}"
MMRAG_ROOT="${MMRAG_ROOT:-/home/wanghui/rag/multimodalrag/downloaded_resource/mmRAG/data/mmRAG_ds}"
SPLIT_FILE="${SPLIT_FILE:-${MMRAG_ROOT}/mmrag_test.json}"
CORPUS_FILE="${CORPUS_FILE:-${MMRAG_ROOT}/processed_documents.json}"
BASELINE_CONTEXT_ROOT="${BASELINE_CONTEXT_ROOT:-${ROOT_DIR}/artifacts/results/20260710_142951_ctx6_allbaseline_contexts}"
OURS_CONTEXT_DOCS="${OURS_CONTEXT_DOCS:-${ROOT_DIR}/artifacts/results/20260708_103622_reader_gpt4omini_unifusion_esr_v36_ctx6/context_docs_unifusion_esr_v36_ctx6_test1286.jsonl}"
OUT_ROOT="${OUT_ROOT:-${ROOT_DIR}/artifacts/results/${RUN_ID}_llama33_70b_ctx6_parallel}"

MAX_QUERIES="${MAX_QUERIES:-1286}"
QA_CONTEXT_K="${QA_CONTEXT_K:-6}"
OPENAI_MODEL="${OPENAI_MODEL:-llama-3.3-70b}"
OPENAI_BASE_URL="${OPENAI_BASE_URL:-${OPENAI_API_BASE:-https://api.key77qiqi.com/v1}}"
OPENAI_TIMEOUT="${OPENAI_TIMEOUT:-300}"
OPENAI_MAX_RETRIES="${OPENAI_MAX_RETRIES:-8}"
OPENAI_RETRY_BACKOFF="${OPENAI_RETRY_BACKOFF:-3}"
OPENAI_MAX_TOKENS="${OPENAI_MAX_TOKENS:-64}"
OPENAI_FAIL_SOFT="${OPENAI_FAIL_SOFT:-1}"

if ! LC_ALL=C grep -q '^[ -~]\+$' <<< "${OPENAI_API_KEY}"; then
  echo "[ERROR] OPENAI_API_KEY contains non-ASCII characters." >&2
  exit 2
fi

mkdir -p "${OUT_ROOT}" "${ROOT_DIR}/logs"

echo "[config] run_id=${RUN_ID}"
echo "[config] out_root=${OUT_ROOT}"
echo "[config] baseline_context_root=${BASELINE_CONTEXT_ROOT}"
echo "[config] ours_context_docs=${OURS_CONTEXT_DOCS}"
echo "[config] model=${OPENAI_MODEL}"
echo "[config] base_url=${OPENAI_BASE_URL}"
echo "[config] parallel_jobs=${PARALLEL_JOBS}"
echo "[config] no_proxy=${no_proxy}"

declare -a LABELS=(
  "dense_concat"
  "naive_rag"
  "carp"
  "tablerag"
  "quasar"
  "unihgkr_dense"
  "tessera_v36"
)

context_file_for() {
  case "$1" in
    dense_concat) echo "${BASELINE_CONTEXT_ROOT}/context_docs_dense_concat_test1286.jsonl" ;;
    naive_rag) echo "${BASELINE_CONTEXT_ROOT}/context_docs_naive_rag_test1286.jsonl" ;;
    carp) echo "${BASELINE_CONTEXT_ROOT}/context_docs_carp_test1286.jsonl" ;;
    tablerag) echo "${BASELINE_CONTEXT_ROOT}/context_docs_tablerag_test1286.jsonl" ;;
    quasar) echo "${BASELINE_CONTEXT_ROOT}/context_docs_quasar_test1286.jsonl" ;;
    unihgkr_dense) echo "${BASELINE_CONTEXT_ROOT}/context_docs_unihgkr_dense_test1286.jsonl" ;;
    tessera_v36) echo "${OURS_CONTEXT_DOCS}" ;;
    *) echo "[ERROR] unknown label: $1" >&2; return 2 ;;
  esac
}

run_one() {
  local label="$1"
  local context_docs
  context_docs="$(context_file_for "${label}")"
  local out_dir="${OUT_ROOT}/${label}"

  if [[ ! -s "${context_docs}" ]]; then
    echo "[ERROR] missing context docs for ${label}: ${context_docs}" >&2
    return 2
  fi

  echo "[RUN-ONE] label=${label}"
  echo "[RUN-ONE] context_docs=${context_docs}"
  "${PYTHON_BIN}" scripts/eval/run_reader_from_context_docs.py \
    --split-file "${SPLIT_FILE}" \
    --corpus-file "${CORPUS_FILE}" \
    --context-docs "${context_docs}" \
    --method-label "${label}" \
    --out-dir "${out_dir}" \
    --max-queries "${MAX_QUERIES}" \
    --qa-context-k "${QA_CONTEXT_K}" \
    --reader openai \
    --openai-model "${OPENAI_MODEL}" \
    --openai-api-key-env OPENAI_API_KEY \
    --openai-base-url "${OPENAI_BASE_URL}" \
    --openai-timeout "${OPENAI_TIMEOUT}" \
    --openai-temperature 0 \
    --openai-max-tokens "${OPENAI_MAX_TOKENS}" \
    --openai-max-retries "${OPENAI_MAX_RETRIES}" \
    --openai-retry-backoff "${OPENAI_RETRY_BACKOFF}" \
    ${OPENAI_FAIL_SOFT:+--openai-fail-soft}
}

declare -a PIDS=()
declare -a PID_LABELS=()
FAIL=0

wait_oldest() {
  local pid="${PIDS[0]}"
  local label="${PID_LABELS[0]}"
  if wait "${pid}"; then
    echo "[DONE-ONE] label=${label} pid=${pid}"
  else
    echo "[FAIL-ONE] label=${label} pid=${pid}" >&2
    FAIL=1
  fi
  PIDS=("${PIDS[@]:1}")
  PID_LABELS=("${PID_LABELS[@]:1}")
}

for label in "${LABELS[@]}"; do
  while (( ${#PIDS[@]} >= PARALLEL_JOBS )); do
    wait_oldest
  done
  method_log="${ROOT_DIR}/logs/${RUN_ID}_reader_${label}.log"
  echo "[LAUNCH] ${label} log=${method_log}"
  run_one "${label}" > "${method_log}" 2>&1 &
  PIDS+=("$!")
  PID_LABELS+=("${label}")
done

while (( ${#PIDS[@]} > 0 )); do
  wait_oldest
done

if (( FAIL != 0 )); then
  echo "[DONE-WITH-ERROR] some reader jobs failed. Check logs/${RUN_ID}_reader_*.log" >&2
  exit 1
fi

echo "[DONE] ${OUT_ROOT}"
