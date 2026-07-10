#!/usr/bin/env bash
# Run Llama-3.3-70B (Ollama llama3.3:latest) reader on fixed context docs for baselines + UniFusion-ESR.
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${ROOT_DIR}"

export PYTHONUNBUFFERED=1

PYTHON_BIN="${PYTHON_BIN:-/home/wanghui/rag/multimodalrag/.conda/unifusion-gpt4o/bin/python}"
MMRAG_ROOT="${MMRAG_ROOT:-/home/wanghui/rag/multimodalrag/downloaded_resource/mmRAG/data/mmRAG_ds}"
SPLIT_FILE="${SPLIT_FILE:-${MMRAG_ROOT}/mmrag_test.json}"
CORPUS_FILE="${CORPUS_FILE:-${MMRAG_ROOT}/processed_documents.json}"

RUN_ID="${RUN_ID:-$(date +%Y%m%d_%H%M%S)}"
OUT_ROOT="${OUT_ROOT:-${ROOT_DIR}/artifacts/results/${RUN_ID}_reader_llama33_70b_context_docs}"
MAX_QUERIES="${MAX_QUERIES:-1286}"
QA_CONTEXT_K="${QA_CONTEXT_K:-6}"

READER="${READER:-ollama}"
OLLAMA_HOST="${OLLAMA_HOST:-http://127.0.0.1:11434}"
OLLAMA_MODEL="${OLLAMA_MODEL:-llama3.3:latest}"
OLLAMA_TIMEOUT="${OLLAMA_TIMEOUT:-900}"
OLLAMA_TEMPERATURE="${OLLAMA_TEMPERATURE:-0}"
OLLAMA_NUM_PREDICT="${OLLAMA_NUM_PREDICT:-64}"
OLLAMA_MAX_RETRIES="${OLLAMA_MAX_RETRIES:-3}"
OLLAMA_RETRY_BACKOFF="${OLLAMA_RETRY_BACKOFF:-5}"
OLLAMA_FAIL_SOFT="${OLLAMA_FAIL_SOFT:-1}"

OPENAI_MODEL="${OPENAI_MODEL:-llama-3.3-70b}"
OPENAI_API_KEY_ENV="${OPENAI_API_KEY_ENV:-OPENAI_API_KEY}"
OPENAI_BASE_URL="${OPENAI_BASE_URL:-${OPENAI_API_BASE:-https://api.key77qiqi.com/v1}}"
OPENAI_TIMEOUT="${OPENAI_TIMEOUT:-180}"
OPENAI_TEMPERATURE="${OPENAI_TEMPERATURE:-0}"
OPENAI_MAX_TOKENS="${OPENAI_MAX_TOKENS:-64}"
OPENAI_MAX_RETRIES="${OPENAI_MAX_RETRIES:-5}"
OPENAI_RETRY_BACKOFF="${OPENAI_RETRY_BACKOFF:-3}"
OPENAI_FAIL_SOFT="${OPENAI_FAIL_SOFT:-1}"

BASELINE_CONTEXT_ROOT="${BASELINE_CONTEXT_ROOT:-${ROOT_DIR}/artifacts/results/table1c_e2e_gpt4omini_full1286_allbaselines}"
OURS_CONTEXT_DOCS="${OURS_CONTEXT_DOCS:-${ROOT_DIR}/artifacts/results/20260708_103622_reader_gpt4omini_unifusion_esr_v36_ctx6/context_docs_unifusion_esr_v36_ctx6_test1286.jsonl}"

mkdir -p "${OUT_ROOT}"

if [[ "${READER}" == "openai" ]]; then
  OPENAI_API_KEY_VALUE="${!OPENAI_API_KEY_ENV-}"
  if [[ -z "${OPENAI_API_KEY_VALUE}" ]]; then
    echo "[ERROR] ${OPENAI_API_KEY_ENV} is missing. Export your real API key before running." >&2
    exit 2
  fi
  if ! LC_ALL=C grep -q '^[ -~]\+$' <<< "${OPENAI_API_KEY_VALUE}"; then
    echo "[ERROR] ${OPENAI_API_KEY_ENV} contains non-ASCII characters. Do not use placeholder text like 你的key." >&2
    exit 2
  fi
fi

run_one() {
  local label="$1"
  local context_docs="$2"
  local out_dir="${OUT_ROOT}/${label}"
  echo "[RUN-ONE] label=${label}"
  echo "[RUN-ONE] context_docs=${context_docs}"
  local cmd=(
    "${PYTHON_BIN}" scripts/eval/run_reader_from_context_docs.py
    --split-file "${SPLIT_FILE}" \
    --corpus-file "${CORPUS_FILE}" \
    --context-docs "${context_docs}" \
    --method-label "${label}" \
    --out-dir "${out_dir}" \
    --max-queries "${MAX_QUERIES}" \
    --qa-context-k "${QA_CONTEXT_K}" \
    --reader "${READER}"
  )

  if [[ "${READER}" == "ollama" ]]; then
    cmd+=(
      --ollama-host "${OLLAMA_HOST}"
      --ollama-model "${OLLAMA_MODEL}"
      --ollama-timeout "${OLLAMA_TIMEOUT}"
      --ollama-temperature "${OLLAMA_TEMPERATURE}"
      --ollama-num-predict "${OLLAMA_NUM_PREDICT}"
      --ollama-max-retries "${OLLAMA_MAX_RETRIES}"
      --ollama-retry-backoff "${OLLAMA_RETRY_BACKOFF}"
    )
    if [[ "${OLLAMA_FAIL_SOFT}" == "1" ]]; then
      cmd+=(--ollama-fail-soft)
    fi
  elif [[ "${READER}" == "openai" ]]; then
    cmd+=(
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
      cmd+=(--openai-fail-soft)
    fi
  else
    echo "[ERROR] unknown READER=${READER}; expected ollama or openai" >&2
    exit 2
  fi
  "${cmd[@]}"
}

METHODS="${METHODS:-dense_concat,naive_rag,carp,tablerag,quasar,unihgkr_dense,unifusion_esr}"
IFS=',' read -r -a method_list <<< "${METHODS}"

for method in "${method_list[@]}"; do
  case "${method}" in
    dense_concat)
      run_one "dense_concat" "${BASELINE_CONTEXT_ROOT}/context_docs_dense_concat_test1286.jsonl"
      ;;
    naive_rag)
      run_one "naive_rag" "${BASELINE_CONTEXT_ROOT}/context_docs_naive_rag_test1286.jsonl"
      ;;
    carp)
      run_one "carp" "${BASELINE_CONTEXT_ROOT}/context_docs_carp_test1286.jsonl"
      ;;
    tablerag)
      run_one "tablerag" "${BASELINE_CONTEXT_ROOT}/context_docs_tablerag_test1286.jsonl"
      ;;
    quasar)
      run_one "quasar" "${BASELINE_CONTEXT_ROOT}/context_docs_quasar_test1286.jsonl"
      ;;
    unihgkr_dense)
      run_one "unihgkr_dense" "${BASELINE_CONTEXT_ROOT}/context_docs_unihgkr_dense_test1286.jsonl"
      ;;
    unifusion_esr)
      run_one "unifusion_esr" "${OURS_CONTEXT_DOCS}"
      ;;
    *)
      echo "[ERROR] unknown method: ${method}" >&2
      exit 2
      ;;
  esac
done

echo "[DONE] Llama-3.3-70B context-doc reader runs -> ${OUT_ROOT}"
