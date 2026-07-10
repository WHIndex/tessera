#!/usr/bin/env bash
# TESSERA retrieval: supervised evidence-set selection over the unified text/table/KG candidate pool.
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${ROOT_DIR}"

RUN_ID="${RUN_ID:-$(date +%Y%m%d_%H%M%S)}"
export PYTHONUNBUFFERED=1
export PYTHON_BIN="${PYTHON_BIN:-/home/wanghui/rag/multimodalrag/.conda/unifusion-gpt4o/bin/python}"

export TESSERA_MODEL="${TESSERA_MODEL:-${ESR_MODEL:-${ROOT_DIR}/artifacts/models/unifusion_esr_v1/evidence_set_reranker.pkl}}"
export BASE_TRACE="${BASE_TRACE:-${ROOT_DIR}/artifacts/results/20260705_162850_paper_retrieval_metrics_unifusion_ser_pr_composer_v16/rankings_debug.jsonl}"
export OUT_DIR="${OUT_DIR:-${ROOT_DIR}/artifacts/results/${RUN_ID}_paper_retrieval_metrics_tessera_v36}"
export CORPUS_MAIN="${CORPUS_MAIN:-${ROOT_DIR}/artifacts/retrieval/corpus_subset_v1.json}"
export CORPUS_STRICT="${CORPUS_STRICT:-${ROOT_DIR}/artifacts/retrieval/corpus_subset_strict_train_dev_v1.json}"
export CORPUS_DEVPOS="${CORPUS_DEVPOS:-${ROOT_DIR}/artifacts/retrieval/corpus_subset_devpos_v2.json}"

for path in "${TESSERA_MODEL}" "${BASE_TRACE}" "${CORPUS_MAIN}"; do
  if [[ ! -f "${path}" ]]; then
    echo "[missing] required file not found: ${path}"
    exit 2
  fi
done

EXTRA_ARGS=()
if [[ -n "${TESSERA_POOL_K_OVERRIDE:-${ESR_POOL_K_OVERRIDE:-}}" ]]; then
  EXTRA_ARGS+=(--pool-k "${TESSERA_POOL_K_OVERRIDE:-${ESR_POOL_K_OVERRIDE}}")
fi
if [[ -n "${TESSERA_PRESERVE_TOP_OVERRIDE:-${ESR_PRESERVE_TOP_OVERRIDE:-}}" ]]; then
  EXTRA_ARGS+=(--preserve-top "${TESSERA_PRESERVE_TOP_OVERRIDE:-${ESR_PRESERVE_TOP_OVERRIDE}}")
fi
if [[ -n "${TESSERA_BLEND_ORIGINAL_WEIGHT_OVERRIDE:-${ESR_BLEND_ORIGINAL_WEIGHT_OVERRIDE:-}}" ]]; then
  EXTRA_ARGS+=(--blend-original-weight "${TESSERA_BLEND_ORIGINAL_WEIGHT_OVERRIDE:-${ESR_BLEND_ORIGINAL_WEIGHT_OVERRIDE}}")
fi
if [[ -n "${TESSERA_TOP1_SWITCH_MARGIN_OVERRIDE:-${ESR_TOP1_SWITCH_MARGIN_OVERRIDE:-}}" ]]; then
  EXTRA_ARGS+=(--top1-switch-margin "${TESSERA_TOP1_SWITCH_MARGIN_OVERRIDE:-${ESR_TOP1_SWITCH_MARGIN_OVERRIDE}}")
fi

"${PYTHON_BIN}" scripts/eval/apply_evidence_set_reranker.py \
  --base-rankings-jsonl "${BASE_TRACE}" \
  --reranker-model "${TESSERA_MODEL}" \
  --corpus-json "${CORPUS_MAIN}" \
  --corpus-json "${CORPUS_STRICT}" \
  --corpus-json "${CORPUS_DEVPOS}" \
  --out-dir "${OUT_DIR}" \
  --method "${METHOD:-tessera}" \
  --base-method "${BASE_METHOD:-unifusion_rag}" \
  --label "TESSERA" \
  --metrics-k "${METRICS_K:-1,5}" \
  --max-queries "${MAX_QUERIES:-0}" \
  --include-base \
  "${EXTRA_ARGS[@]}"

echo "[DONE] TESSERA retrieval: ${OUT_DIR}"
