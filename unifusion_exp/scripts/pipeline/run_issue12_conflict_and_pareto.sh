#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-/home/anaconda/envs/graphrag-yyq/bin/python}"
GPU_LIST="${GPU_LIST:-0,1}"
RUN_TAG="${1:-$(date +%Y%m%d_%H%M%S)}"
MAX_QUERIES="${MAX_QUERIES:-300}"

IFS=',' read -r -a GPUS <<< "${GPU_LIST}"
if [[ ${#GPUS[@]} -eq 0 ]]; then
  echo "[ERR] no GPU configured"
  exit 1
fi

MMRAG_TEST="/home/yongqi.yin/reaserch_paper/downloaded_resource/mmRAG/data/mmRAG_ds/mmrag_test.json"
CORPUS_FILE="${ROOT_DIR}/artifacts/retrieval/corpus_subset_v1.json"
BASE_DIR="${ROOT_DIR}/artifacts/results/table1c_e2e_20260330_llm_qwen25_campe_full1286_all_v3_unihgkrfix"

CONFLICT_DIR="${ROOT_DIR}/artifacts/data/issue12_conflict50_${RUN_TAG}"
PARETO_DIR="${ROOT_DIR}/artifacts/results/issue12_pareto_q300_${RUN_TAG}"
mkdir -p "${CONFLICT_DIR}" "${PARETO_DIR}"

echo "[step1] build Conflict-50 template"
"${PYTHON_BIN}" "${ROOT_DIR}/scripts/data/build_conflict50_template.py" \
  --split-file "${MMRAG_TEST}" \
  --dense-pred-file "${BASE_DIR}/qa_predictions_dense_concat_test1286.jsonl" \
  --unifusion-pred-file "${BASE_DIR}/qa_predictions_unifusion_rag_test1286.jsonl" \
  --corpus-file "${CORPUS_FILE}" \
  --target-size 50 \
  --max-queries 1286 \
  --out-jsonl "${CONFLICT_DIR}/conflict50_template.jsonl" \
  --out-csv "${CONFLICT_DIR}/conflict50_template.csv" \
  --out-summary "${CONFLICT_DIR}/conflict50_template_summary.json"

run_one() {
  local gpu="$1"
  local profile="$2"
  local topk="$3"
  local outdir="${PARETO_DIR}/${profile}_topk_${topk}"
  mkdir -p "${outdir}"

  local late pathw cellw consw
  if [[ "${profile}" == "p0" ]]; then
    late="0.08"
    pathw="0.14"
    cellw="0.00"
    consw="0.00"
  else
    late="0.12"
    pathw="0.14"
    cellw="0.02"
    consw="0.03"
  fi

  echo "[run] gpu=${gpu} profile=${profile} topk=${topk} out=${outdir}"
  CUDA_VISIBLE_DEVICES="${gpu}" "${PYTHON_BIN}" "${ROOT_DIR}/scripts/eval/run_e2e_table1c.py" \
    --model-dir /home/yongqi.yin/reaserch_paper/downloaded_resource/e5-large-v2 \
    --split-file "${MMRAG_TEST}" \
    --corpus-file "${CORPUS_FILE}" \
    --out-dir "${outdir}" \
    --max-queries "${MAX_QUERIES}" \
    --retrieve-topk "${topk}" \
    --qa-context-k 6 \
    --reader ollama \
    --ollama-host http://127.0.0.1:11434 \
    --ollama-model qwen2.5-7B-Instruct:latest \
    --preserve-dense-top 0 \
    --unifusion-late-alpha "${late}" \
    --method-preset targeted \
    --methods dense_concat,unifusion_rag \
    --context-active-threshold 0.40 \
    --context-anchor-dense-k 4 \
    --context-anchor-uni-k 1 \
    --context-redundancy-lambda 0.08 \
    --pathmaxsim-weight "${pathw}" \
    --pathmaxsim-kg-threshold 0.0 \
    --table-cellmaxsim-weight "${cellw}" \
    --table-cellmaxsim-top-cells 160 \
    --context-consistency-weight "${consw}" \
    --router-model /home/yongqi.yin/reaserch_paper/unifusion_exp/runs/router_deberta_full_v1_router_deberta/router_metrics/router_deberta_full_model
}

echo "[step2] run pareto scans (parallel by GPUs=${GPU_LIST})"
configs=(
  "p0 10"
  "p0 20"
  "p0 40"
  "p1 10"
  "p1 20"
  "p1 40"
)

running=0
idx=0
for cfg in "${configs[@]}"; do
  profile="$(echo "${cfg}" | awk '{print $1}')"
  topk="$(echo "${cfg}" | awk '{print $2}')"
  gpu="${GPUS[$((idx % ${#GPUS[@]}))]}"
  run_one "${gpu}" "${profile}" "${topk}" &
  ((running+=1))
  ((idx+=1))
  if [[ ${running} -ge ${#GPUS[@]} ]]; then
    wait -n
    ((running-=1))
  fi
done
wait

echo "[step3] summarize pareto runs"
"${PYTHON_BIN}" "${ROOT_DIR}/scripts/pipeline/summarize_issue12_pareto.py" \
  --scan-root "${PARETO_DIR}" \
  --out-json "${PARETO_DIR}/pareto_summary.json" \
  --out-csv "${PARETO_DIR}/pareto_summary.csv" \
  --out-md "${PARETO_DIR}/pareto_summary.md"

echo "[done] issue12 pipeline complete"
echo "[out] conflict -> ${CONFLICT_DIR}"
echo "[out] pareto -> ${PARETO_DIR}"
