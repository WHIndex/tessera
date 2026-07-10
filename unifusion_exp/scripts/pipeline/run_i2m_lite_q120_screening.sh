#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-/home/anaconda/envs/graphrag-yyq/bin/python}"
GPU_ID="${GPU_ID:-1}"
RUN_TAG="${RUN_TAG:-$(date +%Y%m%d)}"
MAX_QUERIES="${MAX_QUERIES:-120}"
SEEDS="${SEEDS:-20260403 20260404 20260405}"

if [[ -f "${ROOT_DIR}/configs/paths.env" ]]; then
  # shellcheck disable=SC1090
  source "${ROOT_DIR}/configs/paths.env"
fi

MODEL_DIR="${MODEL_DIR:-/home/yongqi.yin/reaserch_paper/downloaded_resource/e5-large-v2}"
CORPUS_FILE="${CORPUS_FILE:-${ROOT_DIR}/artifacts/retrieval/corpus_subset_v1.json}"
OLLAMA_HOST="${OLLAMA_HOST:-http://127.0.0.1:11434}"
OLLAMA_MODEL="${OLLAMA_MODEL:-qwen2.5-7B-Instruct:latest}"
CDG_MIN_GAIN="${CDG_MIN_GAIN:-0.20}"
CDG_SUPPORT_THRESHOLD="${CDG_SUPPORT_THRESHOLD:--1}"
CDG_COMPLEXITY_MIN="${CDG_COMPLEXITY_MIN:-0}"
CDG_ENTROPY_MIN="${CDG_ENTROPY_MIN:--1}"
RUN_BOOTSTRAP="${RUN_BOOTSTRAP:-0}"
RUN_IECC="${RUN_IECC:-0}"
EVAL_PROGRESS_EVERY="${EVAL_PROGRESS_EVERY:-10}"
EVAL_PROGRESS_MIN_SECONDS="${EVAL_PROGRESS_MIN_SECONDS:-90}"
BOOTSTRAP_PROGRESS_EVERY="${BOOTSTRAP_PROGRESS_EVERY:-500}"
BOOTSTRAP_PROGRESS_MIN_SECONDS="${BOOTSTRAP_PROGRESS_MIN_SECONDS:-30}"
IECC_THRESHOLD="${IECC_THRESHOLD:-0.20}"
IECC_MARGIN="${IECC_MARGIN:-0.08}"
IECC_POOL_K="${IECC_POOL_K:-24}"
IECC_ANSWER_BOOST="${IECC_ANSWER_BOOST:-0.55}"
IECC_ENABLE_CALIBRATION="${IECC_ENABLE_CALIBRATION:-1}"
IECC_COMPLEXITY_MIN="${IECC_COMPLEXITY_MIN:-0}"
IECC_ENTROPY_MIN="${IECC_ENTROPY_MIN:--1}"

result_dir() {
  local case_name="$1"
  local seed="$2"
  echo "${ROOT_DIR}/artifacts/results/i2m_q120_ollama_strat_seed${seed}_${case_name}_${RUN_TAG}"
}

run_case() {
  local case_name="$1"
  shift
  local extra_args=("$@")

  for seed in ${SEEDS}; do
    local split_file="${ROOT_DIR}/artifacts/splits/mmrag_test_q120_stratified_modality_seed${seed}.json"
    local out_dir
    out_dir="$(result_dir "${case_name}" "${seed}")"

    echo "[run] case=${case_name} seed=${seed} out=${out_dir}"
    CUDA_VISIBLE_DEVICES="${GPU_ID}" "${PYTHON_BIN}" "${ROOT_DIR}/scripts/eval/run_e2e_table1c.py" \
      --model-dir "${MODEL_DIR}" \
      --split-file "${split_file}" \
      --corpus-file "${CORPUS_FILE}" \
      --out-dir "${out_dir}" \
      --max-queries "${MAX_QUERIES}" \
      --retrieve-topk 20 \
      --qa-context-k 6 \
      --reader ollama \
      --ollama-host "${OLLAMA_HOST}" \
      --ollama-model "${OLLAMA_MODEL}" \
      --methods dense_concat,unifusion_rag \
      --preserve-dense-top 0 \
      --unifusion-late-alpha 0.08 \
      --query-modality-prior-mix 0.35 \
      --context-candidate-expand-k 20 \
      --context-conflict-penalty-weight 0.12 \
      --context-conflict-risk-gating \
      --context-conflict-risk-low 0.06 \
      --context-conflict-risk-high 0.22 \
      --context-conflict-risk-probe-k 12 \
      --qa-objective-retrieval-weight 0.04 \
      --progress-every "${EVAL_PROGRESS_EVERY}" \
      --progress-min-seconds "${EVAL_PROGRESS_MIN_SECONDS}" \
      "${extra_args[@]}"
    echo "[done] case=${case_name} seed=${seed}"
  done
}

run_case "control_zero"
run_case "iadr_lite" \
  --intent-complexity-aware-budgeting
run_case "cdg_lite" \
  --enable-unifusion-consensus-refine \
  --unifusion-consensus-refine-min-gain "${CDG_MIN_GAIN}" \
  --unifusion-consensus-refine-support-threshold "${CDG_SUPPORT_THRESHOLD}" \
  --unifusion-consensus-refine-complexity-min "${CDG_COMPLEXITY_MIN}" \
  --unifusion-consensus-refine-entropy-min "${CDG_ENTROPY_MIN}"
run_case "iadr_cdg_lite" \
  --intent-complexity-aware-budgeting \
  --enable-unifusion-consensus-refine \
  --unifusion-consensus-refine-min-gain "${CDG_MIN_GAIN}" \
  --unifusion-consensus-refine-support-threshold "${CDG_SUPPORT_THRESHOLD}" \
  --unifusion-consensus-refine-complexity-min "${CDG_COMPLEXITY_MIN}" \
  --unifusion-consensus-refine-entropy-min "${CDG_ENTROPY_MIN}"

case_list=("iadr_lite" "cdg_lite" "iadr_cdg_lite")

if [[ "${RUN_IECC}" == "1" ]]; then
  iecc_args=(
    --unifusion-support-retry-threshold "${IECC_THRESHOLD}"
    --unifusion-support-retry-margin "${IECC_MARGIN}"
    --unifusion-support-retry-mode evidence_chain
    --unifusion-support-retry-pool-k "${IECC_POOL_K}"
    --unifusion-support-retry-answer-boost "${IECC_ANSWER_BOOST}"
    --unifusion-support-retry-complexity-min "${IECC_COMPLEXITY_MIN}"
    --unifusion-support-retry-entropy-min "${IECC_ENTROPY_MIN}"
  )
  if [[ "${IECC_ENABLE_CALIBRATION}" == "1" ]]; then
    iecc_args+=(--enable-unifusion-answer-calibration)
  fi

  run_case "iecc_lite" "${iecc_args[@]}"

  combo_args=(
    --intent-complexity-aware-budgeting
    "${iecc_args[@]}"
    --enable-unifusion-consensus-refine
    --unifusion-consensus-refine-min-gain "${CDG_MIN_GAIN}"
    --unifusion-consensus-refine-support-threshold "${CDG_SUPPORT_THRESHOLD}"
    --unifusion-consensus-refine-complexity-min "${CDG_COMPLEXITY_MIN}"
    --unifusion-consensus-refine-entropy-min "${CDG_ENTROPY_MIN}"
  )
  run_case "iadr_iecc_cdg_lite" "${combo_args[@]}"

  case_list+=("iecc_lite" "iadr_iecc_cdg_lite")
fi

echo "[all_done] i2m-lite q120 screening complete, run_tag=${RUN_TAG}, run_iecc=${RUN_IECC}"

if [[ "${RUN_BOOTSTRAP}" == "1" ]]; then
  echo "[step] paired bootstrap against control_zero"
  for seed in ${SEEDS}; do
    gold_file="${ROOT_DIR}/artifacts/splits/mmrag_test_q120_stratified_modality_seed${seed}.json"
    a_dir="$(result_dir "control_zero" "${seed}")"
    a_pred="${a_dir}/qa_predictions_unifusion_rag_test1286.jsonl"

    for case_name in "${case_list[@]}"; do
      b_dir="$(result_dir "${case_name}" "${seed}")"
      b_pred="${b_dir}/qa_predictions_unifusion_rag_test1286.jsonl"
      out_json="${b_dir}/e2e_paired_bootstrap_control_vs_${case_name}.json"
      out_md="${b_dir}/e2e_paired_bootstrap_control_vs_${case_name}.md"

      echo "[bootstrap] seed=${seed} case=${case_name}"
      "${PYTHON_BIN}" "${ROOT_DIR}/scripts/pipeline/e2e_paired_bootstrap.py" \
        --gold-file "${gold_file}" \
        --a-pred-file "${a_pred}" \
        --b-pred-file "${b_pred}" \
        --a-name "control_zero" \
        --b-name "${case_name}" \
        --n-bootstrap 5000 \
        --seed "${seed}" \
        --progress-every "${BOOTSTRAP_PROGRESS_EVERY}" \
        --progress-min-seconds "${BOOTSTRAP_PROGRESS_MIN_SECONDS}" \
        --out-json "${out_json}" \
        --out-md "${out_md}"
    done
  done
  echo "[all_done] paired bootstrap complete"
fi
