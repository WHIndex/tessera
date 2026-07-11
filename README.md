# TESSERA

TESSERA (**Task-aware Evidence Set Selection and Efficient Retrieval Assembly**) is a task-aware evidence selection framework for multimodal RAG over unified text, table, and KG evidence chunks.

## Repository Layout

```text
tessera_exp/
  configs/          # configuration templates
  scripts/          # data, training, evaluation, and analysis entry points
  src/tessera_exp/  # core TESSERA implementation
```

## Core Components

- `src/tessera_exp/e2e/evidence_set_reranker.py`: supervised evidence-set reranker used by TESSERA.
- `src/tessera_exp/e2e/source_evidence_fusion.py`: source-aware evidence fusion utilities.
- `src/tessera_exp/e2e/source_head_selector.py`: source-aware head evidence selection.
- `src/tessera_exp/e2e/submodular_packing.py`: set-level evidence packing and redundancy control.
- `scripts/eval/apply_evidence_set_reranker.py`: apply a trained TESSERA reranker to saved rankings.
- `scripts/eval/run_reader_from_rankings.py`: reader evaluation from saved rankings.
- `scripts/eval/run_reader_from_context_docs.py`: reader evaluation from fixed context documents.
- `scripts/train/train_evidence_set_reranker.py`: train the supervised evidence-set reranker.

## Main Entrypoints

- `tessera_exp/run_tessera_retrieval_experiment.sh`
- `tessera_exp/run_tessera_reader_gpt4omini.sh`
- `tessera_exp/run_tessera_llama33_70b_ctx6_parallel.sh`
