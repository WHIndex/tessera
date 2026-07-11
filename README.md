# TESSERA

TESSERA (**Task-aware Evidence Set Selection and Efficient Retrieval Assembly**) is a multimodal RAG framework for selecting complementary evidence from a unified pool of text, table, and knowledge-graph chunks.

The core idea is to treat retrieval context construction as an evidence-set selection problem. TESSERA first learns a supervised evidence utility function from graded relevance labels, then assembles the final context with a set-level objective that balances utility, query coverage, source complementarity, anchor preservation, redundancy control, and context length.

## Repository Layout

```text
tessera_exp/
  configs/          # configuration templates
  scripts/          # training, retrieval, reader evaluation, and analysis entry points
  src/tessera_exp/  # core implementation
```

## Core Implementation

- `tessera_exp/src/tessera_exp/e2e/evidence_set_reranker.py`: supervised evidence utility scoring and set-level evidence assembly.
- `tessera_exp/scripts/train/train_evidence_set_reranker.py`: training entry point for the TESSERA evidence utility model.
- `tessera_exp/scripts/eval/apply_evidence_set_reranker.py`: applies TESSERA to saved retrieval rankings and reports retrieval metrics.
- `tessera_exp/scripts/eval/run_reader_from_rankings.py`: evaluates end-to-end QA from saved rankings.
- `tessera_exp/scripts/eval/run_reader_from_context_docs.py`: evaluates end-to-end QA from fixed context documents.

## Main Scripts

- `tessera_exp/run_tessera_retrieval_experiment.sh`
- `tessera_exp/run_tessera_reader_gpt4omini.sh`
- `tessera_exp/run_tessera_llama33_70b_ctx6_parallel.sh`

## Data and Models

The code expects mmRAG splits, corpus files, retrieval traces, and model checkpoints to be provided through local paths or environment variables. API keys and machine-specific paths should be configured locally and should not be committed.
