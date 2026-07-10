# TESSERA

TESSERA (**Task-aware Evidence Set Selection and Efficient Retrieval Assembly**) is a task-aware evidence selection framework for multimodal RAG over unified text, table, and KG evidence chunks.

This repository contains the cleaned source code for TESSERA. Local datasets, model checkpoints, experiment logs, generated artifacts, conda environments, and temporary run outputs are intentionally excluded from Git.

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
- `scripts/eval/run_e2e_table1c.py`: end-to-end evaluation driver.
- `scripts/eval/apply_evidence_set_reranker.py`: apply a trained TESSERA reranker to saved rankings.
- `scripts/eval/run_reader_from_rankings.py`: reader evaluation from saved rankings.
- `scripts/train/train_evidence_set_reranker.py`: train the supervised evidence-set reranker.

## Notes

The project expects mmRAG data and local model artifacts to be provided outside the repository. Keep local paths in environment variables or ignored local config files rather than committing machine-specific paths or API keys.
