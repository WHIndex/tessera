# TESSERA

TESSERA is a task-aware evidence selection framework for multimodal RAG over unified text, table, and KG evidence chunks.

The project contains the reproducible experiment code, TESSERA retrieval/reranking modules, reader evaluation scripts, and paper materials used for the mmRAG experiments.

## What To Run

- Retrieval metrics: `bash unifusion_exp/run_tessera_retrieval_experiment.sh`
- GPT-4o-mini reader evaluation: `bash unifusion_exp/run_tessera_reader_gpt4omini.sh`
- Llama-3.3-70B reader evaluation: `bash unifusion_exp/run_tessera_llama33_70b_ctx6_parallel.sh`

The historical `unifusion_*` names are kept as compatibility aliases for old experiment artifacts. New paper-facing scripts and reports use **TESSERA**.

## Do Not Commit

Local conda environments, downloaded datasets, external baseline repositories, logs, runs, model checkpoints, and generated result artifacts are intentionally ignored by `.gitignore`.
