# Eval Scripts

本目录当前可用评估脚本如下：

- `eval_dense_retrieval_subset.py`
- `eval_milvus_retrieval_subset.py`
- `eval_neo4j_2hop_smoke.py`
- `eval_router_thresholds.py`
- `eval_qa.py`
- `eval_unifusion_retrieval_main.py`
- `run_e2e_table1c.py`（投稿主线 E2E 管线）
- `eval_unifusion_v2_simple.py`
- `eval_unifusion_v2_enhanced.py`（历史 experimental，仅探索用途，需显式 `--experimental-ok`）

## 1. 评测分层

- `eval_dense_retrieval_subset.py`：无服务依赖，适合快速检查检索语料和召回。
- `eval_router_thresholds.py`：验证 router 阈值与分类效果。
- `eval_qa.py`：通用 QA 结果打分。
- `run_e2e_table1c.py`：主线端到端评测，负责路由、上下文打包、reader 调用和结果汇总。

## 2. controller 评测说明

`run_e2e_table1c.py` 支持两类 smoke 模式：

1. `--reader extractive`
2. `--reader ollama`
3. `--reader openai`

建议规则如下：

- `reader=extractive` 仅用于 smoke / debug，速度快，但会低估真实 QA 质量。
- `reader=ollama` 和 `reader=openai` 会触发 verifier 驱动的 retry / refine 路径。
- 论文级结果保持默认严格模式，不启用 `--allow-heuristic-router-fallback`。

OpenAI API reader 示例：

```bash
export OPENAI_API_KEY=sk-...

/home/anaconda/envs/graphrag-yyq/bin/python scripts/eval/run_e2e_table1c.py \
  --model-dir ../downloaded_resource/e5-large-v2 \
  --split-file ../downloaded_resource/mmRAG/data/mmRAG_ds/mmrag_dev.json \
  --corpus-file artifacts/retrieval/corpus_subset_devpos_v2_5000.json \
  --out-dir artifacts/results/controller_openai_smoke_dev10_gpt4omini \
  --max-queries 10 \
  --reader openai \
  --openai-model gpt-4o-mini \
  --planner-model artifacts/models/evidence_planner_mmrag_v1 \
  --verifier-model artifacts/models/evidence_verifier_mmrag_v1 \
  --unifusion-support-retry-threshold 0.2 \
  --enable-unifusion-consensus-refine
```

如果要让 verifier 参与控制链，至少需要传入：

- `--planner-model <planner_bundle_dir>`
- `--verifier-model <verifier_bundle_dir>`
- `--unifusion-support-retry-threshold <float>`
- `--enable-unifusion-consensus-refine`

## 3. 输出约定

- 指标文件：JSON（`--out-file` 或脚本默认落盘文件）
- 明细文件：JSON（`--detail-file`，用于 bootstrap 显著性）
- 运行日志：建议重定向到 `artifacts/results/<run-id>/run.log`

## 4. 输入约定

`eval_qa.py` 输入约定：

- gold: JSON list，字段包含 `id`、`answer`
- pred: JSONL，字段包含 `id`、`prediction`（或 `pred` / `answer` 兼容）

## 5. 小样本优先

任何耗时评测都应先做 smoke：

- 查询数先从 5 或 10 开始。
- corpus 先使用小 slice，再扩大到正式主线 corpus。
- 先确认 `planner_model` / `verifier_model` 可以正常加载，再扩大到长任务。
