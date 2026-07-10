# UniFusion 实验跟踪文档

> **创建目的**: 防止任务跑偏，记录关键决策和状态。

---

## 2026-07-01 — Full Paper Retrieval Metrics 已完成

### 结论

- **状态**: 已完成，评测进程已退出。
- **结果目录**: `artifacts/results/paper_retrieval_metrics_test1286_full/`
- **输出文件**:
  - `paper_retrieval_metrics.json`
  - `paper_retrieval_metrics.csv`
  - `paper_retrieval_metrics.md`
  - `paper_retrieval_metrics_detail.json`
- **这次没有使用 OpenAI API**: 该实验只评测检索排序指标，不调用 reader/LLM，所以未设置 API key 也能完成。
- **实际配置**: BGE-large-en-v1.5 dense encoder + BM25 sparse scores + DeBERTa router + local UniHGKR baseline。

### 数据与配置

| 项目 | 值 |
|------|----|
| Split | `mmrag_test.json` |
| Queries | 1286 |
| Corpus | 3,201,548 chunks (`processed_documents.json`) |
| Positive qrels | 9474 / 9474 covered |
| Dense model | `downloaded_resource/bge-large-en-v1.5` |
| Sparse backend | BM25 (`max_features=200000`) |
| Router | `runs/router_deberta_full_v1_router_deberta/router_metrics/router_deberta_full_model` |
| Metrics | NDCG@1/3/5, MAP@1/3/5, Hits@1/3/5 |

### 主结果

| Method | NDCG@1 | NDCG@5 | MAP@1 | MAP@5 | Hits@1 | Hits@5 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Dense-Concat | 0.5892 | **0.5942** | 0.1149 | **0.3928** | 0.7022 | **3.0972** |
| NaiveRAG | 0.5555 | 0.5146 | 0.1105 | 0.3373 | 0.6617 | 2.5226 |
| CARP-Adapter | 0.5819 | 0.5444 | 0.1149 | 0.3510 | 0.6820 | 2.6267 |
| TableRAG-Adapter | 0.5526 | 0.5050 | 0.1135 | 0.3252 | 0.6407 | 2.3779 |
| QUASAR-Adapter | 0.3883 | 0.3504 | 0.0683 | 0.1899 | 0.4619 | 1.7333 |
| UniHGKR-Adapter | 0.1843 | 0.1712 | 0.0388 | 0.0963 | 0.2170 | 0.7932 |
| UniFusion | **0.5993** | 0.5635 | **0.1194** | 0.3653 | **0.7076** | 2.7240 |

### 解读

- UniFusion 在 **NDCG@1、MAP@1、MAP@3、Hits@1、Any-Hit@3/5** 上是最佳。
- Dense-Concat 在 **NDCG@3/5、MAP@5、Hits@3/5** 上更高，说明当前 UniFusion 对 top-1 命中更强，但 top-5 内保留的相关 chunk 数量不如 Dense-Concat。
- `Hits@k` 这里定义为 top-k 里的平均相关 chunk 数量，所以 Hits@5 可以大于 1。
- 如果要跑 reader/F1/EM 或 GPT-4o-mini 生成答案，需要另外启动 OpenAI reader 的 end-to-end 实验，并设置 `OPENAI_API_KEY`。

---

## 2026-07-02 — GPT-4o-mini E2E 全 Baseline 已完成

### 结论

- **状态**: 已完成，1286/1286。
- **结果目录**: `artifacts/results/table1c_e2e_gpt4omini_full1286_allbaselines/`
- **Reader**: OpenAI-compatible API, `gpt-4o-mini`。
- **主要指标**: EM / F1 / mmRAG Official Avg。
- **日志现象**: `api.key77qiqi.com` 偶发 timeout；已通过 fail-soft 继续，最终成功落盘。

### 结果记录

| Method | F1 | EM | mmRAG Official Avg | Recall@10 |
|---|---:|---:|---:|---:|
| Dense-Concat | 0.3880 | 0.2667 | 0.2435 | 0.5267 |
| NaiveRAG | 0.3737 | 0.2574 | 0.2342 | 0.3722 |
| CARP-Adapter | 0.3663 | 0.2512 | 0.2270 | 0.4438 |
| TableRAG-Adapter | 0.3470 | 0.2364 | 0.2159 | 0.3560 |
| QUASAR-Adapter | 0.3201 | 0.2193 | 0.1983 | 0.3518 |
| UniHGKR-Base | 0.2970 | 0.1998 | 0.1838 | 0.1853 |
| UniFusion-RAG | **0.3978** | **0.2753** | **0.2525** | 0.5056 |
| Oracle Gold | 0.5497 | 0.4067 | 0.3802 | 0.9518 |

### 分析

- UniFusion 已经在 F1、EM、mmRAG Official Avg 上超过所有本地复现 baseline。
- Dense-Concat 的 Recall@10 更高，说明 UniFusion 的答案生成效果更好，但 evidence coverage 还有提升空间。
- Slice 结果显示 UniFusion 在 text/table/kg/multi-modal 上整体优于对应 baseline；下一步应集中优化 **multi-modal evidence packing** 和 **table/number query 的上下文选择**。

### 新增改进

- 新模块: `src/unifusion_exp/e2e/unifusion_policy.py`
- 新开关: `--unifusion-policy-context`
- 新脚本: `run_unifusion_experiment.sh`
- 后续实验默认只跑 `unifusion_rag`，baseline 结果固定使用本节记录。

### 2026-07-02 新增 UniFusion Multi-Agent v1

- **目标**: 同时提升 retrieval 指标 `NDCG@1/5, MAP@1/5, Hits@1/5` 和 reader 指标 `EM/F1`。
- **检索新增模块**: `UnifusionRetrievalAgentConfig` + `rerank_unifusion_retrieval()`，位于 `src/unifusion_exp/e2e/unifusion_policy.py`。
- **方法结构**: base-rank agent 保留 UniFusion 原排序，dense/sparse agents 补召回，target/coverage/diversity agents 负责数字/表格/KG/多跳覆盖。
- **默认保护**: `--unifusion-retrieval-preserve-top 1`，优先守住 UniFusion 现有 top-1 优势，再优化 top-5 coverage。
- **E2E 仍使用**: `gpt-4o-mini` OpenAI-compatible API。
- **baseline 策略**: 不再重复跑 baseline；固定引用 `paper_retrieval_metrics_test1286_full/` 和 `table1c_e2e_gpt4omini_full1286_allbaselines/` 的结果。

推荐先跑 retrieval-only，再跑 E2E：

```bash
cd /home/wanghui/rag/multimodalrag/unifusion_exp
mkdir -p logs
nohup bash run_unifusion_retrieval_experiment.sh \
  > logs/paper_retrieval_unifusion_multiagent_v1.log 2>&1 &
```

```bash
cd /home/wanghui/rag/multimodalrag/unifusion_exp
mkdir -p logs
nohup bash run_unifusion_experiment.sh \
  > logs/table1c_e2e_gpt4omini_unifusion_multiagent_v1.log 2>&1 &
```

也可以一次性顺序跑两个：

```bash
cd /home/wanghui/rag/multimodalrag/unifusion_exp
mkdir -p logs
nohup bash run_unifusion_joint_experiment.sh \
  > logs/unifusion_multiagent_joint_v1.log 2>&1 &
```

### 2026-07-02 Multi-Agent v1 结果与瓶颈

- **Retrieval v1**: `artifacts/results/paper_retrieval_metrics_unifusion_multiagent_v1/`
  - NDCG@1 `0.5993`，MAP@1 `0.1194`，Hits@1 `0.7076`，与旧 UniFusion 持平。
  - NDCG@5 `0.5635 -> 0.5825`，MAP@5 `0.3653 -> 0.3752`，Hits@5 `2.7240 -> 2.8826`，top-5 覆盖提升明显。
  - 仍落后 Dense-Concat 的 top-5：Dense NDCG@5 `0.5942`，MAP@5 `0.3928`，Hits@5 `3.0972`。
- **E2E v1**: `artifacts/results/table1c_e2e_gpt4omini_unifusion_multiagent_v1/`
  - F1 `0.3978 -> 0.4140`，EM `0.2753 -> 0.2885`，Official Avg `0.2525 -> 0.2619`。
  - Recall@10 `0.5056 -> 0.5420`，说明 retrieval multi-agent 的召回提升能传导到 reader。
- **最大瓶颈**:
  - 表格题 reader 最弱：table_only Recall@10 `0.7100`，但 EM `0.0847`，F1 `0.1709`，Official Avg `0.0551`。
  - 离线统计显示 context@3 整体含 gold `84.6%`，但表格题只有 `73.5%`。
  - 65/1286 条 GPT-4o-mini 输出 No-evidence/Not-specified 类答案；表格题约 `22.1%` 出现这种失败。

### 2026-07-02 新增 UniFusion Retry v2

- **新增模块**: `UnifusionRetryAgentConfig` + `select_unifusion_retry_context()` + `is_no_evidence_answer()`，位于 `src/unifusion_exp/e2e/unifusion_policy.py`。
- **作用**: 当 reader 输出 No-evidence/Not-specified，或答案支持分过低时，从 UniFusion top candidates + dense candidates 重新打包 context，默认扩展到 `k=5`，并优先补 table/number 证据。
- **默认输出目录**: `artifacts/results/table1c_e2e_gpt4omini_unifusion_retry_v2/`
- **默认打开**:
  - `--unifusion-no-evidence-retry`
  - `--unifusion-no-evidence-retry-context-k 5`
  - `--enable-unifusion-answer-calibration`
  - `--enable-unifusion-answer-type-guard`
  - `--extractive-numeric-consensus`

推荐只跑 E2E v2，不重复跑 retrieval：

```bash
cd /home/wanghui/rag/multimodalrag/unifusion_exp
mkdir -p logs
LOG=logs/$(date +%Y%m%d_%H%M%S)_unifusion_retry_v2_e2e.log
nohup bash run_unifusion_experiment.sh > "$LOG" 2>&1 &
```

### 2026-07-02 新增 Retrieval v2

- **目标**: 继续提升 `NDCG@5 / MAP@5 / Hits@5`，追赶 Dense-Concat 的 top-5 coverage。
- **新增模块**:
  - Dense-rescue agent: `--unifusion-retrieval-dense-rescue-k 2`，在保留 UniFusion top-1 后给 Dense 高置信候选保留少量救援位置。
  - Sibling-expansion agent: `--unifusion-retrieval-sibling-seed-k 8 --unifusion-retrieval-sibling-window 1`，扩展同一页面/实体的相邻 chunk，解决相关证据分布在相邻 chunk 的 Hits@5 缺口。
- **默认输出目录**: `artifacts/results/paper_retrieval_metrics_unifusion_retrieval_v2/`
- **验证**: Python 编译、bash 语法检查、dev1 smoke 均通过。

推荐先跑 full retrieval-only v2：

```bash
cd /home/wanghui/rag/multimodalrag/unifusion_exp
mkdir -p logs
LOG=logs/$(date +%Y%m%d_%H%M%S)_paper_retrieval_unifusion_retrieval_v2.log
nohup bash run_unifusion_retrieval_experiment.sh > "$LOG" 2>&1 &
```

---

## 当前任务目标

将 Table 1c 的 reader 从 extractive 替换为 **qwen3.6:27b (Ollama)**，使用 **BGE-large-en-v1.5** 作为 encoder，运行完整 1286 查询的 end-to-end 评测，然后更新论文数据。

---

## 关键配置

| 组件 | 配置 |
|------|------|
| Encoder | BGE-large-en-v1.5 (`/home/yongqi.yin/reaserch_paper/downloaded_resource/bge-large-en-v1.5`) |
| Reader | qwen3.6:27b via Ollama v0.21.2 on port 11435 |
| Ollama Host | `http://127.0.0.1:11435` |
| Reader 参数 | `temperature=0, num_predict=16, num_ctx=1024` |
| 查询数 | 1286 (`mmrag_test.json`) |
| 语料库 | 102916 (`corpus_subset_v1.json`) |
| 方法数 | 10 (full preset) |
| QA context k | 6 |

---

## 已完成的步骤

1. ✅ BGE-large-en-v1.5 下载并集成
2. ✅ Ollama 升级到 v0.21.2 (port 11435)
3. ✅ qwen3.6:27b 验证可用
4. ✅ 编码缓存已生成 (BGE + UniHGKR query/corpus)
5. ✅ v7 实验已启动 (Ollama reader)

---

## 已知问题

### 1. qwen3.6-27b 输出 `<think>` 标签 (未修复)
```
Response: '\n\n<think>\n\n</think>\n\nParis'
```
当前 `ollama_reader` 没有 strip `<think>` 标签，会污染答案导致 F1/EM 下降。**需要在 `scripts/eval/run_e2e_table1c.py` 的 `ollama_reader()` 函数中添加 strip 逻辑。**

### 2. 速度过慢 (69-86s/查询, ETA 24-30h)
- Ollama 单请求测试: ~2.3s
- 实际每查询: 69-86s (10 个方法顺序执行)
- 瓶颈: 10 个方法 × (检索 + reader) 顺序执行
- 速度在缓慢提升中 (86→69s)

### 3. v6 实验跑错 (extractive reader)
- v6 缺少 `--reader ollama` 参数，默认使用了 extractive
- v6 结果无效 (F1 仅 0.045-0.063)
- v7 是正确的 (带 `--reader ollama`)

---

## 下一步计划

1. **修复 `<think>` 标签**: 修改 `ollama_reader()` strip `<think>...</think>`
2. **速度优化**: 考虑减少方法数或并行 reader
3. **等待 v7 完成**: 预计 24-30 小时 (当前 5/1286)
4. **bootstrap 显著性检验**
5. **更新论文**: `unifusion.tex`, `unifusion.md`, `unifusion_zh.tex`, `unifusion_zh.md`

---

## 缓存文件位置

```
artifacts/retrieval/dense_query_bf01342f0ff6_cls_1286_1e8700f434bf.npy
artifacts/retrieval/dense_corpus_bf01342f0ff6_cls_102916_070f21b5d758.npy
artifacts/retrieval/unihgkr_query_deb119f2daef_cls_1286_1e8700f434bf.npy
# unihgkr_corpus 缓存不存在 (v6 可能没生成或 v7 不需要)
```

---

## 实验结果目录

- v5 (被kill): `artifacts/results/table1c_e2e_20260427_bge_qwen36_full1286_v5/`
- v6 (extractive, 无效): `artifacts/results/table1c_e2e_20260427_bge_qwen36_full1286_v6/`
- v7 (Ollama reader, 进行中): `artifacts/results/table1c_e2e_20260427_bge_qwen36_ollama_v7/`

---

## 重要提醒

- **Ollama v0.21.2** 在 port 11435, 原始 v0.11.2 在 port 11434
- **不要混淆两个 Ollama 实例**
- **启动实验时必须加 `--reader ollama`**
- **qwen3.6-27b 会输出 `<think>` 标签，必须 strip**
- **GPU 1 显存充足 (33-76GB 空闲)，但速度仍慢**

---

*Last updated: 2026-04-28*

## 2026-04-28 01:40 — v8 正式启动

- **实验目录**: `artifacts/results/table1c_e2e_20260428_bge_qwen36_ollama_v8/`
- **配置**: BGE-large-en-v1.5 + qwen3.6:27b (Ollama reader)
- **num_ctx**: 768 (已从 1024 优化)
- **方法**: 10 个 (full preset)
- **进程**: PID 1722238
- **状态**: 运行中，首个查询处理中
- **Keep-alive**: 已部署
