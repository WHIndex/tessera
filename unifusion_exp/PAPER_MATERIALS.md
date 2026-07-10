# UniFusion Paper Materials

整理日期：2026-07-08  
当前论文主方法建议名称：**UniFusion-ESR: Utility-Guided Evidence Set Selection for Multimodal RAG**

这份材料只保留可以写进论文的主线结果。旧版 UniFusion 作为内部迭代版本，不建议放入论文主表。

## 1. Core Message

UniFusion 的核心不是简单融合 dense/sparse/source 特征，而是把多模态 RAG 的检索阶段重新建模为：

> 在 text / table / KG 统一候选空间中，学习一个面向最终回答的 evidence utility model，并选择一组对 reader 最有用的证据集合。

最终最好结果来自：

- retrieval: **v36 ESR**
- reader context budget: **ctx6**
- reader: **GPT-4o-mini**

主结果：

| Method | F1 | EM | mmRAG Official Avg | Recall@10 |
|---|---:|---:|---:|---:|
| Dense-Concat | 0.3880 | 0.2667 | 0.2435 | 0.5267 |
| **UniFusion-ESR (ours)** | **0.4667** | **0.3212** | **0.2952** | **0.6071** |

提升：

- F1: +0.0788 absolute, +20.3% relative
- EM: +0.0544 absolute, +20.4% relative
- mmRAG Official Avg: +0.0517 absolute, +21.2% relative
- Recall@10: +0.0804 absolute, +15.3% relative

Paired bootstrap shows the gains over Dense-Concat are stable:

- F1 delta: +0.0788, 95% CI [0.0613, 0.0963]
- EM delta: +0.0544, 95% CI [0.0365, 0.0731]

## 2. Main E2E Table

GPT-4o-mini reader, full test set, 1286 queries.

| Method | F1 | EM | mmRAG Official Avg | Recall@10 |
|---|---:|---:|---:|---:|
| Dense-Concat | 0.3880 | 0.2667 | 0.2435 | 0.5267 |
| NaiveRAG | 0.3737 | 0.2574 | 0.2342 | 0.3722 |
| CARP-Adapter | 0.3663 | 0.2512 | 0.2270 | 0.4438 |
| TableRAG-Adapter | 0.3470 | 0.2364 | 0.2159 | 0.3560 |
| QUASAR-Adapter | 0.3201 | 0.2193 | 0.1983 | 0.3518 |
| UniHGKR-Base | 0.2970 | 0.1998 | 0.1838 | 0.1853 |
| **UniFusion-ESR (ours)** | **0.4667** | **0.3212** | **0.2952** | **0.6071** |

Result files:

- Baselines: `artifacts/results/table1c_e2e_gpt4omini_full1286_allbaselines/table1c_e2e_metrics.json`
- Ours: `artifacts/results/20260708_103622_reader_gpt4omini_unifusion_esr_v36_ctx6/table1c_e2e_metrics_from_rankings.json`

## 3. Retrieval Table

Full retrieval test set, 1286 queries.

| Method | NDCG@1 | NDCG@5 | MAP@1 | MAP@5 | Hits@1 | Hits@5 | AnyHit@5 |
|---|---:|---:|---:|---:|---:|---:|---:|
| Dense-Concat | 0.5892 | 0.5942 | 0.1149 | 0.3928 | 0.7022 | 3.0972 | 0.8686 |
| v35 CEPS-Cascade | 0.6475 | 0.6173 | 0.1305 | 0.4048 | 0.7527 | 3.0342 | 0.9106 |
| **v36 ESR** | 0.6459 | **0.6544** | 0.1286 | **0.4396** | **0.7636** | **3.2924** | **0.9269** |
| v37 Head-Protected ESR | **0.6475** | 0.6490 | **0.1305** | 0.4335 | 0.7527 | 3.2551 | 0.9253 |
| v38 Over-Protected ESR | **0.6475** | 0.6466 | **0.1305** | 0.4307 | 0.7527 | 3.2317 | 0.9238 |

Interpretation:

- v36 ESR has the best top-k retrieval metrics and the best final E2E result.
- v37 preserves the original head evidence and gives the best top-1 retrieval metrics, but slightly underperforms v36 in E2E ctx6.
- v38 over-protects anchors and degrades both retrieval and E2E compared with v36/v37.

## 4. Ablation Table

GPT-4o-mini reader, full test set, QA context budget = 6.

| Setting | F1 | EM | mmRAG Official Avg | Interpretation |
|---|---:|---:|---:|---|
| w/o ESR Utility, v35 ctx6 | 0.4480 | 0.3056 | 0.2796 | Strong base, but lacks supervised evidence utility learning |
| **UniFusion-ESR, v36 ctx6** | **0.4667** | **0.3212** | **0.2952** | Main method |
| Head-Protected ESR, v37 ctx6 | 0.4605 | 0.3196 | 0.2923 | Risk-control variant, not best E2E |
| Over-Protected ESR, v38 ctx6 | 0.4579 | 0.3188 | 0.2905 | Negative/risk analysis |

Key ablation conclusion:

- ESR Utility is useful: v36 ctx6 improves over v35 ctx6 by +0.0188 F1 and +0.0156 EM.
- Over-protection is harmful: v38 ctx6 underperforms v36 ctx6.
- Head protection is a stability/risk-control variant, not the main final method.

Bootstrap for v36 ctx6 over v35 ctx6:

- F1 delta: +0.0188, 95% CI [0.0085, 0.0299]
- EM delta: +0.0156, 95% CI [0.0047, 0.0272]

## 5. Context Packing Sensitivity

| Setting | F1 | EM | mmRAG Official Avg |
|---|---:|---:|---:|
| v36 ESR ctx3 | 0.4290 | 0.2955 | 0.2723 |
| **v36 ESR ctx6** | **0.4667** | **0.3212** | **0.2952** |
| v37 Head-Protected ctx3 | 0.4401 | 0.3040 | 0.2772 |
| v37 Head-Protected ctx5 | 0.4573 | 0.3173 | 0.2918 |
| v37 Head-Protected ctx6 | 0.4605 | 0.3196 | 0.2923 |

Interpretation:

- Increasing context budget from 3 to 6 converts retrieval gains into answer gains.
- v37 ctx5 and ctx6 are close; ctx6 is only marginally better than ctx5.
- For the paper, use ctx6 as the main accuracy setting and mention ctx5 as an efficiency-friendly setting if needed.

## 6. Slice Analysis

Final main method: v36 ESR ctx6.

| Slice | Count | F1 | EM | Recall@10 | Official |
|---|---:|---:|---:|---:|---:|
| text_only | 315 | 0.5938 | 0.4349 | 0.7354 | 0.3619 |
| table_only | 236 | 0.2221 | 0.1102 | 0.7814 | 0.0847 |
| kg_only | 88 | 0.4544 | 0.3523 | 0.4002 | 0.3707 |
| multi_modal | 647 | 0.4958 | 0.3385 | 0.5092 | 0.3293 |

Key error analysis:

- table_only remains the main bottleneck.
- table_only Recall@10 is high, but F1/EM are low.
- This indicates the bottleneck is not only retrieval, but table reasoning / numeric extraction / row selection in the reader stage.

This is useful for the limitations or future work section:

> Although UniFusion improves evidence retrieval and end-to-end QA, tabular-only questions remain challenging because the answer often requires structured row selection, comparison, counting, or numeric reasoning over semi-structured table text.

## 7. Recommended Paper Contributions

Use these as the contribution bullets.

### Contribution 1: Source-Heterogeneous Evidence Set Formulation

We formulate multimodal RAG retrieval as a **source-heterogeneous evidence set selection** problem. Instead of ranking text, table, and KG chunks independently, UniFusion places them in a unified candidate space and optimizes a set of complementary evidence for downstream answer generation.

Possible paper wording:

> We reformulate multimodal RAG retrieval as source-heterogeneous evidence set selection, where textual, tabular, and KG-derived evidence are jointly ranked in a unified candidate space.

### Contribution 2: Supervised Evidence Utility Ranker

We propose a supervised evidence utility model trained from qrels. The model estimates the usefulness of candidate evidence for a query, instead of relying on manually weighted score fusion.

Possible formulation:

\[
u_\theta(q,d)=f_\theta(\phi(q,d), s(d), r(d), c(q,d))
\]

where:

- \(q\): question
- \(d\): candidate evidence
- \(s(d)\): evidence source type, e.g. text/table/KG
- \(r(d)\): retrieval rank and retrieval score features
- \(c(q,d)\): lexical/entity/answer-type matching features
- \(u_\theta(q,d)\): learned evidence utility

Training labels come from training qrels:

\[
y(q,d)=1 \quad \text{if } d \in \mathcal{E}^{+}(q)
\]

This is the strongest methodological contribution.

### Contribution 3: Evidence Context Packing for Generation

UniFusion does not stop at retrieval metrics. It studies how the selected evidence set should be packed into the reader context. The ctx3 -> ctx6 improvement shows that evidence set selection and context packing jointly determine end-to-end QA performance.

Possible paper wording:

> We further introduce evidence context packing to bridge retrieval and generation, showing that a properly packed evidence set substantially improves GPT-4o-mini EM/F1.

### Contribution 4: Risk-Control Analysis of Evidence Anchors

Head-protected and over-protected variants show that preserving first-hop evidence can stabilize top-1 retrieval, but excessive anchor protection hurts evidence coverage and final QA.

This is better as an **analysis contribution**, not the main method contribution.

## 8. What Not To Claim

Do not claim:

- The conversion of text/table/KG into text chunks is our contribution. This is inherited from the mmRAG setup.
- Head protection is the main improvement. v36 ctx6 is stronger than v37 ctx6.
- v38 is an improvement. It is an over-protection negative case.
- Old UniFusion is a paper baseline. It was an internal iteration and should not appear in the main paper.

## 9. Suggested Paper Structure

### Introduction

- Multimodal RAG requires retrieving evidence from heterogeneous text/table/KG sources.
- Existing retrieval strategies focus on single-document relevance or simple score fusion.
- This misses the fact that multi-hop QA needs a complementary evidence set.
- UniFusion learns answer-oriented evidence utility and packs the selected set into the reader context.

### Method

1. Unified multimodal evidence candidate space
2. Source-aware evidence utility ranker
3. Evidence set selection and reranking
4. Evidence context packing for reader generation

### Experiments

- Dataset: mmRAG full test, 1286 queries
- Retrieval metrics: NDCG@1, NDCG@5, MAP@1, MAP@5, Hits@1, Hits@5, AnyHit@5
- E2E metrics: F1, EM, mmRAG Official Avg, Recall@10
- Reader: GPT-4o-mini
- Baselines: Dense-Concat, NaiveRAG, CARP, TableRAG, QUASAR, UniHGKR

### Ablation

- w/o ESR Utility: v35 ctx6
- UniFusion-ESR: v36 ctx6
- Head-Protected: v37 ctx6
- Over-Protected: v38 ctx6
- Context budget: ctx3 vs ctx5 vs ctx6

### Error Analysis

- table_only has high Recall@10 but low F1/EM.
- Reader struggles with table reasoning, row selection, counting, comparison, and numeric extraction.

## 10. Files To Cite Internally

Main E2E:

- `artifacts/results/20260708_103622_reader_gpt4omini_unifusion_esr_v36_ctx6/table1c_e2e_metrics_from_rankings.json`
- `artifacts/results/20260708_103622_reader_gpt4omini_unifusion_esr_v36_ctx6/table1c_e2e_metrics_from_rankings.md`

Ablations:

- `artifacts/results/20260708_115556_reader_gpt4omini_unifusion_ceps_v35_ctx6/table1c_e2e_metrics_from_rankings.json`
- `artifacts/results/20260707_135732_reader_gpt4omini_unifusion_esr_v37_ctx6/table1c_e2e_metrics_from_rankings.json`
- `artifacts/results/20260708_124732_reader_gpt4omini_unifusion_esr_v38_anchor_guard_ctx6/table1c_e2e_metrics_from_rankings.json`

Retrieval:

- `artifacts/results/20260706_212912_paper_retrieval_metrics_unifusion_esr_v36/paper_retrieval_metrics.csv`
- `artifacts/results/20260706_225116_paper_retrieval_metrics_unifusion_esr_v37_head_protected/paper_retrieval_metrics.csv`
- `artifacts/results/20260707_012020_paper_retrieval_metrics_unifusion_esr_v38_anchor_guard/paper_retrieval_metrics.csv`
- `artifacts/results/paper_retrieval_metrics_test1286_full/paper_retrieval_metrics.csv`

## 11. Next Writing Checklist

- [ ] Write Abstract around the evidence set selection story.
- [ ] Write Method section with equations for evidence utility and set selection.
- [ ] Build final main table from Section 2.
- [ ] Build retrieval table from Section 3.
- [ ] Build ablation table from Section 4.
- [ ] Add context packing sensitivity table from Section 5.
- [ ] Add table-only error analysis paragraph from Section 6.
- [ ] Keep old UniFusion out of the paper narrative.

