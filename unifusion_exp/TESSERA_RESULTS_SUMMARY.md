# TESSERA 实验结果汇总

整理日期：2026-07-09

方法名：**TESSERA**  
全称建议：**Task-aware Evidence Set Selection and Efficient Retrieval Assembly**

## 1. 核心结论

Llama-3.3-70B API 实验已经完整跑完。7 个方法各 1286 条预测，总计 9002 条预测；检查后各方法预测文件均无空答案。

在 GPT-4o-mini 和 Llama-3.3-70B 两个 reader 下，**TESSERA 都是当前所有 baseline 中最好的方法**。这里的 baseline 包括 Dense-Concat、NaiveRAG、CARP、TableRAG、QUASAR、UniHGKR。Oracle Gold 不属于可比较 baseline。

这组结果可以支撑论文中的两个主张：

1. TESSERA 的提升不是依赖某一个特定 reader，而是来自更好的证据选择与上下文组织。
2. TESSERA 在检索召回和端到端生成指标上都优于现有 baseline，说明它对 multimodal RAG 的核心瓶颈有实际改善。

## 2. Llama-3.3-70B API 端到端结果

Reader: `llama-3.3-70b`  
API base: `https://api.key77qiqi.com/v1`  
Run ID: `20260708_155214`  
Result root: `artifacts/results/20260708_155214_reader_llama33_70b_context_docs`

| Method | F1 | EM | mmRAG Official Avg | Recall@10 |
|---|---:|---:|---:|---:|
| Dense-Concat | 0.4214 | 0.2862 | 0.1205 | 0.2918 |
| NaiveRAG | 0.4057 | 0.2760 | 0.1168 | 0.2298 |
| CARP | 0.4056 | 0.2659 | 0.1102 | 0.2370 |
| TableRAG | 0.3908 | 0.2589 | 0.1066 | 0.1858 |
| QUASAR | 0.3529 | 0.2247 | 0.0920 | 0.1254 |
| UniHGKR | 0.3656 | 0.2496 | 0.0962 | 0.0934 |
| **TESSERA** | **0.4816** | **0.3351** | **0.1441** | **0.5509** |

与最强 baseline Dense-Concat 相比：

| Metric | Absolute Gain | Relative Gain |
|---|---:|---:|
| F1 | +0.0602 | +14.3% |
| EM | +0.0490 | +17.1% |
| mmRAG Official Avg | +0.0236 | +19.6% |
| Recall@10 | +0.2591 | +88.8% |

## 3. GPT-4o-mini 端到端结果

Reader: `gpt-4o-mini`  
Main TESSERA setting: v36 ESR, context budget = 6

| Method | F1 | EM | mmRAG Official Avg | Recall@10 |
|---|---:|---:|---:|---:|
| Dense-Concat | 0.3880 | 0.2667 | 0.2435 | 0.5267 |
| NaiveRAG | 0.3737 | 0.2574 | 0.2342 | 0.3722 |
| CARP | 0.3663 | 0.2512 | 0.2270 | 0.4438 |
| TableRAG | 0.3470 | 0.2364 | 0.2159 | 0.3560 |
| QUASAR | 0.3201 | 0.2193 | 0.1983 | 0.3518 |
| UniHGKR | 0.2970 | 0.1998 | 0.1838 | 0.1853 |
| **TESSERA** | **0.4667** | **0.3212** | **0.2952** | **0.6071** |

与最强 baseline Dense-Concat 相比：

| Metric | Absolute Gain | Relative Gain |
|---|---:|---:|
| F1 | +0.0788 | +20.3% |
| EM | +0.0544 | +20.4% |
| mmRAG Official Avg | +0.0517 | +21.2% |
| Recall@10 | +0.0804 | +15.3% |

## 4. 跨 Reader 对比

TESSERA 在两个不同 reader 下都超过所有 baseline：

| Reader | Best Baseline | Best Baseline F1 | TESSERA F1 | F1 Gain |
|---|---|---:|---:|---:|
| GPT-4o-mini | Dense-Concat | 0.3880 | 0.4667 | +0.0788 |
| Llama-3.3-70B | Dense-Concat | 0.4214 | 0.4816 | +0.0602 |

这说明 TESSERA 的提升不是 reader-specific 的偶然现象，而是来自检索证据质量和 evidence context 组织方式的改进。换句话说，TESSERA 改善的是 reader 之前的 evidence selection 阶段，因此不同 reader 都能受益。

## 5. 实验完整性检查

Llama-3.3-70B run completeness：

| Method | Predictions | Empty Predictions |
|---|---:|---:|
| Dense-Concat | 1286 | 0 |
| NaiveRAG | 1286 | 0 |
| CARP | 1286 | 0 |
| TableRAG | 1286 | 0 |
| QUASAR | 1286 | 0 |
| UniHGKR | 1286 | 0 |
| TESSERA | 1286 | 0 |

说明：resume log 中保留了早期由 API key 环境变量错误导致的失败记录，但错误期间写入的空预测已经在最终续跑前清理。最终每个方法的 prediction 文件均包含 1286 条非空预测。

## 6. 论文可用表述

英文表述：

> Across two different LLM readers, GPT-4o-mini and Llama-3.3-70B, TESSERA consistently outperforms all retrieval baselines. This suggests that the improvement comes from better multimodal evidence selection and context assembly, rather than from a reader-specific artifact.

> Compared with Dense-Concat, the strongest baseline, TESSERA improves F1 by 20.3% under GPT-4o-mini and 14.3% under Llama-3.3-70B, demonstrating robust cross-reader generalization.

中文表述：

> 在 GPT-4o-mini 和 Llama-3.3-70B 两个 reader 下，TESSERA 都稳定优于所有对比方法。这说明 TESSERA 的提升主要来自更有效的多源证据选择与上下文组织，而不是某个特定生成模型带来的偶然收益。

> 相比最强 baseline Dense-Concat，TESSERA 在 GPT-4o-mini 下取得 20.3% 的 F1 相对提升，在 Llama-3.3-70B 下取得 14.3% 的 F1 相对提升，说明该方法具有较好的跨 reader 泛化能力。

## 7. 可直接放入论文的结论句

TESSERA consistently achieves the best end-to-end QA performance across two readers and all compared baselines, validating the effectiveness of task-aware evidence set selection for multimodal RAG.
