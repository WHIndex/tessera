# TESSERA 论文结构初稿

方法名：**TESSERA**  
全称：**Task-aware Evidence Set Selection and Efficient Retrieval Assembly**

这版结构按正常论文写法组织，不单独写“任务定义”一节，而是把任务设定融入 Method 中；所有实验放在一个大节里；讨论和局限性合并。

## 1. Introduction

### 1.1 研究背景

说明多模态 RAG 需要同时利用 text、table、KG 等异构证据，单一文本检索或简单拼接难以满足复杂问答需求。

### 1.2 问题动机

指出现有方法通常关注单条 evidence 的相关性排序，但真实多跳或多源问题更需要一组互补 evidence，而不是一条最高分证据。

### 1.3 方法概览

介绍 TESSERA 的核心思想：在统一候选空间中学习 evidence utility，并选择一个面向 reader 的证据集合进行上下文组装。

### 1.4 主要贡献

概括三点贡献：任务感知的 evidence utility model、异构证据集合选择、面向 reader 的 context assembly，以及跨 reader 实验验证。

## 2. Related Work

### 2.1 Retrieval-Augmented Generation

回顾 RAG 中从检索到生成的基本范式，强调传统 RAG 多数以文本 passage 检索为主。

### 2.2 Multimodal and Heterogeneous Evidence Retrieval

讨论 text、table、KG 等异构证据在问答中的使用方式，并说明现有方法往往缺少统一的 evidence selection 视角。

### 2.3 Reranking and Evidence Selection

介绍 reranking、learning-to-rank、evidence selection 等相关工作，引出 TESSERA 与普通 reranker 的区别：TESSERA 优化的是 evidence set，而不是单条 evidence 排序。

## 3. Method

### 3.1 Overview

给出整体框架图，说明输入问题、统一多源 evidence space、候选召回、utility ranker、evidence set selection、context assembly、LLM reader 的完整流程。

### 3.2 Unified Heterogeneous Evidence Space

说明 text、table、KG 已在数据集中被 textualized，TESSERA 的起点是把这些 evidence 放入统一候选空间进行联合检索和排序；强调 textualization 不是本文贡献。

### 3.3 Task-aware Evidence Utility Modeling

介绍 supervised evidence utility ranker：对每个 query-evidence pair 构造特征，并用训练集 qrels 或 answer supervision 学习证据对回答的效用。

#### 3.3.1 Evidence Utility Function

给出公式：

\[
u_\theta(q,d)=f_\theta(\phi(q,d), s(d))
\]

其中 \(q\) 是问题，\(d\) 是候选 evidence，\(\phi(q,d)\) 是 query-evidence 特征，\(s(d)\) 是来源类型。

#### 3.3.2 Supervision Signal

说明训练标签来自训练集中的相关证据或答案支持证据，测试集答案不参与训练，避免数据泄漏。

### 3.4 Evidence Set Selection

说明 TESSERA 不直接取 utility score 最高的 top-k，而是选择一个互补证据集合，综合考虑 utility、coverage、source complementarity 和 redundancy。

#### 3.4.1 Set Objective

给出集合选择目标：

\[
S^*(q)=\arg\max_{S\subset C(q), |S|\le k}
\sum_{d\in S}u_\theta(q,d)
+\lambda \mathrm{Coverage}(q,S)
+\mu \mathrm{Diversity}(S)
-\gamma \mathrm{Redundancy}(S)
\]

### 3.5 Reader-oriented Context Assembly

说明 evidence set 会被组织为 reader 输入上下文，ctx6 的实验说明更充分的 evidence packing 可以把 retrieval gain 转化为 EM/F1 gain。

## 4. Experiments

### 4.1 Experimental Setup

介绍数据集、测试集规模、三类 evidence 来源、reader 设置、评价指标和 baseline。说明主要 reader 为 GPT-4o-mini，另用 Llama-3.3-70B 验证跨 reader 泛化。

### 4.2 Baselines

列出 Dense-Concat、NaiveRAG、CARP、TableRAG、QUASAR、UniHGKR，并简单说明每个 baseline 对应的检索或证据组织方式。

### 4.3 Main Results

放 GPT-4o-mini 主表，重点说明 TESSERA 在 F1、EM、mmRAG Official Avg 和 Recall@10 上均优于所有 baseline。

### 4.4 Cross-Reader Generalization

放 Llama-3.3-70B 表，说明 TESSERA 在另一个 reader 下仍然最佳，证明方法提升来自 evidence selection，而不是 reader 特化。

### 4.5 Retrieval Results

放 NDCG@1、NDCG@5、MAP@1、MAP@5、Hits@1、Hits@5、AnyHit@5 等检索指标，说明 TESSERA 不只提升生成，也提升证据排序质量。

### 4.6 Ablation Study

比较 w/o utility ranker、TESSERA 主方法、head-protected variant、over-protected variant，说明各模块的作用和过度保护 anchor 的风险。

### 4.7 Context Budget Analysis

比较 ctx3、ctx5、ctx6，说明 evidence set selection 需要合理的上下文预算才能充分转化为 reader 性能。

### 4.8 Slice and Error Analysis

按 text_only、table_only、kg_only、multi_modal 分析性能，指出 table_only 仍是主要瓶颈，因为表格问题需要行选择、数值比较、计数等结构化推理。

## 5. Discussion and Limitations

### 5.1 Why Evidence Set Selection Works

讨论为什么多源 QA 需要互补 evidence set，而不是单条最高分证据；结合实验说明 coverage 和 source complementarity 的价值。

### 5.2 Limitations

说明当前方法仍依赖 textualized table/KG evidence，且 table-only 场景仍受 reader 的表格推理能力限制。

### 5.3 Future Work

提出后续可以引入更强的结构化表格推理器、更细粒度 KG path execution，或训练 reader-aware verifier 来进一步提升 answer fidelity。

## 6. Conclusion

总结 TESSERA 将 multimodal RAG 检索重新建模为 task-aware evidence set selection，并在 GPT-4o-mini 与 Llama-3.3-70B 两个 reader 下稳定超过 baseline。

## 附：框架图应包含的内容

框架图建议从左到右画：

1. Query
2. Unified Heterogeneous Evidence Space：Text / Table / KG
3. Candidate Retrieval
4. Task-aware Evidence Utility Ranker
5. Evidence Set Selection：coverage、source complementarity、redundancy control
6. Reader-oriented Context Assembly
7. LLM Reader：GPT-4o-mini / Llama-3.3-70B
8. Final Answer

图中要明确标注：textualization is inherited from the benchmark，不作为本文贡献；本文贡献从 unified evidence selection 开始。
