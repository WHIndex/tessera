# TESSERA 中文论文结构初稿

方法名：**TESSERA**  
全称：**Task-aware Evidence Set Selection and Efficient Retrieval Assembly**

这版结构按中文论文写法整理：引言不拆二级标题，任务定义融入方法章节，所有实验统一放在第 4 章，跨 reader 结果并入主实验，讨论与局限性不拆二级标题。

## 1. 引言

介绍多模态检索增强生成的研究背景，说明真实问答场景中问题往往需要同时利用文本、表格和知识图谱等异构证据；指出现有方法多关注单条证据的相关性排序或简单分数融合，难以显式建模多源证据之间的互补关系；进一步引出本文方法 TESSERA，即在统一异构证据空间中学习面向任务的 evidence utility，并选择一组对 reader 最有用的证据集合；最后总结本文主要贡献和实验结论。

## 2. 相关工作

### 2.1 检索增强生成

回顾 RAG 中“检索证据再生成答案”的基本范式，说明传统 RAG 多以文本 passage 检索为主，通常缺少对异构证据来源的统一建模。

### 2.2 多模态与异构证据问答

介绍 text、table、KG 等证据在复杂问答中的作用，说明多模态 RAG 的关键挑战不只是召回更多证据，而是如何把不同来源的证据放入同一个候选空间中进行比较、选择和组合。

### 2.3 证据排序与证据选择

介绍 reranking、learning-to-rank 和 evidence selection 相关工作，强调 TESSERA 与普通 reranker 的区别：普通 reranker 主要优化单条证据排序，而 TESSERA 优化的是面向 reader 的证据集合。

## 3. 方法

### 3.1 方法概述

给出 TESSERA 的整体框架：输入问题后，系统从统一异构证据空间中召回候选证据，利用任务感知的 evidence utility model 对候选证据打分，再选择互补证据集合并打包为 reader context，最终由 LLM reader 生成答案。

### 3.2 统一异构证据空间

说明数据集已经将文本、表格和知识图谱证据转化为文本化 evidence，本文不把 textualization 作为贡献；TESSERA 的起点是将三类 evidence 放入统一候选空间：

\[
\mathcal{D}=\mathcal{D}_{text}\cup\mathcal{D}_{table}\cup\mathcal{D}_{kg}
\]

在该空间中，每条 evidence 同时具有文本内容和来源类型，使模型能够学习不同问题对不同证据来源的需求。

### 3.3 任务感知的证据效用建模

介绍 supervised evidence utility ranker。对每个 query-evidence pair 构造特征，包括初始检索分数、排序位置、来源类型、词汇重叠、实体匹配、数字匹配、表格或 KG 结构线索等，并学习证据对最终回答的效用：

\[
u_\theta(q,d)=f_\theta(\phi(q,d),s(d))
\]

其中 \(q\) 表示问题，\(d\) 表示候选证据，\(\phi(q,d)\) 表示问题与证据之间的匹配特征，\(s(d)\) 表示证据来源类型。监督信号来自训练集中的相关证据或答案支持证据，测试集答案不参与训练。

### 3.4 证据集合选择

说明 TESSERA 不直接取 utility 最高的 top-k 证据，而是选择一个互补证据集合。集合选择同时考虑证据效用、问题覆盖度、来源互补性和冗余惩罚：

\[
S^*(q)=\arg\max_{S\subset C(q), |S|\le k}
\sum_{d\in S}u_\theta(q,d)
+\lambda \mathrm{Coverage}(q,S)
+\mu \mathrm{Diversity}(S)
-\gamma \mathrm{Redundancy}(S)
\]

该设计的核心目的是让 reader 看到一组覆盖完整、来源互补且不过度重复的证据。

### 3.5 面向 Reader 的上下文组织

说明选出的 evidence set 会被进一步组织为 reader 输入上下文。实验中 ctx6 相比 ctx3 能更好地把 retrieval gain 转化为 EM/F1 gain，说明证据集合选择和上下文预算共同影响端到端问答效果。

## 4. 实验

### 4.1 实验设置

介绍数据集、测试集规模、三类证据来源、评价指标和 reader 设置。检索指标包括 NDCG@1、NDCG@5、MAP@1、MAP@5、Hits@1、Hits@5、AnyHit@5；端到端指标包括 F1、EM、mmRAG Official Avg 和 Recall@10。主要 reader 为 GPT-4o-mini，同时使用 Llama-3.3-70B 验证方法是否具有跨 reader 泛化能力。

### 4.2 对比方法

列出 Dense-Concat、NaiveRAG、CARP、TableRAG、QUASAR 和 UniHGKR 等 baseline，并简要说明它们分别代表简单拼接检索、通用 RAG、表格增强、KG 增强或异构知识检索等不同路线。

### 4.3 主实验结果

展示 GPT-4o-mini 和 Llama-3.3-70B 两组端到端结果。该小节中同时呈现跨 reader 对比，不单独开 Cross-Reader Generalization 小节。重点说明 TESSERA 在两个 reader 下均超过所有 baseline，证明提升来自更好的证据选择与上下文组织，而不是某个特定 reader 的偶然收益。

### 4.4 检索结果

展示 NDCG、MAP、Hits 和 AnyHit 等检索指标，说明 TESSERA 不仅提升最终答案质量，也提升候选证据排序和 top-k 证据覆盖能力。

### 4.5 消融实验

比较去掉 evidence utility ranker、TESSERA 主方法、head-protected variant 和 over-protected variant，分析任务感知效用建模、证据集合选择以及过度 anchor 保护对性能的影响。

### 4.6 补充分析

这里合并写上下文预算分析和分类型误差分析。上下文预算分析比较 ctx3、ctx5、ctx6，说明合理增加 reader context 能让证据选择收益更充分地转化为 EM/F1；分类型分析比较 text-only、table-only、kg-only 和 multi-modal 问题，指出 table-only 仍是主要瓶颈，因为这类问题通常需要表格行选择、数值比较、计数或结构化推理。

说明：4.6 里的两个分析我们都已经做过。上下文预算来自 ctx3/ctx5/ctx6 的 reader 结果；分类型分析来自 text_only、table_only、kg_only、multi_modal 的 slice metrics。如果后面篇幅不够，可以把 4.6 压缩为一段分析文字，而不是单独放大表。

## 5. 讨论与局限性

讨论 TESSERA 为什么有效：多源问答往往需要互补证据集合，而不是单条最高分证据；TESSERA 通过 utility modeling 和 evidence set selection 显式建模这种需求。同时说明当前方法仍依赖 benchmark 提供的 textualized table/KG evidence，表格问题上的表现仍受 reader 结构化推理能力限制。未来可以结合更强的表格推理器、KG path executor 或 reader-aware verifier 来进一步提升答案可靠性。

## 6. 结论

总结本文提出的 TESSERA 方法，将多模态 RAG 的检索阶段重新建模为任务感知的异构证据集合选择问题。实验表明，TESSERA 在 GPT-4o-mini 和 Llama-3.3-70B 两个 reader 下均稳定优于现有 baseline，验证了面向任务的 evidence utility modeling 和 reader-oriented context assembly 对多模态 RAG 的有效性。

## 附：框架图内容

框架图建议从左到右画：

1. Query
2. Unified Heterogeneous Evidence Space：Text / Table / KG
3. Candidate Retrieval
4. Task-aware Evidence Utility Ranker
5. Evidence Set Selection：utility、coverage、source complementarity、redundancy control
6. Reader-oriented Context Assembly
7. LLM Reader：GPT-4o-mini / Llama-3.3-70B
8. Final Answer

图中需要明确：text/table/KG 的文本化来自数据集或 benchmark 设置，不作为本文贡献；本文贡献从统一候选空间中的 evidence utility modeling 和 evidence set selection 开始。
