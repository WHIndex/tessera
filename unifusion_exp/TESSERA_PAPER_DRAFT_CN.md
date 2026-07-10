# TESSERA：面向多模态 RAG 的任务感知异构证据集合选择方法

## 摘要

多模态检索增强生成（Multimodal Retrieval-Augmented Generation, MMRAG）通常需要同时利用文本段落、表格内容和知识图谱三类异构证据。现有方法往往将多模态内容统一转写为文本后，直接使用单一检索器或线性融合策略进行排序，忽略了不同来源证据在问题类型、答案形式和多跳推理过程中的互补性。本文提出 TESSERA（Task-aware Evidence Set Selection and Efficient Retrieval Assembly），将多模态 RAG 中的检索问题建模为任务感知的异构证据集合选择问题，而不是单个证据片段的独立排序问题。TESSERA 首先在统一候选空间中召回 text、table、KG 三类证据；随后利用训练集中的问题、答案和证据监督，学习问题-证据级别的效用函数；最后在效用、覆盖度、来源互补性与冗余控制之间进行联合优化，选择一组适合 Reader 使用的证据集合。实验结果表明，在 GPT-4o-mini 和 Llama-3.3-70B 两种 Reader 下，TESSERA 在 F1、EM、Official Score 和证据覆盖指标上均优于所有对比方法。进一步的消融实验表明，任务感知证据效用建模和适度的证据保护机制能够稳定提升端到端问答性能。

**关键词**：多模态检索增强生成；异构证据选择；证据集合排序；表格问答；知识图谱问答

## 1 引言

检索增强生成（Retrieval-Augmented Generation, RAG）通过在生成前引入外部证据，显著提升了大语言模型在知识密集型任务上的可靠性。随着多模态数据的广泛存在，问答系统需要同时处理自然语言文本、结构化表格以及知识图谱三类信息来源。对于这类任务，系统不仅需要召回与问题相关的片段，还需要判断不同来源证据之间的互补关系。例如，文本证据可能给出实体背景，表格证据可能包含数值答案，知识图谱证据可能提供实体关系或归属信息。简单地把所有证据转写成文本并放入同一个向量检索器，虽然能构建统一检索空间，但并不能充分解决异构证据的选择与组合问题。

现有多模态 RAG 方法主要存在三个不足。第一，检索阶段通常以单个证据片段为单位进行排序，缺少对证据集合整体效用的建模。对于多跳问题，一个证据片段本身未必足够回答问题，但多个来源的证据组合后可能形成完整推理链。第二，排序模型往往偏向 dense 相似度或 sparse lexical matching，难以根据问题类型动态选择最有价值的证据来源。例如，数值型问题更依赖表格，实体关系型问题更依赖 KG，而事实描述型问题更依赖文本。第三，端到端问答性能不仅取决于检索召回，还取决于有限上下文预算下如何组织证据。如果排序结果中冗余证据过多，即使 Evidence Recall@10 较高，Reader 仍可能无法在前几个上下文片段中获得足够明确的答案线索。

为解决上述问题，本文提出 TESSERA，一种面向多模态 RAG 的任务感知异构证据集合选择方法。TESSERA 的核心思想是：多模态 RAG 的关键不是简单地为每个候选证据分配一个相关性分数，而是根据问题意图、答案类型和候选证据来源，在统一候选空间中选择一组互补、低冗余且适合 Reader 使用的证据集合。具体而言，TESSERA 先将 text、table、KG 三类候选证据放入统一检索空间；然后使用训练集中的监督信号学习证据效用模型；最后通过集合选择目标联合考虑证据效用、查询覆盖、来源互补和冗余惩罚，得到最终上下文。

需要强调的是，本文不把“将 text、table、KG 转写为文本 chunk”作为主要贡献。这一步属于多模态 RAG 数据预处理或已有 MMRAG 框架的基础能力。本文关注的是在统一文本化候选空间之后，如何进行任务感知的异构证据选择与上下文组装。我们的贡献可以概括为三点：

1. 提出一种任务感知的异构证据集合选择框架，将多模态 RAG 检索从“单证据排序”转化为“证据集合选择”。
2. 设计一个监督式证据效用建模方法，利用训练集中的问题、答案和证据标注学习不同来源证据在不同问题类型下的贡献。
3. 在 GPT-4o-mini 和 Llama-3.3-70B 两种 Reader 上验证方法有效性，并通过消融实验、上下文预算分析和分类型误差分析说明各模块的作用。

## 2 相关工作

### 2.1 检索增强生成

RAG 方法通常由检索器和生成器两部分组成。检索器从外部语料中获取与问题相关的证据，生成器基于问题和证据产生答案。经典 RAG 系统通常使用 dense retriever、BM25 等 sparse retriever 或二者融合来构造候选集合。近年来，随着大语言模型能力增强，RAG 研究逐渐从单纯提升召回率转向改进证据排序、上下文压缩和生成可信度。

然而，在多模态场景下，传统 RAG 方法面临更复杂的证据组织问题。text、table、KG 三类信息具有不同结构与语义密度，直接将它们混合排序容易导致某一来源证据占据上下文预算，进而削弱跨来源互补性。本文将该问题重新定义为异构证据集合选择问题，强调在有限上下文中选择最有用的一组证据。

### 2.2 多模态与异构证据问答

多模态问答任务通常要求系统综合利用文本、表格、图像或知识图谱等多源信息。对于本文关注的数据设置，三类主要证据来源为 text、table 和 KG。文本证据适合描述事实与背景信息，表格证据适合表达数值、排名和属性列表，KG 证据适合表示实体间关系。已有方法通常将表格和图谱内容序列化为自然语言文本，再与文本段落共同输入检索器。

这种统一文本化策略降低了系统实现复杂度，但也带来一个核心问题：不同来源证据在形式上被统一后，其结构性差异和任务适配性容易被弱化。本文认为，统一文本化只是候选空间构建的前提，真正决定多模态 RAG 效果的是如何在统一空间中保留来源信息、问题类型信息和证据互补性。

### 2.3 证据排序与证据选择

证据排序通常把每个候选文档视为独立样本，学习一个相关性函数并按分数降序排列。该范式适用于单跳、单证据问答，但对多源、多跳问题存在局限：排序最高的单个证据不一定能形成完整答案；多个分数较高的证据也可能高度重复。相较之下，证据集合选择关注候选子集的整体效用，天然适合解决覆盖不足与冗余过高的问题。

TESSERA 借鉴监督排序与集合选择思想，但面向多模态 RAG 的特殊需求引入来源感知特征、问题意图特征和上下文预算约束。与仅依赖 dense/sparse 分数的融合方法不同，TESSERA 学习一个任务感知效用函数，并在推理时结合集合级目标选择证据。

## 3 方法

### 3.1 方法概述

给定一个问题 \(q\)，系统需要从异构证据库中检索并组织证据，供 Reader 生成答案。证据库包含三类来源：

\[
\mathcal{D}=\mathcal{D}^{text}\cup\mathcal{D}^{table}\cup\mathcal{D}^{kg},
\]

其中 \(\mathcal{D}^{text}\) 表示文本证据集合，\(\mathcal{D}^{table}\) 表示表格转写证据集合，\(\mathcal{D}^{kg}\) 表示知识图谱转写证据集合。每个证据片段 \(d\in\mathcal{D}\) 都带有来源标签：

\[
s(d)\in\{text, table, kg\}.
\]

TESSERA 的推理流程包括四步：

1. 从统一候选空间中召回候选证据集合 \(C(q)\)；
2. 为每个候选证据构造任务感知特征 \(\phi(q,d)\)；
3. 使用监督学习得到的效用函数 \(u_\theta(q,d)\) 估计证据对当前问题的贡献；
4. 通过证据集合选择目标得到最终上下文证据集合 \(S^\*\)，并输入 Reader。

整体形式可写为：

\[
S^\*=\operatorname{Select}\left(q, C(q), u_\theta, B\right),
\]

其中 \(B\) 为上下文预算，即最多允许放入 Reader 的证据数量或 token 数量。

### 3.2 统一异构证据空间

TESSERA 不改变前置 MMRAG 框架对不同来源数据的文本化处理。对于 text、table、KG 三类输入，系统分别得到统一格式的文本 chunk。每个 chunk 包含两部分信息：一是可被检索器和 Reader 直接处理的文本内容，二是用于后续排序和选择的元信息，例如来源类型、实体线索、数值线索和候选来源。

对问题 \(q\)，候选召回阶段综合使用 base ranking、dense ranking 和 sparse ranking。设三种召回器得到的候选集合分别为：

\[
C_b(q)=\operatorname{TopM}_b(q,\mathcal{D}),
\]

\[
C_d(q)=\operatorname{TopM}_d(q,\mathcal{D}),
\]

\[
C_s(q)=\operatorname{TopM}_s(q,\mathcal{D}).
\]

最终候选池为三者并集：

\[
C(q)=C_b(q)\cup C_d(q)\cup C_s(q).
\]

这样做的目的是同时保留 dense 检索的语义匹配能力、sparse 检索的词面匹配能力以及已有 baseline 排序中的强先验。对于每个候选证据 \(d\in C(q)\)，我们记录其在不同召回器中的分数和排名，作为后续效用模型的输入。

### 3.3 任务感知的证据效用建模

TESSERA 的核心是学习一个证据效用函数：

\[
u_\theta(q,d)=P_\theta(y=1\mid q,d,\phi(q,d)),
\]

其中 \(y=1\) 表示证据 \(d\) 对回答问题 \(q\) 有效，\(\phi(q,d)\) 是问题-证据特征向量，\(\theta\) 是模型参数。与直接使用 dense 相似度不同，\(u_\theta(q,d)\) 不是衡量文本语义相似度，而是衡量一个证据片段在当前问题下对最终回答的实际效用。

#### 3.3.1 训练标签构造

训练阶段只使用训练集和验证集，不使用测试集答案或测试集标注。mmRAG 数据集中每个问题样本包含问题 \(q_i\)、答案 \(a_i\) 以及 `relevant_chunks` 字段，其中 `relevant_chunks` 给出了与该问题相关或不相关的候选证据标记。在实验代码中，我们先将 `relevant_chunks` 中取值大于 0 的证据转换为检索评估中的 qrels，形成正证据集合 \(\mathcal{E}^{+}(q_i)\)。若候选证据 \(d\) 属于该集合，则记为正样本：

\[
y_i(d)=
\begin{cases}
1, & d\in \mathcal{E}^{+}(q_i),\\
0, & d\notin \mathcal{E}^{+}(q_i).
\end{cases}
\]

候选池中未被标注为正的证据作为负样本。这里的负样本包括两类：一类是 `relevant_chunks` 中显式标记为 0 的证据，另一类是召回器返回但不在正 qrels 中的高排名候选。为避免训练被容易负样本主导，负样本主要从召回器返回的 top pool 中采样，使模型学习区分“看起来相关但不能回答问题”的困难负例。

#### 3.3.2 特征设计

对每个问题-证据对 \((q,d)\)，TESSERA 构造如下特征：

\[
\phi(q,d)=
[\phi_{rank},\phi_{match},\phi_{source},\phi_{target},\phi_{query},\phi_{set}].
\]

其中：

\[
\phi_{rank}=
[s_b(q,d),s_d(q,d),s_s(q,d),rr_b(q,d),rr_d(q,d),rr_s(q,d)]
\]

表示 base、dense、sparse 三类召回分数及 reciprocal rank。若 \(r_m(q,d)\) 是证据 \(d\) 在召回器 \(m\) 中的排名，则：

\[
rr_m(q,d)=\frac{1}{1+r_m(q,d)}.
\]

\(\phi_{match}\) 表示词面和内容匹配特征，包括问题-文档 token overlap、文档对问题 token 的覆盖率、数值匹配、年份匹配、表格数值匹配、KG 实体匹配等。例如，问题 token 集合为 \(T_q\)，证据 token 集合为 \(T_d\)，则问题覆盖率可以写为：

\[
\operatorname{Cov}_{tok}(q,d)=\frac{|T_q\cap T_d|}{|T_q|+\epsilon}.
\]

\(\phi_{source}\) 表示来源相关特征，包括 text/table/KG one-hot 标签以及来源路由概率：

\[
\phi_{source}=[\mathbb{I}(s(d)=text),\mathbb{I}(s(d)=table),\mathbb{I}(s(d)=kg),p(s(d)\mid q)].
\]

\(\phi_{target}\) 表示答案类型相关特征，例如问题是否指向数值、年份、实体、人物、地点等：

\[
\phi_{target}=[z_{num}(q),z_{year}(q),z_{entity}(q),z_{person}(q),z_{loc}(q)].
\]

\(\phi_{query}\) 表示问题复杂度与推理需求，例如是否为直接事实型问题、是否需要多跳推理、问题 slot 数量、内容词数量和覆盖需求：

\[
\phi_{query}=[z_{direct}(q),z_{complex}(q),n_{slot}(q),n_{content}(q),z_{coverage}(q)].
\]

\(\phi_{set}\) 表示候选之间的相对结构特征，例如 dense/sparse 排名一致性、base/dense 排名一致性、dense top-5 分数平坦度、候选是否与 top 证据属于同一来源 family 等。这类特征帮助模型识别“单个检索器高分但不稳定”的候选。

#### 3.3.3 训练目标

证据效用模型采用二分类监督目标。对训练集 \(\mathcal{Q}_{train}\) 中所有候选证据，最小化 binary cross entropy：

\[
\mathcal{L}_{utility}(\theta)=
-\sum_{q\in\mathcal{Q}_{train}}\sum_{d\in C(q)}
\left[
y(q,d)\log u_\theta(q,d)
+(1-y(q,d))\log(1-u_\theta(q,d))
\right].
\]

该目标的含义是让模型学习：在给定问题和候选证据的情况下，哪些证据真正有助于回答问题。由于训练标签只来自训练集，因此不会造成测试集泄漏。

在推理阶段，TESSERA 将学习到的效用分数与基础检索分数融合：

\[
\tilde{s}(q,d)=(1-\alpha)\hat{s}_{base}(q,d)+\alpha u_\theta(q,d),
\]

其中 \(\hat{s}_{base}(q,d)\) 是归一化后的基础排序分数，\(\alpha\in[0,1]\) 控制监督效用模型的权重。实验中使用较高的 \(\alpha\)，使排序结果更多依赖任务感知效用，而不是原始相似度。

### 3.4 证据集合选择

单独按 \(\tilde{s}(q,d)\) 降序选择 top-k 证据仍然可能导致冗余。例如，top-k 中可能全部来自文本来源，或者多个证据重复描述同一实体。为此，TESSERA 将最终上下文构造为一个集合优化问题：

\[
S^\*=\arg\max_{S\subseteq C(q), |S|\le K} J(q,S),
\]

其中 \(K\) 是最多选择的证据数量，目标函数为：

\[
J(q,S)=
\sum_{d\in S}\tilde{s}(q,d)
\lambda \operatorname{Cov}(q,S)
\mu \operatorname{Div}(S)
\eta \operatorname{Anchor}(q,S)
-\gamma \operatorname{Red}(S).
\]

第一项为证据效用总和，鼓励选择高效用证据。第二项为问题覆盖度，鼓励证据集合覆盖问题中的关键 slot：

\[
\operatorname{Cov}(q,S)=
\frac{|\operatorname{Slot}(q)\cap \bigcup_{d\in S}\operatorname{Slot}(d)|}
{|\operatorname{Slot}(q)|+\epsilon}.
\]

第三项为来源多样性，鼓励 text、table、KG 之间形成互补：

\[
\operatorname{Div}(S)=
\frac{|\{s(d):d\in S\}|}{3}.
\]

第四项为 anchor 保护项。若某些候选证据具有极强的检索先验或被多个检索器共同支持，则将其视为 anchor evidence。设 \(\mathcal{A}(q)\) 为 anchor 候选集合，则：

\[
\operatorname{Anchor}(q,S)=
\mathbb{I}(S\cap \mathcal{A}(q)\neq\emptyset).
\]

该项防止监督模型过度改写原始强证据排序，从而降低漏掉显著相关证据的风险。

最后一项为冗余惩罚。设 \(T(d)\) 为证据 \(d\) 的 token 集合，两个证据之间的冗余定义为 Jaccard 相似度：

\[
\operatorname{sim}(d_i,d_j)=
\frac{|T(d_i)\cap T(d_j)|}{|T(d_i)\cup T(d_j)|+\epsilon}.
\]

则集合冗余为：

\[
\operatorname{Red}(S)=
\sum_{d_i,d_j\in S, i<j}\operatorname{sim}(d_i,d_j).
\]

完整优化问题是组合优化。实际推理中，TESSERA 使用贪心近似：从空集合开始，每一步选择使目标函数增益最大的候选证据：

\[
d_t^\*=\arg\max_{d\in C(q)\setminus S_{t-1}}
\left[J(q,S_{t-1}\cup\{d\})-J(q,S_{t-1})\right].
\]

然后更新：

\[
S_t=S_{t-1}\cup\{d_t^\*\}.
\]

当达到上下文预算 \(K\) 或候选增益低于阈值时停止。该过程能够在高效用、覆盖度、多源互补和低冗余之间取得平衡。

### 3.5 面向 Reader 的上下文组织

得到证据集合 \(S^\*\) 后，TESSERA 按集合选择顺序组织上下文：

\[
P(q)=\operatorname{Concat}\left(\operatorname{Format}(d_1),\ldots,\operatorname{Format}(d_K)\right),
\]

其中 \(d_i\in S^\*\)。每个证据在输入 Reader 前保留来源标签，例如 `[text]`、`[table]` 或 `[kg]`，使 Reader 能够区分证据类型。最终答案由 Reader 生成：

\[
\hat{a}=G_{\psi}(q,P(q)).
\]

这里 \(G_{\psi}\) 可以是任意大语言模型。本文分别使用 GPT-4o-mini 和 Llama-3.3-70B 作为 Reader，以验证 TESSERA 是否具备跨 Reader 的泛化能力。

TESSERA 的整体推理过程如下：

```text
Algorithm 1: TESSERA Inference
Input: question q, evidence corpus D, retrievers Rb/Rd/Rs, utility model uθ, budget K
Output: answer â
1. Build candidate set C(q) = Cb(q) ∪ Cd(q) ∪ Cs(q)
2. For each d in C(q):
      compute feature vector φ(q,d)
      compute utility score uθ(q,d)
      compute fused score s~(q,d)
3. Initialize S = ∅
4. While |S| < K:
      choose d* with maximum marginal gain in J(q,S∪{d})
      S = S ∪ {d*}
5. Format selected evidence S into prompt context P(q)
6. Generate answer â = Gψ(q,P(q))
```

## 4 实验

### 4.1 实验设置

本文在包含 text、table、KG 三类证据的多模态 RAG 问答数据集上进行实验。所有方法使用相同的数据划分和相同的候选证据库。TESSERA 的监督模型只使用训练集构造训练样本，测试阶段不使用测试集答案或标注信息。

端到端问答指标包括 F1、Exact Match（EM）和 Official Score。F1 衡量预测答案与标准答案的 token 级重叠，EM 衡量预测答案是否与标准答案完全一致。设预测答案 token 集为 \(P\)，标准答案 token 集为 \(A\)，则：

\[
Precision=\frac{|P\cap A|}{|P|+\epsilon},
\]

\[
Recall=\frac{|P\cap A|}{|A|+\epsilon},
\]

\[
F1=\frac{2\cdot Precision\cdot Recall}{Precision+Recall+\epsilon}.
\]

EM 定义为：

\[
EM=\mathbb{I}(\operatorname{Norm}(\hat{a})=\operatorname{Norm}(a)).
\]

Official Score 表示 mmRAG 官方生成评价协议下的平均得分。该协议不是单一的 token-F1，而是根据子数据集采用不同的答案匹配方式。对于 NQ、TAT 和 CWQ 等答案可能包含集合形式的子数据集，使用集合级 F1：

\[
\operatorname{SetF1}
=\frac{2|\hat{A}\cap A|}
{2|\hat{A}\cap A|+|\hat{A}\setminus A|+|A\setminus \hat{A}|},
\]

其中 \(A\) 为标准答案集合，\(\hat{A}\) 为模型预测答案集合。对于其他子数据集，官方分数采用严格答案匹配：

\[
\operatorname{Official}(q)=
\begin{cases}
\operatorname{SetF1}(\hat{A},A), & q\in\{\mathrm{NQ},\mathrm{TAT},\mathrm{CWQ}\},\\
\mathbb{I}(\hat{a}=a), & \text{otherwise}.
\end{cases}
\]

因此 Official Score 更接近 mmRAG 原始评测协议，通常比归一化 EM/F1 更严格。本文在主实验中同时报告 F1、EM 和 Official Score，以分别反映宽松 token 重叠、严格归一化匹配和数据集官方协议下的性能。

除跨 Reader 泛化实验外，本文的消融实验、上下文预算分析和分类型误差分析均以 GPT-4o-mini 作为默认 Reader。这一设置有两个原因。第一，GPT-4o-mini 是本文主要端到端实验采用的 Reader，在该 Reader 下进行模块消融可以直接解释主结果的性能来源。第二，消融与诊断实验的目标是分析 TESSERA 内部模块的作用，而不是比较不同 Reader 的生成能力，因此保持 Reader 固定能够减少模型差异带来的干扰。Llama-3.3-70B 实验主要用于验证同一证据组织方法在更强 Reader 下是否仍具有泛化优势。

除答案指标外，端到端表中还报告证据覆盖指标，用于诊断输入 Reader 的证据是否覆盖标准相关证据。对于从完整 ranking 直接评估的 GPT-4o-mini 实验，我们报告 Evidence Recall@10。设 \(\operatorname{Top}_{10}(q)\) 为方法返回的前 10 个证据，\(\mathcal{E}^{+}(q)\) 为 mmRAG `relevant_chunks` 中标记为正的证据集合，则：

\[
\operatorname{EvidenceRecall}@10
=\frac{|\operatorname{Top}_{10}(q)\cap \mathcal{E}^{+}(q)|}
{|\mathcal{E}^{+}(q)|+\epsilon}.
\]

需要说明的是，Evidence Recall@10 是诊断性检索指标，不是答案生成指标。我们在主实验表中保留它，是为了说明答案性能提升是否伴随证据覆盖提升；而 Recall@2 或 Recall@5 不放入端到端主表，是因为早期排序质量已经在检索实验中通过 NDCG@1、NDCG@5、MAP@1、MAP@5、Hits@1 和 Hits@5 更细致地报告。

为了使不同 Reader 下的主实验表具有相同含义，GPT-4o-mini 和 Llama-3.3-70B 两张端到端主表均报告 Evidence Recall@10。需要注意的是，该指标只依赖检索排序和标准相关证据标注，不依赖 Reader；因此对于同一个方法，GPT-4o-mini 表和 Llama-3.3-70B 表中的 Evidence Recall@10 数值相同。该列用于说明方法本身的检索覆盖能力，而 F1、EM 和 Official Score 则反映不同 Reader 基于证据生成答案后的端到端效果。

在上下文预算分析中，本文另使用 Context Evidence Recall@K 来衡量实际输入 Reader 的前 \(K\) 条证据覆盖了多少标准相关证据，其定义为：

\[
\operatorname{ContextEvidenceRecall}@K
=\frac{|\operatorname{Ctx}_{K}(q)\cap \mathcal{E}^{+}(q)|}
{|\mathcal{E}^{+}(q)|+\epsilon}.
\]

该指标反映 Reader 实际接收到的上下文证据质量，适合用于分析不同上下文预算 \(K\) 对答案生成的影响。与 Evidence Recall@10 不同，Context Evidence Recall@K 会随着输入 Reader 的证据数量变化而变化。

检索指标包括 NDCG@1、NDCG@5、MAP@1、MAP@5、Hits@1、Hits@5 和 AnyHit@5。DCG@k 定义为：

\[
DCG@k=\sum_{i=1}^{k}\frac{2^{rel_i}-1}{\log_2(i+1)},
\]

其中 \(rel_i\) 表示第 \(i\) 个检索结果的相关性。NDCG@k 为：

\[
NDCG@k=\frac{DCG@k}{IDCG@k}.
\]

Average Precision@k 定义为：

\[
AP@k=\frac{1}{\min(k,|\mathcal{R}|)}
\sum_{i=1}^{k}P@i\cdot rel_i,
\]

其中 \(\mathcal{R}\) 为相关证据集合，\(P@i\) 表示前 \(i\) 个检索结果中的相关证据比例：

\[
P@i=\frac{1}{i}\sum_{j=1}^{i}\mathbb{I}(d_j\in\mathcal{R}).
\]

MAP@k 是所有问题的 AP@k 平均值：

\[
MAP@k=\frac{1}{|\mathcal{Q}|}\sum_{q\in\mathcal{Q}}AP@k(q).
\]

Hits@k 统计 top-k 中命中的相关证据数量。对单个问题 \(q\)，设检索排序前 \(k\) 个证据为 \((d_1,\ldots,d_k)\)，相关证据集合为 \(\mathcal{E}^{+}(q)\)，则：

\[
\operatorname{Hits}@k(q)=
\sum_{i=1}^{k}\mathbb{I}(d_i\in\mathcal{E}^{+}(q)).
\]

数据集上的 Hits@k 为所有问题的平均值：

\[
\operatorname{Hits}@k=
\frac{1}{|\mathcal{Q}|}\sum_{q\in\mathcal{Q}}\operatorname{Hits}@k(q).
\]

AnyHit@k 表示 top-k 中是否至少存在一个相关证据：

\[
\operatorname{AnyHit}@k(q)=
\mathbb{I}\left(\operatorname{Hits}@k(q)>0\right).
\]

数据集上的 AnyHit@k 同样取平均：

\[
\operatorname{AnyHit}@k=
\frac{1}{|\mathcal{Q}|}\sum_{q\in\mathcal{Q}}\operatorname{AnyHit}@k(q).
\]

### 4.2 对比方法

本文比较以下方法：

1. **Dense-Concat**：将所有候选证据拼接或统一处理后使用 dense 检索排序，是最强基础 baseline。
2. **NaiveRAG**：标准 RAG 检索与 Reader 组合，不显式建模多源互补。
3. **CARP**：考虑上下文或候选重排的 RAG baseline。
4. **TableRAG**：偏向表格证据处理的 RAG baseline。
5. **QUASAR**：面向复杂问题的检索增强 baseline。
6. **UniHGKR**：利用异构知识检索的 baseline。
7. **TESSERA**：本文提出的任务感知异构证据集合选择方法。

所有方法在相同测试集上评估。对于 Reader 泛化实验，分别使用 GPT-4o-mini 与 Llama-3.3-70B。

### 4.3 主实验结果

表 1 汇总了 GPT-4o-mini 与 Llama-3.3-70B 两种下游生成模型下的端到端问答结果。TESSERA 在两种下游生成模型的 F1、EM、Official Score 和 Evidence Recall@10 上均取得最优结果，说明其收益并不依赖某一个特定生成模型。

**表 1：端到端问答实验结果**

| Reader | 方法 | F1 | EM | Official | Evidence Recall@10 |
|---|---|---:|---:|---:|---:|
| GPT-4o-mini | Dense-Concat | 0.4116 | 0.2784 | 0.2526 | 0.5267 |
| GPT-4o-mini | NaiveRAG | 0.3830 | 0.2597 | 0.2382 | 0.3722 |
| GPT-4o-mini | CARP | 0.3925 | 0.2691 | 0.2419 | 0.4438 |
| GPT-4o-mini | TableRAG | 0.3729 | 0.2473 | 0.2244 | 0.3560 |
| GPT-4o-mini | QUASAR | 0.3766 | 0.2543 | 0.2297 | 0.3518 |
| GPT-4o-mini | UniHGKR | 0.3227 | 0.2177 | 0.2008 | 0.1853 |
| GPT-4o-mini | **TESSERA** | **0.4628** | **0.3204** | **0.2925** | **0.6071** |
| Llama-3.3-70B | Dense-Concat | 0.4369 | 0.2947 | 0.1103 | 0.5267 |
| Llama-3.3-70B | NaiveRAG | 0.4188 | 0.2823 | 0.1072 | 0.3722 |
| Llama-3.3-70B | CARP | 0.4289 | 0.2869 | 0.1195 | 0.4438 |
| Llama-3.3-70B | TableRAG | 0.4146 | 0.2768 | 0.1235 | 0.3560 |
| Llama-3.3-70B | QUASAR | 0.4096 | 0.2722 | 0.0978 | 0.3518 |
| Llama-3.3-70B | UniHGKR | 0.3851 | 0.2652 | 0.0947 | 0.1853 |
| Llama-3.3-70B | **TESSERA** | **0.4817** | **0.3344** | **0.1386** | **0.6071** |

在 GPT-4o-mini 上，Dense-Concat 是最强基线方法。相较 Dense-Concat，TESSERA 在 F1 上提升 0.0512，约为 12.4%；在 EM 上提升 0.0420，约为 15.1%；在 Official Score 上提升 0.0399，约为 15.8%；在 Evidence Recall@10 上提升 0.0804，约为 15.3%。这说明 TESSERA 不仅提升了证据覆盖，也提升了下游生成模型最终生成答案的准确性。更重要的是，TESSERA 的 F1/EM 提升方向与 Evidence Recall@10 的提升方向一致，说明其收益不是简单的提示词偶然性，而是来自更有效的证据组织。

在 Llama-3.3-70B 上的实验用于检验在更强下游生成模型下，TESSERA 的证据组织优势是否仍能转化为答案性能优势。需要说明的是，下游生成模型本身不负责检索；上下文证据由各方法的前置检索和排序模块产生，Llama-3.3-70B 只基于给定证据生成答案。由于 Evidence Recall@10 与下游生成模型无关，因此同一方法在 GPT-4o-mini 和 Llama-3.3-70B 下的 Evidence Recall@10 数值相同。

在 Llama-3.3-70B 下，Dense-Concat 仍是 F1 和 EM 上最强的基线方法。相较 Dense-Concat，TESSERA 的 F1 提升 0.0447，约为 10.2%；EM 提升 0.0397，约为 13.5%。在 Official Score 上，TableRAG 是最强基线方法，TESSERA 仍比其高 0.0151，约为 12.2%。该结果说明，即使更换为更强的 Llama-3.3-70B 下游生成模型，TESSERA 仍然保持最优的端到端答案性能。

综合两种模型上的结果可以看到，TESSERA 在两种下游生成模型上的表现均优于所有基线方法。由于 Evidence Recall@10 是模型无关的检索覆盖指标，同一方法在两种模型中的该列数值相同。这一设计使表 1 能够同时展示两层信息：一方面，TESSERA 的检索排序能够覆盖更多标准相关证据；另一方面，在 GPT-4o-mini 和 Llama-3.3-70B 两种不同下游生成模型下，这种证据组织优势都能稳定转化为更高的 F1、EM 和 Official Score。

此外，Llama-3.3-70B 的 F1 和 EM 整体高于 GPT-4o-mini，但 Official Score 整体低于 GPT-4o-mini。这并不表示 Llama 的答案质量整体更差，而是因为 Official Score 对原始答案格式更敏感。Llama 更倾向于输出完整句子、解释性短语或带标点的答案；这些预测在归一化 F1/EM 下可能仍然正确，但在部分子数据集的官方严格匹配下会被判错。因此，本文将 F1、EM 和 Official Score 同时报告：F1/EM 反映语义和归一化答案匹配，Official Score 反映更严格的原始评测协议。

### 4.4 检索结果

表 2 展示了检索阶段的主要指标。为保证与端到端实验一致，本节补充所有 baseline 的检索结果。与端到端表中的 Evidence Recall@10 不同，表 2 更关注前几个位置的排序质量，因此报告 NDCG、MAP、Hits 和 AnyHit 在 @1 与 @5 下的表现。NDCG 和 MAP 反映排序质量，Hits@k 反映前 \(k\) 个证据中相关证据的平均命中数量，AnyHit@k 反映前 \(k\) 个证据中至少命中一个相关证据的概率。

**表 2：检索指标对比**

| 方法 | NDCG@1 | NDCG@5 | MAP@1 | MAP@5 | Hits@1 | Hits@5 | AnyHit@5 |
|---|---:|---:|---:|---:|---:|---:|---:|
| Dense-Concat | 0.5892 | 0.5942 | 0.1149 | 0.3928 | 0.7022 | 3.0972 | 0.8686 |
| NaiveRAG | 0.5555 | 0.5146 | 0.1105 | 0.3373 | 0.6617 | 2.5226 | 0.7947 |
| CARP | 0.5819 | 0.5444 | 0.1149 | 0.3510 | 0.6820 | 2.6267 | 0.8538 |
| TableRAG | 0.5526 | 0.5050 | 0.1135 | 0.3252 | 0.6407 | 2.3779 | 0.8180 |
| QUASAR | 0.3883 | 0.3504 | 0.0683 | 0.1899 | 0.4619 | 1.7333 | 0.7815 |
| UniHGKR | 0.1843 | 0.1712 | 0.0388 | 0.0963 | 0.2170 | 0.7932 | 0.4230 |
| **TESSERA** | **0.6459** | **0.6544** | **0.1286** | **0.4396** | **0.7636** | **3.2924** | **0.9269** |

可以看到，Dense-Concat 是最强 baseline，但 TESSERA 在所有检索指标上均超过 Dense-Concat。尤其是在 NDCG@5、MAP@5、Hits@5 和 AnyHit@5 上，TESSERA 的优势更加明显。这与本文方法目标一致：TESSERA 不只优化 top-1 的单点相关性，而是优化前若干个证据组成的整体证据集合。对于多模态 RAG 问题，前 5 个证据的覆盖度和互补性往往比单个 top-1 证据更重要，因为 Reader 需要从多个来源中整合信息。TESSERA 在 Hits@5 和 AnyHit@5 上的提升说明，它能把更多正证据放入早期上下文候选中，从而为端到端问答性能提升提供基础。

### 4.5 消融实验

表 3 展示了 TESSERA 的模块级消融实验。需要说明的是，TESSERA 不是只添加一个单独模块，而是由异构候选证据组织、监督式证据效用建模、证据集合选择和保护策略共同组成。为了避免消融过于零散，本文将消融组织为三个层次：第一，Dense-Concat 表示不使用 TESSERA 的任务感知证据选择；第二，w/o Utility 保留候选组合与证据组织，但去掉监督式证据效用模型；第三，Head-Protected 和 Over-Protected 分别考察对高排名 anchor evidence 的保护强度。

**表 3：消融实验结果（GPT-4o-mini, ctx6）**

| 方法 | F1 | EM | Official |
|---|---:|---:|---:|
| w/o TESSERA Selection | 0.3880 | 0.2667 | 0.2435 |
| w/o Utility | 0.4480 | 0.3056 | 0.2796 |
| **TESSERA** | **0.4667** | **0.3212** | **0.2952** |
| Head-Protected Variant | 0.4605 | 0.3196 | 0.2923 |
| Over-Protected Variant | 0.4579 | 0.3188 | 0.2905 |

与 w/o TESSERA Selection 相比，w/o Utility 的 F1 从 0.3880 提升到 0.4480，说明仅仅进行异构候选组织和证据集合式上下文构造，就已经能够显著改善 Reader 的输入质量。在此基础上，加入监督式证据效用模型后，F1 进一步提升到 0.4667，EM 从 0.3056 提升到 0.3212，说明效用模型能够学习训练集中问题类型、证据来源和答案形式之间的关系，是 TESSERA 的关键增益来源。

Head-Protected 变体略低于 TESSERA，说明保留高置信 anchor evidence 有助于稳定排序，但如果保护过强，会限制模型根据任务需求重组证据集合。Over-Protected 进一步下降，说明过分依赖原始高排名证据会削弱异构证据集合选择的作用。因此，TESSERA 的最终配置采用适度保护策略，而不是简单固定 top evidence。

### 4.6 补充分析

本节补充分析均基于 GPT-4o-mini Reader。这样做是为了固定答案生成模型，使结果变化主要反映 TESSERA 证据选择与上下文组织策略本身的影响。Llama-3.3-70B 的结果在主实验中作为跨 Reader 泛化验证报告，不在消融和诊断分析中重复展开。

#### 4.6.1 上下文预算分析

表 4 展示了不同上下文预算下的结果。上下文预算指最终输入 Reader 的证据数量，而不是检索阶段保留的候选数量。通过改变上下文预算，可以观察 TESSERA 在有限证据槽位下组织 text、table、KG 证据的能力：如果预算过小，证据链可能不完整；如果预算过大，新增证据可能带来冗余或噪声。

**表 4：上下文预算分析**

| 方法 | 上下文预算 K | F1 | EM | Official | Context Evidence Recall@K |
|---|---:|---:|---:|---:|---:|
| TESSERA | 3 | 0.4290 | 0.2955 | 0.2723 | 0.3379 |
| TESSERA | 4 | 0.4529 | 0.3126 | 0.2892 | 0.4247 |
| TESSERA | 5 | 0.4546 | 0.3126 | 0.2910 | 0.4980 |
| **TESSERA** | **6** | **0.4667** | **0.3212** | **0.2952** | 0.5509 |
| TESSERA | 8 | 0.4635 | 0.3196 | 0.2929 | 0.5990 |
| TESSERA | 10 | 0.4631 | 0.3180 | 0.2914 | **0.6071** |

表中报告 Context Evidence Recall@K，而不是 Evidence Recall@10。原因是上下文预算分析关注的是实际输入 Reader 的前 K 条证据；在同一份 TESSERA 排序不变的情况下，Evidence Recall@10 不会随 K 改变，不能反映上下文预算变化的影响。Context Evidence Recall@K 则直接衡量 Reader 实际接收到的 K 条上下文证据覆盖了多少标准相关证据，因此更适合用于分析上下文预算。

从结果可以看到，随着 K 从 3 增加到 6，Context Evidence Recall@K 从 0.3379 提升到 0.5509，F1 从 0.4290 提升到 0.4667，EM 从 0.2955 提升到 0.3212，Official 从 0.2723 提升到 0.2952。这说明较小上下文预算容易导致证据链不完整，尤其是多跳或跨来源问题往往需要多个 text、table、KG 证据共同支撑答案。

当 K 继续增加到 8 或 10 时，Context Evidence Recall@K 仍然上升，分别达到 0.5990 和 0.6071，但答案指标没有继续提升，反而较 K=6 略有下降。这表明更高的证据覆盖并不必然带来更好的端到端答案质量。一个可能原因是，新增证据虽然包含更多相关片段，但同时也会引入冗余信息、长上下文噪声或与答案无关的相邻内容，从而增加 Reader 的定位难度。因此，TESSERA 的目标不是无限增加上下文，而是在有限预算内选择信息密度最高、互补性最强且便于 Reader 使用的一组证据。综合答案指标与上下文覆盖率，本文后续实验采用 K=6 作为默认上下文预算。

#### 4.6.2 分类型误差分析

为了进一步分析 TESSERA 在不同多模态场景中的表现，本文按照测试集中标准相关证据的来源组成对问题进行分组。具体来说，mmRAG 为每个问题提供相关证据标注 `relevant_chunks`，这些证据来自 text、table 或 KG 三类文本化证据。如果一个问题的标准相关证据只来自 text，则记为 text-only；只来自 table，则记为 table-only；只来自 KG，则记为 kg-only；如果标准相关证据同时涉及两类或三类来源，则记为 multi-modal。

因此，表 5 不是在比较不同方法，而是在回答一个诊断问题：同一个 TESSERA 系统在不同证据来源需求的问题上表现是否一致。表中的“数量”表示该类型测试问题的个数；F1、EM 和 Official 是这些问题上的答案质量；Evidence Recall@10 表示 TESSERA 排序前 10 个证据中覆盖了多少该类型问题的标准相关证据。

**表 5：TESSERA 分类型结果（GPT-4o-mini）**

| 问题类型 | 数量 | F1 | EM | Evidence Recall@10 | Official |
|---|---:|---:|---:|---:|---:|
| text-only | 315 | 0.5938 | 0.4349 | 0.7354 | 0.3619 |
| table-only | 236 | 0.2221 | 0.1102 | 0.7814 | 0.0847 |
| kg-only | 88 | 0.4544 | 0.3523 | 0.4002 | 0.3707 |
| multi-modal | 647 | 0.4958 | 0.3385 | 0.5092 | 0.3293 |

从结果可以看到，TESSERA 在 text-only 问题上表现最好，F1 达到 0.5938，说明当答案主要依赖自然语言文本证据时，当前的证据选择和 Reader 都能较好工作。multi-modal 问题的 F1 为 0.4958，也明显高于整体较弱的 table-only 子集，说明 TESSERA 的跨来源证据组织确实能够帮助多源问题，而不是只对单一文本来源有效。

kg-only 问题呈现另一种现象：Evidence Recall@10 为 0.4002，相对较低，但 EM 达到 0.3523，Official 达到 0.3707。这说明 KG 相关证据的主要难点在前端召回和排序；一旦正确 KG 证据进入上下文，Reader 通常能够较直接地抽取实体或关系型答案。

table-only 问题则是当前最明显的瓶颈。它的 Evidence Recall@10 达到 0.7814，是四类问题中最高的，但 F1 只有 0.2221，EM 只有 0.1102。这意味着很多表格相关证据已经被 TESSERA 检索到并排入前 10，但 Reader 仍然没有稳定给出正确答案。因此，table-only 的主要问题不再是“有没有找到相关表格证据”，而是“找到的表格证据是否被压缩、定位和解释成 Reader 容易使用的形式”。

这一发现解释了为什么继续单纯提升检索指标不一定带来同等幅度的端到端收益。对于 table-only 问题，即使相关表格证据已经进入 top-10，Reader 仍可能因为表格转写过长、字段关系不清、数值单位混淆或多行多列定位失败而给出错误答案。因此，后续工作可以在表格证据压缩、表格 cell-level grounding 和数值推理方面进一步改进。

## 5 讨论与局限性

TESSERA 的实验结果表明，多模态 RAG 的关键瓶颈并不只是召回更多证据，而是如何选择一组适合 Reader 使用的异构证据。相比单证据排序，证据集合选择能够更好地处理多跳问题和跨来源问题。监督式证据效用模型能够学习训练集中问题类型与证据来源之间的关系，使系统在不同任务需求下动态调整 text、table、KG 的使用方式。

同时，TESSERA 也存在一些局限。第一，方法依赖训练集中的监督信号。如果训练集中缺少高质量证据标注或答案匹配规则较弱，效用模型的学习效果会受到影响。第二，当前方法主要在文本化后的表格和 KG 证据上工作，没有直接建模原始表格结构或图结构，因此在 table-only 问题上仍存在明显瓶颈。第三，证据集合选择使用贪心近似，虽然效率较高，但无法保证得到全局最优证据集合。第四，当前 Reader 仍可能受到上下文长度、证据顺序和提示模板影响，尤其是在涉及复杂数值推理和表格定位的问题中。

从实验结果看，TESSERA 已经在两个不同 Reader 上稳定优于所有 baseline，说明方法具有一定泛化性。不过，分类型误差分析也表明，表格理解仍是后续最值得深入优化的方向。未来可以将 TESSERA 与更细粒度的表格单元选择、结构化表格编码器或程序化数值推理模块结合，以进一步提升 table-only 问题的 F1 和 EM。

## 6 结论

本文提出 TESSERA，一种面向多模态 RAG 的任务感知异构证据集合选择方法。不同于将 text、table、KG 统一文本化后直接排序的传统做法，TESSERA 在统一候选空间中学习证据效用，并通过集合选择目标联合考虑效用、覆盖度、来源互补性、anchor 保护和冗余控制。实验结果表明，TESSERA 在 GPT-4o-mini 和 Llama-3.3-70B 两种 Reader 下均显著优于现有 baseline，并在检索指标、端到端问答指标和消融实验中表现稳定。进一步分析显示，上下文预算和证据类型对最终效果具有重要影响，其中 table-only 问题仍是当前多模态 RAG 的主要瓶颈。总体而言，TESSERA 为多模态 RAG 中的异构证据整合提供了一种有效且可解释的解决方案。
