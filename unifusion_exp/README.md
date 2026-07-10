# TESSERA 实验工作区

本目录保存 TESSERA 的实验代码、训练脚本、评测脚本和论文材料。TESSERA 的全称是 **Task-aware Evidence Set Selection and Efficient Retrieval Assembly**，面向 mmRAG 中由文本、表格和知识图谱转写得到的统一 evidence chunk 空间，学习并组织更适合下游问答的证据集合。

历史代码中仍保留少量 `unifusion_*` 内部名称，用于兼容早期实验结果和已有 rankings 文件；论文、README、最终运行入口和结果汇总均统一使用 **TESSERA**。

## 目录说明

```text
unifusion_exp/
  configs/                 # 路径模板与实验配置模板
  scripts/
    analysis/              # 误差分析、瓶颈诊断和结果比较
    data/                  # 数据检查、语料构建和索引准备
    eval/                  # 检索评测、reader 评测和端到端实验
    train/                 # 证据效用模型、证据集合重排器等训练入口
  src/unifusion_exp/       # TESSERA 核心实现；包名保留为兼容名
  TESSERA_PAPER_DRAFT_CN.md
  TESSERA_RESULTS_SUMMARY.md
```

`artifacts/`、`logs/`、`runs/`、本地 conda 环境和下载数据均为本机实验产物，不进入 Git。

## 主要入口

检索实验：

```bash
bash run_tessera_retrieval_experiment.sh
```

GPT-4o-mini 端到端 reader 实验：

```bash
bash run_tessera_reader_gpt4omini.sh
```

Llama-3.3-70B 端到端 reader 实验：

```bash
bash run_tessera_llama33_70b_ctx6_parallel.sh
```

## 方法概览

TESSERA 不把 text、table、KG 证据看成彼此独立的检索结果，而是在统一候选空间中进行证据集合选择。整体流程包括：

1. 从统一 evidence chunk 空间中收集 dense、sparse 和结构化邻接候选。
2. 使用训练集标注构造正负证据对，训练 evidence utility model。
3. 在候选池中执行证据集合选择，综合效用、覆盖、来源互补、冗余惩罚和锚点保护。
4. 将选出的 top-k 证据交给下游 reader，并分别报告检索指标与端到端 QA 指标。

训练只使用 train/dev 中的 qrels 或相关证据标注，不使用 test 答案或 test 标注调参。
