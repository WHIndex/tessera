# Train Scripts

本目录当前包含三类训练入口：

## 1. 学习式控制器主线

- `train_evidence_planner.py`
- `train_evidence_verifier.py`

这两条脚本对应论文中的 planner / verifier：

- planner 负责把查询映射为模态与预算先验。
- verifier 负责对 evidence pack 做 support 判别。

## 2. 其他可复用训练入口

- `train_router.py`
- `train_router_deberta.py`
- `train_transe_subgraph.py`

## 3. 参数与输出约定

建议统一支持以下参数：

- `--config`
- `--run-id`
- `--output-dir`
- `--seed`
- `--max-train`
- `--max-val`
- `--max-test`

推荐输出命名：

- bundle：`*_bundle.pkl`
- metrics：`*_metrics.json`

## 4. 使用原则

- 先用小样本确认特征、标签与 bundle 写出，再扩大训练规模。
- planner 与 verifier 尽量使用同一套 Python 环境训练和加载，避免 sklearn 版本差异导致的 bundle 兼容问题。
- 如果只是验证流程，优先使用 `--max-train` / `--max-val` 做 smoke。

## 5. 推荐顺序

1. 先训练 planner。
2. 再训练 verifier。
3. 然后跑 `scripts/eval/run_e2e_table1c.py` 做 controller smoke。
