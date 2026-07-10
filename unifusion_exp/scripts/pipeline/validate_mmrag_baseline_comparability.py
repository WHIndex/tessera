#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

from sklearn.metrics import f1_score
from sklearn.preprocessing import MultiLabelBinarizer


# Extracted from mmRAG paper (arXiv:2505.11180v1, Table 9, Used with Qwen, Over All Chunks)
OFFICIAL_QWEN_BGE_AVG = 0.2223
OFFICIAL_QWEN_ORACLE_AVG = 0.3720


@dataclass
class SplitCheck:
    split_name: str
    official_count: int
    ours_count: int
    intersection: int
    official_only: int
    ours_only: int
    same_order: bool

    @property
    def passed(self) -> bool:
        return (
            self.official_count == self.ours_count
            and self.intersection == self.official_count
            and self.official_only == 0
            and self.ours_only == 0
            and self.same_order
        )


def load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def to_list(value):
    if isinstance(value, list):
        return [str(x) for x in value]
    try:
        parsed = eval(str(value))
        if isinstance(parsed, list):
            return [str(x) for x in parsed]
    except Exception:
        pass
    return [str(value)]


def mmrag_set_f1(gold_list, pred_list) -> float:
    mlb = MultiLabelBinarizer()
    mlb.fit([gold_list, pred_list])
    y_true = mlb.transform([gold_list])
    y_pred = mlb.transform([pred_list])
    return float(f1_score(y_true, y_pred, average="micro"))


def check_split(official_file: Path, ours_file: Path, split_name: str) -> SplitCheck:
    off = load_json(official_file)
    me = load_json(ours_file)
    off_ids = [x["id"] for x in off]
    me_ids = [x["id"] for x in me]
    so = set(off_ids)
    sm = set(me_ids)
    return SplitCheck(
        split_name=split_name,
        official_count=len(off_ids),
        ours_count=len(me_ids),
        intersection=len(so & sm),
        official_only=len(so - sm),
        ours_only=len(sm - so),
        same_order=(off_ids == me_ids),
    )


def compute_mmrag_official_like_avg(test_split: Path, pred_jsonl: Path) -> dict:
    rows = load_json(test_split)
    gold_map = {x["id"]: x["answer"] for x in rows}
    preds = [json.loads(line) for line in pred_jsonl.read_text(encoding="utf-8").splitlines() if line.strip()]

    f1_ds = {"nq", "tat", "cwq"}
    by_ds: dict[str, list[float]] = {}
    for p in preds:
        qid = p["id"]
        pred = p["prediction"]
        gold = gold_map[qid]
        ds = qid.split("_")[0]
        by_ds.setdefault(ds, [])
        if ds in f1_ds:
            score = mmrag_set_f1(to_list(gold), to_list(pred))
        else:
            score = float(str(gold) == str(pred))
        by_ds[ds].append(score)

    ds_means = {k: (sum(v) / len(v) if v else 0.0) for k, v in by_ds.items()}
    macro = sum(ds_means.values()) / max(1, len(ds_means))
    micro = sum(sum(v) for v in by_ds.values()) / max(1, sum(len(v) for v in by_ds.values()))

    return {
        "dataset_means": ds_means,
        "macro6": macro,
        "micro_all": micro,
        "query_count": len(preds),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate Dense baseline comparability against mmRAG official setup")
    parser.add_argument(
        "--official-split-dir",
        type=Path,
        default=Path("/home/yongqi.yin/reaserch_paper/downloaded_resource/mmRAG/data/mmRAG_ds"),
    )
    parser.add_argument(
        "--router-split-dir",
        type=Path,
        default=Path("/home/yongqi.yin/reaserch_paper/unifusion_exp/runs/router_deberta_full_v1_router_deberta/router_data"),
    )
    parser.add_argument(
        "--table1c-metrics",
        type=Path,
        default=Path(
            "/home/yongqi.yin/reaserch_paper/unifusion_exp/artifacts/results/"
            "table1c_e2e_20260330_llm_qwen25_campe_full1286_all_v3_unihgkrfix/table1c_e2e_metrics.json"
        ),
    )
    parser.add_argument(
        "--dense-pred",
        type=Path,
        default=Path(
            "/home/yongqi.yin/reaserch_paper/unifusion_exp/artifacts/results/"
            "table1c_e2e_20260330_llm_qwen25_campe_full1286_all_v3_unihgkrfix/qa_predictions_dense_concat_test1286.jsonl"
        ),
    )
    parser.add_argument(
        "--oracle-pred",
        type=Path,
        default=Path(
            "/home/yongqi.yin/reaserch_paper/unifusion_exp/artifacts/results/"
            "table1c_e2e_20260330_llm_qwen25_campe_full1286_all_v3_unihgkrfix/qa_predictions_oracle_gold_test1286.jsonl"
        ),
    )
    parser.add_argument(
        "--out-json",
        type=Path,
        default=Path(
            "/home/yongqi.yin/reaserch_paper/unifusion_exp/artifacts/results/"
            "mmrag_dense_comparability_audit_20260331.json"
        ),
    )
    args = parser.parse_args()

    checks = [
        check_split(args.official_split_dir / "mmrag_train.json", args.router_split_dir / "router_train.json", "train"),
        check_split(args.official_split_dir / "mmrag_dev.json", args.router_split_dir / "router_val.json", "dev"),
        check_split(args.official_split_dir / "mmrag_test.json", args.router_split_dir / "router_test.json", "test"),
    ]

    dense_like = compute_mmrag_official_like_avg(args.official_split_dir / "mmrag_test.json", args.dense_pred)
    oracle_like = compute_mmrag_official_like_avg(args.official_split_dir / "mmrag_test.json", args.oracle_pred)

    dense_delta = float(dense_like["macro6"] - OFFICIAL_QWEN_BGE_AVG)
    oracle_delta = float(oracle_like["macro6"] - OFFICIAL_QWEN_ORACLE_AVG)

    metrics_meta = load_json(args.table1c_metrics).get("meta", {})
    config_diff = {
        "ours_reader": metrics_meta.get("reader"),
        "ours_ollama_model": metrics_meta.get("ollama_model"),
        "ours_qa_context_k": metrics_meta.get("qa_context_k"),
        "official_qwen_context_k": 3,
        "official_qwen_generation_metric": "dataset-wise mixed metric (NQ/TAT/CWQ use set-F1; others use EM)",
        "ours_generation_metric": "global normalized EM/F1 over all queries",
    }

    result = {
        "split_check": {
            "all_passed": all(c.passed for c in checks),
            "details": [
                {
                    "split": c.split_name,
                    "official_count": c.official_count,
                    "ours_count": c.ours_count,
                    "intersection": c.intersection,
                    "official_only": c.official_only,
                    "ours_only": c.ours_only,
                    "same_order": c.same_order,
                    "passed": c.passed,
                }
                for c in checks
            ],
        },
        "dense_alignment": {
            "official_qwen_bge_macro6": OFFICIAL_QWEN_BGE_AVG,
            "ours_dense_macro6_official_like": dense_like["macro6"],
            "delta_abs": dense_delta,
            "pass_threshold_abs": 0.03,
            "passed": abs(dense_delta) <= 0.03,
            "ours_dense_details": dense_like,
        },
        "oracle_alignment": {
            "official_qwen_oracle_macro6": OFFICIAL_QWEN_ORACLE_AVG,
            "ours_oracle_macro6_official_like": oracle_like["macro6"],
            "delta_abs": oracle_delta,
            "pass_threshold_abs": 0.05,
            "passed": abs(oracle_delta) <= 0.05,
            "ours_oracle_details": oracle_like,
        },
        "config_differences": config_diff,
        "final_gate": {
            "all_three_passed": all(c.passed for c in checks)
            and (abs(dense_delta) <= 0.03)
            and (abs(oracle_delta) <= 0.05)
        },
    }

    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result["final_gate"], ensure_ascii=False))
    print(f"[OK] audit json -> {args.out_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
