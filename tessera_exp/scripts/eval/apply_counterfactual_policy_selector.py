#!/usr/bin/env python3
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
import json
import math
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from tessera_exp.e2e.counterfactual_policy_selector import (  # noqa: E402
    load_counterfactual_policy_selector_bundle,
    query_family,
)


def iter_jsonl(path: Path, max_examples: int = 0) -> list[dict]:
    rows: list[dict] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            rows.append(json.loads(line))
            if int(max_examples) > 0 and len(rows) >= int(max_examples):
                break
    return rows


def qrels_for_row(row: dict) -> dict[str, float]:
    out: dict[str, float] = {}
    for doc_id, raw in (row.get("qrels", {}) or {}).items():
        try:
            val = float(raw)
        except Exception:
            continue
        if val > 0.0:
            out[str(doc_id)] = val
    return out


def ranking_for_row(row: dict, method: str) -> list[str]:
    rankings = row.get("rankings", {}) or {}
    return [str(x) for x in rankings.get(method, [])]


def dcg(grades: list[float]) -> float:
    return float(sum((2.0 ** float(grade) - 1.0) / math.log2(rank + 2.0) for rank, grade in enumerate(grades)))


def method_metrics(rows: list[dict], preds: list[list[str]], ks: list[int]) -> tuple[dict, dict]:
    details: dict[str, list[float]] = {}
    rel_counts: list[int] = []
    for k in ks:
        details[f"ndcg@{k}"] = []
        details[f"map@{k}"] = []
        details[f"hits@{k}"] = []
        details[f"any_hit@{k}"] = []
    for row, pred in zip(rows, preds):
        qrels = qrels_for_row(row)
        rel = {doc_id for doc_id, grade in qrels.items() if float(grade) > 0.0}
        rel_counts.append(len(rel))
        for k in ks:
            top = list(pred)[:k]
            grades = [float(qrels.get(doc_id, 0.0)) for doc_id in top]
            ideal = sorted([float(v) for v in qrels.values() if float(v) > 0.0], reverse=True)[:k]
            idcg = dcg(ideal)
            ndcg_val = dcg(grades) / idcg if idcg > 0.0 else 0.0
            hit_count = 0
            ap_sum = 0.0
            for rank, doc_id in enumerate(top, start=1):
                if doc_id in rel:
                    hit_count += 1
                    ap_sum += hit_count / rank
            map_val = ap_sum / len(rel) if rel else 0.0
            details[f"ndcg@{k}"].append(float(ndcg_val))
            details[f"map@{k}"].append(float(map_val))
            details[f"hits@{k}"].append(float(hit_count))
            details[f"any_hit@{k}"].append(float(hit_count > 0))
    summary = {"avg_positive_qrels": float(np.mean(rel_counts)) if rel_counts else 0.0}
    for key, vals in details.items():
        summary[key] = float(np.mean(vals)) if vals else 0.0
    return summary, {"rel_count": rel_counts, **details}


def write_csv(path: Path, rows: list[dict], metric_keys: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["method", "label", *metric_keys])
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in ["method", "label", *metric_keys]})


def write_markdown(path: Path, rows: list[dict], metric_keys: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    headers = ["Method", *metric_keys]
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for row in rows:
        values = [row["label"]] + [f"{float(row[key]):.4f}" for key in metric_keys]
        lines.append("| " + " | ".join(values) + " |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Apply CEPS to a saved rankings_debug.jsonl file.")
    parser.add_argument("--base-rankings-jsonl", type=Path, required=True)
    parser.add_argument("--selector-model", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--method", type=str, default="tessera_rag")
    parser.add_argument("--label", type=str, default="TESSERA-CEPS")
    parser.add_argument("--metrics-k", type=str, default="1,5")
    parser.add_argument("--switch-margin", type=float, default=None)
    parser.add_argument("--max-queries", type=int, default=0)
    parser.add_argument("--include-base", action="store_true")
    args = parser.parse_args()

    if not args.base_rankings_jsonl.exists():
        raise FileNotFoundError(f"base-rankings-jsonl not found: {args.base_rankings_jsonl}")
    if not args.selector_model.exists():
        raise FileNotFoundError(f"selector-model not found: {args.selector_model}")

    ks = sorted({int(x.strip()) for x in str(args.metrics_k).split(",") if x.strip()})
    if not ks:
        raise ValueError("metrics-k is empty")
    bundle = load_counterfactual_policy_selector_bundle(args.selector_model)
    margin = float(bundle.switch_margin if args.switch_margin is None else args.switch_margin)
    rows = iter_jsonl(args.base_rankings_jsonl, int(args.max_queries))
    preds: list[list[str]] = []
    base_preds: list[list[str]] = []
    debug_rows: list[dict] = []
    policy_counts: Counter[str] = Counter()
    family_policy_counts: dict[str, Counter[str]] = defaultdict(Counter)
    switched = 0
    for i, row in enumerate(rows):
        base = ranking_for_row(row, str(args.method))
        base_preds.append(base)
        ranked, selection = bundle.select(
            query_text=str(row.get("query", "")),
            query_id=str(row.get("query_id", "")),
            base_ranked_doc_ids=base,
            trace=row.get("trace", {}),
            switch_margin=margin,
        )
        preds.append(ranked)
        policy_counts[selection.policy] += 1
        family = query_family(str(row.get("query_id", "")))
        family_policy_counts[family][selection.policy] += 1
        switched += int(selection.switched)
        debug_rows.append(
            {
                "query_index": i,
                "query_id": row.get("query_id", ""),
                "query": row.get("query", ""),
                "qrels": qrels_for_row(row),
                "rankings": {
                    str(args.method): ranked,
                    "_base_tessera": base,
                },
                "trace": {
                    "tessera_ceps_active": 1.0,
                    "tessera_ceps_selected_policy": selection.policy,
                    "tessera_ceps_switched": float(selection.switched),
                    "tessera_ceps_predicted_utility": float(selection.predicted_utility),
                    "tessera_ceps_default_utility": float(selection.default_utility),
                    "tessera_ceps_predicted_margin": float(selection.margin),
                    "tessera_ceps_switch_margin": float(margin),
                    "tessera_ceps_policy_scores": selection.policy_scores,
                    "tessera_ceps_positive_probs": selection.positive_probs,
                },
            }
        )

    summary, detail = method_metrics(rows, preds, ks)
    metric_keys = []
    for metric in ["ndcg", "map", "hits"]:
        for k in ks:
            metric_keys.append(f"{metric}@{k}")
    for k in ks:
        metric_keys.append(f"any_hit@{k}")
    table_rows = [{"method": str(args.method), "label": str(args.label), **summary}]
    details = {"query_ids": [str(row.get("query_id", "")) for row in rows], "methods": {str(args.method): detail}}
    if bool(args.include_base):
        base_summary, base_detail = method_metrics(rows, base_preds, ks)
        table_rows.insert(0, {"method": "base_tessera", "label": "Base TESSERA", **base_summary})
        details["methods"]["base_tessera"] = base_detail

    args.out_dir.mkdir(parents=True, exist_ok=True)
    meta = {
        "base_rankings_jsonl": str(args.base_rankings_jsonl),
        "selector_model": str(args.selector_model),
        "selector_meta_summary": {
            "method_name": (bundle.metadata or {}).get("method_name"),
            "method_formulation": (bundle.metadata or {}).get("method_formulation"),
            "recommended_switch_margin": (bundle.metadata or {}).get("recommended_switch_margin"),
            "recommended_positive_prob_threshold": (bundle.metadata or {}).get("recommended_positive_prob_threshold"),
            "dev_policy_eval": (bundle.metadata or {}).get("dev_policy_eval"),
            "dev_policy_eval_calibrated": (bundle.metadata or {}).get("dev_policy_eval_calibrated"),
            "family_policy_thresholds": (bundle.metadata or {}).get("family_policy_thresholds"),
        },
        "switch_margin": float(margin),
        "policies": list(bundle.policy_labels),
        "queries": int(len(rows)),
        "switched": int(switched),
        "policy_counts": dict(policy_counts),
        "policy_counts_by_family": {family: dict(counter) for family, counter in family_policy_counts.items()},
    }
    payload = {"metadata": meta, "methods": {row["method"]: {k: v for k, v in row.items() if k not in {"method", "label"}} for row in table_rows}}
    (args.out_dir / "paper_retrieval_metrics.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    (args.out_dir / "paper_retrieval_metrics_detail.json").write_text(json.dumps(details, ensure_ascii=False, indent=2), encoding="utf-8")
    with (args.out_dir / "rankings_debug.jsonl").open("w", encoding="utf-8") as f:
        for debug in debug_rows:
            f.write(json.dumps(debug, ensure_ascii=False) + "\n")
    write_csv(args.out_dir / "paper_retrieval_metrics.csv", table_rows, metric_keys)
    write_markdown(args.out_dir / "paper_retrieval_metrics.md", table_rows, metric_keys)
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    print(f"[OK] json -> {args.out_dir / 'paper_retrieval_metrics.json'}")
    print(f"[OK] csv  -> {args.out_dir / 'paper_retrieval_metrics.csv'}")
    print(f"[OK] md   -> {args.out_dir / 'paper_retrieval_metrics.md'}")
    print(f"[DONE] CEPS metrics -> {args.out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
