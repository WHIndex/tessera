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

from unifusion_exp.e2e.counterfactual_policy_selector import (  # noqa: E402
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


def selector_meta(bundle: object) -> dict:
    metadata = getattr(bundle, "metadata", {}) or {}
    return {
        "method_name": metadata.get("method_name"),
        "method_formulation": metadata.get("method_formulation"),
        "score_mode": metadata.get("score_mode"),
        "model_type": metadata.get("model_type"),
        "risk_model_type": metadata.get("risk_model_type"),
        "recommended_switch_margin": metadata.get("recommended_switch_margin"),
        "recommended_positive_prob_threshold": metadata.get("recommended_positive_prob_threshold"),
        "dev_policy_eval": metadata.get("dev_policy_eval"),
        "dev_policy_eval_calibrated": metadata.get("dev_policy_eval_calibrated"),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Apply a safe primary CEPS selector plus an additive CEPS selector.")
    parser.add_argument("--base-rankings-jsonl", type=Path, required=True)
    parser.add_argument("--primary-selector-model", type=Path, required=True)
    parser.add_argument("--addon-selector-model", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--method", type=str, default="unifusion_rag")
    parser.add_argument("--label", type=str, default="UniFusion-Cascade-CEPS")
    parser.add_argument("--metrics-k", type=str, default="1,5")
    parser.add_argument("--primary-switch-margin", type=float, default=None)
    parser.add_argument("--addon-switch-margin", type=float, default=None)
    parser.add_argument("--max-queries", type=int, default=0)
    parser.add_argument("--include-base", action="store_true")
    parser.add_argument("--include-primary", action="store_true")
    args = parser.parse_args()

    if not args.base_rankings_jsonl.exists():
        raise FileNotFoundError(f"base-rankings-jsonl not found: {args.base_rankings_jsonl}")
    if not args.primary_selector_model.exists():
        raise FileNotFoundError(f"primary-selector-model not found: {args.primary_selector_model}")
    if not args.addon_selector_model.exists():
        raise FileNotFoundError(f"addon-selector-model not found: {args.addon_selector_model}")

    ks = sorted({int(x.strip()) for x in str(args.metrics_k).split(",") if x.strip()})
    if not ks:
        raise ValueError("metrics-k is empty")

    primary = load_counterfactual_policy_selector_bundle(args.primary_selector_model)
    addon = load_counterfactual_policy_selector_bundle(args.addon_selector_model)
    primary_margin = float(primary.switch_margin if args.primary_switch_margin is None else args.primary_switch_margin)
    addon_margin = float(addon.switch_margin if args.addon_switch_margin is None else args.addon_switch_margin)

    rows = iter_jsonl(args.base_rankings_jsonl, int(args.max_queries))
    preds: list[list[str]] = []
    base_preds: list[list[str]] = []
    primary_preds: list[list[str]] = []
    debug_rows: list[dict] = []
    primary_policy_counts: Counter[str] = Counter()
    addon_policy_counts: Counter[str] = Counter()
    final_policy_counts: Counter[str] = Counter()
    family_final_policy_counts: dict[str, Counter[str]] = defaultdict(Counter)
    primary_switched = 0
    addon_switched = 0
    final_switched = 0

    for i, row in enumerate(rows):
        query_text = str(row.get("query", ""))
        query_id = str(row.get("query_id", ""))
        trace = row.get("trace", {})
        base = ranking_for_row(row, str(args.method))
        base_preds.append(base)

        primary_ranked, primary_selection = primary.select(
            query_text=query_text,
            query_id=query_id,
            base_ranked_doc_ids=base,
            trace=trace,
            switch_margin=primary_margin,
        )
        addon_ranked, addon_selection = addon.select(
            query_text=query_text,
            query_id=query_id,
            base_ranked_doc_ids=primary_ranked,
            trace=trace,
            switch_margin=addon_margin,
        )

        final_ranked = addon_ranked
        primary_preds.append(primary_ranked)
        preds.append(final_ranked)

        primary_policy_counts[primary_selection.policy] += 1
        addon_policy_counts[addon_selection.policy] += 1
        primary_switched += int(primary_selection.switched)
        addon_switched += int(addon_selection.switched)
        final_changed = bool(primary_selection.switched or addon_selection.switched)
        final_switched += int(final_changed)
        if addon_selection.switched:
            final_policy = f"{primary_selection.policy}+{addon_selection.policy}"
        else:
            final_policy = primary_selection.policy
        final_policy_counts[final_policy] += 1
        family = query_family(query_id)
        family_final_policy_counts[family][final_policy] += 1

        debug_rows.append(
            {
                "query_index": i,
                "query_id": query_id,
                "query": query_text,
                "qrels": qrels_for_row(row),
                "rankings": {
                    str(args.method): final_ranked,
                    "_base_unifusion": base,
                    "_primary_unifusion": primary_ranked,
                },
                "trace": {
                    "unifusion_cascade_active": 1.0,
                    "unifusion_cascade_primary_policy": primary_selection.policy,
                    "unifusion_cascade_primary_switched": float(primary_selection.switched),
                    "unifusion_cascade_primary_margin": float(primary_selection.margin),
                    "unifusion_cascade_primary_scores": primary_selection.policy_scores,
                    "unifusion_cascade_addon_policy": addon_selection.policy,
                    "unifusion_cascade_addon_switched": float(addon_selection.switched),
                    "unifusion_cascade_addon_margin": float(addon_selection.margin),
                    "unifusion_cascade_addon_scores": addon_selection.policy_scores,
                    "unifusion_cascade_addon_positive_probs": addon_selection.positive_probs,
                    "unifusion_cascade_final_policy": final_policy,
                    "unifusion_cascade_final_switched": float(final_changed),
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
    if bool(args.include_primary):
        primary_summary, primary_detail = method_metrics(rows, primary_preds, ks)
        table_rows.insert(0, {"method": "primary_unifusion", "label": "Primary CEPS", **primary_summary})
        details["methods"]["primary_unifusion"] = primary_detail
    if bool(args.include_base):
        base_summary, base_detail = method_metrics(rows, base_preds, ks)
        table_rows.insert(0, {"method": "base_unifusion", "label": "Base UniFusion", **base_summary})
        details["methods"]["base_unifusion"] = base_detail

    args.out_dir.mkdir(parents=True, exist_ok=True)
    meta = {
        "base_rankings_jsonl": str(args.base_rankings_jsonl),
        "primary_selector_model": str(args.primary_selector_model),
        "addon_selector_model": str(args.addon_selector_model),
        "primary_selector_meta_summary": selector_meta(primary),
        "addon_selector_meta_summary": selector_meta(addon),
        "primary_switch_margin": float(primary_margin),
        "addon_switch_margin": float(addon_margin),
        "queries": int(len(rows)),
        "primary_switched": int(primary_switched),
        "addon_switched": int(addon_switched),
        "switched": int(final_switched),
        "primary_policy_counts": dict(primary_policy_counts),
        "addon_policy_counts": dict(addon_policy_counts),
        "policy_counts": dict(final_policy_counts),
        "policy_counts_by_family": {family: dict(counter) for family, counter in family_final_policy_counts.items()},
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
    print(f"[DONE] Cascade CEPS metrics -> {args.out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
