#!/usr/bin/env python3
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
import math
from pathlib import Path
import sys
from typing import Any, Sequence

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from tessera_exp.e2e.counterfactual_policy_selector import (  # noqa: E402
    POLICY_LABELS,
    candidate_rankings,
    query_family,
    retrieval_utility,
)
from tessera_exp.e2e.source_action_policy import source_bucket  # noqa: E402


def iter_jsonl(path: Path, max_examples: int = 0) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            rows.append(json.loads(line))
            if int(max_examples) > 0 and len(rows) >= int(max_examples):
                break
    return rows


def qrels_for_row(row: dict[str, Any]) -> dict[str, float]:
    out: dict[str, float] = {}
    for doc_id, raw in (row.get("qrels", {}) or {}).items():
        try:
            val = float(raw)
        except Exception:
            continue
        if val > 0.0:
            out[str(doc_id)] = val
    return out


def ranking_for_row(row: dict[str, Any], method: str) -> list[str]:
    rankings = row.get("rankings", {}) or {}
    return [str(x) for x in rankings.get(method, [])]


def dcg(grades: Sequence[float]) -> float:
    return float(sum((2.0 ** float(grade) - 1.0) / math.log2(rank + 2.0) for rank, grade in enumerate(grades)))


def metrics_for_ranking(ranking: Sequence[str], qrels: dict[str, float], ks: Sequence[int]) -> dict[str, float]:
    out: dict[str, float] = {}
    rel = {doc_id for doc_id, grade in qrels.items() if float(grade) > 0.0}
    for k in ks:
        top = list(ranking)[: int(k)]
        grades = [float(qrels.get(doc_id, 0.0)) for doc_id in top]
        ideal = sorted([float(v) for v in qrels.values() if float(v) > 0.0], reverse=True)[: int(k)]
        idcg = dcg(ideal)
        hit_count = 0
        ap_sum = 0.0
        for rank, doc_id in enumerate(top, start=1):
            if doc_id in rel:
                hit_count += 1
                ap_sum += hit_count / rank
        out[f"ndcg@{k}"] = float(dcg(grades) / idcg) if idcg > 0.0 else 0.0
        out[f"map@{k}"] = float(ap_sum / len(rel)) if rel else 0.0
        out[f"hits@{k}"] = float(hit_count)
        out[f"any_hit@{k}"] = float(hit_count > 0)
        out[f"rel_coverage@{k}"] = float(hit_count / max(1, len(rel)))
    return out


def average_metric(rows: list[dict[str, float]]) -> dict[str, float]:
    if not rows:
        return {}
    keys = sorted({key for row in rows for key in row})
    return {key: float(np.mean([row.get(key, 0.0) for row in rows])) for key in keys}


def oracle_reorder_within_pool(ranking: Sequence[str], qrels: dict[str, float], pool_k: int) -> list[str]:
    pool = list(ranking)[: int(pool_k)]
    tail = list(ranking)[int(pool_k) :]
    return sorted(pool, key=lambda doc_id: (float(qrels.get(doc_id, 0.0)), -pool.index(doc_id)), reverse=True) + tail


def parse_compare(raw_items: Sequence[str]) -> list[tuple[str, Path]]:
    out: list[tuple[str, Path]] = []
    for raw in raw_items:
        if "=" not in raw:
            raise ValueError(f"--compare expects label=path, got: {raw}")
        label, path = raw.split("=", 1)
        out.append((label.strip(), Path(path.strip())))
    return out


def load_rows_by_query_id(path: Path, method: str, max_examples: int = 0) -> dict[str, dict[str, Any]]:
    rows = iter_jsonl(path, max_examples=max_examples)
    out: dict[str, dict[str, Any]] = {}
    for idx, row in enumerate(rows):
        query_id = str(row.get("query_id") or f"idx_{idx}")
        if method not in (row.get("rankings", {}) or {}):
            continue
        out[query_id] = row
    return out


def summarize_deltas(
    base_rows: list[dict[str, Any]],
    *,
    method: str,
    ks: Sequence[int],
    compares: list[tuple[str, Path]],
    max_examples: int,
) -> dict[str, Any]:
    base_by_id = {str(row.get("query_id") or f"idx_{i}"): row for i, row in enumerate(base_rows)}
    out: dict[str, Any] = {}
    for label, path in compares:
        other_by_id = load_rows_by_query_id(path, method=method, max_examples=max_examples)
        metric_deltas: list[dict[str, float]] = []
        improved = Counter()
        worsened = Counter()
        tied = Counter()
        changed_top5 = 0
        overlap_top5: list[float] = []
        for query_id, base_row in base_by_id.items():
            other_row = other_by_id.get(query_id)
            if other_row is None:
                continue
            qrels = qrels_for_row(base_row)
            base_rank = ranking_for_row(base_row, method)
            other_rank = ranking_for_row(other_row, method)
            if not qrels or not base_rank or not other_rank:
                continue
            base_metrics = metrics_for_ranking(base_rank, qrels, ks)
            other_metrics = metrics_for_ranking(other_rank, qrels, ks)
            delta = {key: float(other_metrics.get(key, 0.0) - base_metrics.get(key, 0.0)) for key in base_metrics}
            metric_deltas.append(delta)
            for key, val in delta.items():
                if val > 1e-12:
                    improved[key] += 1
                elif val < -1e-12:
                    worsened[key] += 1
                else:
                    tied[key] += 1
            if list(base_rank[:5]) != list(other_rank[:5]):
                changed_top5 += 1
            union = set(base_rank[:5]) | set(other_rank[:5])
            overlap_top5.append(float(len(set(base_rank[:5]) & set(other_rank[:5])) / max(1, len(union))))
        out[label] = {
            "path": str(path),
            "matched_queries": int(len(metric_deltas)),
            "mean_delta": average_metric(metric_deltas),
            "improved_counts": dict(improved),
            "worsened_counts": dict(worsened),
            "tied_counts": dict(tied),
            "changed_top5": int(changed_top5),
            "changed_top5_rate": float(changed_top5 / max(1, len(metric_deltas))),
            "mean_top5_jaccard": float(np.mean(overlap_top5)) if overlap_top5 else 0.0,
        }
    return out


def policy_oracle(
    row: dict[str, Any],
    *,
    method: str,
    policy_labels: Sequence[str],
    topk: int,
    pool_k: int,
    utility_weights: dict[str, float],
) -> dict[str, Any]:
    ranking = ranking_for_row(row, method)
    qrels = qrels_for_row(row)
    if not ranking or not qrels:
        return {}
    candidates = candidate_rankings(
        ranking,
        query_id=str(row.get("query_id", "")),
        topk=int(topk),
        pool_k=int(pool_k),
        policy_labels=policy_labels,
    )
    utilities: dict[str, float] = {}
    metrics: dict[str, dict[str, float]] = {}
    for policy, candidate in candidates.items():
        utilities[policy] = retrieval_utility(
            candidate,
            qrels,
            topk=int(topk),
            ndcg1_weight=float(utility_weights["ndcg1"]),
            ndcg5_weight=float(utility_weights["ndcg5"]),
            map5_weight=float(utility_weights["map5"]),
            hits5_weight=float(utility_weights["hits5"]),
        )
        metrics[policy] = metrics_for_ranking(candidate, qrels, [1, topk])
    default = "keep_current"
    best_policy = max(policy_labels, key=lambda label: (utilities.get(label, 0.0), -list(policy_labels).index(label)))
    return {
        "best_policy": best_policy,
        "best_utility": float(utilities.get(best_policy, 0.0)),
        "default_utility": float(utilities.get(default, 0.0)),
        "gain": float(utilities.get(best_policy, 0.0) - utilities.get(default, 0.0)),
        "best_metrics": metrics.get(best_policy, {}),
        "default_metrics": metrics.get(default, {}),
    }


def diagnose(
    rows: list[dict[str, Any]],
    *,
    method: str,
    ks: Sequence[int],
    pool_depths: Sequence[int],
    enable_policy_oracle: bool,
    policy_labels: Sequence[str],
    policy_pool_k: int,
) -> dict[str, Any]:
    actual_metrics: list[dict[str, float]] = []
    oracle_by_pool: dict[int, list[dict[str, float]]] = {int(pool): [] for pool in pool_depths}
    per_family_metrics: dict[str, list[dict[str, float]]] = defaultdict(list)
    per_family_counts: Counter[str] = Counter()
    per_family_error_modes: dict[str, Counter[str]] = defaultdict(Counter)
    source_rows: dict[str, Counter[str]] = defaultdict(Counter)
    top1_miss_but_top5_hit: list[dict[str, Any]] = []
    top5_miss_but_top10_hit: list[dict[str, Any]] = []
    no_relevant_top10: list[dict[str, Any]] = []
    underfilled_examples: list[dict[str, Any]] = []
    qrel_count_values: list[int] = []
    rank_len_values: list[int] = []

    policy_oracle_rows: list[dict[str, float]] = []
    policy_oracle_counts: Counter[str] = Counter()
    policy_oracle_positive = 0
    policy_oracle_examples: list[dict[str, Any]] = []

    for idx, row in enumerate(rows):
        qrels = qrels_for_row(row)
        ranking = ranking_for_row(row, method)
        if not qrels or not ranking:
            continue
        query_id = str(row.get("query_id") or f"idx_{idx}")
        family = query_family(query_id)
        rel = set(qrels)
        qrel_count_values.append(len(qrels))
        rank_len_values.append(len(ranking))
        metrics = metrics_for_ranking(ranking, qrels, ks)
        actual_metrics.append(metrics)
        per_family_metrics[family].append(metrics)
        per_family_counts[family] += 1
        for pool in pool_depths:
            oracle_rank = oracle_reorder_within_pool(ranking, qrels, int(pool))
            oracle_by_pool[int(pool)].append(metrics_for_ranking(oracle_rank, qrels, ks))

        top1_hit = bool(ranking and ranking[0] in rel)
        top5_hits = len(rel & set(ranking[:5]))
        top10_hits = len(rel & set(ranking[:10]))
        if not top1_hit and top5_hits > 0:
            per_family_error_modes[family]["top1_rerank_error"] += 1
            if len(top1_miss_but_top5_hit) < 20:
                top1_miss_but_top5_hit.append(
                    {
                        "query_id": query_id,
                        "family": family,
                        "query": row.get("query", ""),
                        "qrels_in_top5": [doc for doc in ranking[:5] if doc in rel],
                        "top5": ranking[:5],
                    }
                )
        if top5_hits == 0 and top10_hits > 0:
            per_family_error_modes[family]["top5_rerank_error"] += 1
            if len(top5_miss_but_top10_hit) < 20:
                top5_miss_but_top10_hit.append(
                    {
                        "query_id": query_id,
                        "family": family,
                        "query": row.get("query", ""),
                        "qrels_in_top10": [doc for doc in ranking[:10] if doc in rel],
                        "top10": ranking[:10],
                    }
                )
        if top10_hits == 0:
            per_family_error_modes[family]["candidate_recall_error_top10"] += 1
            if len(no_relevant_top10) < 20:
                no_relevant_top10.append(
                    {
                        "query_id": query_id,
                        "family": family,
                        "query": row.get("query", ""),
                        "top10_sources": [source_bucket(doc) for doc in ranking[:10]],
                    }
                )
        ideal_hits5 = min(5, len(qrels))
        if top5_hits < ideal_hits5:
            per_family_error_modes[family]["top5_underfilled"] += 1
            if len(underfilled_examples) < 20:
                underfilled_examples.append(
                    {
                        "query_id": query_id,
                        "family": family,
                        "query": row.get("query", ""),
                        "hits5": top5_hits,
                        "ideal_hits5": ideal_hits5,
                        "hits10": top10_hits,
                        "qrels": list(qrels)[:12],
                        "top10": ranking[:10],
                    }
                )

        for doc_id in qrels:
            source_rows[family][f"qrel_{source_bucket(doc_id)}"] += 1
        for doc_id in ranking[:5]:
            source_rows[family][f"top5_{source_bucket(doc_id)}"] += 1
            if doc_id in rel:
                source_rows[family][f"hit5_{source_bucket(doc_id)}"] += 1

        if enable_policy_oracle:
            oracle = policy_oracle(
                row,
                method=method,
                policy_labels=policy_labels,
                topk=5,
                pool_k=int(policy_pool_k),
                utility_weights={"ndcg1": 0.30, "ndcg5": 0.30, "map5": 0.28, "hits5": 0.12},
            )
            if oracle:
                policy_oracle_counts[str(oracle["best_policy"])] += 1
                if float(oracle["gain"]) > 1e-12:
                    policy_oracle_positive += 1
                    if len(policy_oracle_examples) < 25:
                        policy_oracle_examples.append(
                            {
                                "query_id": query_id,
                                "family": family,
                                "query": row.get("query", ""),
                                "best_policy": oracle["best_policy"],
                                "gain": oracle["gain"],
                                "default_metrics": oracle["default_metrics"],
                                "best_metrics": oracle["best_metrics"],
                            }
                        )
                delta = {
                    f"oracle_delta_{key}": float(oracle["best_metrics"].get(key, 0.0) - oracle["default_metrics"].get(key, 0.0))
                    for key in oracle["default_metrics"]
                }
                delta["oracle_utility_gain"] = float(oracle["gain"])
                policy_oracle_rows.append(delta)

    actual = average_metric(actual_metrics)
    pool_oracle_summary = {
        str(pool): {
            "metrics": average_metric(items),
            "delta_vs_actual": {
                key: float(average_metric(items).get(key, 0.0) - actual.get(key, 0.0))
                for key in actual
                if key in average_metric(items)
            },
        }
        for pool, items in oracle_by_pool.items()
    }
    families = {}
    for family, items in sorted(per_family_metrics.items()):
        fam_metrics = average_metric(items)
        families[family] = {
            "queries": int(per_family_counts[family]),
            "metrics": fam_metrics,
            "error_modes": dict(per_family_error_modes[family]),
            "source_counts": dict(source_rows[family]),
        }

    root_cause = {
        "top1_rerank_error_rate": float(sum(c["top1_rerank_error"] for c in per_family_error_modes.values()) / max(1, len(actual_metrics))),
        "top5_candidate_recall_error_top10_rate": float(sum(c["candidate_recall_error_top10"] for c in per_family_error_modes.values()) / max(1, len(actual_metrics))),
        "top5_underfilled_rate": float(sum(c["top5_underfilled"] for c in per_family_error_modes.values()) / max(1, len(actual_metrics))),
        "policy_oracle_positive_rate": float(policy_oracle_positive / max(1, len(policy_oracle_rows))) if enable_policy_oracle else 0.0,
    }
    root_cause["interpretation"] = []
    if pool_oracle_summary.get("10", {}).get("delta_vs_actual", {}).get("hits@5", 0.0) < 0.15:
        root_cause["interpretation"].append("Top-10 内可重排带来的 Hits@5 空间很小，主要不是简单 rerank 能解决。")
    else:
        root_cause["interpretation"].append("Top-10 内仍有可重排空间，reranker/selector 仍可能提升。")
    if root_cause["top1_rerank_error_rate"] > 0.15:
        root_cause["interpretation"].append("大量 query 的相关证据已在 top5 但不在 top1，NDCG@1/Hits@1 受 head rerank 限制。")
    if root_cause["top5_candidate_recall_error_top10_rate"] > 0.05:
        root_cause["interpretation"].append("不少 query 在 top10 都没有相关证据，存在候选召回或源路由瓶颈。")
    if enable_policy_oracle and root_cause["policy_oracle_positive_rate"] > 0.15:
        root_cause["interpretation"].append("Policy oracle 有明显空间，但 selector 没有稳定吃到，属于策略选择学习瓶颈。")

    return {
        "method": method,
        "queries": int(len(actual_metrics)),
        "avg_qrels": float(np.mean(qrel_count_values)) if qrel_count_values else 0.0,
        "avg_ranking_length": float(np.mean(rank_len_values)) if rank_len_values else 0.0,
        "actual_metrics": actual,
        "pool_oracle": pool_oracle_summary,
        "families": families,
        "root_cause": root_cause,
        "policy_oracle": {
            "enabled": bool(enable_policy_oracle),
            "positive_queries": int(policy_oracle_positive),
            "positive_rate": root_cause["policy_oracle_positive_rate"],
            "best_policy_counts": dict(policy_oracle_counts),
            "mean_oracle_delta": average_metric(policy_oracle_rows),
            "examples": policy_oracle_examples,
        },
        "examples": {
            "top1_miss_but_top5_hit": top1_miss_but_top5_hit,
            "top5_miss_but_top10_hit": top5_miss_but_top10_hit,
            "no_relevant_top10": no_relevant_top10,
            "top5_underfilled": underfilled_examples,
        },
    }


def fmt(value: float) -> str:
    return f"{value:.6f}"


def write_markdown(report: dict[str, Any], path: Path) -> None:
    lines: list[str] = []
    lines.append("# TESSERA Retrieval Ceiling Diagnosis")
    lines.append("")
    lines.append(f"- Method: `{report['method']}`")
    lines.append(f"- Queries: {report['queries']}")
    lines.append(f"- Avg qrels/query: {report['avg_qrels']:.3f}")
    lines.append(f"- Avg ranking length: {report['avg_ranking_length']:.1f}")
    lines.append("")
    lines.append("## Current Metrics")
    metric_keys = ["ndcg@1", "ndcg@5", "map@1", "map@5", "hits@1", "hits@5", "any_hit@5", "rel_coverage@5"]
    lines.append("| metric | value |")
    lines.append("|---|---:|")
    for key in metric_keys:
        if key in report["actual_metrics"]:
            lines.append(f"| {key} | {fmt(report['actual_metrics'][key])} |")
    lines.append("")
    lines.append("## Pool Oracle")
    lines.append("| pool | NDCG@1 Δ | NDCG@5 Δ | MAP@5 Δ | Hits@5 Δ |")
    lines.append("|---:|---:|---:|---:|---:|")
    for pool, payload in sorted(report["pool_oracle"].items(), key=lambda item: int(item[0])):
        delta = payload["delta_vs_actual"]
        lines.append(
            f"| {pool} | {fmt(delta.get('ndcg@1', 0.0))} | {fmt(delta.get('ndcg@5', 0.0))} | "
            f"{fmt(delta.get('map@5', 0.0))} | {fmt(delta.get('hits@5', 0.0))} |"
        )
    lines.append("")
    lines.append("## Root Cause Signals")
    rc = report["root_cause"]
    lines.append(f"- Top1 rerank-error rate: {100.0 * rc['top1_rerank_error_rate']:.2f}%")
    lines.append(f"- Top10 candidate-recall error rate: {100.0 * rc['top5_candidate_recall_error_top10_rate']:.2f}%")
    lines.append(f"- Top5 underfilled rate: {100.0 * rc['top5_underfilled_rate']:.2f}%")
    if report["policy_oracle"]["enabled"]:
        lines.append(f"- Policy-oracle positive rate: {100.0 * rc['policy_oracle_positive_rate']:.2f}%")
    for item in rc.get("interpretation", []):
        lines.append(f"- {item}")
    lines.append("")
    lines.append("## Family Breakdown")
    lines.append("| family | queries | NDCG@1 | NDCG@5 | Hits@5 | AnyHit@5 | top1 rerank err | top10 recall err |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|")
    for family, payload in sorted(report["families"].items()):
        metrics = payload["metrics"]
        errors = payload["error_modes"]
        q = max(1, int(payload["queries"]))
        lines.append(
            f"| {family} | {payload['queries']} | {fmt(metrics.get('ndcg@1', 0.0))} | "
            f"{fmt(metrics.get('ndcg@5', 0.0))} | {fmt(metrics.get('hits@5', 0.0))} | "
            f"{fmt(metrics.get('any_hit@5', 0.0))} | "
            f"{100.0 * errors.get('top1_rerank_error', 0) / q:.2f}% | "
            f"{100.0 * errors.get('candidate_recall_error_top10', 0) / q:.2f}% |"
        )
    if report["policy_oracle"]["enabled"]:
        lines.append("")
        lines.append("## Policy Oracle")
        po = report["policy_oracle"]
        lines.append(f"- Positive queries: {po['positive_queries']} ({100.0 * po['positive_rate']:.2f}%)")
        lines.append(f"- Best-policy counts: `{json.dumps(po['best_policy_counts'], ensure_ascii=False)}`")
        lines.append(f"- Mean oracle utility gain: {fmt(po['mean_oracle_delta'].get('oracle_utility_gain', 0.0))}")
        for key in ["oracle_delta_ndcg@1", "oracle_delta_ndcg@5", "oracle_delta_map@5", "oracle_delta_hits@5"]:
            if key in po["mean_oracle_delta"]:
                lines.append(f"- {key}: {fmt(po['mean_oracle_delta'][key])}")
    lines.append("")
    lines.append("## Example Buckets")
    for bucket_name, examples in report["examples"].items():
        lines.append(f"### {bucket_name}")
        for ex in examples[:8]:
            lines.append(f"- `{ex.get('query_id')}` [{ex.get('family')}] {ex.get('query')}")
        lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Diagnose retrieval ceilings and error sources for TESSERA rankings.")
    parser.add_argument("--debug-jsonl", type=Path, required=True)
    parser.add_argument("--method", type=str, default="tessera_rag")
    parser.add_argument("--ks", type=str, default="1,5")
    parser.add_argument("--pool-depths", type=str, default="5,10")
    parser.add_argument("--compare", action="append", default=[], help="Optional label=rankings_debug.jsonl comparison.")
    parser.add_argument("--enable-policy-oracle", action="store_true")
    parser.add_argument("--policy-pool-k", type=int, default=30)
    parser.add_argument("--policy-labels", type=str, default=",".join(POLICY_LABELS))
    parser.add_argument("--max-examples", type=int, default=0)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()

    rows = iter_jsonl(args.debug_jsonl, int(args.max_examples))
    ks = sorted({int(x.strip()) for x in str(args.ks).split(",") if x.strip()})
    pool_depths = sorted({int(x.strip()) for x in str(args.pool_depths).split(",") if x.strip()})
    policy_labels = [x.strip() for x in str(args.policy_labels).split(",") if x.strip()]
    if "keep_current" not in policy_labels:
        policy_labels.insert(0, "keep_current")

    report = diagnose(
        rows,
        method=str(args.method),
        ks=ks,
        pool_depths=pool_depths,
        enable_policy_oracle=bool(args.enable_policy_oracle),
        policy_labels=policy_labels,
        policy_pool_k=int(args.policy_pool_k),
    )
    compares = parse_compare(args.compare)
    if compares:
        report["run_deltas"] = summarize_deltas(
            rows,
            method=str(args.method),
            ks=ks,
            compares=compares,
            max_examples=int(args.max_examples),
        )

    args.out_dir.mkdir(parents=True, exist_ok=True)
    json_path = args.out_dir / "retrieval_ceiling_diagnosis.json"
    md_path = args.out_dir / "retrieval_ceiling_diagnosis.md"
    json_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    write_markdown(report, md_path)
    print(json.dumps({"json": str(json_path), "markdown": str(md_path), "queries": report["queries"]}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
