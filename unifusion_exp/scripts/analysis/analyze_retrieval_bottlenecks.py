#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import mean
from typing import Any


def _as_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return default


def _trace_value(trace: dict[str, Any], key: str, default: float = 0.0) -> float:
    value = trace.get(key, default)
    if isinstance(value, list):
        if not value:
            return default
        value = value[0]
    return _as_float(value, default)


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def _relevant_set(row: dict[str, Any]) -> set[str]:
    qrels = row.get("qrels") or {}
    out: set[str] = set()
    if isinstance(qrels, dict):
        for doc_id, grade in qrels.items():
            if _as_float(grade) > 0:
                out.add(str(doc_id))
    return out


def _avg(values: list[float]) -> float:
    return float(mean(values)) if values else 0.0


def _pct(value: float) -> str:
    return f"{100.0 * value:.2f}%"


def analyze(rows: list[dict[str, Any]], method: str, ks: list[int]) -> dict[str, Any]:
    if not rows:
        raise ValueError("No rows found in rankings debug jsonl")
    max_k = max(ks)
    total = len(rows)

    hit_any = {k: [] for k in ks}
    hit_count = {k: [] for k in ks}
    rel_coverage = {k: [] for k in ks}
    missing_qrels = 0
    method_missing = 0

    pool_any: list[float] = []
    pool_hits: list[float] = []
    pool_cov: list[float] = []
    dense_added: list[float] = []
    sparse_added: list[float] = []
    graph_added: list[float] = []
    local_changed: list[float] = []
    local_slot_cov: list[float] = []

    candidate_miss_top = {k: 0 for k in ks}
    rerank_miss_top = {k: 0 for k in ks}
    solved_by_top = {k: 0 for k in ks}

    hard_examples: list[dict[str, Any]] = []
    rerank_examples: list[dict[str, Any]] = []

    for row in rows:
        rel = _relevant_set(row)
        if not rel:
            missing_qrels += 1
        rankings = row.get("rankings") or {}
        ranking = [str(x) for x in rankings.get(method, [])]
        if not ranking:
            method_missing += 1
        trace = row.get("trace") or {}
        cand_any = _trace_value(trace, "unifusion_v9_candidate_pool_any_hit", default=-1.0)
        cand_hits = _trace_value(trace, "unifusion_v9_candidate_pool_hits", default=0.0)
        cand_cov = _trace_value(trace, "unifusion_v9_candidate_pool_rel_coverage", default=0.0)
        if cand_any >= 0:
            pool_any.append(cand_any)
        pool_hits.append(cand_hits)
        pool_cov.append(cand_cov)
        dense_added.append(_trace_value(trace, "unifusion_v9_dense_added"))
        sparse_added.append(_trace_value(trace, "unifusion_v9_sparse_added"))
        graph_added.append(_trace_value(trace, "unifusion_v9_graph_added"))
        local_changed.append(_trace_value(trace, "unifusion_v9_local_changed_count"))
        local_slot_cov.append(_trace_value(trace, "unifusion_v9_local_slot_coverage"))

        top_hits_at_max = len(rel & set(ranking[:max_k])) if rel else 0
        for k in ks:
            top = set(ranking[:k])
            hits = len(rel & top) if rel else 0
            hit_any[k].append(float(hits > 0))
            hit_count[k].append(float(hits))
            rel_coverage[k].append(float(hits / max(1, len(rel))))
            if hits > 0:
                solved_by_top[k] += 1
            elif cand_any == 1.0:
                rerank_miss_top[k] += 1
            elif cand_any == 0.0:
                candidate_miss_top[k] += 1

        if cand_any == 0.0 and len(hard_examples) < 12:
            hard_examples.append(
                {
                    "query_id": row.get("query_id"),
                    "query": row.get("query"),
                    "candidate_pool_rel_coverage": cand_cov,
                    "top_hits": top_hits_at_max,
                }
            )
        if cand_any == 1.0 and top_hits_at_max == 0 and len(rerank_examples) < 12:
            rerank_examples.append(
                {
                    "query_id": row.get("query_id"),
                    "query": row.get("query"),
                    "candidate_pool_hits": cand_hits,
                    "candidate_pool_rel_coverage": cand_cov,
                    "topk": ranking[:max_k],
                }
            )

    summary: dict[str, Any] = {
        "method": method,
        "queries": total,
        "missing_qrels": missing_qrels,
        "method_missing": method_missing,
        "candidate_pool_any_hit": _avg(pool_any),
        "candidate_pool_hits": _avg(pool_hits),
        "candidate_pool_rel_coverage": _avg(pool_cov),
        "v9_dense_added_avg": _avg(dense_added),
        "v9_sparse_added_avg": _avg(sparse_added),
        "v9_graph_added_avg": _avg(graph_added),
        "v9_local_changed_avg": _avg(local_changed),
        "v9_local_slot_coverage_avg": _avg(local_slot_cov),
        "hard_candidate_miss_examples": hard_examples,
        "hard_rerank_miss_examples": rerank_examples,
    }
    for k in ks:
        summary[f"top{k}_any_hit"] = _avg(hit_any[k])
        summary[f"top{k}_hit_count"] = _avg(hit_count[k])
        summary[f"top{k}_rel_coverage"] = _avg(rel_coverage[k])
        summary[f"top{k}_candidate_miss_rate"] = float(candidate_miss_top[k] / max(1, total))
        summary[f"top{k}_rerank_miss_rate"] = float(rerank_miss_top[k] / max(1, total))
        summary[f"top{k}_solved_rate"] = float(solved_by_top[k] / max(1, total))
    return summary


def write_markdown(summary: dict[str, Any], out_path: Path) -> None:
    ks = sorted(
        int(k.removeprefix("top").removesuffix("_any_hit"))
        for k in summary
        if k.startswith("top") and k.endswith("_any_hit")
    )
    lines = [
        f"# Retrieval Bottleneck Analysis",
        "",
        f"- Method: `{summary['method']}`",
        f"- Queries: {summary['queries']}",
        f"- Candidate-pool any-hit: {_pct(summary['candidate_pool_any_hit'])}",
        f"- Candidate-pool relevance coverage: {summary['candidate_pool_rel_coverage']:.4f}",
        f"- Avg v9 additions: dense {summary['v9_dense_added_avg']:.1f}, sparse {summary['v9_sparse_added_avg']:.1f}, graph {summary['v9_graph_added_avg']:.1f}",
        f"- Avg local rerank changes: {summary['v9_local_changed_avg']:.2f}; slot coverage {summary['v9_local_slot_coverage_avg']:.4f}",
        "",
        "| k | Top-k any-hit | Top-k rel coverage | Candidate miss | Rerank miss |",
        "|---:|---:|---:|---:|---:|",
    ]
    for k in ks:
        lines.append(
            "| {k} | {any_hit} | {cov:.4f} | {cand_miss} | {rerank_miss} |".format(
                k=k,
                any_hit=_pct(summary[f"top{k}_any_hit"]),
                cov=summary[f"top{k}_rel_coverage"],
                cand_miss=_pct(summary[f"top{k}_candidate_miss_rate"]),
                rerank_miss=_pct(summary[f"top{k}_rerank_miss_rate"]),
            )
        )
    lines.append("")
    lines.append("## Candidate Miss Examples")
    for ex in summary.get("hard_candidate_miss_examples", []):
        lines.append(f"- `{ex.get('query_id')}` {ex.get('query')}")
    lines.append("")
    lines.append("## Rerank Miss Examples")
    for ex in summary.get("hard_rerank_miss_examples", []):
        lines.append(
            f"- `{ex.get('query_id')}` hits={ex.get('candidate_pool_hits')} "
            f"coverage={ex.get('candidate_pool_rel_coverage'):.4f} {ex.get('query')}"
        )
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--debug-jsonl", type=Path, required=True)
    parser.add_argument("--method", default="unifusion_rag")
    parser.add_argument("--ks", default="1,5")
    parser.add_argument("--out-json", type=Path, default=None)
    parser.add_argument("--out-md", type=Path, default=None)
    args = parser.parse_args()

    rows = _read_jsonl(args.debug_jsonl)
    ks = sorted({int(x.strip()) for x in args.ks.split(",") if x.strip()})
    summary = analyze(rows, args.method, ks)
    if args.out_json:
        args.out_json.parent.mkdir(parents=True, exist_ok=True)
        args.out_json.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    if args.out_md:
        args.out_md.parent.mkdir(parents=True, exist_ok=True)
        write_markdown(summary, args.out_md)
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
