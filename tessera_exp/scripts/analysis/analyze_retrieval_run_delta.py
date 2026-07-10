#!/usr/bin/env python3
from __future__ import annotations

import argparse
from collections import defaultdict
import json
import math
from pathlib import Path
from statistics import mean


TRACE_KEYS = [
    "tessera_ser_changed_count",
    "tessera_ser_topk_overlap",
    "tessera_source_evidence_changed_count",
    "tessera_source_evidence_topk_overlap",
    "tessera_source_evidence_budget_changed_count",
    "tessera_source_evidence_budget_tail_selected",
    "tessera_source_evidence_budget_sibling_selected",
    "tessera_source_evidence_budget_source_quota_selected",
    "tessera_source_evidence_budget_reference_selected",
    "tessera_source_head_top1_changed",
    "tessera_gee_expand_triggered",
    "tessera_gee_expand_graph_added",
]


def query_family(query_id: str) -> str:
    return str(query_id).split("_", 1)[0]


def source_bucket(doc_id: str) -> str:
    doc = str(doc_id)
    if doc.startswith(("nq_", "triviaqa_")):
        return "text"
    if doc.startswith(("ott_", "tat_")):
        return "table"
    return "kg"


def load_rankings(run_dir: Path) -> dict[str, dict]:
    path = run_dir / "rankings_debug.jsonl"
    out: dict[str, dict] = {}
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            ranking = row.get("rankings", {}).get("tessera_rag", [])
            out[str(row["query_id"])] = {
                "query": str(row.get("query", "")),
                "qrels": {str(k): float(v) for k, v in dict(row.get("qrels", {})).items()},
                "ranking": [str(x) for x in ranking],
                "trace": dict(row.get("trace", {})),
            }
    return out


def retrieval_metrics(qrels: dict[str, float], ranking: list[str], k: int) -> dict[str, float]:
    rels = [float(qrels.get(doc_id, 0.0)) for doc_id in ranking[:k]]
    gains = [(2.0**rel - 1.0) if rel > 0 else 0.0 for rel in rels]
    dcg = sum(gain / math.log2(i + 2) for i, gain in enumerate(gains))
    ideal = sorted((float(v) for v in qrels.values() if float(v) > 0), reverse=True)[:k]
    idcg = sum((2.0**rel - 1.0) / math.log2(i + 2) for i, rel in enumerate(ideal))
    hits = sum(1.0 for rel in rels if rel > 0)
    found = 0.0
    ap = 0.0
    for i, rel in enumerate(rels, 1):
        if rel > 0:
            found += 1.0
            ap += found / float(i)
    total_positive = max(1.0, sum(1.0 for v in qrels.values() if float(v) > 0))
    return {
        "ndcg": dcg / idcg if idcg > 0 else 0.0,
        "map": ap / total_positive,
        "hits": hits,
        "any_hit": 1.0 if hits > 0 else 0.0,
    }


def avg(values: list[float]) -> float:
    return float(mean(values)) if values else 0.0


def summarize_run(name: str, rows: dict[str, dict]) -> list[str]:
    buckets: dict[str, list[tuple[dict[str, float], dict[str, float], dict]]] = defaultdict(list)
    for query_id, row in rows.items():
        m1 = retrieval_metrics(row["qrels"], row["ranking"], 1)
        m5 = retrieval_metrics(row["qrels"], row["ranking"], 5)
        buckets[query_family(query_id)].append((m1, m5, row))

    lines = [f"RUN {name} queries={len(rows)}"]
    for family in sorted(buckets):
        vals = buckets[family]
        lines.append(
            f"  {family:9s} n={len(vals):4d} "
            f"ndcg1={avg([x[0]['ndcg'] for x in vals]):.6f} "
            f"ndcg5={avg([x[1]['ndcg'] for x in vals]):.6f} "
            f"map5={avg([x[1]['map'] for x in vals]):.6f} "
            f"hits5={avg([x[1]['hits'] for x in vals]):.6f} "
            f"any5={avg([x[1]['any_hit'] for x in vals]):.6f}"
        )
    return lines


def source_composition(rows: dict[str, dict], k: int = 5) -> dict[str, float]:
    counts: dict[str, float] = defaultdict(float)
    total = 0.0
    for row in rows.values():
        for doc_id in row["ranking"][:k]:
            counts[source_bucket(doc_id)] += 1.0
            total += 1.0
    return {key: counts[key] / total for key in sorted(counts)} if total else {}


def trace_averages(rows: dict[str, dict]) -> dict[str, float]:
    buckets: dict[str, list[float]] = defaultdict(list)
    for row in rows.values():
        trace = row.get("trace", {})
        for key in TRACE_KEYS:
            value = trace.get(key)
            if isinstance(value, (int, float)):
                buckets[key].append(float(value))
    return {key: avg(values) for key, values in buckets.items()}


def compare_runs(base_name: str, base_rows: dict[str, dict], cand_name: str, cand_rows: dict[str, dict]) -> list[str]:
    common = sorted(set(base_rows) & set(cand_rows))
    buckets: dict[str, list[tuple[float, float, float, float, float, str]]] = defaultdict(list)
    regressions: list[tuple[float, float, str, dict, dict]] = []
    for query_id in common:
        base = base_rows[query_id]
        cand = cand_rows[query_id]
        base1 = retrieval_metrics(base["qrels"], base["ranking"], 1)
        base5 = retrieval_metrics(base["qrels"], base["ranking"], 5)
        cand1 = retrieval_metrics(cand["qrels"], cand["ranking"], 1)
        cand5 = retrieval_metrics(cand["qrels"], cand["ranking"], 5)
        delta = (
            cand1["ndcg"] - base1["ndcg"],
            cand5["ndcg"] - base5["ndcg"],
            cand5["map"] - base5["map"],
            cand5["hits"] - base5["hits"],
            cand5["any_hit"] - base5["any_hit"],
            query_id,
        )
        buckets[query_family(query_id)].append(delta)
        regressions.append((delta[1], delta[3], query_id, base, cand))

    lines = [f"DELTA {cand_name} - {base_name} common={len(common)}"]
    for family in sorted(buckets):
        vals = buckets[family]
        better = sum(1 for x in vals if x[1] > 1e-9)
        worse = sum(1 for x in vals if x[1] < -1e-9)
        lines.append(
            f"  {family:9s} n={len(vals):4d} "
            f"d_ndcg1={avg([x[0] for x in vals]):+.6f} "
            f"d_ndcg5={avg([x[1] for x in vals]):+.6f} "
            f"d_map5={avg([x[2] for x in vals]):+.6f} "
            f"d_hits5={avg([x[3] for x in vals]):+.6f} "
            f"d_any5={avg([x[4] for x in vals]):+.6f} "
            f"better5={better} worse5={worse}"
        )

    lines.append("SOURCE_TOP5")
    lines.append(f"  {base_name}: {json.dumps(source_composition(base_rows), sort_keys=True)}")
    lines.append(f"  {cand_name}: {json.dumps(source_composition(cand_rows), sort_keys=True)}")

    lines.append("TRACE_AVG")
    for key, value in sorted(trace_averages(cand_rows).items()):
        lines.append(f"  {cand_name}.{key}={value:.6f}")

    lines.append("WORST_NDCG5_REGRESSIONS")
    for d_ndcg5, d_hits5, query_id, base, cand in sorted(regressions)[: int(8)]:
        lines.append(
            f"  {query_id} d_ndcg5={d_ndcg5:+.6f} d_hits5={d_hits5:+.1f} query={base['query'][:120]}"
        )
        lines.append(f"    qrels={json.dumps(base['qrels'], sort_keys=True)}")
        lines.append(f"    {base_name}_top5={base['ranking'][:5]}")
        lines.append(f"    {cand_name}_top5={cand['ranking'][:5]}")
    return lines


def main() -> int:
    parser = argparse.ArgumentParser(description="Analyze query-level retrieval deltas between two TESSERA runs.")
    parser.add_argument("--base-run", type=Path, required=True)
    parser.add_argument("--candidate-run", type=Path, required=True)
    parser.add_argument("--base-name", default="base")
    parser.add_argument("--candidate-name", default="candidate")
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    base_rows = load_rankings(args.base_run)
    candidate_rows = load_rankings(args.candidate_run)
    lines: list[str] = []
    lines.extend(summarize_run(args.base_name, base_rows))
    lines.append("")
    lines.extend(summarize_run(args.candidate_name, candidate_rows))
    lines.append("")
    lines.extend(compare_runs(args.base_name, base_rows, args.candidate_name, candidate_rows))
    text = "\n".join(lines) + "\n"
    print(text)
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text, encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
