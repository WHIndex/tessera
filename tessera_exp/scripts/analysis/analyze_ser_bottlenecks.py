#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
import sys
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

try:
    from tessera_exp.e2e.objectives import infer_qa_target_type
except Exception:  # pragma: no cover
    infer_qa_target_type = None


METRICS = ["ndcg@1", "ndcg@5", "map@1", "map@5", "hits@1", "hits@5", "any_hit@5"]
TRACE_KEYS = [
    "tessera_ser_changed_count",
    "tessera_ser_topk_overlap",
    "tessera_ser_mean_score",
    "tessera_gee_expand_triggered",
    "tessera_gee_expand_graph_added",
    "tessera_gee_expand_output_candidates",
    "tessera_retrieval_agent_forced_hits",
    "tessera_retrieval_agent_coverage",
    "tessera_ser_evidence_set_cardinality_need",
    "tessera_ser_evidence_set_slot_coverage",
    "tessera_ser_evidence_set_family_count",
    "tessera_ser_evidence_set_anchor_hits",
]


def load_detail(path: Path, method: str) -> dict[str, dict[str, float]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    qids = [str(q) for q in data["query_ids"]]
    method_data = data["methods"][method]
    return {
        qid: {key: float(method_data[key][idx]) for key in method_data}
        for idx, qid in enumerate(qids)
    }


def load_debug(path: Path | None) -> dict[str, dict[str, Any]]:
    if path is None or not path.exists():
        return {}
    out: dict[str, dict[str, Any]] = {}
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            out[str(row.get("query_id"))] = row
    return out


def load_queries(path: Path | None) -> dict[str, str]:
    if path is None or not path.exists():
        return {}
    rows = json.loads(path.read_text(encoding="utf-8"))
    return {str(row.get("id")): str(row.get("query", "")) for row in rows}


def prefix(qid: str) -> str:
    return qid.split("_", 1)[0]


def rel_bin(rel_count: float) -> str:
    n = int(rel_count)
    if n <= 2:
        return "1-2"
    if n <= 5:
        return "3-5"
    if n <= 10:
        return "6-10"
    return "11+"


def target_type(query: str) -> str:
    if infer_qa_target_type is None:
        return "unknown"
    return str(infer_qa_target_type(query))


def mean(vals: list[float]) -> float:
    return float(sum(vals) / len(vals)) if vals else 0.0


def summarize_group(
    qids: list[str],
    baseline: dict[str, dict[str, float]],
    candidate: dict[str, dict[str, float]],
    key_fn,
) -> list[dict[str, Any]]:
    groups: dict[str, list[str]] = defaultdict(list)
    for qid in qids:
        groups[str(key_fn(qid))].append(qid)
    out: list[dict[str, Any]] = []
    for name, ids in sorted(groups.items(), key=lambda kv: kv[0]):
        row: dict[str, Any] = {"group": name, "n": len(ids)}
        for metric in METRICS:
            deltas = [candidate[qid][metric] - baseline[qid][metric] for qid in ids]
            row[f"{metric}_delta"] = mean(deltas)
            row[f"{metric}_improve"] = sum(1 for x in deltas if x > 1e-12)
            row[f"{metric}_hurt"] = sum(1 for x in deltas if x < -1e-12)
        out.append(row)
    return out


def trace_summary(qids: list[str], debug: dict[str, dict[str, Any]]) -> dict[str, float]:
    out: dict[str, float] = {}
    for key in TRACE_KEYS:
        vals = [
            float(debug[qid].get("trace", {}).get(key, 0.0))
            for qid in qids
            if qid in debug
        ]
        if vals:
            out[key] = mean(vals)
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description="Analyze SER/GEE retrieval bottlenecks from detail and debug files.")
    parser.add_argument("--baseline-name", default="baseline")
    parser.add_argument("--candidate-name", default="candidate")
    parser.add_argument("--baseline-detail", type=Path, required=True)
    parser.add_argument("--candidate-detail", type=Path, required=True)
    parser.add_argument("--baseline-debug", type=Path, default=None)
    parser.add_argument("--candidate-debug", type=Path, default=None)
    parser.add_argument("--split-file", type=Path, default=None)
    parser.add_argument("--method", default="tessera_rag")
    parser.add_argument("--primary-metric", default="hits@5", choices=METRICS)
    parser.add_argument("--top-n", type=int, default=20)
    parser.add_argument("--out-json", type=Path, required=True)
    parser.add_argument("--out-md", type=Path, required=True)
    args = parser.parse_args()

    baseline = load_detail(args.baseline_detail, args.method)
    candidate = load_detail(args.candidate_detail, args.method)
    common_qids = [qid for qid in candidate if qid in baseline]
    queries = load_queries(args.split_file)
    baseline_debug = load_debug(args.baseline_debug)
    candidate_debug = load_debug(args.candidate_debug)

    overall: dict[str, Any] = {"queries": len(common_qids)}
    for metric in METRICS:
        deltas = [candidate[qid][metric] - baseline[qid][metric] for qid in common_qids]
        overall[f"{metric}_baseline"] = mean([baseline[qid][metric] for qid in common_qids])
        overall[f"{metric}_candidate"] = mean([candidate[qid][metric] for qid in common_qids])
        overall[f"{metric}_delta"] = mean(deltas)
        overall[f"{metric}_improve"] = sum(1 for x in deltas if x > 1e-12)
        overall[f"{metric}_hurt"] = sum(1 for x in deltas if x < -1e-12)
        overall[f"{metric}_same"] = sum(1 for x in deltas if abs(x) <= 1e-12)

    primary = args.primary_metric
    hurt_qids = [qid for qid in common_qids if candidate[qid][primary] - baseline[qid][primary] < -1e-12]
    improve_qids = [qid for qid in common_qids if candidate[qid][primary] - baseline[qid][primary] > 1e-12]
    same_qids = [qid for qid in common_qids if abs(candidate[qid][primary] - baseline[qid][primary]) <= 1e-12]

    top_hurt = sorted(
        common_qids,
        key=lambda qid: candidate[qid][primary] - baseline[qid][primary],
    )[: max(1, int(args.top_n))]
    examples = []
    for qid in top_hurt:
        b_rank = baseline_debug.get(qid, {}).get("rankings", {}).get(args.method, [])[:10]
        c_rank = candidate_debug.get(qid, {}).get("rankings", {}).get(args.method, [])[:10]
        examples.append(
            {
                "query_id": qid,
                "query": queries.get(qid, ""),
                "rel_count": candidate[qid].get("rel_count", 0.0),
                f"{primary}_baseline": baseline[qid][primary],
                f"{primary}_candidate": candidate[qid][primary],
                f"{primary}_delta": candidate[qid][primary] - baseline[qid][primary],
                "baseline_top10": b_rank,
                "candidate_top10": c_rank,
                "candidate_trace": {
                    key: candidate_debug.get(qid, {}).get("trace", {}).get(key)
                    for key in TRACE_KEYS
                    if key in candidate_debug.get(qid, {}).get("trace", {})
                },
            }
        )

    report = {
        "baseline_name": args.baseline_name,
        "candidate_name": args.candidate_name,
        "primary_metric": primary,
        "overall": overall,
        "by_rel_count": summarize_group(common_qids, baseline, candidate, lambda qid: rel_bin(candidate[qid].get("rel_count", 0.0))),
        "by_dataset": summarize_group(common_qids, baseline, candidate, prefix),
        "by_target_type": summarize_group(common_qids, baseline, candidate, lambda qid: target_type(queries.get(qid, ""))),
        "trace": {
            "hurt": trace_summary(hurt_qids, candidate_debug),
            "improve": trace_summary(improve_qids, candidate_debug),
            "same": trace_summary(same_qids, candidate_debug),
        },
        "top_hurt_examples": examples,
    }

    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_md.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    lines = [
        f"# {args.candidate_name} vs {args.baseline_name}",
        "",
        f"Primary metric: `{primary}`",
        "",
        "## Overall",
        "",
        "| Metric | Baseline | Candidate | Delta | Improve | Hurt | Same |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for metric in METRICS:
        lines.append(
            f"| {metric} | {overall[f'{metric}_baseline']:.6f} | {overall[f'{metric}_candidate']:.6f} | "
            f"{overall[f'{metric}_delta']:+.6f} | {overall[f'{metric}_improve']} | "
            f"{overall[f'{metric}_hurt']} | {overall[f'{metric}_same']} |"
        )
    for title, key in [("By Rel Count", "by_rel_count"), ("By Dataset", "by_dataset"), ("By Target Type", "by_target_type")]:
        lines.extend(["", f"## {title}", "", f"| Group | N | {primary} Delta | NDCG@5 Delta | MAP@5 Delta | Hurt | Improve |", "|---|---:|---:|---:|---:|---:|---:|"])
        for row in report[key]:
            lines.append(
                f"| {row['group']} | {row['n']} | {row[f'{primary}_delta']:+.6f} | "
                f"{row['ndcg@5_delta']:+.6f} | {row['map@5_delta']:+.6f} | "
                f"{row[f'{primary}_hurt']} | {row[f'{primary}_improve']} |"
            )
    lines.extend(["", "## Trace Means", ""])
    for group, vals in report["trace"].items():
        lines.append(f"### {group}")
        for key, value in vals.items():
            lines.append(f"- `{key}`: {value:.6f}")
        lines.append("")
    lines.extend(["## Top Hurt Examples", ""])
    for ex in examples[: min(10, len(examples))]:
        lines.append(f"- `{ex['query_id']}` delta={ex[f'{primary}_delta']:+.6f} rel={ex['rel_count']}: {ex['query']}")
    args.out_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"[OK] json -> {args.out_json}")
    print(f"[OK] md   -> {args.out_md}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
