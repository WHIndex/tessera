from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from statistics import mean
from typing import Any


DEFAULT_METRICS = ["NDCG@1", "NDCG@5", "MAP@1", "MAP@5", "Hits@1", "Hits@5", "AnyHit@5"]
TRACE_KEYS = [
    "tessera_v9_candidate_pool_any_hit",
    "tessera_v9_candidate_pool_rel_coverage",
    "tessera_ser_changed_count",
    "tessera_v10_conservative_rerank",
    "tessera_v10_changed_count",
    "tessera_v10_restored_from_reference",
    "tessera_v10_accepted_new",
    "tessera_v10_rejected_new",
    "tessera_v10_effective_margin",
    "tessera_v10_effective_relevance_floor",
    "tessera_v10_reference_topk_overlap_before",
    "tessera_v10_reference_topk_overlap_after",
]


def _load_metrics(run_dir: Path, method: str) -> dict[str, float]:
    csv_path = run_dir / "paper_retrieval_metrics.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"metrics csv not found: {csv_path}")
    with csv_path.open("r", encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            if row.get("method") == method:
                return {
                    key: float(row[key])
                    for key in row
                    if key not in {"method", "label"} and str(row.get(key, "")).strip() != ""
                }
    raise ValueError(f"method {method!r} not found in {csv_path}")


def _trace_values(payload: dict[str, Any], key: str) -> list[float]:
    trace = payload.get("trace", {})
    values = trace.get(key, [])
    if not isinstance(values, list):
        return []
    out: list[float] = []
    for value in values:
        try:
            out.append(float(value))
        except Exception:
            continue
    return out


def _load_trace_means(run_dir: Path) -> dict[str, float]:
    path = run_dir / "rankings_debug.jsonl"
    if not path.exists():
        return {}
    buckets: dict[str, list[float]] = {key: [] for key in TRACE_KEYS}
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError:
                continue
            for key in TRACE_KEYS:
                buckets[key].extend(_trace_values(payload, key))
    return {f"{key}_avg": mean(values) for key, values in buckets.items() if values}


def _format_delta(value: float, base: float | None) -> str:
    if base is None:
        return ""
    delta = value - base
    return f"{delta:+.6f}"


def main() -> int:
    parser = argparse.ArgumentParser(description="Compare TESSERA retrieval run directories.")
    parser.add_argument("run_dirs", nargs="+", type=Path)
    parser.add_argument("--method", default="tessera_rag")
    parser.add_argument("--metrics", default=",".join(DEFAULT_METRICS))
    parser.add_argument("--out-md", type=Path, default=None)
    parser.add_argument("--out-json", type=Path, default=None)
    args = parser.parse_args()

    metric_keys = [x.strip() for x in str(args.metrics).split(",") if x.strip()]
    rows: list[dict[str, Any]] = []
    baseline: dict[str, float] | None = None
    for idx, run_dir in enumerate(args.run_dirs):
        run_dir = run_dir.resolve()
        metrics = _load_metrics(run_dir, method=str(args.method))
        trace = _load_trace_means(run_dir)
        if idx == 0:
            baseline = metrics
        row: dict[str, Any] = {"run": run_dir.name, "path": str(run_dir)}
        for key in metric_keys:
            if key in metrics:
                row[key] = metrics[key]
                row[f"{key}_delta"] = _format_delta(metrics[key], baseline.get(key) if baseline else None)
        row.update(trace)
        rows.append(row)

    headers = ["run"]
    for key in metric_keys:
        headers.extend([key, f"{key}_delta"])
    trace_headers = sorted({key for row in rows for key in row if key.endswith("_avg")})
    headers.extend(trace_headers)

    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for row in rows:
        vals = []
        for key in headers:
            value = row.get(key, "")
            if isinstance(value, float):
                vals.append(f"{value:.6f}")
            else:
                vals.append(str(value))
        lines.append("| " + " | ".join(vals) + " |")
    md = "\n".join(lines) + "\n"
    print(md)

    if args.out_md is not None:
        args.out_md.parent.mkdir(parents=True, exist_ok=True)
        args.out_md.write_text(md, encoding="utf-8")
    if args.out_json is not None:
        args.out_json.parent.mkdir(parents=True, exist_ok=True)
        args.out_json.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
