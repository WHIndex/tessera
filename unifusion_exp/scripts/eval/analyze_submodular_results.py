#!/usr/bin/env python3
"""
Analyze and compare submodular packing results against baselines.
Usage:
    python analyze_submodular_results.py <result_dir>
"""
from __future__ import annotations

import json
import sys
from pathlib import Path


def load_metrics(result_dir: str) -> dict[str, dict[str, float]] | None:
    p = Path(result_dir) / "table1c_e2e_metrics.json"
    if not p.exists():
        return None
    with open(p) as f:
        data = json.load(f)
    out: dict[str, dict[str, float]] = {}
    for row in data.get("rows", []):
        method = row.get("method", "unknown")
        out[method] = {
            "f1": float(row.get("f1", 0.0)),
            "exact_match": float(row.get("exact_match", 0.0)),
            "recall_at_10": float(row.get("recall_at_10", 0.0)),
            "p95_latency_ms": float(row.get("p95_latency_ms", 0.0)),
        }
    return out


def compare(metrics: dict[str, dict[str, float]]) -> None:
    baseline = metrics.get("dense_concat")
    unifusion = metrics.get("unifusion_rag")
    submod = metrics.get("unifusion_submod")

    if baseline is None:
        print("ERROR: dense_concat baseline not found")
        return

    print("=" * 60)
    print("Submodular Packing Smoke Test Results")
    print("=" * 60)

    def _fmt(name: str, vals: dict[str, float]) -> str:
        return (
            f"{name:20s}  F1={vals['f1']:.4f}  EM={vals['exact_match']:.4f}  "
            f"R@10={vals['recall_at_10']:.4f}  P95={vals['p95_latency_ms']:.1f}ms"
        )

    print(_fmt("Dense-Concat", baseline))
    if unifusion:
        print(_fmt("UniFusion-RAG", unifusion))
    if submod:
        print(_fmt("UniFusion-Submod", submod))

    print("-" * 60)
    if unifusion:
        df1 = unifusion["f1"] - baseline["f1"]
        dem = unifusion["exact_match"] - baseline["exact_match"]
        dr10 = unifusion["recall_at_10"] - baseline["recall_at_10"]
        print(f"UniFusion vs Dense   ΔF1={df1:+.4f}  ΔEM={dem:+.4f}  ΔR@10={dr10:+.4f}")
    if submod:
        df1 = submod["f1"] - baseline["f1"]
        dem = submod["exact_match"] - baseline["exact_match"]
        dr10 = submod["recall_at_10"] - baseline["recall_at_10"]
        print(f"Submod   vs Dense    ΔF1={df1:+.4f}  ΔEM={dem:+.4f}  ΔR@10={dr10:+.4f}")
    if unifusion and submod:
        df1 = submod["f1"] - unifusion["f1"]
        dem = submod["exact_match"] - unifusion["exact_match"]
        dr10 = submod["recall_at_10"] - unifusion["recall_at_10"]
        print(f"Submod   vs UniFusion ΔF1={df1:+.4f}  ΔEM={dem:+.4f}  ΔR@10={dr10:+.4f}")
    print("=" * 60)


def main() -> None:
    if len(sys.argv) < 2:
        print(f"Usage: {sys.argv[0]} <result_dir>")
        sys.exit(1)
    result_dir = sys.argv[1]
    metrics = load_metrics(result_dir)
    if metrics is None:
        print(f"ERROR: metrics file not found in {result_dir}")
        sys.exit(1)
    compare(metrics)


if __name__ == "__main__":
    main()
