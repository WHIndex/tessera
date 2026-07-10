#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def load_json(path: Path):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def parse_profile_and_topk(run_name: str) -> tuple[str, int | None]:
    profile = "unknown"
    topk = None
    parts = run_name.split("_")
    if parts:
        profile = parts[0]
    for i, p in enumerate(parts):
        if p == "topk" and i + 1 < len(parts):
            try:
                topk = int(parts[i + 1])
            except Exception:
                topk = None
            break
    return profile, topk


def pareto_front(points: list[dict]) -> set[str]:
    keep = set()
    for i, a in enumerate(points):
        dominated = False
        for j, b in enumerate(points):
            if i == j:
                continue
            better_or_equal_latency = float(b["p95_latency_ms"]) <= float(a["p95_latency_ms"])
            better_or_equal_f1 = float(b["f1"]) >= float(a["f1"])
            strictly_better = (
                float(b["p95_latency_ms"]) < float(a["p95_latency_ms"])
                or float(b["f1"]) > float(a["f1"])
            )
            if better_or_equal_latency and better_or_equal_f1 and strictly_better:
                dominated = True
                break
        if not dominated:
            keep.add(str(a["run_name"]))
    return keep


def main() -> int:
    parser = argparse.ArgumentParser(description="Summarize Issue12 Pareto scan runs")
    parser.add_argument("--scan-root", type=Path, required=True)
    parser.add_argument("--out-json", type=Path, required=True)
    parser.add_argument("--out-csv", type=Path, required=True)
    parser.add_argument("--out-md", type=Path, required=True)
    args = parser.parse_args()

    if not args.scan_root.exists():
        raise FileNotFoundError(args.scan_root)

    rows = []
    for run_dir in sorted([p for p in args.scan_root.iterdir() if p.is_dir()]):
        mpath = run_dir / "table1c_e2e_metrics.json"
        if not mpath.exists():
            continue
        data = load_json(mpath)
        if not data:
            continue
        methods = data.get("methods", {})
        dense = methods.get("dense_concat", {})
        uni = methods.get("unifusion_rag", {})
        profile, topk = parse_profile_and_topk(run_dir.name)
        rows.append(
            {
                "run_name": run_dir.name,
                "profile": profile,
                "topk": topk,
                "f1": float(uni.get("f1", 0.0)),
                "em": float(uni.get("exact_match", 0.0)),
                "recall@10": float(uni.get("recall@10", 0.0)),
                "p95_latency_ms": float(uni.get("p95_latency_ms", 0.0)),
                "dense_f1": float(dense.get("f1", 0.0)),
                "dense_em": float(dense.get("exact_match", 0.0)),
                "dense_recall@10": float(dense.get("recall@10", 0.0)),
                "dense_p95_latency_ms": float(dense.get("p95_latency_ms", 0.0)),
                "delta_f1_vs_dense": float(uni.get("f1", 0.0)) - float(dense.get("f1", 0.0)),
                "delta_em_vs_dense": float(uni.get("exact_match", 0.0)) - float(dense.get("exact_match", 0.0)),
                "delta_p95_vs_dense_ms": float(uni.get("p95_latency_ms", 0.0)) - float(dense.get("p95_latency_ms", 0.0)),
            }
        )

    pareto_set = pareto_front(rows)
    for r in rows:
        r["is_pareto"] = 1 if r["run_name"] in pareto_set else 0

    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")

    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    with args.out_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "run_name",
                "profile",
                "topk",
                "f1",
                "em",
                "recall@10",
                "p95_latency_ms",
                "dense_f1",
                "dense_em",
                "dense_recall@10",
                "dense_p95_latency_ms",
                "delta_f1_vs_dense",
                "delta_em_vs_dense",
                "delta_p95_vs_dense_ms",
                "is_pareto",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    md = [
        "# Issue12 Pareto Summary",
        "",
        "| run | profile | topk | uni f1 | uni em | uni r10 | uni p95(ms) | delta f1 vs dense | delta em vs dense | delta p95(ms) | pareto |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for r in sorted(rows, key=lambda x: (x["p95_latency_ms"], -x["f1"])):
        md.append(
            "| {run_name} | {profile} | {topk} | {f1:.4f} | {em:.4f} | {r10:.4f} | {p95:.2f} | {df1:+.4f} | {dem:+.4f} | {dp95:+.2f} | {pareto} |".format(
                run_name=r["run_name"],
                profile=r["profile"],
                topk=r["topk"],
                f1=r["f1"],
                em=r["em"],
                r10=r["recall@10"],
                p95=r["p95_latency_ms"],
                df1=r["delta_f1_vs_dense"],
                dem=r["delta_em_vs_dense"],
                dp95=r["delta_p95_vs_dense_ms"],
                pareto="yes" if r["is_pareto"] else "no",
            )
        )
    args.out_md.parent.mkdir(parents=True, exist_ok=True)
    args.out_md.write_text("\n".join(md) + "\n", encoding="utf-8")

    print(f"[OK] runs={len(rows)}")
    print(f"[OK] json -> {args.out_json}")
    print(f"[OK] csv -> {args.out_csv}")
    print(f"[OK] markdown -> {args.out_md}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
