#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path


def find_metrics_files(runs_dir: Path):
    return sorted(runs_dir.glob("**/*_metrics.json"))


def find_retrieval_files(results_dir: Path):
    files = []
    for pat in ["retrieval_*.json", "tessera_*.json"]:
        files.extend(results_dir.glob(pat))
    return sorted(files)


def find_service_files(results_dir: Path):
    files = []
    for pat in ["neo4j_*.json", "milvus_*.json"]:
        files.extend(results_dir.glob(pat))
    return sorted(files)


def find_significance_files(results_dir: Path):
    return sorted(results_dir.glob("significance_*.json"))


def main() -> int:
    parser = argparse.ArgumentParser(description="Collect metrics files into a markdown report")
    parser.add_argument("--runs-dir", type=Path, required=True)
    parser.add_argument("--out-md", type=Path, required=True)
    args = parser.parse_args()

    files = find_metrics_files(args.runs_dir)
    retrieval_files = find_retrieval_files(Path("artifacts/results"))
    service_files = find_service_files(Path("artifacts/results"))
    significance_files = find_significance_files(Path("artifacts/results"))

    lines = [
        "# Run Metrics Report",
        "",
        f"total metric files: {len(files)}",
        f"total retrieval files: {len(retrieval_files)}",
        f"total service files: {len(service_files)}",
        f"total significance files: {len(significance_files)}",
        "",
    ]

    for fp in files:
        try:
            data = json.loads(fp.read_text(encoding="utf-8"))
        except Exception:
            continue
        lines.append(f"## {fp.stem}")
        lines.append(f"- path: {fp}")
        for k in [
            "run_id",
            "threshold",
            "train_rows",
            "val_rows",
            "val_micro_f1",
            "val_subset_acc",
            "test_rows",
            "test_micro_f1",
            "test_subset_acc",
        ]:
            if k in data:
                lines.append(f"- {k}: {data[k]}")
        lines.append("")

    if retrieval_files:
        lines.append("# Retrieval Results")
        lines.append("")
        for fp in retrieval_files:
            try:
                data = json.loads(fp.read_text(encoding="utf-8"))
            except Exception:
                continue
            lines.append(f"## {fp.stem}")
            lines.append(f"- path: {fp}")
            for k in ["queries", "corpus", "recall@5", "recall@10", "recall@20"]:
                if k in data:
                    lines.append(f"- {k}: {data[k]}")
            lines.append("")

    if service_files:
        lines.append("# Service Results")
        lines.append("")
        for fp in service_files:
            try:
                data = json.loads(fp.read_text(encoding="utf-8"))
            except Exception:
                continue
            lines.append(f"## {fp.stem}")
            lines.append(f"- path: {fp}")
            for k in [
                "triples_loaded",
                "nodes",
                "rels",
                "elapsed_sec",
                "entities",
                "hit_entities",
                "hit_rate",
                "latency_ms_avg",
                "latency_ms_p95",
            ]:
                if k in data:
                    lines.append(f"- {k}: {data[k]}")
            lines.append("")

    if significance_files:
        lines.append("# Significance Results")
        lines.append("")
        for fp in significance_files:
            try:
                data = json.loads(fp.read_text(encoding="utf-8"))
            except Exception:
                continue
            lines.append(f"## {fp.stem}")
            lines.append(f"- path: {fp}")
            rows = data.get("rows", [])
            lines.append(f"- rows: {len(rows)}")
            for row in rows:
                if row.get("type") == "delta_ci" and row.get("metric") in {"recall@10", "recall@20", "hit_rate"}:
                    lines.append(
                        "- delta: "
                        f"{row.get('method')} {row.get('metric')} "
                        f"delta={row.get('delta')} "
                        f"ci95=[{row.get('delta_ci95_low')}, {row.get('delta_ci95_high')}] "
                        f"p={row.get('p_value')}"
                    )
            lines.append("")

    args.out_md.parent.mkdir(parents=True, exist_ok=True)
    args.out_md.write_text("\n".join(lines), encoding="utf-8")
    print(f"[OK] report -> {args.out_md}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
