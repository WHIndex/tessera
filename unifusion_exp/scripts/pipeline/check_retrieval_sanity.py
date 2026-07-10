#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path


def load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def collect_positive_qrels(rows: list[dict]) -> set[str]:
    out = set()
    for r in rows:
        for chunk_id, label in r.get("relevant_chunks", {}).items():
            try:
                if float(label) > 0:
                    out.add(chunk_id)
            except Exception:
                continue
    return out


def get_metric(d: dict, key: str, default: float | None = None):
    x = d.get(key, default)
    if x is None:
        return None
    try:
        return float(x)
    except Exception:
        return default


def main() -> int:
    parser = argparse.ArgumentParser(description="Sanity checks for retrieval metrics/results")
    parser.add_argument("--dense-metrics", type=Path, required=True)
    parser.add_argument("--main-metrics", type=Path, default=None, help="Unified main-vs-baseline result json")
    parser.add_argument("--main-detail", type=Path, default=None, help="Detailed json from unified evaluation")
    parser.add_argument("--split-file", type=Path, default=None, help="Optional split json for corpus-conditioning audit")
    parser.add_argument("--corpus-file", type=Path, default=None, help="Optional corpus json for corpus-conditioning audit")
    parser.add_argument("--out-json", type=Path, default=Path("artifacts/results/retrieval_sanity_report_v1.json"))
    parser.add_argument("--out-md", type=Path, default=Path("artifacts/results/retrieval_sanity_report_v1.md"))
    parser.add_argument("--main-regression-tol", type=float, default=0.0)
    parser.add_argument("--main-dense-overlap-warn", type=float, default=0.9)
    parser.add_argument("--main-dense-overlap-fail", type=float, default=0.98)
    parser.add_argument("--max-query-conditioned-ratio", type=float, default=0.6)
    parser.add_argument("--strict", action="store_true", help="Exit non-zero if any FAILED check")
    args = parser.parse_args()

    dense = load_json(args.dense_metrics)
    checks = []

    def add_check(name: str, status: str, detail: str):
        checks.append({"name": name, "status": status, "detail": detail})

    dense_q = int(dense.get("queries", 0))
    dense_corpus = int(dense.get("corpus", 0))
    dense_any5 = get_metric(dense, "any_hit@5", 0.0)
    dense_rec5 = get_metric(dense, "recall@5", 0.0)
    dense_rec10 = get_metric(dense, "recall@10", 0.0)
    dense_cov = get_metric(dense, "qrels_coverage_in_corpus", get_metric(dense, "qrels_coverage_in_results", 0.0))
    dense_no_pos = int(dense.get("queries_without_positive_qrels", 0))
    dense_avg_pos = get_metric(dense, "avg_positive_qrels", 0.0)

    if dense_q <= 0 or dense_corpus <= 0:
        add_check("dense_file_valid", "FAILED", "queries/corpus must be > 0")
    else:
        add_check("dense_file_valid", "PASSED", f"queries={dense_q}, corpus={dense_corpus}")

    if dense_no_pos > 0:
        add_check("dense_positive_qrels", "FAILED", f"queries_without_positive_qrels={dense_no_pos}")
    else:
        add_check("dense_positive_qrels", "PASSED", "all queries have positive qrels")

    if dense_cov is None or dense_cov < 0.8:
        add_check("dense_qrels_coverage", "FAILED", f"coverage={dense_cov}")
    elif dense_cov < 0.95:
        add_check("dense_qrels_coverage", "WARN", f"coverage={dense_cov:.4f}")
    else:
        add_check("dense_qrels_coverage", "PASSED", f"coverage={dense_cov:.4f}")

    if dense_any5 is not None and dense_rec5 is not None and (dense_any5 - dense_rec5) > 0.2:
        add_check(
            "anyhit_vs_recall_gap",
            "WARN",
            f"any_hit@5={dense_any5:.4f} recall@5={dense_rec5:.4f} gap={dense_any5-dense_rec5:.4f}; do not treat any-hit as recall",
        )
    else:
        add_check("anyhit_vs_recall_gap", "PASSED", f"any_hit@5={dense_any5:.4f} recall@5={dense_rec5:.4f}")

    if dense_avg_pos is not None and dense_avg_pos < 1.0:
        add_check("avg_positive_qrels", "WARN", f"avg_positive_qrels={dense_avg_pos:.4f}")
    else:
        add_check("avg_positive_qrels", "PASSED", f"avg_positive_qrels={dense_avg_pos:.4f}")

    if args.main_metrics is not None and args.main_metrics.exists():
        main = load_json(args.main_metrics)
        methods = main.get("methods", {})
        dense_m = methods.get("baseline_dense", {})
        main_m = methods.get("main_unifusion", {})

        d10 = get_metric(dense_m, "recall@10", None)
        m10 = get_metric(main_m, "recall@10", None)
        d20 = get_metric(dense_m, "recall@20", None)
        m20 = get_metric(main_m, "recall@20", None)

        if d10 is None or m10 is None:
            add_check("main_vs_dense_r10", "FAILED", "missing recall@10 in main metrics")
        elif m10 + args.main_regression_tol < d10:
            add_check("main_vs_dense_r10", "FAILED", f"main={m10:.4f} < dense={d10:.4f} (tol={args.main_regression_tol:.4f})")
        else:
            add_check("main_vs_dense_r10", "PASSED", f"main={m10:.4f}, dense={d10:.4f}")

        if d20 is None or m20 is None:
            add_check("main_vs_dense_r20", "FAILED", "missing recall@20 in main metrics")
        elif m20 + args.main_regression_tol < d20:
            add_check("main_vs_dense_r20", "FAILED", f"main={m20:.4f} < dense={d20:.4f} (tol={args.main_regression_tol:.4f})")
        else:
            add_check("main_vs_dense_r20", "PASSED", f"main={m20:.4f}, dense={d20:.4f}")

    if args.main_detail is not None and args.main_detail.exists():
        detail = load_json(args.main_detail)
        diag = detail.get("main_diagnostics", {}).get("summary", {})
        overlap = get_metric(diag, "avg_main_dense_overlap_at_k", None)
        new_docs = get_metric(diag, "avg_main_new_over_dense_at_k", None)
        if overlap is None:
            add_check("main_dense_overlap", "WARN", "main_diagnostics.summary.avg_main_dense_overlap_at_k missing")
        elif overlap >= args.main_dense_overlap_fail:
            add_check("main_dense_overlap", "FAILED", f"avg_overlap={overlap:.4f} >= fail={args.main_dense_overlap_fail:.4f}")
        elif overlap >= args.main_dense_overlap_warn:
            add_check("main_dense_overlap", "WARN", f"avg_overlap={overlap:.4f} >= warn={args.main_dense_overlap_warn:.4f}")
        else:
            add_check("main_dense_overlap", "PASSED", f"avg_overlap={overlap:.4f}")

        if new_docs is None:
            add_check("main_new_docs", "WARN", "main_diagnostics.summary.avg_main_new_over_dense_at_k missing")
        elif new_docs <= 0:
            add_check("main_new_docs", "FAILED", f"avg_new_over_dense={new_docs:.4f}")
        elif new_docs < 1.0:
            add_check("main_new_docs", "WARN", f"avg_new_over_dense={new_docs:.4f}")
        else:
            add_check("main_new_docs", "PASSED", f"avg_new_over_dense={new_docs:.4f}")

    if args.split_file is not None and args.corpus_file is not None and args.split_file.exists() and args.corpus_file.exists():
        rows = load_json(args.split_file)
        corpus = load_json(args.corpus_file)
        corpus_ids = {x.get("id") for x in corpus if isinstance(x, dict) and x.get("id")}
        qrels = collect_positive_qrels(rows)
        in_corpus = qrels & corpus_ids
        conditioned_ratio = len(in_corpus) / max(1, len(corpus_ids))
        qrels_coverage = len(in_corpus) / max(1, len(qrels))

        if conditioned_ratio > args.max_query_conditioned_ratio:
            add_check(
                "corpus_query_conditioning",
                "WARN",
                f"qrel_docs_in_corpus_ratio={conditioned_ratio:.4f} > threshold={args.max_query_conditioned_ratio:.4f}",
            )
        else:
            add_check(
                "corpus_query_conditioning",
                "PASSED",
                f"qrel_docs_in_corpus_ratio={conditioned_ratio:.4f}",
            )

        if qrels_coverage < 0.8:
            add_check("corpus_qrels_coverage", "WARN", f"qrels_coverage={qrels_coverage:.4f}")
        else:
            add_check("corpus_qrels_coverage", "PASSED", f"qrels_coverage={qrels_coverage:.4f}")

    failed = [c for c in checks if c["status"] == "FAILED"]
    warn = [c for c in checks if c["status"] == "WARN"]

    summary = {
        "failed": len(failed),
        "warn": len(warn),
        "checks": checks,
        "dense_metrics": str(args.dense_metrics),
        "main_metrics": str(args.main_metrics) if args.main_metrics else None,
        "main_detail": str(args.main_detail) if args.main_detail else None,
        "split_file": str(args.split_file) if args.split_file else None,
        "corpus_file": str(args.corpus_file) if args.corpus_file else None,
    }

    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    md = [
        "# Retrieval Sanity Report",
        "",
        f"- dense_metrics: {args.dense_metrics}",
        f"- main_metrics: {args.main_metrics}",
        f"- main_detail: {args.main_detail}",
        f"- split_file: {args.split_file}",
        f"- corpus_file: {args.corpus_file}",
        f"- failed: {len(failed)}",
        f"- warn: {len(warn)}",
        "",
        "| Check | Status | Detail |",
        "|---|---|---|",
    ]
    for c in checks:
        md.append(f"| {c['name']} | {c['status']} | {c['detail']} |")
    args.out_md.write_text("\n".join(md) + "\n", encoding="utf-8")

    print(f"[OK] json -> {args.out_json}")
    print(f"[OK] md -> {args.out_md}")

    if args.strict and failed:
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
