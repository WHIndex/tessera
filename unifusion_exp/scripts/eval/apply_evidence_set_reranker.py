#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
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

from unifusion_exp.e2e.evidence_set_reranker import load_evidence_set_reranker_bundle  # noqa: E402


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


def ranking_for_row(row: dict[str, Any], method: str, fallback_method: str | None = None) -> list[str]:
    rankings = row.get("rankings", {}) or {}
    selected = rankings.get(method)
    if selected is None and fallback_method:
        selected = rankings.get(fallback_method)
    return [str(x) for x in (selected or [])]


def qrels_for_row(row: dict[str, Any]) -> dict[str, float]:
    out: dict[str, float] = {}
    for doc_id, raw in (row.get("qrels", {}) or {}).items():
        try:
            grade = float(raw)
        except Exception:
            continue
        if grade > 0.0:
            out[str(doc_id)] = grade
    return out


def collect_needed_doc_ids(
    rows: Sequence[dict[str, Any]], *, method: str, pool_k: int, fallback_method: str | None = None
) -> set[str]:
    needed: set[str] = set()
    for row in rows:
        needed.update(ranking_for_row(row, method, fallback_method)[: int(pool_k)])
        needed.update(qrels_for_row(row))
    return needed


def load_corpus_texts(paths: Sequence[Path], needed_ids: set[str] | None = None) -> dict[str, str]:
    corpus: dict[str, str] = {}
    needed = set(needed_ids or [])
    for path in paths:
        if not path.exists():
            raise FileNotFoundError(f"corpus-json not found: {path}")
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, dict):
            for doc_id, payload in data.items():
                doc_id = str(doc_id)
                if needed and doc_id not in needed:
                    continue
                if isinstance(payload, dict):
                    text = str(payload.get("text", "") or payload.get("contents", "") or "")
                else:
                    text = str(payload or "")
                if text and doc_id not in corpus:
                    corpus[doc_id] = text
            continue
        if not isinstance(data, list):
            continue
        for item in data:
            if not isinstance(item, dict):
                continue
            doc_id = str(item.get("id", "") or item.get("doc_id", "") or item.get("chunk_id", ""))
            if not doc_id:
                continue
            if needed and doc_id not in needed:
                continue
            text = str(item.get("text", "") or item.get("contents", "") or "")
            if text and doc_id not in corpus:
                corpus[doc_id] = text
    return corpus


def dcg(grades: Sequence[float]) -> float:
    return float(sum((2.0 ** float(grade) - 1.0) / math.log2(rank + 2.0) for rank, grade in enumerate(grades)))


def method_metrics(rows: Sequence[dict[str, Any]], preds: Sequence[Sequence[str]], ks: Sequence[int]) -> tuple[dict[str, float], dict[str, list[float]]]:
    details: dict[str, list[float]] = {}
    rel_counts: list[int] = []
    for k in ks:
        details[f"ndcg@{k}"] = []
        details[f"map@{k}"] = []
        details[f"hits@{k}"] = []
        details[f"any_hit@{k}"] = []
    for row, pred in zip(rows, preds):
        qrels = qrels_for_row(row)
        rel = {doc_id for doc_id, grade in qrels.items() if grade > 0.0}
        rel_counts.append(len(rel))
        for k in ks:
            top = list(pred)[: int(k)]
            grades = [float(qrels.get(doc_id, 0.0)) for doc_id in top]
            ideal = sorted([float(v) for v in qrels.values() if float(v) > 0.0], reverse=True)[: int(k)]
            idcg = dcg(ideal)
            ndcg = dcg(grades) / idcg if idcg > 0.0 else 0.0
            hits = 0
            ap_sum = 0.0
            for rank, doc_id in enumerate(top, start=1):
                if doc_id in rel:
                    hits += 1
                    ap_sum += hits / rank
            details[f"ndcg@{k}"].append(float(ndcg))
            details[f"map@{k}"].append(float(ap_sum / len(rel) if rel else 0.0))
            details[f"hits@{k}"].append(float(hits))
            details[f"any_hit@{k}"].append(float(hits > 0))
    summary = {"avg_positive_qrels": float(np.mean(rel_counts)) if rel_counts else 0.0}
    for key, vals in details.items():
        summary[key] = float(np.mean(vals)) if vals else 0.0
    return summary, {"rel_count": rel_counts, **details}


def write_csv(path: Path, rows: list[dict[str, Any]], metric_keys: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["method", "label", *metric_keys])
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in ["method", "label", *metric_keys]})


def write_markdown(path: Path, rows: list[dict[str, Any]], metric_keys: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    headers = ["Method", *metric_keys]
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for row in rows:
        values = [str(row["label"])] + [f"{float(row[key]):.4f}" for key in metric_keys]
        lines.append("| " + " | ".join(values) + " |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def top_score_map(scores: dict[str, float], n: int = 8) -> dict[str, float]:
    return {
        doc_id: float(score)
        for doc_id, score in sorted(scores.items(), key=lambda item: item[1], reverse=True)[: int(n)]
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Apply a supervised Evidence-Set Reranker to saved rankings_debug.jsonl.")
    parser.add_argument("--base-rankings-jsonl", type=Path, required=True)
    parser.add_argument("--reranker-model", type=Path, required=True)
    parser.add_argument("--corpus-json", type=Path, action="append", required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--method", type=str, default="tessera")
    parser.add_argument("--base-method", type=str, default="unifusion_rag")
    parser.add_argument("--label", type=str, default="TESSERA")
    parser.add_argument("--metrics-k", type=str, default="1,5")
    parser.add_argument("--max-queries", type=int, default=0)
    parser.add_argument("--pool-k", type=int, default=None)
    parser.add_argument("--preserve-top", type=int, default=None)
    parser.add_argument("--blend-original-weight", type=float, default=None)
    parser.add_argument("--top1-switch-margin", type=float, default=None)
    parser.add_argument("--anchor-guard", action="store_true")
    parser.add_argument("--anchor-guard-topk", type=int, default=None)
    parser.add_argument("--anchor-guard-max-restores", type=int, default=None)
    parser.add_argument("--anchor-guard-min-model-score", type=float, default=None)
    parser.add_argument("--include-base", action="store_true")
    args = parser.parse_args()

    if not args.base_rankings_jsonl.exists():
        raise FileNotFoundError(f"base-rankings-jsonl not found: {args.base_rankings_jsonl}")
    if not args.reranker_model.exists():
        raise FileNotFoundError(f"reranker-model not found: {args.reranker_model}")

    rows = iter_jsonl(args.base_rankings_jsonl, int(args.max_queries))
    bundle = load_evidence_set_reranker_bundle(args.reranker_model)
    if args.pool_k is not None:
        bundle = bundle.with_config(pool_k=int(args.pool_k))
    if args.preserve_top is not None:
        bundle = bundle.with_config(preserve_top=int(args.preserve_top))
    if args.blend_original_weight is not None:
        bundle = bundle.with_config(blend_original_weight=float(args.blend_original_weight))
    if args.top1_switch_margin is not None:
        bundle = bundle.with_config(top1_switch_margin=float(args.top1_switch_margin))
    if bool(args.anchor_guard):
        bundle = bundle.with_config(anchor_guard_enabled=True)
    if args.anchor_guard_topk is not None:
        bundle = bundle.with_config(anchor_guard_topk=int(args.anchor_guard_topk))
    if args.anchor_guard_max_restores is not None:
        bundle = bundle.with_config(anchor_guard_max_restores=int(args.anchor_guard_max_restores))
    if args.anchor_guard_min_model_score is not None:
        bundle = bundle.with_config(anchor_guard_min_model_score=float(args.anchor_guard_min_model_score))

    ks = sorted({int(x.strip()) for x in str(args.metrics_k).split(",") if x.strip()})
    fallback_method = str(args.base_method) if str(args.base_method) != str(args.method) else None
    needed = collect_needed_doc_ids(
        rows,
        method=str(args.method),
        fallback_method=fallback_method,
        pool_k=int(bundle.config.pool_k),
    )
    corpus_texts = load_corpus_texts(args.corpus_json, needed)

    preds: list[list[str]] = []
    base_preds: list[list[str]] = []
    debug_rows: list[dict[str, Any]] = []
    changed = 0
    top1_switched = 0
    anchor_guard_restored = 0
    anchor_guard_queries = 0
    missing_text = 0
    for i, row in enumerate(rows):
        base = ranking_for_row(row, str(args.method), fallback_method)
        base_preds.append(base)
        missing_text += sum(1 for doc_id in base[: int(bundle.config.pool_k)] if doc_id not in corpus_texts)
        result = bundle.rerank(
            query_text=str(row.get("query", "")),
            query_id=str(row.get("query_id", "")),
            ranked_doc_ids=base,
            trace=row.get("trace", {}) or {},
            corpus_texts=corpus_texts,
        )
        preds.append(result.ranked_doc_ids)
        changed += int(result.changed_count > 0)
        top1_switched += int(result.switched_top1)
        anchor_guard_restored += int(result.anchor_guard_restored)
        anchor_guard_queries += int(result.anchor_guard_restored > 0)
        debug_rows.append(
            {
                "query_index": i,
                "query_id": row.get("query_id", ""),
                "query": row.get("query", ""),
                "qrels": qrels_for_row(row),
                "rankings": {
                    str(args.method): result.ranked_doc_ids,
                    "tessera": result.ranked_doc_ids,
                    "unifusion_rag": result.ranked_doc_ids,
                    "_base_tessera": base,
                },
                "trace": {
                    **(row.get("trace", {}) or {}),
                    "tessera_active": 1.0,
                    "tessera_pool_k": float(bundle.config.pool_k),
                    "tessera_preserve_top": float(bundle.config.preserve_top),
                    "tessera_blend_original_weight": float(bundle.config.blend_original_weight),
                    "tessera_top1_switch_margin": float(bundle.config.top1_switch_margin),
                    "tessera_changed_count": float(result.changed_count),
                    "tessera_switched_top1": float(result.switched_top1),
                    "tessera_pool_size": float(result.pool_size),
                    "tessera_anchor_guard_enabled": float(bool(bundle.config.anchor_guard_enabled)),
                    "tessera_anchor_guard_restored": float(result.anchor_guard_restored),
                    "tessera_anchor_guard_anchor_count": float(len(result.anchor_guard_anchors)),
                    "tessera_anchor_guard_anchors": list(result.anchor_guard_anchors[:8]),
                    "tessera_model_scores_top": top_score_map(result.model_scores),
                    "tessera_final_scores_top": top_score_map(result.final_scores),
                },
            }
        )

    summary, detail = method_metrics(rows, preds, ks)
    metric_keys: list[str] = []
    for metric in ["ndcg", "map", "hits"]:
        for k in ks:
            metric_keys.append(f"{metric}@{k}")
    for k in ks:
        metric_keys.append(f"any_hit@{k}")
    table_rows = [{"method": str(args.method), "label": str(args.label), **summary}]
    details = {"query_ids": [str(row.get("query_id", "")) for row in rows], "methods": {str(args.method): detail}}
    if bool(args.include_base):
        base_summary, base_detail = method_metrics(rows, base_preds, ks)
        table_rows.insert(0, {"method": "base_tessera", "label": "Base TESSERA", **base_summary})
        details["methods"]["base_tessera"] = base_detail

    args.out_dir.mkdir(parents=True, exist_ok=True)
    meta = {
        "base_rankings_jsonl": str(args.base_rankings_jsonl),
        "reranker_model": str(args.reranker_model),
        "corpus_json": [str(x) for x in args.corpus_json],
        "reranker_metadata": bundle.metadata,
        "config": bundle.config.__dict__,
        "queries": int(len(rows)),
        "changed_queries": int(changed),
        "top1_switched_queries": int(top1_switched),
        "anchor_guard_queries": int(anchor_guard_queries),
        "anchor_guard_restored": int(anchor_guard_restored),
        "corpus_loaded": int(len(corpus_texts)),
        "needed_doc_ids": int(len(needed)),
        "missing_text_in_pool": int(missing_text),
    }
    payload = {
        "metadata": meta,
        "methods": {row["method"]: {k: v for k, v in row.items() if k not in {"method", "label"}} for row in table_rows},
    }
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
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
