#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import re
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[2]

DOC_ID_RE = re.compile(r"^\[DOC_ID:\s*([^\]]+)\]\s*\n?", re.DOTALL)


def qrels_for_row(row: dict[str, Any]) -> dict[str, float]:
    out: dict[str, float] = {}
    for chunk_id, label in row.get("relevant_chunks", {}).items():
        try:
            grade = float(label)
        except Exception:
            continue
        if grade > 0:
            out[str(chunk_id)] = grade
    return out


def dcg(grades: list[float]) -> float:
    total = 0.0
    for rank, grade in enumerate(grades, start=1):
        total += (2.0 ** float(grade) - 1.0) / math.log2(rank + 1.0)
    return total


def method_metrics(rows: list[dict[str, Any]], preds: list[list[str]], ks: list[int]) -> tuple[dict, dict]:
    details: dict[str, list[float]] = {}
    rel_counts: list[int] = []
    for k in ks:
        details[f"ndcg@{k}"] = []
        details[f"map@{k}"] = []
        details[f"hits@{k}"] = []
        details[f"any_hit@{k}"] = []

    for row, pred in zip(rows, preds):
        qrels = qrels_for_row(row)
        rel = {doc_id for doc_id, grade in qrels.items() if grade > 0}
        rel_counts.append(len(rel))

        for k in ks:
            top = pred[:k]
            top_grades = [float(qrels.get(doc_id, 0.0)) for doc_id in top]
            ideal_grades = sorted(qrels.values(), reverse=True)[:k]
            idcg = dcg(ideal_grades)
            ndcg_val = dcg(top_grades) / idcg if idcg > 0 else 0.0

            hit_count = 0
            ap_sum = 0.0
            for rank, doc_id in enumerate(top, start=1):
                if doc_id in rel:
                    hit_count += 1
                    ap_sum += hit_count / rank
            map_val = ap_sum / len(rel) if rel else 0.0

            details[f"ndcg@{k}"].append(float(ndcg_val))
            details[f"map@{k}"].append(float(map_val))
            details[f"hits@{k}"].append(float(hit_count))
            details[f"any_hit@{k}"].append(float(hit_count > 0))

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
            writer.writerow({k: row.get(k, "") for k in ["method", "label", *metric_keys]})


def write_markdown(path: Path, rows: list[dict[str, Any]], metric_keys: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    headers = ["Method", *metric_keys]
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for row in rows:
        vals = [row["label"]] + [f"{float(row[k]):.4f}" for k in metric_keys]
        lines.append("| " + " | ".join(vals) + " |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def tagged_doc(doc_id: str, text: str) -> str:
    return f"[DOC_ID: {doc_id}]\n{text}"


def parse_doc_id(doc_text: str, text_to_doc_id: dict[str, str]) -> str | None:
    match = DOC_ID_RE.match(doc_text or "")
    if match:
        return match.group(1).strip()
    return text_to_doc_id.get(doc_text)


def filter_rows_with_corpus_positives(
    rows: list[dict[str, Any]],
    corpus_doc_ids: set[str],
    max_queries: int,
) -> list[dict[str, Any]]:
    kept: list[dict[str, Any]] = []
    for row in rows:
        qrels = qrels_for_row(row)
        if any(doc_id in corpus_doc_ids for doc_id in qrels):
            kept.append(row)
        if len(kept) >= max_queries:
            break
    return kept


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate HippoRAG2 on the mmRAG paper retrieval metrics.")
    parser.add_argument("--hipporag2-root", type=Path, required=True)
    parser.add_argument("--split-file", type=Path, required=True)
    parser.add_argument("--corpus-file", type=Path, required=True)
    parser.add_argument("--out-json", type=Path, required=True)
    parser.add_argument("--out-csv", type=Path, required=True)
    parser.add_argument("--out-md", type=Path, required=True)
    parser.add_argument("--detail-json", type=Path, default=None)
    parser.add_argument("--save-rankings-jsonl", type=Path, default=None)
    parser.add_argument("--save-dir", type=Path, required=True)
    parser.add_argument("--max-queries", type=int, default=1286)
    parser.add_argument("--max-corpus", type=int, default=0)
    parser.add_argument("--filter-queries-with-positive-in-corpus", action="store_true")
    parser.add_argument("--retrieve-topk", type=int, default=10)
    parser.add_argument("--metrics-k", type=str, default="1,5")
    parser.add_argument("--llm-model-name", type=str, default="gpt-4o-mini")
    parser.add_argument("--llm-base-url", type=str, default=None)
    parser.add_argument("--embedding-model-name", type=str, default="text-embedding-3-small")
    parser.add_argument("--embedding-base-url", type=str, default=None)
    parser.add_argument("--embedding-batch-size", type=int, default=64)
    parser.add_argument("--linking-top-k", type=int, default=5)
    parser.add_argument("--force-index-from-scratch", action="store_true")
    parser.add_argument("--force-openie-from-scratch", action="store_true")
    parser.add_argument("--skip-index", action="store_true")
    parser.add_argument("--no-save-openie", action="store_true")
    parser.add_argument("--enable-ontology", action="store_true")
    parser.add_argument("--enable-query-plan-ontology", action="store_true")
    args = parser.parse_args()

    if int(args.max_queries) < 1:
        raise ValueError("--max-queries must be >= 1")
    if int(args.retrieve_topk) < 1:
        raise ValueError("--retrieve-topk must be >= 1")

    hipporag2_root = args.hipporag2_root.resolve()
    if not hipporag2_root.exists():
        raise FileNotFoundError(f"HippoRAG2 root not found: {hipporag2_root}")
    sys.path.insert(0, str(hipporag2_root))

    try:
        from src.hipporag.HippoRAG import HippoRAG
        from src.hipporag.utils.config_utils import BaseConfig
    except Exception as exc:
        raise RuntimeError(
            "Failed to import HippoRAG2. Run this script in a HippoRAG2 environment, "
            "for example `conda run -n hipporag python ...`, or install ../HippoRAG2/requirements.txt."
        ) from exc

    split_rows_all = json.loads(args.split_file.read_text(encoding="utf-8"))
    corpus_all = json.loads(args.corpus_file.read_text(encoding="utf-8"))
    if int(args.max_corpus) > 0:
        corpus = corpus_all[: int(args.max_corpus)]
    else:
        corpus = corpus_all
    doc_ids = [str(doc["id"]) for doc in corpus]
    doc_texts = [str(doc.get("text", "")) for doc in corpus]
    doc_id_set = set(doc_ids)

    if args.filter_queries_with_positive_in_corpus:
        rows = filter_rows_with_corpus_positives(split_rows_all, doc_id_set, max_queries=int(args.max_queries))
    else:
        rows = split_rows_all[: int(args.max_queries)]

    q_texts = [str(row.get("query", "")) for row in rows]
    q_ids = [str(row.get("id", f"q_{idx}")) for idx, row in enumerate(rows)]
    ks = sorted({int(k.strip()) for k in args.metrics_k.split(",") if k.strip()})
    retrieve_topk = max(int(args.retrieve_topk), max(ks))

    qrels_total = 0
    qrels_in_corpus = 0
    queries_with_positive_in_corpus = 0
    for row in rows:
        qrels = qrels_for_row(row)
        in_count = sum(1 for doc_id in qrels if doc_id in doc_id_set)
        qrels_total += len(qrels)
        qrels_in_corpus += in_count
        queries_with_positive_in_corpus += int(in_count > 0)

    tagged_docs = [tagged_doc(doc_id, text) for doc_id, text in zip(doc_ids, doc_texts)]
    text_to_doc_id = {text: doc_id for doc_id, text in zip(doc_ids, tagged_docs)}
    text_to_doc_id.update({text: doc_id for doc_id, text in zip(doc_ids, doc_texts)})

    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.save_dir.mkdir(parents=True, exist_ok=True)

    config = BaseConfig()
    config.save_dir = str(args.save_dir)
    config.llm_name = str(args.llm_model_name)
    config.llm_base_url = args.llm_base_url
    config.embedding_model_name = str(args.embedding_model_name)
    config.embedding_base_url = args.embedding_base_url
    config.embedding_batch_size = int(args.embedding_batch_size)
    config.retrieval_top_k = int(retrieve_topk)
    config.linking_top_k = int(args.linking_top_k)
    config.force_index_from_scratch = bool(args.force_index_from_scratch)
    config.force_openie_from_scratch = bool(args.force_openie_from_scratch)
    config.save_openie = not bool(args.no_save_openie)
    if hasattr(config, "enable_ontology"):
        config.enable_ontology = bool(args.enable_ontology)
    if hasattr(config, "enable_query_plan_ontology"):
        config.enable_query_plan_ontology = bool(args.enable_query_plan_ontology)

    print(
        "[stage] HippoRAG2 "
        f"queries={len(rows)} corpus={len(corpus)} retrieve_topk={retrieve_topk} "
        f"qrels_coverage={qrels_in_corpus}/{qrels_total} "
        f"save_dir={args.save_dir}",
        flush=True,
    )
    hipporag = HippoRAG(global_config=config)

    start = time.perf_counter()
    if not args.skip_index:
        print("[stage] indexing HippoRAG2 corpus", flush=True)
        hipporag.index(docs=tagged_docs)
        print(f"[stage] indexing done elapsed={time.perf_counter() - start:.1f}s", flush=True)
    else:
        print("[stage] skip index; using existing HippoRAG2 index in save_dir", flush=True)

    print("[stage] retrieving", flush=True)
    retrieval_results = hipporag.retrieve(queries=q_texts, num_to_retrieve=retrieve_topk)

    preds: list[list[str]] = []
    rankings_debug_f = None
    if args.save_rankings_jsonl is not None:
        args.save_rankings_jsonl.parent.mkdir(parents=True, exist_ok=True)
        rankings_debug_f = args.save_rankings_jsonl.open("w", encoding="utf-8")
    try:
        for idx, result in enumerate(retrieval_results):
            pred_ids: list[str] = []
            seen: set[str] = set()
            scores = result.doc_scores.tolist() if getattr(result, "doc_scores", None) is not None else []
            for doc_text in result.docs:
                doc_id = parse_doc_id(str(doc_text), text_to_doc_id)
                if doc_id is None or doc_id in seen:
                    continue
                pred_ids.append(doc_id)
                seen.add(doc_id)
            preds.append(pred_ids)
            if rankings_debug_f is not None:
                rankings_debug_f.write(
                    json.dumps(
                        {
                            "query_index": idx,
                            "query_id": q_ids[idx],
                            "query": q_texts[idx],
                            "rankings": {"hipporag2": pred_ids[:retrieve_topk]},
                            "scores": scores[:retrieve_topk],
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )
    finally:
        if rankings_debug_f is not None:
            rankings_debug_f.close()

    summary, detail = method_metrics(rows, preds, ks)
    metric_keys: list[str] = []
    for metric_name in ["ndcg", "map", "hits"]:
        for k in ks:
            metric_keys.append(f"{metric_name}@{k}")
    metric_keys.extend([f"any_hit@{k}" for k in ks])

    table_rows = [{"method": "hipporag2", "label": "HippoRAG2", **summary}]
    out = {
        "meta": {
            "queries": len(rows),
            "corpus": len(corpus),
            "split_file": str(args.split_file),
            "corpus_file": str(args.corpus_file),
            "max_corpus": int(args.max_corpus),
            "filter_queries_with_positive_in_corpus": bool(args.filter_queries_with_positive_in_corpus),
            "hipporag2_root": str(hipporag2_root),
            "hipporag2_save_dir": str(args.save_dir),
            "llm_model_name": str(args.llm_model_name),
            "llm_base_url": args.llm_base_url,
            "embedding_model_name": str(args.embedding_model_name),
            "embedding_base_url": args.embedding_base_url,
            "embedding_batch_size": int(args.embedding_batch_size),
            "retrieve_topk": int(retrieve_topk),
            "metrics_k": ks,
            "force_index_from_scratch": bool(args.force_index_from_scratch),
            "force_openie_from_scratch": bool(args.force_openie_from_scratch),
            "skip_index": bool(args.skip_index),
            "qrels_positive_total": int(qrels_total),
            "qrels_positive_in_corpus": int(qrels_in_corpus),
            "qrels_coverage_in_corpus": float(qrels_in_corpus / max(1, qrels_total)),
            "queries_with_positive_in_corpus": int(queries_with_positive_in_corpus),
            "hits_definition": "average count of relevant chunks in top-k; can exceed 1.0",
            "map_definition": "truncated AP@k divided by total positive qrels for the query",
            "ndcg_definition": "graded NDCG using qrel labels >0 and gains 2^label-1",
        },
        "methods": {"hipporag2": summary},
    }

    args.out_json.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    write_csv(args.out_csv, table_rows, metric_keys)
    write_markdown(args.out_md, table_rows, metric_keys)
    if args.detail_json is not None:
        args.detail_json.parent.mkdir(parents=True, exist_ok=True)
        details = {"query_ids": q_ids, "methods": {"hipporag2": detail}}
        args.detail_json.write_text(json.dumps(details, ensure_ascii=False, indent=2), encoding="utf-8")

    print(json.dumps(out, ensure_ascii=False, indent=2), flush=True)
    print(f"[OK] json -> {args.out_json}", flush=True)
    print(f"[OK] csv  -> {args.out_csv}", flush=True)
    print(f"[OK] md   -> {args.out_md}", flush=True)
    print(f"[DONE] HippoRAG2 retrieval metrics -> {args.out_json.parent}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
