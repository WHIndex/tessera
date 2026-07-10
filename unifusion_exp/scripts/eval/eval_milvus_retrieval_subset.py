#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib
import json
from pathlib import Path
import sys

import numpy as np
from pymilvus import MilvusClient

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

e5_mod = importlib.import_module("unifusion_exp.utils.e5_embed")
encode_texts = e5_mod.encode_texts
load_e5 = e5_mod.load_e5


def positive_relevant_ids(row: dict) -> set[str]:
    rel = set()
    for chunk_id, label in row.get("relevant_chunks", {}).items():
        try:
            if float(label) > 0:
                rel.add(chunk_id)
        except Exception:
            continue
    return rel


def evaluate_at_k(rows, preds, k, id_set):
    any_hit = []
    recall = []
    precision = []
    rel_count = []
    rel_in_corpus_count = []
    for r, p in zip(rows, preds):
        rel = positive_relevant_ids(r)
        topk = set(p[:k])
        rel_count.append(len(rel))
        rel_in = len(rel & id_set)
        rel_in_corpus_count.append(rel_in)
        inter = len(rel & topk)
        any_hit.append(1 if inter > 0 else 0)
        recall.append(inter / max(1, rel_in))
        precision.append(inter / k)
    return any_hit, recall, precision, rel_count, rel_in_corpus_count


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate retrieval using Milvus search")
    parser.add_argument("--milvus-uri", type=str, default="http://127.0.0.1:19530")
    parser.add_argument("--collection", type=str, required=True)
    parser.add_argument("--model-dir", type=str, required=True)
    parser.add_argument("--split-file", type=Path, required=True)
    parser.add_argument("--out-file", type=Path, required=True)
    parser.add_argument("--detail-file", type=Path, default=None)
    parser.add_argument("--max-queries", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--search-k", type=int, default=20)
    args = parser.parse_args()

    rows = json.loads(args.split_file.read_text(encoding="utf-8"))[: args.max_queries]
    q_texts = [r.get("query", "") for r in rows]

    tokenizer, model, device, resolved = load_e5(args.model_dir)
    print(f"[stage] encode queries={len(q_texts)} model={resolved} device={device}")
    qv = encode_texts(q_texts, tokenizer, model, device, batch_size=args.batch_size)

    client = MilvusClient(uri=args.milvus_uri)
    preds = []
    pred_id_set = set()
    for i in range(len(qv)):
        ans = client.search(
            collection_name=args.collection,
            data=[qv[i].tolist()],
            limit=args.search_k,
            output_fields=["doc_id"],
            search_params={"metric_type": "COSINE", "params": {"nprobe": 16}},
        )[0]
        pred_ids = []
        for hit in ans:
            ent = hit.get("entity", {})
            if isinstance(ent, dict) and "doc_id" in ent:
                pred_ids.append(ent["doc_id"])
                pred_id_set.add(ent["doc_id"])
        preds.append(pred_ids)
        if i % 50 == 0:
            print(f"[search] {i+1}/{len(qv)}")

    any5, rec5, pre5, rel_count, rel_in_count = evaluate_at_k(rows, preds, 5, pred_id_set)
    any10, rec10, pre10, _, _ = evaluate_at_k(rows, preds, 10, pred_id_set)
    any20, rec20, pre20, _, _ = evaluate_at_k(rows, preds, 20, pred_id_set)

    metrics = {
        "queries": len(rows),
        "collection": args.collection,
        "avg_positive_qrels": float(np.mean(rel_count)) if rel_count else 0.0,
        "qrels_coverage_in_results": float(np.sum(rel_in_count) / max(1, np.sum(rel_count))),
        "queries_without_positive_qrels": int(sum(1 for x in rel_count if x == 0)),
        "any_hit@5": float(np.mean(any5)) if any5 else 0.0,
        "any_hit@10": float(np.mean(any10)) if any10 else 0.0,
        "any_hit@20": float(np.mean(any20)) if any20 else 0.0,
        "recall@5": float(np.mean(rec5)) if rec5 else 0.0,
        "recall@10": float(np.mean(rec10)) if rec10 else 0.0,
        "recall@20": float(np.mean(rec20)) if rec20 else 0.0,
        "precision@5": float(np.mean(pre5)) if pre5 else 0.0,
        "precision@10": float(np.mean(pre10)) if pre10 else 0.0,
        "precision@20": float(np.mean(pre20)) if pre20 else 0.0,
    }
    args.out_file.parent.mkdir(parents=True, exist_ok=True)
    args.out_file.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")

    if args.detail_file is not None:
        detail = {
            "queries": len(rows),
            "query_ids": [r.get("id", f"q_{i}") for i, r in enumerate(rows)],
            "rel_count": rel_count,
            "rel_in_results_count": rel_in_count,
            "any_hit@5": any5,
            "any_hit@10": any10,
            "any_hit@20": any20,
            "hit@5": any5,
            "hit@10": any10,
            "hit@20": any20,
            "recall@5": rec5,
            "recall@10": rec10,
            "recall@20": rec20,
            "precision@5": pre5,
            "precision@10": pre10,
            "precision@20": pre20,
        }
        args.detail_file.parent.mkdir(parents=True, exist_ok=True)
        args.detail_file.write_text(json.dumps(detail, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"[OK] detail -> {args.detail_file}")

    print(json.dumps(metrics, ensure_ascii=False, indent=2))
    print(f"[OK] saved -> {args.out_file}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
