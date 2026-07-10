#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import sys
import time
from pathlib import Path

import numpy as np
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
if str(Path(__file__).resolve().parent) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parent))

import run_e2e_table1c as e2e  # noqa: E402


DEFAULT_METHODS = [
    "dense_concat",
    "naive_rag",
    "carp",
    "tablerag",
    "quasar",
    "tessera_rag",
]

METHOD_LABELS = {
    "dense_concat": "Dense-Concat",
    "naive_rag": "NaiveRAG",
    "carp": "CARP-Adapter",
    "tablerag": "TableRAG-Adapter",
    "quasar": "QUASAR-Adapter",
    "unihgkr_dense": "UniHGKR-Adapter",
    "tessera_rag": "TESSERA",
    "ablation_no_redundancy_e2e": "TESSERA w/o RP",
    "ablation_no_pathmaxsim_e2e": "TESSERA w/o PathHint",
}


def parse_methods(raw: str, include_unihgkr: bool) -> list[str]:
    if raw.strip():
        methods = [x.strip() for x in raw.split(",") if x.strip()]
    else:
        methods = list(DEFAULT_METHODS)
        if include_unihgkr:
            methods.insert(-1, "unihgkr_dense")
    allowed = set(METHOD_LABELS)
    unknown = [m for m in methods if m not in allowed]
    if unknown:
        raise ValueError(f"Unknown methods: {unknown}. Allowed: {sorted(allowed)}")
    return methods


def qrels_for_row(row: dict) -> dict[str, float]:
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


def method_metrics(rows: list[dict], preds: list[list[str]], ks: list[int]) -> tuple[dict, dict]:
    buckets: dict[str, list[float]] = {}
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


def make_cache_key(ids: list[str], max_items: int = 2048) -> str:
    if not ids:
        return "empty"
    if len(ids) <= max_items:
        sampled = ids
    else:
        step = max(1, len(ids) // max_items)
        sampled = ids[::step][:max_items]
    payload = "|".join(sampled) + f"|n={len(ids)}"
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()[:12]


def build_tfidf_scores(corpus_texts: list[str], query_texts: list[str], max_features: int) -> np.ndarray:
    vec = TfidfVectorizer(stop_words="english", ngram_range=(1, 2), max_features=max_features, min_df=2)
    c_mat = vec.fit_transform(corpus_texts)
    q_mat = vec.transform(query_texts)
    return (q_mat @ c_mat.T).toarray().astype(np.float32)


def build_bm25_scores(
    corpus_texts: list[str],
    query_texts: list[str],
    max_features: int,
    k1: float = 1.2,
    b: float = 0.75,
) -> np.ndarray:
    vec = CountVectorizer(stop_words="english", ngram_range=(1, 2), max_features=max_features, min_df=2)
    c_mat = vec.fit_transform(corpus_texts).astype(np.float32).tocsr()
    n_docs = c_mat.shape[0]
    doc_len = np.asarray(c_mat.sum(axis=1), dtype=np.float32).reshape(-1)
    avgdl = float(np.mean(doc_len)) if doc_len.size else 0.0
    avgdl = max(avgdl, 1e-6)
    df = np.asarray((c_mat > 0).sum(axis=0), dtype=np.float32).reshape(-1)
    idf = np.log((n_docs - df + 0.5) / (df + 0.5) + 1.0).astype(np.float32)

    indptr = c_mat.indptr
    indices = c_mat.indices
    data = c_mat.data
    norm = k1 * (1.0 - b + b * doc_len / avgdl)
    for row_idx in range(n_docs):
        start, end = indptr[row_idx], indptr[row_idx + 1]
        if start == end:
            continue
        tf = data[start:end]
        data[start:end] = idf[indices[start:end]] * (tf * (k1 + 1.0)) / (tf + norm[row_idx])

    q_mat = vec.transform(query_texts).astype(np.float32)
    if q_mat.nnz:
        q_mat.data[:] = 1.0
    return (q_mat @ c_mat.T).toarray().astype(np.float32)


def load_or_build_sparse_scores(args, q_texts: list[str], doc_texts: list[str], q_key: str, c_key: str) -> np.ndarray:
    cache = args.cache_dir / (
        f"{args.sparse_backend}_scores_{len(q_texts)}x{len(doc_texts)}_"
        f"mf{args.sparse_max_features}_{q_key}_{c_key}.npy"
    )
    if cache.exists():
        arr = np.load(cache)
        if arr.shape == (len(q_texts), len(doc_texts)):
            print(f"[cache] sparse scores -> {cache}")
            return arr

    print(f"[stage] building {args.sparse_backend} sparse scores")
    if args.sparse_backend == "bm25":
        arr = build_bm25_scores(doc_texts, q_texts, max_features=args.sparse_max_features)
    else:
        arr = build_tfidf_scores(doc_texts, q_texts, max_features=args.sparse_max_features)
    np.save(cache, arr)
    return arr


def write_csv(path: Path, rows: list[dict], metric_keys: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["method", "label", *metric_keys])
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in ["method", "label", *metric_keys]})


def write_markdown(path: Path, rows: list[dict], metric_keys: list[str]) -> None:
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


def main() -> int:
    parser = argparse.ArgumentParser(description="Run paper-style retrieval metrics for TESSERA and baselines")
    parser.add_argument("--model-dir", type=str, required=True)
    parser.add_argument("--split-file", type=Path, required=True)
    parser.add_argument("--corpus-file", type=Path, required=True)
    parser.add_argument("--out-json", type=Path, required=True)
    parser.add_argument("--out-csv", type=Path, required=True)
    parser.add_argument("--out-md", type=Path, required=True)
    parser.add_argument("--detail-json", type=Path, default=None)
    parser.add_argument("--save-rankings-jsonl", type=Path, default=None)
    parser.add_argument("--cache-dir", type=Path, default=Path("artifacts/retrieval"))
    parser.add_argument("--max-queries", type=int, default=1286)
    parser.add_argument("--retrieve-topk", type=int, default=10)
    parser.add_argument("--metrics-k", type=str, default="1,3,5")
    parser.add_argument("--methods", type=str, default="")
    parser.add_argument("--include-unihgkr", action="store_true")
    parser.add_argument("--unihgkr-model-dir", type=Path, default=ROOT.parent / "downloaded_resource/compmix-ir-benchmarks/ZhishanQ-UniHGKR-base")
    parser.add_argument("--unihgkr-batch-size", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--sparse-backend", choices=["bm25", "tfidf"], default="bm25")
    parser.add_argument("--sparse-max-features", type=int, default=200000)
    parser.add_argument("--router-metrics", type=Path, default=ROOT / "runs/router_deberta_full_v1_router_deberta/router_metrics/router_deberta_full_metrics.json")
    parser.add_argument("--router-model", type=Path, default=None)
    parser.add_argument("--router-threshold", type=float, default=0.5)
    parser.add_argument("--router-batch-size", type=int, default=64)
    parser.add_argument("--allow-heuristic-router-fallback", action="store_true")
    parser.add_argument("--preserve-dense-top", type=int, default=0)
    parser.add_argument("--tessera-late-alpha", type=float, default=0.08)
    parser.add_argument("--query-modality-prior-mix", type=float, default=0.35)
    parser.add_argument("--routing-uncertainty-threshold", type=float, default=0.75)
    parser.add_argument("--pathmaxsim-weight", type=float, default=0.14)
    parser.add_argument("--pathmaxsim-kg-threshold", type=float, default=0.0)
    parser.add_argument("--table-cellmaxsim-weight", type=float, default=0.0)
    parser.add_argument("--table-cellmaxsim-top-cells", type=int, default=160)
    parser.add_argument("--adapter-plus-mode", action="store_true")
    parser.add_argument("--adapter-official-lite", action="store_true")
    parser.add_argument("--qa-objective-retrieval-weight", type=float, default=0.04)
    parser.add_argument("--qa-objective-targeted-only", action="store_true")
    parser.add_argument("--tessera-candidate-pool-k", type=int, default=80)
    parser.add_argument("--tessera-retrieval-multi-agent", action="store_true")
    parser.add_argument("--tessera-retrieval-agent-pool-k", type=int, default=120)
    parser.add_argument("--tessera-retrieval-dense-pool-k", type=int, default=80)
    parser.add_argument("--tessera-retrieval-sparse-pool-k", type=int, default=80)
    parser.add_argument("--tessera-retrieval-preserve-top", type=int, default=1)
    parser.add_argument("--tessera-retrieval-base-weight", type=float, default=0.50)
    parser.add_argument("--tessera-retrieval-dense-weight", type=float, default=0.16)
    parser.add_argument("--tessera-retrieval-sparse-weight", type=float, default=0.10)
    parser.add_argument("--tessera-retrieval-target-weight", type=float, default=0.10)
    parser.add_argument("--tessera-retrieval-coverage-weight", type=float, default=0.06)
    parser.add_argument("--tessera-retrieval-diversity-weight", type=float, default=0.04)
    parser.add_argument("--tessera-retrieval-dense-rescue-k", type=int, default=0)
    parser.add_argument("--tessera-retrieval-dense-rescue-pool-k", type=int, default=12)
    parser.add_argument("--tessera-retrieval-sibling-seed-k", type=int, default=0)
    parser.add_argument("--tessera-retrieval-sibling-window", type=int, default=0)
    parser.add_argument("--tessera-retrieval-sibling-weight", type=float, default=0.0)
    parser.add_argument("--tessera-retrieval-moe", action="store_true")
    parser.add_argument("--tessera-moe-pool-k", type=int, default=260)
    parser.add_argument("--tessera-moe-prf-seed-k", type=int, default=6)
    parser.add_argument("--tessera-moe-prf-dense-seed-k", type=int, default=6)
    parser.add_argument("--tessera-moe-prf-sparse-seed-k", type=int, default=6)
    parser.add_argument("--tessera-moe-prf-max-terms", type=int, default=48)
    parser.add_argument("--tessera-moe-sibling-seed-k", type=int, default=6)
    parser.add_argument("--tessera-moe-sibling-window", type=int, default=1)
    parser.add_argument("--tessera-moe-sibling-weight", type=float, default=0.03)
    parser.add_argument("--tessera-ser-ranker", type=Path, default=None)
    parser.add_argument("--tessera-ser-candidate-pool-k", type=int, default=180)
    parser.add_argument("--tessera-ser-dense-pool-k", type=int, default=120)
    parser.add_argument("--tessera-ser-sparse-pool-k", type=int, default=120)
    parser.add_argument("--tessera-ser-preserve-top", type=int, default=1)
    parser.add_argument("--tessera-ser-blend-weight", type=float, default=0.65)
    parser.add_argument("--tessera-ser-diversity-weight", type=float, default=0.02)
    parser.add_argument("--tessera-ser-evidence-rescue-k", type=int, default=0)
    parser.add_argument("--tessera-ser-evidence-rescue-pool-k", type=int, default=24)
    parser.add_argument("--tessera-ser-evidence-preserve-top", type=int, default=3)
    parser.add_argument("--tessera-ser-evidence-redundancy-weight", type=float, default=0.04)
    parser.add_argument("--tessera-ser-evidence-min-gain", type=float, default=0.03)
    parser.add_argument("--tessera-ser-plan-adaptive", action="store_true")
    parser.add_argument("--tessera-ser-plan-dense-weight", type=float, default=0.12)
    parser.add_argument("--tessera-ser-plan-sparse-weight", type=float, default=0.03)
    parser.add_argument("--tessera-ser-plan-lexical-weight", type=float, default=0.03)
    parser.add_argument("--tessera-ser-plan-slot-weight", type=float, default=0.04)
    parser.add_argument("--tessera-ser-evidence-set-selection", action="store_true")
    parser.add_argument("--tessera-ser-evidence-set-preserve-top", type=int, default=2)
    parser.add_argument("--tessera-ser-evidence-set-pool-k", type=int, default=220)
    parser.add_argument("--tessera-ser-evidence-set-cardinality-threshold", type=float, default=0.46)
    parser.add_argument("--tessera-ser-evidence-set-learned-weight", type=float, default=0.22)
    parser.add_argument("--tessera-ser-evidence-set-base-weight", type=float, default=0.30)
    parser.add_argument("--tessera-ser-evidence-set-dense-weight", type=float, default=0.18)
    parser.add_argument("--tessera-ser-evidence-set-sparse-weight", type=float, default=0.10)
    parser.add_argument("--tessera-ser-evidence-set-probe-weight", type=float, default=0.10)
    parser.add_argument("--tessera-ser-evidence-set-slot-weight", type=float, default=0.12)
    parser.add_argument("--tessera-ser-evidence-set-anchor-weight", type=float, default=0.16)
    parser.add_argument("--tessera-ser-evidence-set-family-weight", type=float, default=0.05)
    parser.add_argument("--tessera-ser-evidence-set-redundancy-weight", type=float, default=0.018)
    parser.add_argument("--tessera-graph-evidence-expansion", action="store_true")
    parser.add_argument("--tessera-gee-post-rerank", action="store_true")
    parser.add_argument("--tessera-gee-candidate-pool-k", type=int, default=420)
    parser.add_argument("--tessera-gee-dense-pool-k", type=int, default=260)
    parser.add_argument("--tessera-gee-sparse-pool-k", type=int, default=220)
    parser.add_argument("--tessera-gee-graph-seed-k", type=int, default=18)
    parser.add_argument("--tessera-gee-graph-window", type=int, default=1)
    parser.add_argument("--tessera-gee-preserve-top", type=int, default=2)
    parser.add_argument("--tessera-gee-trigger-threshold", type=float, default=0.58)
    parser.add_argument("--tessera-gee-base-weight", type=float, default=0.28)
    parser.add_argument("--tessera-gee-dense-weight", type=float, default=0.34)
    parser.add_argument("--tessera-gee-sparse-weight", type=float, default=0.14)
    parser.add_argument("--tessera-gee-probe-weight", type=float, default=0.12)
    parser.add_argument("--tessera-gee-graph-weight", type=float, default=0.06)
    parser.add_argument("--tessera-gee-slot-weight", type=float, default=0.06)
    parser.add_argument("--tessera-gee-sibling-weight", type=float, default=0.04)
    parser.add_argument("--tessera-gee-redundancy-weight", type=float, default=0.012)
    parser.add_argument("--tessera-v9", action="store_true")
    parser.add_argument("--tessera-v9-local-rerank", action="store_true")
    parser.add_argument("--tessera-v9-dense-pool-k", type=int, default=1200)
    parser.add_argument("--tessera-v9-sparse-pool-k", type=int, default=1800)
    parser.add_argument("--tessera-v9-candidate-pool-k", type=int, default=900)
    parser.add_argument("--tessera-v9-graph-seed-k", type=int, default=36)
    parser.add_argument("--tessera-v9-graph-window", type=int, default=1)
    parser.add_argument("--tessera-v9-preserve-top", type=int, default=0)
    parser.add_argument("--tessera-v9-base-weight", type=float, default=0.28)
    parser.add_argument("--tessera-v9-dense-weight", type=float, default=0.30)
    parser.add_argument("--tessera-v9-sparse-weight", type=float, default=0.20)
    parser.add_argument("--tessera-v9-probe-weight", type=float, default=0.16)
    parser.add_argument("--tessera-v9-graph-weight", type=float, default=0.08)
    parser.add_argument("--tessera-v9-slot-weight", type=float, default=0.08)
    parser.add_argument("--tessera-v9-diversity-weight", type=float, default=0.018)
    parser.add_argument("--tessera-v9-modality-weight", type=float, default=0.04)
    parser.add_argument("--tessera-v10-conservative-rerank", action="store_true")
    parser.add_argument("--tessera-v10-preserve-top", type=int, default=1)
    parser.add_argument("--tessera-v10-direct-preserve-top", type=int, default=2)
    parser.add_argument("--tessera-v10-reference-pool-k", type=int, default=40)
    parser.add_argument("--tessera-v10-candidate-pool-k", type=int, default=120)
    parser.add_argument("--tessera-v10-reference-weight", type=float, default=0.54)
    parser.add_argument("--tessera-v10-current-weight", type=float, default=0.24)
    parser.add_argument("--tessera-v10-base-weight", type=float, default=0.10)
    parser.add_argument("--tessera-v10-dense-weight", type=float, default=0.07)
    parser.add_argument("--tessera-v10-sparse-weight", type=float, default=0.04)
    parser.add_argument("--tessera-v10-probe-weight", type=float, default=0.04)
    parser.add_argument("--tessera-v10-slot-weight", type=float, default=0.03)
    parser.add_argument("--tessera-v10-diversity-weight", type=float, default=0.012)
    parser.add_argument("--tessera-v10-margin", type=float, default=0.035)
    parser.add_argument("--tessera-v10-relevance-floor", type=float, default=0.18)
    parser.add_argument("--tessera-source-evidence-fusion", action="store_true")
    parser.add_argument("--tessera-source-evidence-topk", type=int, default=5)
    parser.add_argument("--tessera-source-evidence-candidate-pool-k", type=int, default=80)
    parser.add_argument("--tessera-source-evidence-preserve-top", type=int, default=1)
    parser.add_argument("--tessera-source-evidence-base-weight", type=float, default=0.34)
    parser.add_argument("--tessera-source-evidence-dense-weight", type=float, default=0.16)
    parser.add_argument("--tessera-source-evidence-sparse-weight", type=float, default=0.08)
    parser.add_argument("--tessera-source-evidence-reference-weight", type=float, default=0.14)
    parser.add_argument("--tessera-source-evidence-lexical-weight", type=float, default=0.10)
    parser.add_argument("--tessera-source-evidence-modality-prior-weight", type=float, default=0.12)
    parser.add_argument("--tessera-source-evidence-source-balance-weight", type=float, default=0.10)
    parser.add_argument("--tessera-source-evidence-target-family-weight", type=float, default=0.08)
    parser.add_argument("--tessera-source-evidence-diversity-weight", type=float, default=0.025)
    parser.add_argument("--tessera-source-evidence-replacement-margin", type=float, default=0.01)
    parser.add_argument("--tessera-source-evidence-min-candidate-score", type=float, default=0.08)
    parser.add_argument("--tessera-source-evidence-dense-guard", action="store_true")
    parser.add_argument("--tessera-source-evidence-dense-guard-topn", type=int, default=5)
    parser.add_argument("--tessera-source-evidence-dense-guard-prefixes", type=str, default="")
    parser.add_argument("--tessera-source-evidence-dense-guard-weight", type=float, default=0.22)
    parser.add_argument("--tessera-source-evidence-dense-rank-weight", type=float, default=0.10)
    parser.add_argument("--tessera-source-evidence-current-rank-weight", type=float, default=0.06)
    parser.add_argument("--tessera-source-evidence-source-balance-prefixes", type=str, default="")
    parser.add_argument("--tessera-source-evidence-max-changed-slots", type=int, default=0)
    parser.add_argument("--tessera-source-evidence-slot-acceptance-guard", action="store_true")
    parser.add_argument("--tessera-source-evidence-slot-acceptance-prefixes", type=str, default="")
    parser.add_argument("--tessera-source-evidence-slot-acceptance-margin", type=float, default=0.02)
    parser.add_argument("--tessera-source-evidence-budget-composer", action="store_true")
    parser.add_argument("--tessera-source-evidence-budget-prefixes", type=str, default="")
    parser.add_argument("--tessera-source-evidence-budget-candidate-pool-k", type=int, default=180)
    parser.add_argument("--tessera-source-evidence-budget-start-slot", type=int, default=4)
    parser.add_argument("--tessera-source-evidence-budget-max-selected", type=int, default=2)
    parser.add_argument("--tessera-source-evidence-budget-score-weight", type=float, default=0.10)
    parser.add_argument("--tessera-source-evidence-budget-sibling-weight", type=float, default=0.16)
    parser.add_argument("--tessera-source-evidence-budget-source-quota-weight", type=float, default=0.08)
    parser.add_argument("--tessera-source-evidence-budget-tail-rank-weight", type=float, default=0.08)
    parser.add_argument("--tessera-source-evidence-budget-reference-weight", type=float, default=0.10)
    parser.add_argument("--tessera-source-evidence-budget-margin", type=float, default=0.006)
    parser.add_argument("--tessera-source-evidence-budget-redundancy-weight", type=float, default=0.01)
    parser.add_argument("--tessera-source-evidence-sibling-filler", action="store_true")
    parser.add_argument("--tessera-source-evidence-sibling-filler-prefixes", type=str, default="")
    parser.add_argument("--tessera-source-evidence-sibling-filler-candidate-pool-k", type=int, default=120)
    parser.add_argument("--tessera-source-evidence-sibling-filler-start-slot", type=int, default=4)
    parser.add_argument("--tessera-source-evidence-sibling-filler-max-selected", type=int, default=1)
    parser.add_argument("--tessera-source-evidence-sibling-filler-tail-topn", type=int, default=10)
    parser.add_argument("--tessera-source-evidence-sibling-filler-reference-topn", type=int, default=10)
    parser.add_argument("--tessera-source-evidence-sibling-filler-margin", type=float, default=0.02)
    parser.add_argument("--tessera-source-evidence-sibling-filler-sibling-weight", type=float, default=0.22)
    parser.add_argument("--tessera-source-evidence-sibling-filler-reference-weight", type=float, default=0.18)
    parser.add_argument("--tessera-source-evidence-sibling-filler-tail-weight", type=float, default=0.10)
    parser.add_argument("--tessera-source-evidence-sibling-filler-dense-weight", type=float, default=0.08)
    parser.add_argument("--tessera-source-evidence-sibling-filler-source-weight", type=float, default=0.08)
    parser.add_argument("--tessera-source-evidence-sibling-filler-redundancy-weight", type=float, default=0.008)
    parser.add_argument("--tessera-source-evidence-slot-verifier", action="store_true")
    parser.add_argument("--tessera-source-evidence-slot-verifier-prefixes", type=str, default="")
    parser.add_argument("--tessera-source-evidence-slot-verifier-candidate-pool-k", type=int, default=220)
    parser.add_argument("--tessera-source-evidence-slot-verifier-start-slot", type=int, default=4)
    parser.add_argument("--tessera-source-evidence-slot-verifier-max-selected", type=int, default=2)
    parser.add_argument("--tessera-source-evidence-slot-verifier-tail-topn", type=int, default=12)
    parser.add_argument("--tessera-source-evidence-slot-verifier-reference-topn", type=int, default=12)
    parser.add_argument("--tessera-source-evidence-slot-verifier-dense-topn", type=int, default=24)
    parser.add_argument("--tessera-source-evidence-slot-verifier-margin", type=float, default=0.025)
    parser.add_argument("--tessera-source-evidence-slot-verifier-min-score", type=float, default=0.42)
    parser.add_argument("--tessera-source-evidence-slot-verifier-model", type=Path, default=None)
    parser.add_argument("--tessera-source-evidence-slot-verifier-model-threshold", type=float, default=0.68)
    parser.add_argument("--tessera-source-evidence-slot-verifier-static-weight", type=float, default=0.20)
    parser.add_argument("--tessera-source-evidence-slot-verifier-reference-weight", type=float, default=0.20)
    parser.add_argument("--tessera-source-evidence-slot-verifier-dense-weight", type=float, default=0.14)
    parser.add_argument("--tessera-source-evidence-slot-verifier-tail-weight", type=float, default=0.12)
    parser.add_argument("--tessera-source-evidence-slot-verifier-sibling-weight", type=float, default=0.12)
    parser.add_argument("--tessera-source-evidence-slot-verifier-source-weight", type=float, default=0.10)
    parser.add_argument("--tessera-source-evidence-slot-verifier-lexical-weight", type=float, default=0.08)
    parser.add_argument("--tessera-source-evidence-slot-verifier-family-weight", type=float, default=0.06)
    parser.add_argument("--tessera-source-evidence-slot-verifier-redundancy-weight", type=float, default=0.012)
    parser.add_argument("--tessera-source-evidence-kg-preservation-guard", action="store_true")
    parser.add_argument("--tessera-source-evidence-kg-preservation-prefixes", type=str, default="cwq,webqsp")
    parser.add_argument("--tessera-source-evidence-kg-preservation-min-kg", type=int, default=1)
    parser.add_argument("--tessera-source-evidence-kg-preservation-candidate-pool-k", type=int, default=160)
    parser.add_argument("--tessera-source-evidence-kg-preservation-start-slot", type=int, default=2)
    parser.add_argument("--tessera-source-evidence-kg-preservation-margin", type=float, default=0.015)
    parser.add_argument("--tessera-source-evidence-kg-preservation-reference-weight", type=float, default=0.24)
    parser.add_argument("--tessera-source-evidence-kg-preservation-dense-weight", type=float, default=0.16)
    parser.add_argument("--tessera-source-evidence-kg-preservation-current-weight", type=float, default=0.12)
    parser.add_argument("--tessera-source-evidence-kg-preservation-family-weight", type=float, default=0.10)
    parser.add_argument("--tessera-source-evidence-kg-preservation-lexical-weight", type=float, default=0.06)
    parser.add_argument("--tessera-source-evidence-kg-verifier-model", type=Path, default=None)
    parser.add_argument("--tessera-source-evidence-kg-verifier-weight", type=float, default=0.0)
    parser.add_argument("--tessera-source-evidence-kg-verifier-min-score", type=float, default=0.0)
    parser.add_argument("--tessera-source-evidence-kg-verify-existing", action="store_true")
    parser.add_argument("--tessera-source-evidence-kg-verify-existing-max-replacements", type=int, default=1)
    parser.add_argument("--tessera-source-budgeter-model", type=Path, default=None)
    parser.add_argument("--tessera-source-budgeter-top1-guard", action="store_true")
    parser.add_argument("--tessera-source-budgeter-need-threshold", type=float, default=0.45)
    parser.add_argument("--tessera-source-budgeter-non-kg-top1-max-kg", type=int, default=1)
    parser.add_argument("--tessera-source-head-selector", action="store_true")
    parser.add_argument("--tessera-source-head-topn", type=int, default=5)
    parser.add_argument("--tessera-source-head-source-weight", type=float, default=0.42)
    parser.add_argument("--tessera-source-head-same-query-weight", type=float, default=0.16)
    parser.add_argument("--tessera-source-head-position-weight", type=float, default=0.16)
    parser.add_argument("--tessera-source-head-reference-weight", type=float, default=0.12)
    parser.add_argument("--tessera-source-head-lexical-weight", type=float, default=0.10)
    parser.add_argument("--tessera-source-head-base-weight", type=float, default=0.08)
    parser.add_argument("--tessera-source-head-dense-weight", type=float, default=0.05)
    parser.add_argument("--tessera-source-head-sparse-weight", type=float, default=0.04)
    parser.add_argument("--tessera-source-head-margin", type=float, default=0.015)
    parser.add_argument("--tessera-source-head-off-source-margin", type=float, default=0.04)
    parser.add_argument("--tessera-source-action-policy-model", type=Path, default=None)
    parser.add_argument("--tessera-source-action-policy-min-prob", type=float, default=0.42)
    parser.add_argument("--tessera-source-action-policy-topk", type=int, default=5)
    parser.add_argument("--tessera-source-action-policy-pool-k", type=int, default=10)
    parser.add_argument("--tessera-final-evidence-composer", action="store_true")
    parser.add_argument("--tessera-final-evidence-topk", type=int, default=5)
    parser.add_argument("--tessera-final-evidence-candidate-pool-k", type=int, default=120)
    parser.add_argument("--tessera-final-evidence-dense-pool-k", type=int, default=80)
    parser.add_argument("--tessera-final-evidence-sparse-pool-k", type=int, default=80)
    parser.add_argument("--tessera-final-evidence-preserve-top", type=int, default=1)
    parser.add_argument("--tessera-final-evidence-max-replacements", type=int, default=1)
    parser.add_argument("--tessera-final-evidence-min-candidate-score", type=float, default=0.62)
    parser.add_argument("--tessera-final-evidence-replacement-margin", type=float, default=0.08)
    parser.add_argument("--tessera-final-evidence-min-query-overlap", type=float, default=0.0)
    parser.add_argument("--tessera-final-evidence-source-need-weight", type=float, default=0.035)
    parser.add_argument("--tessera-final-evidence-redundancy-weight", type=float, default=0.025)
    parser.add_argument("--tessera-final-evidence-verifier-model", type=Path, default=None)
    parser.add_argument("--tessera-final-evidence-verifier-threshold", type=float, default=0.70)
    parser.add_argument("--tessera-final-evidence-verifier-margin", type=float, default=0.0)
    args = parser.parse_args()

    if int(args.tessera_candidate_pool_k) < 0:
        raise ValueError("tessera-candidate-pool-k must be >= 0")
    if int(args.tessera_retrieval_agent_pool_k) < 1:
        raise ValueError("tessera-retrieval-agent-pool-k must be >= 1")
    if int(args.tessera_retrieval_dense_pool_k) < 1:
        raise ValueError("tessera-retrieval-dense-pool-k must be >= 1")
    if int(args.tessera_retrieval_sparse_pool_k) < 1:
        raise ValueError("tessera-retrieval-sparse-pool-k must be >= 1")
    if int(args.tessera_retrieval_preserve_top) < 0:
        raise ValueError("tessera-retrieval-preserve-top must be >= 0")
    if int(args.tessera_retrieval_dense_rescue_k) < 0:
        raise ValueError("tessera-retrieval-dense-rescue-k must be >= 0")
    if int(args.tessera_retrieval_dense_rescue_pool_k) < 1:
        raise ValueError("tessera-retrieval-dense-rescue-pool-k must be >= 1")
    if int(args.tessera_retrieval_sibling_seed_k) < 0:
        raise ValueError("tessera-retrieval-sibling-seed-k must be >= 0")
    if int(args.tessera_retrieval_sibling_window) < 0:
        raise ValueError("tessera-retrieval-sibling-window must be >= 0")
    if float(args.tessera_retrieval_sibling_weight) < 0.0:
        raise ValueError("tessera-retrieval-sibling-weight must be >= 0")
    if int(args.tessera_moe_pool_k) < 1:
        raise ValueError("tessera-moe-pool-k must be >= 1")
    if int(args.tessera_moe_prf_seed_k) < 0:
        raise ValueError("tessera-moe-prf-seed-k must be >= 0")
    if int(args.tessera_moe_prf_dense_seed_k) < 0:
        raise ValueError("tessera-moe-prf-dense-seed-k must be >= 0")
    if int(args.tessera_moe_prf_sparse_seed_k) < 0:
        raise ValueError("tessera-moe-prf-sparse-seed-k must be >= 0")
    if int(args.tessera_moe_prf_max_terms) < 1:
        raise ValueError("tessera-moe-prf-max-terms must be >= 1")
    if int(args.tessera_moe_sibling_seed_k) < 0:
        raise ValueError("tessera-moe-sibling-seed-k must be >= 0")
    if int(args.tessera_moe_sibling_window) < 0:
        raise ValueError("tessera-moe-sibling-window must be >= 0")
    if float(args.tessera_moe_sibling_weight) < 0.0:
        raise ValueError("tessera-moe-sibling-weight must be >= 0")
    if args.tessera_ser_ranker is not None and not Path(args.tessera_ser_ranker).exists():
        raise FileNotFoundError(f"tessera-ser-ranker not found: {args.tessera_ser_ranker}")
    if int(args.tessera_ser_candidate_pool_k) < 1:
        raise ValueError("tessera-ser-candidate-pool-k must be >= 1")
    if int(args.tessera_ser_dense_pool_k) < 1:
        raise ValueError("tessera-ser-dense-pool-k must be >= 1")
    if int(args.tessera_ser_sparse_pool_k) < 1:
        raise ValueError("tessera-ser-sparse-pool-k must be >= 1")
    if int(args.tessera_ser_preserve_top) < 0:
        raise ValueError("tessera-ser-preserve-top must be >= 0")
    if float(args.tessera_ser_blend_weight) < 0.0 or float(args.tessera_ser_blend_weight) > 1.0:
        raise ValueError("tessera-ser-blend-weight must be in [0, 1]")
    if float(args.tessera_ser_diversity_weight) < 0.0:
        raise ValueError("tessera-ser-diversity-weight must be >= 0")
    if int(args.tessera_ser_evidence_rescue_k) < 0:
        raise ValueError("tessera-ser-evidence-rescue-k must be >= 0")
    if int(args.tessera_ser_evidence_rescue_pool_k) < 1:
        raise ValueError("tessera-ser-evidence-rescue-pool-k must be >= 1")
    if int(args.tessera_ser_evidence_preserve_top) < 0:
        raise ValueError("tessera-ser-evidence-preserve-top must be >= 0")
    if float(args.tessera_ser_evidence_redundancy_weight) < 0.0:
        raise ValueError("tessera-ser-evidence-redundancy-weight must be >= 0")
    if float(args.tessera_ser_evidence_min_gain) < 0.0:
        raise ValueError("tessera-ser-evidence-min-gain must be >= 0")
    if int(args.tessera_ser_evidence_set_preserve_top) < 0:
        raise ValueError("tessera-ser-evidence-set-preserve-top must be >= 0")
    if int(args.tessera_ser_evidence_set_pool_k) < 1:
        raise ValueError("tessera-ser-evidence-set-pool-k must be >= 1")
    if (
        float(args.tessera_ser_evidence_set_cardinality_threshold) < 0.0
        or float(args.tessera_ser_evidence_set_cardinality_threshold) > 1.0
    ):
        raise ValueError("tessera-ser-evidence-set-cardinality-threshold must be in [0, 1]")
    for name in [
        "tessera_ser_plan_dense_weight",
        "tessera_ser_plan_sparse_weight",
        "tessera_ser_plan_lexical_weight",
        "tessera_ser_plan_slot_weight",
        "tessera_ser_evidence_set_learned_weight",
        "tessera_ser_evidence_set_base_weight",
        "tessera_ser_evidence_set_dense_weight",
        "tessera_ser_evidence_set_sparse_weight",
        "tessera_ser_evidence_set_probe_weight",
        "tessera_ser_evidence_set_slot_weight",
        "tessera_ser_evidence_set_anchor_weight",
        "tessera_ser_evidence_set_family_weight",
        "tessera_ser_evidence_set_redundancy_weight",
        "tessera_gee_base_weight",
        "tessera_gee_dense_weight",
        "tessera_gee_sparse_weight",
        "tessera_gee_probe_weight",
        "tessera_gee_graph_weight",
        "tessera_gee_slot_weight",
        "tessera_gee_sibling_weight",
        "tessera_gee_redundancy_weight",
        "tessera_v10_reference_weight",
        "tessera_v10_current_weight",
        "tessera_v10_base_weight",
        "tessera_v10_dense_weight",
        "tessera_v10_sparse_weight",
        "tessera_v10_probe_weight",
        "tessera_v10_slot_weight",
        "tessera_v10_diversity_weight",
        "tessera_source_evidence_base_weight",
        "tessera_source_evidence_dense_weight",
        "tessera_source_evidence_sparse_weight",
        "tessera_source_evidence_reference_weight",
        "tessera_source_evidence_lexical_weight",
        "tessera_source_evidence_modality_prior_weight",
        "tessera_source_evidence_source_balance_weight",
        "tessera_source_evidence_target_family_weight",
        "tessera_source_evidence_diversity_weight",
        "tessera_source_evidence_dense_guard_weight",
        "tessera_source_evidence_dense_rank_weight",
        "tessera_source_evidence_current_rank_weight",
        "tessera_source_evidence_slot_verifier_static_weight",
        "tessera_source_evidence_slot_verifier_reference_weight",
        "tessera_source_evidence_slot_verifier_dense_weight",
        "tessera_source_evidence_slot_verifier_tail_weight",
        "tessera_source_evidence_slot_verifier_sibling_weight",
        "tessera_source_evidence_slot_verifier_source_weight",
        "tessera_source_evidence_slot_verifier_lexical_weight",
        "tessera_source_evidence_slot_verifier_family_weight",
        "tessera_source_head_source_weight",
        "tessera_source_head_same_query_weight",
        "tessera_source_head_position_weight",
        "tessera_source_head_reference_weight",
        "tessera_source_head_lexical_weight",
        "tessera_source_head_base_weight",
        "tessera_source_head_dense_weight",
        "tessera_source_head_sparse_weight",
    ]:
        if float(getattr(args, name)) < 0.0:
            raise ValueError(f"{name.replace('_', '-')} must be >= 0")
    for name in [
        "tessera_gee_candidate_pool_k",
        "tessera_gee_dense_pool_k",
        "tessera_gee_sparse_pool_k",
        "tessera_gee_graph_seed_k",
    ]:
        if int(getattr(args, name)) < 1:
            raise ValueError(f"{name.replace('_', '-')} must be >= 1")
    if int(args.tessera_gee_graph_window) < 0:
        raise ValueError("tessera-gee-graph-window must be >= 0")
    if int(args.tessera_gee_preserve_top) < 0:
        raise ValueError("tessera-gee-preserve-top must be >= 0")
    if float(args.tessera_gee_trigger_threshold) < 0.0 or float(args.tessera_gee_trigger_threshold) > 1.0:
        raise ValueError("tessera-gee-trigger-threshold must be in [0, 1]")
    for name in [
        "tessera_v10_preserve_top",
        "tessera_v10_direct_preserve_top",
    ]:
        if int(getattr(args, name)) < 0:
            raise ValueError(f"{name.replace('_', '-')} must be >= 0")
    for name in [
        "tessera_v10_reference_pool_k",
        "tessera_v10_candidate_pool_k",
    ]:
        if int(getattr(args, name)) < 1:
            raise ValueError(f"{name.replace('_', '-')} must be >= 1")
    if float(args.tessera_v10_relevance_floor) < 0.0:
        raise ValueError("tessera-v10-relevance-floor must be >= 0")
    if int(args.tessera_source_evidence_topk) < 1:
        raise ValueError("tessera-source-evidence-topk must be >= 1")
    if int(args.tessera_source_evidence_candidate_pool_k) < 1:
        raise ValueError("tessera-source-evidence-candidate-pool-k must be >= 1")
    if int(args.tessera_source_evidence_preserve_top) < 0:
        raise ValueError("tessera-source-evidence-preserve-top must be >= 0")
    if float(args.tessera_source_evidence_replacement_margin) < 0.0:
        raise ValueError("tessera-source-evidence-replacement-margin must be >= 0")
    if float(args.tessera_source_evidence_min_candidate_score) < 0.0:
        raise ValueError("tessera-source-evidence-min-candidate-score must be >= 0")
    if int(args.tessera_source_evidence_dense_guard_topn) < 1:
        raise ValueError("tessera-source-evidence-dense-guard-topn must be >= 1")
    if int(args.tessera_source_evidence_max_changed_slots) < 0:
        raise ValueError("tessera-source-evidence-max-changed-slots must be >= 0")
    if float(args.tessera_source_evidence_slot_acceptance_margin) < 0.0:
        raise ValueError("tessera-source-evidence-slot-acceptance-margin must be >= 0")
    if int(args.tessera_source_evidence_budget_candidate_pool_k) < 1:
        raise ValueError("tessera-source-evidence-budget-candidate-pool-k must be >= 1")
    if int(args.tessera_source_evidence_budget_start_slot) < 1:
        raise ValueError("tessera-source-evidence-budget-start-slot must be >= 1")
    if int(args.tessera_source_evidence_budget_max_selected) < 0:
        raise ValueError("tessera-source-evidence-budget-max-selected must be >= 0")
    if float(args.tessera_source_evidence_budget_margin) < 0.0:
        raise ValueError("tessera-source-evidence-budget-margin must be >= 0")
    if float(args.tessera_source_evidence_budget_redundancy_weight) < 0.0:
        raise ValueError("tessera-source-evidence-budget-redundancy-weight must be >= 0")
    if int(args.tessera_source_evidence_sibling_filler_candidate_pool_k) < 1:
        raise ValueError("tessera-source-evidence-sibling-filler-candidate-pool-k must be >= 1")
    if int(args.tessera_source_evidence_sibling_filler_start_slot) < 1:
        raise ValueError("tessera-source-evidence-sibling-filler-start-slot must be >= 1")
    if int(args.tessera_source_evidence_sibling_filler_max_selected) < 0:
        raise ValueError("tessera-source-evidence-sibling-filler-max-selected must be >= 0")
    if int(args.tessera_source_evidence_sibling_filler_tail_topn) < 1:
        raise ValueError("tessera-source-evidence-sibling-filler-tail-topn must be >= 1")
    if int(args.tessera_source_evidence_sibling_filler_reference_topn) < 1:
        raise ValueError("tessera-source-evidence-sibling-filler-reference-topn must be >= 1")
    if float(args.tessera_source_evidence_sibling_filler_margin) < 0.0:
        raise ValueError("tessera-source-evidence-sibling-filler-margin must be >= 0")
    if float(args.tessera_source_evidence_sibling_filler_redundancy_weight) < 0.0:
        raise ValueError("tessera-source-evidence-sibling-filler-redundancy-weight must be >= 0")
    if int(args.tessera_source_evidence_slot_verifier_candidate_pool_k) < 1:
        raise ValueError("tessera-source-evidence-slot-verifier-candidate-pool-k must be >= 1")
    if int(args.tessera_source_evidence_slot_verifier_start_slot) < 1:
        raise ValueError("tessera-source-evidence-slot-verifier-start-slot must be >= 1")
    if int(args.tessera_source_evidence_slot_verifier_max_selected) < 0:
        raise ValueError("tessera-source-evidence-slot-verifier-max-selected must be >= 0")
    if int(args.tessera_source_evidence_slot_verifier_tail_topn) < 1:
        raise ValueError("tessera-source-evidence-slot-verifier-tail-topn must be >= 1")
    if int(args.tessera_source_evidence_slot_verifier_reference_topn) < 1:
        raise ValueError("tessera-source-evidence-slot-verifier-reference-topn must be >= 1")
    if int(args.tessera_source_evidence_slot_verifier_dense_topn) < 1:
        raise ValueError("tessera-source-evidence-slot-verifier-dense-topn must be >= 1")
    if float(args.tessera_source_evidence_slot_verifier_margin) < 0.0:
        raise ValueError("tessera-source-evidence-slot-verifier-margin must be >= 0")
    if float(args.tessera_source_evidence_slot_verifier_min_score) < 0.0:
        raise ValueError("tessera-source-evidence-slot-verifier-min-score must be >= 0")
    if (
        float(args.tessera_source_evidence_slot_verifier_model_threshold) < 0.0
        or float(args.tessera_source_evidence_slot_verifier_model_threshold) > 1.0
    ):
        raise ValueError("tessera-source-evidence-slot-verifier-model-threshold must be in [0, 1]")
    if args.tessera_source_evidence_slot_verifier_model is not None and not Path(args.tessera_source_evidence_slot_verifier_model).exists():
        raise FileNotFoundError(
            f"tessera-source-evidence-slot-verifier-model not found: {args.tessera_source_evidence_slot_verifier_model}"
        )
    if args.tessera_source_evidence_kg_verifier_model is not None and not Path(args.tessera_source_evidence_kg_verifier_model).exists():
        raise FileNotFoundError(
            f"tessera-source-evidence-kg-verifier-model not found: {args.tessera_source_evidence_kg_verifier_model}"
        )
    if float(args.tessera_source_evidence_kg_verifier_weight) < 0.0:
        raise ValueError("tessera-source-evidence-kg-verifier-weight must be >= 0")
    if (
        float(args.tessera_source_evidence_kg_verifier_min_score) < 0.0
        or float(args.tessera_source_evidence_kg_verifier_min_score) > 1.0
    ):
        raise ValueError("tessera-source-evidence-kg-verifier-min-score must be in [0, 1]")
    if int(args.tessera_source_evidence_kg_verify_existing_max_replacements) < 0:
        raise ValueError("tessera-source-evidence-kg-verify-existing-max-replacements must be >= 0")
    if args.tessera_source_budgeter_model is not None and not Path(args.tessera_source_budgeter_model).exists():
        raise FileNotFoundError(f"tessera-source-budgeter-model not found: {args.tessera_source_budgeter_model}")
    if (
        float(args.tessera_source_budgeter_need_threshold) < 0.0
        or float(args.tessera_source_budgeter_need_threshold) > 1.0
    ):
        raise ValueError("tessera-source-budgeter-need-threshold must be in [0, 1]")
    if int(args.tessera_source_budgeter_non_kg_top1_max_kg) < 0:
        raise ValueError("tessera-source-budgeter-non-kg-top1-max-kg must be >= 0")
    if float(args.tessera_source_evidence_slot_verifier_redundancy_weight) < 0.0:
        raise ValueError("tessera-source-evidence-slot-verifier-redundancy-weight must be >= 0")
    if int(args.tessera_source_head_topn) < 1:
        raise ValueError("tessera-source-head-topn must be >= 1")
    if float(args.tessera_source_head_margin) < 0.0:
        raise ValueError("tessera-source-head-margin must be >= 0")
    if float(args.tessera_source_head_off_source_margin) < 0.0:
        raise ValueError("tessera-source-head-off-source-margin must be >= 0")
    if args.tessera_source_action_policy_model is not None and not Path(
        args.tessera_source_action_policy_model
    ).exists():
        raise FileNotFoundError(
            f"tessera-source-action-policy-model not found: {args.tessera_source_action_policy_model}"
        )
    if (
        float(args.tessera_source_action_policy_min_prob) < 0.0
        or float(args.tessera_source_action_policy_min_prob) > 1.0
    ):
        raise ValueError("tessera-source-action-policy-min-prob must be in [0, 1]")
    if int(args.tessera_source_action_policy_topk) < 1:
        raise ValueError("tessera-source-action-policy-topk must be >= 1")
    if int(args.tessera_source_action_policy_pool_k) < 1:
        raise ValueError("tessera-source-action-policy-pool-k must be >= 1")
    if int(args.tessera_final_evidence_topk) < 1:
        raise ValueError("tessera-final-evidence-topk must be >= 1")
    if int(args.tessera_final_evidence_candidate_pool_k) < 1:
        raise ValueError("tessera-final-evidence-candidate-pool-k must be >= 1")
    if int(args.tessera_final_evidence_dense_pool_k) < 1:
        raise ValueError("tessera-final-evidence-dense-pool-k must be >= 1")
    if int(args.tessera_final_evidence_sparse_pool_k) < 1:
        raise ValueError("tessera-final-evidence-sparse-pool-k must be >= 1")
    if int(args.tessera_final_evidence_preserve_top) < 0:
        raise ValueError("tessera-final-evidence-preserve-top must be >= 0")
    if int(args.tessera_final_evidence_max_replacements) < 0:
        raise ValueError("tessera-final-evidence-max-replacements must be >= 0")
    for _name in (
        "tessera_final_evidence_min_candidate_score",
        "tessera_final_evidence_replacement_margin",
        "tessera_final_evidence_min_query_overlap",
        "tessera_final_evidence_source_need_weight",
        "tessera_final_evidence_redundancy_weight",
        "tessera_final_evidence_verifier_margin",
    ):
        if float(getattr(args, _name)) < 0.0:
            raise ValueError(f"{_name.replace('_', '-')} must be >= 0")
    if args.tessera_final_evidence_verifier_model is not None and not Path(
        args.tessera_final_evidence_verifier_model
    ).exists():
        raise FileNotFoundError(
            f"tessera-final-evidence-verifier-model not found: {args.tessera_final_evidence_verifier_model}"
        )
    if (
        float(args.tessera_final_evidence_verifier_threshold) < 0.0
        or float(args.tessera_final_evidence_verifier_threshold) > 1.0
    ):
        raise ValueError("tessera-final-evidence-verifier-threshold must be in [0, 1]")
    for _name in (
        "tessera_retrieval_base_weight",
        "tessera_retrieval_dense_weight",
        "tessera_retrieval_sparse_weight",
        "tessera_retrieval_target_weight",
        "tessera_retrieval_coverage_weight",
        "tessera_retrieval_diversity_weight",
    ):
        if float(getattr(args, _name)) < 0.0:
            raise ValueError(f"{_name.replace('_', '-')} must be >= 0")

    args.cache_dir.mkdir(parents=True, exist_ok=True)
    rows = json.loads(args.split_file.read_text(encoding="utf-8"))[: args.max_queries]
    corpus = json.loads(args.corpus_file.read_text(encoding="utf-8"))
    methods = parse_methods(args.methods, include_unihgkr=bool(args.include_unihgkr))
    ranking_methods = [m for m in methods if m != "unihgkr_dense"]
    ks = sorted({int(k.strip()) for k in args.metrics_k.split(",") if k.strip()})

    q_texts = [str(r.get("query", "")) for r in rows]
    q_ids = [str(r.get("id", f"q_{i}")) for i, r in enumerate(rows)]
    doc_ids = [str(d["id"]) for d in corpus]
    doc_texts = [str(d.get("text", "")) for d in corpus]
    doc_id_to_idx = {did: i for i, did in enumerate(doc_ids)}

    qrels_total = 0
    qrels_in_corpus = 0
    for row in rows:
        qrels = qrels_for_row(row)
        qrels_total += len(qrels)
        qrels_in_corpus += sum(1 for did in qrels if did in doc_id_to_idx)

    q_key = make_cache_key(q_ids)
    c_key = make_cache_key(doc_ids)

    tokenizer, model, device, resolved = e2e.load_e5(args.model_dir)
    pooling_mode = e2e.detect_st_pooling_mode(resolved)
    embed_backend = os.environ.get("TESSERA_EMBED_BACKEND", "hf").strip().lower() or "hf"
    query_prefix = os.environ.get("TESSERA_QUERY_PREFIX", "")
    doc_prefix = os.environ.get("TESSERA_DOC_PREFIX", "")
    q_prefix_key = hashlib.sha1(f"{embed_backend}|{pooling_mode}|{query_prefix}".encode("utf-8")).hexdigest()[:10]
    c_prefix_key = hashlib.sha1(f"{embed_backend}|{pooling_mode}|{doc_prefix}".encode("utf-8")).hexdigest()[:10]
    print(
        f"[stage] model={resolved} backend={embed_backend} device={device} "
        f"queries={len(q_texts)} corpus={len(doc_texts)} pooling={pooling_mode}"
    )
    model_key = hashlib.sha1(str(resolved).encode("utf-8")).hexdigest()[:12]
    q_cache = args.cache_dir / f"dense_query_{model_key}_{pooling_mode}_{q_prefix_key}_{len(q_texts)}_{q_key}.npy"
    c_cache = args.cache_dir / f"dense_corpus_{model_key}_{pooling_mode}_{c_prefix_key}_{len(doc_texts)}_{c_key}.npy"

    if q_cache.exists() and np.load(q_cache, mmap_mode="r").shape[0] == len(q_texts):
        qv = np.load(q_cache)
    else:
        qv = e2e.encode_texts(
            q_texts,
            tokenizer,
            model,
            device,
            batch_size=args.batch_size,
            pooling_mode=pooling_mode,
            query_prefix=query_prefix,
        )
        np.save(q_cache, qv)
    if c_cache.exists() and np.load(c_cache, mmap_mode="r").shape[0] == len(doc_texts):
        cv = np.load(c_cache)
    else:
        cv = e2e.encode_texts(
            doc_texts,
            tokenizer,
            model,
            device,
            batch_size=args.batch_size,
            pooling_mode=pooling_mode,
            query_prefix=doc_prefix,
        )
        np.save(c_cache, cv)
    dense_scores = qv @ cv.T
    sparse_scores = load_or_build_sparse_scores(args, q_texts, doc_texts, q_key, c_key)

    router_model_path = e2e.resolve_router_model_path(args.router_model, args.router_metrics)
    _, router_probs, router_entropy, router_source = e2e.infer_router_predictions(
        q_texts,
        router_model_dir=router_model_path,
        threshold=float(args.router_threshold),
        batch_size=int(args.router_batch_size),
        allow_heuristic_fallback=bool(args.allow_heuristic_router_fallback),
    )
    print(f"[stage] router_source={router_source}")

    table_doc_indices = np.asarray([i for i, did in enumerate(doc_ids) if e2e.source_bucket(did) == "table"], dtype=np.int64)
    kg_doc_indices = np.asarray([i for i, did in enumerate(doc_ids) if e2e.source_bucket(did) == "kg"], dtype=np.int64)
    print("[stage] using lazy document token stores")
    doc_tokens = e2e.LazyDocTokenStore(doc_texts)
    doc_prefix_tokens = e2e.LazyDocPrefixTokenStore(doc_tokens)
    doc_token_lists = [[] for _ in doc_ids]
    doc_signal_tokens = [set() for _ in doc_ids]
    doc_numeric_literals = [set() for _ in doc_ids]
    tessera_ser_bundle = None
    tessera_ser_meta_summary = {}
    if args.tessera_ser_ranker is not None:
        if e2e.load_ser_ranker_bundle is None:
            raise RuntimeError("ser_ranker module is required for --tessera-ser-ranker")
        tessera_ser_bundle = e2e.load_ser_ranker_bundle(args.tessera_ser_ranker)
        ser_meta = dict(getattr(tessera_ser_bundle, "meta", {}) or {})
        split_guard = dict(ser_meta.get("split_guard", {}) or {})
        tessera_ser_meta_summary = {
            "method_name": ser_meta.get("method_name"),
            "method_formulation": ser_meta.get("method_formulation"),
            "train_file": split_guard.get("train_file"),
            "dev_file": split_guard.get("dev_file"),
            "corpus_file": split_guard.get("corpus_file"),
            "train_queries": split_guard.get("train_queries"),
            "dev_queries": split_guard.get("dev_queries"),
            "train_dev_overlap": split_guard.get("train_dev_overlap"),
            "dev_average_precision": ser_meta.get("dev_average_precision"),
            "dev_roc_auc": ser_meta.get("dev_roc_auc"),
        }
        print(f"[stage] loaded TESSERA-SER ranker: {args.tessera_ser_ranker}")
    tessera_slot_verifier_bundle = None
    if args.tessera_source_evidence_slot_verifier_model is not None:
        if e2e.load_pairwise_slot_verifier_bundle is None:
            raise RuntimeError(
                "pairwise_slot_verifier module is required for --tessera-source-evidence-slot-verifier-model"
            )
        tessera_slot_verifier_bundle = e2e.load_pairwise_slot_verifier_bundle(
            args.tessera_source_evidence_slot_verifier_model
        )
        print(
            f"[stage] loaded Pairwise Evidence Slot Verifier: "
            f"{args.tessera_source_evidence_slot_verifier_model}"
        )
    tessera_final_evidence_verifier_bundle = None
    tessera_final_evidence_verifier_meta_summary = {}
    if args.tessera_final_evidence_verifier_model is not None:
        if e2e.load_pairwise_slot_verifier_bundle is None:
            raise RuntimeError(
                "pairwise_slot_verifier module is required for --tessera-final-evidence-verifier-model"
            )
        tessera_final_evidence_verifier_bundle = e2e.load_pairwise_slot_verifier_bundle(
            args.tessera_final_evidence_verifier_model
        )
        erv_meta = dict(getattr(tessera_final_evidence_verifier_bundle, "metadata", {}) or {})
        tessera_final_evidence_verifier_meta_summary = {
            "method_name": erv_meta.get("method_name"),
            "method_formulation": erv_meta.get("method_formulation"),
            "recommended_threshold": erv_meta.get("recommended_threshold"),
            "enabled_families": erv_meta.get("enabled_families"),
            "train_examples": erv_meta.get("train_examples"),
            "val_examples": erv_meta.get("val_examples"),
            "train_dev_overlap": erv_meta.get("train_dev_overlap"),
        }
        print(f"[stage] loaded Evidence Replacement Verifier: {args.tessera_final_evidence_verifier_model}")
    tessera_kg_verifier_bundle = None
    tessera_kg_verifier_meta_summary = {}
    if args.tessera_source_evidence_kg_verifier_model is not None:
        if e2e.load_kg_consistency_bundle is None:
            raise RuntimeError(
                "kg_consistency_verifier module is required for --tessera-source-evidence-kg-verifier-model"
            )
        tessera_kg_verifier_bundle = e2e.load_kg_consistency_bundle(
            args.tessera_source_evidence_kg_verifier_model
        )
        kg_meta = dict(getattr(tessera_kg_verifier_bundle, "metadata", {}) or {})
        split_guard = dict(kg_meta.get("split_guard", {}) or {})
        tessera_kg_verifier_meta_summary = {
            "method_name": kg_meta.get("method_name"),
            "method_formulation": kg_meta.get("method_formulation"),
            "train_file": split_guard.get("train_file"),
            "dev_file": split_guard.get("dev_file"),
            "corpus_file": split_guard.get("corpus_file"),
            "train_queries": split_guard.get("train_queries"),
            "dev_queries": split_guard.get("dev_queries"),
            "train_dev_overlap": split_guard.get("train_dev_overlap"),
            "dev_average_precision": kg_meta.get("dev_average_precision"),
            "dev_roc_auc": kg_meta.get("dev_roc_auc"),
        }
        print(
            f"[stage] loaded KG Entity-Relation Consistency Verifier: "
            f"{args.tessera_source_evidence_kg_verifier_model}"
        )
    tessera_source_budgeter_bundle = None
    tessera_source_budgeter_meta_summary = {}
    if args.tessera_source_budgeter_model is not None:
        if e2e.load_source_budgeter_bundle is None:
            raise RuntimeError("source_budgeter module is required for --tessera-source-budgeter-model")
        tessera_source_budgeter_bundle = e2e.load_source_budgeter_bundle(args.tessera_source_budgeter_model)
        budget_meta = dict(getattr(tessera_source_budgeter_bundle, "metadata", {}) or {})
        split_guard = dict(budget_meta.get("split_guard", {}) or {})
        tessera_source_budgeter_meta_summary = {
            "method_name": budget_meta.get("method_name"),
            "method_formulation": budget_meta.get("method_formulation"),
            "train_file": split_guard.get("train_file"),
            "dev_file": split_guard.get("dev_file"),
            "train_queries": split_guard.get("train_queries"),
            "dev_queries": split_guard.get("dev_queries"),
            "train_dev_overlap": split_guard.get("train_dev_overlap"),
            "dev_top1_accuracy": budget_meta.get("dev_top1_accuracy"),
            "need_metrics": budget_meta.get("need_metrics"),
        }
        print(f"[stage] loaded Query-Adaptive Source Budgeter: {args.tessera_source_budgeter_model}")

    tessera_source_action_policy_bundle = None
    tessera_source_action_policy_meta_summary = {}
    if args.tessera_source_action_policy_model is not None:
        if e2e.load_source_action_policy_bundle is None:
            raise RuntimeError("source_action_policy module is required for --tessera-source-action-policy-model")
        tessera_source_action_policy_bundle = e2e.load_source_action_policy_bundle(
            args.tessera_source_action_policy_model
        )
        action_meta = dict(getattr(tessera_source_action_policy_bundle, "metadata", {}) or {})
        split_guard = dict(action_meta.get("split_guard", {}) or {})
        tessera_source_action_policy_meta_summary = {
            "method_name": action_meta.get("method_name"),
            "method_formulation": action_meta.get("method_formulation"),
            "train_rankings_jsonl": split_guard.get("train_rankings_jsonl"),
            "dev_rankings_jsonl": split_guard.get("dev_rankings_jsonl"),
            "train_queries": split_guard.get("train_queries"),
            "dev_queries": split_guard.get("dev_queries"),
            "train_dev_overlap": split_guard.get("train_dev_overlap"),
            "dev_action_accuracy": action_meta.get("dev_action_accuracy"),
            "dev_regression_mae": action_meta.get("dev_regression_mae"),
            "dev_policy_eval": action_meta.get("dev_policy_eval"),
            "action_stats": action_meta.get("action_stats"),
        }
        print(f"[stage] loaded Source Action/Utility Policy: {args.tessera_source_action_policy_model}")
    need_table_structs = "tablerag" in ranking_methods and float(args.table_cellmaxsim_weight) > 0.0
    doc_table_structs = [
        e2e.extract_table_structure_tokens(t, max_cells=max(64, int(args.table_cellmaxsim_top_cells)))
        if need_table_structs and e2e.source_bucket(did) == "table"
        else None
        for did, t in zip(doc_ids, doc_texts)
    ]
    doc_kg_path_sets = [None for _ in doc_ids]

    preds: dict[str, list[list[str]]] = {m: [] for m in methods}
    rankings_debug_f = None
    if args.save_rankings_jsonl is not None:
        args.save_rankings_jsonl.parent.mkdir(parents=True, exist_ok=True)
        rankings_debug_f = args.save_rankings_jsonl.open("w", encoding="utf-8")

    start = time.perf_counter()
    try:
        for i, row in enumerate(rows):
            per_query_trace: dict[str, list[float]] | None = {} if rankings_debug_f is not None else None
            rankings = e2e.build_rankings_for_query(
            q_texts[i],
            q_ids[i],
            dense_scores[i],
            sparse_scores[i],
            doc_ids,
            doc_texts,
            doc_tokens,
            doc_prefix_tokens,
            doc_token_lists,
            doc_signal_tokens,
            doc_numeric_literals,
            table_doc_indices,
            kg_doc_indices,
            doc_table_structs,
            doc_kg_path_sets,
            {},
            {},
            None,
            retrieve_topk=max(args.retrieve_topk, max(ks)),
            preserve_dense_top=args.preserve_dense_top,
            tessera_late_alpha=args.tessera_late_alpha,
            router_prob=np.asarray(router_probs[i], dtype=np.float32),
            query_modality_prior_mix=args.query_modality_prior_mix,
            query_modality_prior_adaptive=False,
            query_modality_prior_entropy_scale=0.30,
            query_modality_prior_disagreement_scale=0.25,
            query_modality_prior_min=0.0,
            query_modality_prior_max=0.85,
            router_entropy=float(router_entropy[i]),
            uncertainty_threshold=args.routing_uncertainty_threshold,
            pathmaxsim_weight=args.pathmaxsim_weight,
            pathmaxsim_kg_threshold=args.pathmaxsim_kg_threshold,
            table_cellmaxsim_weight=args.table_cellmaxsim_weight,
            table_cellmaxsim_top_cells=args.table_cellmaxsim_top_cells,
            innovation_scheme2=False,
            scheme2_cross_modal_weight=0.0,
            scheme2_token_maxsim_weight=0.0,
            adapter_plus_mode=bool(args.adapter_plus_mode),
            adapter_official_lite=bool(args.adapter_official_lite),
            heavy_schemeb_mode=False,
            heavy_table_encoder_weight=0.0,
            heavy_kg_path_weight=0.0,
            heavy_token_late_weight=0.0,
            heavy_query_max_tokens=16,
            heavy_table_max_cells=128,
            heavy_token_doc_max_tokens=160,
            heavy_table_backend="hash",
            heavy_table_tapas_topn=0,
            heavy_table_max_rows=32,
            heavy_table_max_cols=16,
            heavy_table_agg_cell_logit=0.0,
            heavy_table_agg_row_logit=-0.4,
            heavy_table_agg_col_logit=-0.6,
            heavy_table_agg_temp=0.35,
            heavy_kg_backend="token",
            heavy_kg_gnn_topn=0,
            heavy_kg_max_hops=2,
            heavy_kg_max_paths=64,
            heavy_kg_contrastive_temp=0.12,
            heavy_kg_hard_negative_mode="cross_doc_hard",
            heavy_kg_hard_negative_topdocs=3,
            heavy_kg_hard_negative_max_paths=24,
            heavy_token_cross_modal_weight=0.0,
            heavy_branch_candidate_expand_k=0,
            heavy_branch_candidate_table_weight=0.55,
            heavy_branch_candidate_kg_weight=0.55,
            heavy_branch_candidate_max_total=1400,
            heavy_score_calibration="none",
            heavy_score_calibration_nonzero_only=False,
            qa_objective_retrieval_weight=args.qa_objective_retrieval_weight,
            qa_objective_targeted_only=bool(args.qa_objective_targeted_only),
            upo_lite_retrieval_weight=0.0,
            upo_lite_targeted_only=True,
            retrieval_conflict_penalty_weight=0.0,
            retrieval_conflict_targeted_only=True,
            retrieval_conflict_table_kg_only=False,
            retrieval_conflict_risk_gating=True,
            retrieval_conflict_risk_low=0.12,
            retrieval_conflict_risk_high=0.32,
            retrieval_conflict_risk_probe_k=8,
            retrieval_conflict_anchor_k=4,
            retrieval_conflict_max_literals_per_doc=0,
            retrieval_conflict_sensitive_target_scale=1.0,
            conflict_bundle=None,
            selected_methods=set(ranking_methods),
            tessera_candidate_pool_k=int(args.tessera_candidate_pool_k),
            tessera_retrieval_multi_agent=bool(args.tessera_retrieval_multi_agent),
            tessera_retrieval_agent_pool_k=int(args.tessera_retrieval_agent_pool_k),
            tessera_retrieval_dense_pool_k=int(args.tessera_retrieval_dense_pool_k),
            tessera_retrieval_sparse_pool_k=int(args.tessera_retrieval_sparse_pool_k),
            tessera_retrieval_preserve_top=int(args.tessera_retrieval_preserve_top),
            tessera_retrieval_base_weight=float(args.tessera_retrieval_base_weight),
            tessera_retrieval_dense_weight=float(args.tessera_retrieval_dense_weight),
            tessera_retrieval_sparse_weight=float(args.tessera_retrieval_sparse_weight),
            tessera_retrieval_target_weight=float(args.tessera_retrieval_target_weight),
            tessera_retrieval_coverage_weight=float(args.tessera_retrieval_coverage_weight),
            tessera_retrieval_diversity_weight=float(args.tessera_retrieval_diversity_weight),
            tessera_retrieval_dense_rescue_k=int(args.tessera_retrieval_dense_rescue_k),
            tessera_retrieval_dense_rescue_pool_k=int(args.tessera_retrieval_dense_rescue_pool_k),
            tessera_retrieval_sibling_seed_k=int(args.tessera_retrieval_sibling_seed_k),
            tessera_retrieval_sibling_window=int(args.tessera_retrieval_sibling_window),
            tessera_retrieval_sibling_weight=float(args.tessera_retrieval_sibling_weight),
            tessera_retrieval_moe=bool(args.tessera_retrieval_moe),
            tessera_moe_pool_k=int(args.tessera_moe_pool_k),
            tessera_moe_prf_seed_k=int(args.tessera_moe_prf_seed_k),
            tessera_moe_prf_dense_seed_k=int(args.tessera_moe_prf_dense_seed_k),
            tessera_moe_prf_sparse_seed_k=int(args.tessera_moe_prf_sparse_seed_k),
            tessera_moe_prf_max_terms=int(args.tessera_moe_prf_max_terms),
            tessera_moe_sibling_seed_k=int(args.tessera_moe_sibling_seed_k),
            tessera_moe_sibling_window=int(args.tessera_moe_sibling_window),
            tessera_moe_sibling_weight=float(args.tessera_moe_sibling_weight),
            tessera_ser_bundle=tessera_ser_bundle,
            tessera_ser_candidate_pool_k=int(args.tessera_ser_candidate_pool_k),
            tessera_ser_dense_pool_k=int(args.tessera_ser_dense_pool_k),
            tessera_ser_sparse_pool_k=int(args.tessera_ser_sparse_pool_k),
            tessera_ser_preserve_top=int(args.tessera_ser_preserve_top),
            tessera_ser_blend_weight=float(args.tessera_ser_blend_weight),
            tessera_ser_diversity_weight=float(args.tessera_ser_diversity_weight),
            tessera_ser_evidence_rescue_k=int(args.tessera_ser_evidence_rescue_k),
            tessera_ser_evidence_rescue_pool_k=int(args.tessera_ser_evidence_rescue_pool_k),
            tessera_ser_evidence_preserve_top=int(args.tessera_ser_evidence_preserve_top),
            tessera_ser_evidence_redundancy_weight=float(args.tessera_ser_evidence_redundancy_weight),
            tessera_ser_evidence_min_gain=float(args.tessera_ser_evidence_min_gain),
            tessera_ser_plan_adaptive=bool(args.tessera_ser_plan_adaptive),
            tessera_ser_plan_dense_weight=float(args.tessera_ser_plan_dense_weight),
            tessera_ser_plan_sparse_weight=float(args.tessera_ser_plan_sparse_weight),
            tessera_ser_plan_lexical_weight=float(args.tessera_ser_plan_lexical_weight),
            tessera_ser_plan_slot_weight=float(args.tessera_ser_plan_slot_weight),
            tessera_ser_evidence_set_selection=bool(args.tessera_ser_evidence_set_selection),
            tessera_ser_evidence_set_preserve_top=int(args.tessera_ser_evidence_set_preserve_top),
            tessera_ser_evidence_set_pool_k=int(args.tessera_ser_evidence_set_pool_k),
            tessera_ser_evidence_set_cardinality_threshold=float(args.tessera_ser_evidence_set_cardinality_threshold),
            tessera_ser_evidence_set_learned_weight=float(args.tessera_ser_evidence_set_learned_weight),
            tessera_ser_evidence_set_base_weight=float(args.tessera_ser_evidence_set_base_weight),
            tessera_ser_evidence_set_dense_weight=float(args.tessera_ser_evidence_set_dense_weight),
            tessera_ser_evidence_set_sparse_weight=float(args.tessera_ser_evidence_set_sparse_weight),
            tessera_ser_evidence_set_probe_weight=float(args.tessera_ser_evidence_set_probe_weight),
            tessera_ser_evidence_set_slot_weight=float(args.tessera_ser_evidence_set_slot_weight),
            tessera_ser_evidence_set_anchor_weight=float(args.tessera_ser_evidence_set_anchor_weight),
            tessera_ser_evidence_set_family_weight=float(args.tessera_ser_evidence_set_family_weight),
            tessera_ser_evidence_set_redundancy_weight=float(args.tessera_ser_evidence_set_redundancy_weight),
            tessera_graph_evidence_expansion=bool(args.tessera_graph_evidence_expansion),
            tessera_gee_post_rerank=bool(args.tessera_gee_post_rerank),
            tessera_gee_candidate_pool_k=int(args.tessera_gee_candidate_pool_k),
            tessera_gee_dense_pool_k=int(args.tessera_gee_dense_pool_k),
            tessera_gee_sparse_pool_k=int(args.tessera_gee_sparse_pool_k),
            tessera_gee_graph_seed_k=int(args.tessera_gee_graph_seed_k),
            tessera_gee_graph_window=int(args.tessera_gee_graph_window),
            tessera_gee_preserve_top=int(args.tessera_gee_preserve_top),
            tessera_gee_trigger_threshold=float(args.tessera_gee_trigger_threshold),
            tessera_gee_base_weight=float(args.tessera_gee_base_weight),
            tessera_gee_dense_weight=float(args.tessera_gee_dense_weight),
            tessera_gee_sparse_weight=float(args.tessera_gee_sparse_weight),
            tessera_gee_probe_weight=float(args.tessera_gee_probe_weight),
            tessera_gee_graph_weight=float(args.tessera_gee_graph_weight),
            tessera_gee_slot_weight=float(args.tessera_gee_slot_weight),
            tessera_gee_sibling_weight=float(args.tessera_gee_sibling_weight),
            tessera_gee_redundancy_weight=float(args.tessera_gee_redundancy_weight),
            doc_id_to_idx=doc_id_to_idx,
            debug_trace=per_query_trace,
            tessera_v9_enabled=bool(args.tessera_v9),
            tessera_v9_local_rerank=bool(args.tessera_v9_local_rerank),
            tessera_v9_dense_pool_k=int(args.tessera_v9_dense_pool_k),
            tessera_v9_sparse_pool_k=int(args.tessera_v9_sparse_pool_k),
            tessera_v9_candidate_pool_k=int(args.tessera_v9_candidate_pool_k),
            tessera_v9_graph_seed_k=int(args.tessera_v9_graph_seed_k),
            tessera_v9_graph_window=int(args.tessera_v9_graph_window),
            tessera_v9_preserve_top=int(args.tessera_v9_preserve_top),
            tessera_v9_base_weight=float(args.tessera_v9_base_weight),
            tessera_v9_dense_weight=float(args.tessera_v9_dense_weight),
            tessera_v9_sparse_weight=float(args.tessera_v9_sparse_weight),
            tessera_v9_probe_weight=float(args.tessera_v9_probe_weight),
            tessera_v9_graph_weight=float(args.tessera_v9_graph_weight),
            tessera_v9_slot_weight=float(args.tessera_v9_slot_weight),
            tessera_v9_diversity_weight=float(args.tessera_v9_diversity_weight),
            tessera_v9_modality_weight=float(args.tessera_v9_modality_weight),
            tessera_v10_conservative_rerank=bool(args.tessera_v10_conservative_rerank),
            tessera_v10_preserve_top=int(args.tessera_v10_preserve_top),
            tessera_v10_direct_preserve_top=int(args.tessera_v10_direct_preserve_top),
            tessera_v10_reference_pool_k=int(args.tessera_v10_reference_pool_k),
            tessera_v10_candidate_pool_k=int(args.tessera_v10_candidate_pool_k),
            tessera_v10_reference_weight=float(args.tessera_v10_reference_weight),
            tessera_v10_current_weight=float(args.tessera_v10_current_weight),
            tessera_v10_base_weight=float(args.tessera_v10_base_weight),
            tessera_v10_dense_weight=float(args.tessera_v10_dense_weight),
            tessera_v10_sparse_weight=float(args.tessera_v10_sparse_weight),
            tessera_v10_probe_weight=float(args.tessera_v10_probe_weight),
            tessera_v10_slot_weight=float(args.tessera_v10_slot_weight),
            tessera_v10_diversity_weight=float(args.tessera_v10_diversity_weight),
            tessera_v10_margin=float(args.tessera_v10_margin),
            tessera_v10_relevance_floor=float(args.tessera_v10_relevance_floor),
            tessera_source_evidence_fusion=bool(args.tessera_source_evidence_fusion),
            tessera_source_evidence_slot_verifier_bundle=tessera_slot_verifier_bundle,
            tessera_source_evidence_topk=int(args.tessera_source_evidence_topk),
            tessera_source_evidence_candidate_pool_k=int(args.tessera_source_evidence_candidate_pool_k),
            tessera_source_evidence_preserve_top=int(args.tessera_source_evidence_preserve_top),
            tessera_source_evidence_base_weight=float(args.tessera_source_evidence_base_weight),
            tessera_source_evidence_dense_weight=float(args.tessera_source_evidence_dense_weight),
            tessera_source_evidence_sparse_weight=float(args.tessera_source_evidence_sparse_weight),
            tessera_source_evidence_reference_weight=float(args.tessera_source_evidence_reference_weight),
            tessera_source_evidence_lexical_weight=float(args.tessera_source_evidence_lexical_weight),
            tessera_source_evidence_modality_prior_weight=float(args.tessera_source_evidence_modality_prior_weight),
            tessera_source_evidence_source_balance_weight=float(args.tessera_source_evidence_source_balance_weight),
            tessera_source_evidence_target_family_weight=float(args.tessera_source_evidence_target_family_weight),
            tessera_source_evidence_diversity_weight=float(args.tessera_source_evidence_diversity_weight),
            tessera_source_evidence_replacement_margin=float(args.tessera_source_evidence_replacement_margin),
            tessera_source_evidence_min_candidate_score=float(args.tessera_source_evidence_min_candidate_score),
            tessera_source_evidence_dense_guard=bool(args.tessera_source_evidence_dense_guard),
            tessera_source_evidence_dense_guard_topn=int(args.tessera_source_evidence_dense_guard_topn),
            tessera_source_evidence_dense_guard_prefixes=str(args.tessera_source_evidence_dense_guard_prefixes),
            tessera_source_evidence_dense_guard_weight=float(args.tessera_source_evidence_dense_guard_weight),
            tessera_source_evidence_dense_rank_weight=float(args.tessera_source_evidence_dense_rank_weight),
            tessera_source_evidence_current_rank_weight=float(args.tessera_source_evidence_current_rank_weight),
            tessera_source_evidence_source_balance_prefixes=str(args.tessera_source_evidence_source_balance_prefixes),
            tessera_source_evidence_max_changed_slots=int(args.tessera_source_evidence_max_changed_slots),
            tessera_source_evidence_slot_acceptance_guard=bool(args.tessera_source_evidence_slot_acceptance_guard),
            tessera_source_evidence_slot_acceptance_prefixes=str(args.tessera_source_evidence_slot_acceptance_prefixes),
            tessera_source_evidence_slot_acceptance_margin=float(args.tessera_source_evidence_slot_acceptance_margin),
            tessera_source_evidence_budget_composer=bool(args.tessera_source_evidence_budget_composer),
            tessera_source_evidence_budget_prefixes=str(args.tessera_source_evidence_budget_prefixes),
            tessera_source_evidence_budget_candidate_pool_k=int(args.tessera_source_evidence_budget_candidate_pool_k),
            tessera_source_evidence_budget_start_slot=int(args.tessera_source_evidence_budget_start_slot),
            tessera_source_evidence_budget_max_selected=int(args.tessera_source_evidence_budget_max_selected),
            tessera_source_evidence_budget_score_weight=float(args.tessera_source_evidence_budget_score_weight),
            tessera_source_evidence_budget_sibling_weight=float(args.tessera_source_evidence_budget_sibling_weight),
            tessera_source_evidence_budget_source_quota_weight=float(args.tessera_source_evidence_budget_source_quota_weight),
            tessera_source_evidence_budget_tail_rank_weight=float(args.tessera_source_evidence_budget_tail_rank_weight),
            tessera_source_evidence_budget_reference_weight=float(args.tessera_source_evidence_budget_reference_weight),
            tessera_source_evidence_budget_margin=float(args.tessera_source_evidence_budget_margin),
            tessera_source_evidence_budget_redundancy_weight=float(args.tessera_source_evidence_budget_redundancy_weight),
            tessera_source_evidence_sibling_filler=bool(args.tessera_source_evidence_sibling_filler),
            tessera_source_evidence_sibling_filler_prefixes=str(args.tessera_source_evidence_sibling_filler_prefixes),
            tessera_source_evidence_sibling_filler_candidate_pool_k=int(args.tessera_source_evidence_sibling_filler_candidate_pool_k),
            tessera_source_evidence_sibling_filler_start_slot=int(args.tessera_source_evidence_sibling_filler_start_slot),
            tessera_source_evidence_sibling_filler_max_selected=int(args.tessera_source_evidence_sibling_filler_max_selected),
            tessera_source_evidence_sibling_filler_tail_topn=int(args.tessera_source_evidence_sibling_filler_tail_topn),
            tessera_source_evidence_sibling_filler_reference_topn=int(args.tessera_source_evidence_sibling_filler_reference_topn),
            tessera_source_evidence_sibling_filler_margin=float(args.tessera_source_evidence_sibling_filler_margin),
            tessera_source_evidence_sibling_filler_sibling_weight=float(args.tessera_source_evidence_sibling_filler_sibling_weight),
            tessera_source_evidence_sibling_filler_reference_weight=float(args.tessera_source_evidence_sibling_filler_reference_weight),
            tessera_source_evidence_sibling_filler_tail_weight=float(args.tessera_source_evidence_sibling_filler_tail_weight),
            tessera_source_evidence_sibling_filler_dense_weight=float(args.tessera_source_evidence_sibling_filler_dense_weight),
            tessera_source_evidence_sibling_filler_source_weight=float(args.tessera_source_evidence_sibling_filler_source_weight),
            tessera_source_evidence_sibling_filler_redundancy_weight=float(args.tessera_source_evidence_sibling_filler_redundancy_weight),
            tessera_source_evidence_slot_verifier=bool(args.tessera_source_evidence_slot_verifier),
            tessera_source_evidence_slot_verifier_prefixes=str(args.tessera_source_evidence_slot_verifier_prefixes),
            tessera_source_evidence_slot_verifier_candidate_pool_k=int(args.tessera_source_evidence_slot_verifier_candidate_pool_k),
            tessera_source_evidence_slot_verifier_start_slot=int(args.tessera_source_evidence_slot_verifier_start_slot),
            tessera_source_evidence_slot_verifier_max_selected=int(args.tessera_source_evidence_slot_verifier_max_selected),
            tessera_source_evidence_slot_verifier_tail_topn=int(args.tessera_source_evidence_slot_verifier_tail_topn),
            tessera_source_evidence_slot_verifier_reference_topn=int(args.tessera_source_evidence_slot_verifier_reference_topn),
            tessera_source_evidence_slot_verifier_dense_topn=int(args.tessera_source_evidence_slot_verifier_dense_topn),
            tessera_source_evidence_slot_verifier_margin=float(args.tessera_source_evidence_slot_verifier_margin),
            tessera_source_evidence_slot_verifier_min_score=float(args.tessera_source_evidence_slot_verifier_min_score),
            tessera_source_evidence_slot_verifier_model_threshold=float(args.tessera_source_evidence_slot_verifier_model_threshold),
            tessera_source_evidence_slot_verifier_static_weight=float(args.tessera_source_evidence_slot_verifier_static_weight),
            tessera_source_evidence_slot_verifier_reference_weight=float(args.tessera_source_evidence_slot_verifier_reference_weight),
            tessera_source_evidence_slot_verifier_dense_weight=float(args.tessera_source_evidence_slot_verifier_dense_weight),
            tessera_source_evidence_slot_verifier_tail_weight=float(args.tessera_source_evidence_slot_verifier_tail_weight),
            tessera_source_evidence_slot_verifier_sibling_weight=float(args.tessera_source_evidence_slot_verifier_sibling_weight),
            tessera_source_evidence_slot_verifier_source_weight=float(args.tessera_source_evidence_slot_verifier_source_weight),
            tessera_source_evidence_slot_verifier_lexical_weight=float(args.tessera_source_evidence_slot_verifier_lexical_weight),
            tessera_source_evidence_slot_verifier_family_weight=float(args.tessera_source_evidence_slot_verifier_family_weight),
            tessera_source_evidence_slot_verifier_redundancy_weight=float(args.tessera_source_evidence_slot_verifier_redundancy_weight),
            tessera_source_evidence_kg_preservation_guard=bool(args.tessera_source_evidence_kg_preservation_guard),
            tessera_source_evidence_kg_preservation_prefixes=str(args.tessera_source_evidence_kg_preservation_prefixes),
            tessera_source_evidence_kg_preservation_min_kg=int(args.tessera_source_evidence_kg_preservation_min_kg),
            tessera_source_evidence_kg_preservation_candidate_pool_k=int(args.tessera_source_evidence_kg_preservation_candidate_pool_k),
            tessera_source_evidence_kg_preservation_start_slot=int(args.tessera_source_evidence_kg_preservation_start_slot),
            tessera_source_evidence_kg_preservation_margin=float(args.tessera_source_evidence_kg_preservation_margin),
            tessera_source_evidence_kg_preservation_reference_weight=float(args.tessera_source_evidence_kg_preservation_reference_weight),
            tessera_source_evidence_kg_preservation_dense_weight=float(args.tessera_source_evidence_kg_preservation_dense_weight),
            tessera_source_evidence_kg_preservation_current_weight=float(args.tessera_source_evidence_kg_preservation_current_weight),
            tessera_source_evidence_kg_preservation_family_weight=float(args.tessera_source_evidence_kg_preservation_family_weight),
            tessera_source_evidence_kg_preservation_lexical_weight=float(args.tessera_source_evidence_kg_preservation_lexical_weight),
            tessera_source_evidence_kg_verifier_bundle=tessera_kg_verifier_bundle,
            tessera_source_evidence_kg_verifier_weight=float(args.tessera_source_evidence_kg_verifier_weight),
            tessera_source_evidence_kg_verifier_min_score=float(args.tessera_source_evidence_kg_verifier_min_score),
            tessera_source_budgeter_bundle=tessera_source_budgeter_bundle,
            tessera_source_budgeter_top1_guard=bool(args.tessera_source_budgeter_top1_guard),
            tessera_source_budgeter_need_threshold=float(args.tessera_source_budgeter_need_threshold),
            tessera_source_budgeter_non_kg_top1_max_kg=int(args.tessera_source_budgeter_non_kg_top1_max_kg),
            tessera_source_head_selector=bool(args.tessera_source_head_selector),
            tessera_source_head_topn=int(args.tessera_source_head_topn),
            tessera_source_head_source_weight=float(args.tessera_source_head_source_weight),
            tessera_source_head_same_query_weight=float(args.tessera_source_head_same_query_weight),
            tessera_source_head_position_weight=float(args.tessera_source_head_position_weight),
            tessera_source_head_reference_weight=float(args.tessera_source_head_reference_weight),
            tessera_source_head_lexical_weight=float(args.tessera_source_head_lexical_weight),
            tessera_source_head_base_weight=float(args.tessera_source_head_base_weight),
            tessera_source_head_dense_weight=float(args.tessera_source_head_dense_weight),
            tessera_source_head_sparse_weight=float(args.tessera_source_head_sparse_weight),
            tessera_source_head_margin=float(args.tessera_source_head_margin),
            tessera_source_head_off_source_margin=float(args.tessera_source_head_off_source_margin),
            tessera_source_action_policy_bundle=tessera_source_action_policy_bundle,
            tessera_source_action_policy_min_prob=float(args.tessera_source_action_policy_min_prob),
            tessera_source_action_policy_topk=int(args.tessera_source_action_policy_topk),
            tessera_source_action_policy_pool_k=int(args.tessera_source_action_policy_pool_k),
            tessera_final_evidence_composer=bool(args.tessera_final_evidence_composer),
            tessera_final_evidence_topk=int(args.tessera_final_evidence_topk),
            tessera_final_evidence_candidate_pool_k=int(args.tessera_final_evidence_candidate_pool_k),
            tessera_final_evidence_dense_pool_k=int(args.tessera_final_evidence_dense_pool_k),
            tessera_final_evidence_sparse_pool_k=int(args.tessera_final_evidence_sparse_pool_k),
            tessera_final_evidence_preserve_top=int(args.tessera_final_evidence_preserve_top),
            tessera_final_evidence_max_replacements=int(args.tessera_final_evidence_max_replacements),
            tessera_final_evidence_min_candidate_score=float(args.tessera_final_evidence_min_candidate_score),
            tessera_final_evidence_replacement_margin=float(args.tessera_final_evidence_replacement_margin),
            tessera_final_evidence_min_query_overlap=float(args.tessera_final_evidence_min_query_overlap),
            tessera_final_evidence_source_need_weight=float(args.tessera_final_evidence_source_need_weight),
            tessera_final_evidence_redundancy_weight=float(args.tessera_final_evidence_redundancy_weight),
            tessera_final_evidence_verifier_bundle=tessera_final_evidence_verifier_bundle,
            tessera_final_evidence_verifier_threshold=float(args.tessera_final_evidence_verifier_threshold),
            tessera_final_evidence_verifier_margin=float(args.tessera_final_evidence_verifier_margin),
            )
            for m in ranking_methods:
                preds[m].append([doc_ids[j] for j in rankings[m]])
            if rankings_debug_f is not None:
                qrels = qrels_for_row(row)
                rel = {doc_id for doc_id, grade in qrels.items() if grade > 0}
                v9_pool = rankings.get("_tessera_v9_candidate_pool", [])
                if per_query_trace is not None and v9_pool:
                    v9_pool_ids = [doc_ids[j] for j in v9_pool if 0 <= int(j) < len(doc_ids)]
                    per_query_trace.setdefault("tessera_v9_candidate_pool_hits", []).append(
                        float(sum(1 for did in v9_pool_ids if did in rel))
                    )
                    per_query_trace.setdefault("tessera_v9_candidate_pool_any_hit", []).append(
                        float(any(did in rel for did in v9_pool_ids))
                    )
                    per_query_trace.setdefault("tessera_v9_candidate_pool_rel_coverage", []).append(
                        float(sum(1 for did in v9_pool_ids if did in rel) / max(1, len(rel)))
                    )
                trace_out = {
                    key: (vals[0] if len(vals) == 1 else vals)
                    for key, vals in (per_query_trace or {}).items()
                }
                rankings_debug_f.write(
                    json.dumps(
                        {
                            "query_index": i,
                            "query_id": q_ids[i],
                            "query": q_texts[i],
                            "qrels": qrels,
                            "rankings": {
                                m: [doc_ids[j] for j in rankings[m][: max(args.retrieve_topk, max(ks))]]
                                for m in ranking_methods
                            },
                            "trace": trace_out,
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )
            if (i + 1) % 50 == 0 or i + 1 == len(rows):
                elapsed = time.perf_counter() - start
                print(f"[progress] {i + 1}/{len(rows)} elapsed={elapsed:.1f}s", flush=True)
    finally:
        if rankings_debug_f is not None:
            rankings_debug_f.close()

    if "unihgkr_dense" in methods:
        if e2e.AutoTokenizer is None or e2e.AutoModel is None or e2e.torch is None:
            raise RuntimeError("UniHGKR baseline requires transformers + torch")
        if not args.unihgkr_model_dir.exists():
            raise FileNotFoundError(f"UniHGKR model dir not found: {args.unihgkr_model_dir}")
        pooling = e2e.detect_st_pooling_mode(args.unihgkr_model_dir)
        do_normalize = e2e.has_st_normalize(args.unihgkr_model_dir)
        uh_key = hashlib.sha1(str(args.unihgkr_model_dir.resolve()).encode("utf-8")).hexdigest()[:12]
        uq_cache = args.cache_dir / f"unihgkr_query_{uh_key}_{pooling}_{len(q_texts)}_{q_key}.npy"
        uc_cache = args.cache_dir / f"unihgkr_corpus_{uh_key}_{pooling}_{len(doc_texts)}_{c_key}.npy"
        print(f"[stage] loading UniHGKR baseline model from {args.unihgkr_model_dir}")
        uh_tokenizer = e2e.AutoTokenizer.from_pretrained(str(args.unihgkr_model_dir))
        uh_model = e2e.AutoModel.from_pretrained(str(args.unihgkr_model_dir))
        uh_device = e2e.torch.device("cuda" if e2e.torch.cuda.is_available() else "cpu")
        uh_model.to(uh_device)
        if uq_cache.exists() and np.load(uq_cache, mmap_mode="r").shape[0] == len(q_texts):
            uqv = np.load(uq_cache)
        else:
            uqv = e2e.encode_texts_with_hf_encoder(
                q_texts, uh_tokenizer, uh_model, uh_device,
                batch_size=int(args.unihgkr_batch_size), pooling_mode=pooling, do_normalize=do_normalize,
            )
            np.save(uq_cache, uqv)
        if uc_cache.exists() and np.load(uc_cache, mmap_mode="r").shape[0] == len(doc_texts):
            ucv = np.load(uc_cache)
        else:
            ucv = e2e.encode_texts_with_hf_encoder(
                doc_texts, uh_tokenizer, uh_model, uh_device,
                batch_size=int(args.unihgkr_batch_size), pooling_mode=pooling, do_normalize=do_normalize,
            )
            np.save(uc_cache, ucv)
        uh_scores = uqv @ ucv.T
        preds["unihgkr_dense"] = [
            [doc_ids[j] for j in e2e.topk_indices(uh_scores[i], max(args.retrieve_topk, max(ks))).tolist()]
            for i in range(len(rows))
        ]

    metric_keys = []
    for metric_name in ["ndcg", "map", "hits"]:
        for k in ks:
            metric_keys.append(f"{metric_name}@{k}")
    metric_keys.extend([f"any_hit@{k}" for k in ks])

    table_rows: list[dict] = []
    details = {"query_ids": q_ids, "methods": {}}
    metrics = {}
    for m in methods:
        summary, detail = method_metrics(rows, preds[m], ks)
        metrics[m] = summary
        details["methods"][m] = detail
        table_rows.append({"method": m, "label": METHOD_LABELS[m], **summary})

    out = {
        "meta": {
            "queries": len(rows),
            "corpus": len(corpus),
            "split_file": str(args.split_file),
            "corpus_file": str(args.corpus_file),
            "model_dir": str(resolved),
            "sparse_backend": args.sparse_backend,
            "sparse_max_features": int(args.sparse_max_features),
            "router_source": str(router_source),
            "methods": methods,
            "metrics_k": ks,
            "save_rankings_jsonl": str(args.save_rankings_jsonl) if args.save_rankings_jsonl is not None else None,
            "tessera_candidate_pool_k": int(args.tessera_candidate_pool_k),
            "tessera_retrieval_multi_agent": bool(args.tessera_retrieval_multi_agent),
            "tessera_retrieval_agent_pool_k": int(args.tessera_retrieval_agent_pool_k),
            "tessera_retrieval_dense_pool_k": int(args.tessera_retrieval_dense_pool_k),
            "tessera_retrieval_sparse_pool_k": int(args.tessera_retrieval_sparse_pool_k),
            "tessera_retrieval_preserve_top": int(args.tessera_retrieval_preserve_top),
            "tessera_retrieval_base_weight": float(args.tessera_retrieval_base_weight),
            "tessera_retrieval_dense_weight": float(args.tessera_retrieval_dense_weight),
            "tessera_retrieval_sparse_weight": float(args.tessera_retrieval_sparse_weight),
            "tessera_retrieval_target_weight": float(args.tessera_retrieval_target_weight),
            "tessera_retrieval_coverage_weight": float(args.tessera_retrieval_coverage_weight),
            "tessera_retrieval_diversity_weight": float(args.tessera_retrieval_diversity_weight),
            "tessera_retrieval_dense_rescue_k": int(args.tessera_retrieval_dense_rescue_k),
            "tessera_retrieval_dense_rescue_pool_k": int(args.tessera_retrieval_dense_rescue_pool_k),
            "tessera_retrieval_sibling_seed_k": int(args.tessera_retrieval_sibling_seed_k),
            "tessera_retrieval_sibling_window": int(args.tessera_retrieval_sibling_window),
            "tessera_retrieval_sibling_weight": float(args.tessera_retrieval_sibling_weight),
            "tessera_retrieval_moe": bool(args.tessera_retrieval_moe),
            "tessera_moe_pool_k": int(args.tessera_moe_pool_k),
            "tessera_moe_prf_seed_k": int(args.tessera_moe_prf_seed_k),
            "tessera_moe_prf_dense_seed_k": int(args.tessera_moe_prf_dense_seed_k),
            "tessera_moe_prf_sparse_seed_k": int(args.tessera_moe_prf_sparse_seed_k),
            "tessera_moe_prf_max_terms": int(args.tessera_moe_prf_max_terms),
            "tessera_moe_sibling_seed_k": int(args.tessera_moe_sibling_seed_k),
            "tessera_moe_sibling_window": int(args.tessera_moe_sibling_window),
            "tessera_moe_sibling_weight": float(args.tessera_moe_sibling_weight),
            "tessera_ser_ranker": str(args.tessera_ser_ranker) if args.tessera_ser_ranker is not None else None,
            "tessera_ser_loaded": bool(tessera_ser_bundle is not None),
            "tessera_ser_meta_summary": tessera_ser_meta_summary,
            "tessera_ser_candidate_pool_k": int(args.tessera_ser_candidate_pool_k),
            "tessera_ser_dense_pool_k": int(args.tessera_ser_dense_pool_k),
            "tessera_ser_sparse_pool_k": int(args.tessera_ser_sparse_pool_k),
            "tessera_ser_preserve_top": int(args.tessera_ser_preserve_top),
            "tessera_ser_blend_weight": float(args.tessera_ser_blend_weight),
            "tessera_ser_diversity_weight": float(args.tessera_ser_diversity_weight),
            "tessera_ser_evidence_rescue_k": int(args.tessera_ser_evidence_rescue_k),
            "tessera_ser_evidence_rescue_pool_k": int(args.tessera_ser_evidence_rescue_pool_k),
            "tessera_ser_evidence_preserve_top": int(args.tessera_ser_evidence_preserve_top),
            "tessera_ser_evidence_redundancy_weight": float(args.tessera_ser_evidence_redundancy_weight),
            "tessera_ser_evidence_min_gain": float(args.tessera_ser_evidence_min_gain),
            "tessera_ser_plan_adaptive": bool(args.tessera_ser_plan_adaptive),
            "tessera_ser_plan_dense_weight": float(args.tessera_ser_plan_dense_weight),
            "tessera_ser_plan_sparse_weight": float(args.tessera_ser_plan_sparse_weight),
            "tessera_ser_plan_lexical_weight": float(args.tessera_ser_plan_lexical_weight),
            "tessera_ser_plan_slot_weight": float(args.tessera_ser_plan_slot_weight),
            "tessera_ser_evidence_set_selection": bool(args.tessera_ser_evidence_set_selection),
            "tessera_ser_evidence_set_preserve_top": int(args.tessera_ser_evidence_set_preserve_top),
            "tessera_ser_evidence_set_pool_k": int(args.tessera_ser_evidence_set_pool_k),
            "tessera_ser_evidence_set_cardinality_threshold": float(args.tessera_ser_evidence_set_cardinality_threshold),
            "tessera_ser_evidence_set_learned_weight": float(args.tessera_ser_evidence_set_learned_weight),
            "tessera_ser_evidence_set_base_weight": float(args.tessera_ser_evidence_set_base_weight),
            "tessera_ser_evidence_set_dense_weight": float(args.tessera_ser_evidence_set_dense_weight),
            "tessera_ser_evidence_set_sparse_weight": float(args.tessera_ser_evidence_set_sparse_weight),
            "tessera_ser_evidence_set_probe_weight": float(args.tessera_ser_evidence_set_probe_weight),
            "tessera_ser_evidence_set_slot_weight": float(args.tessera_ser_evidence_set_slot_weight),
            "tessera_ser_evidence_set_anchor_weight": float(args.tessera_ser_evidence_set_anchor_weight),
            "tessera_ser_evidence_set_family_weight": float(args.tessera_ser_evidence_set_family_weight),
            "tessera_ser_evidence_set_redundancy_weight": float(args.tessera_ser_evidence_set_redundancy_weight),
            "tessera_graph_evidence_expansion": bool(args.tessera_graph_evidence_expansion),
            "tessera_gee_post_rerank": bool(args.tessera_gee_post_rerank),
            "tessera_gee_candidate_pool_k": int(args.tessera_gee_candidate_pool_k),
            "tessera_gee_dense_pool_k": int(args.tessera_gee_dense_pool_k),
            "tessera_gee_sparse_pool_k": int(args.tessera_gee_sparse_pool_k),
            "tessera_gee_graph_seed_k": int(args.tessera_gee_graph_seed_k),
            "tessera_gee_graph_window": int(args.tessera_gee_graph_window),
            "tessera_gee_preserve_top": int(args.tessera_gee_preserve_top),
            "tessera_gee_trigger_threshold": float(args.tessera_gee_trigger_threshold),
            "tessera_gee_base_weight": float(args.tessera_gee_base_weight),
            "tessera_gee_dense_weight": float(args.tessera_gee_dense_weight),
            "tessera_gee_sparse_weight": float(args.tessera_gee_sparse_weight),
            "tessera_gee_probe_weight": float(args.tessera_gee_probe_weight),
            "tessera_gee_graph_weight": float(args.tessera_gee_graph_weight),
            "tessera_gee_slot_weight": float(args.tessera_gee_slot_weight),
            "tessera_gee_sibling_weight": float(args.tessera_gee_sibling_weight),
            "tessera_gee_redundancy_weight": float(args.tessera_gee_redundancy_weight),
            "tessera_v9": bool(args.tessera_v9),
            "tessera_v9_local_rerank": bool(args.tessera_v9_local_rerank),
            "tessera_v9_dense_pool_k": int(args.tessera_v9_dense_pool_k),
            "tessera_v9_sparse_pool_k": int(args.tessera_v9_sparse_pool_k),
            "tessera_v9_candidate_pool_k": int(args.tessera_v9_candidate_pool_k),
            "tessera_v9_graph_seed_k": int(args.tessera_v9_graph_seed_k),
            "tessera_v9_graph_window": int(args.tessera_v9_graph_window),
            "tessera_v9_preserve_top": int(args.tessera_v9_preserve_top),
            "tessera_v9_base_weight": float(args.tessera_v9_base_weight),
            "tessera_v9_dense_weight": float(args.tessera_v9_dense_weight),
            "tessera_v9_sparse_weight": float(args.tessera_v9_sparse_weight),
            "tessera_v9_probe_weight": float(args.tessera_v9_probe_weight),
            "tessera_v9_graph_weight": float(args.tessera_v9_graph_weight),
            "tessera_v9_slot_weight": float(args.tessera_v9_slot_weight),
            "tessera_v9_diversity_weight": float(args.tessera_v9_diversity_weight),
            "tessera_v9_modality_weight": float(args.tessera_v9_modality_weight),
            "tessera_v10_conservative_rerank": bool(args.tessera_v10_conservative_rerank),
            "tessera_v10_preserve_top": int(args.tessera_v10_preserve_top),
            "tessera_v10_direct_preserve_top": int(args.tessera_v10_direct_preserve_top),
            "tessera_v10_reference_pool_k": int(args.tessera_v10_reference_pool_k),
            "tessera_v10_candidate_pool_k": int(args.tessera_v10_candidate_pool_k),
            "tessera_v10_reference_weight": float(args.tessera_v10_reference_weight),
            "tessera_v10_current_weight": float(args.tessera_v10_current_weight),
            "tessera_v10_base_weight": float(args.tessera_v10_base_weight),
            "tessera_v10_dense_weight": float(args.tessera_v10_dense_weight),
            "tessera_v10_sparse_weight": float(args.tessera_v10_sparse_weight),
            "tessera_v10_probe_weight": float(args.tessera_v10_probe_weight),
            "tessera_v10_slot_weight": float(args.tessera_v10_slot_weight),
            "tessera_v10_diversity_weight": float(args.tessera_v10_diversity_weight),
            "tessera_v10_margin": float(args.tessera_v10_margin),
            "tessera_v10_relevance_floor": float(args.tessera_v10_relevance_floor),
            "tessera_source_evidence_fusion": bool(args.tessera_source_evidence_fusion),
            "tessera_source_evidence_topk": int(args.tessera_source_evidence_topk),
            "tessera_source_evidence_candidate_pool_k": int(args.tessera_source_evidence_candidate_pool_k),
            "tessera_source_evidence_preserve_top": int(args.tessera_source_evidence_preserve_top),
            "tessera_source_evidence_base_weight": float(args.tessera_source_evidence_base_weight),
            "tessera_source_evidence_dense_weight": float(args.tessera_source_evidence_dense_weight),
            "tessera_source_evidence_sparse_weight": float(args.tessera_source_evidence_sparse_weight),
            "tessera_source_evidence_reference_weight": float(args.tessera_source_evidence_reference_weight),
            "tessera_source_evidence_lexical_weight": float(args.tessera_source_evidence_lexical_weight),
            "tessera_source_evidence_modality_prior_weight": float(args.tessera_source_evidence_modality_prior_weight),
            "tessera_source_evidence_source_balance_weight": float(args.tessera_source_evidence_source_balance_weight),
            "tessera_source_evidence_target_family_weight": float(args.tessera_source_evidence_target_family_weight),
            "tessera_source_evidence_diversity_weight": float(args.tessera_source_evidence_diversity_weight),
            "tessera_source_evidence_replacement_margin": float(args.tessera_source_evidence_replacement_margin),
            "tessera_source_evidence_min_candidate_score": float(args.tessera_source_evidence_min_candidate_score),
            "tessera_source_evidence_dense_guard": bool(args.tessera_source_evidence_dense_guard),
            "tessera_source_evidence_dense_guard_topn": int(args.tessera_source_evidence_dense_guard_topn),
            "tessera_source_evidence_dense_guard_prefixes": str(args.tessera_source_evidence_dense_guard_prefixes),
            "tessera_source_evidence_dense_guard_weight": float(args.tessera_source_evidence_dense_guard_weight),
            "tessera_source_evidence_dense_rank_weight": float(args.tessera_source_evidence_dense_rank_weight),
            "tessera_source_evidence_current_rank_weight": float(args.tessera_source_evidence_current_rank_weight),
            "tessera_source_evidence_source_balance_prefixes": str(args.tessera_source_evidence_source_balance_prefixes),
            "tessera_source_evidence_max_changed_slots": int(args.tessera_source_evidence_max_changed_slots),
            "tessera_source_evidence_slot_acceptance_guard": bool(args.tessera_source_evidence_slot_acceptance_guard),
            "tessera_source_evidence_slot_acceptance_prefixes": str(args.tessera_source_evidence_slot_acceptance_prefixes),
            "tessera_source_evidence_slot_acceptance_margin": float(args.tessera_source_evidence_slot_acceptance_margin),
            "tessera_source_evidence_budget_composer": bool(args.tessera_source_evidence_budget_composer),
            "tessera_source_evidence_budget_prefixes": str(args.tessera_source_evidence_budget_prefixes),
            "tessera_source_evidence_budget_candidate_pool_k": int(args.tessera_source_evidence_budget_candidate_pool_k),
            "tessera_source_evidence_budget_start_slot": int(args.tessera_source_evidence_budget_start_slot),
            "tessera_source_evidence_budget_max_selected": int(args.tessera_source_evidence_budget_max_selected),
            "tessera_source_evidence_budget_score_weight": float(args.tessera_source_evidence_budget_score_weight),
            "tessera_source_evidence_budget_sibling_weight": float(args.tessera_source_evidence_budget_sibling_weight),
            "tessera_source_evidence_budget_source_quota_weight": float(args.tessera_source_evidence_budget_source_quota_weight),
            "tessera_source_evidence_budget_tail_rank_weight": float(args.tessera_source_evidence_budget_tail_rank_weight),
            "tessera_source_evidence_budget_reference_weight": float(args.tessera_source_evidence_budget_reference_weight),
            "tessera_source_evidence_budget_margin": float(args.tessera_source_evidence_budget_margin),
            "tessera_source_evidence_budget_redundancy_weight": float(args.tessera_source_evidence_budget_redundancy_weight),
            "tessera_source_evidence_sibling_filler": bool(args.tessera_source_evidence_sibling_filler),
            "tessera_source_evidence_sibling_filler_prefixes": str(args.tessera_source_evidence_sibling_filler_prefixes),
            "tessera_source_evidence_sibling_filler_candidate_pool_k": int(args.tessera_source_evidence_sibling_filler_candidate_pool_k),
            "tessera_source_evidence_sibling_filler_start_slot": int(args.tessera_source_evidence_sibling_filler_start_slot),
            "tessera_source_evidence_sibling_filler_max_selected": int(args.tessera_source_evidence_sibling_filler_max_selected),
            "tessera_source_evidence_sibling_filler_tail_topn": int(args.tessera_source_evidence_sibling_filler_tail_topn),
            "tessera_source_evidence_sibling_filler_reference_topn": int(args.tessera_source_evidence_sibling_filler_reference_topn),
            "tessera_source_evidence_sibling_filler_margin": float(args.tessera_source_evidence_sibling_filler_margin),
            "tessera_source_evidence_sibling_filler_sibling_weight": float(args.tessera_source_evidence_sibling_filler_sibling_weight),
            "tessera_source_evidence_sibling_filler_reference_weight": float(args.tessera_source_evidence_sibling_filler_reference_weight),
            "tessera_source_evidence_sibling_filler_tail_weight": float(args.tessera_source_evidence_sibling_filler_tail_weight),
            "tessera_source_evidence_sibling_filler_dense_weight": float(args.tessera_source_evidence_sibling_filler_dense_weight),
            "tessera_source_evidence_sibling_filler_source_weight": float(args.tessera_source_evidence_sibling_filler_source_weight),
            "tessera_source_evidence_sibling_filler_redundancy_weight": float(args.tessera_source_evidence_sibling_filler_redundancy_weight),
            "tessera_source_evidence_slot_verifier": bool(args.tessera_source_evidence_slot_verifier),
            "tessera_source_evidence_slot_verifier_prefixes": str(args.tessera_source_evidence_slot_verifier_prefixes),
            "tessera_source_evidence_slot_verifier_candidate_pool_k": int(args.tessera_source_evidence_slot_verifier_candidate_pool_k),
            "tessera_source_evidence_slot_verifier_start_slot": int(args.tessera_source_evidence_slot_verifier_start_slot),
            "tessera_source_evidence_slot_verifier_max_selected": int(args.tessera_source_evidence_slot_verifier_max_selected),
            "tessera_source_evidence_slot_verifier_tail_topn": int(args.tessera_source_evidence_slot_verifier_tail_topn),
            "tessera_source_evidence_slot_verifier_reference_topn": int(args.tessera_source_evidence_slot_verifier_reference_topn),
            "tessera_source_evidence_slot_verifier_dense_topn": int(args.tessera_source_evidence_slot_verifier_dense_topn),
            "tessera_source_evidence_slot_verifier_margin": float(args.tessera_source_evidence_slot_verifier_margin),
            "tessera_source_evidence_slot_verifier_min_score": float(args.tessera_source_evidence_slot_verifier_min_score),
            "tessera_source_evidence_slot_verifier_model": str(args.tessera_source_evidence_slot_verifier_model)
            if args.tessera_source_evidence_slot_verifier_model is not None
            else None,
            "tessera_source_evidence_slot_verifier_model_loaded": bool(tessera_slot_verifier_bundle is not None),
            "tessera_source_evidence_slot_verifier_model_threshold": float(args.tessera_source_evidence_slot_verifier_model_threshold),
            "tessera_source_evidence_slot_verifier_static_weight": float(args.tessera_source_evidence_slot_verifier_static_weight),
            "tessera_source_evidence_slot_verifier_reference_weight": float(args.tessera_source_evidence_slot_verifier_reference_weight),
            "tessera_source_evidence_slot_verifier_dense_weight": float(args.tessera_source_evidence_slot_verifier_dense_weight),
            "tessera_source_evidence_slot_verifier_tail_weight": float(args.tessera_source_evidence_slot_verifier_tail_weight),
            "tessera_source_evidence_slot_verifier_sibling_weight": float(args.tessera_source_evidence_slot_verifier_sibling_weight),
            "tessera_source_evidence_slot_verifier_source_weight": float(args.tessera_source_evidence_slot_verifier_source_weight),
            "tessera_source_evidence_slot_verifier_lexical_weight": float(args.tessera_source_evidence_slot_verifier_lexical_weight),
            "tessera_source_evidence_slot_verifier_family_weight": float(args.tessera_source_evidence_slot_verifier_family_weight),
            "tessera_source_evidence_slot_verifier_redundancy_weight": float(args.tessera_source_evidence_slot_verifier_redundancy_weight),
            "tessera_source_evidence_kg_preservation_guard": bool(args.tessera_source_evidence_kg_preservation_guard),
            "tessera_source_evidence_kg_preservation_prefixes": str(args.tessera_source_evidence_kg_preservation_prefixes),
            "tessera_source_evidence_kg_preservation_min_kg": int(args.tessera_source_evidence_kg_preservation_min_kg),
            "tessera_source_evidence_kg_preservation_candidate_pool_k": int(args.tessera_source_evidence_kg_preservation_candidate_pool_k),
            "tessera_source_evidence_kg_preservation_start_slot": int(args.tessera_source_evidence_kg_preservation_start_slot),
            "tessera_source_evidence_kg_preservation_margin": float(args.tessera_source_evidence_kg_preservation_margin),
            "tessera_source_evidence_kg_preservation_reference_weight": float(args.tessera_source_evidence_kg_preservation_reference_weight),
            "tessera_source_evidence_kg_preservation_dense_weight": float(args.tessera_source_evidence_kg_preservation_dense_weight),
            "tessera_source_evidence_kg_preservation_current_weight": float(args.tessera_source_evidence_kg_preservation_current_weight),
            "tessera_source_evidence_kg_preservation_family_weight": float(args.tessera_source_evidence_kg_preservation_family_weight),
            "tessera_source_evidence_kg_preservation_lexical_weight": float(args.tessera_source_evidence_kg_preservation_lexical_weight),
            "tessera_source_evidence_kg_verifier_model": str(args.tessera_source_evidence_kg_verifier_model)
            if args.tessera_source_evidence_kg_verifier_model is not None
            else None,
            "tessera_source_evidence_kg_verifier_model_loaded": bool(tessera_kg_verifier_bundle is not None),
            "tessera_source_evidence_kg_verifier_meta_summary": tessera_kg_verifier_meta_summary,
            "tessera_source_evidence_kg_verifier_weight": float(args.tessera_source_evidence_kg_verifier_weight),
            "tessera_source_evidence_kg_verifier_min_score": float(args.tessera_source_evidence_kg_verifier_min_score),
            "tessera_source_budgeter_model": str(args.tessera_source_budgeter_model)
            if args.tessera_source_budgeter_model is not None
            else None,
            "tessera_source_budgeter_model_loaded": bool(tessera_source_budgeter_bundle is not None),
            "tessera_source_budgeter_meta_summary": tessera_source_budgeter_meta_summary,
            "tessera_source_budgeter_top1_guard": bool(args.tessera_source_budgeter_top1_guard),
            "tessera_source_budgeter_need_threshold": float(args.tessera_source_budgeter_need_threshold),
            "tessera_source_budgeter_non_kg_top1_max_kg": int(args.tessera_source_budgeter_non_kg_top1_max_kg),
            "tessera_source_head_selector": bool(args.tessera_source_head_selector),
            "tessera_source_head_topn": int(args.tessera_source_head_topn),
            "tessera_source_head_source_weight": float(args.tessera_source_head_source_weight),
            "tessera_source_head_same_query_weight": float(args.tessera_source_head_same_query_weight),
            "tessera_source_head_position_weight": float(args.tessera_source_head_position_weight),
            "tessera_source_head_reference_weight": float(args.tessera_source_head_reference_weight),
            "tessera_source_head_lexical_weight": float(args.tessera_source_head_lexical_weight),
            "tessera_source_head_base_weight": float(args.tessera_source_head_base_weight),
            "tessera_source_head_dense_weight": float(args.tessera_source_head_dense_weight),
            "tessera_source_head_sparse_weight": float(args.tessera_source_head_sparse_weight),
            "tessera_source_head_margin": float(args.tessera_source_head_margin),
            "tessera_source_head_off_source_margin": float(args.tessera_source_head_off_source_margin),
            "tessera_source_action_policy_model": str(args.tessera_source_action_policy_model)
            if args.tessera_source_action_policy_model is not None
            else None,
            "tessera_source_action_policy_model_loaded": bool(tessera_source_action_policy_bundle is not None),
            "tessera_source_action_policy_meta_summary": tessera_source_action_policy_meta_summary,
            "tessera_source_action_policy_min_prob": float(args.tessera_source_action_policy_min_prob),
            "tessera_source_action_policy_topk": int(args.tessera_source_action_policy_topk),
            "tessera_source_action_policy_pool_k": int(args.tessera_source_action_policy_pool_k),
            "tessera_final_evidence_composer": bool(args.tessera_final_evidence_composer),
            "tessera_final_evidence_topk": int(args.tessera_final_evidence_topk),
            "tessera_final_evidence_candidate_pool_k": int(args.tessera_final_evidence_candidate_pool_k),
            "tessera_final_evidence_dense_pool_k": int(args.tessera_final_evidence_dense_pool_k),
            "tessera_final_evidence_sparse_pool_k": int(args.tessera_final_evidence_sparse_pool_k),
            "tessera_final_evidence_preserve_top": int(args.tessera_final_evidence_preserve_top),
            "tessera_final_evidence_max_replacements": int(args.tessera_final_evidence_max_replacements),
            "tessera_final_evidence_min_candidate_score": float(args.tessera_final_evidence_min_candidate_score),
            "tessera_final_evidence_replacement_margin": float(args.tessera_final_evidence_replacement_margin),
            "tessera_final_evidence_min_query_overlap": float(args.tessera_final_evidence_min_query_overlap),
            "tessera_final_evidence_source_need_weight": float(args.tessera_final_evidence_source_need_weight),
            "tessera_final_evidence_redundancy_weight": float(args.tessera_final_evidence_redundancy_weight),
            "tessera_final_evidence_verifier_model": str(args.tessera_final_evidence_verifier_model)
            if args.tessera_final_evidence_verifier_model is not None
            else None,
            "tessera_final_evidence_verifier_model_loaded": bool(
                tessera_final_evidence_verifier_bundle is not None
            ),
            "tessera_final_evidence_verifier_meta_summary": tessera_final_evidence_verifier_meta_summary,
            "tessera_final_evidence_verifier_threshold": float(args.tessera_final_evidence_verifier_threshold),
            "tessera_final_evidence_verifier_margin": float(args.tessera_final_evidence_verifier_margin),
            "qrels_positive_total": int(qrels_total),
            "qrels_positive_in_corpus": int(qrels_in_corpus),
            "qrels_coverage_in_corpus": float(qrels_in_corpus / max(1, qrels_total)),
            "hits_definition": "average count of relevant chunks in top-k; can exceed 1.0",
            "map_definition": "truncated AP@k divided by total positive qrels for the query",
            "ndcg_definition": "graded NDCG using qrel labels >0 and gains 2^label-1",
        },
        "methods": metrics,
    }

    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    write_csv(args.out_csv, table_rows, metric_keys)
    write_markdown(args.out_md, table_rows, metric_keys)
    if args.detail_json is not None:
        args.detail_json.parent.mkdir(parents=True, exist_ok=True)
        args.detail_json.write_text(json.dumps(details, ensure_ascii=False, indent=2), encoding="utf-8")

    print(json.dumps(out, ensure_ascii=False, indent=2))
    print(f"[OK] json -> {args.out_json}")
    print(f"[OK] csv  -> {args.out_csv}")
    print(f"[OK] md   -> {args.out_md}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
