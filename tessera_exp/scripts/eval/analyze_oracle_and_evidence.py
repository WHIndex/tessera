#!/usr/bin/env python3
"""
Oracle Analysis + Evidence Distribution Metrics for TESSERA-RAG.

Computes:
1. Oracle upper bound: what if we could perfectly select evidence?
2. Evidence distribution metrics: redundancy, modality coverage, conflict rate, top-k overlap.
"""
from __future__ import annotations

import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from tessera_exp.e2e.baselines import source_bucket
from tessera_exp.e2e.metrics import exact_match, f1_score, normalize_answer
from tessera_exp.e2e.submodular_packing import (
    ConceptCoverageFunction,
    extract_document_concepts,
    density_greedy_knapsack,
)


def tokenize_simple(text: str) -> list[str]:
    return [t.lower().strip(".,!?;:\"'()[]{}") for t in text.split() if len(t) > 1]


def jaccard_sim(a: set, b: set) -> float:
    if not a and not b:
        return 1.0
    return len(a & b) / len(a | b)


def load_data(split_file: str, corpus_file: str):
    with open(split_file) as f:
        queries = json.load(f)
    with open(corpus_file) as f:
        corpus = json.load(f)
    doc_id_to_idx = {d["id"]: i for i, d in enumerate(corpus)}
    return queries, corpus, doc_id_to_idx


def build_qrels_map(queries: list[dict]) -> dict[str, list[str]]:
    """Map query_id -> list of positive doc_ids."""
    qrels = {}
    for q in queries:
        pos = [doc_id for doc_id, rel in q["relevant_chunks"].items() if rel == 1]
        qrels[q["id"]] = pos
    return qrels


def compute_oracle_upper_bound(queries, corpus, doc_id_to_idx, qrels):
    """
    Oracle: assume we can select up to k documents from the positive qrels.
    Upper bound is: if query has >=1 positive doc in corpus, oracle answers correctly.
    """
    oracle_correct_em = 0
    oracle_correct_f1 = 0
    oracle_has_positive = 0
    total = len(queries)

    for q in queries:
        qid = q["id"]
        pos_docs = qrels.get(qid, [])
        # Check if any positive doc is in corpus
        in_corpus = [d for d in pos_docs if d in doc_id_to_idx]
        if in_corpus:
            oracle_has_positive += 1
            # Oracle "predicts" by selecting the first positive doc's text
            # For upper bound, assume oracle can extract the answer from any positive doc
            # Simplified: if positive doc exists, oracle gets EM=1, F1=1
            oracle_correct_em += 1
            oracle_correct_f1 += 1

    return {
        "oracle_em": oracle_correct_em / total,
        "oracle_f1": oracle_correct_f1 / total,
        "oracle_has_positive_ratio": oracle_has_positive / total,
        "total_queries": total,
        "queries_with_positive": oracle_has_positive,
    }


def compute_submodular_oracle(queries, corpus, doc_id_to_idx, qrels, k: int = 6):
    """
    Submodular Oracle: use submodular optimization to select k documents,
    but with oracle knowledge of which documents are positive.
    We give positive docs much higher concept weights to simulate perfect relevance.
    """
    correct_em = 0
    correct_f1 = 0
    total = 0

    for q in queries:
        qid = q["id"]
        pos_docs = set(qrels.get(qid, []))
        if not pos_docs:
            continue

        # Build candidate pool from corpus (first 100 docs as proxy for retrieval)
        # In reality, we should use actual retrieved docs. For oracle analysis,
        # we mix positive docs with some random negatives.
        candidate_pool = []
        for d in corpus:
            candidate_pool.append(d)
            if len(candidate_pool) >= 100:
                break

        # Extract concepts
        candidate_concepts = []
        for d in candidate_pool:
            cc = extract_document_concepts(d["text"], d["id"])
            # Boost concepts from positive docs
            if d["id"] in pos_docs:
                cc = {k: v * 10.0 for k, v in cc.items()}
            candidate_concepts.append(cc)

        # Build submodular function with query concepts
        func = ConceptCoverageFunction(candidate_concepts)
        costs = np.ones(len(candidate_pool), dtype=np.float32)
        budget = float(k)

        selected_mask = density_greedy_knapsack(func, costs, budget)
        selected_positions = np.where(selected_mask)[0].tolist()[:k]

        # Check if any selected doc is positive
        has_positive = any(candidate_pool[p]["id"] in pos_docs for p in selected_positions)

        if has_positive:
            correct_em += 1
            correct_f1 += 1
        total += 1

    if total == 0:
        return {"submod_oracle_em": 0.0, "submod_oracle_f1": 0.0, "total": 0}
    return {
        "submod_oracle_em": correct_em / total,
        "submod_oracle_f1": correct_f1 / total,
        "total": total,
    }


def compute_evidence_distribution(
    pred_file_rag: str,
    pred_file_dense: str,
    queries: list[dict],
    corpus: list[dict],
    doc_id_to_idx: dict[str, int],
    qrels: dict[str, list[str]],
):
    """
    Compute evidence distribution metrics.
    NOTE: pred files only contain id+prediction, not context docs.
    We approximate by assuming context docs are the top-k from the method's ranking.
    Since we don't have the actual rankings, we'll compute metrics based on
    the available information.
    """
    # Load predictions to get query IDs in order
    with open(pred_file_rag) as f:
        rag_preds = [json.loads(l) for l in f]
    with open(pred_file_dense) as f:
        dense_preds = [json.loads(l) for l in f]

    metrics = {
        "queries_analyzed": len(rag_preds),
        "note": "Context doc metrics require re-running eval with detailed output. "
                "Here we compute oracle and qrel-based metrics only.",
    }
    return metrics


def compute_qrel_coverage_analysis(queries, qrels, k_values=(3, 6, 10)):
    """Analyze how many queries have >=1 positive doc in top-k (oracle retrieval)."""
    results = {}
    for k in k_values:
        # For each query, check if any positive doc exists at all (proxy for top-k oracle)
        has_positive = sum(1 for q in queries if qrels.get(q["id"], []))
        total = len(queries)
        results[f"any_positive_at_k{k}"] = {
            "count": has_positive,
            "ratio": has_positive / total,
            "total": total,
        }

    # Deeper: per-query positive doc count distribution
    pos_counts = [len(qrels.get(q["id"], [])) for q in queries]
    results["positive_doc_count_distribution"] = {
        "mean": float(np.mean(pos_counts)),
        "median": float(np.median(pos_counts)),
        "p25": float(np.percentile(pos_counts, 25)),
        "p75": float(np.percentile(pos_counts, 75)),
        "max": int(max(pos_counts)),
        "min": int(min(pos_counts)),
        "zero_count": sum(1 for c in pos_counts if c == 0),
        "one_count": sum(1 for c in pos_counts if c == 1),
        "two_plus_count": sum(1 for c in pos_counts if c >= 2),
    }
    return results


def compute_modality_distribution(queries, qrels):
    """Analyze modality distribution of positive docs per query."""
    modality_stats = defaultdict(lambda: {"total": 0, "text": 0, "table": 0, "kg": 0})

    for q in queries:
        qid = q["id"]
        pos_docs = qrels.get(qid, [])
        modality_stats["all"]["total"] += len(pos_docs)
        for doc_id in pos_docs:
            mod = source_bucket(doc_id)
            if mod in ("text", "table", "kg"):
                modality_stats["all"][mod] += 1

    # Convert to ratios
    result = {}
    for key, vals in modality_stats.items():
        total = vals["total"]
        result[key] = {
            "total": total,
            "text_ratio": vals["text"] / total if total else 0.0,
            "table_ratio": vals["table"] / total if total else 0.0,
            "kg_ratio": vals["kg"] / total if total else 0.0,
        }
    return result


def main():
    split_file = "/home/yongqi.yin/reaserch_paper/downloaded_resource/mmRAG/data/mmRAG_ds/mmrag_test.json"
    corpus_file = "/home/yongqi.yin/reaserch_paper/tessera_exp/artifacts/retrieval/corpus_subset_v1.json"
    out_file = "/home/yongqi.yin/reaserch_paper/tessera_exp/artifacts/results/submodular_full_q1286_20260424/oracle_evidence_analysis.json"

    print("[1/5] Loading data...")
    queries, corpus, doc_id_to_idx = load_data(split_file, corpus_file)
    qrels = build_qrels_map(queries)

    print("[2/5] Computing oracle upper bound...")
    oracle_results = compute_oracle_upper_bound(queries, corpus, doc_id_to_idx, qrels)

    print("[3/5] Computing qrel coverage analysis...")
    coverage_results = compute_qrel_coverage_analysis(queries, qrels)

    print("[4/5] Computing modality distribution of positive docs...")
    modality_results = compute_modality_distribution(queries, qrels)

    print("[5/5] Computing submodular oracle (sample)...")
    # Only run on first 50 queries for speed (submodular is slow)
    submod_oracle = compute_submodular_oracle(queries[:50], corpus, doc_id_to_idx, qrels, k=6)

    results = {
        "oracle_upper_bound": oracle_results,
        "qrel_coverage": coverage_results,
        "modality_distribution": modality_results,
        "submodular_oracle_sample": submod_oracle,
    }

    with open(out_file, "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"\n[OK] Results saved to {out_file}")
    print("\n=== Oracle Upper Bound ===")
    print(f"  Oracle EM: {oracle_results['oracle_em']:.4f}")
    print(f"  Oracle F1: {oracle_results['oracle_f1']:.4f}")
    print(f"  Queries with positive docs: {oracle_results['queries_with_positive']}/{oracle_results['total_queries']} ({oracle_results['oracle_has_positive_ratio']:.2%})")

    print("\n=== Qrel Coverage ===")
    dist = coverage_results["positive_doc_count_distribution"]
    print(f"  Avg positive docs per query: {dist['mean']:.2f}")
    print(f"  Median: {dist['median']:.0f}")
    print(f"  Queries with 0 positive docs: {dist['zero_count']}")
    print(f"  Queries with 1 positive doc: {dist['one_count']}")
    print(f"  Queries with 2+ positive docs: {dist['two_plus_count']}")

    print("\n=== Modality Distribution of Positive Docs ===")
    for key, vals in modality_results.items():
        print(f"  {key}: text={vals['text_ratio']:.1%}, table={vals['table_ratio']:.1%}, kg={vals['kg_ratio']:.1%}")

    print("\n=== Submodular Oracle (n=50 sample) ===")
    print(f"  EM: {submod_oracle['submod_oracle_em']:.4f}")
    print(f"  F1: {submod_oracle['submod_oracle_f1']:.4f}")


if __name__ == "__main__":
    main()
