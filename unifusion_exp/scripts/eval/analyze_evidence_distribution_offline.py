#!/usr/bin/env python3
"""
Offline Evidence Distribution Analysis for UniFusion-RAG.
Computes metrics from qrels and corpus without re-running evaluation.
"""
from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from unifusion_exp.e2e.baselines import source_bucket
from unifusion_exp.e2e.metrics import exact_match, f1_score, normalize_answer


def load_data(split_file: str, corpus_file: str):
    with open(split_file) as f:
        queries = json.load(f)
    with open(corpus_file) as f:
        corpus = json.load(f)
    doc_map = {d["id"]: d for d in corpus}
    return queries, corpus, doc_map


def analyze_evidence_distribution(queries, doc_map):
    """Analyze the distribution of evidence in qrels."""
    results = {}

    # 1. Per-query positive doc count by modality
    per_query_modality = []
    all_positive_text_lengths = []
    all_negative_text_lengths = []
    
    for q in queries:
        pos_docs = [doc_id for doc_id, rel in q["relevant_chunks"].items() if rel == 1]
        neg_docs = [doc_id for doc_id, rel in q["relevant_chunks"].items() if rel == 0]
        
        mod_counts = {"text": 0, "table": 0, "kg": 0}
        for doc_id in pos_docs:
            mod = source_bucket(doc_id)
            if mod in mod_counts:
                mod_counts[mod] += 1
        per_query_modality.append(mod_counts)
        
        for doc_id in pos_docs:
            if doc_id in doc_map:
                all_positive_text_lengths.append(len(doc_map[doc_id]["text"].split()))
        for doc_id in neg_docs:
            if doc_id in doc_map:
                all_negative_text_lengths.append(len(doc_map[doc_id]["text"].split()))

    # Aggregate modality stats
    total_text = sum(p["text"] for p in per_query_modality)
    total_table = sum(p["table"] for p in per_query_modality)
    total_kg = sum(p["kg"] for p in per_query_modality)
    total_pos = total_text + total_table + total_kg
    
    results["positive_doc_modality_distribution"] = {
        "text_ratio": total_text / total_pos if total_pos else 0,
        "table_ratio": total_table / total_pos if total_pos else 0,
        "kg_ratio": total_kg / total_pos if total_pos else 0,
        "total_positive_docs": total_pos,
    }

    # Per-query modality coverage patterns
    queries_with_text_only = sum(1 for p in per_query_modality if p["table"] == 0 and p["kg"] == 0 and p["text"] > 0)
    queries_with_table_only = sum(1 for p in per_query_modality if p["text"] == 0 and p["kg"] == 0 and p["table"] > 0)
    queries_with_kg_only = sum(1 for p in per_query_modality if p["text"] == 0 and p["table"] == 0 and p["kg"] > 0)
    queries_with_multi_modal = sum(1 for p in per_query_modality if (p["text"] > 0) + (p["table"] > 0) + (p["kg"] > 0) >= 2)
    
    results["query_modality_coverage_patterns"] = {
        "text_only": queries_with_text_only,
        "table_only": queries_with_table_only,
        "kg_only": queries_with_kg_only,
        "multi_modal": queries_with_multi_modal,
        "total": len(queries),
    }

    # 2. Text length distribution
    results["positive_doc_text_length"] = {
        "mean": float(np.mean(all_positive_text_lengths)) if all_positive_text_lengths else 0,
        "median": float(np.median(all_positive_text_lengths)) if all_positive_text_lengths else 0,
        "p25": float(np.percentile(all_positive_text_lengths, 25)) if all_positive_text_lengths else 0,
        "p75": float(np.percentile(all_positive_text_lengths, 75)) if all_positive_text_lengths else 0,
        "p95": float(np.percentile(all_positive_text_lengths, 95)) if all_positive_text_lengths else 0,
        "count": len(all_positive_text_lengths),
    }
    results["negative_doc_text_length"] = {
        "mean": float(np.mean(all_negative_text_lengths)) if all_negative_text_lengths else 0,
        "median": float(np.median(all_negative_text_lengths)) if all_negative_text_lengths else 0,
        "count": len(all_negative_text_lengths),
    }

    # 3. Per-modality answerability (simple heuristic: does answer text appear in positive doc?)
    modality_answerability = {"text": {"em": 0, "f1": 0, "count": 0},
                              "table": {"em": 0, "f1": 0, "count": 0},
                              "kg": {"em": 0, "f1": 0, "count": 0}}
    
    for q in queries:
        gold = normalize_answer(str(q.get("answer", "")))
        if not gold:
            continue
        
        pos_docs = [doc_id for doc_id, rel in q["relevant_chunks"].items() if rel == 1]
        for doc_id in pos_docs:
            if doc_id not in doc_map:
                continue
            doc_text = normalize_answer(doc_map[doc_id]["text"])
            mod = source_bucket(doc_id)
            if mod not in modality_answerability:
                continue
            
            em = exact_match(gold, doc_text)
            f1 = f1_score(gold, doc_text)
            modality_answerability[mod]["em"] += em
            modality_answerability[mod]["f1"] += f1
            modality_answerability[mod]["count"] += 1

    for mod, vals in modality_answerability.items():
        if vals["count"] > 0:
            vals["em_rate"] = vals["em"] / vals["count"]
            vals["f1_rate"] = vals["f1"] / vals["count"]
        else:
            vals["em_rate"] = 0
            vals["f1_rate"] = 0
    
    results["per_modality_answerability_in_positive_docs"] = modality_answerability

    # 4. Budget constraint analysis
    # How many positive docs per query? What's the distribution?
    pos_counts = [sum(p.values()) for p in per_query_modality]
    results["positive_doc_count_per_query"] = {
        "mean": float(np.mean(pos_counts)),
        "median": float(np.median(pos_counts)),
        "p25": float(np.percentile(pos_counts, 25)),
        "p75": float(np.percentile(pos_counts, 75)),
        "p95": float(np.percentile(pos_counts, 95)),
        "queries_with_0": sum(1 for c in pos_counts if c == 0),
        "queries_with_1": sum(1 for c in pos_counts if c == 1),
        "queries_with_2_5": sum(1 for c in pos_counts if 2 <= c <= 5),
        "queries_with_6_plus": sum(1 for c in pos_counts if c >= 6),
    }

    return results


def main():
    split_file = "/home/yongqi.yin/reaserch_paper/downloaded_resource/mmRAG/data/mmRAG_ds/mmrag_test.json"
    corpus_file = "/home/yongqi.yin/reaserch_paper/unifusion_exp/artifacts/retrieval/corpus_subset_v1.json"
    out_file = "/home/yongqi.yin/reaserch_paper/unifusion_exp/artifacts/results/submodular_full_q1286_20260424/evidence_distribution_offline.json"

    print("[1/2] Loading data...")
    queries, corpus, doc_map = load_data(split_file, corpus_file)

    print("[2/2] Analyzing evidence distribution...")
    results = analyze_evidence_distribution(queries, doc_map)

    with open(out_file, "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"\n[OK] Results saved to {out_file}")
    
    print("\n=== Positive Doc Modality Distribution ===")
    d = results["positive_doc_modality_distribution"]
    print(f"  Text: {d['text_ratio']:.1%}, Table: {d['table_ratio']:.1%}, KG: {d['kg_ratio']:.1%}")
    
    print("\n=== Query Modality Coverage Patterns ===")
    p = results["query_modality_coverage_patterns"]
    print(f"  Text-only queries: {p['text_only']} ({p['text_only']/p['total']:.1%})")
    print(f"  Table-only queries: {p['table_only']} ({p['table_only']/p['total']:.1%})")
    print(f"  KG-only queries: {p['kg_only']} ({p['kg_only']/p['total']:.1%})")
    print(f"  Multi-modal queries: {p['multi_modal']} ({p['multi_modal']/p['total']:.1%})")
    
    print("\n=== Positive Doc Text Length ===")
    l = results["positive_doc_text_length"]
    print(f"  Mean: {l['mean']:.1f} tokens, Median: {l['median']:.1f}, P95: {l['p95']:.1f}")
    
    print("\n=== Per-Modality Answerability in Positive Docs ===")
    for mod, vals in results["per_modality_answerability_in_positive_docs"].items():
        print(f"  {mod}: EM={vals['em_rate']:.4f}, F1={vals['f1_rate']:.4f}, n={vals['count']}")
    
    print("\n=== Positive Doc Count Per Query ===")
    c = results["positive_doc_count_per_query"]
    print(f"  Mean: {c['mean']:.2f}, Median: {c['median']:.0f}")
    print(f"  0 pos: {c['queries_with_0']}, 1 pos: {c['queries_with_1']}, 2-5 pos: {c['queries_with_2_5']}, 6+ pos: {c['queries_with_6_plus']}")


if __name__ == "__main__":
    main()
