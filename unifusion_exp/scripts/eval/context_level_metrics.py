#!/usr/bin/env python3
"""Compute context-level metrics from saved context doc IDs."""
import json
import sys
from collections import Counter
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from unifusion_exp.e2e.baselines import source_bucket


def jaccard(a: set, b: set) -> float:
    if not a and not b:
        return 1.0
    return len(a & b) / len(a | b)


def tokenize_simple(text: str) -> set[str]:
    return set(t.lower().strip(".,!?;:\"'()[]{}") for t in text.split() if len(t) > 1)


def load_context_docs(ctx_file: str):
    result = {}
    with open(ctx_file) as f:
        for line in f:
            item = json.loads(line)
            result[item["id"]] = item["context_doc_ids"]
    return result


def compute_metrics(queries, ctx_docs_dense, ctx_docs_uni, doc_map):
    metrics = {
        "modality_coverage": {},
        "redundancy": {},
        "qrel_match": {},
        "comparison": {},
    }
    
    # 1. Modality coverage
    for method_name, ctx_docs in [("dense_concat", ctx_docs_dense), ("unifusion_rag", ctx_docs_uni)]:
        mod_counts = Counter()
        per_query_mod_ratios = []
        
        for q in queries:
            qid = q["id"]
            docs = ctx_docs.get(qid, [])
            query_mods = Counter()
            for doc_id in docs:
                mod = source_bucket(doc_id)
                query_mods[mod] += 1
                mod_counts[mod] += 1
            
            if docs:
                ratios = {mod: count / len(docs) for mod, count in query_mods.items()}
                per_query_mod_ratios.append(ratios)
        
        # Aggregate
        total_docs = sum(mod_counts.values())
        avg_ratios = {}
        if per_query_mod_ratios:
            for mod in ["text", "table", "kg"]:
                avg_ratios[mod] = np.mean([r.get(mod, 0.0) for r in per_query_mod_ratios])
        
        metrics["modality_coverage"][method_name] = {
            "total_docs": total_docs,
            "overall_ratios": {mod: count / total_docs if total_docs else 0 for mod, count in mod_counts.items()},
            "avg_per_query_ratios": avg_ratios,
        }
    
    # 2. Redundancy (pairwise Jaccard of token sets)
    for method_name, ctx_docs in [("dense_concat", ctx_docs_dense), ("unifusion_rag", ctx_docs_uni)]:
        all_pairwise_jaccards = []
        
        for q in queries:
            qid = q["id"]
            docs = ctx_docs.get(qid, [])
            if len(docs) < 2:
                continue
            
            # Get token sets for each doc
            token_sets = []
            for doc_id in docs:
                text = doc_map.get(doc_id, "")
                token_sets.append(tokenize_simple(text))
            
            # Compute pairwise Jaccard
            pair_jaccards = []
            for i in range(len(token_sets)):
                for j in range(i + 1, len(token_sets)):
                    pair_jaccards.append(jaccard(token_sets[i], token_sets[j]))
            
            if pair_jaccards:
                all_pairwise_jaccards.extend(pair_jaccards)
        
        metrics["redundancy"][method_name] = {
            "mean_jaccard": float(np.mean(all_pairwise_jaccards)) if all_pairwise_jaccards else 0.0,
            "median_jaccard": float(np.median(all_pairwise_jaccards)) if all_pairwise_jaccards else 0.0,
            "p90_jaccard": float(np.percentile(all_pairwise_jaccards, 90)) if all_pairwise_jaccards else 0.0,
            "n_pairs": len(all_pairwise_jaccards),
        }
    
    # 3. Qrel match rate
    for method_name, ctx_docs in [("dense_concat", ctx_docs_dense), ("unifusion_rag", ctx_docs_uni)]:
        total_pos_in_ctx = 0
        total_queries_with_pos = 0
        
        for q in queries:
            qid = q["id"]
            pos_docs = set(doc_id for doc_id, rel in q["relevant_chunks"].items() if rel == 1)
            ctx = set(ctx_docs.get(qid, []))
            
            matched = len(pos_docs & ctx)
            total_pos_in_ctx += matched
            if matched > 0:
                total_queries_with_pos += 1
        
        metrics["qrel_match"][method_name] = {
            "total_positive_docs_in_context": total_pos_in_ctx,
            "queries_with_at_least_one_positive": total_queries_with_pos,
            "ratio_queries_with_positive": total_queries_with_pos / len(queries),
        }
    
    # 4. Comparison: overlap between Dense and UniFusion contexts
    overlap_ratios = []
    for q in queries:
        qid = q["id"]
        dense_ctx = set(ctx_docs_dense.get(qid, []))
        uni_ctx = set(ctx_docs_uni.get(qid, []))
        if dense_ctx or uni_ctx:
            overlap = len(dense_ctx & uni_ctx) / len(dense_ctx | uni_ctx)
            overlap_ratios.append(overlap)
    
    metrics["comparison"]["context_overlap"] = {
        "mean": float(np.mean(overlap_ratios)),
        "median": float(np.median(overlap_ratios)),
        "p25": float(np.percentile(overlap_ratios, 25)),
        "p75": float(np.percentile(overlap_ratios, 75)),
    }
    
    return metrics


def main():
    split_file = "/home/yongqi.yin/reaserch_paper/downloaded_resource/mmRAG/data/mmRAG_ds/mmrag_test.json"
    corpus_file = "/home/yongqi.yin/reaserch_paper/unifusion_exp/artifacts/retrieval/corpus_subset_v1.json"
    results_dir = "artifacts/results/context_docs_full1286_extractive_20260424"
    out_file = f"{results_dir}/context_level_metrics.json"
    
    with open(split_file) as f:
        queries = json.load(f)
    with open(corpus_file) as f:
        corpus = json.load(f)
    doc_map = {d["id"]: d["text"] for d in corpus}
    
    ctx_docs_dense = load_context_docs(f"{results_dir}/context_docs_dense_concat_test1286.jsonl")
    ctx_docs_uni = load_context_docs(f"{results_dir}/context_docs_unifusion_rag_test1286.jsonl")
    
    metrics = compute_metrics(queries, ctx_docs_dense, ctx_docs_uni, doc_map)
    
    with open(out_file, "w") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)
    
    print(f"[OK] Context-level metrics saved to {out_file}\n")
    
    print("=== Modality Coverage (per-query average) ===")
    for method in ["dense_concat", "unifusion_rag"]:
        label = "Dense-Concat" if method == "dense_concat" else "UniFusion-RAG"
        ratios = metrics["modality_coverage"][method]["avg_per_query_ratios"]
        print(f"  {label}: text={ratios.get('text', 0):.2%}, table={ratios.get('table', 0):.2%}, kg={ratios.get('kg', 0):.2%}")
    
    print("\n=== Redundancy (pairwise token Jaccard) ===")
    for method in ["dense_concat", "unifusion_rag"]:
        label = "Dense-Concat" if method == "dense_concat" else "UniFusion-RAG"
        r = metrics["redundancy"][method]
        print(f"  {label}: mean={r['mean_jaccard']:.4f}, median={r['median_jaccard']:.4f}, p90={r['p90_jaccard']:.4f}")
    
    print("\n=== Qrel Match ===")
    for method in ["dense_concat", "unifusion_rag"]:
        label = "Dense-Concat" if method == "dense_concat" else "UniFusion-RAG"
        q = metrics["qrel_match"][method]
        print(f"  {label}: {q['queries_with_at_least_one_positive']}/{len(queries)} queries have ≥1 positive ({q['ratio_queries_with_positive']:.1%})")
    
    print("\n=== Context Overlap (Dense vs UniFusion) ===")
    o = metrics["comparison"]["context_overlap"]
    print(f"  mean={o['mean']:.4f}, median={o['median']:.4f}, p25={o['p25']:.4f}, p75={o['p75']:.4f}")


if __name__ == "__main__":
    main()
