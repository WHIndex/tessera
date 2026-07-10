#!/usr/bin/env python3
"""Error typology analysis for TESSERA-RAG vs Dense-Concat."""
import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from tessera_exp.e2e.baselines import source_bucket
from tessera_exp.e2e.metrics import exact_match, f1_score, normalize_answer


def categorize_error(gold: str, pred: str) -> str:
    gold_norm = normalize_answer(gold)
    pred_norm = normalize_answer(pred)
    em = exact_match(gold_norm, pred_norm)
    f1 = f1_score(gold_norm, pred_norm)
    if em >= 0.99:
        return "correct"
    elif f1 >= 0.5:
        return "partial"
    elif f1 > 0:
        return "wrong_type"
    else:
        return "completely_wrong"


def analyze_query_modality(query: dict) -> str:
    pos_docs = [doc_id for doc_id, rel in query["relevant_chunks"].items() if rel == 1]
    mods = set()
    for doc_id in pos_docs:
        mod = source_bucket(doc_id)
        if mod in ("text", "table", "kg"):
            mods.add(mod)
    if len(mods) == 0:
        return "unknown"
    elif len(mods) == 1:
        return list(mods)[0] + "_only"
    else:
        return "multi_modal"


def main():
    split_file = "/home/yongqi.yin/reaserch_paper/downloaded_resource/mmRAG/data/mmRAG_ds/mmrag_test.json"
    results_dir = "artifacts/results/table1c_e2e_20260328_llm_qwen25_campe_full1286_all_v2_ablationfix"
    out_file = f"{results_dir}/error_typology_analysis.json"
    
    with open(split_file) as f:
        queries = json.load(f)
    
    methods = {"dense_concat": "Dense-Concat", "tessera_rag": "TESSERA-RAG"}
    preds = {}
    for method in methods:
        pred_file = f"{results_dir}/qa_predictions_{method}_test1286.jsonl"
        p = {}
        with open(pred_file) as f:
            for line in f:
                item = json.loads(line)
                p[item["id"]] = item["prediction"]
        preds[method] = p
    
    overall = {m: Counter() for m in methods}
    by_modality = {}
    comparative = {"uni_better": [], "dense_better": [], "both_correct": [], "both_wrong": []}
    
    for q in queries:
        qid = q["id"]
        gold = str(q.get("answer", ""))
        modality = analyze_query_modality(q)
        
        dense_pred = preds["dense_concat"].get(qid, "")
        uni_pred = preds["tessera_rag"].get(qid, "")
        
        dense_err = categorize_error(gold, dense_pred)
        uni_err = categorize_error(gold, uni_pred)
        
        overall["dense_concat"][dense_err] += 1
        overall["tessera_rag"][uni_err] += 1
        
        if modality not in by_modality:
            by_modality[modality] = {m: Counter() for m in methods}
        by_modality[modality]["dense_concat"][dense_err] += 1
        by_modality[modality]["tessera_rag"][uni_err] += 1
        
        if uni_err == "correct" and dense_err != "correct":
            comparative["uni_better"].append(qid)
        elif dense_err == "correct" and uni_err != "correct":
            comparative["dense_better"].append(qid)
        elif uni_err == "correct" and dense_err == "correct":
            comparative["both_correct"].append(qid)
        else:
            comparative["both_wrong"].append(qid)
    
    results = {
        "overall": {k: dict(v) for k, v in overall.items()},
        "by_modality": {mod: {m: dict(c) for m, c in vals.items()} for mod, vals in by_modality.items()},
        "comparative": {k: len(v) for k, v in comparative.items()},
        "comparative_qids": comparative,
    }
    
    with open(out_file, "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    print(f"[OK] Error typology saved to {out_file}\n")
    total = len(queries)
    print("=== Overall Error Distribution ===")
    for method, label in methods.items():
        print(f"\n{label}:")
        for err_type, count in sorted(overall[method].items(), key=lambda x: -x[1]):
            print(f"  {err_type:20s}: {count:4d} ({count/total*100:.1f}%)")
    
    print("\n=== Comparative (Exact Match) ===")
    for key, val in comparative.items():
        if key != "comparative_qids":
            print(f"  {key:20s}: {len(val):4d} ({len(val)/total*100:.1f}%)")
    
    print("\n=== Error by Modality ===")
    for mod in sorted(by_modality.keys()):
        mod_total = sum(by_modality[mod]["dense_concat"].values())
        print(f"\n{mod} (n={mod_total}):")
        for method, label in methods.items():
            correct = by_modality[mod][method].get("correct", 0)
            print(f"  {label} correct: {correct}/{mod_total} ({correct/mod_total*100:.1f}%)")

if __name__ == "__main__":
    main()
