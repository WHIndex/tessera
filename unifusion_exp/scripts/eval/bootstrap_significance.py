#!/usr/bin/env python3
"""Paired bootstrap significance test for UniFusion-RAG vs baselines."""
import json
import sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from unifusion_exp.e2e.metrics import exact_match, f1_score, normalize_answer


def load_predictions(pred_file: str):
    preds = {}
    with open(pred_file) as f:
        for line in f:
            p = json.loads(line)
            preds[p["id"]] = p["prediction"]
    return preds


def bootstrap_paired(
    scores_a: np.ndarray,
    scores_b: np.ndarray,
    n_bootstrap: int = 5000,
    seed: int = 42,
):
    rng = np.random.RandomState(seed)
    n = len(scores_a)
    deltas = scores_b - scores_a
    observed = float(deltas.mean())

    boot_means = []
    for _ in range(n_bootstrap):
        idx = rng.randint(0, n, size=n)
        boot_means.append(float((scores_b[idx] - scores_a[idx]).mean()))

    boot_means = np.array(boot_means)
    ci_low = float(np.percentile(boot_means, 2.5))
    ci_high = float(np.percentile(boot_means, 97.5))

    # Two-sided p-value: proportion of bootstrap samples with opposite sign
    p_value = float(np.mean(boot_means <= 0)) * 2.0
    if p_value > 1.0:
        p_value = 2.0 * (1.0 - float(np.mean(boot_means <= 0)))

    return {
        "observed_delta": observed,
        "ci_95": [ci_low, ci_high],
        "p_value_two_sided": p_value,
        "n_samples": n,
        "n_bootstrap": n_bootstrap,
        "mean_a": float(scores_a.mean()),
        "mean_b": float(scores_b.mean()),
    }


def main():
    split_file = "/home/yongqi.yin/reaserch_paper/downloaded_resource/mmRAG/data/mmRAG_ds/mmrag_test.json"
    results_dir = "artifacts/results/table1c_e2e_20260328_llm_qwen25_campe_full1286_all_v2_ablationfix"
    out_file = f"{results_dir}/paired_bootstrap_results.json"

    with open(split_file) as f:
        queries = json.load(f)
    gold_map = {q["id"]: str(q.get("answer", "")) for q in queries}

    methods_to_compare = [
        ("dense_concat", "unifusion_rag", "UniFusion-RAG vs Dense-Concat"),
        ("dense_concat", "ablation_no_redundancy_e2e", "w/o Redundancy vs Dense-Concat"),
        ("unifusion_rag", "ablation_no_redundancy_e2e", "UniFusion-RAG vs w/o Redundancy"),
        ("unifusion_rag", "ablation_no_pathmaxsim_e2e", "UniFusion-RAG vs w/o PathMaxSim"),
    ]

    # Load all predictions
    all_preds = {}
    for method in ["dense_concat", "unifusion_rag", "ablation_no_redundancy_e2e", "ablation_no_pathmaxsim_e2e"]:
        pred_file = f"{results_dir}/qa_predictions_{method}_test1286.jsonl"
        all_preds[method] = load_predictions(pred_file)

    results = {}
    for method_a, method_b, label in methods_to_compare:
        preds_a = all_preds[method_a]
        preds_b = all_preds[method_b]

        f1_a, f1_b = [], []
        em_a, em_b = [], []

        for qid, gold in gold_map.items():
            pred_a = normalize_answer(preds_a.get(qid, ""))
            pred_b = normalize_answer(preds_b.get(qid, ""))
            gold_norm = normalize_answer(gold)

            f1_a.append(f1_score(gold_norm, pred_a))
            f1_b.append(f1_score(gold_norm, pred_b))
            em_a.append(exact_match(gold_norm, pred_a))
            em_b.append(exact_match(gold_norm, pred_b))

        f1_a = np.array(f1_a)
        f1_b = np.array(f1_b)
        em_a = np.array(em_a)
        em_b = np.array(em_b)

        results[label] = {
            "f1": bootstrap_paired(f1_a, f1_b),
            "em": bootstrap_paired(em_a, em_b),
        }

    with open(out_file, "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"[OK] Bootstrap results saved to {out_file}\n")
    for label, res in results.items():
        print(f"=== {label} ===")
        print(f"  F1: delta={res['f1']['observed_delta']:+.4f}, "
              f"CI=[{res['f1']['ci_95'][0]:+.4f}, {res['f1']['ci_95'][1]:+.4f}], "
              f"p={res['f1']['p_value_two_sided']:.4f}")
        print(f"  EM: delta={res['em']['observed_delta']:+.4f}, "
              f"CI=[{res['em']['ci_95'][0]:+.4f}, {res['em']['ci_95'][1]:+.4f}], "
              f"p={res['em']['p_value_two_sided']:.4f}")
        print()


if __name__ == "__main__":
    main()
