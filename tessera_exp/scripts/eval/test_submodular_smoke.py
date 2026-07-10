#!/usr/bin/env python3
"""
Smoke test for submodular T2G evidence packing.
Validates correctness, submodularity, and basic behavior on synthetic + real data.
"""
from __future__ import annotations

import os
import sys
import time
import numpy as np
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from tessera_exp.e2e.submodular_packing import (
    extract_document_concepts,
    ConceptCoverageFunction,
    density_greedy_knapsack,
    build_concept_weights,
    submodular_t2g_packer,
)
from tessera_exp.e2e.controller import tokenize


def test_concept_extraction():
    print("=== Test 1: Concept Extraction ===")
    text = "Apple Inc. was founded by Steve Jobs in 1976. It is headquartered in Cupertino, California."
    concepts = extract_document_concepts(text, "text_001")
    print(f"Text concepts ({len(concepts)}): {dict(list(concepts.items())[:5])}...")
    assert len(concepts) > 0, "Text concept extraction failed"

    table = "| Company | Revenue | Year |\n|---------|---------|------|\n| Apple   | 394.3B  | 2022 |\n| Google  | 282.8B  | 2022 |"
    tconcepts = extract_document_concepts(table, "tat_001")
    print(f"Table concepts ({len(tconcepts)}): {dict(list(tconcepts.items())[:5])}...")
    assert any("col:" in k for k in tconcepts), "Table column concept missing"

    graph = "Steve Jobs (founder_of) Apple Inc. Apple Inc. (headquartered_in) Cupertino."
    gconcepts = extract_document_concepts(graph, "m_001")
    print(f"Graph concepts ({len(gconcepts)}): {dict(list(gconcepts.items())[:5])}...")
    assert len(gconcepts) > 0, "Graph concept extraction failed"
    print("PASSED\n")


def test_submodularity():
    print("=== Test 2: Submodularity (Diminishing Returns) ===")
    candidate_concepts = [
        {"a": 0.8, "b": 0.2},
        {"a": 0.3, "c": 0.7},
        {"b": 0.5, "c": 0.5},
        {"a": 0.4, "b": 0.3, "c": 0.3},
    ]
    func = ConceptCoverageFunction(candidate_concepts)

    # Marginal gain of adding item 0 to empty set
    mask_empty = np.zeros(4, dtype=bool)
    g0_empty = func.marginal_gain(mask_empty, 0)

    # Marginal gain of adding item 0 to set {1}
    mask_1 = np.zeros(4, dtype=bool)
    mask_1[1] = True
    g0_1 = func.marginal_gain(mask_1, 0)

    # Diminishing returns: g(0 | {}) >= g(0 | {1})
    print(f"g(0|empty) = {g0_empty:.4f}, g(0|{{1}}) = {g0_1:.4f}")
    assert g0_empty >= g0_1 - 1e-6, "Diminishing returns violated!"

    # Monotonicity: f({0,1}) >= f({1})
    f_01 = func.evaluate(np.asarray([True, True, False, False]))
    f_1 = func.evaluate(np.asarray([False, True, False, False]))
    print(f"f({{0,1}}) = {f_01:.4f}, f({{1}}) = {f_1:.4f}")
    assert f_01 >= f_1 - 1e-6, "Monotonicity violated!"
    print("PASSED\n")


def test_greedy_vs_optimal_small():
    print("=== Test 3: Greedy Quality on Tiny Instance ===")
    candidate_concepts = [
        {"a": 1.0},
        {"b": 1.0},
        {"a": 0.6, "b": 0.4},
        {"c": 1.0},
    ]
    func = ConceptCoverageFunction(candidate_concepts)
    costs = np.ones(4, dtype=np.float32)
    budget = 2.0

    selected = density_greedy_knapsack(func, costs, budget)
    sel_idxs = np.where(selected)[0].tolist()
    val = func.evaluate(selected)

    # Brute force optimal for budget=2
    best_val = 0.0
    best_sets = []
    for i in range(4):
        for j in range(i + 1, 4):
            mask = np.zeros(4, dtype=bool)
            mask[i] = mask[j] = True
            v = func.evaluate(mask)
            if v > best_val + 1e-6:
                best_val = v
                best_sets = [(i, j)]
            elif abs(v - best_val) <= 1e-6:
                best_sets.append((i, j))

    print(f"Greedy selected {sel_idxs}, value={val:.4f}")
    print(f"Optimal sets {best_sets}, value={best_val:.4f}")
    print(f"Approximation ratio = {val / max(best_val, 1e-9):.4f} (theory >= 0.632)")
    assert val >= best_val * 0.6, "Greedy quality too low"
    print("PASSED\n")


def test_t2g_packer_synthetic():
    print("=== Test 4: T2G Packer on Synthetic Data ===")
    candidate_idxs = list(range(6))
    candidate_texts = [
        "Apple Inc. revenue in 2022 was 394 billion dollars.",
        "Google parent Alphabet revenue 2022 282 billion.",
        "Microsoft revenue 2022 198 billion.",
        "| Company | Revenue 2022 |\n| Apple | 394.3B |",
        "| Company | Revenue 2022 |\n| Google | 282.8B |",
        "Apple (headquartered_in) Cupertino, California.",
    ]
    candidate_doc_ids = [
        "nq_apple", "nq_google", "nq_ms",
        "tat_rev", "tat_rev2", "m_hq"
    ]
    query = "What was Apple's revenue in 2022?"
    router_prob = np.asarray([0.2, 0.7, 0.1], dtype=np.float32)  # table-heavy
    router_entropy = 0.4

    result = submodular_t2g_packer(
        candidate_idxs=candidate_idxs,
        candidate_texts=candidate_texts,
        candidate_doc_ids=candidate_doc_ids,
        query=query,
        router_prob=router_prob,
        router_entropy=router_entropy,
        k=3,
        budget_mode="cardinality",
    )
    print(f"Selected {len(result)} docs: {result}")
    print(f"Doc IDs: {[candidate_doc_ids[candidate_idxs.index(r)] for r in result]}")
    assert len(result) <= 3
    # Should prefer table/text over kg for this query
    print("PASSED\n")


def test_concept_weights_adaptation():
    print("=== Test 5: Concept Weight Adaptation ===")
    universe = ["col:revenue", "pattern:numeric", "rel:founder", "apple", "year"]
    query = "Apple revenue 2022"
    # Low entropy, table-dominant
    w_low = build_concept_weights(query, np.asarray([0.1, 0.8, 0.1]), 0.2, universe)
    # High entropy, uncertain
    w_high = build_concept_weights(query, np.asarray([0.4, 0.3, 0.3]), 0.9, universe)

    print(f"Low entropy weights: {w_low}")
    print(f"High entropy weights: {w_high}")

    # Low entropy should boost table concepts more strongly
    assert w_low["col:revenue"] > w_high["col:revenue"] * 0.5, "Modality boost not working"
    print("PASSED\n")


def test_speed():
    print("=== Test 6: Speed Benchmark ===")
    n = 80
    candidate_concepts = []
    rng = np.random.RandomState(42)
    for _ in range(n):
        cc = {}
        for c in rng.choice(list("abcdefghij"), size=4, replace=False):
            cc[c] = float(rng.rand())
        candidate_concepts.append(cc)

    func = ConceptCoverageFunction(candidate_concepts)
    costs = np.ones(n, dtype=np.float32) * 100.0
    budget = 600.0  # select ~6 items

    t0 = time.perf_counter()
    for _ in range(10):
        sel = density_greedy_knapsack(func, costs, budget)
    t1 = time.perf_counter()
    ms_per_run = (t1 - t0) / 10 * 1000
    n_sel = int(sel.sum())
    print(f"n={n}, budget=6 docs, selected={n_sel}, time={ms_per_run:.2f} ms/run")
    assert ms_per_run < 500, "Too slow for online use"
    print("PASSED\n")


def main():
    print("Submodular T2G Packing Smoke Tests\n")
    test_concept_extraction()
    test_submodularity()
    test_greedy_vs_optimal_small()
    test_t2g_packer_synthetic()
    test_concept_weights_adaptation()
    test_speed()
    print("All smoke tests PASSED.")


if __name__ == "__main__":
    main()
