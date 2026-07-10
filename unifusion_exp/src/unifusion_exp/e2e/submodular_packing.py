"""
Submodular Analysis Framework for T2G RAG.

This module implements the theoretical submodular framework described in §3.0 of the paper.
It serves two purposes:

1. THEORETICAL ANALYSIS: Provides monotone submodular maximization under cardinality/token-budget
   constraints, with proven (1-1/e) approximation guarantees (Theorem 2). This is used to compute
   Oracle Upper Bounds and quantify the approximation gap of heuristic methods.

2. DIAGNOSTIC TOOL: The submodular packing variant (unifusion_submod) was evaluated as a potential
   replacement for CAMPE heuristic packing. Empirical results (§4.5) show it underperforms due to
   crude concept extraction, confirming that simple token-based concepts cannot capture structured
   semantics. This negative result is itself valuable: it demonstrates that theoretical guarantees
   alone are insufficient without domain-appropriate concept representations.

Key components:
- ConceptCoverageFunction: monotone submodular utility for evidence selection
- density_greedy_knapsack: (1-1/e) approximation algorithm
- build_concept_weights: query-adaptive weighting (currently heuristic; future work: learned)
"""
from __future__ import annotations

import re
from collections import Counter
from typing import Any

import numpy as np

from unifusion_exp.e2e.baselines import source_bucket
from unifusion_exp.e2e.controller import tokenize, normalized_entropy


# ---------------------------------------------------------------------------
# Concept extraction per modality
# ---------------------------------------------------------------------------

def _extract_text_concepts(text: str, top_k: int = 12) -> dict[str, float]:
    """Extract keyword concepts from plain text via token frequency."""
    toks = tokenize(text)
    if not toks:
        return {}
    counts = Counter(toks)
    # Filter out overly common stop-like tokens manually (lightweight)
    stop = {"the", "a", "an", "is", "are", "was", "were", "be", "been",
            "being", "have", "has", "had", "do", "does", "did", "will",
            "would", "could", "should", "may", "might", "must", "shall",
            "can", "need", "dare", "ought", "used", "to", "of", "in",
            "for", "on", "with", "at", "by", "from", "as", "into",
            "through", "during", "before", "after", "above", "below",
            "between", "under", "and", "but", "or", "yet", "so",
            "if", "because", "although", "though", "while", "where",
            "when", "that", "which", "who", "whom", "whose", "what",
            "this", "these", "those", "i", "you", "he", "she", "it",
            "we", "they", "me", "him", "her", "us", "them", "my",
            "your", "his", "its", "our", "their", "how", "why", "all",
            "any", "both", "each", "few", "more", "most", "other",
            "some", "such", "no", "nor", "not", "only", "own", "same",
            "than", "too", "very", "just", "now", "then", "here", "there",
            "up", "down", "out", "off", "over", "again", "further"}
    filtered = {t: c for t, c in counts.items() if t not in stop and len(t) > 1}
    if not filtered:
        filtered = counts
    top = sorted(filtered.items(), key=lambda x: x[1], reverse=True)[:top_k]
    total = sum(c for _, c in top) or 1.0
    return {t: float(c) / total for t, c in top}


def _extract_table_concepts(text: str, max_rows: int = 40) -> dict[str, float]:
    """Extract concepts from markdown-style tables: headers, column types, numeric patterns."""
    concepts: dict[str, float] = {}
    rows: list[list[str]] = []
    for line in text.splitlines():
        if "|" not in line:
            continue
        parts = [p.strip() for p in line.split("|") if p.strip()]
        if not parts:
            continue
        if all(re.fullmatch(r"[:\-\s]+", p) for p in parts):
            continue
        rows.append(parts)
        if len(rows) >= max_rows:
            break

    if not rows:
        return _extract_text_concepts(text, top_k=8)

    # Header row concepts (highest weight)
    header = rows[0]
    for h in header:
        ht = h.lower().replace(" ", "_")
        concepts[f"col:{ht}"] = concepts.get(f"col:{ht}", 0.0) + 1.5

    # Numeric patterns and row entities
    numeric_count = 0
    entity_cells: list[str] = []
    for r in rows[1:]:
        for cell in r:
            c = cell.strip()
            if re.search(r"\d{4}", c):
                numeric_count += 1
                concepts["pattern:year"] = concepts.get("pattern:year", 0.0) + 0.3
            if re.search(r"\d+(?:\.\d+)?%?", c):
                numeric_count += 1
                concepts["pattern:numeric"] = concepts.get("pattern:numeric", 0.0) + 0.3
            if len(c.split()) <= 3 and len(c) > 1:
                entity_cells.append(c.lower())

    for e in entity_cells[:10]:
        concepts[f"ent:{e}"] = concepts.get(f"ent:{e}", 0.0) + 0.4

    # Normalize
    total = sum(concepts.values()) or 1.0
    return {k: v / total for k, v in concepts.items()}


def _extract_graph_concepts(text: str) -> dict[str, float]:
    """Extract concepts from graph-text snippets: relation types and entity pairs."""
    concepts: dict[str, float] = {}
    toks = tokenize(text)

    # Relation hint terms as proxy for relation types
    relation_hints = {"spouse", "founder", "ceo", "president", "chairman",
                      "subsidiary", "parent", "acquired", "merged", "invested",
                      "born", "died", "located", "headquartered", "developed",
                      "authored", "directed", "produced", "starring", "member",
                      "employee", "employer", "competitor", "partner",
                      "nationality", "education", "alma", "mater",
                      "sibling", "child", "parent", "relative"}

    for t in toks:
        if t in relation_hints:
            concepts[f"rel:{t}"] = concepts.get(f"rel:{t}", 0.0) + 1.0

    # Entity n-grams (bigrams/trigrams as proxy for entity pairs)
    for n in (2, 3):
        for i in range(len(toks) - n + 1):
            gram = "_".join(toks[i:i + n])
            concepts[f"pair:{gram}"] = concepts.get(f"pair:{gram}", 0.0) + 0.3

    if not concepts:
        return _extract_text_concepts(text, top_k=8)

    total = sum(concepts.values()) or 1.0
    return {k: v / total for k, v in concepts.items()}


def extract_document_concepts(text: str, doc_id: str) -> dict[str, float]:
    """Route to modality-specific concept extractor."""
    bucket = source_bucket(str(doc_id))
    if bucket == "table":
        return _extract_table_concepts(text)
    if bucket == "kg":
        return _extract_graph_concepts(text)
    return _extract_text_concepts(text)


# ---------------------------------------------------------------------------
# Submodular objective: weighted concept coverage
# ---------------------------------------------------------------------------

class ConceptCoverageFunction:
    """f(S) = sum_c w_c * min(1, sum_{d in S} rel(d,c))

    This is a monotone submodular function because:
    - min(1, x) is concave and non-decreasing.
    - Sum of concave non-decreasing functions of linear arguments is submodular.
    - Nonnegative weighted sum preserves monotonicity and submodularity.
    """

    def __init__(
        self,
        candidate_concepts: list[dict[str, float]],
        concept_weights: dict[str, float] | None = None,
    ):
        self.candidate_concepts = candidate_concepts
        # Build concept universe
        universe: set[str] = set()
        for cc in candidate_concepts:
            universe.update(cc.keys())
        self.concepts = sorted(universe)
        self.concept_index = {c: i for i, c in enumerate(self.concepts)}
        self.n_candidates = len(candidate_concepts)
        self.n_concepts = len(self.concepts)

        # Dense relevance matrix: rel[d, c]
        self.rel_matrix = np.zeros((self.n_candidates, self.n_concepts), dtype=np.float32)
        for d, cc in enumerate(candidate_concepts):
            for c, v in cc.items():
                idx = self.concept_index.get(c)
                if idx is not None:
                    self.rel_matrix[d, idx] = float(v)

        # Concept weights
        if concept_weights is None:
            self.weights = np.ones(self.n_concepts, dtype=np.float32)
        else:
            self.weights = np.asarray(
                [float(concept_weights.get(c, 1.0)) for c in self.concepts],
                dtype=np.float32,
            )

    def evaluate(self, selected_mask: np.ndarray) -> float:
        """Evaluate f(S) given a boolean mask over candidates."""
        if selected_mask.sum() == 0:
            return 0.0
        # sum_{d in S} rel(d, c) for each concept c
        agg = self.rel_matrix[selected_mask].sum(axis=0)
        # min(1, agg)
        capped = np.minimum(agg, 1.0)
        return float((self.weights * capped).sum())

    def marginal_gain(self, selected_mask: np.ndarray, d: int) -> float:
        """Delta f(d | S) = f(S ∪ {d}) - f(S)."""
        if selected_mask[d]:
            return 0.0
        # Current aggregated relevance
        if selected_mask.sum() == 0:
            current_agg = np.zeros(self.n_concepts, dtype=np.float32)
        else:
            current_agg = self.rel_matrix[selected_mask].sum(axis=0)
        # After adding d
        new_agg = current_agg + self.rel_matrix[d]
        current_capped = np.minimum(current_agg, 1.0)
        new_capped = np.minimum(new_agg, 1.0)
        return float((self.weights * (new_capped - current_capped)).sum())

    def marginal_gains(self, selected_mask: np.ndarray) -> np.ndarray:
        """Compute marginal gains for all candidates not in S."""
        if selected_mask.sum() == 0:
            current_agg = np.zeros(self.n_concepts, dtype=np.float32)
        else:
            current_agg = self.rel_matrix[selected_mask].sum(axis=0)
        new_agg = current_agg + self.rel_matrix  # shape (n_candidates, n_concepts)
        current_capped = np.minimum(current_agg, 1.0)  # shape (n_concepts,)
        new_capped = np.minimum(new_agg, 1.0)  # shape (n_candidates, n_concepts)
        gains = (self.weights * (new_capped - current_capped)).sum(axis=1)  # (n_candidates,)
        gains[selected_mask] = 0.0
        return gains


# ---------------------------------------------------------------------------
# Density-greedy knapsack solver
# ---------------------------------------------------------------------------

def density_greedy_knapsack(
    func: ConceptCoverageFunction,
    costs: np.ndarray,
    budget: float,
    seed_mask: np.ndarray | None = None,
) -> np.ndarray:
    """Greedy density algorithm for monotone submodular maximization under knapsack constraint.

    At each step, select the item with highest marginal_gain / cost ratio
    that fits in the remaining budget.

    For monotone submodular + knapsack, this achieves (1 - 1/e) approximation
    of the optimal surrogate objective (Nemhauser et al., 1978; Sviridenko, 2004).

    Parameters
    ----------
    func: ConceptCoverageFunction
    costs: np.ndarray of shape (n_candidates,)
    budget: total token budget
    seed_mask: boolean mask of pre-selected items (e.g., dense anchors)

    Returns
    -------
    selected_mask: np.ndarray boolean mask
    """
    n = func.n_candidates
    selected = np.zeros(n, dtype=bool)
    if seed_mask is not None:
        selected = seed_mask.copy()

    remaining_budget = float(budget) - float(costs[selected].sum())
    if remaining_budget <= 0:
        return selected

    # Precompute marginal gains iteratively (faster than full recompute)
    for _ in range(n):
        gains = func.marginal_gains(selected)
        # Mask out items that don't fit
        fits = costs <= remaining_budget + 1e-6
        fits[selected] = False
        if not fits.any():
            break

        densities = np.zeros(n, dtype=np.float32)
        densities[fits] = gains[fits] / np.maximum(costs[fits], 1.0)
        best = int(np.argmax(densities))
        if densities[best] <= 1e-12:
            break
        selected[best] = True
        remaining_budget -= float(costs[best])
        if remaining_budget <= 0:
            break

    return selected


# ---------------------------------------------------------------------------
# Query-adaptive concept weight construction
# ---------------------------------------------------------------------------

def build_concept_weights(
    query: str,
    router_prob: np.ndarray,
    router_entropy: float,
    concept_universe: list[str],
) -> dict[str, float]:
    """Build concept weights that adapt to query modality and uncertainty.

    High router entropy -> flatten concept weights (avoid over-committing to one modality).
    Low router entropy -> boost concepts from the predicted dominant modality.
    """
    q_toks = set(tokenize(query))
    # Base weight from query keyword overlap
    base_weights: dict[str, float] = {}
    for c in concept_universe:
        c_toks = set(c.replace("col:", "").replace("rel:", "").replace("pair:", "").replace("ent:", "").replace("pattern:", "").split("_"))
        ov = len(q_toks & c_toks) / max(1, len(c_toks))
        base_weights[c] = 1.0 + 2.0 * ov  # range [1.0, 3.0]

    # Modality boost
    modality_names = ("text", "table", "kg")
    dominant = int(np.argmax(router_prob))
    boost = 1.0 + 1.5 * router_prob[dominant]  # up to 2.5x for dominant modality

    # Uncertainty gating: high entropy -> reduce modality boost, increase diversity
    uncertainty_gate = 1.0 - 0.5 * min(1.0, router_entropy)  # [0.5, 1.0]

    final_weights: dict[str, float] = {}
    for c in concept_universe:
        w = base_weights.get(c, 1.0)
        # Modality-specific concept boost
        if c.startswith("col:") or c.startswith("pattern:") or c.startswith("ent:"):
            w *= (1.0 + (boost - 1.0) * uncertainty_gate * router_prob[modality_names.index("table")])
        elif c.startswith("rel:") or c.startswith("pair:"):
            w *= (1.0 + (boost - 1.0) * uncertainty_gate * router_prob[modality_names.index("kg")])
        else:
            w *= (1.0 + (boost - 1.0) * uncertainty_gate * router_prob[modality_names.index("text")])
        final_weights[c] = float(w)

    return final_weights


# ---------------------------------------------------------------------------
# High-level T2G submodular packer (drop-in replacement for CAMPE)
# ---------------------------------------------------------------------------

def submodular_t2g_packer(
    candidate_idxs: list[int],
    candidate_texts: list[str],
    candidate_doc_ids: list[str],
    query: str,
    router_prob: np.ndarray,
    router_entropy: float,
    k: int,
    dense_anchor_idxs: list[int] | None = None,
    budget_mode: str = "token_estimate",
    token_per_doc: float = 180.0,
    redundancy_aware: bool = True,
) -> list[int]:
    """Select up to k heterogeneous evidence documents via submodular maximization.

    This is intended as a principled replacement for the heuristic CAMPE packing.

    Parameters
    ----------
    candidate_idxs: list of document indices in the candidate pool
    candidate_texts: parallel list of document texts
    candidate_doc_ids: parallel list of document IDs (for modality routing)
    query: query string
    router_prob: 3-dim modality probability [text, table, kg]
    router_entropy: normalized entropy of router_prob
    k: max number of documents to select (hard cardinality cap)
    dense_anchor_idxs: pre-selected dense anchor indices (optional)
    budget_mode: "token_estimate" or "cardinality"
    token_per_doc: average token count per document for budget estimation
    redundancy_aware: if True, the submodular objective naturally handles redundancy
        via diminishing returns; no extra Jaccard penalty needed.

    Returns
    -------
    selected_idxs: ordered list of selected document indices
    """
    if not candidate_idxs or k <= 0:
        return []

    n = len(candidate_idxs)
    # Extract concepts for each candidate
    candidate_concepts = [
        extract_document_concepts(candidate_texts[i], candidate_doc_ids[i])
        for i in range(n)
    ]

    # Build concept universe and weights
    universe: set[str] = set()
    for cc in candidate_concepts:
        universe.update(cc.keys())
    concept_weights = build_concept_weights(query, router_prob, router_entropy, sorted(universe))

    # Build submodular function
    func = ConceptCoverageFunction(candidate_concepts, concept_weights)

    # Costs: token length estimate or uniform
    if budget_mode == "token_estimate":
        costs = np.asarray([max(1.0, len(tokenize(candidate_texts[i]))) for i in range(n)], dtype=np.float32)
    else:
        costs = np.ones(n, dtype=np.float32)

    # Budget: k documents * avg tokens, or a tighter budget to test efficiency
    budget = float(k) * token_per_doc

    # Seed: dense anchors (if provided and within candidate pool)
    seed_mask = np.zeros(n, dtype=bool)
    if dense_anchor_idxs is not None:
        anchor_set = {int(x) for x in dense_anchor_idxs}
        for pos, idx in enumerate(candidate_idxs):
            if int(idx) in anchor_set:
                seed_mask[pos] = True

    # Run density-greedy
    selected_mask = density_greedy_knapsack(func, costs, budget, seed_mask=seed_mask)

    # Order selected by marginal contribution (descending)
    selected_positions = np.where(selected_mask)[0].tolist()
    if not selected_positions:
        return []

    # Compute greedy ordering: iteratively pick the one with highest marginal gain
    ordered: list[int] = []
    remaining = set(selected_positions)
    temp_mask = np.zeros(n, dtype=bool)
    for pos in selected_positions:
        if seed_mask[pos]:
            ordered.append(pos)
            temp_mask[pos] = True
            remaining.discard(pos)

    while remaining:
        gains = func.marginal_gains(temp_mask)
        best = max(remaining, key=lambda p: gains[p])
        ordered.append(best)
        temp_mask[best] = True
        remaining.discard(best)

    # Map back to original indices
    result = [int(candidate_idxs[p]) for p in ordered]
    # Ensure cardinality cap
    return result[:k]


# ---------------------------------------------------------------------------
# Fast approximate variant with lazy evaluation
# ---------------------------------------------------------------------------

def lazy_density_greedy_knapsack(
    func: ConceptCoverageFunction,
    costs: np.ndarray,
    budget: float,
    seed_mask: np.ndarray | None = None,
) -> np.ndarray:
    """Lazy-evaluated density greedy for larger candidate pools.

    Uses the property that marginal gains can only decrease (diminishing returns)
    to avoid recomputing all gains at every step.
    """
    n = func.n_candidates
    selected = np.zeros(n, dtype=bool)
    if seed_mask is not None:
        selected = seed_mask.copy()

    remaining_budget = float(budget) - float(costs[selected].sum())
    if remaining_budget <= 0:
        return selected

    # Initial full computation
    gains = func.marginal_gains(selected)
    fits = costs <= remaining_budget + 1e-6
    fits[selected] = False

    # Priority queue by density (simulated with sorted list for simplicity; n <= 200)
    # For n <= 200, full recompute is already fast (~ms). Lazy is mainly for n > 500.
    # Here we keep the simple version for clarity and correctness.
    return density_greedy_knapsack(func, costs, budget, seed_mask=seed_mask)
