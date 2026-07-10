#!/usr/bin/env python3
from __future__ import annotations

import argparse
from functools import lru_cache
import hashlib
import importlib
import json
import os
import re
import time
from collections import defaultdict
from pathlib import Path
from typing import Callable

import numpy as np
import requests
from sklearn.feature_extraction.text import TfidfVectorizer

try:
    import torch
except Exception:  # pragma: no cover - optional dependency at runtime
    torch = None

try:
    from transformers import AutoModel, AutoModelForSequenceClassification, AutoTokenizer
except Exception:  # pragma: no cover - optional dependency at runtime
    AutoModel = None
    AutoModelForSequenceClassification = None
    AutoTokenizer = None

try:
    from transformers import TapasModel, TapasTokenizer
except Exception:  # pragma: no cover - optional dependency at runtime
    TapasModel = None
    TapasTokenizer = None

try:
    import pandas as pd
except Exception:  # pragma: no cover - optional dependency at runtime
    pd = None

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in os.sys.path:
    os.sys.path.insert(0, str(SRC))

mod = importlib.import_module("tessera_exp.utils.e5_embed")
load_e5 = mod.load_e5
encode_texts = mod.encode_texts
metrics_mod = importlib.import_module("tessera_exp.e2e.metrics")
normalize_answer = metrics_mod.normalize_answer
exact_match = metrics_mod.exact_match
f1_score = metrics_mod.f1_score
mmrag_official_generation_score = metrics_mod.mmrag_official_generation_score
percentile95_ms = metrics_mod.percentile95_ms

report_mod = importlib.import_module("tessera_exp.e2e.reporting")
build_table1c_markdown = report_mod.build_table1c_markdown

methods_mod = importlib.import_module("tessera_exp.e2e.methods")
METHOD_KEYS = methods_mod.METHOD_KEYS
METHOD_LABELS = methods_mod.METHOD_LABELS
METHOD_MODALITY_COVERAGE = methods_mod.METHOD_MODALITY_COVERAGE
METHOD_PRESETS = methods_mod.METHOD_PRESETS
resolve_selected_methods = methods_mod.resolve_selected_methods
resolve_reuse_methods = methods_mod.resolve_reuse_methods

baseline_mod = importlib.import_module("tessera_exp.e2e.baselines")
build_dense_concat_ranking = baseline_mod.build_dense_concat_ranking
build_naive_rag_ranking = baseline_mod.build_naive_rag_ranking
build_adapter_candidate_pool = baseline_mod.build_adapter_candidate_pool
build_carp_ranking = baseline_mod.build_carp_ranking
build_tablerag_ranking = baseline_mod.build_tablerag_ranking
build_quasar_ranking = baseline_mod.build_quasar_ranking
table_intent_gate = baseline_mod.table_intent_gate
source_bucket = baseline_mod.source_bucket

objectives_mod = importlib.import_module("tessera_exp.e2e.objectives")
blend_router_with_query_prior = objectives_mod.blend_router_with_query_prior
blend_router_with_query_prior_adaptive = objectives_mod.blend_router_with_query_prior_adaptive
infer_qa_target_type = objectives_mod.infer_qa_target_type
infer_upo_lite_concept = objectives_mod.infer_upo_lite_concept
infer_upo_lite_modality_prior = objectives_mod.infer_upo_lite_modality_prior
infer_query_intent_type = objectives_mod.infer_query_intent_type
estimate_query_complexity_level = objectives_mod.estimate_query_complexity_level
qa_objective_retrieval_score = objectives_mod.qa_objective_retrieval_score

controller_mod = importlib.import_module("tessera_exp.e2e.controller")
PlannerBundle = controller_mod.PlannerBundle
VerifierBundle = controller_mod.VerifierBundle
ConflictBundle = controller_mod.ConflictBundle
normalize_prob_vector = controller_mod.normalize_prob_vector

try:
    from tessera_exp.e2e.submodular_packing import submodular_t2g_packer
except Exception:
    submodular_t2g_packer = None  # type: ignore[misc,assignment]

try:
    policy_mod = importlib.import_module("tessera_exp.e2e.tessera_policy")
    TESSERAPolicyConfig = policy_mod.TESSERAPolicyConfig
    select_tessera_policy_context = policy_mod.select_tessera_policy_context
    TESSERARetrievalAgentConfig = policy_mod.TESSERARetrievalAgentConfig
    rerank_tessera_retrieval = policy_mod.rerank_tessera_retrieval
    TESSERAMoERetrievalConfig = policy_mod.TESSERAMoERetrievalConfig
    rerank_tessera_moe_retrieval = policy_mod.rerank_tessera_moe_retrieval
    TESSERARetryAgentConfig = policy_mod.TESSERARetryAgentConfig
    select_tessera_retry_context = policy_mod.select_tessera_retry_context
    TESSERATableNumberAgentConfig = policy_mod.TESSERATableNumberAgentConfig
    select_tessera_table_number_answer = policy_mod.select_tessera_table_number_answer
    is_no_evidence_answer = policy_mod.is_no_evidence_answer
except Exception:
    policy_mod = None
    TESSERAPolicyConfig = None
    select_tessera_policy_context = None
    TESSERARetrievalAgentConfig = None
    rerank_tessera_retrieval = None
    TESSERAMoERetrievalConfig = None
    rerank_tessera_moe_retrieval = None
    TESSERARetryAgentConfig = None
    select_tessera_retry_context = None
    TESSERATableNumberAgentConfig = None
    select_tessera_table_number_answer = None
    is_no_evidence_answer = None

try:
    ser_mod = importlib.import_module("tessera_exp.e2e.ser_ranker")
    SERRankerConfig = ser_mod.SERRankerConfig
    FinalEvidenceComposerConfig = ser_mod.FinalEvidenceComposerConfig
    load_ser_ranker_bundle = ser_mod.load_ser_ranker_bundle
    rerank_with_ser = ser_mod.rerank_with_ser
    compose_final_with_ser = ser_mod.compose_final_with_ser
except Exception:
    ser_mod = None
    SERRankerConfig = None
    FinalEvidenceComposerConfig = None
    load_ser_ranker_bundle = None
    rerank_with_ser = None
    compose_final_with_ser = None

try:
    graph_evidence_mod = importlib.import_module("tessera_exp.e2e.graph_evidence")
    GraphEvidenceConfig = graph_evidence_mod.GraphEvidenceConfig
    expand_graph_evidence_candidates = graph_evidence_mod.expand_graph_evidence_candidates
    expand_and_rerank_graph_evidence = graph_evidence_mod.expand_and_rerank_graph_evidence
except Exception:
    graph_evidence_mod = None
    GraphEvidenceConfig = None
    expand_graph_evidence_candidates = None
    expand_and_rerank_graph_evidence = None

try:
    v9_mod = importlib.import_module("tessera_exp.e2e.tessera_v9")
    V9CandidateConfig = v9_mod.V9CandidateConfig
    expand_v9_candidates = v9_mod.expand_v9_candidates
    rerank_v9_local_evidence = v9_mod.rerank_v9_local_evidence
except Exception:
    v9_mod = None
    V9CandidateConfig = None
    expand_v9_candidates = None
    rerank_v9_local_evidence = None

try:
    v10_mod = importlib.import_module("tessera_exp.e2e.tessera_v10")
    V10RerankConfig = v10_mod.V10RerankConfig
    apply_v10_conservative_gate = v10_mod.apply_v10_conservative_gate
except Exception:
    v10_mod = None
    V10RerankConfig = None
    apply_v10_conservative_gate = None

try:
    source_head_mod = importlib.import_module("tessera_exp.e2e.source_head_selector")
    SourceHeadSelectorConfig = source_head_mod.SourceHeadSelectorConfig
    apply_source_aware_head_selector = source_head_mod.apply_source_aware_head_selector
except Exception:
    source_head_mod = None
    SourceHeadSelectorConfig = None
    apply_source_aware_head_selector = None

try:
    source_evidence_mod = importlib.import_module("tessera_exp.e2e.source_evidence_fusion")
    SourceEvidenceFusionConfig = source_evidence_mod.SourceEvidenceFusionConfig
    apply_source_evidence_fusion = source_evidence_mod.apply_source_evidence_fusion
except Exception:
    source_evidence_mod = None
    SourceEvidenceFusionConfig = None
    apply_source_evidence_fusion = None

try:
    pesv_mod = importlib.import_module("tessera_exp.e2e.pairwise_slot_verifier")
    load_pairwise_slot_verifier_bundle = pesv_mod.load_pairwise_slot_verifier_bundle
except Exception:
    pesv_mod = None
    load_pairwise_slot_verifier_bundle = None

try:
    kgcv_mod = importlib.import_module("tessera_exp.e2e.kg_consistency_verifier")
    load_kg_consistency_bundle = kgcv_mod.load_kg_consistency_bundle
except Exception:
    kgcv_mod = None
    load_kg_consistency_bundle = None

try:
    source_budgeter_mod = importlib.import_module("tessera_exp.e2e.source_budgeter")
    SOURCE_BUDGET_LABELS = source_budgeter_mod.SOURCE_LABELS
    load_source_budgeter_bundle = source_budgeter_mod.load_source_budgeter_bundle
except Exception:
    source_budgeter_mod = None
    SOURCE_BUDGET_LABELS = ["text", "table", "kg"]
    load_source_budgeter_bundle = None

try:
    source_action_mod = importlib.import_module("tessera_exp.e2e.source_action_policy")
    SOURCE_ACTION_LABELS = source_action_mod.ACTION_LABELS
    load_source_action_policy_bundle = source_action_mod.load_source_action_policy_bundle
    apply_source_action_to_ranked_idxs = source_action_mod.apply_source_action_to_ranked_idxs
except Exception:
    source_action_mod = None
    SOURCE_ACTION_LABELS = []
    load_source_action_policy_bundle = None
    apply_source_action_to_ranked_idxs = None

eval_mod = importlib.import_module("tessera_exp.e2e.evaluation")
eval_positive_relevant_ids = eval_mod.positive_relevant_ids
build_query_modality_distribution = eval_mod.build_query_modality_distribution
evaluate_predictions = eval_mod.evaluate_predictions
write_predictions_jsonl = eval_mod.write_predictions_jsonl

ROUTER_LABELS = ["text", "table", "kg"]
ROUTER_LABEL_TO_IDX = {k: i for i, k in enumerate(ROUTER_LABELS)}

RELATION_HINT_TERMS = [
    "parent",
    "subsidiary",
    "competitor",
    "founded",
    "founded by",
    "owned",
    "acquired",
    "headquarter",
    "located",
    "spouse",
    "capital",
]
RELATION_HINT_ATOMS = {
    "parent",
    "subsidiary",
    "competitor",
    "founded",
    "owned",
    "acquired",
    "headquarter",
    "located",
    "spouse",
    "capital",
}
CONSISTENCY_KEYWORDS = {
    "revenue",
    "profit",
    "loss",
    "capacity",
    "population",
    "bankrupt",
    "winner",
    "champion",
    "price",
    "score",
    "year",
}

HEAVY_TOKEN_VEC_DIM = 48
HEAVY_GNN_VEC_DIM = HEAVY_TOKEN_VEC_DIM
HEAVY_GNN_LAYERS = 2

NUMERIC_LITERAL_RE = re.compile(
    r"\b\d{1,3}(?:,\d{3})*(?:\.\d+)?(?:\s*(?:million|billion|thousand))?\b",
    flags=re.IGNORECASE,
)
YEAR_LITERAL_RE = re.compile(r"\b(?:1[0-9]{3}|20[0-9]{2})\b")

SUBQUERY_STOPWORDS = {
    "the",
    "a",
    "an",
    "of",
    "to",
    "in",
    "on",
    "for",
    "from",
    "by",
    "with",
    "at",
    "as",
    "is",
    "are",
    "was",
    "were",
    "be",
    "been",
    "being",
    "what",
    "which",
    "who",
    "whom",
    "whose",
    "when",
    "where",
    "why",
    "how",
    "did",
    "does",
    "do",
    "that",
    "this",
    "these",
    "those",
    "and",
    "or",
}

LOCATION_HINT_TOKENS = {
    "city",
    "country",
    "state",
    "province",
    "district",
    "county",
    "region",
    "capital",
    "located",
    "location",
    "airport",
    "river",
    "mountain",
    "village",
    "town",
}


def tokenize(text: str) -> set[str]:
    return set(re.findall(r"[a-z0-9]+", text.lower()))


def tokenize_list(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", text.lower())


class LazyDocTokenStore:
    def __init__(self, doc_texts: list[str]):
        self.doc_texts = doc_texts
        self.cache: dict[int, set[str]] = {}

    def __len__(self) -> int:
        return len(self.doc_texts)

    def __getitem__(self, index: int) -> set[str]:
        i = int(index)
        if i < 0 or i >= len(self.doc_texts):
            raise IndexError(i)
        if i not in self.cache:
            self.cache[i] = tokenize(self.doc_texts[i])
        return self.cache[i]

    def cache_size(self) -> int:
        return len(self.cache)


class LazyDocPrefixTokenStore:
    def __init__(self, doc_tokens: LazyDocTokenStore):
        self.doc_tokens = doc_tokens
        self.cache: dict[int, set[str]] = {}

    def __len__(self) -> int:
        return len(self.doc_tokens)

    def __getitem__(self, index: int) -> set[str]:
        i = int(index)
        if i < 0 or i >= len(self.doc_tokens):
            raise IndexError(i)
        if i not in self.cache:
            self.cache[i] = {tok[:4] for tok in self.doc_tokens[i] if len(tok) >= 4}
        return self.cache[i]

    def cache_size(self) -> int:
        return len(self.cache)


def decompose_query_segments(query: str) -> list[set[str]]:
    raw_parts = re.split(r"[,;\.?]|\band\b|\bwhich\b|\bthat\b|\bwho\b|\bwhen\b|\bwhere\b", query.lower())
    segs: list[set[str]] = []
    seen: set[frozenset[str]] = set()
    for part in raw_parts:
        toks = [t for t in re.findall(r"[a-z0-9]+", part) if t not in SUBQUERY_STOPWORDS]
        if len(toks) < 2:
            continue
        seg = set(toks)
        key = frozenset(seg)
        if key in seen:
            continue
        seen.add(key)
        segs.append(seg)
    return segs


def lightweight_context_rerank(
    ranked_idxs: list[int],
    query_tokens: set[str],
    query_segments: list[set[str]],
    qa_target: str,
    upo_concept: str,
    upo_rerank_bonus: float,
    doc_ids: list[str],
    doc_tokens: list[set[str]],
    doc_signal_tokens: list[set[str]],
    doc_numeric_literals: list[set[str]],
    probs: dict[str, float],
    topn: int,
    weight: float,
) -> list[int]:
    n = min(len(ranked_idxs), max(0, int(topn)))
    w = min(1.0, max(0.0, float(weight)))
    if n <= 1 or w <= 0.0:
        return [int(j) for j in ranked_idxs]

    top = [int(j) for j in ranked_idxs[:n]]
    rest = [int(j) for j in ranked_idxs[n:]]
    query_signal = {x for x in query_tokens if (x in CONSISTENCY_KEYWORDS or x in RELATION_HINT_ATOMS)}

    scored: list[tuple[float, int, int]] = []
    for pos, j in enumerate(top):
        bucket = source_bucket(doc_ids[j])
        d_toks = doc_tokens[j]
        has_year = any(YEAR_LITERAL_RE.search(x) is not None for x in doc_numeric_literals[j])
        q_overlap = len(query_tokens & d_toks) / max(1, len(query_tokens)) if query_tokens else 0.0

        seg_cov = 0.0
        if query_segments:
            covered = 0
            for seg in query_segments:
                ov = len(seg & d_toks) / max(1, len(seg))
                if ov >= 0.45:
                    covered += 1
            seg_cov = covered / max(1, len(query_segments))

        signal_cov = 0.0
        if query_signal:
            signal_cov = len(query_signal & doc_signal_tokens[j]) / max(1, len(query_signal))

        target_bonus = 0.0
        if qa_target == "number":
            if doc_numeric_literals[j]:
                target_bonus += 0.72
            if bucket == "table":
                target_bonus += 0.20
            elif bucket == "kg":
                target_bonus -= 0.05
        elif qa_target == "year":
            if has_year:
                target_bonus += 0.60
            if bucket in {"text", "table"}:
                target_bonus += 0.14
            elif bucket == "kg":
                target_bonus -= 0.04
        elif qa_target == "location":
            if d_toks & LOCATION_HINT_TOKENS:
                target_bonus += 0.45
            if bucket in {"text", "table"}:
                target_bonus += 0.12
            elif bucket == "kg":
                target_bonus -= 0.05

        concept_bonus = 0.0
        concept = str(upo_concept).strip().lower()
        if concept == "relation":
            if doc_signal_tokens[j] & RELATION_HINT_ATOMS:
                concept_bonus += 0.42
            if bucket == "kg":
                concept_bonus += 0.16
        elif concept == "entity":
            if bucket in {"text", "kg"}:
                concept_bonus += 0.14
            elif bucket == "table":
                concept_bonus += 0.06
        elif concept == "open":
            if bucket == "text":
                concept_bonus += 0.10
        elif concept == "year":
            if has_year:
                concept_bonus += 0.40
            if bucket in {"text", "table"}:
                concept_bonus += 0.10

        target_bonus += max(0.0, float(upo_rerank_bonus)) * concept_bonus
        target_bonus = min(1.0, max(0.0, target_bonus))

        modality_prior = float(probs.get(bucket, 0.0))
        rerank_signal = (
            0.42 * q_overlap
            + 0.20 * seg_cov
            + 0.18 * signal_cov
            + 0.14 * target_bonus
            + 0.06 * modality_prior
        )
        rank_prior = 1.0 / float(pos + 1)
        final_score = (1.0 - w) * rank_prior + w * rerank_signal
        scored.append((float(final_score), -pos, j))

    scored.sort(key=lambda x: (x[0], x[1]), reverse=True)
    reranked_top = [j for _, _, j in scored]
    out = reranked_top + rest
    seen: set[int] = set()
    dedup: list[int] = []
    for j in out:
        if j in seen:
            continue
        seen.add(j)
        dedup.append(j)
    return dedup


def external_context_rerank(
    ranked_idxs: list[int],
    query: str,
    doc_texts: list[str],
    topn: int,
    endpoint: str,
    timeout: int,
) -> list[int]:
    n = min(len(ranked_idxs), max(0, int(topn)))
    if n <= 1:
        return [int(j) for j in ranked_idxs]

    top = [int(j) for j in ranked_idxs[:n]]
    rest = [int(j) for j in ranked_idxs[n:]]
    documents = [doc_texts[j] for j in top]
    payload = {"query": query, "documents": documents, "top_n": n}
    response = requests.post(endpoint, json=payload, timeout=timeout)
    response.raise_for_status()
    raw_payload = response.json()

    scores: list[float] | None = None
    if isinstance(raw_payload, dict):
        raw_results = raw_payload.get("results")
        if isinstance(raw_results, list) and raw_results:
            if isinstance(raw_results[0], dict):
                if len(raw_results) == len(documents):
                    if "index" in raw_results[0]:
                        score_map = {}
                        for item in raw_results:
                            if not isinstance(item, dict) or "index" not in item:
                                continue
                            score_map[int(item["index"])] = float(item.get("relevance_score", item.get("score", 0.0)))
                        if len(score_map) == len(documents):
                            scores = [score_map[i] for i in range(len(documents))]
                    elif "relevance_score" in raw_results[0] or "score" in raw_results[0]:
                        scores = [float(item.get("relevance_score", item.get("score"))) for item in raw_results]
            elif len(raw_results) == len(documents):
                scores = [float(item) for item in raw_results]

    if scores is None:
        raise ValueError(f"Unsupported rerank response payload from {endpoint!r}: {raw_payload!r}")
    if len(scores) != len(documents):
        raise ValueError(
            f"Rerank response length mismatch from {endpoint!r}: expected {len(documents)} scores, got {len(scores)}"
        )

    scored = [(float(score), -pos, j) for pos, (score, j) in enumerate(zip(scores, top))]
    scored.sort(key=lambda item: (item[0], item[1]), reverse=True)
    reranked_top = [j for _, _, j in scored]
    out = reranked_top + rest
    seen: set[int] = set()
    dedup: list[int] = []
    for j in out:
        if j in seen:
            continue
        seen.add(j)
        dedup.append(j)
    return dedup


@lru_cache(maxsize=300000)
def token_hash_vector(token: str, dim: int = HEAVY_TOKEN_VEC_DIM) -> np.ndarray:
    # Deterministic token embedding without extra model dependency.
    digest = hashlib.blake2b(token.encode("utf-8"), digest_size=16).digest()
    seed = int.from_bytes(digest[:8], byteorder="little", signed=False)
    rng = np.random.default_rng(seed)
    vec = rng.standard_normal(int(dim)).astype(np.float32)
    norm = float(np.linalg.norm(vec))
    if norm > 1e-9:
        vec = vec / norm
    return vec


def token_priority(token: str) -> float:
    score = 1.0
    if token.isdigit():
        score += 0.8
    if token in RELATION_HINT_ATOMS:
        score += 0.6
    score += min(0.6, 0.06 * max(0, len(token) - 4))
    return score


def normalize_numeric_literal(raw: str) -> str:
    val = str(raw).strip().lower().replace(",", "")
    return re.sub(r"\s+", " ", val)


def extract_numeric_literals(text: str) -> set[str]:
    out: set[str] = set()
    for m in NUMERIC_LITERAL_RE.finditer(str(text)):
        lit = normalize_numeric_literal(m.group(0))
        if lit:
            out.add(lit)
    return out


def pick_literal_consensus(top_scored_sentences: list[tuple[str, float]], pattern) -> str:
    support: dict[str, float] = {}
    surface: dict[str, str] = {}
    for sent, sent_sc in top_scored_sentences:
        seen_local: set[str] = set()
        for m in pattern.finditer(sent):
            raw = m.group(0)
            key = normalize_numeric_literal(raw)
            if not key or key in seen_local:
                continue
            seen_local.add(key)
            support[key] = support.get(key, 0.0) + max(0.0, float(sent_sc)) + 0.20
            if key not in surface:
                surface[key] = raw

    if not support:
        return ""
    best_key = max(support.items(), key=lambda kv: (kv[1], -len(kv[0])))[0]
    return surface.get(best_key, best_key)


def build_token_matrix(tokens: list[str], max_tokens: int) -> np.ndarray:
    if not tokens:
        return np.zeros((0, HEAVY_TOKEN_VEC_DIM), dtype=np.float32)
    uniq = []
    seen = set()
    for t in tokens:
        if t in seen or len(t) < 2:
            continue
        seen.add(t)
        uniq.append(t)
    if not uniq:
        return np.zeros((0, HEAVY_TOKEN_VEC_DIM), dtype=np.float32)
    uniq.sort(key=token_priority, reverse=True)
    sel = uniq[: max(1, int(max_tokens))]
    mats = [token_hash_vector(t) for t in sel]
    if not mats:
        return np.zeros((0, HEAVY_TOKEN_VEC_DIM), dtype=np.float32)
    return np.asarray(mats, dtype=np.float32)


def maxsim_score(query_mat: np.ndarray, item_mat: np.ndarray) -> float:
    if query_mat.size == 0 or item_mat.size == 0:
        return 0.0
    sim = query_mat @ item_mat.T
    return float(np.mean(np.max(sim, axis=1)))


def embed_token_set(token_set: set[str], max_tokens: int) -> np.ndarray | None:
    if not token_set:
        return None
    mat = build_token_matrix(list(token_set), max_tokens=max_tokens)
    if mat.size == 0:
        return None
    vec = np.mean(mat, axis=0)
    norm = float(np.linalg.norm(vec))
    if norm <= 1e-9:
        return None
    return (vec / norm).astype(np.float32)


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


def topk_indices(scores: np.ndarray, k: int) -> np.ndarray:
    if len(scores) <= k:
        return np.argsort(-scores)
    idx = np.argpartition(-scores, kth=k - 1)[:k]
    return idx[np.argsort(-scores[idx])]


def normalize_scores(values: np.ndarray) -> np.ndarray:
    if values.size == 0:
        return values
    lo = float(values.min())
    hi = float(values.max())
    if hi - lo < 1e-9:
        return np.zeros_like(values)
    return (values - lo) / (hi - lo)


def calibrate_component_scores(values: np.ndarray, mode: str, nonzero_only: bool = True) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float32)
    if arr.size == 0:
        return arr

    m = str(mode).lower().strip()
    if m in {"", "none"}:
        return arr

    if nonzero_only:
        mask = np.abs(arr) > 1e-9
        if int(mask.sum()) < 2:
            return np.zeros_like(arr)
        work = arr[mask]
    else:
        mask = np.ones(arr.shape, dtype=bool)
        work = arr

    if m == "minmax":
        norm = normalize_scores(work)
    elif m == "zscore":
        mu = float(np.mean(work))
        std = float(np.std(work))
        if std <= 1e-9:
            norm = np.zeros_like(work)
        else:
            z = (work - mu) / std
            z = np.clip(z, -6.0, 6.0)
            norm = 1.0 / (1.0 + np.exp(-z))
    elif m == "robust":
        med = float(np.median(work))
        mad = float(np.median(np.abs(work - med)))
        denom = max(1e-6, 1.4826 * mad)
        z = (work - med) / denom
        z = np.clip(z, -6.0, 6.0)
        norm = 1.0 / (1.0 + np.exp(-z))
    elif m == "rank":
        order = np.argsort(work)
        ranks = np.empty_like(order, dtype=np.float32)
        ranks[order] = np.arange(len(order), dtype=np.float32)
        norm = ranks / max(1.0, float(len(order) - 1))
    else:
        return arr

    out = np.zeros_like(arr)
    out[mask] = np.asarray(norm, dtype=np.float32)
    return out


def jaccard_overlap(a: set[str], b: set[str]) -> float:
    if not a or not b:
        return 0.0
    inter = len(a & b)
    if inter == 0:
        return 0.0
    return inter / max(1, len(a | b))


def greedy_diverse_topk(
    candidate_idxs: np.ndarray,
    base_scores: np.ndarray,
    doc_tokens: list[set[str]],
    k: int,
    redundancy_lambda: float,
    redundancy_mode: str = "exact",
    candidate_pool_k: int = 0,
) -> list[int]:
    if len(candidate_idxs) == 0 or k <= 0:
        return []
    pool_limit = min(int(candidate_pool_k), len(candidate_idxs))
    if pool_limit > 0 and len(candidate_idxs) > pool_limit:
        pool_limit = min(len(candidate_idxs), max(pool_limit, int(k)))
        cutoff = len(base_scores) - pool_limit
        top_pool_pos = np.argpartition(base_scores, cutoff)[cutoff:]
        top_pool_pos = top_pool_pos[np.argsort(base_scores[top_pool_pos])[::-1]]
        candidate_idxs = candidate_idxs[top_pool_pos]
        base_scores = base_scores[top_pool_pos]
    remaining = list(range(len(candidate_idxs)))
    selected_pos: list[int] = []
    selected_union: set[str] = set()
    fast_mode = str(redundancy_mode).lower().strip() in {"union", "fast", "approx"}
    while remaining and len(selected_pos) < k:
        best_pos = remaining[0]
        best_score = -1e9
        for pos in remaining:
            score = float(base_scores[pos])
            if redundancy_lambda > 0.0 and selected_pos:
                tok = doc_tokens[candidate_idxs[pos]]
                if fast_mode:
                    if selected_union:
                        score -= redundancy_lambda * (len(tok & selected_union) / max(1, len(tok)))
                else:
                    max_red = 0.0
                    for sp in selected_pos:
                        red = jaccard_overlap(tok, doc_tokens[candidate_idxs[sp]])
                        if red > max_red:
                            max_red = red
                    score -= redundancy_lambda * max_red
            if score > best_score:
                best_score = score
                best_pos = pos
        selected_pos.append(best_pos)
        if fast_mode and redundancy_lambda > 0.0:
            selected_union |= doc_tokens[candidate_idxs[best_pos]]
        remaining.remove(best_pos)
    return [int(candidate_idxs[p]) for p in selected_pos]


def relation_hint_bonus(query: str, doc_text: str, bucket: str) -> float:
    if bucket != "kg":
        return 0.0
    q = query.lower()
    d = doc_text.lower()
    q_hits = sum(1 for t in RELATION_HINT_TERMS if t in q)
    if q_hits == 0:
        return 0.0
    d_hits = sum(1 for t in RELATION_HINT_TERMS if t in d)
    if d_hits == 0:
        return 0.0
    return min(0.12, 0.02 * float(q_hits + d_hits))


def extract_table_structure_tokens(text: str, max_rows: int = 80, max_cells: int = 320) -> dict[str, list[set[str]]] | None:
    rows: list[list[set[str]]] = []
    for line in text.splitlines():
        if "|" not in line:
            continue
        parts = [p.strip() for p in line.split("|")]
        parts = [p for p in parts if p]
        if not parts:
            continue
        if all(re.fullmatch(r"[:\-\s]+", p) for p in parts):
            continue
        row_tok = [tokenize(p) for p in parts if p]
        row_tok = [x for x in row_tok if x]
        if not row_tok:
            continue
        rows.append(row_tok)
        if len(rows) >= max_rows:
            break

    if not rows:
        return None

    cell_sets: list[set[str]] = []
    row_sets: list[set[str]] = []
    max_cols = max(len(r) for r in rows)
    col_sets: list[set[str]] = [set() for _ in range(max_cols)]

    for r in rows:
        r_union: set[str] = set()
        for ci, cset in enumerate(r):
            if not cset:
                continue
            cell_sets.append(cset)
            r_union |= cset
            col_sets[ci] |= cset
            if len(cell_sets) >= max_cells:
                break
        if r_union:
            row_sets.append(r_union)
        if len(cell_sets) >= max_cells:
            break

    col_sets = [x for x in col_sets if x]
    if not cell_sets:
        return None
    header_set: set[str] = set()
    if rows and rows[0]:
        for c in rows[0]:
            header_set |= c
    return {"cell": cell_sets, "row": row_sets, "col": col_sets, "header": [header_set] if header_set else []}


def cellmaxsim_like_score(
    query_tokens: set[str],
    table_struct: dict[str, list[set[str]]] | None,
    top_cells: int,
) -> float:
    if not query_tokens or table_struct is None:
        return 0.0

    cell_sets = table_struct.get("cell", [])[: max(1, top_cells)]
    if not cell_sets:
        return 0.0
    row_sets = table_struct.get("row", [])
    col_sets = table_struct.get("col", [])

    cell_union: set[str] = set()
    for c in cell_sets:
        cell_union |= c
    token_coverage = len(query_tokens & cell_union) / max(1, len(query_tokens))

    max_cell = max(jaccard_overlap(query_tokens, c) for c in cell_sets)
    max_row = max((jaccard_overlap(query_tokens, r) for r in row_sets), default=0.0)
    max_col = max((jaccard_overlap(query_tokens, c) for c in col_sets), default=0.0)

    return 0.45 * token_coverage + 0.30 * max_cell + 0.15 * max_row + 0.10 * max_col

def table_schema_alignment_score(
    query_tokens: set[str],
    table_struct: dict[str, list[set[str]]] | None,
) -> float:
    if not query_tokens or table_struct is None:
        return 0.0
    header_sets = table_struct.get("header", [])
    if not header_sets:
        return 0.0
    header = header_sets[0]
    if not header:
        return 0.0
    cov = len(query_tokens & header) / max(1, len(query_tokens))
    return float(cov)


def token_maxsim_like_score(
    query_tokens: set[str],
    doc_token_set: set[str],
    doc_prefix4_set: set[str],
) -> float:
    if not query_tokens or not doc_token_set:
        return 0.0
    total_w = 0.0
    hit_w = 0.0
    for qt in query_tokens:
        w = 1.0 + min(0.30, 0.05 * max(0, len(qt) - 4))
        if qt.isdigit():
            w += 0.20
        total_w += w
        if qt in doc_token_set:
            hit_w += w
            continue
        if len(qt) >= 4 and qt[:4] in doc_prefix4_set:
            hit_w += 0.55 * w
    if total_w <= 1e-9:
        return 0.0
    return float(hit_w / total_w)


def table_cell_encoder_score(
    query_mat: np.ndarray,
    table_struct: dict[str, list[set[str]]] | None,
    max_cells: int,
) -> float:
    if query_mat.size == 0 or table_struct is None:
        return 0.0

    cell_sets = table_struct.get("cell", [])[: max(1, int(max_cells))]
    row_sets = table_struct.get("row", [])
    col_sets = table_struct.get("col", [])

    cell_vecs = []
    for s in cell_sets:
        v = embed_token_set(s, max_tokens=12)
        if v is not None:
            cell_vecs.append(v)
    if not cell_vecs:
        return 0.0
    cell_mat = np.asarray(cell_vecs, dtype=np.float32)
    cell_score = maxsim_score(query_mat, cell_mat)

    row_vecs = []
    for s in row_sets[:48]:
        v = embed_token_set(s, max_tokens=16)
        if v is not None:
            row_vecs.append(v)
    col_vecs = []
    for s in col_sets[:48]:
        v = embed_token_set(s, max_tokens=16)
        if v is not None:
            col_vecs.append(v)

    row_score = maxsim_score(query_mat, np.asarray(row_vecs, dtype=np.float32)) if row_vecs else 0.0
    col_score = maxsim_score(query_mat, np.asarray(col_vecs, dtype=np.float32)) if col_vecs else 0.0
    return 0.65 * cell_score + 0.20 * row_score + 0.15 * col_score


def extract_kg_path_sets(text: str, max_paths: int = 64) -> list[set[str]]:
    paths: list[set[str]] = []
    chunks = re.split(r"[\n;]+", text)
    for ch in chunks:
        low = ch.lower()
        has_rel = any(tok in low for tok in RELATION_HINT_ATOMS)
        has_path_marker = ("->" in ch) or ("|" in ch) or ("/m/" in ch)
        if not has_rel and not has_path_marker:
            continue
        toks = tokenize(ch)
        if 2 <= len(toks) <= 24:
            paths.append(toks)
            if len(paths) >= max_paths:
                break

    if paths:
        return paths[: max_paths]

    # Fallback: sliding windows over whole text to avoid empty path features.
    seq = tokenize_list(text)
    win = 8
    step = 4
    for i in range(0, max(0, len(seq) - win + 1), step):
        s = set(seq[i : i + win])
        if len(s) >= 3:
            paths.append(s)
            if len(paths) >= max_paths:
                break
    return paths[: max_paths]


def kg_path_encoder_score(query_mat: np.ndarray, path_sets: list[set[str]] | None) -> float:
    if query_mat.size == 0 or not path_sets:
        return 0.0
    vecs = []
    for s in path_sets[:64]:
        v = embed_token_set(s, max_tokens=18)
        if v is not None:
            vecs.append(v)
    if not vecs:
        return 0.0
    path_mat = np.asarray(vecs, dtype=np.float32)
    return maxsim_score(query_mat, path_mat)


def token_late_interaction_score(
    query_mat: np.ndarray,
    doc_token_list: list[str],
    max_doc_tokens: int = 96,
) -> float:
    if query_mat.size == 0 or not doc_token_list:
        return 0.0
    doc_mat = build_token_matrix(doc_token_list, max_tokens=max(24, int(max_doc_tokens)))
    return maxsim_score(query_mat, doc_mat)


def softmax_1d(values: np.ndarray, temp: float = 1.0) -> np.ndarray:
    x = np.asarray(values, dtype=np.float32)
    t = max(1e-6, float(temp))
    x = x / t
    x = x - float(np.max(x))
    e = np.exp(x)
    return e / max(1e-9, float(np.sum(e)))


def normalize_rows(mat: np.ndarray) -> np.ndarray:
    if mat.size == 0:
        return mat
    n = np.linalg.norm(mat, axis=1, keepdims=True)
    return mat / np.clip(n, 1e-9, None)


def parse_markdown_table_rows(text: str, max_rows: int = 80, max_cols: int = 24) -> list[list[str]]:
    rows: list[list[str]] = []
    for line in text.splitlines():
        if "|" not in line:
            continue
        parts = [p.strip() for p in line.split("|")]
        parts = [p for p in parts if p]
        if not parts:
            continue
        if all(re.fullmatch(r"[:\-\s]+", p) for p in parts):
            continue
        rows.append(parts[: max(1, int(max_cols))])
        if len(rows) >= max(1, int(max_rows)):
            break
    if not rows:
        return []
    width = max(len(r) for r in rows)
    return [r + [""] * (width - len(r)) for r in rows]


_TAPAS_BUNDLE_CACHE: dict[str, dict] = {}


def resolve_local_hf_snapshot_path(model_name_or_path: str) -> str:
    name = str(model_name_or_path).strip()
    if not name:
        return name

    p = Path(name).expanduser()
    if not p.exists() or not p.is_dir():
        return name

    if (p / "config.json").exists():
        return str(p)

    refs_main = p / "refs" / "main"
    snapshots = p / "snapshots"

    if refs_main.exists() and snapshots.exists():
        try:
            rev = refs_main.read_text(encoding="utf-8").strip()
        except Exception:
            rev = ""
        if rev:
            candidate = snapshots / rev
            if candidate.exists() and (candidate / "config.json").exists():
                return str(candidate)

    if snapshots.exists():
        for candidate in sorted(snapshots.iterdir()):
            if candidate.is_dir() and (candidate / "config.json").exists():
                return str(candidate)

    return str(p)


def maybe_load_tapas_bundle(model_name_or_path: str) -> dict | None:
    name = str(model_name_or_path).strip()
    if not name:
        return None
    if TapasTokenizer is None or TapasModel is None or torch is None or pd is None:
        return None
    resolved = resolve_local_hf_snapshot_path(name)
    cache_key = resolved
    if cache_key in _TAPAS_BUNDLE_CACHE:
        return _TAPAS_BUNDLE_CACHE[cache_key]

    local_only = Path(resolved).exists()
    try:
        tok = TapasTokenizer.from_pretrained(resolved, local_files_only=local_only)
        mdl = TapasModel.from_pretrained(resolved, local_files_only=local_only)
        mdl.eval()
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        mdl.to(device)
    except Exception:
        return None
    bundle = {
        "tokenizer": tok,
        "model": mdl,
        "device": device,
        "model_path": resolved,
        "local_files_only": local_only,
    }
    _TAPAS_BUNDLE_CACHE[cache_key] = bundle
    _TAPAS_BUNDLE_CACHE[name] = bundle
    return bundle


def tapas_query_token_matrix(query: str, bundle: dict, max_tokens: int = 24) -> np.ndarray:
    if not query.strip() or pd is None or torch is None:
        return np.zeros((0, 0), dtype=np.float32)
    tok = bundle["tokenizer"]
    mdl = bundle["model"]
    device = bundle["device"]
    dummy = pd.DataFrame({"c0": ["placeholder"]})
    enc = tok(table=dummy, queries=[query], truncation=True, return_tensors="pt")
    enc = {k: v.to(device) for k, v in enc.items()}
    with torch.no_grad():
        out = mdl(**enc)

    hid = out.last_hidden_state[0].detach().cpu().numpy().astype(np.float32)
    att = enc["attention_mask"][0].detach().cpu().numpy()
    ids = enc["input_ids"][0].detach().cpu().numpy()
    special_ids = set(tok.all_special_ids)

    if "token_type_ids" in enc:
        token_type_ids = enc["token_type_ids"][0].detach().cpu().numpy()
        seg = token_type_ids[:, 0]
    else:
        seg = np.zeros_like(att)

    keep = [
        i
        for i in range(len(att))
        if int(att[i]) > 0 and int(ids[i]) not in special_ids and int(seg[i]) == 0
    ]
    if not keep:
        return np.zeros((0, hid.shape[1]), dtype=np.float32)
    sel = hid[keep][: max(1, int(max_tokens))]
    return normalize_rows(sel.astype(np.float32))


def tapas_table_repr(
    table_text: str,
    bundle: dict,
    max_rows: int,
    max_cols: int,
    max_cells: int,
) -> dict[str, np.ndarray] | None:
    if pd is None or torch is None:
        return None
    rows = parse_markdown_table_rows(table_text, max_rows=max_rows, max_cols=max_cols)
    if not rows:
        return None
    df = pd.DataFrame(rows, columns=[f"c{i}" for i in range(len(rows[0]))])
    tok = bundle["tokenizer"]
    mdl = bundle["model"]
    device = bundle["device"]
    enc = tok(table=df, queries=[""], truncation=True, return_tensors="pt")
    enc = {k: v.to(device) for k, v in enc.items()}
    with torch.no_grad():
        out = mdl(**enc)

    hid = out.last_hidden_state[0].detach().cpu().numpy().astype(np.float32)
    att = enc["attention_mask"][0].detach().cpu().numpy()
    if "token_type_ids" not in enc:
        return None
    token_type_ids = enc["token_type_ids"][0].detach().cpu().numpy()
    col_ids = token_type_ids[:, 1]
    row_ids = token_type_ids[:, 2]

    cell_map: dict[tuple[int, int], list[np.ndarray]] = defaultdict(list)
    for i in range(len(att)):
        if int(att[i]) <= 0:
            continue
        r = int(row_ids[i]) - 1
        c = int(col_ids[i]) - 1
        if r < 0 or c < 0:
            continue
        cell_map[(r, c)].append(hid[i])

    if not cell_map:
        return None

    sorted_cells = sorted(cell_map.keys())[: max(1, int(max_cells))]
    cell_vecs: list[np.ndarray] = []
    row_map: dict[int, list[np.ndarray]] = defaultdict(list)
    col_map: dict[int, list[np.ndarray]] = defaultdict(list)
    for r, c in sorted_cells:
        vec = np.mean(np.asarray(cell_map[(r, c)], dtype=np.float32), axis=0)
        cell_vecs.append(vec)
        row_map[r].append(vec)
        col_map[c].append(vec)

    cell_mat = normalize_rows(np.asarray(cell_vecs, dtype=np.float32)) if cell_vecs else np.zeros((0, 0), dtype=np.float32)
    row_mat = normalize_rows(
        np.asarray([np.mean(np.asarray(v, dtype=np.float32), axis=0) for _, v in sorted(row_map.items())], dtype=np.float32)
    ) if row_map else np.zeros((0, cell_mat.shape[1] if cell_mat.size else 0), dtype=np.float32)
    col_mat = normalize_rows(
        np.asarray([np.mean(np.asarray(v, dtype=np.float32), axis=0) for _, v in sorted(col_map.items())], dtype=np.float32)
    ) if col_map else np.zeros((0, cell_mat.shape[1] if cell_mat.size else 0), dtype=np.float32)
    return {"cell": cell_mat, "row": row_mat, "col": col_mat}


def learnable_table_aggregation_score(
    cell_score: float,
    row_score: float,
    col_score: float,
    cell_logit: float,
    row_logit: float,
    col_logit: float,
    temperature: float,
) -> float:
    logits = np.asarray(
        [float(cell_logit) * float(cell_score), float(row_logit) * float(row_score), float(col_logit) * float(col_score)],
        dtype=np.float32,
    )
    weights = softmax_1d(logits, temp=float(temperature))
    return float(weights[0] * cell_score + weights[1] * row_score + weights[2] * col_score)


def table_cell_encoder_score_tapas(
    query_mat: np.ndarray,
    table_repr: dict[str, np.ndarray] | None,
    cell_logit: float,
    row_logit: float,
    col_logit: float,
    temperature: float,
) -> float:
    if query_mat.size == 0 or not table_repr:
        return 0.0
    cell_mat = table_repr.get("cell", np.zeros((0, 0), dtype=np.float32))
    row_mat = table_repr.get("row", np.zeros((0, 0), dtype=np.float32))
    col_mat = table_repr.get("col", np.zeros((0, 0), dtype=np.float32))
    cell_score = maxsim_score(query_mat, cell_mat) if cell_mat.size > 0 else 0.0
    row_score = maxsim_score(query_mat, row_mat) if row_mat.size > 0 else 0.0
    col_score = maxsim_score(query_mat, col_mat) if col_mat.size > 0 else 0.0
    return learnable_table_aggregation_score(
        cell_score,
        row_score,
        col_score,
        cell_logit=cell_logit,
        row_logit=row_logit,
        col_logit=col_logit,
        temperature=temperature,
    )


def extract_kg_graph_edges(text: str, max_edges: int = 256) -> list[tuple[str, str, str]]:
    edges: list[tuple[str, str, str]] = []
    chunks = [x.strip() for x in re.split(r"[\n;]+", text) if x.strip()]
    rel_re = re.compile(
        r"\s*([^,;|]+?)\s+(parent|subsidiary|competitor|founded by|founded|owned|acquired|headquarter|located|spouse|capital)\s+([^,;|]+)",
        flags=re.IGNORECASE,
    )
    for ch in chunks:
        if len(edges) >= max(1, int(max_edges)):
            break
        if "|" in ch:
            parts = [p.strip() for p in ch.split("|") if p.strip()]
            if len(parts) >= 3:
                edges.append((parts[0], parts[1].lower(), parts[2]))
                continue
        if "->" in ch:
            parts = [p.strip() for p in ch.split("->") if p.strip()]
            if len(parts) >= 2:
                for a, b in zip(parts[:-1], parts[1:]):
                    edges.append((a, "path", b))
                    if len(edges) >= max(1, int(max_edges)):
                        break
                continue
        m = rel_re.match(ch)
        if m is not None:
            edges.append((m.group(1).strip(), m.group(2).lower(), m.group(3).strip()))

    if edges:
        return edges[: max(1, int(max_edges))]

    seq = tokenize_list(text)
    for i in range(0, max(0, len(seq) - 1)):
        edges.append((seq[i], "cooccur", seq[i + 1]))
        if len(edges) >= max(1, int(max_edges)):
            break
    return edges[: max(1, int(max_edges))]


def build_kg_gnn_doc_repr(
    text: str,
    max_hops: int,
    max_paths: int,
    gnn_layers: int = HEAVY_GNN_LAYERS,
) -> dict[str, np.ndarray] | None:
    edges = extract_kg_graph_edges(text, max_edges=256)
    if not edges:
        return None

    node_set = set()
    for h, _, t in edges:
        node_set.add(h)
        node_set.add(t)
    if not node_set:
        return None

    nodes = sorted(node_set)
    idx_of = {n: i for i, n in enumerate(nodes)}
    n_nodes = len(nodes)
    dim = HEAVY_GNN_VEC_DIM
    h0 = np.zeros((n_nodes, dim), dtype=np.float32)
    for i, n in enumerate(nodes):
        v = embed_token_set(tokenize(n), max_tokens=10)
        if v is None:
            v = token_hash_vector(n, dim=dim)
        h0[i] = v
    h = normalize_rows(h0)

    neigh: list[list[tuple[int, np.ndarray]]] = [[] for _ in range(n_nodes)]
    for hs, rel, ts in edges:
        i = idx_of[hs]
        j = idx_of[ts]
        rv = token_hash_vector(rel, dim=dim)
        neigh[i].append((j, rv))
        neigh[j].append((i, rv))

    for _ in range(max(1, int(gnn_layers))):
        h_new = np.zeros_like(h)
        for i in range(n_nodes):
            msg = []
            for j, rv in neigh[i][:16]:
                msg.append(0.85 * h[j] + 0.15 * rv)
            nei = np.mean(np.asarray(msg, dtype=np.float32), axis=0) if msg else np.zeros((dim,), dtype=np.float32)
            h_new[i] = 0.60 * h[i] + 0.40 * nei
        h = normalize_rows(h_new)

    starts = sorted(range(n_nodes), key=lambda i: len(neigh[i]), reverse=True)[:32]
    paths: list[list[int]] = []

    def dfs(cur: int, stack: list[int], depth: int) -> None:
        if len(paths) >= max(1, int(max_paths)):
            return
        if depth >= 2:
            paths.append(stack.copy())
        if depth >= max(2, int(max_hops)):
            return
        for nxt, _ in neigh[cur][:6]:
            if len(stack) >= 2 and nxt == stack[-2]:
                continue
            stack.append(nxt)
            dfs(nxt, stack, depth + 1)
            stack.pop()
            if len(paths) >= max(1, int(max_paths)):
                return

    for s in starts:
        dfs(s, [s], 0)
        if len(paths) >= max(1, int(max_paths)):
            break

    if not paths:
        return None

    path_vecs = []
    for p in paths[: max(1, int(max_paths))]:
        vec = np.mean(h[p], axis=0)
        path_vecs.append(vec)
    path_mat = normalize_rows(np.asarray(path_vecs, dtype=np.float32))

    if path_mat.shape[0] >= 2:
        neg_mat = path_mat[np.roll(np.arange(path_mat.shape[0]), 1)]
    else:
        neg_mat = normalize_rows(
            np.asarray(
                [
                    token_hash_vector("neg_path_a", dim=dim),
                    token_hash_vector("neg_path_b", dim=dim),
                    token_hash_vector("neg_path_c", dim=dim),
                ],
                dtype=np.float32,
            )
        )
    return {"path": path_mat, "neg": neg_mat}


def contrastive_path_alignment_score(
    query_mat: np.ndarray,
    path_mat: np.ndarray,
    neg_mat: np.ndarray,
    temperature: float,
) -> float:
    if query_mat.size == 0 or path_mat.size == 0:
        return 0.0
    q = np.mean(query_mat, axis=0)
    qn = float(np.linalg.norm(q))
    if qn <= 1e-9:
        return 0.0
    q = q / qn

    pos = path_mat @ q
    pos = np.sort(pos)[-min(4, len(pos)) :]
    if pos.size == 0:
        return 0.0

    if neg_mat.size == 0:
        neg = np.asarray([], dtype=np.float32)
    else:
        neg = neg_mat @ q
        neg = np.sort(neg)[-min(16, len(neg)) :]

    t = max(1e-6, float(temperature))
    pos_logits = pos / t
    neg_sum = float(np.sum(np.exp(neg / t))) if neg.size > 0 else 0.0
    nce = pos_logits - np.log(np.exp(pos_logits) + neg_sum + 1e-9)
    return float(1.0 / (1.0 + np.exp(-float(np.mean(nce)))))


def kg_path_encoder_score_gnn(
    query_mat: np.ndarray,
    kg_doc_repr: dict[str, np.ndarray] | None,
    contrastive_temp: float,
    extra_neg_mat: np.ndarray | None = None,
) -> float:
    if query_mat.size == 0 or not kg_doc_repr:
        return 0.0
    path_mat = kg_doc_repr.get("path", np.zeros((0, 0), dtype=np.float32))
    neg_mat = kg_doc_repr.get("neg", np.zeros((0, 0), dtype=np.float32))
    if extra_neg_mat is not None and getattr(extra_neg_mat, "size", 0) > 0:
        if neg_mat.size == 0:
            neg_mat = normalize_rows(np.asarray(extra_neg_mat, dtype=np.float32))
        else:
            neg_mat = normalize_rows(
                np.concatenate([neg_mat, np.asarray(extra_neg_mat, dtype=np.float32)], axis=0)
            )
    if path_mat.size == 0:
        return 0.0
    maxsim = maxsim_score(query_mat, path_mat)
    contrastive = contrastive_path_alignment_score(query_mat, path_mat, neg_mat, temperature=contrastive_temp)
    return 0.70 * maxsim + 0.30 * contrastive


def build_query_aware_kg_hard_negatives(
    query_mat: np.ndarray,
    kg_top_positions: list[int],
    uni_candidate: np.ndarray,
    kg_gnn_cache: dict[int, dict[str, np.ndarray] | None],
    doc_ids: list[str],
    doc_texts: list[str],
    max_hops: int,
    max_paths: int,
    top_docs: int,
    max_neg_paths: int,
) -> dict[int, np.ndarray]:
    if query_mat.size == 0 or len(kg_top_positions) < 2:
        return {}

    q = np.mean(query_mat, axis=0)
    qn = float(np.linalg.norm(q))
    if qn <= 1e-9:
        return {}
    q = (q / qn).astype(np.float32)

    doc_centroids: dict[int, np.ndarray] = {}
    doc_qsim: dict[int, float] = {}

    for pos in kg_top_positions:
        j = int(uni_candidate[pos])
        if j not in kg_gnn_cache:
            kg_gnn_cache[j] = build_kg_gnn_doc_repr(
                doc_texts[j],
                max_hops=max_hops,
                max_paths=max_paths,
            )
        rep = kg_gnn_cache.get(j)
        if not rep:
            continue
        path_mat = rep.get("path", np.zeros((0, 0), dtype=np.float32))
        if path_mat.size == 0:
            continue
        cent = np.mean(path_mat, axis=0)
        cn = float(np.linalg.norm(cent))
        if cn <= 1e-9:
            continue
        cent = (cent / cn).astype(np.float32)
        doc_centroids[j] = cent
        doc_qsim[j] = float(cent @ q)

    if len(doc_centroids) < 2:
        return {}

    hard_neg_map: dict[int, np.ndarray] = {}
    top_docs_eff = max(1, int(top_docs))
    max_neg_paths_eff = max(1, int(max_neg_paths))

    for anchor in doc_centroids.keys():
        anchor_cent = doc_centroids[anchor]
        cand = []
        for other, other_cent in doc_centroids.items():
            if other == anchor:
                continue
            sim_anchor = float(anchor_cent @ other_cent)
            sim_query = float(doc_qsim.get(other, 0.0))
            hard_score = 0.65 * sim_anchor + 0.35 * sim_query
            cand.append((hard_score, other))
        if not cand:
            continue
        cand.sort(key=lambda x: x[0], reverse=True)

        neg_vecs: list[np.ndarray] = []
        for _, other in cand[:top_docs_eff]:
            rep = kg_gnn_cache.get(other)
            if not rep:
                continue
            path_mat = rep.get("path", np.zeros((0, 0), dtype=np.float32))
            if path_mat.size == 0:
                continue
            for v in path_mat[: max(1, max_neg_paths_eff // top_docs_eff)]:
                neg_vecs.append(v)
                if len(neg_vecs) >= max_neg_paths_eff:
                    break
            if len(neg_vecs) >= max_neg_paths_eff:
                break
        if neg_vecs:
            hard_neg_map[anchor] = normalize_rows(np.asarray(neg_vecs, dtype=np.float32))

    return hard_neg_map


def build_modality_seed_token_mats(
    candidate: np.ndarray,
    mixed_scores: np.ndarray,
    doc_ids: list[str],
    doc_token_lists: list[list[str]],
    per_bucket_docs: int = 8,
) -> dict[str, np.ndarray]:
    out: dict[str, list[str]] = {"text": [], "table": [], "kg": []}
    used: dict[str, int] = {"text": 0, "table": 0, "kg": 0}
    order = np.argsort(-mixed_scores)
    for pos in order.tolist():
        j = int(candidate[pos])
        b = source_bucket(doc_ids[j])
        if used[b] >= max(1, int(per_bucket_docs)):
            continue
        out[b].extend(doc_token_lists[j][:48])
        used[b] += 1
        if all(used[k] >= max(1, int(per_bucket_docs)) for k in used.keys()):
            break

    mats: dict[str, np.ndarray] = {}
    for b in ("text", "table", "kg"):
        mats[b] = build_token_matrix(out[b], max_tokens=96)
    return mats


def token_cross_modal_interaction_score(
    cand_mat: np.ndarray,
    cand_bucket: str,
    seed_mats: dict[str, np.ndarray],
) -> float:
    if cand_mat.size == 0:
        return 0.0
    vals = []
    for b in ("text", "table", "kg"):
        if b == cand_bucket:
            continue
        m = seed_mats.get(b, np.zeros((0, 0), dtype=np.float32))
        if m.size == 0:
            continue
        vals.append(maxsim_score(cand_mat, m))
    if not vals:
        return 0.0
    return float(np.mean(vals))


def extract_consistency_signal_tokens(text: str) -> set[str]:
    lower = text.lower()
    out = set()
    out |= {kw for kw in CONSISTENCY_KEYWORDS if kw in lower}
    out |= {kw for kw in RELATION_HINT_TERMS if kw in lower}
    out |= extract_numeric_literals(text)
    return out


def cross_modal_conflict_penalty(
    cand_idx: int,
    selected_idxs: list[int],
    doc_ids: list[str],
    query_tokens: set[str],
    doc_tokens: list[set[str]],
    doc_signal_tokens: list[set[str]],
    doc_numeric_literals: list[set[str]],
    table_kg_only: bool = False,
    max_literals_per_doc: int = 0,
) -> float:
    if not selected_idxs:
        return 0.0
    cand_nums = doc_numeric_literals[cand_idx]
    if not cand_nums:
        return 0.0
    if int(max_literals_per_doc) > 0 and len(cand_nums) > int(max_literals_per_doc):
        return 0.0
    cand_qov = len(query_tokens & doc_tokens[cand_idx]) / max(1, len(query_tokens)) if query_tokens else 0.0
    if cand_qov <= 0.0:
        return 0.0
    cand_kw = {x for x in doc_signal_tokens[cand_idx] if (x in CONSISTENCY_KEYWORDS or x in RELATION_HINT_ATOMS)}

    cand_bucket = source_bucket(doc_ids[cand_idx])
    best = 0.0
    for j in selected_idxs:
        other_bucket = source_bucket(doc_ids[j])
        if other_bucket == cand_bucket:
            continue
        if bool(table_kg_only) and {cand_bucket, other_bucket} != {"table", "kg"}:
            continue
        other_nums = doc_numeric_literals[j]
        if not other_nums:
            continue
        if int(max_literals_per_doc) > 0 and len(other_nums) > int(max_literals_per_doc):
            continue
        other_qov = len(query_tokens & doc_tokens[j]) / max(1, len(query_tokens)) if query_tokens else 0.0
        if other_qov <= 0.0:
            continue
        other_kw = {x for x in doc_signal_tokens[j] if (x in CONSISTENCY_KEYWORDS or x in RELATION_HINT_ATOMS)}
        kw_overlap = cand_kw & other_kw
        if not kw_overlap and (query_tokens & doc_tokens[cand_idx] & doc_tokens[j]) == set():
            continue
        if cand_nums & other_nums:
            continue
        kw_ratio = len(kw_overlap) / max(1, len(cand_kw | other_kw)) if (cand_kw or other_kw) else 0.5
        qov_term = min(1.0, cand_qov + other_qov)
        penalty = min(1.0, 0.30 + 0.30 * qov_term + 0.40 * kw_ratio)
        if penalty > best:
            best = penalty
    return best

def estimate_query_conflict_risk(
    query: str,
    candidate_idxs: list[int],
    query_tokens: set[str],
    doc_ids: list[str],
    doc_texts: list[str],
    doc_tokens: list[set[str]],
    doc_signal_tokens: list[set[str]],
    doc_numeric_literals: list[set[str]],
    conflict_bundle: ConflictBundle | None = None,
    table_kg_only: bool = False,
    probe_k: int = 12,
    max_literals_per_doc: int = 0,
) -> float:
    if not candidate_idxs:
        return 0.0
    probe = [int(j) for j in candidate_idxs[: max(2, int(probe_k))]]

    if conflict_bundle is not None:
        probe_contexts = [str(doc_texts[j]) for j in probe if 0 <= j < len(doc_texts)]
        probe_doc_ids = [str(doc_ids[j]) for j in probe if 0 <= j < len(doc_ids)]
        if probe_contexts:
            conflict_probability, _ = conflict_bundle.predict_conflict_probability(
                query=query,
                contexts=probe_contexts,
                doc_ids=probe_doc_ids if len(probe_doc_ids) == len(probe_contexts) else None,
                table_kg_only=bool(table_kg_only),
                probe_k=int(probe_k),
                max_literals_per_doc=int(max_literals_per_doc),
            )
            return float(conflict_probability)

    valid_pairs = 0
    conflict_pairs = 0

    for i in range(len(probe)):
        a = probe[i]
        nums_a = doc_numeric_literals[a]
        if not nums_a:
            continue
        if int(max_literals_per_doc) > 0 and len(nums_a) > int(max_literals_per_doc):
            continue
        qov_a = len(query_tokens & doc_tokens[a]) / max(1, len(query_tokens)) if query_tokens else 0.0
        if qov_a <= 0.0:
            continue
        bucket_a = source_bucket(doc_ids[a])
        kw_a = {x for x in doc_signal_tokens[a] if (x in CONSISTENCY_KEYWORDS or x in RELATION_HINT_ATOMS)}

        for b in probe[i + 1 :]:
            nums_b = doc_numeric_literals[b]
            if not nums_b:
                continue
            if int(max_literals_per_doc) > 0 and len(nums_b) > int(max_literals_per_doc):
                continue
            qov_b = len(query_tokens & doc_tokens[b]) / max(1, len(query_tokens)) if query_tokens else 0.0
            if qov_b <= 0.0:
                continue
            bucket_b = source_bucket(doc_ids[b])
            if bucket_a == bucket_b:
                continue
            if bool(table_kg_only) and {bucket_a, bucket_b} != {"table", "kg"}:
                continue

            kw_b = {x for x in doc_signal_tokens[b] if (x in CONSISTENCY_KEYWORDS or x in RELATION_HINT_ATOMS)}
            kw_overlap = kw_a & kw_b
            if not kw_overlap and (query_tokens & doc_tokens[a] & doc_tokens[b]) == set():
                continue

            valid_pairs += 1
            if (nums_a & nums_b) == set():
                conflict_pairs += 1

    if valid_pairs <= 0:
        return 0.0
    return float(conflict_pairs / valid_pairs)


def cross_modal_consistency_score(
    cand_idx: int,
    selected_idxs: list[int],
    doc_ids: list[str],
    doc_signal_tokens: list[set[str]],
) -> float:
    if not selected_idxs:
        return 0.0
    s_cand = doc_signal_tokens[cand_idx]
    if not s_cand:
        return 0.0
    b_cand = source_bucket(doc_ids[cand_idx])
    best = 0.0
    for j in selected_idxs:
        b_other = source_bucket(doc_ids[j])
        if b_other == b_cand:
            continue
        if {b_cand, b_other} != {"table", "kg"}:
            continue
        s_other = doc_signal_tokens[j]
        if not s_other:
            continue
        sim = jaccard_overlap(s_cand, s_other)
        if sim > best:
            best = sim
    return best


def scheme2_uncertainty_factor(router_entropy: float, uncertainty_threshold: float) -> float:
    # Map entropy to [0, 1] with a smooth high-uncertainty emphasis.
    lo = max(0.0, float(uncertainty_threshold) - 0.25)
    hi = max(lo + 1e-6, min(1.0, float(uncertainty_threshold) + 0.20))
    x = (float(router_entropy) - lo) / (hi - lo)
    return float(np.clip(x, 0.0, 1.0))


def build_cross_modal_seed_sets(
    candidate: np.ndarray,
    mixed_scores: np.ndarray,
    doc_ids: list[str],
    doc_tokens: list[set[str]],
    per_bucket: int = 12,
) -> dict[str, list[set[str]]]:
    seeds: dict[str, list[set[str]]] = {"text": [], "table": [], "kg": []}
    order = np.argsort(-mixed_scores)
    for pos in order.tolist():
        j = int(candidate[pos])
        b = source_bucket(doc_ids[j])
        if len(seeds[b]) >= max(1, int(per_bucket)):
            continue
        tok = doc_tokens[j]
        if not tok:
            continue
        seeds[b].append(tok)
        if all(len(seeds[k]) >= max(1, int(per_bucket)) for k in seeds.keys()):
            break
    return seeds


def cross_modal_agreement_bonus(
    cand_idx: int,
    cand_bucket: str,
    doc_tokens: list[set[str]],
    seed_sets: dict[str, list[set[str]]],
) -> float:
    tok = doc_tokens[cand_idx]
    if not tok:
        return 0.0
    sims: list[float] = []
    for other_bucket in ("text", "table", "kg"):
        if other_bucket == cand_bucket:
            continue
        seeds = seed_sets.get(other_bucket, [])
        if not seeds:
            continue
        best = 0.0
        for s in seeds[:6]:
            sim = jaccard_overlap(tok, s)
            if sim > best:
                best = sim
        sims.append(best)
    if not sims:
        return 0.0
    return float(np.mean(sims))


def qrel_modalities(row: dict) -> np.ndarray:
    y = np.zeros(3, dtype=np.int64)
    for chunk_id, label in row.get("relevant_chunks", {}).items():
        try:
            if float(label) <= 0:
                continue
        except Exception:
            continue
        b = source_bucket(chunk_id)
        y[ROUTER_LABEL_TO_IDX[b]] = 1
    if int(y.sum()) == 0:
        y[ROUTER_LABEL_TO_IDX["text"]] = 1
    return y


def normalized_entropy(probs: np.ndarray) -> np.ndarray:
    p = np.clip(probs, 1e-8, 1.0)
    ent = -(p * np.log(p) + (1.0 - p) * np.log(1.0 - p)).sum(axis=1)
    return ent / (probs.shape[1] * np.log(2.0))


def logits_to_multihot(logits: np.ndarray, threshold: float) -> tuple[np.ndarray, np.ndarray]:
    probs = 1.0 / (1.0 + np.exp(-logits))
    pred = (probs >= threshold).astype(np.int64)
    empty_mask = np.sum(pred, axis=1) == 0
    if np.any(empty_mask):
        top_idx = np.argmax(probs[empty_mask], axis=1)
        pred[empty_mask] = 0
        pred[empty_mask, top_idx] = 1
    return pred, probs


def heuristic_router_probs(query: str) -> np.ndarray:
    q = query.lower()
    probs = np.asarray([0.45, 0.35, 0.20], dtype=np.float32)
    table_hits = any(x in q for x in ["table", "row", "column", "how many", "total", "rate", "percentage"])
    kg_hits = any(x in q for x in ["relation", "parent", "subsidiary", "founded by", "who", "which company"]) 
    text_hits = any(x in q for x in ["why", "explain", "describe", "summary", "reason"])

    if table_hits:
        probs += np.asarray([0.05, 0.25, -0.05], dtype=np.float32)
    if kg_hits:
        probs += np.asarray([-0.05, 0.00, 0.20], dtype=np.float32)
    if text_hits:
        probs += np.asarray([0.15, -0.05, -0.05], dtype=np.float32)

    probs = np.clip(probs, 1e-3, None)
    probs = probs / probs.sum()
    return probs


def infer_router_predictions(
    queries: list[str],
    router_model_dir: Path | None,
    threshold: float,
    batch_size: int,
    allow_heuristic_fallback: bool,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, str]:
    n = len(queries)
    if n == 0:
        empty = np.zeros((0, 3), dtype=np.int64)
        return empty, empty.astype(np.float32), np.zeros((0,), dtype=np.float32), "empty"

    model_attempted = (
        router_model_dir is not None
        and router_model_dir.exists()
        and AutoTokenizer is not None
        and AutoModelForSequenceClassification is not None
        and torch is not None
    )

    if model_attempted:
        try:
            tokenizer = AutoTokenizer.from_pretrained(str(router_model_dir))
            model = AutoModelForSequenceClassification.from_pretrained(str(router_model_dir))
            model.eval()
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            model.to(device)
            logits_all = []
            for i in range(0, n, batch_size):
                batch_q = queries[i : i + batch_size]
                enc = tokenizer(
                    batch_q,
                    truncation=True,
                    padding=True,
                    max_length=256,
                    return_tensors="pt",
                )
                enc = {k: v.to(device) for k, v in enc.items()}
                with torch.no_grad():
                    out = model(**enc)
                logits_all.append(out.logits.detach().cpu().numpy())
            logits = np.concatenate(logits_all, axis=0)
            pred, probs = logits_to_multihot(logits, threshold=threshold)
            ent = normalized_entropy(probs)
            return pred, probs.astype(np.float32), ent.astype(np.float32), f"model:{router_model_dir}"
        except Exception as e:
            if not allow_heuristic_fallback:
                raise RuntimeError(
                    f"Router model inference failed and heuristic fallback is disabled: {e}"
                ) from e

    if not allow_heuristic_fallback:
        raise RuntimeError(
            "Router model is unavailable but heuristic fallback is disabled. "
            "Provide --router-model (or a resolvable model from --router-metrics) or set --allow-heuristic-router-fallback."
        )

    probs = np.vstack([heuristic_router_probs(q) for q in queries]).astype(np.float32)
    pred = np.zeros((n, 3), dtype=np.int64)
    top_idx = np.argmax(probs, axis=1)
    pred[np.arange(n), top_idx] = 1
    ent = normalized_entropy(probs)
    return pred, probs, ent.astype(np.float32), "heuristic"


def detect_st_pooling_mode(model_dir: Path) -> str:
    cfg = model_dir / "1_Pooling" / "config.json"
    if not cfg.exists():
        return "mean"
    try:
        obj = json.loads(cfg.read_text(encoding="utf-8"))
    except Exception:
        return "mean"
    if bool(obj.get("pooling_mode_cls_token", False)):
        return "cls"
    if bool(obj.get("pooling_mode_mean_tokens", False)):
        return "mean"
    if bool(obj.get("pooling_mode_max_tokens", False)):
        return "max"
    return "mean"


def has_st_normalize(model_dir: Path) -> bool:
    modules_path = model_dir / "modules.json"
    if not modules_path.exists():
        return True
    try:
        modules = json.loads(modules_path.read_text(encoding="utf-8"))
    except Exception:
        return True
    for m in modules:
        t = str(m.get("type", "")).lower()
        if t.endswith(".normalize"):
            return True
    return False


def encode_texts_with_hf_encoder(
    texts: list[str],
    tokenizer,
    model,
    device,
    batch_size: int,
    max_length: int = 512,
    pooling_mode: str = "mean",
    do_normalize: bool = True,
) -> np.ndarray:
    if torch is None:
        raise RuntimeError("torch is required for HF encoder embeddings")
    chunks: list[np.ndarray] = []
    model.eval()
    step = max(1, int(batch_size))
    for i in range(0, len(texts), step):
        batch = texts[i : i + step]
        enc = tokenizer(
            batch,
            truncation=True,
            padding=True,
            max_length=max_length,
            return_tensors="pt",
        )
        enc = {k: v.to(device) for k, v in enc.items()}
        with torch.no_grad():
            out = model(**enc)
            hidden = out.last_hidden_state
            if pooling_mode == "cls":
                pooled = hidden[:, 0, :]
            elif pooling_mode == "max":
                mask = enc["attention_mask"].unsqueeze(-1).to(hidden.dtype)
                neg = torch.full_like(hidden, -1e9)
                masked = torch.where(mask > 0, hidden, neg)
                pooled = masked.max(dim=1).values
            else:
                mask = enc["attention_mask"].unsqueeze(-1).to(hidden.dtype)
                pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-9)
            if do_normalize:
                pooled = torch.nn.functional.normalize(pooled, p=2, dim=1)
        chunks.append(pooled.detach().cpu().numpy().astype(np.float32))
    if not chunks:
        return np.zeros((0, 1), dtype=np.float32)
    return np.concatenate(chunks, axis=0)


def routing_subset_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    if y_true.size == 0:
        return 0.0
    return float(np.mean(np.all(y_true == y_pred, axis=1)))


def routing_micro_f1(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    tp = float(np.sum((y_true == 1) & (y_pred == 1)))
    fp = float(np.sum((y_true == 0) & (y_pred == 1)))
    fn = float(np.sum((y_true == 1) & (y_pred == 0)))
    p = tp / max(1e-9, tp + fp)
    r = tp / max(1e-9, tp + fn)
    return 0.0 if (p + r) < 1e-9 else float(2 * p * r / (p + r))


def extractive_reader(query: str, contexts: list[str], numeric_consensus: bool = False) -> str:
    if not contexts:
        return ""
    q_tokens = tokenize(query)

    sentences: list[str] = []
    for c in contexts:
        for s in re.split(r"[\n\.\!\?;]+", c):
            s = s.strip()
            if len(s) >= 5:
                sentences.append(s)

    if not sentences:
        return ""

    def sent_score(s: str) -> float:
        s_tokens = tokenize(s)
        if not s_tokens:
            return 0.0
        inter = len(q_tokens & s_tokens)
        return inter / max(1, len(q_tokens)) + 0.3 * inter / max(1, len(s_tokens))

    scored_sentences = [(s, sent_score(s)) for s in sentences]
    scored_sentences.sort(key=lambda x: x[1], reverse=True)
    top_scored_sentences = scored_sentences[: min(12, len(scored_sentences))]
    top_sentences = [s for s, _ in top_scored_sentences]
    best = top_sentences[0]

    ql = query.strip().lower()
    is_count = bool(re.match(r"^(how many|how much|number of)\b", ql)) or any(
        x in ql for x in [" capacity", " population", " sold", " purchases", " total"]
    )
    is_year = bool(re.match(r"^(when|what year|which year)\b", ql)) or " date" in ql
    is_person = bool(re.match(r"^(who|which person|whose)\b", ql))
    is_location = bool(re.match(r"^(where|which country|which city|which canton|in which)\b", ql))

    if is_count:
        if bool(numeric_consensus):
            vote = pick_literal_consensus(top_scored_sentences, NUMERIC_LITERAL_RE)
            if vote:
                return vote
        for s in top_sentences:
            m = NUMERIC_LITERAL_RE.search(s)
            if m:
                return m.group(0)

    if is_year:
        if bool(numeric_consensus):
            vote = pick_literal_consensus(top_scored_sentences, YEAR_LITERAL_RE)
            if vote:
                return vote
        for s in top_sentences:
            m = YEAR_LITERAL_RE.search(s)
            if m:
                return m.group(0)

    if is_person:
        for s in top_sentences:
            caps = re.findall(r"\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+){0,4}\b", s)
            if caps:
                return caps[0]

    if is_location:
        for s in top_sentences:
            m = re.search(r"\b([A-Z][A-Za-z]+(?:\s+[A-Z][A-Za-z]+){0,4})\b", s)
            if m:
                return m.group(1)

    toks = best.split()
    return " ".join(toks[: min(14, len(toks))])


def answer_support_score(answer: str, contexts: list[str]) -> float:
    a = normalize_answer(answer)
    if not a:
        return -1e6
    a_toks = set(a.split())
    if not a_toks:
        return -1e6

    best_f = 0.0
    has_exact = 0.0
    for c in contexts:
        c_norm = normalize_answer(c)
        if a in c_norm:
            has_exact = 1.0
        for s in re.split(r"[\n\.!\?;]+", c):
            s_norm = normalize_answer(s)
            if not s_norm:
                continue
            s_toks = set(s_norm.split())
            inter = len(a_toks & s_toks)
            if inter == 0:
                continue
            p = inter / max(1, len(a_toks))
            r = inter / max(1, len(s_toks))
            f = 0.0 if (p + r) < 1e-9 else 2 * p * r / (p + r)
            if f > best_f:
                best_f = f

    # Prefer concise noun phrase / value-like answers.
    length_penalty = max(0.0, (len(a_toks) - 8) * 0.03)
    return 1.2 * has_exact + best_f - length_penalty


def calibrate_tessera_answer(
    query: str,
    raw_answer: str,
    contexts: list[str],
    extractive_numeric_consensus: bool = False,
) -> str:
    raw = raw_answer.strip()
    ext = extractive_reader(query, contexts, numeric_consensus=bool(extractive_numeric_consensus)).strip()
    if not raw and not ext:
        return ""
    if raw and not ext:
        return raw
    if ext and not raw:
        return ext
    if normalize_answer(ext) == normalize_answer(raw):
        return raw

    raw_score = answer_support_score(raw, contexts)
    ext_score = answer_support_score(ext, contexts)

    target = str(infer_qa_target_type(query)).strip().lower()
    if target not in {"number", "year"}:
        return raw
    # Conservative replacement guard: avoid calibration-induced collapse.
    # For numeric/year queries, a smaller margin is acceptable.
    margin = 0.16 if target in {"number", "year"} else 0.30
    if ext_score >= raw_score + margin:
        return ext
    return raw


def apply_tessera_answer_type_guard(
    query: str,
    raw_answer: str,
    contexts: list[str],
    extractive_numeric_consensus: bool = False,
) -> str:
    raw = raw_answer.strip()
    if not raw:
        return raw

    target = str(infer_qa_target_type(query)).strip().lower()
    # Keep this guard narrowly scoped to number queries only.
    # Year/date questions can require full date strings rather than bare years.
    if target != "number":
        return raw

    ext = extractive_reader(query, contexts, numeric_consensus=bool(extractive_numeric_consensus)).strip()
    if not ext:
        return raw

    raw_match = NUMERIC_LITERAL_RE.search(raw)
    ext_match = NUMERIC_LITERAL_RE.search(ext)

    if not ext_match:
        return raw
    ext_lit = ext_match.group(0)
    if not raw_match:
        return ext_lit

    raw_lit = raw_match.group(0)
    # Conservative canonicalization: only when raw/ext agree on the literal
    # and raw is verbose beyond that literal (single-literal answers only).
    if normalize_answer(raw_lit) == normalize_answer(ext_lit):
        raw_norm = normalize_answer(raw)
        lit_norm = normalize_answer(raw_lit)
        if raw_norm != lit_norm and len(raw_norm.split()) >= 2 and len(NUMERIC_LITERAL_RE.findall(raw)) == 1:
            return ext_lit
    return raw


def maybe_apply_tessera_table_number_agent(
    query: str,
    raw_answer: str,
    contexts: list[str],
    ctx_doc_ids: list[str],
    args: argparse.Namespace,
) -> tuple[str, bool, dict[str, float]]:
    if select_tessera_table_number_answer is None or TESSERATableNumberAgentConfig is None:
        raise RuntimeError("tessera_policy module is required for --tessera-table-number-agent")

    def strict_table_number_target(q: str) -> str:
        ql = str(q or "").strip().lower()
        if re.match(
            r"^(who|which person|whose|where|which country|which city|which county|"
            r"which state|which province|in which|what currency|what group|"
            r"what term|what is the term|what is the name|what was the name|"
            r"which company|which team|which album|which film|which movie)\b",
            ql,
        ):
            return ""
        if re.match(r"^(what months|which months)\b", ql) or (
            re.match(r"^when\b", ql) and re.search(r"\b(month|months|season|seasons)\b", ql)
        ):
            return ""
        if re.match(r"^(when|what year|which year|in what year)\b", ql):
            return "year"
        if re.match(r"^(how many|how much|number of)\b", ql):
            return "number"
        if re.match(
            r"^(what|which)\s+(?:is|was|were|are|did)?\s*(?:the\s+)?"
            r"(?:total|number|population|capacity|percentage|percent|amount|"
            r"price|cost|score|rank|age|revenue|income|sales|profit)\b",
            ql,
        ):
            return "number"
        if re.match(r"^(what|which)\s+(?:is|was|were|are)?\s*(?:the\s+)?(?:year|date)\b", ql):
            return "year"
        return ""

    target = strict_table_number_target(query)
    debug = {
        "attempted": 0.0,
        "candidates": 0.0,
        "best_score": 0.0,
        "raw_no_evidence": 0.0,
        "raw_has_literal": 0.0,
        "base_support": float(answer_support_score(raw_answer, contexts)),
        "support": 0.0,
        "accept": 0.0,
    }
    if target not in {"number", "year"}:
        return raw_answer, False, debug

    force_trigger = (
        float(args.tessera_table_number_agent_low_support_threshold) >= 0.0
        and debug["base_support"] < float(args.tessera_table_number_agent_low_support_threshold)
    )
    cfg = TESSERATableNumberAgentConfig(
        min_score=float(args.tessera_table_number_agent_min_score),
    )
    candidate_answer, trace = select_tessera_table_number_answer(
        query=query,
        current_answer=raw_answer,
        contexts=contexts,
        doc_ids=ctx_doc_ids,
        target_type=target,
        source_bucket_fn=source_bucket,
        force_trigger=bool(force_trigger),
        config=cfg,
    )
    debug.update(
        {
            "attempted": float(trace.attempted),
            "candidates": float(trace.candidate_count),
            "best_score": float(trace.best_score),
            "raw_no_evidence": float(trace.raw_no_evidence),
            "raw_has_literal": float(trace.raw_has_literal),
        }
    )
    if not trace.accepted:
        return raw_answer, False, debug

    candidate_support = float(answer_support_score(candidate_answer, contexts))
    accept = candidate_support >= float(args.tessera_table_number_agent_min_support)
    if not trace.raw_no_evidence and trace.raw_has_literal:
        accept = accept and candidate_support >= debug["base_support"] + float(
            args.tessera_table_number_agent_margin
        )
    if force_trigger:
        accept = accept and (
            candidate_support >= debug["base_support"] + float(args.tessera_table_number_agent_margin)
            or debug["base_support"] < float(args.tessera_table_number_agent_min_support)
        )
    debug["support"] = candidate_support
    debug["accept"] = float(accept)
    if accept:
        return candidate_answer, True, debug
    return raw_answer, False, debug


def refine_tessera_answer_with_consensus(
    query: str,
    raw_answer: str,
    contexts: list[str],
    extractive_numeric_consensus: bool = False,
    min_gain: float = 0.20,
    targeted_only: bool = True,
) -> tuple[str, bool, bool]:
    target = str(infer_qa_target_type(query)).strip().lower()
    if targeted_only and target not in {"number", "year", "location"}:
        return raw_answer, False, False

    raw = raw_answer.strip()
    ext = extractive_reader(query, contexts, numeric_consensus=bool(extractive_numeric_consensus)).strip()
    if not ext:
        return raw_answer, False, False
    if normalize_answer(raw) == normalize_answer(ext):
        return raw_answer, False, False

    if target == "number" and NUMERIC_LITERAL_RE.search(ext) is None:
        return raw_answer, False, False
    if target == "year" and YEAR_LITERAL_RE.search(ext) is None:
        return raw_answer, False, False

    attempted = True
    if not raw:
        return ext, attempted, True

    raw_score = answer_support_score(raw, contexts)
    ext_score = answer_support_score(ext, contexts)
    gain = float(ext_score - raw_score)
    if gain >= float(max(0.0, min_gain)):
        return ext, attempted, True
    return raw_answer, attempted, False


def build_evidence_chain_retry_context(
    query: str,
    current_answer: str,
    current_ctx_idxs: list[int],
    dense_ranked_idxs: list[int],
    doc_ids: list[str],
    doc_tokens: list[set[str]],
    k: int,
    pool_k: int,
    answer_boost: float,
    target_type: str,
) -> list[int]:
    topk = max(1, int(k))
    pool = max(topk, int(pool_k))

    q_tokens = tokenize(query)
    a_tokens = tokenize(current_answer)
    cur_set = {int(x) for x in current_ctx_idxs}

    candidates: list[int] = []
    seen: set[int] = set()
    for idx in list(current_ctx_idxs) + list(dense_ranked_idxs[:pool]):
        j = int(idx)
        if j in seen:
            continue
        if j < 0 or j >= len(doc_tokens):
            continue
        seen.add(j)
        candidates.append(j)

    if not candidates:
        if current_ctx_idxs:
            return [int(x) for x in current_ctx_idxs[:topk]]
        return [int(x) for x in dense_ranked_idxs[:topk]]

    answer_w = float(max(0.0, answer_boost))
    scored: list[tuple[int, float, float, float]] = []
    for j in candidates:
        d_toks = doc_tokens[j]
        q_overlap = len(q_tokens & d_toks) / max(1, len(q_tokens)) if q_tokens else 0.0
        a_overlap = len(a_tokens & d_toks) / max(1, len(a_tokens)) if a_tokens else 0.0

        modality_bonus = 0.0
        bucket = source_bucket(doc_ids[j])
        if target_type in {"number", "year"} and bucket == "table":
            modality_bonus += 0.08
        if target_type in {"person", "entity", "location"} and bucket == "kg":
            modality_bonus += 0.05

        anchor_bonus = 0.03 if j in cur_set else 0.0
        score = q_overlap + answer_w * a_overlap + modality_bonus + anchor_bonus
        scored.append((j, float(score), float(q_overlap), float(a_overlap)))

    scored.sort(key=lambda x: (x[1], x[2], x[3], -x[0]), reverse=True)
    selected = [j for j, _, _, _ in scored[:topk]]

    # Force at least one new dense candidate when all selections are old context.
    if selected and set(selected).issubset(cur_set):
        for j, s, _, _ in scored:
            if j not in cur_set and s > 1e-9:
                selected[-1] = j
                break

    out: list[int] = []
    used: set[int] = set()
    for j in selected:
        if j in used:
            continue
        out.append(j)
        used.add(j)

    if len(out) < topk:
        for j, _, _, _ in scored:
            if j in used:
                continue
            out.append(j)
            used.add(j)
            if len(out) >= topk:
                break
    return out[:topk]


def ollama_reader(host: str, model: str, query: str, contexts: list[str], timeout_s: int = 300) -> str:
    context_block = "\n\n".join([f"[doc{i+1}] {c[:1200]}" for i, c in enumerate(contexts[:6])])
    prompt = (
        "/no_think\n\n"
        "Answer the question using only the provided evidence. "
        "Return a short answer phrase only, no explanation.\n\n"
        f"Question: {query}\n\n"
        f"Evidence:\n{context_block}\n\n"
        "Short Answer:\n\n"
        "/no_think"
    )
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "options": {"temperature": 0, "num_predict": 32, "num_ctx": 768},
    }
    max_retries = 3
    last_err = None
    for attempt in range(max_retries):
        try:
            r = requests.post(f"{host.rstrip('/')}/api/generate", json=payload, timeout=timeout_s)
            r.raise_for_status()
            out = r.json().get("response", "").strip()
            # Strip <think>...</think> reasoning tags from qwen3.6-27b output.
            # Handle both closed tags (<think>...</think>) and truncated opens
            # where num_predict cuts off before </think>.
            if "<think>" in out:
                # Case 1: closed <think>...</think>
                out = re.sub(r"<think>.*?</think>", "", out, flags=re.DOTALL).strip()
                # Case 2: truncated <think>... (no closing tag)
                if "<think>" in out:
                    out = out.split("<think>")[0].strip()
            out = out.replace("\n", " ").strip()
            if len(out) > 200:
                out = out[:200]
            return out
        except (requests.exceptions.ReadTimeout, requests.exceptions.ConnectionError) as e:
            last_err = e
            print(f"[ollama_reader] timeout on attempt {attempt + 1}/{max_retries}, retrying in 2s...")
            time.sleep(2)
        except requests.exceptions.HTTPError as e:
            last_err = e
            print(f"[ollama_reader] HTTP error on attempt {attempt + 1}/{max_retries}: {e}")
            time.sleep(2)
    raise RuntimeError(f"[ollama_reader] failed after {max_retries} attempts: {last_err}")


def openai_reader(
    model: str,
    query: str,
    contexts: list[str],
    timeout_s: int = 120,
    temperature: float = 0.0,
    max_tokens: int = 64,
    base_url: str = "",
    api_key_env: str = "OPENAI_API_KEY",
    max_retries: int = 3,
    retry_backoff_s: float = 2.0,
    fail_soft: bool = False,
) -> str:
    api_key = os.environ.get(api_key_env, "").strip()
    if not api_key:
        raise RuntimeError(f"[openai_reader] missing API key env var: {api_key_env}")

    try:
        from openai import OpenAI
    except Exception as e:  # pragma: no cover - runtime dependency check
        raise RuntimeError("OpenAI Python SDK is required for --reader openai. Install with: pip install openai") from e

    context_block = "\n\n".join([f"[doc{i+1}] {c[:1200]}" for i, c in enumerate(contexts[:6])])
    messages = [
        {
            "role": "system",
            "content": (
                "Answer questions using only the provided evidence. "
                "Return a short answer phrase only, no explanation."
            ),
        },
        {
            "role": "user",
            "content": f"Question: {query}\n\nEvidence:\n{context_block}\n\nShort Answer:",
        },
    ]
    client_kwargs = {"api_key": api_key, "timeout": timeout_s}
    if str(base_url).strip():
        client_kwargs["base_url"] = str(base_url).strip()
    client = OpenAI(**client_kwargs)

    last_err = None
    retries = max(1, int(max_retries))
    for attempt in range(retries):
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=float(temperature),
                max_tokens=int(max_tokens),
            )
            out = ""
            if resp.choices:
                out = (resp.choices[0].message.content or "").strip()
            out = re.sub(r"<think>.*?</think>", "", out, flags=re.DOTALL).strip()
            out = out.replace("\n", " ").strip()
            if len(out) > 200:
                out = out[:200]
            return out
        except Exception as e:
            last_err = e
            print(f"[openai_reader] error on attempt {attempt + 1}/{retries}: {e}", flush=True)
            time.sleep(max(0.0, float(retry_backoff_s)) * float(attempt + 1))
    if fail_soft:
        print(f"[openai_reader] fail-soft: returning empty answer after {retries} failed attempts: {last_err}", flush=True)
        return ""
    raise RuntimeError(f"[openai_reader] failed after {retries} attempts: {last_err}")


def build_rankings_for_query(
    query: str,
    query_id: str,
    d_row: np.ndarray,
    s_row: np.ndarray,
    doc_ids: list[str],
    doc_texts: list[str],
    doc_tokens: list[set[str]],
    doc_prefix_tokens: list[set[str]],
    doc_token_lists: list[list[str]],
    doc_signal_tokens: list[set[str]],
    doc_numeric_literals: list[set[str]],
    table_doc_indices: np.ndarray,
    kg_doc_indices: np.ndarray,
    doc_table_structs: list[dict[str, list[set[str]]] | None],
    doc_kg_path_sets: list[list[set[str]] | None],
    table_tapas_cache: dict[int, dict[str, np.ndarray] | None],
    kg_gnn_cache: dict[int, dict[str, np.ndarray] | None],
    tapas_bundle: dict | None,
    retrieve_topk: int,
    preserve_dense_top: int,
    tessera_late_alpha: float,
    router_prob: np.ndarray,
    query_modality_prior_mix: float,
    query_modality_prior_adaptive: bool,
    query_modality_prior_entropy_scale: float,
    query_modality_prior_disagreement_scale: float,
    query_modality_prior_min: float,
    query_modality_prior_max: float,
    router_entropy: float,
    uncertainty_threshold: float,
    pathmaxsim_weight: float,
    pathmaxsim_kg_threshold: float,
    table_cellmaxsim_weight: float,
    table_cellmaxsim_top_cells: int,
    innovation_scheme2: bool,
    scheme2_cross_modal_weight: float,
    scheme2_token_maxsim_weight: float,
    adapter_plus_mode: bool,
    adapter_official_lite: bool,
    heavy_schemeb_mode: bool,
    heavy_table_encoder_weight: float,
    heavy_kg_path_weight: float,
    heavy_token_late_weight: float,
    heavy_query_max_tokens: int,
    heavy_table_max_cells: int,
    heavy_token_doc_max_tokens: int,
    heavy_table_backend: str,
    heavy_table_tapas_topn: int,
    heavy_table_max_rows: int,
    heavy_table_max_cols: int,
    heavy_table_agg_cell_logit: float,
    heavy_table_agg_row_logit: float,
    heavy_table_agg_col_logit: float,
    heavy_table_agg_temp: float,
    heavy_kg_backend: str,
    heavy_kg_gnn_topn: int,
    heavy_kg_max_hops: int,
    heavy_kg_max_paths: int,
    heavy_kg_contrastive_temp: float,
    heavy_kg_hard_negative_mode: str,
    heavy_kg_hard_negative_topdocs: int,
    heavy_kg_hard_negative_max_paths: int,
    heavy_token_cross_modal_weight: float,
    heavy_branch_candidate_expand_k: int,
    heavy_branch_candidate_table_weight: float,
    heavy_branch_candidate_kg_weight: float,
    heavy_branch_candidate_max_total: int,
    heavy_score_calibration: str,
    heavy_score_calibration_nonzero_only: bool,
    qa_objective_retrieval_weight: float,
    qa_objective_targeted_only: bool,
    upo_lite_retrieval_weight: float,
    upo_lite_targeted_only: bool,
    retrieval_conflict_penalty_weight: float,
    retrieval_conflict_targeted_only: bool,
    retrieval_conflict_table_kg_only: bool,
    retrieval_conflict_risk_gating: bool,
    retrieval_conflict_risk_low: float,
    retrieval_conflict_risk_high: float,
    retrieval_conflict_risk_probe_k: int,
    retrieval_conflict_anchor_k: int,
    retrieval_conflict_max_literals_per_doc: int,
    retrieval_conflict_sensitive_target_scale: float,
    conflict_bundle: ConflictBundle | None,
    selected_methods: set[str],
    tessera_candidate_pool_k: int,
    tessera_retrieval_multi_agent: bool,
    tessera_retrieval_agent_pool_k: int,
    tessera_retrieval_dense_pool_k: int,
    tessera_retrieval_sparse_pool_k: int,
    tessera_retrieval_preserve_top: int,
    tessera_retrieval_base_weight: float,
    tessera_retrieval_dense_weight: float,
    tessera_retrieval_sparse_weight: float,
    tessera_retrieval_target_weight: float,
    tessera_retrieval_coverage_weight: float,
    tessera_retrieval_diversity_weight: float,
    tessera_retrieval_dense_rescue_k: int,
    tessera_retrieval_dense_rescue_pool_k: int,
    tessera_retrieval_sibling_seed_k: int,
    tessera_retrieval_sibling_window: int,
    tessera_retrieval_sibling_weight: float,
    tessera_retrieval_moe: bool,
    tessera_moe_pool_k: int,
    tessera_moe_prf_seed_k: int,
    tessera_moe_prf_dense_seed_k: int,
    tessera_moe_prf_sparse_seed_k: int,
    tessera_moe_prf_max_terms: int,
    tessera_moe_sibling_seed_k: int,
    tessera_moe_sibling_window: int,
    tessera_moe_sibling_weight: float,
    tessera_ser_bundle: object | None,
    tessera_ser_candidate_pool_k: int,
    tessera_ser_dense_pool_k: int,
    tessera_ser_sparse_pool_k: int,
    tessera_ser_preserve_top: int,
    tessera_ser_blend_weight: float,
    tessera_ser_diversity_weight: float,
    tessera_ser_evidence_rescue_k: int,
    tessera_ser_evidence_rescue_pool_k: int,
    tessera_ser_evidence_preserve_top: int,
    tessera_ser_evidence_redundancy_weight: float,
    tessera_ser_evidence_min_gain: float,
    tessera_ser_plan_adaptive: bool,
    tessera_ser_plan_dense_weight: float,
    tessera_ser_plan_sparse_weight: float,
    tessera_ser_plan_lexical_weight: float,
    tessera_ser_plan_slot_weight: float,
    tessera_ser_evidence_set_selection: bool,
    tessera_ser_evidence_set_preserve_top: int,
    tessera_ser_evidence_set_pool_k: int,
    tessera_ser_evidence_set_cardinality_threshold: float,
    tessera_ser_evidence_set_learned_weight: float,
    tessera_ser_evidence_set_base_weight: float,
    tessera_ser_evidence_set_dense_weight: float,
    tessera_ser_evidence_set_sparse_weight: float,
    tessera_ser_evidence_set_probe_weight: float,
    tessera_ser_evidence_set_slot_weight: float,
    tessera_ser_evidence_set_anchor_weight: float,
    tessera_ser_evidence_set_family_weight: float,
    tessera_ser_evidence_set_redundancy_weight: float,
    tessera_graph_evidence_expansion: bool,
    tessera_gee_post_rerank: bool,
    tessera_gee_candidate_pool_k: int,
    tessera_gee_dense_pool_k: int,
    tessera_gee_sparse_pool_k: int,
    tessera_gee_graph_seed_k: int,
    tessera_gee_graph_window: int,
    tessera_gee_preserve_top: int,
    tessera_gee_trigger_threshold: float,
    tessera_gee_base_weight: float,
    tessera_gee_dense_weight: float,
    tessera_gee_sparse_weight: float,
    tessera_gee_probe_weight: float,
    tessera_gee_graph_weight: float,
    tessera_gee_slot_weight: float,
    tessera_gee_sibling_weight: float,
    tessera_gee_redundancy_weight: float,
    doc_id_to_idx: dict[str, int] | None,
    debug_trace: dict[str, list[float]] | None = None,
    tessera_v9_enabled: bool = False,
    tessera_v9_local_rerank: bool = False,
    tessera_v9_dense_pool_k: int = 1200,
    tessera_v9_sparse_pool_k: int = 1800,
    tessera_v9_candidate_pool_k: int = 900,
    tessera_v9_graph_seed_k: int = 36,
    tessera_v9_graph_window: int = 1,
    tessera_v9_preserve_top: int = 0,
    tessera_v9_base_weight: float = 0.28,
    tessera_v9_dense_weight: float = 0.30,
    tessera_v9_sparse_weight: float = 0.20,
    tessera_v9_probe_weight: float = 0.16,
    tessera_v9_graph_weight: float = 0.08,
    tessera_v9_slot_weight: float = 0.08,
    tessera_v9_diversity_weight: float = 0.018,
    tessera_v9_modality_weight: float = 0.04,
    tessera_v10_conservative_rerank: bool = False,
    tessera_v10_preserve_top: int = 1,
    tessera_v10_direct_preserve_top: int = 2,
    tessera_v10_reference_pool_k: int = 40,
    tessera_v10_candidate_pool_k: int = 120,
    tessera_v10_reference_weight: float = 0.54,
    tessera_v10_current_weight: float = 0.24,
    tessera_v10_base_weight: float = 0.10,
    tessera_v10_dense_weight: float = 0.07,
    tessera_v10_sparse_weight: float = 0.04,
    tessera_v10_probe_weight: float = 0.04,
    tessera_v10_slot_weight: float = 0.03,
    tessera_v10_diversity_weight: float = 0.012,
    tessera_v10_margin: float = 0.035,
    tessera_v10_relevance_floor: float = 0.18,
    tessera_source_evidence_fusion: bool = False,
    tessera_source_evidence_slot_verifier_bundle: object | None = None,
    tessera_source_evidence_topk: int = 5,
    tessera_source_evidence_candidate_pool_k: int = 80,
    tessera_source_evidence_preserve_top: int = 1,
    tessera_source_evidence_base_weight: float = 0.34,
    tessera_source_evidence_dense_weight: float = 0.16,
    tessera_source_evidence_sparse_weight: float = 0.08,
    tessera_source_evidence_reference_weight: float = 0.14,
    tessera_source_evidence_lexical_weight: float = 0.10,
    tessera_source_evidence_modality_prior_weight: float = 0.12,
    tessera_source_evidence_source_balance_weight: float = 0.10,
    tessera_source_evidence_target_family_weight: float = 0.08,
    tessera_source_evidence_diversity_weight: float = 0.025,
    tessera_source_evidence_replacement_margin: float = 0.01,
    tessera_source_evidence_min_candidate_score: float = 0.08,
    tessera_source_evidence_dense_guard: bool = False,
    tessera_source_evidence_dense_guard_topn: int = 5,
    tessera_source_evidence_dense_guard_prefixes: str = "",
    tessera_source_evidence_dense_guard_weight: float = 0.22,
    tessera_source_evidence_dense_rank_weight: float = 0.10,
    tessera_source_evidence_current_rank_weight: float = 0.06,
    tessera_source_evidence_source_balance_prefixes: str = "",
    tessera_source_evidence_max_changed_slots: int = 0,
    tessera_source_evidence_slot_acceptance_guard: bool = False,
    tessera_source_evidence_slot_acceptance_prefixes: str = "",
    tessera_source_evidence_slot_acceptance_margin: float = 0.02,
    tessera_source_evidence_budget_composer: bool = False,
    tessera_source_evidence_budget_prefixes: str = "",
    tessera_source_evidence_budget_candidate_pool_k: int = 180,
    tessera_source_evidence_budget_start_slot: int = 4,
    tessera_source_evidence_budget_max_selected: int = 2,
    tessera_source_evidence_budget_score_weight: float = 0.10,
    tessera_source_evidence_budget_sibling_weight: float = 0.16,
    tessera_source_evidence_budget_source_quota_weight: float = 0.08,
    tessera_source_evidence_budget_tail_rank_weight: float = 0.08,
    tessera_source_evidence_budget_reference_weight: float = 0.10,
    tessera_source_evidence_budget_margin: float = 0.006,
    tessera_source_evidence_budget_redundancy_weight: float = 0.01,
    tessera_source_evidence_sibling_filler: bool = False,
    tessera_source_evidence_sibling_filler_prefixes: str = "",
    tessera_source_evidence_sibling_filler_candidate_pool_k: int = 120,
    tessera_source_evidence_sibling_filler_start_slot: int = 4,
    tessera_source_evidence_sibling_filler_max_selected: int = 1,
    tessera_source_evidence_sibling_filler_tail_topn: int = 10,
    tessera_source_evidence_sibling_filler_reference_topn: int = 10,
    tessera_source_evidence_sibling_filler_margin: float = 0.02,
    tessera_source_evidence_sibling_filler_sibling_weight: float = 0.22,
    tessera_source_evidence_sibling_filler_reference_weight: float = 0.18,
    tessera_source_evidence_sibling_filler_tail_weight: float = 0.10,
    tessera_source_evidence_sibling_filler_dense_weight: float = 0.08,
    tessera_source_evidence_sibling_filler_source_weight: float = 0.08,
    tessera_source_evidence_sibling_filler_redundancy_weight: float = 0.008,
    tessera_source_evidence_slot_verifier: bool = False,
    tessera_source_evidence_slot_verifier_prefixes: str = "",
    tessera_source_evidence_slot_verifier_candidate_pool_k: int = 220,
    tessera_source_evidence_slot_verifier_start_slot: int = 4,
    tessera_source_evidence_slot_verifier_max_selected: int = 2,
    tessera_source_evidence_slot_verifier_tail_topn: int = 12,
    tessera_source_evidence_slot_verifier_reference_topn: int = 12,
    tessera_source_evidence_slot_verifier_dense_topn: int = 24,
    tessera_source_evidence_slot_verifier_margin: float = 0.025,
    tessera_source_evidence_slot_verifier_min_score: float = 0.42,
    tessera_source_evidence_slot_verifier_model_threshold: float = 0.68,
    tessera_source_evidence_slot_verifier_static_weight: float = 0.20,
    tessera_source_evidence_slot_verifier_reference_weight: float = 0.20,
    tessera_source_evidence_slot_verifier_dense_weight: float = 0.14,
    tessera_source_evidence_slot_verifier_tail_weight: float = 0.12,
    tessera_source_evidence_slot_verifier_sibling_weight: float = 0.12,
    tessera_source_evidence_slot_verifier_source_weight: float = 0.10,
    tessera_source_evidence_slot_verifier_lexical_weight: float = 0.08,
    tessera_source_evidence_slot_verifier_family_weight: float = 0.06,
    tessera_source_evidence_slot_verifier_redundancy_weight: float = 0.012,
    tessera_source_evidence_kg_preservation_guard: bool = False,
    tessera_source_evidence_kg_preservation_prefixes: str = "cwq,webqsp",
    tessera_source_evidence_kg_preservation_min_kg: int = 1,
    tessera_source_evidence_kg_preservation_candidate_pool_k: int = 160,
    tessera_source_evidence_kg_preservation_start_slot: int = 2,
    tessera_source_evidence_kg_preservation_margin: float = 0.015,
    tessera_source_evidence_kg_preservation_reference_weight: float = 0.24,
    tessera_source_evidence_kg_preservation_dense_weight: float = 0.16,
    tessera_source_evidence_kg_preservation_current_weight: float = 0.12,
    tessera_source_evidence_kg_preservation_family_weight: float = 0.10,
    tessera_source_evidence_kg_preservation_lexical_weight: float = 0.06,
    tessera_source_evidence_kg_verifier_bundle: object | None = None,
    tessera_source_evidence_kg_verifier_weight: float = 0.0,
    tessera_source_evidence_kg_verifier_min_score: float = 0.0,
    tessera_source_evidence_kg_verify_existing: bool = False,
    tessera_source_evidence_kg_verify_existing_max_replacements: int = 1,
    tessera_source_budgeter_bundle: object | None = None,
    tessera_source_budgeter_top1_guard: bool = False,
    tessera_source_budgeter_need_threshold: float = 0.45,
    tessera_source_budgeter_non_kg_top1_max_kg: int = 1,
    tessera_source_head_selector: bool = False,
    tessera_source_head_topn: int = 5,
    tessera_source_head_source_weight: float = 0.42,
    tessera_source_head_same_query_weight: float = 0.16,
    tessera_source_head_position_weight: float = 0.16,
    tessera_source_head_reference_weight: float = 0.12,
    tessera_source_head_lexical_weight: float = 0.10,
    tessera_source_head_base_weight: float = 0.08,
    tessera_source_head_dense_weight: float = 0.05,
    tessera_source_head_sparse_weight: float = 0.04,
    tessera_source_head_margin: float = 0.015,
    tessera_source_head_off_source_margin: float = 0.04,
    tessera_source_action_policy_bundle: object | None = None,
    tessera_source_action_policy_min_prob: float = 0.42,
    tessera_source_action_policy_topk: int = 5,
    tessera_source_action_policy_pool_k: int = 10,
    tessera_final_evidence_composer: bool = False,
    tessera_final_evidence_topk: int = 5,
    tessera_final_evidence_candidate_pool_k: int = 120,
    tessera_final_evidence_dense_pool_k: int = 80,
    tessera_final_evidence_sparse_pool_k: int = 80,
    tessera_final_evidence_preserve_top: int = 1,
    tessera_final_evidence_max_replacements: int = 1,
    tessera_final_evidence_min_candidate_score: float = 0.62,
    tessera_final_evidence_replacement_margin: float = 0.08,
    tessera_final_evidence_min_query_overlap: float = 0.0,
    tessera_final_evidence_source_need_weight: float = 0.035,
    tessera_final_evidence_redundancy_weight: float = 0.025,
    tessera_final_evidence_verifier_bundle: object | None = None,
    tessera_final_evidence_verifier_threshold: float = 0.70,
    tessera_final_evidence_verifier_margin: float = 0.0,
) -> dict[str, list[int]]:
    selected_method_set = set(selected_methods)
    t_profile0 = time.perf_counter()
    q_tokens = tokenize(query)
    q_token_list = tokenize_list(query)
    upo_concept = str(infer_upo_lite_concept(query)).strip().lower()
    upo_prior = infer_upo_lite_modality_prior(query, query_id=query_id)
    if bool(query_modality_prior_adaptive):
        blended_router_prob, query_modality_prior, prior_mix_eff, prior_disagreement = blend_router_with_query_prior_adaptive(
            router_prob=router_prob,
            query=query,
            query_id=query_id,
            prior_mix=float(query_modality_prior_mix),
            router_entropy=float(router_entropy),
            uncertainty_threshold=float(uncertainty_threshold),
            entropy_scale=float(query_modality_prior_entropy_scale),
            disagreement_scale=float(query_modality_prior_disagreement_scale),
            mix_min=float(query_modality_prior_min),
            mix_max=float(query_modality_prior_max),
        )
    else:
        blended_router_prob, query_modality_prior, prior_mix_eff, prior_disagreement = blend_router_with_query_prior(
            router_prob=router_prob,
            query=query,
            query_id=query_id,
            prior_mix=float(query_modality_prior_mix),
        )
    table_p = float(blended_router_prob[ROUTER_LABEL_TO_IDX["table"]])
    text_p = float(blended_router_prob[ROUTER_LABEL_TO_IDX["text"]])
    kg_p = float(blended_router_prob[ROUTER_LABEL_TO_IDX["kg"]])
    if debug_trace is not None:
        debug_trace.setdefault("query_prior_mix_eff", []).append(float(prior_mix_eff))
        debug_trace.setdefault("query_prior_disagreement", []).append(float(prior_disagreement))
        debug_trace.setdefault("router_table_prob_blend", []).append(table_p)
        debug_trace.setdefault("router_text_prob_blend", []).append(text_p)
        debug_trace.setdefault("router_kg_prob_blend", []).append(kg_p)
        debug_trace.setdefault("query_prior_table", []).append(float(query_modality_prior[1]))
        debug_trace.setdefault("query_prior_text", []).append(float(query_modality_prior[0]))
        debug_trace.setdefault("query_prior_kg", []).append(float(query_modality_prior[2]))
    source_budget: dict[str, float | str] | None = None
    source_budget_allowed_top1: set[str] | None = None
    if tessera_source_budgeter_bundle is not None:
        pred = tessera_source_budgeter_bundle.predict(query, query_id)
        source_budget = {"top1_source": str(pred.top1_source)}
        for label in SOURCE_BUDGET_LABELS:
            source_budget[f"need_{label}"] = float(pred.need_probs.get(label, 0.0))
            source_budget[f"top1_prob_{label}"] = float(pred.top1_probs.get(label, 0.0))
        if bool(tessera_source_budgeter_top1_guard) and str(pred.top1_source) in SOURCE_BUDGET_LABELS:
            source_budget_allowed_top1 = {str(pred.top1_source)}
        if debug_trace is not None:
            debug_trace.setdefault("tessera_source_budgeter_active", []).append(1.0)
            debug_trace.setdefault("tessera_source_budgeter_top1_source_index", []).append(
                float(SOURCE_BUDGET_LABELS.index(str(pred.top1_source)) if str(pred.top1_source) in SOURCE_BUDGET_LABELS else -1)
            )
            for label in SOURCE_BUDGET_LABELS:
                debug_trace.setdefault(f"tessera_source_budgeter_top1_prob_{label}", []).append(
                    float(pred.top1_probs.get(label, 0.0))
                )
                debug_trace.setdefault(f"tessera_source_budgeter_need_prob_{label}", []).append(
                    float(pred.need_probs.get(label, 0.0))
                )
    elif debug_trace is not None:
        debug_trace.setdefault("tessera_source_budgeter_active", []).append(0.0)
    table_gate = table_intent_gate(query, table_p)
    gated_table_cell_weight = float(table_cellmaxsim_weight) * table_gate
    q_heavy_mat = np.zeros((0, HEAVY_TOKEN_VEC_DIM), dtype=np.float32)
    if heavy_schemeb_mode:
        q_heavy_mat = build_token_matrix(q_token_list, max_tokens=max(8, int(heavy_query_max_tokens)))

    rankings: dict[str, list[int]] = {}
    dense_top400 = topk_indices(d_row, 400)
    sparse_top500 = topk_indices(s_row, 500)
    # Always keep a dense reference ranking for TESSERA anchoring/diagnostics.
    # It is only evaluated as a baseline when dense_concat is selected.
    rankings["dense_concat"] = dense_top400[:retrieve_topk].tolist()
    rankings["dense_context_pool"] = dense_top400.tolist()
    if "dense_concat" in selected_method_set:
        rankings["dense_concat"] = dense_top400[:retrieve_topk].tolist()
    if "naive_rag" in selected_method_set:
        naive_candidates = sparse_top500[:500]
        naive_sorted = naive_candidates[np.argsort(-d_row[naive_candidates])]
        rankings["naive_rag"] = naive_sorted[:retrieve_topk].tolist()

    if debug_trace is not None:
        debug_trace.setdefault("ranking_topk_ms", []).append((time.perf_counter() - t_profile0) * 1000.0)

    t_profile1 = time.perf_counter()
    candidate, d_norm, s_norm = build_adapter_candidate_pool(
        d_row=d_row,
        s_row=s_row,
        dense_top400=dense_top400,
        sparse_top500=sparse_top500,
    )
    if "carp" in selected_method_set:
        rankings["carp"] = build_carp_ranking(
            query_tokens=q_tokens,
            candidate=candidate,
            d_norm=d_norm,
            s_norm=s_norm,
            dense_top400=dense_top400,
            doc_ids=doc_ids,
            doc_tokens=doc_tokens,
            adapter_plus_mode=bool(adapter_plus_mode),
            adapter_official_lite=bool(adapter_official_lite),
            source_bucket_fn=source_bucket,
            retrieve_topk=retrieve_topk,
        ).tolist()
    if "tablerag" in selected_method_set:
        rankings["tablerag"] = build_tablerag_ranking(
            query_tokens=q_tokens,
            candidate=candidate,
            d_norm=d_norm,
            s_norm=s_norm,
            doc_ids=doc_ids,
            doc_table_structs=doc_table_structs,
            table_gate=float(table_gate),
            table_cellmaxsim_weight=float(table_cellmaxsim_weight),
            table_cellmaxsim_top_cells=int(table_cellmaxsim_top_cells),
            adapter_plus_mode=bool(adapter_plus_mode),
            adapter_official_lite=bool(adapter_official_lite),
            source_bucket_fn=source_bucket,
            cellmaxsim_like_score_fn=cellmaxsim_like_score,
            table_schema_alignment_score_fn=table_schema_alignment_score,
            retrieve_topk=retrieve_topk,
        ).tolist()
    if "quasar" in selected_method_set:
        rankings["quasar"] = build_quasar_ranking(
            candidate=candidate,
            d_norm=d_norm,
            s_norm=s_norm,
            doc_ids=doc_ids,
            text_p=float(text_p),
            table_p=float(table_p),
            kg_p=float(kg_p),
            adapter_official_lite=bool(adapter_official_lite),
            source_bucket_fn=source_bucket,
            retrieve_topk=retrieve_topk,
        )

    if debug_trace is not None:
        debug_trace.setdefault("ranking_candidate_ms", []).append((time.perf_counter() - t_profile1) * 1000.0)

    # TESSERA ranking uses route-aware fusion + token-level late bonus +
    # graph relation hint + diversity-aware decoding.
    t_profile2 = time.perf_counter()
    candidate_sources: list[np.ndarray] = [dense_top400[:500], sparse_top500[:500]]
    if heavy_schemeb_mode and int(heavy_branch_candidate_expand_k) > 0:
        extra_k = int(heavy_branch_candidate_expand_k)

        if table_doc_indices.size > 0:
            tw = float(np.clip(heavy_branch_candidate_table_weight, 0.0, 1.0))
            table_scores = tw * d_row[table_doc_indices] + (1.0 - tw) * s_row[table_doc_indices]
            table_top_local = topk_indices(table_scores, min(extra_k, int(table_doc_indices.size)))
            candidate_sources.append(table_doc_indices[table_top_local])

        if kg_doc_indices.size > 0:
            kw = float(np.clip(heavy_branch_candidate_kg_weight, 0.0, 1.0))
            kg_scores = kw * d_row[kg_doc_indices] + (1.0 - kw) * s_row[kg_doc_indices]
            kg_top_local = topk_indices(kg_scores, min(extra_k, int(kg_doc_indices.size)))
            candidate_sources.append(kg_doc_indices[kg_top_local])

    uni_candidate = np.unique(np.concatenate(candidate_sources)).astype(np.int64)
    max_total = max(0, int(heavy_branch_candidate_max_total))
    if max_total > 0 and len(uni_candidate) > max_total:
        prune_mix = 0.5 * d_row[uni_candidate] + 0.5 * s_row[uni_candidate]
        keep = topk_indices(prune_mix, max_total)
        uni_candidate = uni_candidate[keep]
    if debug_trace is not None:
        debug_trace.setdefault("uni_candidate_size", []).append(float(len(uni_candidate)))
    uni_d_norm = normalize_scores(d_row[uni_candidate])
    uni_s_norm = normalize_scores(s_row[uni_candidate])
    late_bonus = np.asarray(
        [len(q_tokens & doc_tokens[j]) / max(1, len(q_tokens)) for j in uni_candidate],
        dtype=np.float32,
    )
    if float(router_entropy) >= float(uncertainty_threshold):
        w_dense, w_sparse = 0.68, 0.32
    else:
        w_dense, w_sparse = 0.76, 0.24

    scheme2_unc = 0.0
    late_alpha_eff = float(tessera_late_alpha)
    modality_bonus_weight = 0.10
    cross_modal_bonus_weight = 0.0
    table_cell_scale = 1.0
    if innovation_scheme2:
        scheme2_unc = scheme2_uncertainty_factor(float(router_entropy), float(uncertainty_threshold))
        late_alpha_eff = float(tessera_late_alpha) * (1.0 + 0.25 * scheme2_unc)
        modality_bonus_weight = 0.10 + 0.06 * scheme2_unc
        cross_modal_bonus_weight = float(scheme2_cross_modal_weight) * (0.40 + 0.60 * scheme2_unc)
        table_cell_scale = 1.0 + 0.50 * table_gate

    modality_bonus = np.asarray(
        [
            table_p if source_bucket(doc_ids[j]) == "table" else (kg_p if source_bucket(doc_ids[j]) == "kg" else text_p)
            for j in uni_candidate
        ],
        dtype=np.float32,
    )
    path_bonus = np.asarray(
        [relation_hint_bonus(query, doc_texts[j], source_bucket(doc_ids[j])) for j in uni_candidate],
        dtype=np.float32,
    )
    if float(gated_table_cell_weight) > 0.0:
        table_cell_arr = np.asarray(
            [
                cellmaxsim_like_score(q_tokens, doc_table_structs[j], top_cells=int(table_cellmaxsim_top_cells))
                if source_bucket(doc_ids[j]) == "table"
                else 0.0
                for j in uni_candidate
            ],
            dtype=np.float32,
        )
    else:
        table_cell_arr = np.zeros_like(uni_d_norm)

    cross_bonus_arr = np.zeros_like(uni_d_norm)
    if innovation_scheme2 and cross_modal_bonus_weight > 0.0:
        mixed_scores = 0.50 * uni_d_norm + 0.50 * uni_s_norm
        seed_sets = build_cross_modal_seed_sets(
            uni_candidate,
            mixed_scores,
            doc_ids,
            doc_tokens,
            per_bucket=12,
        )
        cross_bonus_arr = np.asarray(
            [
                cross_modal_agreement_bonus(
                    int(j),
                    source_bucket(doc_ids[int(j)]),
                    doc_tokens,
                    seed_sets,
                )
                for j in uni_candidate
            ],
            dtype=np.float32,
        )

    token_maxsim_arr = np.zeros_like(uni_d_norm)
    token_maxsim_weight = 0.0
    if innovation_scheme2 and float(scheme2_token_maxsim_weight) > 0.0:
        token_maxsim_weight = float(scheme2_token_maxsim_weight) * (0.60 + 0.40 * scheme2_unc)
        token_maxsim_arr = np.asarray(
            [
                token_maxsim_like_score(
                    q_tokens,
                    doc_tokens[int(j)],
                    doc_prefix_tokens[int(j)],
                )
                for j in uni_candidate
            ],
            dtype=np.float32,
        )

    target_type = str(infer_qa_target_type(query)).strip().lower()
    qa_objective_weight_eff = max(0.0, float(qa_objective_retrieval_weight))
    if qa_objective_weight_eff > 0.0 and bool(qa_objective_targeted_only):
        if target_type not in {"number", "year", "boolean"}:
            qa_objective_weight_eff = 0.0
    qa_objective_arr = np.zeros_like(uni_d_norm)
    if qa_objective_weight_eff > 0.0:
        qa_objective_arr = np.asarray(
            [
                qa_objective_retrieval_score(
                    query=query,
                    query_tokens=q_tokens,
                    doc_text=doc_texts[int(j)],
                    doc_tokens=doc_tokens[int(j)],
                    bucket=source_bucket(doc_ids[int(j)]),
                )
                for j in uni_candidate
            ],
            dtype=np.float32,
        )

    upo_retrieval_weight_eff = max(0.0, float(upo_lite_retrieval_weight))
    if upo_retrieval_weight_eff > 0.0 and bool(upo_lite_targeted_only):
        if upo_concept not in {"number", "year", "location", "relation"}:
            upo_retrieval_weight_eff = 0.0
    upo_modality_arr = np.asarray(
        [
            float(upo_prior[ROUTER_LABEL_TO_IDX[source_bucket(doc_ids[int(j)])]])
            for j in uni_candidate
        ],
        dtype=np.float32,
    )

    kg_factor = max(0.0, (kg_p - float(pathmaxsim_kg_threshold)) / max(1e-6, 1.0 - float(pathmaxsim_kg_threshold)))
    mixed_scores = 0.50 * uni_d_norm + 0.50 * uni_s_norm
    heavy_table_arr = np.zeros_like(uni_d_norm)
    heavy_kg_path_arr = np.zeros_like(uni_d_norm)
    heavy_token_late_arr = np.zeros_like(uni_d_norm)
    heavy_token_cross_arr = np.zeros_like(uni_d_norm)
    heavy_table_weight_eff = 0.0
    heavy_kg_path_weight_eff = 0.0
    heavy_token_late_weight_eff = 0.0
    heavy_token_cross_weight_eff = 0.0
    tapas_query_ready = 0.0
    if heavy_schemeb_mode and q_heavy_mat.size > 0:
        heavy_table_weight_eff = max(0.0, float(heavy_table_encoder_weight)) * (0.45 + 0.55 * table_gate)
        heavy_kg_path_weight_eff = max(0.0, float(heavy_kg_path_weight)) * kg_factor
        heavy_token_late_weight_eff = max(0.0, float(heavy_token_late_weight))
        heavy_token_cross_weight_eff = max(0.0, float(heavy_token_cross_modal_weight))
        if innovation_scheme2:
            heavy_kg_path_weight_eff *= 0.40 + 0.60 * scheme2_unc
            heavy_token_late_weight_eff *= 0.80 + 0.20 * scheme2_unc
            heavy_token_cross_weight_eff *= 0.70 + 0.30 * scheme2_unc

        if heavy_table_weight_eff > 0.0:
            table_backend = str(heavy_table_backend).lower().strip()
            q_table_mat = q_heavy_mat
            tapas_ready = False
            if table_backend == "tapas" and tapas_bundle is not None:
                q_tapas = tapas_query_token_matrix(query, tapas_bundle, max_tokens=max(8, int(heavy_query_max_tokens)))
                if q_tapas.size > 0:
                    q_table_mat = q_tapas
                    tapas_ready = True
                    tapas_query_ready = 1.0

            table_pos = [p for p, j in enumerate(uni_candidate.tolist()) if source_bucket(doc_ids[int(j)]) == "table"]
            table_rank = sorted(table_pos, key=lambda p: float(mixed_scores[p]), reverse=True)
            table_topn = int(heavy_table_tapas_topn)
            if table_topn <= 0:
                table_top = set(table_rank)
            else:
                table_top = set(table_rank[: max(1, table_topn)])

            tab_vals = []
            for pos, j in enumerate(uni_candidate.tolist()):
                j = int(j)
                if source_bucket(doc_ids[j]) != "table":
                    tab_vals.append(0.0)
                    continue

                use_tapas = table_backend == "tapas" and tapas_ready and pos in table_top
                if use_tapas:
                    if j not in table_tapas_cache:
                        table_tapas_cache[j] = tapas_table_repr(
                            doc_texts[j],
                            tapas_bundle,
                            max_rows=int(heavy_table_max_rows),
                            max_cols=int(heavy_table_max_cols),
                            max_cells=int(heavy_table_max_cells),
                        )
                    sc = table_cell_encoder_score_tapas(
                        q_table_mat,
                        table_tapas_cache.get(j),
                        cell_logit=float(heavy_table_agg_cell_logit),
                        row_logit=float(heavy_table_agg_row_logit),
                        col_logit=float(heavy_table_agg_col_logit),
                        temperature=float(heavy_table_agg_temp),
                    )
                    if sc <= 1e-9:
                        sc = table_cell_encoder_score(
                            q_heavy_mat,
                            doc_table_structs[j],
                            max_cells=int(heavy_table_max_cells),
                        )
                else:
                    sc = table_cell_encoder_score(
                        q_heavy_mat,
                        doc_table_structs[j],
                        max_cells=int(heavy_table_max_cells),
                    )
                tab_vals.append(sc)
            heavy_table_arr = np.asarray(tab_vals, dtype=np.float32)

        if heavy_kg_path_weight_eff > 0.0:
            kg_backend = str(heavy_kg_backend).lower().strip()
            kg_pos = [p for p, j in enumerate(uni_candidate.tolist()) if source_bucket(doc_ids[int(j)]) == "kg"]
            kg_rank = sorted(kg_pos, key=lambda p: float(mixed_scores[p]), reverse=True)
            kg_topn = int(heavy_kg_gnn_topn)
            if kg_topn <= 0:
                kg_top = set(kg_rank)
                kg_top_positions = list(kg_rank)
            else:
                kg_top_positions = kg_rank[: max(1, kg_topn)]
                kg_top = set(kg_top_positions)

            hard_neg_mode = str(heavy_kg_hard_negative_mode).lower().strip()
            hard_neg_map: dict[int, np.ndarray] = {}
            if kg_backend == "gnn" and hard_neg_mode == "cross_doc_hard":
                hard_neg_map = build_query_aware_kg_hard_negatives(
                    query_mat=q_heavy_mat,
                    kg_top_positions=kg_top_positions,
                    uni_candidate=uni_candidate,
                    kg_gnn_cache=kg_gnn_cache,
                    doc_ids=doc_ids,
                    doc_texts=doc_texts,
                    max_hops=int(heavy_kg_max_hops),
                    max_paths=int(heavy_kg_max_paths),
                    top_docs=int(heavy_kg_hard_negative_topdocs),
                    max_neg_paths=int(heavy_kg_hard_negative_max_paths),
                )

            kg_vals = []
            for pos, j in enumerate(uni_candidate.tolist()):
                j = int(j)
                if source_bucket(doc_ids[j]) != "kg":
                    kg_vals.append(0.0)
                    continue
                use_gnn = kg_backend == "gnn" and pos in kg_top
                if use_gnn:
                    if j not in kg_gnn_cache:
                        kg_gnn_cache[j] = build_kg_gnn_doc_repr(
                            doc_texts[j],
                            max_hops=int(heavy_kg_max_hops),
                            max_paths=int(heavy_kg_max_paths),
                        )
                    sc = kg_path_encoder_score_gnn(
                        q_heavy_mat,
                        kg_gnn_cache.get(j),
                        contrastive_temp=float(heavy_kg_contrastive_temp),
                        extra_neg_mat=hard_neg_map.get(j),
                    )
                    if sc <= 1e-9:
                        sc = kg_path_encoder_score(q_heavy_mat, doc_kg_path_sets[j])
                else:
                    sc = kg_path_encoder_score(q_heavy_mat, doc_kg_path_sets[j])
                kg_vals.append(sc)
            heavy_kg_path_arr = np.asarray(kg_vals, dtype=np.float32)

        if heavy_token_late_weight_eff > 0.0:
            heavy_token_late_arr = np.asarray(
                [
                    token_late_interaction_score(
                        q_heavy_mat,
                        doc_token_lists[int(j)],
                        max_doc_tokens=int(heavy_token_doc_max_tokens),
                    )
                    for j in uni_candidate
                ],
                dtype=np.float32,
            )

        if heavy_token_cross_weight_eff > 0.0:
            seed_mats = build_modality_seed_token_mats(
                uni_candidate,
                mixed_scores,
                doc_ids,
                doc_token_lists,
                per_bucket_docs=8,
            )
            cross_vals = []
            for j in uni_candidate.tolist():
                j = int(j)
                cand_mat = build_token_matrix(doc_token_lists[j], max_tokens=max(24, int(heavy_token_doc_max_tokens)))
                cross_vals.append(
                    token_cross_modal_interaction_score(
                        cand_mat,
                        source_bucket(doc_ids[j]),
                        seed_mats,
                    )
                )
            heavy_token_cross_arr = np.asarray(cross_vals, dtype=np.float32)

    heavy_calibration_mode = str(heavy_score_calibration).lower().strip()
    if heavy_schemeb_mode and heavy_calibration_mode not in {"", "none"}:
        heavy_table_arr = calibrate_component_scores(
            heavy_table_arr,
            mode=heavy_calibration_mode,
            nonzero_only=bool(heavy_score_calibration_nonzero_only),
        )
        heavy_kg_path_arr = calibrate_component_scores(
            heavy_kg_path_arr,
            mode=heavy_calibration_mode,
            nonzero_only=bool(heavy_score_calibration_nonzero_only),
        )
        heavy_token_late_arr = calibrate_component_scores(
            heavy_token_late_arr,
            mode=heavy_calibration_mode,
            nonzero_only=bool(heavy_score_calibration_nonzero_only),
        )
        heavy_token_cross_arr = calibrate_component_scores(
            heavy_token_cross_arr,
            mode=heavy_calibration_mode,
            nonzero_only=bool(heavy_score_calibration_nonzero_only),
        )

    if debug_trace is not None:
        debug_trace.setdefault("heavy_table_weight_eff", []).append(float(heavy_table_weight_eff))
        debug_trace.setdefault("heavy_kg_path_weight_eff", []).append(float(heavy_kg_path_weight_eff))
        debug_trace.setdefault("heavy_token_late_weight_eff", []).append(float(heavy_token_late_weight_eff))
        debug_trace.setdefault("heavy_token_cross_weight_eff", []).append(float(heavy_token_cross_weight_eff))
        debug_trace.setdefault("qa_objective_weight_eff", []).append(float(qa_objective_weight_eff))
        debug_trace.setdefault("upo_lite_retrieval_weight_eff", []).append(float(upo_retrieval_weight_eff))
        debug_trace.setdefault("upo_lite_prior_table", []).append(float(upo_prior[1]))
        debug_trace.setdefault("upo_lite_prior_text", []).append(float(upo_prior[0]))
        debug_trace.setdefault("upo_lite_prior_kg", []).append(float(upo_prior[2]))
        debug_trace.setdefault("heavy_table_score_mean", []).append(float(np.mean(heavy_table_arr)) if heavy_table_arr.size else 0.0)
        debug_trace.setdefault("heavy_kg_score_mean", []).append(float(np.mean(heavy_kg_path_arr)) if heavy_kg_path_arr.size else 0.0)
        debug_trace.setdefault("heavy_token_late_score_mean", []).append(float(np.mean(heavy_token_late_arr)) if heavy_token_late_arr.size else 0.0)
        debug_trace.setdefault("heavy_token_cross_score_mean", []).append(float(np.mean(heavy_token_cross_arr)) if heavy_token_cross_arr.size else 0.0)
        debug_trace.setdefault("qa_objective_score_mean", []).append(float(np.mean(qa_objective_arr)) if qa_objective_arr.size else 0.0)
        debug_trace.setdefault("tapas_query_ready", []).append(float(tapas_query_ready))

    uni_base_score_no_conflict = (
        w_dense * uni_d_norm
        + w_sparse * uni_s_norm
        + late_alpha_eff * late_bonus
        + modality_bonus_weight * modality_bonus
        + float(pathmaxsim_weight) * kg_factor * path_bonus
        + gated_table_cell_weight * table_cell_scale * table_cell_arr
        + cross_modal_bonus_weight * cross_bonus_arr
        + token_maxsim_weight * token_maxsim_arr
        + qa_objective_weight_eff * qa_objective_arr
        + upo_retrieval_weight_eff * upo_modality_arr
        + heavy_table_weight_eff * heavy_table_arr
        + heavy_kg_path_weight_eff * heavy_kg_path_arr
        + heavy_token_late_weight_eff * heavy_token_late_arr
        + heavy_token_cross_weight_eff * heavy_token_cross_arr
    )

    retrieval_conflict_weight_eff = max(0.0, float(retrieval_conflict_penalty_weight))
    if retrieval_conflict_weight_eff > 0.0 and bool(retrieval_conflict_targeted_only):
        if target_type not in {"number", "year", "location"}:
            retrieval_conflict_weight_eff = 0.0
    if retrieval_conflict_weight_eff > 0.0 and target_type in {"number", "location"}:
        retrieval_conflict_weight_eff *= float(retrieval_conflict_sensitive_target_scale)

    retrieval_conflict_pen_arr = np.zeros_like(uni_d_norm)
    retrieval_conflict_risk = 0.0
    retrieval_conflict_risk_scale = 1.0
    if retrieval_conflict_weight_eff > 0.0:
        order_pos = np.argsort(-uni_base_score_no_conflict)
        candidate_order = [int(uni_candidate[pos]) for pos in order_pos.tolist()]

        if bool(retrieval_conflict_risk_gating):
            retrieval_conflict_risk = estimate_query_conflict_risk(
                query=query,
                candidate_idxs=candidate_order,
                query_tokens=q_tokens,
                doc_ids=doc_ids,
                doc_texts=doc_texts,
                doc_tokens=doc_tokens,
                doc_signal_tokens=doc_signal_tokens,
                doc_numeric_literals=doc_numeric_literals,
                conflict_bundle=conflict_bundle,
                table_kg_only=bool(retrieval_conflict_table_kg_only),
                probe_k=int(retrieval_conflict_risk_probe_k),
                max_literals_per_doc=int(retrieval_conflict_max_literals_per_doc),
            )
            lo = float(retrieval_conflict_risk_low)
            hi = max(lo + 1e-6, float(retrieval_conflict_risk_high))
            if retrieval_conflict_risk <= lo:
                retrieval_conflict_risk_scale = 0.10
            elif retrieval_conflict_risk >= hi:
                retrieval_conflict_risk_scale = 1.00
            else:
                retrieval_conflict_risk_scale = 0.10 + 0.90 * ((retrieval_conflict_risk - lo) / (hi - lo))
            retrieval_conflict_weight_eff *= retrieval_conflict_risk_scale

        anchor_k = max(2, int(retrieval_conflict_anchor_k))
        anchor_idxs = candidate_order[:anchor_k]
        retrieval_conflict_pen_arr = np.asarray(
            [
                cross_modal_conflict_penalty(
                    cand_idx=int(j),
                    selected_idxs=[a for a in anchor_idxs if a != int(j)],
                    doc_ids=doc_ids,
                    query_tokens=q_tokens,
                    doc_tokens=doc_tokens,
                    doc_signal_tokens=doc_signal_tokens,
                    doc_numeric_literals=doc_numeric_literals,
                    table_kg_only=bool(retrieval_conflict_table_kg_only),
                    max_literals_per_doc=int(retrieval_conflict_max_literals_per_doc),
                )
                for j in uni_candidate
            ],
            dtype=np.float32,
        )

    if debug_trace is not None:
        debug_trace.setdefault("retrieval_conflict_weight_eff", []).append(float(retrieval_conflict_weight_eff))
        debug_trace.setdefault("retrieval_conflict_risk", []).append(float(retrieval_conflict_risk))
        debug_trace.setdefault("retrieval_conflict_risk_scale", []).append(float(retrieval_conflict_risk_scale))
        debug_trace.setdefault("retrieval_conflict_penalty_mean", []).append(
            float(np.mean(retrieval_conflict_pen_arr)) if retrieval_conflict_pen_arr.size else 0.0
        )
        debug_trace.setdefault("retrieval_conflict_penalty_nonzero_rate", []).append(
            float(np.mean((retrieval_conflict_pen_arr > 1e-9).astype(np.float32))) if retrieval_conflict_pen_arr.size else 0.0
        )

    uni_base_score = uni_base_score_no_conflict - retrieval_conflict_weight_eff * retrieval_conflict_pen_arr

    v10_pre_v9_rank_all: list[int] = []
    if (
        bool(tessera_v10_conservative_rerank)
        and ("tessera_rag" in selected_method_set or "tessera_submod" in selected_method_set)
        and len(uni_candidate) > 0
    ):
        pre_pool_k = int(tessera_candidate_pool_k)
        if pre_pool_k <= 0:
            pre_pool_k = max(retrieve_topk * 4, 80)
        v10_pre_v9_rank_all = greedy_diverse_topk(
            uni_candidate,
            uni_base_score,
            doc_tokens,
            k=retrieve_topk,
            redundancy_lambda=0.03,
            redundancy_mode="union",
            candidate_pool_k=pre_pool_k,
        )

    v9_cfg = None
    if bool(tessera_v9_enabled):
        if expand_v9_candidates is None or V9CandidateConfig is None:
            raise RuntimeError("tessera_v9 module is required for --tessera-v9")
        v9_cfg = V9CandidateConfig(
            dense_pool_k=int(tessera_v9_dense_pool_k),
            sparse_pool_k=int(tessera_v9_sparse_pool_k),
            candidate_pool_k=int(tessera_v9_candidate_pool_k),
            graph_seed_k=int(tessera_v9_graph_seed_k),
            graph_window=int(tessera_v9_graph_window),
            preserve_top=int(tessera_v9_preserve_top),
            base_weight=float(tessera_v9_base_weight),
            dense_weight=float(tessera_v9_dense_weight),
            sparse_weight=float(tessera_v9_sparse_weight),
            probe_weight=float(tessera_v9_probe_weight),
            graph_weight=float(tessera_v9_graph_weight),
            slot_weight=float(tessera_v9_slot_weight),
            diversity_weight=float(tessera_v9_diversity_weight),
            modality_weight=float(tessera_v9_modality_weight),
        )
        v9_seed_order = [
            int(uni_candidate[int(pos)])
            for pos in np.argsort(np.asarray(uni_base_score, dtype=np.float32))[::-1][
                : max(retrieve_topk, min(len(uni_candidate), int(tessera_v9_graph_seed_k)))
            ].tolist()
        ]
        uni_candidate, uni_base_score, v9_expand_trace = expand_v9_candidates(
            query_text=query,
            candidate_idxs=uni_candidate,
            candidate_base_scores=uni_base_score,
            dense_scores=d_row,
            sparse_scores=s_row,
            doc_ids=doc_ids,
            doc_id_to_idx=doc_id_to_idx,
            doc_tokens=doc_tokens,
            router_prob=blended_router_prob,
            target_type=target_type,
            source_bucket_fn=source_bucket,
            config=v9_cfg,
        )
        rankings["_tessera_v9_candidate_pool"] = uni_candidate.tolist()
        rankings["_tessera_v9_seed_order"] = v9_seed_order
        if debug_trace is not None:
            debug_trace.setdefault("tessera_v9_enabled", []).append(1.0)
            debug_trace.setdefault("tessera_v9_input_candidates", []).append(
                float(getattr(v9_expand_trace, "input_candidate_count", 0))
            )
            debug_trace.setdefault("tessera_v9_output_candidates", []).append(
                float(getattr(v9_expand_trace, "output_candidate_count", 0))
            )
            debug_trace.setdefault("tessera_v9_dense_added", []).append(
                float(getattr(v9_expand_trace, "dense_added", 0))
            )
            debug_trace.setdefault("tessera_v9_sparse_added", []).append(
                float(getattr(v9_expand_trace, "sparse_added", 0))
            )
            debug_trace.setdefault("tessera_v9_graph_added", []).append(
                float(getattr(v9_expand_trace, "graph_added", 0))
            )
            debug_trace.setdefault("tessera_v9_probe_count", []).append(
                float(getattr(v9_expand_trace, "probe_count", 0))
            )
            debug_trace.setdefault("tessera_v9_complex_need", []).append(
                float(getattr(v9_expand_trace, "complex_need", 0.0))
            )
            debug_trace.setdefault("tessera_v9_direct_need", []).append(
                float(getattr(v9_expand_trace, "direct_need", 0.0))
            )
    elif debug_trace is not None:
        debug_trace.setdefault("tessera_v9_enabled", []).append(0.0)

    uni_rank_all: list[int] = []
    uni_nored_rank_all: list[int] = []
    if "tessera_rag" in selected_method_set or "tessera_submod" in selected_method_set:
        pool_k = int(tessera_candidate_pool_k)
        if pool_k <= 0:
            pool_k = max(retrieve_topk * 4, 80)
        uni_rank_all = greedy_diverse_topk(
            uni_candidate,
            uni_base_score,
            doc_tokens,
            k=retrieve_topk,
            redundancy_lambda=0.03,
            redundancy_mode="union",
            candidate_pool_k=pool_k,
        )
    if "ablation_no_redundancy_e2e" in selected_method_set:
        pool_k = int(tessera_candidate_pool_k)
        if pool_k <= 0:
            pool_k = max(retrieve_topk * 4, 80)
        uni_nored_rank_all = greedy_diverse_topk(
            uni_candidate,
            uni_base_score,
            doc_tokens,
            k=retrieve_topk,
            redundancy_lambda=0.0,
            redundancy_mode="union",
            candidate_pool_k=pool_k,
        )

    ser_candidate = uni_candidate
    ser_base_score = uni_base_score
    if (
        bool(tessera_graph_evidence_expansion)
        and ("tessera_rag" in selected_method_set or "tessera_submod" in selected_method_set)
        and uni_rank_all
    ):
        if expand_graph_evidence_candidates is None or GraphEvidenceConfig is None:
            raise RuntimeError("graph_evidence module is required for --tessera-graph-evidence-expansion")
        gee_cfg = GraphEvidenceConfig(
            candidate_pool_k=int(tessera_gee_candidate_pool_k),
            dense_pool_k=int(tessera_gee_dense_pool_k),
            sparse_pool_k=int(tessera_gee_sparse_pool_k),
            graph_seed_k=int(tessera_gee_graph_seed_k),
            graph_window=int(tessera_gee_graph_window),
            preserve_top=int(tessera_gee_preserve_top),
            trigger_threshold=float(tessera_gee_trigger_threshold),
            base_weight=float(tessera_gee_base_weight),
            dense_weight=float(tessera_gee_dense_weight),
            sparse_weight=float(tessera_gee_sparse_weight),
            probe_weight=float(tessera_gee_probe_weight),
            graph_weight=float(tessera_gee_graph_weight),
            slot_weight=float(tessera_gee_slot_weight),
            sibling_weight=float(tessera_gee_sibling_weight),
            redundancy_weight=float(tessera_gee_redundancy_weight),
        )
        ser_candidate, ser_base_score, gee_expand_trace = expand_graph_evidence_candidates(
            query_text=query,
            current_ranked_idxs=uni_rank_all,
            candidate_idxs=uni_candidate,
            candidate_base_scores=uni_base_score,
            dense_scores=d_row,
            sparse_scores=s_row,
            dense_ranked_idxs=dense_top400.tolist(),
            sparse_ranked_idxs=sparse_top500.tolist(),
            doc_ids=doc_ids,
            doc_id_to_idx=doc_id_to_idx,
            doc_tokens=doc_tokens,
            target_type=target_type,
            config=gee_cfg,
        )
        if debug_trace is not None:
            debug_trace.setdefault("tessera_gee_expand_triggered", []).append(
                float(getattr(gee_expand_trace, "triggered", False))
            )
            debug_trace.setdefault("tessera_gee_expand_input_candidates", []).append(
                float(getattr(gee_expand_trace, "input_candidate_count", 0))
            )
            debug_trace.setdefault("tessera_gee_expand_output_candidates", []).append(
                float(getattr(gee_expand_trace, "output_candidate_count", 0))
            )
            debug_trace.setdefault("tessera_gee_expand_graph_added", []).append(
                float(getattr(gee_expand_trace, "graph_added", 0))
            )
            debug_trace.setdefault("tessera_gee_expand_boosted_existing", []).append(
                float(getattr(gee_expand_trace, "boosted_existing", 0))
            )
            debug_trace.setdefault("tessera_gee_expand_probe_count", []).append(
                float(getattr(gee_expand_trace, "probe_count", 0))
            )
            debug_trace.setdefault("tessera_gee_expand_coverage_need", []).append(
                float(getattr(gee_expand_trace, "coverage_need", 0.0))
            )
            debug_trace.setdefault("tessera_gee_expand_complex_need", []).append(
                float(getattr(gee_expand_trace, "complex_need", 0.0))
            )
            debug_trace.setdefault("tessera_gee_expand_trigger_score", []).append(
                float(getattr(gee_expand_trace, "trigger_score", 0.0))
            )

    if (
        bool(tessera_retrieval_multi_agent)
        and ("tessera_rag" in selected_method_set or "tessera_submod" in selected_method_set)
        and uni_rank_all
    ):
        if rerank_tessera_retrieval is None or TESSERARetrievalAgentConfig is None:
            raise RuntimeError("tessera_policy module is required for --tessera-retrieval-multi-agent")
        retrieval_agent_cfg = TESSERARetrievalAgentConfig(
            candidate_pool_k=int(tessera_retrieval_agent_pool_k),
            dense_pool_k=int(tessera_retrieval_dense_pool_k),
            sparse_pool_k=int(tessera_retrieval_sparse_pool_k),
            preserve_top=int(tessera_retrieval_preserve_top),
            base_weight=float(tessera_retrieval_base_weight),
            dense_weight=float(tessera_retrieval_dense_weight),
            sparse_weight=float(tessera_retrieval_sparse_weight),
            target_weight=float(tessera_retrieval_target_weight),
            coverage_weight=float(tessera_retrieval_coverage_weight),
            diversity_weight=float(tessera_retrieval_diversity_weight),
            dense_rescue_k=int(tessera_retrieval_dense_rescue_k),
            dense_rescue_pool_k=int(tessera_retrieval_dense_rescue_pool_k),
            sibling_seed_k=int(tessera_retrieval_sibling_seed_k),
            sibling_window=int(tessera_retrieval_sibling_window),
            sibling_weight=float(tessera_retrieval_sibling_weight),
        )
        uni_rank_all, retrieval_agent_trace = rerank_tessera_retrieval(
            query=query,
            current_ranked_idxs=uni_rank_all,
            candidate_idxs=uni_candidate,
            candidate_scores=uni_base_score,
            dense_ranked_idxs=dense_top400.tolist(),
            sparse_ranked_idxs=sparse_top500.tolist(),
            doc_ids=doc_ids,
            doc_id_to_idx=doc_id_to_idx,
            doc_texts=doc_texts,
            doc_tokens=doc_tokens,
            doc_numeric_literals=doc_numeric_literals,
            router_prob=blended_router_prob,
            router_entropy=float(router_entropy),
            k=retrieve_topk,
            target_type=target_type,
            upo_concept=upo_concept,
            source_bucket_fn=source_bucket,
            config=retrieval_agent_cfg,
        )
        if debug_trace is not None:
            debug_trace.setdefault("tessera_retrieval_agent_preserve_count", []).append(
                float(retrieval_agent_trace.preserve_count)
            )
            debug_trace.setdefault("tessera_retrieval_agent_dense_added", []).append(
                float(retrieval_agent_trace.dense_added)
            )
            debug_trace.setdefault("tessera_retrieval_agent_sparse_added", []).append(
                float(retrieval_agent_trace.sparse_added)
            )
            debug_trace.setdefault("tessera_retrieval_agent_dense_rescue_added", []).append(
                float(retrieval_agent_trace.dense_rescue_added)
            )
            debug_trace.setdefault("tessera_retrieval_agent_sibling_added", []).append(
                float(retrieval_agent_trace.sibling_added)
            )
            debug_trace.setdefault("tessera_retrieval_agent_forced_hits", []).append(
                float(retrieval_agent_trace.forced_hits)
            )
            debug_trace.setdefault("tessera_retrieval_agent_coverage", []).append(
                float(retrieval_agent_trace.coverage)
            )

    if (
        bool(tessera_retrieval_moe)
        and ("tessera_rag" in selected_method_set or "tessera_submod" in selected_method_set)
        and uni_rank_all
    ):
        if rerank_tessera_moe_retrieval is None or TESSERAMoERetrievalConfig is None:
            raise RuntimeError("tessera_policy module is required for --tessera-retrieval-moe")
        moe_cfg = TESSERAMoERetrievalConfig(
            candidate_pool_k=int(tessera_moe_pool_k),
            prf_seed_k=int(tessera_moe_prf_seed_k),
            prf_dense_seed_k=int(tessera_moe_prf_dense_seed_k),
            prf_sparse_seed_k=int(tessera_moe_prf_sparse_seed_k),
            prf_max_terms=int(tessera_moe_prf_max_terms),
            sibling_seed_k=int(tessera_moe_sibling_seed_k),
            sibling_window=int(tessera_moe_sibling_window),
            sibling_weight=float(tessera_moe_sibling_weight),
        )
        uni_rank_all, moe_trace = rerank_tessera_moe_retrieval(
            query=query,
            current_ranked_idxs=uni_rank_all,
            candidate_idxs=uni_candidate,
            candidate_scores=uni_base_score,
            dense_ranked_idxs=dense_top400.tolist(),
            sparse_ranked_idxs=sparse_top500.tolist(),
            doc_ids=doc_ids,
            doc_id_to_idx=doc_id_to_idx,
            doc_texts=doc_texts,
            doc_tokens=doc_tokens,
            doc_numeric_literals=doc_numeric_literals,
            router_prob=blended_router_prob,
            router_entropy=float(router_entropy),
            k=retrieve_topk,
            target_type=target_type,
            upo_concept=upo_concept,
            source_bucket_fn=source_bucket,
            config=moe_cfg,
        )
        if debug_trace is not None:
            debug_trace.setdefault("tessera_moe_table_like", []).append(float(moe_trace.table_like))
            debug_trace.setdefault("tessera_moe_kg_like", []).append(float(moe_trace.kg_like))
            debug_trace.setdefault("tessera_moe_prf_terms", []).append(float(moe_trace.prf_terms))
            debug_trace.setdefault("tessera_moe_sibling_added", []).append(float(moe_trace.sibling_added))
            debug_trace.setdefault("tessera_moe_coverage", []).append(float(moe_trace.coverage))

    if (
        tessera_ser_bundle is not None
        and ("tessera_rag" in selected_method_set or "tessera_submod" in selected_method_set)
        and uni_rank_all
    ):
        if rerank_with_ser is None or SERRankerConfig is None:
            raise RuntimeError("ser_ranker module is required for --tessera-ser-ranker")
        ser_cfg = SERRankerConfig(
            preserve_top=int(tessera_ser_preserve_top),
            candidate_pool_k=int(tessera_ser_candidate_pool_k),
            dense_pool_k=int(tessera_ser_dense_pool_k),
            sparse_pool_k=int(tessera_ser_sparse_pool_k),
            blend_weight=float(tessera_ser_blend_weight),
            diversity_weight=float(tessera_ser_diversity_weight),
            evidence_rescue_k=int(tessera_ser_evidence_rescue_k),
            evidence_rescue_pool_k=int(tessera_ser_evidence_rescue_pool_k),
            evidence_preserve_top=int(tessera_ser_evidence_preserve_top),
            evidence_redundancy_weight=float(tessera_ser_evidence_redundancy_weight),
            evidence_min_gain=float(tessera_ser_evidence_min_gain),
            plan_adaptive=bool(tessera_ser_plan_adaptive),
            plan_dense_weight=float(tessera_ser_plan_dense_weight),
            plan_sparse_weight=float(tessera_ser_plan_sparse_weight),
            plan_lexical_weight=float(tessera_ser_plan_lexical_weight),
            plan_slot_weight=float(tessera_ser_plan_slot_weight),
            evidence_set_selection=bool(tessera_ser_evidence_set_selection),
            evidence_set_preserve_top=int(tessera_ser_evidence_set_preserve_top),
            evidence_set_pool_k=int(tessera_ser_evidence_set_pool_k),
            evidence_set_cardinality_threshold=float(tessera_ser_evidence_set_cardinality_threshold),
            evidence_set_learned_weight=float(tessera_ser_evidence_set_learned_weight),
            evidence_set_base_weight=float(tessera_ser_evidence_set_base_weight),
            evidence_set_dense_weight=float(tessera_ser_evidence_set_dense_weight),
            evidence_set_sparse_weight=float(tessera_ser_evidence_set_sparse_weight),
            evidence_set_probe_weight=float(tessera_ser_evidence_set_probe_weight),
            evidence_set_slot_weight=float(tessera_ser_evidence_set_slot_weight),
            evidence_set_anchor_weight=float(tessera_ser_evidence_set_anchor_weight),
            evidence_set_family_weight=float(tessera_ser_evidence_set_family_weight),
            evidence_set_redundancy_weight=float(tessera_ser_evidence_set_redundancy_weight),
        )
        old_uni_rank = list(uni_rank_all)
        uni_rank_all, ser_trace = rerank_with_ser(
            query_tokens=q_tokens,
            current_ranked_idxs=uni_rank_all,
            candidate_idxs=ser_candidate,
            candidate_base_scores=ser_base_score,
            dense_scores=d_row,
            sparse_scores=s_row,
            dense_ranked_idxs=dense_top400.tolist(),
            sparse_ranked_idxs=sparse_top500.tolist(),
            doc_ids=doc_ids,
            doc_tokens=doc_tokens,
            doc_numeric_literals=doc_numeric_literals,
            doc_texts=doc_texts,
            router_prob=blended_router_prob,
            target_type=target_type,
            source_bucket_fn=source_bucket,
            k=retrieve_topk,
            bundle=tessera_ser_bundle,
            config=ser_cfg,
            query_text=query,
        )
        if debug_trace is not None:
            debug_trace.setdefault("tessera_ser_candidate_count", []).append(float(ser_trace.candidate_count))
            debug_trace.setdefault("tessera_ser_preserve_count", []).append(float(ser_trace.preserve_count))
            debug_trace.setdefault("tessera_ser_changed_count", []).append(float(ser_trace.changed_count))
            debug_trace.setdefault("tessera_ser_mean_score", []).append(float(ser_trace.mean_score))
            debug_trace.setdefault("tessera_ser_max_score", []).append(float(ser_trace.max_score))
            debug_trace.setdefault("tessera_ser_evidence_rescue_added", []).append(
                float(getattr(ser_trace, "evidence_rescue_added", 0))
            )
            debug_trace.setdefault("tessera_ser_evidence_rescue_pool", []).append(
                float(getattr(ser_trace, "evidence_rescue_pool", 0))
            )
            debug_trace.setdefault("tessera_ser_plan_direct_score", []).append(
                float(getattr(ser_trace, "plan_direct_score", 0.0))
            )
            debug_trace.setdefault("tessera_ser_plan_complex_score", []).append(
                float(getattr(ser_trace, "plan_complex_score", 0.0))
            )
            debug_trace.setdefault("tessera_ser_plan_slot_count", []).append(
                float(getattr(ser_trace, "plan_slot_count", 0))
            )
            debug_trace.setdefault("tessera_ser_evidence_set_enabled", []).append(
                float(getattr(ser_trace, "evidence_set_enabled", False))
            )
            debug_trace.setdefault("tessera_ser_evidence_set_cardinality_need", []).append(
                float(getattr(ser_trace, "evidence_set_cardinality_need", 0.0))
            )
            debug_trace.setdefault("tessera_ser_evidence_set_slot_coverage", []).append(
                float(getattr(ser_trace, "evidence_set_slot_coverage", 0.0))
            )
            debug_trace.setdefault("tessera_ser_evidence_set_family_count", []).append(
                float(getattr(ser_trace, "evidence_set_family_count", 0))
            )
            debug_trace.setdefault("tessera_ser_evidence_set_anchor_hits", []).append(
                float(getattr(ser_trace, "evidence_set_anchor_hits", 0))
            )
            if old_uni_rank:
                overlap = len(set(old_uni_rank[:retrieve_topk]) & set(uni_rank_all[:retrieve_topk])) / max(1, retrieve_topk)
                debug_trace.setdefault("tessera_ser_topk_overlap", []).append(float(overlap))

    v10_pre_local_rank_all = list(uni_rank_all)
    if (
        bool(tessera_v9_enabled)
        and bool(tessera_v9_local_rerank)
        and ("tessera_rag" in selected_method_set or "tessera_submod" in selected_method_set)
        and uni_rank_all
    ):
        if rerank_v9_local_evidence is None:
            raise RuntimeError("tessera_v9 module is required for --tessera-v9-local-rerank")
        if v9_cfg is None and V9CandidateConfig is not None:
            v9_cfg = V9CandidateConfig(
                dense_pool_k=int(tessera_v9_dense_pool_k),
                sparse_pool_k=int(tessera_v9_sparse_pool_k),
                candidate_pool_k=int(tessera_v9_candidate_pool_k),
                graph_seed_k=int(tessera_v9_graph_seed_k),
                graph_window=int(tessera_v9_graph_window),
                preserve_top=int(tessera_v9_preserve_top),
                base_weight=float(tessera_v9_base_weight),
                dense_weight=float(tessera_v9_dense_weight),
                sparse_weight=float(tessera_v9_sparse_weight),
                probe_weight=float(tessera_v9_probe_weight),
                graph_weight=float(tessera_v9_graph_weight),
                slot_weight=float(tessera_v9_slot_weight),
                diversity_weight=float(tessera_v9_diversity_weight),
                modality_weight=float(tessera_v9_modality_weight),
            )
        old_uni_rank = list(uni_rank_all)
        uni_rank_all, v9_rerank_trace = rerank_v9_local_evidence(
            query_text=query,
            current_ranked_idxs=uni_rank_all,
            candidate_idxs=ser_candidate,
            candidate_scores=ser_base_score,
            dense_scores=d_row,
            sparse_scores=s_row,
            doc_ids=doc_ids,
            doc_tokens=doc_tokens,
            target_type=target_type,
            k=retrieve_topk,
            config=v9_cfg,
        )
        if debug_trace is not None:
            debug_trace.setdefault("tessera_v9_local_rerank", []).append(1.0)
            debug_trace.setdefault("tessera_v9_local_changed_count", []).append(
                float(getattr(v9_rerank_trace, "changed_count", 0))
            )
            debug_trace.setdefault("tessera_v9_local_slot_coverage", []).append(
                float(getattr(v9_rerank_trace, "slot_coverage", 0.0))
            )
            if old_uni_rank:
                overlap = len(set(old_uni_rank[:retrieve_topk]) & set(uni_rank_all[:retrieve_topk])) / max(1, retrieve_topk)
                debug_trace.setdefault("tessera_v9_local_topk_overlap", []).append(float(overlap))
    elif debug_trace is not None and bool(tessera_v9_enabled):
        debug_trace.setdefault("tessera_v9_local_rerank", []).append(0.0)

    if (
        bool(tessera_v10_conservative_rerank)
        and not bool(tessera_gee_post_rerank)
        and ("tessera_rag" in selected_method_set or "tessera_submod" in selected_method_set)
        and uni_rank_all
    ):
        if apply_v10_conservative_gate is None or V10RerankConfig is None:
            raise RuntimeError("tessera_v10 module is required for --tessera-v10-conservative-rerank")
        v10_cfg = V10RerankConfig(
            preserve_top=int(tessera_v10_preserve_top),
            direct_preserve_top=int(tessera_v10_direct_preserve_top),
            reference_pool_k=int(tessera_v10_reference_pool_k),
            candidate_pool_k=int(tessera_v10_candidate_pool_k),
            reference_weight=float(tessera_v10_reference_weight),
            current_weight=float(tessera_v10_current_weight),
            base_weight=float(tessera_v10_base_weight),
            dense_weight=float(tessera_v10_dense_weight),
            sparse_weight=float(tessera_v10_sparse_weight),
            probe_weight=float(tessera_v10_probe_weight),
            slot_weight=float(tessera_v10_slot_weight),
            diversity_weight=float(tessera_v10_diversity_weight),
            margin=float(tessera_v10_margin),
            relevance_floor=float(tessera_v10_relevance_floor),
        )
        old_uni_rank = list(uni_rank_all)
        reference_groups = [g for g in [v10_pre_local_rank_all, v10_pre_v9_rank_all] if g]
        uni_rank_all, v10_trace = apply_v10_conservative_gate(
            query_text=query,
            current_ranked_idxs=uni_rank_all,
            reference_ranked_groups=reference_groups,
            candidate_idxs=ser_candidate,
            candidate_scores=ser_base_score,
            dense_scores=d_row,
            sparse_scores=s_row,
            doc_ids=doc_ids,
            doc_tokens=doc_tokens,
            target_type=target_type,
            k=retrieve_topk,
            config=v10_cfg,
        )
        if debug_trace is not None:
            debug_trace.setdefault("tessera_v10_conservative_rerank", []).append(1.0)
            debug_trace.setdefault("tessera_v10_candidate_count", []).append(
                float(getattr(v10_trace, "candidate_count", 0))
            )
            debug_trace.setdefault("tessera_v10_reference_count", []).append(
                float(getattr(v10_trace, "reference_count", 0))
            )
            debug_trace.setdefault("tessera_v10_preserve_count", []).append(
                float(getattr(v10_trace, "preserve_count", 0))
            )
            debug_trace.setdefault("tessera_v10_restored_from_reference", []).append(
                float(getattr(v10_trace, "restored_from_reference", 0))
            )
            debug_trace.setdefault("tessera_v10_accepted_new", []).append(
                float(getattr(v10_trace, "accepted_new", 0))
            )
            debug_trace.setdefault("tessera_v10_rejected_new", []).append(
                float(getattr(v10_trace, "rejected_new", 0))
            )
            debug_trace.setdefault("tessera_v10_changed_count", []).append(
                float(getattr(v10_trace, "changed_count", 0))
            )
            debug_trace.setdefault("tessera_v10_direct_need", []).append(
                float(getattr(v10_trace, "direct_need", 0.0))
            )
            debug_trace.setdefault("tessera_v10_complex_need", []).append(
                float(getattr(v10_trace, "complex_need", 0.0))
            )
            debug_trace.setdefault("tessera_v10_effective_margin", []).append(
                float(getattr(v10_trace, "effective_margin", 0.0))
            )
            debug_trace.setdefault("tessera_v10_effective_relevance_floor", []).append(
                float(getattr(v10_trace, "effective_relevance_floor", 0.0))
            )
            debug_trace.setdefault("tessera_v10_slot_coverage", []).append(
                float(getattr(v10_trace, "slot_coverage", 0.0))
            )
            debug_trace.setdefault("tessera_v10_reference_topk_overlap_before", []).append(
                float(getattr(v10_trace, "reference_topk_overlap_before", 0.0))
            )
            debug_trace.setdefault("tessera_v10_reference_topk_overlap_after", []).append(
                float(getattr(v10_trace, "reference_topk_overlap_after", 0.0))
            )
            if old_uni_rank:
                overlap = len(set(old_uni_rank[:retrieve_topk]) & set(uni_rank_all[:retrieve_topk])) / max(1, retrieve_topk)
                debug_trace.setdefault("tessera_v10_topk_overlap", []).append(float(overlap))
    elif debug_trace is not None and not bool(tessera_gee_post_rerank):
        debug_trace.setdefault("tessera_v10_conservative_rerank", []).append(0.0)

    v10_pre_final_rank_all = list(uni_rank_all)
    if (
        bool(tessera_graph_evidence_expansion)
        and bool(tessera_gee_post_rerank)
        and ("tessera_rag" in selected_method_set or "tessera_submod" in selected_method_set)
        and uni_rank_all
    ):
        if expand_and_rerank_graph_evidence is None or GraphEvidenceConfig is None:
            raise RuntimeError("graph_evidence module is required for --tessera-graph-evidence-expansion")
        gee_cfg = GraphEvidenceConfig(
            candidate_pool_k=int(tessera_gee_candidate_pool_k),
            dense_pool_k=int(tessera_gee_dense_pool_k),
            sparse_pool_k=int(tessera_gee_sparse_pool_k),
            graph_seed_k=int(tessera_gee_graph_seed_k),
            graph_window=int(tessera_gee_graph_window),
            preserve_top=int(tessera_gee_preserve_top),
            trigger_threshold=float(tessera_gee_trigger_threshold),
            base_weight=float(tessera_gee_base_weight),
            dense_weight=float(tessera_gee_dense_weight),
            sparse_weight=float(tessera_gee_sparse_weight),
            probe_weight=float(tessera_gee_probe_weight),
            graph_weight=float(tessera_gee_graph_weight),
            slot_weight=float(tessera_gee_slot_weight),
            sibling_weight=float(tessera_gee_sibling_weight),
            redundancy_weight=float(tessera_gee_redundancy_weight),
        )
        old_uni_rank = list(uni_rank_all)
        uni_rank_all, gee_trace = expand_and_rerank_graph_evidence(
            query_text=query,
            current_ranked_idxs=uni_rank_all,
            candidate_idxs=ser_candidate,
            candidate_base_scores=ser_base_score,
            dense_scores=d_row,
            sparse_scores=s_row,
            dense_ranked_idxs=dense_top400.tolist(),
            sparse_ranked_idxs=sparse_top500.tolist(),
            doc_ids=doc_ids,
            doc_id_to_idx=doc_id_to_idx,
            doc_tokens=doc_tokens,
            target_type=target_type,
            source_bucket_fn=source_bucket,
            k=retrieve_topk,
            config=gee_cfg,
        )
        if debug_trace is not None:
            debug_trace.setdefault("tessera_gee_triggered", []).append(float(getattr(gee_trace, "triggered", False)))
            debug_trace.setdefault("tessera_gee_pool_size", []).append(float(getattr(gee_trace, "pool_size", 0)))
            debug_trace.setdefault("tessera_gee_graph_added", []).append(float(getattr(gee_trace, "graph_added", 0)))
            debug_trace.setdefault("tessera_gee_probe_count", []).append(float(getattr(gee_trace, "probe_count", 0)))
            debug_trace.setdefault("tessera_gee_coverage_need", []).append(float(getattr(gee_trace, "coverage_need", 0.0)))
            debug_trace.setdefault("tessera_gee_complex_need", []).append(float(getattr(gee_trace, "complex_need", 0.0)))
            debug_trace.setdefault("tessera_gee_changed_count", []).append(float(getattr(gee_trace, "changed_count", 0)))
            if old_uni_rank:
                overlap = len(set(old_uni_rank[:retrieve_topk]) & set(uni_rank_all[:retrieve_topk])) / max(1, retrieve_topk)
                debug_trace.setdefault("tessera_gee_topk_overlap", []).append(float(overlap))

    if (
        bool(tessera_v10_conservative_rerank)
        and bool(tessera_gee_post_rerank)
        and ("tessera_rag" in selected_method_set or "tessera_submod" in selected_method_set)
        and uni_rank_all
    ):
        if apply_v10_conservative_gate is None or V10RerankConfig is None:
            raise RuntimeError("tessera_v10 module is required for --tessera-v10-conservative-rerank")
        v10_cfg = V10RerankConfig(
            preserve_top=int(tessera_v10_preserve_top),
            direct_preserve_top=int(tessera_v10_direct_preserve_top),
            reference_pool_k=int(tessera_v10_reference_pool_k),
            candidate_pool_k=int(tessera_v10_candidate_pool_k),
            reference_weight=float(tessera_v10_reference_weight),
            current_weight=float(tessera_v10_current_weight),
            base_weight=float(tessera_v10_base_weight),
            dense_weight=float(tessera_v10_dense_weight),
            sparse_weight=float(tessera_v10_sparse_weight),
            probe_weight=float(tessera_v10_probe_weight),
            slot_weight=float(tessera_v10_slot_weight),
            diversity_weight=float(tessera_v10_diversity_weight),
            margin=float(tessera_v10_margin),
            relevance_floor=float(tessera_v10_relevance_floor),
        )
        old_uni_rank = list(uni_rank_all)
        reference_groups = [g for g in [v10_pre_final_rank_all, v10_pre_local_rank_all, v10_pre_v9_rank_all] if g]
        uni_rank_all, v10_trace = apply_v10_conservative_gate(
            query_text=query,
            current_ranked_idxs=uni_rank_all,
            reference_ranked_groups=reference_groups,
            candidate_idxs=ser_candidate,
            candidate_scores=ser_base_score,
            dense_scores=d_row,
            sparse_scores=s_row,
            doc_ids=doc_ids,
            doc_tokens=doc_tokens,
            target_type=target_type,
            k=retrieve_topk,
            config=v10_cfg,
        )
        if debug_trace is not None:
            debug_trace.setdefault("tessera_v10_conservative_rerank", []).append(1.0)
            debug_trace.setdefault("tessera_v10_candidate_count", []).append(
                float(getattr(v10_trace, "candidate_count", 0))
            )
            debug_trace.setdefault("tessera_v10_reference_count", []).append(
                float(getattr(v10_trace, "reference_count", 0))
            )
            debug_trace.setdefault("tessera_v10_preserve_count", []).append(
                float(getattr(v10_trace, "preserve_count", 0))
            )
            debug_trace.setdefault("tessera_v10_restored_from_reference", []).append(
                float(getattr(v10_trace, "restored_from_reference", 0))
            )
            debug_trace.setdefault("tessera_v10_accepted_new", []).append(
                float(getattr(v10_trace, "accepted_new", 0))
            )
            debug_trace.setdefault("tessera_v10_rejected_new", []).append(
                float(getattr(v10_trace, "rejected_new", 0))
            )
            debug_trace.setdefault("tessera_v10_changed_count", []).append(
                float(getattr(v10_trace, "changed_count", 0))
            )
            debug_trace.setdefault("tessera_v10_direct_need", []).append(
                float(getattr(v10_trace, "direct_need", 0.0))
            )
            debug_trace.setdefault("tessera_v10_complex_need", []).append(
                float(getattr(v10_trace, "complex_need", 0.0))
            )
            debug_trace.setdefault("tessera_v10_effective_margin", []).append(
                float(getattr(v10_trace, "effective_margin", 0.0))
            )
            debug_trace.setdefault("tessera_v10_effective_relevance_floor", []).append(
                float(getattr(v10_trace, "effective_relevance_floor", 0.0))
            )
            debug_trace.setdefault("tessera_v10_slot_coverage", []).append(
                float(getattr(v10_trace, "slot_coverage", 0.0))
            )
            debug_trace.setdefault("tessera_v10_reference_topk_overlap_before", []).append(
                float(getattr(v10_trace, "reference_topk_overlap_before", 0.0))
            )
            debug_trace.setdefault("tessera_v10_reference_topk_overlap_after", []).append(
                float(getattr(v10_trace, "reference_topk_overlap_after", 0.0))
            )
            if old_uni_rank:
                overlap = len(set(old_uni_rank[:retrieve_topk]) & set(uni_rank_all[:retrieve_topk])) / max(1, retrieve_topk)
                debug_trace.setdefault("tessera_v10_topk_overlap", []).append(float(overlap))
    elif debug_trace is not None and bool(tessera_gee_post_rerank):
        debug_trace.setdefault("tessera_v10_conservative_rerank", []).append(0.0)

    if (
        bool(tessera_source_evidence_fusion)
        and ("tessera_rag" in selected_method_set or "tessera_submod" in selected_method_set)
        and uni_rank_all
    ):
        if apply_source_evidence_fusion is None or SourceEvidenceFusionConfig is None:
            raise RuntimeError("source_evidence_fusion module is required for --tessera-source-evidence-fusion")
        source_evidence_cfg = SourceEvidenceFusionConfig(
            topk=int(tessera_source_evidence_topk),
            candidate_pool_k=int(tessera_source_evidence_candidate_pool_k),
            preserve_top=int(tessera_source_evidence_preserve_top),
            base_weight=float(tessera_source_evidence_base_weight),
            dense_weight=float(tessera_source_evidence_dense_weight),
            sparse_weight=float(tessera_source_evidence_sparse_weight),
            reference_weight=float(tessera_source_evidence_reference_weight),
            lexical_weight=float(tessera_source_evidence_lexical_weight),
            modality_prior_weight=float(tessera_source_evidence_modality_prior_weight),
            source_balance_weight=float(tessera_source_evidence_source_balance_weight),
            target_family_weight=float(tessera_source_evidence_target_family_weight),
            diversity_weight=float(tessera_source_evidence_diversity_weight),
            replacement_margin=float(tessera_source_evidence_replacement_margin),
            min_candidate_score=float(tessera_source_evidence_min_candidate_score),
            dense_guard=bool(tessera_source_evidence_dense_guard),
            dense_guard_topn=int(tessera_source_evidence_dense_guard_topn),
            dense_guard_prefixes=str(tessera_source_evidence_dense_guard_prefixes),
            dense_guard_weight=float(tessera_source_evidence_dense_guard_weight),
            dense_rank_weight=float(tessera_source_evidence_dense_rank_weight),
            current_rank_weight=float(tessera_source_evidence_current_rank_weight),
            source_balance_prefixes=str(tessera_source_evidence_source_balance_prefixes),
            max_changed_slots=int(tessera_source_evidence_max_changed_slots),
            slot_acceptance_guard=bool(tessera_source_evidence_slot_acceptance_guard),
            slot_acceptance_prefixes=str(tessera_source_evidence_slot_acceptance_prefixes),
            slot_acceptance_margin=float(tessera_source_evidence_slot_acceptance_margin),
            budget_composer=bool(tessera_source_evidence_budget_composer),
            budget_prefixes=str(tessera_source_evidence_budget_prefixes),
            budget_candidate_pool_k=int(tessera_source_evidence_budget_candidate_pool_k),
            budget_start_slot=int(tessera_source_evidence_budget_start_slot),
            budget_max_selected=int(tessera_source_evidence_budget_max_selected),
            budget_score_weight=float(tessera_source_evidence_budget_score_weight),
            budget_sibling_weight=float(tessera_source_evidence_budget_sibling_weight),
            budget_source_quota_weight=float(tessera_source_evidence_budget_source_quota_weight),
            budget_tail_rank_weight=float(tessera_source_evidence_budget_tail_rank_weight),
            budget_reference_weight=float(tessera_source_evidence_budget_reference_weight),
            budget_margin=float(tessera_source_evidence_budget_margin),
            budget_redundancy_weight=float(tessera_source_evidence_budget_redundancy_weight),
            sibling_filler=bool(tessera_source_evidence_sibling_filler),
            sibling_filler_prefixes=str(tessera_source_evidence_sibling_filler_prefixes),
            sibling_filler_candidate_pool_k=int(tessera_source_evidence_sibling_filler_candidate_pool_k),
            sibling_filler_start_slot=int(tessera_source_evidence_sibling_filler_start_slot),
            sibling_filler_max_selected=int(tessera_source_evidence_sibling_filler_max_selected),
            sibling_filler_tail_topn=int(tessera_source_evidence_sibling_filler_tail_topn),
            sibling_filler_reference_topn=int(tessera_source_evidence_sibling_filler_reference_topn),
            sibling_filler_margin=float(tessera_source_evidence_sibling_filler_margin),
            sibling_filler_sibling_weight=float(tessera_source_evidence_sibling_filler_sibling_weight),
            sibling_filler_reference_weight=float(tessera_source_evidence_sibling_filler_reference_weight),
            sibling_filler_tail_weight=float(tessera_source_evidence_sibling_filler_tail_weight),
            sibling_filler_dense_weight=float(tessera_source_evidence_sibling_filler_dense_weight),
            sibling_filler_source_weight=float(tessera_source_evidence_sibling_filler_source_weight),
            sibling_filler_redundancy_weight=float(tessera_source_evidence_sibling_filler_redundancy_weight),
            slot_verifier=bool(tessera_source_evidence_slot_verifier),
            slot_verifier_prefixes=str(tessera_source_evidence_slot_verifier_prefixes),
            slot_verifier_candidate_pool_k=int(tessera_source_evidence_slot_verifier_candidate_pool_k),
            slot_verifier_start_slot=int(tessera_source_evidence_slot_verifier_start_slot),
            slot_verifier_max_selected=int(tessera_source_evidence_slot_verifier_max_selected),
            slot_verifier_tail_topn=int(tessera_source_evidence_slot_verifier_tail_topn),
            slot_verifier_reference_topn=int(tessera_source_evidence_slot_verifier_reference_topn),
            slot_verifier_dense_topn=int(tessera_source_evidence_slot_verifier_dense_topn),
            slot_verifier_margin=float(tessera_source_evidence_slot_verifier_margin),
            slot_verifier_min_score=float(tessera_source_evidence_slot_verifier_min_score),
            slot_verifier_model_threshold=float(tessera_source_evidence_slot_verifier_model_threshold),
            slot_verifier_static_weight=float(tessera_source_evidence_slot_verifier_static_weight),
            slot_verifier_reference_weight=float(tessera_source_evidence_slot_verifier_reference_weight),
            slot_verifier_dense_weight=float(tessera_source_evidence_slot_verifier_dense_weight),
            slot_verifier_tail_weight=float(tessera_source_evidence_slot_verifier_tail_weight),
            slot_verifier_sibling_weight=float(tessera_source_evidence_slot_verifier_sibling_weight),
            slot_verifier_source_weight=float(tessera_source_evidence_slot_verifier_source_weight),
            slot_verifier_lexical_weight=float(tessera_source_evidence_slot_verifier_lexical_weight),
            slot_verifier_family_weight=float(tessera_source_evidence_slot_verifier_family_weight),
            slot_verifier_redundancy_weight=float(tessera_source_evidence_slot_verifier_redundancy_weight),
            kg_preservation_guard=bool(tessera_source_evidence_kg_preservation_guard),
            kg_preservation_prefixes=str(tessera_source_evidence_kg_preservation_prefixes),
            kg_preservation_min_kg=int(tessera_source_evidence_kg_preservation_min_kg),
            kg_preservation_candidate_pool_k=int(tessera_source_evidence_kg_preservation_candidate_pool_k),
            kg_preservation_start_slot=int(tessera_source_evidence_kg_preservation_start_slot),
            kg_preservation_margin=float(tessera_source_evidence_kg_preservation_margin),
            kg_preservation_reference_weight=float(tessera_source_evidence_kg_preservation_reference_weight),
            kg_preservation_dense_weight=float(tessera_source_evidence_kg_preservation_dense_weight),
            kg_preservation_current_weight=float(tessera_source_evidence_kg_preservation_current_weight),
            kg_preservation_family_weight=float(tessera_source_evidence_kg_preservation_family_weight),
            kg_preservation_lexical_weight=float(tessera_source_evidence_kg_preservation_lexical_weight),
            kg_preservation_verifier_weight=float(tessera_source_evidence_kg_verifier_weight),
            kg_preservation_verifier_min_score=float(tessera_source_evidence_kg_verifier_min_score),
            kg_preservation_verify_existing=bool(tessera_source_evidence_kg_verify_existing),
            kg_preservation_verify_existing_max_replacements=int(tessera_source_evidence_kg_verify_existing_max_replacements),
            source_budget_gate=bool(tessera_source_budgeter_bundle is not None),
            source_budget_need_threshold=float(tessera_source_budgeter_need_threshold),
            source_budget_non_kg_top1_max_kg=int(tessera_source_budgeter_non_kg_top1_max_kg),
        )
        old_uni_rank = list(uni_rank_all)
        reference_groups = [g for g in [v10_pre_final_rank_all, v10_pre_local_rank_all, v10_pre_v9_rank_all] if g]
        uni_rank_all, source_evidence_trace = apply_source_evidence_fusion(
            query_id=query_id,
            query_text=query,
            current_ranked_idxs=uni_rank_all,
            reference_ranked_groups=reference_groups,
            candidate_idxs=ser_candidate,
            candidate_scores=ser_base_score,
            dense_scores=d_row,
            sparse_scores=s_row,
            dense_ranked_idxs=dense_top400.tolist(),
            router_prob=blended_router_prob,
            doc_ids=doc_ids,
            doc_tokens=doc_tokens,
            doc_texts=doc_texts,
            source_bucket_fn=source_bucket,
            config=source_evidence_cfg,
            slot_verifier_bundle=tessera_source_evidence_slot_verifier_bundle,
            kg_verifier_bundle=tessera_source_evidence_kg_verifier_bundle,
            source_budget=source_budget,
        )
        if debug_trace is not None:
            debug_trace.setdefault("tessera_source_evidence_fusion", []).append(1.0)
            debug_trace.setdefault("tessera_source_evidence_changed_count", []).append(
                float(getattr(source_evidence_trace, "changed_count", 0))
            )
            debug_trace.setdefault("tessera_source_evidence_candidate_count", []).append(
                float(getattr(source_evidence_trace, "candidate_count", 0))
            )
            debug_trace.setdefault("tessera_source_evidence_preserved_count", []).append(
                float(getattr(source_evidence_trace, "preserved_count", 0))
            )
            debug_trace.setdefault("tessera_source_evidence_rescued_from_below_topk", []).append(
                float(getattr(source_evidence_trace, "rescued_from_below_topk", 0))
            )
            debug_trace.setdefault("tessera_source_evidence_source_count_before", []).append(
                float(getattr(source_evidence_trace, "source_count_before", 0))
            )
            debug_trace.setdefault("tessera_source_evidence_source_count_after", []).append(
                float(getattr(source_evidence_trace, "source_count_after", 0))
            )
            debug_trace.setdefault("tessera_source_evidence_text_count_before", []).append(
                float(getattr(source_evidence_trace, "text_count_before", 0))
            )
            debug_trace.setdefault("tessera_source_evidence_table_count_before", []).append(
                float(getattr(source_evidence_trace, "table_count_before", 0))
            )
            debug_trace.setdefault("tessera_source_evidence_kg_count_before", []).append(
                float(getattr(source_evidence_trace, "kg_count_before", 0))
            )
            debug_trace.setdefault("tessera_source_evidence_text_count_after", []).append(
                float(getattr(source_evidence_trace, "text_count_after", 0))
            )
            debug_trace.setdefault("tessera_source_evidence_table_count_after", []).append(
                float(getattr(source_evidence_trace, "table_count_after", 0))
            )
            debug_trace.setdefault("tessera_source_evidence_kg_count_after", []).append(
                float(getattr(source_evidence_trace, "kg_count_after", 0))
            )
            debug_trace.setdefault("tessera_source_evidence_topk_overlap", []).append(
                float(getattr(source_evidence_trace, "topk_overlap", 0.0))
            )
            debug_trace.setdefault("tessera_source_evidence_mean_selected_score", []).append(
                float(getattr(source_evidence_trace, "mean_selected_score", 0.0))
            )
            debug_trace.setdefault("tessera_source_evidence_dense_guard_active", []).append(
                float(getattr(source_evidence_trace, "dense_guard_active", 0))
            )
            debug_trace.setdefault("tessera_source_evidence_source_balance_active", []).append(
                float(getattr(source_evidence_trace, "source_balance_active", 0))
            )
            debug_trace.setdefault("tessera_source_evidence_dense_guard_candidates", []).append(
                float(getattr(source_evidence_trace, "dense_guard_candidates", 0))
            )
            debug_trace.setdefault("tessera_source_evidence_dense_guard_selected", []).append(
                float(getattr(source_evidence_trace, "dense_guard_selected", 0))
            )
            debug_trace.setdefault("tessera_source_evidence_max_changed_slots", []).append(
                float(getattr(source_evidence_trace, "max_changed_slots", 0))
            )
            debug_trace.setdefault("tessera_source_evidence_slot_acceptance_active", []).append(
                float(getattr(source_evidence_trace, "slot_acceptance_active", 0))
            )
            debug_trace.setdefault("tessera_source_evidence_slot_acceptance_rejected", []).append(
                float(getattr(source_evidence_trace, "slot_acceptance_rejected", 0))
            )
            debug_trace.setdefault("tessera_source_evidence_slot_acceptance_selected", []).append(
                float(getattr(source_evidence_trace, "slot_acceptance_selected", 0))
            )
            debug_trace.setdefault("tessera_source_evidence_budget_composer_active", []).append(
                float(getattr(source_evidence_trace, "budget_composer_active", 0))
            )
            debug_trace.setdefault("tessera_source_evidence_budget_candidate_count", []).append(
                float(getattr(source_evidence_trace, "budget_candidate_count", 0))
            )
            debug_trace.setdefault("tessera_source_evidence_budget_changed_count", []).append(
                float(getattr(source_evidence_trace, "budget_changed_count", 0))
            )
            debug_trace.setdefault("tessera_source_evidence_budget_tail_selected", []).append(
                float(getattr(source_evidence_trace, "budget_tail_selected", 0))
            )
            debug_trace.setdefault("tessera_source_evidence_budget_sibling_selected", []).append(
                float(getattr(source_evidence_trace, "budget_sibling_selected", 0))
            )
            debug_trace.setdefault("tessera_source_evidence_budget_source_quota_selected", []).append(
                float(getattr(source_evidence_trace, "budget_source_quota_selected", 0))
            )
            debug_trace.setdefault("tessera_source_evidence_budget_reference_selected", []).append(
                float(getattr(source_evidence_trace, "budget_reference_selected", 0))
            )
            debug_trace.setdefault("tessera_source_evidence_sibling_filler_active", []).append(
                float(getattr(source_evidence_trace, "sibling_filler_active", 0))
            )
            debug_trace.setdefault("tessera_source_evidence_sibling_filler_candidate_count", []).append(
                float(getattr(source_evidence_trace, "sibling_filler_candidate_count", 0))
            )
            debug_trace.setdefault("tessera_source_evidence_sibling_filler_changed_count", []).append(
                float(getattr(source_evidence_trace, "sibling_filler_changed_count", 0))
            )
            debug_trace.setdefault("tessera_source_evidence_sibling_filler_tail_selected", []).append(
                float(getattr(source_evidence_trace, "sibling_filler_tail_selected", 0))
            )
            debug_trace.setdefault("tessera_source_evidence_sibling_filler_sibling_selected", []).append(
                float(getattr(source_evidence_trace, "sibling_filler_sibling_selected", 0))
            )
            debug_trace.setdefault("tessera_source_evidence_sibling_filler_reference_selected", []).append(
                float(getattr(source_evidence_trace, "sibling_filler_reference_selected", 0))
            )
            debug_trace.setdefault("tessera_source_evidence_sibling_filler_dense_selected", []).append(
                float(getattr(source_evidence_trace, "sibling_filler_dense_selected", 0))
            )
            debug_trace.setdefault("tessera_source_evidence_sibling_filler_rejected", []).append(
                float(getattr(source_evidence_trace, "sibling_filler_rejected", 0))
            )
            debug_trace.setdefault("tessera_source_evidence_slot_verifier_active", []).append(
                float(getattr(source_evidence_trace, "slot_verifier_active", 0))
            )
            debug_trace.setdefault("tessera_source_evidence_slot_verifier_candidate_count", []).append(
                float(getattr(source_evidence_trace, "slot_verifier_candidate_count", 0))
            )
            debug_trace.setdefault("tessera_source_evidence_slot_verifier_changed_count", []).append(
                float(getattr(source_evidence_trace, "slot_verifier_changed_count", 0))
            )
            debug_trace.setdefault("tessera_source_evidence_slot_verifier_accepted", []).append(
                float(getattr(source_evidence_trace, "slot_verifier_accepted", 0))
            )
            debug_trace.setdefault("tessera_source_evidence_slot_verifier_rejected", []).append(
                float(getattr(source_evidence_trace, "slot_verifier_rejected", 0))
            )
            debug_trace.setdefault("tessera_source_evidence_slot_verifier_tail_selected", []).append(
                float(getattr(source_evidence_trace, "slot_verifier_tail_selected", 0))
            )
            debug_trace.setdefault("tessera_source_evidence_slot_verifier_sibling_selected", []).append(
                float(getattr(source_evidence_trace, "slot_verifier_sibling_selected", 0))
            )
            debug_trace.setdefault("tessera_source_evidence_slot_verifier_reference_selected", []).append(
                float(getattr(source_evidence_trace, "slot_verifier_reference_selected", 0))
            )
            debug_trace.setdefault("tessera_source_evidence_slot_verifier_dense_selected", []).append(
                float(getattr(source_evidence_trace, "slot_verifier_dense_selected", 0))
            )
            debug_trace.setdefault("tessera_source_evidence_slot_verifier_source_selected", []).append(
                float(getattr(source_evidence_trace, "slot_verifier_source_selected", 0))
            )
            debug_trace.setdefault("tessera_source_evidence_slot_verifier_model_active", []).append(
                float(getattr(source_evidence_trace, "slot_verifier_model_active", 0))
            )
            debug_trace.setdefault("tessera_source_evidence_kg_guard_active", []).append(
                float(getattr(source_evidence_trace, "kg_guard_active", 0))
            )
            debug_trace.setdefault("tessera_source_evidence_kg_guard_candidate_count", []).append(
                float(getattr(source_evidence_trace, "kg_guard_candidate_count", 0))
            )
            debug_trace.setdefault("tessera_source_evidence_kg_guard_recovered", []).append(
                float(getattr(source_evidence_trace, "kg_guard_recovered", 0))
            )
            debug_trace.setdefault("tessera_source_evidence_kg_guard_rejected", []).append(
                float(getattr(source_evidence_trace, "kg_guard_rejected", 0))
            )
            debug_trace.setdefault("tessera_source_evidence_kg_guard_verifier_active", []).append(
                float(getattr(source_evidence_trace, "kg_guard_verifier_active", 0))
            )
            debug_trace.setdefault("tessera_source_evidence_kg_guard_verifier_mean_score", []).append(
                float(getattr(source_evidence_trace, "kg_guard_verifier_mean_score", 0.0))
            )
            debug_trace.setdefault("tessera_source_evidence_kg_guard_verifier_rejected", []).append(
                float(getattr(source_evidence_trace, "kg_guard_verifier_rejected", 0))
            )
            debug_trace.setdefault("tessera_source_evidence_source_budget_gate_active", []).append(
                float(getattr(source_evidence_trace, "source_budget_gate_active", 0))
            )
            debug_trace.setdefault("tessera_source_evidence_kg_guard_effective_min_kg", []).append(
                float(getattr(source_evidence_trace, "kg_guard_effective_min_kg", 0))
            )
            if old_uni_rank:
                overlap = len(set(old_uni_rank[:retrieve_topk]) & set(uni_rank_all[:retrieve_topk])) / max(1, retrieve_topk)
                debug_trace.setdefault("tessera_source_evidence_full_topk_overlap", []).append(float(overlap))
    elif debug_trace is not None:
        debug_trace.setdefault("tessera_source_evidence_fusion", []).append(0.0)

    if (
        bool(tessera_source_head_selector)
        and ("tessera_rag" in selected_method_set or "tessera_submod" in selected_method_set)
        and uni_rank_all
    ):
        if apply_source_aware_head_selector is None or SourceHeadSelectorConfig is None:
            raise RuntimeError("source_head_selector module is required for --tessera-source-head-selector")
        head_cfg = SourceHeadSelectorConfig(
            topn=int(tessera_source_head_topn),
            source_weight=float(tessera_source_head_source_weight),
            same_query_weight=float(tessera_source_head_same_query_weight),
            position_weight=float(tessera_source_head_position_weight),
            reference_weight=float(tessera_source_head_reference_weight),
            lexical_weight=float(tessera_source_head_lexical_weight),
            base_weight=float(tessera_source_head_base_weight),
            dense_weight=float(tessera_source_head_dense_weight),
            sparse_weight=float(tessera_source_head_sparse_weight),
            margin=float(tessera_source_head_margin),
            off_source_margin=float(tessera_source_head_off_source_margin),
        )
        old_uni_rank = list(uni_rank_all)
        reference_groups = [g for g in [v10_pre_final_rank_all, v10_pre_local_rank_all, v10_pre_v9_rank_all] if g]
        uni_rank_all, head_trace = apply_source_aware_head_selector(
            query_id=query_id,
            query_text=query,
            current_ranked_idxs=uni_rank_all,
            reference_ranked_groups=reference_groups,
            candidate_idxs=ser_candidate,
            candidate_scores=ser_base_score,
            dense_scores=d_row,
            sparse_scores=s_row,
            doc_ids=doc_ids,
            doc_tokens=doc_tokens,
            config=head_cfg,
            allowed_top1_sources=source_budget_allowed_top1,
        )
        if debug_trace is not None:
            debug_trace.setdefault("tessera_source_head_selector", []).append(1.0)
            debug_trace.setdefault("tessera_source_head_attempted", []).append(
                float(getattr(head_trace, "attempted", 0))
            )
            debug_trace.setdefault("tessera_source_head_changed", []).append(
                float(getattr(head_trace, "changed", 0))
            )
            debug_trace.setdefault("tessera_source_head_aligned_before", []).append(
                float(getattr(head_trace, "top_family_aligned_before", 0))
            )
            debug_trace.setdefault("tessera_source_head_aligned_after", []).append(
                float(getattr(head_trace, "top_family_aligned_after", 0))
            )
            debug_trace.setdefault("tessera_source_head_source_candidate_count", []).append(
                float(getattr(head_trace, "source_candidate_count", 0))
            )
            debug_trace.setdefault("tessera_source_head_selected_rank", []).append(
                float(getattr(head_trace, "selected_rank", 0))
            )
            debug_trace.setdefault("tessera_source_head_selected_score", []).append(
                float(getattr(head_trace, "selected_score", 0.0))
            )
            debug_trace.setdefault("tessera_source_head_old_score", []).append(
                float(getattr(head_trace, "old_score", 0.0))
            )
            if old_uni_rank:
                debug_trace.setdefault("tessera_source_head_top1_changed", []).append(
                    float(bool(uni_rank_all and old_uni_rank[0] != uni_rank_all[0]))
                )
    elif debug_trace is not None:
        debug_trace.setdefault("tessera_source_head_selector", []).append(0.0)

    if (
        tessera_source_action_policy_bundle is not None
        and ("tessera_rag" in selected_method_set or "tessera_submod" in selected_method_set)
        and uni_rank_all
    ):
        if apply_source_action_to_ranked_idxs is None:
            raise RuntimeError("source_action_policy module is required for --tessera-source-action-policy-model")
        old_uni_rank = list(uni_rank_all)
        ranked_doc_ids_for_policy = [str(doc_ids[j]) for j in old_uni_rank if 0 <= int(j) < len(doc_ids)]
        action_pred = tessera_source_action_policy_bundle.predict(
            query_text=query,
            query_id=query_id,
            ranked_doc_ids=ranked_doc_ids_for_policy,
            trace=debug_trace,
        )
        action_labels = list(getattr(tessera_source_action_policy_bundle, "action_labels", SOURCE_ACTION_LABELS) or [])
        action = str(getattr(action_pred, "action", "keep_current"))
        confidence = float(getattr(action_pred, "confidence", 0.0))
        should_apply = action != "keep_current" and confidence >= float(tessera_source_action_policy_min_prob)
        if should_apply:
            uni_rank_all = apply_source_action_to_ranked_idxs(
                action,
                uni_rank_all,
                doc_ids,
                topk=int(tessera_source_action_policy_topk),
                pool_k=int(tessera_source_action_policy_pool_k),
            )
        if debug_trace is not None:
            overlap = len(set(old_uni_rank[:retrieve_topk]) & set(uni_rank_all[:retrieve_topk])) / max(1, retrieve_topk)
            debug_trace.setdefault("tessera_source_action_policy_active", []).append(1.0)
            debug_trace.setdefault("tessera_source_action_policy_applied", []).append(float(should_apply))
            debug_trace.setdefault("tessera_source_action_policy_confidence", []).append(confidence)
            debug_trace.setdefault("tessera_source_action_policy_topk_overlap", []).append(float(overlap))
            debug_trace.setdefault("tessera_source_action_policy_action_index", []).append(
                float(action_labels.index(action) if action in action_labels else -1)
            )
    elif debug_trace is not None:
        debug_trace.setdefault("tessera_source_action_policy_active", []).append(0.0)

    if (
        bool(tessera_final_evidence_composer)
        and ("tessera_rag" in selected_method_set or "tessera_submod" in selected_method_set)
        and uni_rank_all
    ):
        if compose_final_with_ser is None or FinalEvidenceComposerConfig is None:
            raise RuntimeError("ser_ranker module is required for --tessera-final-evidence-composer")
        if tessera_ser_bundle is None:
            raise RuntimeError("--tessera-final-evidence-composer requires --tessera-ser-ranker")
        final_cfg = FinalEvidenceComposerConfig(
            topk=int(tessera_final_evidence_topk),
            preserve_top=int(tessera_final_evidence_preserve_top),
            candidate_pool_k=int(tessera_final_evidence_candidate_pool_k),
            dense_pool_k=int(tessera_final_evidence_dense_pool_k),
            sparse_pool_k=int(tessera_final_evidence_sparse_pool_k),
            max_replacements=int(tessera_final_evidence_max_replacements),
            min_candidate_score=float(tessera_final_evidence_min_candidate_score),
            replacement_margin=float(tessera_final_evidence_replacement_margin),
            min_query_overlap=float(tessera_final_evidence_min_query_overlap),
            source_need_weight=float(tessera_final_evidence_source_need_weight),
            redundancy_weight=float(tessera_final_evidence_redundancy_weight),
            replacement_verifier_threshold=float(tessera_final_evidence_verifier_threshold),
            replacement_verifier_margin=float(tessera_final_evidence_verifier_margin),
        )
        old_uni_rank = list(uni_rank_all)
        uni_rank_all, final_trace = compose_final_with_ser(
            query_id=query_id,
            query_text=query,
            query_tokens=q_tokens,
            current_ranked_idxs=uni_rank_all,
            candidate_idxs=ser_candidate,
            candidate_base_scores=ser_base_score,
            dense_scores=d_row,
            sparse_scores=s_row,
            dense_ranked_idxs=dense_top400.tolist(),
            sparse_ranked_idxs=sparse_top500.tolist(),
            doc_ids=doc_ids,
            doc_tokens=doc_tokens,
            doc_numeric_literals=doc_numeric_literals,
            router_prob=blended_router_prob,
            target_type=target_type,
            source_bucket_fn=source_bucket,
            bundle=tessera_ser_bundle,
            config=final_cfg,
            source_budget=source_budget,
            doc_texts=doc_texts,
            replacement_verifier_bundle=tessera_final_evidence_verifier_bundle,
        )
        if debug_trace is not None:
            topk = max(1, int(tessera_final_evidence_topk))
            overlap = len(set(old_uni_rank[:topk]) & set(uni_rank_all[:topk])) / max(1, topk)
            debug_trace.setdefault("tessera_final_evidence_composer", []).append(1.0)
            debug_trace.setdefault("tessera_final_evidence_candidate_count", []).append(
                float(getattr(final_trace, "candidate_count", 0))
            )
            debug_trace.setdefault("tessera_final_evidence_preserve_count", []).append(
                float(getattr(final_trace, "preserve_count", 0))
            )
            debug_trace.setdefault("tessera_final_evidence_changed_count", []).append(
                float(getattr(final_trace, "changed_count", 0))
            )
            debug_trace.setdefault("tessera_final_evidence_replacement_count", []).append(
                float(getattr(final_trace, "replacement_count", 0))
            )
            debug_trace.setdefault("tessera_final_evidence_rejected_count", []).append(
                float(getattr(final_trace, "rejected_count", 0))
            )
            debug_trace.setdefault("tessera_final_evidence_mean_candidate_score", []).append(
                float(getattr(final_trace, "mean_candidate_score", 0.0))
            )
            debug_trace.setdefault("tessera_final_evidence_max_candidate_score", []).append(
                float(getattr(final_trace, "max_candidate_score", 0.0))
            )
            debug_trace.setdefault("tessera_final_evidence_min_replaced_score", []).append(
                float(getattr(final_trace, "min_replaced_score", 0.0))
            )
            debug_trace.setdefault("tessera_final_evidence_max_inserted_score", []).append(
                float(getattr(final_trace, "max_inserted_score", 0.0))
            )
            debug_trace.setdefault("tessera_final_evidence_topk_overlap", []).append(float(overlap))
            debug_trace.setdefault("tessera_final_evidence_verifier_active", []).append(
                float(getattr(final_trace, "verifier_active", 0))
            )
            debug_trace.setdefault("tessera_final_evidence_verifier_scored", []).append(
                float(getattr(final_trace, "verifier_scored", 0))
            )
            debug_trace.setdefault("tessera_final_evidence_verifier_accepted", []).append(
                float(getattr(final_trace, "verifier_accepted", 0))
            )
            debug_trace.setdefault("tessera_final_evidence_verifier_rejected", []).append(
                float(getattr(final_trace, "verifier_rejected", 0))
            )
            debug_trace.setdefault("tessera_final_evidence_verifier_max_score", []).append(
                float(getattr(final_trace, "verifier_max_score", 0.0))
            )
    elif debug_trace is not None:
        debug_trace.setdefault("tessera_final_evidence_composer", []).append(0.0)

    uni_nopath_base_no_conflict = (
        w_dense * uni_d_norm
        + w_sparse * uni_s_norm
        + late_alpha_eff * late_bonus
        + modality_bonus_weight * modality_bonus
        + gated_table_cell_weight * table_cell_scale * table_cell_arr
        + cross_modal_bonus_weight * cross_bonus_arr
        + token_maxsim_weight * token_maxsim_arr
        + qa_objective_weight_eff * qa_objective_arr
        + upo_retrieval_weight_eff * upo_modality_arr
        + heavy_table_weight_eff * heavy_table_arr
        + heavy_token_late_weight_eff * heavy_token_late_arr
        + heavy_token_cross_weight_eff * heavy_token_cross_arr
    )
    uni_nopath_base = uni_nopath_base_no_conflict - retrieval_conflict_weight_eff * retrieval_conflict_pen_arr
    uni_nopath_rank_all = greedy_diverse_topk(
        uni_candidate,
        uni_nopath_base,
        doc_tokens,
        k=retrieve_topk,
        redundancy_lambda=0.03,
        redundancy_mode="union",
        candidate_pool_k=int(tessera_candidate_pool_k) if int(tessera_candidate_pool_k) > 0 else max(retrieve_topk * 4, 80),
    )

    preserve_k = min(max(0, int(preserve_dense_top)), retrieve_topk)
    preserve = dense_top400[:preserve_k].tolist()

    def apply_preserve(rank_list: list[int]) -> list[int]:
        seen_local = set(preserve)
        out = preserve[:]
        for j in rank_list:
            if j in seen_local:
                continue
            out.append(j)
            seen_local.add(j)
            if len(out) >= retrieve_topk:
                break
        return out[:retrieve_topk]

    if "tessera_rag" in selected_method_set:
        rankings["tessera_rag"] = apply_preserve(uni_rank_all)
    if "tessera_submod" in selected_method_set:
        rankings["tessera_submod"] = apply_preserve(uni_rank_all)
    if "ablation_no_redundancy_e2e" in selected_method_set:
        rankings["ablation_no_redundancy_e2e"] = apply_preserve(uni_nored_rank_all)
    if "ablation_no_pathmaxsim_e2e" in selected_method_set:
        rankings["ablation_no_pathmaxsim_e2e"] = apply_preserve(uni_nopath_rank_all)

    if debug_trace is not None:
        debug_trace.setdefault("ranking_tessera_ms", []).append((time.perf_counter() - t_profile2) * 1000.0)

    return rankings

def select_modality_adaptive_context(
    ranked_idxs: list[int],
    dense_ranked_idxs: list[int],
    query: str,
    doc_ids: list[str],
    doc_texts: list[str],
    doc_tokens: list[set[str]],
    doc_signal_tokens: list[set[str]],
    doc_numeric_literals: list[set[str]],
    router_prob: np.ndarray,
    router_entropy: float,
    k: int,
    active_threshold: float,
    anchor_dense_k: int,
    anchor_uni_k: int,
    context_candidate_expand_k: int,
    redundancy_lambda: float,
    enable_hard_redundancy_filter: bool,
    consistency_weight: float,
    context_conflict_penalty_weight: float,
    context_conflict_targeted_only: bool,
    context_conflict_table_kg_only: bool,
    context_conflict_risk_gating: bool,
    context_conflict_risk_low: float,
    context_conflict_risk_high: float,
    context_conflict_risk_probe_k: int,
    context_conflict_sensitive_target_scale: float,
    context_conflict_max_literals_per_doc: int,
    conflict_bundle: ConflictBundle | None,
    context_number_table_quota_min: int,
    context_subquery_coverage_weight: float,
    context_light_rerank_weight: float,
    context_light_rerank_topn: int,
    context_light_rerank_targeted_only: bool,
    context_strong_rerank_endpoint: str,
    context_strong_rerank_topn: int,
    context_strong_rerank_timeout: int,
    context_upo_lite_quota_mix: float,
    context_upo_lite_rerank_bonus: float,
    context_qa_objective_weight: float,
    context_qa_modality_bias: float,
    debug_trace: dict[str, list[float]] | None = None,
) -> list[int]:
    if k <= 0 or not ranked_idxs:
        return []
    ranked_idxs = [int(j) for j in ranked_idxs]
    dense_ranked_idxs = [int(j) for j in dense_ranked_idxs]
    q_tokens = tokenize(query)
    query_segments = decompose_query_segments(query)
    covered_segments: set[int] = set()
    text_p = float(router_prob[ROUTER_LABEL_TO_IDX["text"]])
    table_p = float(router_prob[ROUTER_LABEL_TO_IDX["table"]])
    kg_p = float(router_prob[ROUTER_LABEL_TO_IDX["kg"]])

    # CAMPE: confidence-adaptive multi-perspective evidence packing.
    # High uncertainty -> rely more on dense anchors, avoid forced all-modality coverage.
    probs = {"text": text_p, "table": table_p, "kg": kg_p}
    qa_target: str | None = None
    upo_concept = str(infer_upo_lite_concept(query)).strip().lower()
    upo_quota_mix = float(np.clip(context_upo_lite_quota_mix, 0.0, 1.0))
    if upo_quota_mix > 1e-9:
        upo_prior = infer_upo_lite_modality_prior(query)
        arr = np.asarray([probs["text"], probs["table"], probs["kg"]], dtype=np.float32)
        arr = (1.0 - upo_quota_mix) * arr + upo_quota_mix * upo_prior
        arr = np.clip(arr, 1e-6, None)
        arr = arr / max(1e-6, float(arr.sum()))
        probs = {"text": float(arr[0]), "table": float(arr[1]), "kg": float(arr[2])}
    active = [b for b, p in probs.items() if p >= active_threshold]
    if not active:
        active = [max(probs, key=probs.get)]
    consistency_weight_local = float(consistency_weight) if len(active) >= 2 else 0.0
    qa_obj_weight_local = max(0.0, float(context_qa_objective_weight))
    if qa_obj_weight_local > 0.0:
        # Under high uncertainty, slightly increase QA-target evidence bias in context packing.
        qa_obj_weight_local *= 0.85 + 0.30 * min(1.0, max(0.0, float(router_entropy)))
        qa_obj_weight_local = min(0.35, qa_obj_weight_local)
        qa_target = infer_qa_target_type(query)
        # Keep this bonus only where lexical answerability cues are relatively reliable.
        if qa_target not in {"number", "year", "boolean"}:
            qa_obj_weight_local = 0.0

    qa_bias_strength = max(0.0, float(context_qa_modality_bias))
    if qa_bias_strength > 0.0:
        if qa_target is None:
            qa_target = infer_qa_target_type(query)
        bias = np.zeros(3, dtype=np.float32)  # text, table, kg
        if qa_target == "number":
            bias += np.asarray([0.03, 0.16, -0.10], dtype=np.float32)
        elif qa_target == "year":
            bias += np.asarray([0.08, 0.08, -0.06], dtype=np.float32)
        elif qa_target == "boolean":
            bias += np.asarray([0.10, 0.02, -0.08], dtype=np.float32)
        if float(np.max(np.abs(bias))) > 1e-9:
            arr = np.asarray([probs["text"], probs["table"], probs["kg"]], dtype=np.float32)
            arr = np.clip(arr + float(qa_bias_strength) * bias, 1e-6, None)
            arr = arr / max(1e-6, float(arr.sum()))
            probs = {"text": float(arr[0]), "table": float(arr[1]), "kg": float(arr[2])}

    light_rerank_applied = 0.0
    light_rerank_shift = 0.0
    light_w = min(1.0, max(0.0, float(context_light_rerank_weight)))
    light_topn = max(1, int(context_light_rerank_topn))
    if light_w > 0.0 and light_topn > 1:
        if qa_target is None:
            qa_target = infer_qa_target_type(query)
        apply_light_rerank = True
        if bool(context_light_rerank_targeted_only) and qa_target not in {"number", "year", "location"} and upo_concept != "relation":
            apply_light_rerank = False
        if apply_light_rerank:
            before = ranked_idxs[: min(len(ranked_idxs), light_topn)]
            ranked_idxs = lightweight_context_rerank(
                ranked_idxs=ranked_idxs,
                query_tokens=q_tokens,
                query_segments=query_segments,
                qa_target=str(qa_target),
                upo_concept=upo_concept,
                upo_rerank_bonus=float(context_upo_lite_rerank_bonus),
                doc_ids=doc_ids,
                doc_tokens=doc_tokens,
                doc_signal_tokens=doc_signal_tokens,
                doc_numeric_literals=doc_numeric_literals,
                probs=probs,
                topn=light_topn,
                weight=light_w,
            )
            after_pos = {int(j): p for p, j in enumerate(ranked_idxs[: len(before)])}
            if before:
                moved = [abs(p - int(after_pos.get(int(j), p))) for p, j in enumerate(before)]
                light_rerank_shift = float(np.mean(moved)) if moved else 0.0
            light_rerank_applied = 1.0

    strong_rerank_applied = 0.0
    strong_rerank_shift = 0.0
    strong_endpoint = str(context_strong_rerank_endpoint).strip()
    strong_topn = max(1, int(context_strong_rerank_topn))
    if strong_endpoint and strong_topn > 1:
        before = ranked_idxs[: min(len(ranked_idxs), strong_topn)]
        ranked_idxs = external_context_rerank(
            ranked_idxs=ranked_idxs,
            query=query,
            doc_texts=doc_texts,
            topn=strong_topn,
            endpoint=strong_endpoint,
            timeout=int(context_strong_rerank_timeout),
        )
        after_pos = {int(j): p for p, j in enumerate(ranked_idxs[: len(before)])}
        if before:
            moved = [abs(p - int(after_pos.get(int(j), p))) for p, j in enumerate(before)]
            strong_rerank_shift = float(np.mean(moved)) if moved else 0.0
        strong_rerank_applied = 1.0

    conflict_penalty_weight_local = max(0.0, float(context_conflict_penalty_weight))
    if conflict_penalty_weight_local > 0.0:
        if qa_target is None:
            qa_target = infer_qa_target_type(query)
        # Integrate CAMPE-C into mainline conservatively:
        # keep conflict suppression focused on numeric/year questions,
        # where cross-modal literal contradiction is most actionable.
        if bool(context_conflict_targeted_only) and qa_target not in {"number", "year"}:
            conflict_penalty_weight_local = 0.0
        # CRAG-style corrective policy: number/location queries are sensitive
        # to aggressive conflict suppression, so we weaken the penalty there.
        if qa_target in {"number", "location"}:
            conflict_penalty_weight_local *= float(context_conflict_sensitive_target_scale)
        conflict_penalty_weight_local *= 0.80 + 0.40 * min(1.0, max(0.0, float(router_entropy)))
        conflict_penalty_weight_local = min(0.35, conflict_penalty_weight_local)
    else:
        conflict_penalty_weight_local = 0.0

    active_mass = sum(probs[b] for b in active)
    quotas = {"text": 0, "table": 0, "kg": 0}
    for b in active:
        quotas[b] = max(1, int(round(k * (probs[b] / max(1e-9, active_mass)))))

    # Query-intent quota shaping: numeric questions are often table-heavy.
    # A minimum table quota can reduce number-query misses caused by under-exposure.
    min_table_quota = max(0, int(context_number_table_quota_min))
    if min_table_quota > 0:
        if qa_target is None:
            qa_target = infer_qa_target_type(query)
        if qa_target == "number":
            quotas["table"] = max(quotas["table"], min(k, min_table_quota))

    while sum(quotas.values()) > k:
        order = sorted(quotas.keys(), key=lambda x: quotas[x], reverse=True)
        b = order[0]
        if qa_target == "number" and min_table_quota > 0 and b == "table" and len(order) > 1:
            # Prefer shrinking non-table buckets first for numeric queries.
            for cand_b in order[1:]:
                if quotas[cand_b] > 1:
                    b = cand_b
                    break
        if quotas[b] > 1:
            quotas[b] -= 1
        else:
            break

    selected: list[int] = []
    selected_set: set[int] = set()

    def try_add(j: int) -> bool:
        if j in selected_set:
            return False
        tok = doc_tokens[j]
        max_red = 0.0
        for x in selected:
            red = jaccard_overlap(tok, doc_tokens[x])
            if red > max_red:
                max_red = red
        if enable_hard_redundancy_filter and max_red >= 0.9:
            return False
        selected.append(j)
        selected_set.add(j)
        return True

    dense_anchor = max(1, int(anchor_dense_k)) if router_entropy >= 0.7 else int(anchor_dense_k)
    for j in dense_ranked_idxs[: max(0, dense_anchor)]:
        if len(selected) >= k:
            break
        try_add(j)

    for j in ranked_idxs[: max(0, int(anchor_uni_k))]:
        if len(selected) >= k:
            break
        try_add(j)

    used_quota = {"text": 0, "table": 0, "kg": 0}
    for j in selected:
        used_quota[source_bucket(doc_ids[j])] += 1
    if query_segments:
        for sid, seg in enumerate(query_segments):
            for j in selected:
                ov = len(seg & doc_tokens[j]) / max(1, len(seg))
                if ov >= 0.45:
                    covered_segments.add(sid)
                    break

    expand_k = max(0, int(context_candidate_expand_k))
    primary_k = max(len(ranked_idxs), int(k) * 4)
    rank_pos_uni: dict[int, int] = {}
    rank_pos_dense: dict[int, int] = {}
    for p, j in enumerate(ranked_idxs[:primary_k]):
        rank_pos_uni[int(j)] = p
    if expand_k > 0:
        for p, j in enumerate(dense_ranked_idxs[: max(expand_k, int(k) * 2)]):
            rank_pos_dense[int(j)] = p

    candidate_order: list[int] = []
    candidate_seen: set[int] = set()

    def push_candidates(seq: list[int], limit: int) -> None:
        for j in seq[: max(0, int(limit))]:
            jj = int(j)
            if jj in candidate_seen:
                continue
            candidate_seen.add(jj)
            candidate_order.append(jj)

    push_candidates(ranked_idxs, primary_k)
    if expand_k > 0:
        push_candidates(dense_ranked_idxs, expand_k)

    conflict_risk = 0.0
    conflict_risk_scale = 1.0
    if conflict_penalty_weight_local > 0.0 and bool(context_conflict_risk_gating):
        conflict_risk = estimate_query_conflict_risk(
            query=query,
            candidate_idxs=candidate_order,
            query_tokens=q_tokens,
            doc_ids=doc_ids,
            doc_texts=doc_texts,
            doc_tokens=doc_tokens,
            doc_signal_tokens=doc_signal_tokens,
            doc_numeric_literals=doc_numeric_literals,
            conflict_bundle=conflict_bundle,
            table_kg_only=bool(context_conflict_table_kg_only),
            probe_k=int(context_conflict_risk_probe_k),
            max_literals_per_doc=int(context_conflict_max_literals_per_doc),
        )
        low = max(0.0, min(1.0, float(context_conflict_risk_low)))
        high = max(low + 1e-6, min(1.0, float(context_conflict_risk_high)))
        if conflict_risk <= low:
            conflict_risk_scale = 0.0
        elif conflict_risk >= high:
            conflict_risk_scale = 1.0
        else:
            conflict_risk_scale = (conflict_risk - low) / max(1e-6, high - low)
        conflict_penalty_weight_local *= float(conflict_risk_scale)

    candidate_score: list[tuple[int, float]] = []
    conflict_eval = 0
    conflict_trigger = 0
    conflict_pen_sum = 0.0
    subq_weight = max(0.0, float(context_subquery_coverage_weight))
    for j in candidate_order:
        if j in selected_set:
            continue
        bucket = source_bucket(doc_ids[j])
        q_ov = len(q_tokens & doc_tokens[j]) / max(1, len(q_tokens)) if q_tokens else 0.0
        uni_prior = 1.0 / float(rank_pos_uni[j] + 1) if j in rank_pos_uni else 0.0
        if expand_k > 0:
            dense_prior = 1.0 / float(rank_pos_dense[j] + 1) if j in rank_pos_dense else 0.0
            rank_signal = 0.62 * uni_prior + 0.38 * dense_prior
            rank_w, overlap_w, mod_w_coef = 0.34, 0.28, 0.16
        else:
            rank_signal = uni_prior
            rank_w, overlap_w, mod_w_coef = 0.38, 0.30, 0.18
        mod_w = probs[bucket]
        consistency = cross_modal_consistency_score(j, selected, doc_ids, doc_signal_tokens)
        qa_obj = 0.0
        if qa_obj_weight_local > 0.0:
            qa_obj = qa_objective_retrieval_score(
                query=query,
                query_tokens=q_tokens,
                doc_text=doc_texts[j],
                doc_tokens=doc_tokens[j],
                bucket=bucket,
            )
        conflict_pen = 0.0
        if conflict_penalty_weight_local > 0.0:
            conflict_pen = cross_modal_conflict_penalty(
                cand_idx=j,
                selected_idxs=selected,
                doc_ids=doc_ids,
                query_tokens=q_tokens,
                doc_tokens=doc_tokens,
                doc_signal_tokens=doc_signal_tokens,
                doc_numeric_literals=doc_numeric_literals,
                table_kg_only=bool(context_conflict_table_kg_only),
                max_literals_per_doc=int(context_conflict_max_literals_per_doc),
            )
            conflict_eval += 1
            conflict_pen_sum += float(conflict_pen)
            if conflict_pen > 1e-9:
                conflict_trigger += 1
        subq_bonus = 0.0
        if subq_weight > 0.0 and query_segments:
            newly = 0
            for sid, seg in enumerate(query_segments):
                if sid in covered_segments:
                    continue
                ov = len(seg & doc_tokens[j]) / max(1, len(seg))
                if ov >= 0.45:
                    newly += 1
            if newly > 0:
                subq_bonus = subq_weight * (float(newly) / float(len(query_segments)))
        sc = (
            rank_w * rank_signal
            + overlap_w * q_ov
            + mod_w_coef * mod_w
            + consistency_weight_local * consistency
            + qa_obj_weight_local * qa_obj
            + subq_bonus
            - conflict_penalty_weight_local * conflict_pen
        )
        candidate_score.append((j, sc))
    candidate_score.sort(key=lambda x: x[1], reverse=True)

    for j, base_sc in candidate_score:
        if len(selected) >= k:
            break
        bucket = source_bucket(doc_ids[j])
        if bucket in quotas and used_quota[bucket] >= quotas[bucket]:
            continue
        tok = doc_tokens[j]
        max_red = 0.0
        for x in selected:
            red = jaccard_overlap(tok, doc_tokens[x])
            if red > max_red:
                max_red = red
        sc = base_sc - float(redundancy_lambda) * max_red
        # Avoid over-pruning under conflict-aware scoring in LLM settings.
        # A soft floor keeps some diversity while still preferring high-score candidates.
        score_floor = -0.15 if conflict_penalty_weight_local > 0.0 else -1e-6
        if sc < score_floor:
            continue
        if try_add(j):
            used_quota[bucket] += 1
            if subq_weight > 0.0 and query_segments:
                for sid, seg in enumerate(query_segments):
                    if sid in covered_segments:
                        continue
                    ov = len(seg & doc_tokens[j]) / max(1, len(seg))
                    if ov >= 0.45:
                        covered_segments.add(sid)

    if len(selected) < k:
        fallback = list(ranked_idxs)
        if expand_k > 0:
            for j in dense_ranked_idxs:
                if j in fallback:
                    continue
                fallback.append(j)
        for j in fallback:
            if j in selected_set:
                continue
            selected.append(j)
            selected_set.add(j)
            if len(selected) >= k:
                break

    if debug_trace is not None:
        upo_concept_to_id = {
            "number": 0.0,
            "year": 1.0,
            "location": 2.0,
            "relation": 3.0,
            "entity": 4.0,
            "open": 5.0,
        }
        debug_trace.setdefault("context_candidate_pool_size", []).append(float(len(candidate_score)))
        debug_trace.setdefault("context_selected_size", []).append(float(len(selected[:k])))
        debug_trace.setdefault("context_light_rerank_applied", []).append(float(light_rerank_applied))
        debug_trace.setdefault("context_light_rerank_shift", []).append(float(light_rerank_shift))
        debug_trace.setdefault("context_strong_rerank_applied", []).append(float(strong_rerank_applied))
        debug_trace.setdefault("context_strong_rerank_shift", []).append(float(strong_rerank_shift))
        debug_trace.setdefault("context_upo_lite_quota_mix", []).append(float(upo_quota_mix))
        debug_trace.setdefault("context_upo_lite_concept_id", []).append(float(upo_concept_to_id.get(upo_concept, 5.0)))
        debug_trace.setdefault("context_conflict_penalty_weight_eff", []).append(float(conflict_penalty_weight_local))
        debug_trace.setdefault("context_conflict_risk", []).append(float(conflict_risk))
        debug_trace.setdefault("context_conflict_risk_scale", []).append(float(conflict_risk_scale))
        if query_segments:
            debug_trace.setdefault("context_subquery_coverage", []).append(float(len(covered_segments) / max(1, len(query_segments))))
        else:
            debug_trace.setdefault("context_subquery_coverage", []).append(0.0)
        if conflict_eval > 0:
            debug_trace.setdefault("context_conflict_penalty_trigger_rate", []).append(float(conflict_trigger / max(1, conflict_eval)))
            debug_trace.setdefault("context_conflict_penalty_mean", []).append(float(conflict_pen_sum / max(1, conflict_eval)))
        else:
            debug_trace.setdefault("context_conflict_penalty_trigger_rate", []).append(0.0)
            debug_trace.setdefault("context_conflict_penalty_mean", []).append(0.0)
    return selected[:k]


def get_router_acc(router_metrics_path: Path) -> float | None:
    if not router_metrics_path.exists():
        return None
    try:
        d = json.loads(router_metrics_path.read_text(encoding="utf-8"))
    except Exception:
        return None
    x = d.get("test_subset_acc", None)
    if x is None:
        return None
    try:
        return float(x)
    except Exception:
        return None


def load_prediction_jsonl(path: Path) -> list[str]:
    out: list[str] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            out.append(str(row.get("prediction", row.get("pred", row.get("answer", "")))))
    return out


def load_context_docs_jsonl(path: Path) -> list[list[str]]:
    out: list[list[str]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            ids = row.get("context_doc_ids", [])
            if isinstance(ids, list):
                out.append([str(x) for x in ids])
            else:
                out.append([])
    return out


def append_prediction_jsonl(path: Path, row: dict, pred: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps({"id": row.get("id"), "prediction": pred}, ensure_ascii=False) + "\n")


def append_context_docs_jsonl(path: Path, row: dict, ctx_doc_ids: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps({"id": row.get("id"), "context_doc_ids": ctx_doc_ids}, ensure_ascii=False) + "\n")


def resolve_router_model_path(router_model: Path | None, router_metrics: Path) -> Path | None:
    if router_model is not None and router_model.exists():
        return router_model
    base = router_metrics.parent if router_metrics is not None else None
    if base is not None and base.exists():
        cand = base / "router_deberta_full_model"
        if cand.exists():
            return cand
        alt = base / "router_deberta_smoke_model"
        if alt.exists():
            return alt
    return None


def resolve_intent_complexity_tier(level: int) -> str:
    lv = int(level)
    if lv <= 2:
        return "simple"
    if lv == 3:
        return "medium"
    return "complex"


def resolve_budget_by_tier(
    tier: str,
    simple_value: int,
    medium_value: int,
    complex_value: int,
    minimum: int,
) -> int:
    if tier == "simple":
        value = int(simple_value)
    elif tier == "medium":
        value = int(medium_value)
    else:
        value = int(complex_value)
    return max(int(minimum), value)


def format_duration(seconds: float) -> str:
    total = max(0, int(round(float(seconds))))
    hours, rem = divmod(total, 3600)
    minutes, secs = divmod(rem, 60)
    if hours > 0:
        return f"{hours}h{minutes:02d}m{secs:02d}s"
    if minutes > 0:
        return f"{minutes}m{secs:02d}s"
    return f"{secs}s"


def main() -> int:
    parser = argparse.ArgumentParser(description="Run end-to-end QA benchmarks for Table 1c")
    parser.add_argument("--model-dir", type=str, required=True)
    parser.add_argument("--split-file", type=Path, required=True)
    parser.add_argument("--corpus-file", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, default=Path("artifacts/results/table1c_e2e_20260327"))
    parser.add_argument("--cache-dir", type=Path, default=Path("artifacts/retrieval"))
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--max-queries", type=int, default=1286)
    parser.add_argument(
        "--progress-every",
        type=int,
        default=10,
        help="Print query-loop progress every N queries. 0 disables count-based progress.",
    )
    parser.add_argument(
        "--progress-min-seconds",
        type=float,
        default=60.0,
        help="Best-effort heartbeat progress interval (seconds), checked between queries. 0 disables time-based progress.",
    )
    parser.add_argument("--retrieve-topk", type=int, default=20)
    parser.add_argument("--qa-context-k", type=int, default=6)
    parser.add_argument("--preserve-dense-top", type=int, default=0)
    parser.add_argument("--tessera-late-alpha", type=float, default=0.08)
    parser.add_argument("--reader", choices=["extractive", "ollama", "openai"], default="extractive")
    parser.add_argument("--ollama-host", type=str, default="http://127.0.0.1:11434")
    parser.add_argument("--ollama-model", type=str, default="qwen3.6:27b")
    parser.add_argument("--openai-model", type=str, default="gpt-4o-mini")
    parser.add_argument("--openai-api-key-env", type=str, default="OPENAI_API_KEY")
    parser.add_argument("--openai-base-url", type=str, default="")
    parser.add_argument("--openai-timeout", type=int, default=120)
    parser.add_argument("--openai-temperature", type=float, default=0.0)
    parser.add_argument("--openai-max-tokens", type=int, default=64)
    parser.add_argument("--openai-max-retries", type=int, default=3)
    parser.add_argument("--openai-retry-backoff", type=float, default=2.0)
    parser.add_argument(
        "--openai-fail-soft",
        action="store_true",
        help="Return an empty answer after repeated OpenAI reader failures instead of aborting the full run.",
    )
    parser.add_argument(
        "--router-metrics",
        type=Path,
        default=ROOT / "runs/router_deberta_full_v1_router_deberta/router_metrics/router_deberta_full_metrics.json",
    )
    parser.add_argument(
        "--router-model",
        type=Path,
        default=None,
        help="Router model dir for per-query routing inference. If omitted, auto-resolve from --router-metrics dir.",
    )
    parser.add_argument("--router-threshold", type=float, default=0.5)
    parser.add_argument("--router-batch-size", type=int, default=64)
    parser.add_argument(
        "--planner-model",
        type=Path,
        default=None,
        help="Optional learned evidence planner bundle dir or pickle file.",
    )
    parser.add_argument(
        "--planner-mix",
        type=float,
        default=0.35,
        help="Blend strength for the learned planner prior when mixing with router probabilities.",
    )
    parser.add_argument(
        "--verifier-model",
        type=Path,
        default=None,
        help="Optional learned evidence verifier bundle dir or pickle file.",
    )
    parser.add_argument(
        "--conflict-model",
        type=Path,
        default=None,
        help="Optional learned conflict scorer bundle dir or pickle file.",
    )
    parser.add_argument(
        "--query-modality-prior-mix",
        type=float,
        default=0.35,
        help="Blend weight for query-derived modality prior vs router probability (0=router only, 1=query prior only).",
    )
    parser.add_argument(
        "--query-modality-prior-adaptive",
        action="store_true",
        help="Enable adaptive prior mixing based on router uncertainty and router-prior disagreement.",
    )
    parser.add_argument(
        "--query-modality-prior-entropy-scale",
        type=float,
        default=0.30,
        help="Extra prior-mix scale contributed by router entropy above uncertainty threshold.",
    )
    parser.add_argument(
        "--query-modality-prior-disagreement-scale",
        type=float,
        default=0.25,
        help="Extra prior-mix scale contributed by router-vs-prior disagreement.",
    )
    parser.add_argument(
        "--query-modality-prior-min",
        type=float,
        default=0.0,
        help="Lower clip for adaptive query-modality prior mix.",
    )
    parser.add_argument(
        "--query-modality-prior-max",
        type=float,
        default=0.85,
        help="Upper clip for adaptive query-modality prior mix.",
    )
    parser.add_argument(
        "--upo-lite-retrieval-weight",
        type=float,
        default=0.0,
        help="Weight of UPO-Lite concept-modality prior signal in TESSERA retrieval ranking. 0 disables this signal.",
    )
    parser.add_argument(
        "--upo-lite-targeted-only",
        dest="upo_lite_targeted_only",
        action="store_true",
        help="Apply UPO-Lite retrieval prior only on {number, year, location, relation} concepts.",
    )
    parser.add_argument(
        "--upo-lite-global",
        dest="upo_lite_targeted_only",
        action="store_false",
        help="Apply UPO-Lite retrieval prior on all concept types.",
    )
    parser.set_defaults(upo_lite_targeted_only=True)
    parser.add_argument(
        "--retrieval-conflict-penalty-weight",
        type=float,
        default=0.0,
        help="Penalty weight for cross-modal numeric conflicts during TESSERA retrieval ranking. 0 disables it.",
    )
    parser.add_argument(
        "--retrieval-conflict-targeted-only",
        dest="retrieval_conflict_targeted_only",
        action="store_true",
        help="Apply retrieval conflict penalty only for number/year/location queries.",
    )
    parser.add_argument(
        "--retrieval-conflict-global",
        dest="retrieval_conflict_targeted_only",
        action="store_false",
        help="Apply retrieval conflict penalty to all query types.",
    )
    parser.set_defaults(retrieval_conflict_targeted_only=True)
    parser.add_argument(
        "--retrieval-conflict-table-kg-only",
        dest="retrieval_conflict_table_kg_only",
        action="store_true",
        help="Apply retrieval conflict penalty only on table<->kg cross-modal pairs.",
    )
    parser.add_argument(
        "--retrieval-conflict-all-cross-modal",
        dest="retrieval_conflict_table_kg_only",
        action="store_false",
        help="Apply retrieval conflict penalty on all cross-modal pairs (default).",
    )
    parser.set_defaults(retrieval_conflict_table_kg_only=False)
    parser.add_argument(
        "--retrieval-conflict-risk-gating",
        action="store_true",
        help="Enable query-level conflict-risk gating to adaptively scale retrieval conflict penalty weight.",
    )
    parser.add_argument(
        "--retrieval-conflict-no-risk-gating",
        dest="retrieval_conflict_risk_gating",
        action="store_false",
        help="Disable retrieval conflict-risk gating.",
    )
    parser.set_defaults(retrieval_conflict_risk_gating=False)
    parser.add_argument(
        "--retrieval-conflict-risk-low",
        type=float,
        default=0.06,
        help="Low-risk threshold for retrieval conflict-risk gating.",
    )
    parser.add_argument(
        "--retrieval-conflict-risk-high",
        type=float,
        default=0.22,
        help="High-risk threshold for retrieval conflict-risk gating.",
    )
    parser.add_argument(
        "--retrieval-conflict-risk-probe-k",
        type=int,
        default=12,
        help="Number of top candidates used to estimate retrieval conflict risk.",
    )
    parser.add_argument(
        "--retrieval-conflict-anchor-k",
        type=int,
        default=12,
        help="Number of top TESSERA candidates used as anchors for retrieval conflict penalty.",
    )
    parser.add_argument(
        "--retrieval-conflict-max-literals-per-doc",
        type=int,
        default=0,
        help="If >0, skip retrieval conflict checks for docs with more than this many numeric literals.",
    )
    parser.add_argument(
        "--retrieval-conflict-sensitive-target-scale",
        type=float,
        default=1.0,
        help="Extra scale on retrieval conflict penalty for sensitive targets {number, location}; <1.0 weakens penalty.",
    )
    parser.add_argument("--routing-uncertainty-threshold", type=float, default=0.75)
    parser.add_argument("--context-active-threshold", type=float, default=0.40)
    parser.add_argument("--context-anchor-dense-k", type=int, default=4)
    parser.add_argument("--context-anchor-uni-k", type=int, default=1)
    parser.add_argument("--context-dense-pool-k", type=int, default=20)
    parser.add_argument(
        "--intent-complexity-aware-budgeting",
        action="store_true",
        help="Enable IADR-lite: query-intent complexity aware dynamic context budgets.",
    )
    parser.add_argument(
        "--intent-complexity-context-k-simple",
        type=int,
        default=5,
        help="Context size for simple queries (complexity levels 1-2).",
    )
    parser.add_argument(
        "--intent-complexity-context-k-medium",
        type=int,
        default=6,
        help="Context size for medium queries (complexity level 3).",
    )
    parser.add_argument(
        "--intent-complexity-context-k-complex",
        type=int,
        default=8,
        help="Context size for complex queries (complexity levels 4-5).",
    )
    parser.add_argument(
        "--intent-complexity-dense-pool-k-simple",
        type=int,
        default=16,
        help="Dense context pool size for simple queries.",
    )
    parser.add_argument(
        "--intent-complexity-dense-pool-k-medium",
        type=int,
        default=20,
        help="Dense context pool size for medium queries.",
    )
    parser.add_argument(
        "--intent-complexity-dense-pool-k-complex",
        type=int,
        default=28,
        help="Dense context pool size for complex queries.",
    )
    parser.add_argument(
        "--intent-complexity-candidate-expand-k-simple",
        type=int,
        default=0,
        help="Extra dense candidates for simple queries.",
    )
    parser.add_argument(
        "--intent-complexity-candidate-expand-k-medium",
        type=int,
        default=2,
        help="Extra dense candidates for medium queries.",
    )
    parser.add_argument(
        "--intent-complexity-candidate-expand-k-complex",
        type=int,
        default=6,
        help="Extra dense candidates for complex queries.",
    )
    parser.add_argument(
        "--context-number-table-quota-min",
        type=int,
        default=0,
        help="Minimum table quota during CAMPE context selection for number queries. 0 disables this constraint.",
    )
    parser.add_argument(
        "--context-subquery-coverage-weight",
        type=float,
        default=0.0,
        help="Weight of subquery-coverage bonus during CAMPE context selection. 0 disables this term.",
    )
    parser.add_argument(
        "--context-light-rerank-weight",
        type=float,
        default=0.0,
        help="Weight of lightweight rerank signal on top-N retrieval candidates before context packing. 0 disables rerank.",
    )
    parser.add_argument(
        "--context-light-rerank-topn",
        type=int,
        default=20,
        help="Number of top retrieval candidates used by lightweight reranker.",
    )
    parser.add_argument(
        "--context-upo-lite-rerank-bonus",
        type=float,
        default=0.0,
        help="Extra UPO-Lite concept bonus strength in lightweight context reranker. 0 keeps legacy behavior.",
    )
    parser.add_argument(
        "--context-light-rerank-targeted-only",
        dest="context_light_rerank_targeted_only",
        action="store_true",
        help="Apply lightweight reranker only for {number, location} queries.",
    )
    parser.add_argument(
        "--context-light-rerank-global",
        dest="context_light_rerank_targeted_only",
        action="store_false",
        help="Apply lightweight reranker for all query types (default).",
    )
    parser.add_argument(
        "--context-strong-rerank-endpoint",
        default="",
        help="HTTP endpoint for an external reranker service. Empty disables the strong reranker baseline.",
    )
    parser.add_argument(
        "--context-strong-rerank-topn",
        type=int,
        default=20,
        help="Number of top retrieval candidates sent to the external reranker.",
    )
    parser.add_argument(
        "--context-strong-rerank-timeout",
        type=int,
        default=120,
        help="Timeout in seconds for external reranker requests.",
    )
    parser.add_argument(
        "--tessera-candidate-pool-k",
        type=int,
        default=80,
        help="Cap on the TESSERA candidate pool before greedy redundancy scoring; 0 keeps the auto fallback.",
    )
    parser.add_argument(
        "--tessera-retrieval-multi-agent",
        action="store_true",
        help="Enable modular multi-agent retrieval reranking for tessera_rag only.",
    )
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
    parser.add_argument(
        "--tessera-retrieval-moe",
        action="store_true",
        help="Enable query-planned mixture-of-experts retrieval reranking for TESSERA.",
    )
    parser.add_argument("--tessera-moe-pool-k", type=int, default=260)
    parser.add_argument("--tessera-moe-prf-seed-k", type=int, default=6)
    parser.add_argument("--tessera-moe-prf-dense-seed-k", type=int, default=6)
    parser.add_argument("--tessera-moe-prf-sparse-seed-k", type=int, default=6)
    parser.add_argument("--tessera-moe-prf-max-terms", type=int, default=48)
    parser.add_argument("--tessera-moe-sibling-seed-k", type=int, default=6)
    parser.add_argument("--tessera-moe-sibling-window", type=int, default=1)
    parser.add_argument("--tessera-moe-sibling-weight", type=float, default=0.03)
    parser.add_argument(
        "--tessera-ser-ranker",
        type=Path,
        default=None,
        help="Path to a trained TESSERA-SER retrieval-level supervised reranker bundle.",
    )
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
    parser.add_argument(
        "--tessera-graph-evidence-expansion",
        action="store_true",
        help="Enable query-decomposed graph evidence candidate expansion before TESSERA-SER retrieval.",
    )
    parser.add_argument(
        "--tessera-gee-post-rerank",
        action="store_true",
        help="Also run the legacy GEE post-SER reranker. Off by default because v5 hurt top-5 metrics.",
    )
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
    parser.add_argument(
        "--tessera-source-evidence-slot-verifier",
        action="store_true",
        help="Enable evidence slot verifier for conservative learned-style top-k tail replacements.",
    )
    parser.add_argument("--tessera-source-evidence-slot-verifier-prefixes", type=str, default="")
    parser.add_argument("--tessera-source-evidence-slot-verifier-candidate-pool-k", type=int, default=220)
    parser.add_argument("--tessera-source-evidence-slot-verifier-start-slot", type=int, default=4)
    parser.add_argument("--tessera-source-evidence-slot-verifier-max-selected", type=int, default=2)
    parser.add_argument("--tessera-source-evidence-slot-verifier-tail-topn", type=int, default=12)
    parser.add_argument("--tessera-source-evidence-slot-verifier-reference-topn", type=int, default=12)
    parser.add_argument("--tessera-source-evidence-slot-verifier-dense-topn", type=int, default=24)
    parser.add_argument("--tessera-source-evidence-slot-verifier-margin", type=float, default=0.025)
    parser.add_argument("--tessera-source-evidence-slot-verifier-min-score", type=float, default=0.42)
    parser.add_argument(
        "--tessera-source-evidence-slot-verifier-model",
        type=Path,
        default=None,
        help="Path to a trained Pairwise Evidence Slot Verifier bundle.",
    )
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
    parser.add_argument(
        "--tessera-source-evidence-kg-verifier-model",
        type=Path,
        default=None,
        help="Optional Entity-Relation Consistency Verifier bundle for KG preservation.",
    )
    parser.add_argument("--tessera-source-evidence-kg-verifier-weight", type=float, default=0.0)
    parser.add_argument("--tessera-source-evidence-kg-verifier-min-score", type=float, default=0.0)
    parser.add_argument("--tessera-source-evidence-kg-verify-existing", action="store_true")
    parser.add_argument("--tessera-source-evidence-kg-verify-existing-max-replacements", type=int, default=1)
    parser.add_argument(
        "--tessera-source-budgeter-model",
        type=Path,
        default=None,
        help="Path to the query-adaptive source budgeter bundle trained on train/dev qrel source distributions.",
    )
    parser.add_argument(
        "--tessera-source-budgeter-top1-guard",
        action="store_true",
        help="Constrain source-head top1 replacement to the budgeter-predicted source.",
    )
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
    parser.add_argument(
        "--tessera-source-action-policy-model",
        type=Path,
        default=None,
        help="Path to a trained counterfactual source-action policy bundle.",
    )
    parser.add_argument("--tessera-source-action-policy-min-prob", type=float, default=0.42)
    parser.add_argument("--tessera-source-action-policy-topk", type=int, default=5)
    parser.add_argument("--tessera-source-action-policy-pool-k", type=int, default=10)
    parser.add_argument(
        "--tessera-final-evidence-composer",
        action="store_true",
        help="Enable the conservative SER-guided final evidence-set composer for top-k retrieval.",
    )
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
    parser.add_argument(
        "--tessera-final-evidence-verifier-model",
        type=Path,
        default=None,
        help="Optional Evidence Replacement Verifier bundle for final top-k replacement decisions.",
    )
    parser.add_argument("--tessera-final-evidence-verifier-threshold", type=float, default=0.70)
    parser.add_argument("--tessera-final-evidence-verifier-margin", type=float, default=0.0)
    parser.add_argument(
        "--tessera-policy-context",
        action="store_true",
        help="Enable modular TESSERA policy-guided evidence packing for tessera_rag only.",
    )
    parser.add_argument("--tessera-policy-pool-k", type=int, default=80)
    parser.add_argument("--tessera-policy-dense-pool-k", type=int, default=40)
    parser.add_argument("--tessera-policy-target-weight", type=float, default=0.18)
    parser.add_argument("--tessera-policy-coverage-weight", type=float, default=0.16)
    parser.add_argument("--tessera-policy-diversity-weight", type=float, default=0.08)
    parser.add_argument(
        "--tessera-no-evidence-retry",
        action="store_true",
        help="Retry TESSERA reader calls when the first answer says evidence is missing or support is low.",
    )
    parser.add_argument("--tessera-no-evidence-retry-context-k", type=int, default=5)
    parser.add_argument("--tessera-no-evidence-retry-pool-k", type=int, default=120)
    parser.add_argument("--tessera-no-evidence-retry-dense-pool-k", type=int, default=120)
    parser.add_argument("--tessera-no-evidence-retry-min-support", type=float, default=0.08)
    parser.add_argument("--tessera-no-evidence-retry-margin", type=float, default=0.02)
    parser.add_argument("--tessera-no-evidence-retry-low-support-threshold", type=float, default=-1.0)
    parser.add_argument(
        "--tessera-table-number-agent",
        action="store_true",
        help="Enable a local table/number evidence agent that can replace weak number/year LLM answers.",
    )
    parser.add_argument("--tessera-table-number-agent-min-score", type=float, default=0.34)
    parser.add_argument("--tessera-table-number-agent-min-support", type=float, default=0.08)
    parser.add_argument("--tessera-table-number-agent-margin", type=float, default=0.02)
    parser.add_argument("--tessera-table-number-agent-low-support-threshold", type=float, default=0.05)
    parser.set_defaults(context_light_rerank_targeted_only=False)
    parser.add_argument("--context-redundancy-lambda", type=float, default=0.08)
    parser.add_argument(
        "--enable-tessera-answer-calibration",
        action="store_true",
        help="Enable evidence-grounded answer calibration for TESSERA-family methods.",
    )
    parser.add_argument(
        "--tessera-support-retry-threshold",
        type=float,
        default=-1.0,
        help="If >=0, trigger one dense-context retry when TESSERA answer-support score is below this threshold.",
    )
    parser.add_argument(
        "--tessera-support-retry-margin",
        type=float,
        default=0.08,
        help="Minimum support-score improvement required to accept the retry answer.",
    )
    parser.add_argument(
        "--tessera-support-retry-targeted-only",
        dest="tessera_support_retry_targeted_only",
        action="store_true",
        help="Apply support-based retry only for number/year/location queries.",
    )
    parser.add_argument(
        "--tessera-support-retry-global",
        dest="tessera_support_retry_targeted_only",
        action="store_false",
        help="Apply support-based retry for all query types.",
    )
    parser.set_defaults(tessera_support_retry_targeted_only=True)
    parser.add_argument(
        "--tessera-support-retry-mode",
        choices=["dense", "evidence_chain"],
        default="dense",
        help="Retry context construction mode: dense keeps dense top-k; evidence_chain builds support-oriented mixed context.",
    )
    parser.add_argument(
        "--tessera-support-retry-pool-k",
        type=int,
        default=24,
        help="Candidate pool size from dense ranking when retry mode is evidence_chain.",
    )
    parser.add_argument(
        "--tessera-support-retry-answer-boost",
        type=float,
        default=0.55,
        help="Answer-token overlap boost used by evidence_chain retry context scoring.",
    )
    parser.add_argument(
        "--tessera-support-retry-complexity-min",
        type=int,
        default=0,
        help="Minimum query complexity level (1-5) required to allow support retry. 0 disables this gate.",
    )
    parser.add_argument(
        "--tessera-support-retry-entropy-min",
        type=float,
        default=-1.0,
        help="Minimum router entropy required to allow support retry. -1 disables this gate.",
    )
    parser.add_argument(
        "--enable-tessera-consensus-refine",
        action="store_true",
        help="Enable CDG-lite conservative consensus refinement using extractive evidence candidate.",
    )
    parser.add_argument(
        "--tessera-consensus-refine-min-gain",
        type=float,
        default=0.20,
        help="Minimum evidence-support gain required to replace the current answer with consensus candidate.",
    )
    parser.add_argument(
        "--tessera-consensus-refine-support-threshold",
        type=float,
        default=-1.0,
        help="Only run consensus refinement when answer support is below this threshold. -1 disables this gate.",
    )
    parser.add_argument(
        "--tessera-consensus-refine-complexity-min",
        type=int,
        default=0,
        help="Minimum query complexity level (1-5) required to run consensus refinement. 0 disables this gate.",
    )
    parser.add_argument(
        "--tessera-consensus-refine-entropy-min",
        type=float,
        default=-1.0,
        help="Minimum router entropy required to run consensus refinement. -1 disables this gate.",
    )
    parser.add_argument(
        "--tessera-consensus-refine-targeted-only",
        dest="tessera_consensus_refine_targeted_only",
        action="store_true",
        help="Apply consensus refinement only for number/year/location queries.",
    )
    parser.add_argument(
        "--tessera-consensus-refine-global",
        dest="tessera_consensus_refine_targeted_only",
        action="store_false",
        help="Apply consensus refinement for all query types.",
    )
    parser.set_defaults(tessera_consensus_refine_targeted_only=True)
    parser.add_argument(
        "--enable-tessera-answer-type-guard",
        action="store_true",
        help="Enable conservative type-consistency guard for TESSERA answers on number/year queries.",
    )
    parser.add_argument(
        "--allow-heuristic-router-fallback",
        action="store_true",
        help="Allow heuristic router fallback if DeBERTa model is unavailable. For paper-grade runs, keep this disabled.",
    )
    parser.add_argument(
        "--include-oracle-row",
        action="store_true",
        help="Include an Oracle row in output tables. Disabled by default to avoid mixing synthetic upper bounds into measured rows.",
    )
    parser.add_argument(
        "--include-oracle-measured-row",
        action="store_true",
        help="Include a measured Oracle row by feeding gold evidence to the same reader.",
    )
    parser.add_argument(
        "--official-mmrag-mode",
        action="store_true",
        help="Use mmRAG official generation-style settings: force qa_context_k=3 and report mixed official score.",
    )
    parser.add_argument(
        "--method-preset",
        choices=sorted(METHOD_PRESETS.keys()),
        default="targeted",
        help="Method preset for resource-efficient runs. Use full for complete Table1c runs.",
    )
    parser.add_argument(
        "--methods",
        type=str,
        default=None,
        help="Comma-separated method keys to run. Overrides --method-preset when provided.",
    )
    parser.add_argument(
        "--reuse-methods",
        type=str,
        default="",
        help="Comma-separated method keys whose existing qa_predictions_*.jsonl should be reused from --out-dir.",
    )
    parser.add_argument("--pathmaxsim-weight", type=float, default=0.14)
    parser.add_argument("--pathmaxsim-kg-threshold", type=float, default=0.0)
    parser.add_argument(
        "--innovation-scheme2",
        action="store_true",
        help="Enable scheme2 ranking improvements: uncertainty-aware interaction and cross-modal agreement bonus.",
    )
    parser.add_argument(
        "--scheme2-cross-modal-weight",
        type=float,
        default=0.12,
        help="Weight of scheme2 cross-modal agreement bonus in TESSERA ranking.",
    )
    parser.add_argument(
        "--scheme2-token-maxsim-weight",
        type=float,
        default=0.02,
        help="Weight of scheme2 token-level MaxSim-like bonus in TESSERA ranking.",
    )
    parser.add_argument(
        "--schemeb-heavy-mode",
        action="store_true",
        help="Enable heavy schemeB modules: table cell encoder, KG path encoder, and token-level late interaction.",
    )
    parser.add_argument(
        "--heavy-table-encoder-weight",
        type=float,
        default=0.08,
        help="Weight of heavy table cell/row/column encoder score in TESSERA ranking.",
    )
    parser.add_argument(
        "--heavy-kg-path-weight",
        type=float,
        default=0.08,
        help="Weight of heavy KG path encoder score in TESSERA ranking.",
    )
    parser.add_argument(
        "--heavy-token-late-weight",
        type=float,
        default=0.06,
        help="Weight of heavy token-level late interaction score in TESSERA ranking.",
    )
    parser.add_argument(
        "--heavy-query-max-tokens",
        type=int,
        default=28,
        help="Max number of query tokens used by heavy vector interaction modules.",
    )
    parser.add_argument(
        "--heavy-table-max-cells",
        type=int,
        default=256,
        help="Max number of table cells used in heavy table encoder per table doc.",
    )
    parser.add_argument(
        "--heavy-token-doc-max-tokens",
        type=int,
        default=96,
        help="Max number of document tokens used in heavy token late interaction per doc.",
    )
    parser.add_argument(
        "--heavy-table-backend",
        choices=["hash", "tapas"],
        default="hash",
        help="Backend for heavy table encoder. 'tapas' enables TAPAS cell embeddings with row/col learnable aggregation.",
    )
    parser.add_argument(
        "--heavy-table-tapas-model",
        type=str,
        default="",
        help="Local path or model id for TAPAS encoder. Empty means fallback to hash backend.",
    )
    parser.add_argument(
        "--heavy-table-tapas-topn",
        type=int,
        default=24,
        help="Apply TAPAS scoring to top-N table candidates per query, fallback to hash scorer for the rest.",
    )
    parser.add_argument(
        "--heavy-table-max-rows",
        type=int,
        default=48,
        help="Max rows parsed from markdown table for TAPAS encoding.",
    )
    parser.add_argument(
        "--heavy-table-max-cols",
        type=int,
        default=16,
        help="Max cols parsed from markdown table for TAPAS encoding.",
    )
    parser.add_argument(
        "--heavy-table-agg-cell-logit",
        type=float,
        default=1.20,
        help="Learnable aggregation logit scale for cell branch in TAPAS table encoder.",
    )
    parser.add_argument(
        "--heavy-table-agg-row-logit",
        type=float,
        default=1.00,
        help="Learnable aggregation logit scale for row branch in TAPAS table encoder.",
    )
    parser.add_argument(
        "--heavy-table-agg-col-logit",
        type=float,
        default=0.90,
        help="Learnable aggregation logit scale for col branch in TAPAS table encoder.",
    )
    parser.add_argument(
        "--heavy-table-agg-temp",
        type=float,
        default=0.70,
        help="Softmax temperature for learnable row/col/cell aggregation.",
    )
    parser.add_argument(
        "--heavy-kg-backend",
        choices=["token", "gnn"],
        default="gnn",
        help="Backend for heavy KG path encoder. 'gnn' enables GraphSAGE-style multi-hop path encoding.",
    )
    parser.add_argument(
        "--heavy-kg-gnn-topn",
        type=int,
        default=48,
        help="Apply GNN path scoring to top-N KG candidates per query, fallback to token path scorer for the rest.",
    )
    parser.add_argument(
        "--heavy-kg-max-hops",
        type=int,
        default=3,
        help="Max hops for KG multi-hop path sampling in GNN backend.",
    )
    parser.add_argument(
        "--heavy-kg-max-paths",
        type=int,
        default=64,
        help="Max sampled paths per KG candidate in GNN backend.",
    )
    parser.add_argument(
        "--heavy-kg-contrastive-temp",
        type=float,
        default=0.12,
        help="Temperature for contrastive path-query alignment scoring.",
    )
    parser.add_argument(
        "--heavy-kg-hard-negative-mode",
        choices=["roll", "cross_doc_hard"],
        default="cross_doc_hard",
        help="Hard-negative strategy for KG contrastive branch.",
    )
    parser.add_argument(
        "--heavy-kg-hard-negative-topdocs",
        type=int,
        default=3,
        help="When using cross_doc_hard, number of similar KG docs used to build hard negatives.",
    )
    parser.add_argument(
        "--heavy-kg-hard-negative-max-paths",
        type=int,
        default=24,
        help="When using cross_doc_hard, max negative path vectors per KG candidate.",
    )
    parser.add_argument(
        "--heavy-token-cross-modal-weight",
        type=float,
        default=0.02,
        help="Weight of token-level cross-modal fine-grained interaction bonus.",
    )
    parser.add_argument(
        "--heavy-branch-candidate-expand-k",
        type=int,
        default=0,
        help="Additional top-k candidates to pull from table/kg branch-specific pools for TESSERA candidate expansion.",
    )
    parser.add_argument(
        "--heavy-branch-candidate-table-weight",
        type=float,
        default=0.55,
        help="Dense weight used to rank table branch candidates during expansion (sparse weight is 1-w).",
    )
    parser.add_argument(
        "--heavy-branch-candidate-kg-weight",
        type=float,
        default=0.55,
        help="Dense weight used to rank KG branch candidates during expansion (sparse weight is 1-w).",
    )
    parser.add_argument(
        "--heavy-branch-candidate-max-total",
        type=int,
        default=1400,
        help="Upper bound of TESSERA candidate pool size after branch expansion.",
    )
    parser.add_argument(
        "--heavy-remove-hard-caps",
        dest="heavy_remove_hard_caps",
        action="store_true",
        help="Disable heavy top-N and candidate pool hard caps (set table/kg topn and candidate max-total to unlimited). Default: enabled.",
    )
    parser.add_argument(
        "--heavy-keep-hard-caps",
        dest="heavy_remove_hard_caps",
        action="store_false",
        help="Keep heavy top-N and candidate pool hard caps (legacy behavior).",
    )
    parser.set_defaults(heavy_remove_hard_caps=True)
    parser.add_argument(
        "--heavy-table-tapas-required",
        action="store_true",
        help="Fail fast if TAPAS backend is requested but unavailable, instead of silently falling back to hash.",
    )
    parser.add_argument(
        "--heavy-score-calibration",
        choices=["none", "minmax", "zscore", "robust", "rank"],
        default="none",
        help="Per-query calibration mode for heavy component scores before linear fusion.",
    )
    parser.add_argument(
        "--heavy-score-calibration-nonzero-only",
        action="store_true",
        help="Apply heavy score calibration only on non-zero candidates, keeping untouched zeros at 0.",
    )
    parser.add_argument(
        "--qa-objective-retrieval-weight",
        type=float,
        default=0.04,
        help="Weight of QA-objective reverse constraint score when ranking TESSERA candidates.",
    )
    parser.add_argument(
        "--qa-objective-targeted-only",
        dest="qa_objective_targeted_only",
        action="store_true",
        help="Apply retrieval QA-objective bonus only for number/year/boolean query types.",
    )
    parser.add_argument(
        "--qa-objective-global",
        dest="qa_objective_targeted_only",
        action="store_false",
        help="Apply retrieval QA-objective bonus to all query types (default behavior).",
    )
    parser.set_defaults(qa_objective_targeted_only=False)
    parser.add_argument(
        "--adapter-plus-mode",
        action="store_true",
        help="Enable stronger adapter formulas for CARP/TableRAG/QUASAR under the same unified protocol.",
    )
    parser.add_argument(
        "--adapter-official-lite",
        action="store_true",
        help="Enable an official-lite adapter protocol (TableRAG row/column/schema emphasis + QUASAR dynamic quotas).",
    )
    parser.add_argument(
        "--table-cellmaxsim-weight",
        type=float,
        default=0.0,
        help="Weight of table cell-level MaxSim-like bonus in ranking. 0 disables the bonus.",
    )
    parser.add_argument(
        "--table-cellmaxsim-top-cells",
        type=int,
        default=160,
        help="Max number of parsed table cells used for cell-level scoring per table doc.",
    )
    parser.add_argument(
        "--context-consistency-weight",
        type=float,
        default=0.0,
        help="Weight of cross-modal consistency bonus in CAMPE context selection. 0 disables it.",
    )
    parser.add_argument(
        "--context-candidate-expand-k",
        type=int,
        default=0,
        help="Extra candidates from dense ranking to include in CAMPE context scoring pool.",
    )
    parser.add_argument(
        "--context-qa-objective-weight",
        type=float,
        default=0.0,
        help="Weight of QA-objective score in CAMPE context candidate scoring. 0 disables it.",
    )
    parser.add_argument(
        "--context-qa-modality-bias",
        type=float,
        default=0.0,
        help="Strength of query-type-conditioned modality bias in CAMPE context quotas.",
    )
    parser.add_argument(
        "--context-upo-lite-quota-mix",
        type=float,
        default=0.0,
        help="Blend ratio of UPO-Lite concept prior into CAMPE context quotas (0=off, 1=UPO-only).",
    )
    parser.add_argument(
        "--context-conflict-penalty-weight",
        type=float,
        default=0.0,
        help="Penalty weight for cross-modal numeric conflicts during CAMPE context selection. 0 disables it.",
    )
    parser.add_argument(
        "--context-conflict-targeted-only",
        dest="context_conflict_targeted_only",
        action="store_true",
        help="Apply conflict penalty only for number/year queries.",
    )
    parser.add_argument(
        "--context-conflict-global",
        dest="context_conflict_targeted_only",
        action="store_false",
        help="Apply conflict penalty to all query types (default).",
    )
    parser.set_defaults(context_conflict_targeted_only=False)
    parser.add_argument(
        "--context-conflict-table-kg-only",
        dest="context_conflict_table_kg_only",
        action="store_true",
        help="Apply conflict penalty only on table<->kg cross-modal pairs.",
    )
    parser.add_argument(
        "--context-conflict-all-cross-modal",
        dest="context_conflict_table_kg_only",
        action="store_false",
        help="Apply conflict penalty on all cross-modal pairs (default).",
    )
    parser.set_defaults(context_conflict_table_kg_only=False)
    parser.add_argument(
        "--context-conflict-risk-gating",
        action="store_true",
        help="Enable query-level conflict-risk gating to adaptively scale conflict penalty weight.",
    )
    parser.add_argument(
        "--context-conflict-no-risk-gating",
        dest="context_conflict_risk_gating",
        action="store_false",
        help="Disable query-level conflict-risk gating.",
    )
    parser.set_defaults(context_conflict_risk_gating=False)
    parser.add_argument(
        "--context-conflict-risk-low",
        type=float,
        default=0.06,
        help="Low-risk threshold for conflict-risk gating; below this, conflict penalty is mostly off.",
    )
    parser.add_argument(
        "--context-conflict-risk-high",
        type=float,
        default=0.22,
        help="High-risk threshold for conflict-risk gating; above this, full conflict penalty is used.",
    )
    parser.add_argument(
        "--context-conflict-risk-probe-k",
        type=int,
        default=12,
        help="Number of top candidates used to estimate query-level conflict risk.",
    )
    parser.add_argument(
        "--context-conflict-max-literals-per-doc",
        type=int,
        default=0,
        help="If >0, skip conflict checks for docs with more than this many numeric literals (reduces false conflicts on number-dense docs).",
    )
    parser.add_argument(
        "--context-conflict-sensitive-target-scale",
        type=float,
        default=1.0,
        help="Extra scale on conflict penalty for sensitive targets {number, location}; <1.0 weakens penalty.",
    )
    parser.add_argument(
        "--extractive-numeric-consensus",
        action="store_true",
        help="When using extractive reader, choose numeric/year answers by weighted evidence consensus across top sentences.",
    )
    parser.add_argument(
        "--unihgkr-model-dir",
        type=Path,
        default=ROOT.parent / "downloaded_resource/compmix-ir-benchmarks/ZhishanQ-UniHGKR-base",
        help="Path to UniHGKR dense encoder model used for a fair same-protocol baseline.",
    )
    parser.add_argument(
        "--unihgkr-batch-size",
        type=int,
        default=64,
        help="Batch size for UniHGKR embedding inference.",
    )
    args = parser.parse_args()

    if int(args.progress_every) < 0:
        raise ValueError("progress-every must be >= 0")
    if float(args.progress_min_seconds) < 0.0:
        raise ValueError("progress-min-seconds must be >= 0")
    if int(args.openai_max_retries) < 1:
        raise ValueError("openai-max-retries must be >= 1")
    if float(args.openai_retry_backoff) < 0.0:
        raise ValueError("openai-retry-backoff must be >= 0")

    if float(args.query_modality_prior_min) > float(args.query_modality_prior_max):
        raise ValueError(
            "query-modality-prior-min must be <= query-modality-prior-max"
        )
    if float(args.upo_lite_retrieval_weight) < 0.0:
        raise ValueError("upo-lite-retrieval-weight must be >= 0")
    if float(args.retrieval_conflict_penalty_weight) < 0.0:
        raise ValueError("retrieval-conflict-penalty-weight must be >= 0")
    if float(args.retrieval_conflict_risk_low) < 0.0 or float(args.retrieval_conflict_risk_low) > 1.0:
        raise ValueError("retrieval-conflict-risk-low must be in [0,1]")
    if float(args.retrieval_conflict_risk_high) < 0.0 or float(args.retrieval_conflict_risk_high) > 1.0:
        raise ValueError("retrieval-conflict-risk-high must be in [0,1]")
    if float(args.retrieval_conflict_risk_high) < float(args.retrieval_conflict_risk_low):
        raise ValueError("retrieval-conflict-risk-high must be >= retrieval-conflict-risk-low")
    if int(args.retrieval_conflict_risk_probe_k) < 2:
        raise ValueError("retrieval-conflict-risk-probe-k must be >= 2")
    if int(args.retrieval_conflict_anchor_k) < 2:
        raise ValueError("retrieval-conflict-anchor-k must be >= 2")
    if int(args.retrieval_conflict_max_literals_per_doc) < 0:
        raise ValueError("retrieval-conflict-max-literals-per-doc must be >= 0")
    if float(args.retrieval_conflict_sensitive_target_scale) < 0.0 or float(args.retrieval_conflict_sensitive_target_scale) > 1.0:
        raise ValueError("retrieval-conflict-sensitive-target-scale must be in [0,1]")
    if float(args.context_qa_objective_weight) < 0.0:
        raise ValueError("context-qa-objective-weight must be >= 0")
    if float(args.context_qa_modality_bias) < 0.0:
        raise ValueError("context-qa-modality-bias must be >= 0")
    if float(args.context_upo_lite_quota_mix) < 0.0 or float(args.context_upo_lite_quota_mix) > 1.0:
        raise ValueError("context-upo-lite-quota-mix must be in [0,1]")
    if float(args.context_upo_lite_rerank_bonus) < 0.0 or float(args.context_upo_lite_rerank_bonus) > 1.0:
        raise ValueError("context-upo-lite-rerank-bonus must be in [0,1]")
    if float(args.context_conflict_penalty_weight) < 0.0:
        raise ValueError("context-conflict-penalty-weight must be >= 0")
    if float(args.context_conflict_risk_low) < 0.0 or float(args.context_conflict_risk_low) > 1.0:
        raise ValueError("context-conflict-risk-low must be in [0,1]")
    if float(args.context_conflict_risk_high) < 0.0 or float(args.context_conflict_risk_high) > 1.0:
        raise ValueError("context-conflict-risk-high must be in [0,1]")
    if float(args.context_conflict_risk_high) < float(args.context_conflict_risk_low):
        raise ValueError("context-conflict-risk-high must be >= context-conflict-risk-low")
    if int(args.context_conflict_risk_probe_k) < 2:
        raise ValueError("context-conflict-risk-probe-k must be >= 2")
    if float(args.context_conflict_sensitive_target_scale) < 0.0 or float(args.context_conflict_sensitive_target_scale) > 1.0:
        raise ValueError("context-conflict-sensitive-target-scale must be in [0,1]")
    if int(args.context_conflict_max_literals_per_doc) < 0:
        raise ValueError("context-conflict-max-literals-per-doc must be >= 0")
    if int(args.context_dense_pool_k) <= 0:
        raise ValueError("context-dense-pool-k must be > 0")
    if int(args.context_number_table_quota_min) < 0:
        raise ValueError("context-number-table-quota-min must be >= 0")
    if float(args.context_subquery_coverage_weight) < 0.0:
        raise ValueError("context-subquery-coverage-weight must be >= 0")
    if float(args.context_light_rerank_weight) < 0.0 or float(args.context_light_rerank_weight) > 1.0:
        raise ValueError("context-light-rerank-weight must be in [0,1]")
    if int(args.context_light_rerank_topn) < 1:
        raise ValueError("context-light-rerank-topn must be >= 1")
    if int(args.context_strong_rerank_topn) < 1:
        raise ValueError("context-strong-rerank-topn must be >= 1")
    if int(args.context_strong_rerank_timeout) < 1:
        raise ValueError("context-strong-rerank-timeout must be >= 1")
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
    if float(args.tessera_source_evidence_slot_verifier_redundancy_weight) < 0.0:
        raise ValueError("tessera-source-evidence-slot-verifier-redundancy-weight must be >= 0")
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
    if int(args.tessera_policy_pool_k) < 1:
        raise ValueError("tessera-policy-pool-k must be >= 1")
    if int(args.tessera_policy_dense_pool_k) < 1:
        raise ValueError("tessera-policy-dense-pool-k must be >= 1")
    if float(args.tessera_policy_target_weight) < 0.0:
        raise ValueError("tessera-policy-target-weight must be >= 0")
    if float(args.tessera_policy_coverage_weight) < 0.0:
        raise ValueError("tessera-policy-coverage-weight must be >= 0")
    if float(args.tessera_policy_diversity_weight) < 0.0:
        raise ValueError("tessera-policy-diversity-weight must be >= 0")
    if int(args.tessera_no_evidence_retry_context_k) < 1:
        raise ValueError("tessera-no-evidence-retry-context-k must be >= 1")
    if int(args.tessera_no_evidence_retry_pool_k) < 1:
        raise ValueError("tessera-no-evidence-retry-pool-k must be >= 1")
    if int(args.tessera_no_evidence_retry_dense_pool_k) < 1:
        raise ValueError("tessera-no-evidence-retry-dense-pool-k must be >= 1")
    if float(args.tessera_no_evidence_retry_min_support) < -1.0:
        raise ValueError("tessera-no-evidence-retry-min-support must be >= -1")
    if float(args.tessera_no_evidence_retry_margin) < 0.0:
        raise ValueError("tessera-no-evidence-retry-margin must be >= 0")
    if float(args.tessera_no_evidence_retry_low_support_threshold) < -1.0:
        raise ValueError("tessera-no-evidence-retry-low-support-threshold must be >= -1")
    if float(args.tessera_table_number_agent_min_score) < 0.0:
        raise ValueError("tessera-table-number-agent-min-score must be >= 0")
    if float(args.tessera_table_number_agent_min_support) < -1.0:
        raise ValueError("tessera-table-number-agent-min-support must be >= -1")
    if float(args.tessera_table_number_agent_margin) < 0.0:
        raise ValueError("tessera-table-number-agent-margin must be >= 0")
    if float(args.tessera_table_number_agent_low_support_threshold) < -1.0:
        raise ValueError("tessera-table-number-agent-low-support-threshold must be >= -1")
    if int(args.context_candidate_expand_k) < 0:
        raise ValueError("context-candidate-expand-k must be >= 0")
    if int(args.intent_complexity_context_k_simple) < 1:
        raise ValueError("intent-complexity-context-k-simple must be >= 1")
    if int(args.intent_complexity_context_k_medium) < 1:
        raise ValueError("intent-complexity-context-k-medium must be >= 1")
    if int(args.intent_complexity_context_k_complex) < 1:
        raise ValueError("intent-complexity-context-k-complex must be >= 1")
    if int(args.intent_complexity_dense_pool_k_simple) < 1:
        raise ValueError("intent-complexity-dense-pool-k-simple must be >= 1")
    if int(args.intent_complexity_dense_pool_k_medium) < 1:
        raise ValueError("intent-complexity-dense-pool-k-medium must be >= 1")
    if int(args.intent_complexity_dense_pool_k_complex) < 1:
        raise ValueError("intent-complexity-dense-pool-k-complex must be >= 1")
    if int(args.intent_complexity_candidate_expand_k_simple) < 0:
        raise ValueError("intent-complexity-candidate-expand-k-simple must be >= 0")
    if int(args.intent_complexity_candidate_expand_k_medium) < 0:
        raise ValueError("intent-complexity-candidate-expand-k-medium must be >= 0")
    if int(args.intent_complexity_candidate_expand_k_complex) < 0:
        raise ValueError("intent-complexity-candidate-expand-k-complex must be >= 0")
    if float(args.tessera_support_retry_threshold) < -1.0:
        raise ValueError("tessera-support-retry-threshold must be >= -1")
    if float(args.tessera_support_retry_margin) < 0.0:
        raise ValueError("tessera-support-retry-margin must be >= 0")
    if int(args.tessera_support_retry_pool_k) < 1:
        raise ValueError("tessera-support-retry-pool-k must be >= 1")
    if float(args.tessera_support_retry_answer_boost) < 0.0:
        raise ValueError("tessera-support-retry-answer-boost must be >= 0")
    if int(args.tessera_support_retry_complexity_min) < 0 or int(args.tessera_support_retry_complexity_min) > 5:
        raise ValueError("tessera-support-retry-complexity-min must be in [0,5]")
    if float(args.tessera_support_retry_entropy_min) < -1.0 or float(args.tessera_support_retry_entropy_min) > 1.0:
        raise ValueError("tessera-support-retry-entropy-min must be in [-1,1]")
    if float(args.tessera_consensus_refine_min_gain) < 0.0:
        raise ValueError("tessera-consensus-refine-min-gain must be >= 0")
    if float(args.tessera_consensus_refine_support_threshold) < -1.0:
        raise ValueError("tessera-consensus-refine-support-threshold must be >= -1")
    if int(args.tessera_consensus_refine_complexity_min) < 0 or int(args.tessera_consensus_refine_complexity_min) > 5:
        raise ValueError("tessera-consensus-refine-complexity-min must be in [0,5]")
    if float(args.tessera_consensus_refine_entropy_min) < -1.0 or float(args.tessera_consensus_refine_entropy_min) > 1.0:
        raise ValueError("tessera-consensus-refine-entropy-min must be in [-1,1]")

    if bool(args.heavy_remove_hard_caps):
        args.heavy_table_tapas_topn = 0
        args.heavy_kg_gnn_topn = 0
        args.heavy_branch_candidate_max_total = 0

    effective_qa_context_k = int(args.qa_context_k)
    if args.official_mmrag_mode:
        if effective_qa_context_k != 3:
            print(f"[official-mmrag-mode] overriding qa_context_k {effective_qa_context_k} -> 3")
        effective_qa_context_k = 3
        if bool(args.intent_complexity_aware_budgeting):
            print("[official-mmrag-mode] disabling intent-complexity-aware-budgeting to keep fixed qa_context_k=3 protocol.")
            args.intent_complexity_aware_budgeting = False

    if bool(args.intent_complexity_aware_budgeting):
        print(
            "[config] intent-complexity-aware-budgeting on: "
            f"context_k(simple/medium/complex)="
            f"{int(args.intent_complexity_context_k_simple)}/"
            f"{int(args.intent_complexity_context_k_medium)}/"
            f"{int(args.intent_complexity_context_k_complex)}, "
            f"dense_pool_k="
            f"{int(args.intent_complexity_dense_pool_k_simple)}/"
            f"{int(args.intent_complexity_dense_pool_k_medium)}/"
            f"{int(args.intent_complexity_dense_pool_k_complex)}, "
            f"candidate_expand_k="
            f"{int(args.intent_complexity_candidate_expand_k_simple)}/"
            f"{int(args.intent_complexity_candidate_expand_k_medium)}/"
            f"{int(args.intent_complexity_candidate_expand_k_complex)}"
        )

    selected_methods = resolve_selected_methods(
        method_preset=str(args.method_preset),
        methods_raw=args.methods,
    )
    reuse_methods = resolve_reuse_methods(str(args.reuse_methods))

    print(f"[config] method_preset={args.method_preset} methods={selected_methods}")
    if reuse_methods:
        print(f"[config] reuse_methods={reuse_methods}")

    rows = json.loads(args.split_file.read_text(encoding="utf-8"))[: args.max_queries]
    corpus = json.loads(args.corpus_file.read_text(encoding="utf-8"))

    q_ids = [r.get("id", f"q_{i}") for i, r in enumerate(rows)]
    q_texts = [r.get("query", "") for r in rows]
    gold_answers = [str(r.get("answer", "")) for r in rows]
    y_router_true = np.asarray([qrel_modalities(r) for r in rows], dtype=np.int64)

    doc_ids = [d["id"] for d in corpus]
    doc_texts = [d.get("text", "") for d in corpus]
    doc_id_to_idx = {did: i for i, did in enumerate(doc_ids)}

    qrels_positive_total = 0
    qrels_positive_in_corpus = 0
    queries_with_missing_qrels_in_corpus = 0
    for r in rows:
        rel_ids = eval_positive_relevant_ids(r)
        if not rel_ids:
            continue
        in_corpus = sum(1 for cid in rel_ids if cid in doc_id_to_idx)
        qrels_positive_total += len(rel_ids)
        qrels_positive_in_corpus += in_corpus
        if in_corpus < len(rel_ids):
            queries_with_missing_qrels_in_corpus += 1
    qrels_coverage_in_corpus = float(qrels_positive_in_corpus / max(1, qrels_positive_total))
    if qrels_coverage_in_corpus >= 0.99 and "test" in str(args.split_file).lower():
        print(
            "[warn] qrels_coverage_in_corpus is near 1.0 on a test split. "
            "This indicates a strongly qrel-augmented/transductive corpus setting and can inflate absolute baseline performance."
        )

    table_doc_indices = np.asarray([i for i, did in enumerate(doc_ids) if source_bucket(did) == "table"], dtype=np.int64)
    kg_doc_indices = np.asarray([i for i, did in enumerate(doc_ids) if source_bucket(did) == "kg"], dtype=np.int64)
    print("[stage] using lazy document token stores")
    doc_tokens = LazyDocTokenStore(doc_texts)
    doc_prefix_tokens = LazyDocPrefixTokenStore(doc_tokens)
    use_heavy_schemeb = bool(args.schemeb_heavy_mode)
    use_cellmaxsim = float(args.table_cellmaxsim_weight) > 0.0
    use_heavy_table_encoder = use_heavy_schemeb and float(args.heavy_table_encoder_weight) > 0.0
    use_heavy_kg_encoder = use_heavy_schemeb and float(args.heavy_kg_path_weight) > 0.0
    use_heavy_token_late = use_heavy_schemeb and float(args.heavy_token_late_weight) > 0.0
    use_heavy_token_cross = use_heavy_schemeb and float(args.heavy_token_cross_modal_weight) > 0.0
    use_consistency = float(args.context_consistency_weight) > 0.0
    use_retrieval_conflict_penalty = float(args.retrieval_conflict_penalty_weight) > 0.0
    use_conflict_penalty = float(args.context_conflict_penalty_weight) > 0.0 or use_retrieval_conflict_penalty

    heavy_table_backend_effective = str(args.heavy_table_backend).lower().strip()
    heavy_table_tapas_model_resolved = str(args.heavy_table_tapas_model).strip()
    tapas_bundle = None
    if use_heavy_table_encoder and heavy_table_backend_effective == "tapas":
        tapas_bundle = maybe_load_tapas_bundle(str(args.heavy_table_tapas_model))
        if tapas_bundle is None:
            msg = (
                "TAPAS backend requested but unavailable. "
                "Please verify heavy-table-tapas-model path/checkpoint format and required runtime deps."
            )
            if bool(args.heavy_table_tapas_required):
                raise RuntimeError(msg)
            print(f"[warn] {msg} Fallback to hash table encoder.")
            heavy_table_backend_effective = "hash"
        else:
            heavy_table_tapas_model_resolved = str(tapas_bundle.get("model_path", heavy_table_tapas_model_resolved))

    table_tapas_cache: dict[int, dict[str, np.ndarray] | None] = {}
    kg_gnn_cache: dict[int, dict[str, np.ndarray] | None] = {}

    doc_token_lists: list[list[str]] = [[] for _ in doc_ids]
    if use_heavy_token_late or use_heavy_token_cross:
        print("[stage] building document token lists for heavy token interactions")
        doc_token_lists = [tokenize_list(t) for t in doc_texts]

    doc_table_structs: list[dict[str, list[set[str]]] | None] = [None for _ in doc_ids]
    if use_cellmaxsim or use_heavy_table_encoder:
        print("[stage] building table cell structures for table encoders")
        doc_table_structs = [
            extract_table_structure_tokens(
                t,
                max_cells=max(
                    64,
                    int(max(args.table_cellmaxsim_top_cells, args.heavy_table_max_cells)),
                ),
            )
            if source_bucket(did) == "table"
            else None
            for did, t in zip(doc_ids, doc_texts)
        ]

    doc_kg_path_sets: list[list[set[str]] | None] = [None for _ in doc_ids]
    if use_heavy_kg_encoder:
        print("[stage] building KG path token sets for heavy path encoder")
        doc_kg_path_sets = [
            extract_kg_path_sets(t)
            if source_bucket(did) == "kg"
            else None
            for did, t in zip(doc_ids, doc_texts)
        ]

    doc_signal_tokens: list[set[str]] = [set() for _ in doc_ids]
    if use_consistency or use_conflict_penalty:
        print("[stage] building cross-modal consistency signal tokens")
        doc_signal_tokens = [extract_consistency_signal_tokens(t) for t in doc_texts]
    doc_numeric_literals: list[set[str]] = [set() for _ in doc_ids]
    if use_conflict_penalty:
        print("[stage] building numeric literals for conflict retrieval features")
        doc_numeric_literals = [extract_numeric_literals(t) for t in doc_texts]

    args.out_dir.mkdir(parents=True, exist_ok=True)
    args.cache_dir.mkdir(parents=True, exist_ok=True)

    q_key = make_cache_key(q_ids)
    c_key = make_cache_key(doc_ids)
    sparse_cache = args.cache_dir / f"tfidf_scores_{len(q_texts)}x{len(doc_texts)}_{q_key}_{c_key}.npy"

    tokenizer, model, device, resolved = load_e5(args.model_dir)
    embed_backend = os.environ.get("TESSERA_EMBED_BACKEND", "hf").strip().lower() or "hf"
    query_prefix = os.environ.get("TESSERA_QUERY_PREFIX", "")
    doc_prefix = os.environ.get("TESSERA_DOC_PREFIX", "")
    print(
        f"[stage] model={resolved} backend={embed_backend} device={device} "
        f"queries={len(q_texts)} corpus={len(doc_texts)}"
    )

    # Auto-detect pooling mode and normalization from model config (BGE uses CLS, E5 uses mean)
    pooling_mode = detect_st_pooling_mode(resolved)
    do_normalize = has_st_normalize(resolved)
    print(f"[stage] detected pooling_mode={pooling_mode} normalize={do_normalize}")

    model_key = hashlib.sha1(str(resolved).encode("utf-8")).hexdigest()[:12]
    q_prefix_key = hashlib.sha1(f"{embed_backend}|{pooling_mode}|{query_prefix}".encode("utf-8")).hexdigest()[:10]
    c_prefix_key = hashlib.sha1(f"{embed_backend}|{pooling_mode}|{doc_prefix}".encode("utf-8")).hexdigest()[:10]
    q_cache = args.cache_dir / f"dense_query_{model_key}_{pooling_mode}_{q_prefix_key}_{len(q_texts)}_{q_key}.npy"
    c_cache = args.cache_dir / f"dense_corpus_{model_key}_{pooling_mode}_{c_prefix_key}_{len(doc_texts)}_{c_key}.npy"

    if q_cache.exists() and np.load(q_cache, mmap_mode="r").shape[0] == len(q_texts):
        qv = np.load(q_cache)
    else:
        qv = encode_texts(
            q_texts, tokenizer, model, device,
            batch_size=args.batch_size, pooling_mode=pooling_mode, query_prefix=query_prefix,
        )
        np.save(q_cache, qv)

    if c_cache.exists() and np.load(c_cache, mmap_mode="r").shape[0] == len(doc_texts):
        cv = np.load(c_cache)
    else:
        cv = encode_texts(
            doc_texts, tokenizer, model, device,
            batch_size=args.batch_size, pooling_mode=pooling_mode, query_prefix=doc_prefix,
        )
        np.save(c_cache, cv)

    dense_scores = qv @ cv.T

    if sparse_cache.exists():
        sparse_scores = np.load(sparse_cache)
        if sparse_scores.shape != (len(q_texts), len(doc_texts)):
            sparse_scores = None
    else:
        sparse_scores = None

    if sparse_scores is None:
        vec = TfidfVectorizer(stop_words="english", ngram_range=(1, 2), max_features=200000, min_df=2)
        c_mat = vec.fit_transform(doc_texts)
        q_mat = vec.transform(q_texts)
        sparse_scores = (q_mat @ c_mat.T).toarray().astype(np.float32)
        np.save(sparse_cache, sparse_scores)

    unihgkr_scores = None
    if "unihgkr_dense" in selected_methods:
        if AutoTokenizer is None or AutoModel is None or torch is None:
            raise RuntimeError("UniHGKR baseline requires transformers + torch runtime")
        if not args.unihgkr_model_dir.exists():
            raise FileNotFoundError(f"UniHGKR model dir not found: {args.unihgkr_model_dir}")

        pooling_mode = detect_st_pooling_mode(args.unihgkr_model_dir)
        do_normalize = has_st_normalize(args.unihgkr_model_dir)
        uh_model_key = hashlib.sha1(str(args.unihgkr_model_dir.resolve()).encode("utf-8")).hexdigest()[:12]
        uq_cache = args.cache_dir / f"unihgkr_query_{uh_model_key}_{pooling_mode}_{len(q_texts)}_{q_key}.npy"
        uc_cache = args.cache_dir / f"unihgkr_corpus_{uh_model_key}_{pooling_mode}_{len(doc_texts)}_{c_key}.npy"
        print(f"[stage] loading UniHGKR dense baseline model from {args.unihgkr_model_dir}")
        print(f"[stage] UniHGKR pooling={pooling_mode} normalize={do_normalize}")
        uh_tokenizer = AutoTokenizer.from_pretrained(str(args.unihgkr_model_dir))
        uh_model = AutoModel.from_pretrained(str(args.unihgkr_model_dir))
        uh_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        uh_model.to(uh_device)

        if uq_cache.exists() and np.load(uq_cache, mmap_mode="r").shape[0] == len(q_texts):
            uqv = np.load(uq_cache)
        else:
            uqv = encode_texts_with_hf_encoder(
                q_texts,
                uh_tokenizer,
                uh_model,
                uh_device,
                batch_size=int(args.unihgkr_batch_size),
                pooling_mode=pooling_mode,
                do_normalize=do_normalize,
            )
            np.save(uq_cache, uqv)

        if uc_cache.exists() and np.load(uc_cache, mmap_mode="r").shape[0] == len(doc_texts):
            ucv = np.load(uc_cache)
        else:
            ucv = encode_texts_with_hf_encoder(
                doc_texts,
                uh_tokenizer,
                uh_model,
                uh_device,
                batch_size=int(args.unihgkr_batch_size),
                pooling_mode=pooling_mode,
                do_normalize=do_normalize,
            )
            np.save(uc_cache, ucv)

        unihgkr_scores = uqv @ ucv.T

    reader_fn: Callable[[str, list[str]], str]
    is_llm_reader = args.reader in {"ollama", "openai"}
    if args.reader == "extractive":
        print("[warn] Using extractive reader (heuristic). This mode is for smoke/debug and may under-estimate real QA quality.")
        reader_fn = lambda q, c: extractive_reader(
            q,
            c,
            numeric_consensus=bool(args.extractive_numeric_consensus),
        )
    elif args.reader == "ollama":
        reader_fn = lambda q, c: ollama_reader(args.ollama_host, args.ollama_model, q, c)
    else:
        reader_fn = lambda q, c: openai_reader(
            args.openai_model,
            q,
            c,
            timeout_s=int(args.openai_timeout),
            temperature=float(args.openai_temperature),
            max_tokens=int(args.openai_max_tokens),
            base_url=str(args.openai_base_url),
            api_key_env=str(args.openai_api_key_env),
            max_retries=int(args.openai_max_retries),
            retry_backoff_s=float(args.openai_retry_backoff),
            fail_soft=bool(args.openai_fail_soft),
        )

    router_model_path = resolve_router_model_path(args.router_model, args.router_metrics)
    y_router_pred, router_probs, router_entropy, router_source = infer_router_predictions(
        q_texts,
        router_model_dir=router_model_path,
        threshold=float(args.router_threshold),
        batch_size=int(args.router_batch_size),
        allow_heuristic_fallback=bool(args.allow_heuristic_router_fallback),
    )
    router_subset_acc_run = routing_subset_accuracy(y_router_true, y_router_pred)
    router_micro_f1_run = routing_micro_f1(y_router_true, y_router_pred)

    planner_bundle = None
    if args.planner_model is not None:
        planner_bundle = PlannerBundle.load(args.planner_model)
        print(f"[controller] loaded planner: {args.planner_model}")

    verifier_bundle = None
    if args.verifier_model is not None:
        verifier_bundle = VerifierBundle.load(args.verifier_model)
        print(f"[controller] loaded verifier: {args.verifier_model}")

    conflict_bundle = None
    if args.conflict_model is not None:
        conflict_bundle = ConflictBundle.load(args.conflict_model)
        print(f"[controller] loaded conflict scorer: {args.conflict_model}")

    tessera_ser_bundle = None
    tessera_ser_meta_summary = {}
    if args.tessera_ser_ranker is not None:
        if load_ser_ranker_bundle is None:
            raise RuntimeError("ser_ranker module is required for --tessera-ser-ranker")
        tessera_ser_bundle = load_ser_ranker_bundle(args.tessera_ser_ranker)
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
        print(f"[retrieval] loaded TESSERA-SER ranker: {args.tessera_ser_ranker}")

    tessera_slot_verifier_bundle = None
    if args.tessera_source_evidence_slot_verifier_model is not None:
        if load_pairwise_slot_verifier_bundle is None:
            raise RuntimeError(
                "pairwise_slot_verifier module is required for --tessera-source-evidence-slot-verifier-model"
            )
        tessera_slot_verifier_bundle = load_pairwise_slot_verifier_bundle(
            args.tessera_source_evidence_slot_verifier_model
        )
        print(
            f"[retrieval] loaded Pairwise Evidence Slot Verifier: "
            f"{args.tessera_source_evidence_slot_verifier_model}"
        )

    tessera_final_evidence_verifier_bundle = None
    tessera_final_evidence_verifier_meta_summary = {}
    if args.tessera_final_evidence_verifier_model is not None:
        if load_pairwise_slot_verifier_bundle is None:
            raise RuntimeError(
                "pairwise_slot_verifier module is required for --tessera-final-evidence-verifier-model"
            )
        tessera_final_evidence_verifier_bundle = load_pairwise_slot_verifier_bundle(
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
        print(
            f"[retrieval] loaded Evidence Replacement Verifier: "
            f"{args.tessera_final_evidence_verifier_model}"
        )

    tessera_kg_verifier_bundle = None
    tessera_kg_verifier_meta_summary = {}
    if args.tessera_source_evidence_kg_verifier_model is not None:
        if load_kg_consistency_bundle is None:
            raise RuntimeError(
                "kg_consistency_verifier module is required for --tessera-source-evidence-kg-verifier-model"
            )
        tessera_kg_verifier_bundle = load_kg_consistency_bundle(
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
            f"[retrieval] loaded KG Entity-Relation Consistency Verifier: "
            f"{args.tessera_source_evidence_kg_verifier_model}"
        )

    tessera_source_budgeter_bundle = None
    tessera_source_budgeter_meta_summary = {}
    if args.tessera_source_budgeter_model is not None:
        if load_source_budgeter_bundle is None:
            raise RuntimeError("source_budgeter module is required for --tessera-source-budgeter-model")
        tessera_source_budgeter_bundle = load_source_budgeter_bundle(args.tessera_source_budgeter_model)
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
        print(f"[retrieval] loaded Query-Adaptive Source Budgeter: {args.tessera_source_budgeter_model}")

    tessera_source_action_policy_bundle = None
    tessera_source_action_policy_meta_summary = {}
    if args.tessera_source_action_policy_model is not None:
        if load_source_action_policy_bundle is None:
            raise RuntimeError("source_action_policy module is required for --tessera-source-action-policy-model")
        tessera_source_action_policy_bundle = load_source_action_policy_bundle(args.tessera_source_action_policy_model)
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
        print(f"[retrieval] loaded Source Action/Utility Policy: {args.tessera_source_action_policy_model}")

    verifier_feature_version = int(verifier_bundle.metadata.get("feature_version", 1) or 1) if verifier_bundle is not None else 1
    conflict_feature_version = int(conflict_bundle.metadata.get("feature_version", 1) or 1) if conflict_bundle is not None else 1
    retry_support_source = "answer_support_score"
    if verifier_bundle is not None and verifier_feature_version >= 2:
        retry_support_source = "verifier_support_prob"

    per_method_preds: dict[str, list[str]] = {m: [] for m in selected_methods}
    per_method_context_docs: dict[str, list[list[str]]] = {m: [] for m in selected_methods}  # context doc IDs per query
    per_method_top10: dict[str, list[list[str]]] = {m: [] for m in selected_methods}
    # Per-method QA stage latency (context construction + reader call), in seconds.
    per_method_latency: dict[str, list[float]] = {m: [] for m in selected_methods}
    # Shared retrieval/ranking latency per query (applies to all methods), in seconds.
    retrieval_latency: list[float] = []
    oracle_gold_preds: list[str] = []
    oracle_gold_top10: list[list[str]] = []
    oracle_gold_latency: list[float] = []
    preloaded_preds: dict[str, list[str]] = {}
    preloaded_context_docs: dict[str, list[list[str]]] = {}
    previous_p95: dict[str, float] = {}

    prev_metrics_path = args.out_dir / "table1c_e2e_metrics.json"
    if prev_metrics_path.exists():
        try:
            prev = json.loads(prev_metrics_path.read_text(encoding="utf-8"))
            for mk, mv in prev.get("methods", {}).items():
                if isinstance(mv, dict) and mv.get("p95_latency_ms") is not None:
                    previous_p95[mk] = float(mv["p95_latency_ms"])
        except Exception:
            previous_p95 = {}

    for m in reuse_methods:
        if m not in selected_methods:
            continue
        pred_file = args.out_dir / f"qa_predictions_{m}_test1286.jsonl"
        if not pred_file.exists():
            continue
        try:
            preds = load_prediction_jsonl(pred_file)
            if len(preds) > 0:
                # Resource-efficient reuse: allow taking first-N predictions when
                # running a prefix subset of the same split order, and allow
                # prefix checkpoint resume for interrupted long API runs.
                preloaded_preds[m] = preds[: len(rows)]
                ctx_file = args.out_dir / f"context_docs_{m}_test1286.jsonl"
                if ctx_file.exists():
                    preloaded_context_docs[m] = load_context_docs_jsonl(ctx_file)[: len(rows)]
        except Exception:
            continue

    if preloaded_preds:
        loaded_counts = {m: len(v) for m, v in sorted(preloaded_preds.items())}
        print(f"[reuse] loaded predictions for methods={loaded_counts}")

    dense_uni_same_top10 = 0
    dense_uni_overlap_sum = 0.0
    schemeb_debug_trace: dict[str, list[float]] = {}
    context_debug_trace: dict[str, list[float]] = {}
    support_retry_attempted = 0
    support_retry_applied = 0
    no_evidence_retry_attempted = 0
    no_evidence_retry_applied = 0
    table_number_agent_attempted = 0
    table_number_agent_applied = 0
    consensus_refine_attempted = 0
    consensus_refine_applied = 0
    controller_planner_confidence: list[float] = []
    controller_planner_entropy: list[float] = []
    controller_planner_mix: list[float] = []
    controller_verifier_support: list[float] = []
    controller_query_modality_prior_mix: list[float] = []
    controller_context_active_threshold: list[float] = []
    controller_context_anchor_dense_k: list[float] = []
    controller_context_anchor_uni_k: list[float] = []
    controller_context_redundancy_lambda: list[float] = []
    controller_context_conflict_penalty_weight: list[float] = []
    controller_retrieval_conflict_penalty_weight: list[float] = []
    intent_complexity_level_hist: defaultdict[str, int] = defaultdict(int)
    intent_complexity_tier_hist: defaultdict[str, int] = defaultdict(int)
    intent_type_hist: defaultdict[str, int] = defaultdict(int)
    effective_context_k_trace: list[int] = []
    effective_dense_pool_k_trace: list[int] = []
    effective_candidate_expand_k_trace: list[int] = []

    start_all = time.perf_counter()
    total_queries = len(rows)
    progress_every = int(args.progress_every)
    progress_min_seconds = float(args.progress_min_seconds)
    last_progress_ts = start_all
    if total_queries > 0:
        print(
            f"[progress-config] every={progress_every} heartbeat_seconds={progress_min_seconds:.1f} total={total_queries}"
        )
    for i, row in enumerate(rows):
        t_rank0 = time.perf_counter()
        query_intent_type = infer_query_intent_type(q_texts[i])
        query_complexity_level = int(estimate_query_complexity_level(q_texts[i]))
        query_complexity_tier = resolve_intent_complexity_tier(query_complexity_level)

        effective_context_k_i = int(effective_qa_context_k)
        effective_dense_pool_k_i = int(args.context_dense_pool_k)
        effective_candidate_expand_k_i = int(args.context_candidate_expand_k)
        if bool(args.intent_complexity_aware_budgeting):
            effective_context_k_i = resolve_budget_by_tier(
                query_complexity_tier,
                int(args.intent_complexity_context_k_simple),
                int(args.intent_complexity_context_k_medium),
                int(args.intent_complexity_context_k_complex),
                minimum=1,
            )
            effective_dense_pool_k_i = resolve_budget_by_tier(
                query_complexity_tier,
                int(args.intent_complexity_dense_pool_k_simple),
                int(args.intent_complexity_dense_pool_k_medium),
                int(args.intent_complexity_dense_pool_k_complex),
                minimum=1,
            )
            effective_candidate_expand_k_i = resolve_budget_by_tier(
                query_complexity_tier,
                int(args.intent_complexity_candidate_expand_k_simple),
                int(args.intent_complexity_candidate_expand_k_medium),
                int(args.intent_complexity_candidate_expand_k_complex),
                minimum=0,
            )
            intent_complexity_level_hist[str(query_complexity_level)] += 1
            intent_complexity_tier_hist[query_complexity_tier] += 1
            intent_type_hist[query_intent_type] += 1

        effective_query_modality_prior_mix_i = float(args.query_modality_prior_mix)
        effective_context_active_threshold_i = float(args.context_active_threshold)
        effective_context_anchor_dense_k_i = int(args.context_anchor_dense_k)
        effective_context_anchor_uni_k_i = int(args.context_anchor_uni_k)
        effective_context_dense_pool_k_i = int(effective_dense_pool_k_i)
        effective_context_candidate_expand_k_i = int(effective_candidate_expand_k_i)
        effective_context_redundancy_lambda_i = float(args.context_redundancy_lambda)
        effective_context_conflict_penalty_weight_i = float(args.context_conflict_penalty_weight)
        effective_context_number_table_quota_min_i = int(args.context_number_table_quota_min)
        effective_context_light_rerank_weight_i = float(args.context_light_rerank_weight)
        effective_context_qa_objective_weight_i = float(args.context_qa_objective_weight)
        effective_context_qa_modality_bias_i = float(args.context_qa_modality_bias)
        effective_retrieval_conflict_penalty_weight_i = float(args.retrieval_conflict_penalty_weight)

        router_prob_with_planner_i = np.asarray(router_probs[i], dtype=np.float32)
        planner_info = None
        if planner_bundle is not None:
            planner_blended_router_prob_i, planner_info = planner_bundle.blend_router_prob(
                router_prob=router_prob_with_planner_i,
                query=q_texts[i],
                query_id=q_ids[i],
                base_mix=float(args.planner_mix),
                router_entropy=float(router_entropy[i]),
            )
            router_prob_with_planner_i = planner_blended_router_prob_i
            planner_prior = np.asarray(planner_info.get("modality_prior", np.full(3, 1.0 / 3.0)), dtype=np.float32)
            planner_confidence = float(planner_info.get("class_confidence", 0.0))
            planner_multi_mass = float(max(0.0, 1.0 - float(np.max(planner_prior)) if planner_prior.size else 1.0))
            planner_adjust = planner_bundle.derive_packing_adjustments(
                base_query_modality_prior_mix=effective_query_modality_prior_mix_i,
                base_context_active_threshold=effective_context_active_threshold_i,
                base_context_anchor_dense_k=effective_context_anchor_dense_k_i,
                base_context_anchor_uni_k=effective_context_anchor_uni_k_i,
                base_context_dense_pool_k=effective_context_dense_pool_k_i,
                base_context_candidate_expand_k=effective_context_candidate_expand_k_i,
                base_context_redundancy_lambda=effective_context_redundancy_lambda_i,
                base_context_conflict_penalty_weight=effective_context_conflict_penalty_weight_i,
                base_context_number_table_quota_min=effective_context_number_table_quota_min_i,
                base_context_light_rerank_weight=effective_context_light_rerank_weight_i,
                base_context_qa_objective_weight=effective_context_qa_objective_weight_i,
                base_context_qa_modality_bias=effective_context_qa_modality_bias_i,
                query=q_texts[i],
                query_id=q_ids[i],
                router_entropy=float(router_entropy[i]),
            )
            effective_query_modality_prior_mix_i = float(planner_adjust["query_modality_prior_mix"])
            effective_context_active_threshold_i = float(planner_adjust["context_active_threshold"])
            effective_context_anchor_dense_k_i = max(1, int(planner_adjust["context_anchor_dense_k"]))
            effective_context_anchor_uni_k_i = max(1, int(planner_adjust["context_anchor_uni_k"]))
            effective_context_dense_pool_k_i = max(effective_context_dense_pool_k_i, int(planner_adjust["context_dense_pool_k"]))
            effective_context_candidate_expand_k_i = max(effective_context_candidate_expand_k_i, int(planner_adjust["context_candidate_expand_k"]))
            effective_context_redundancy_lambda_i = float(planner_adjust["context_redundancy_lambda"])
            effective_context_conflict_penalty_weight_i = float(planner_adjust["context_conflict_penalty_weight"])
            effective_context_number_table_quota_min_i = int(planner_adjust["context_number_table_quota_min"])
            effective_context_light_rerank_weight_i = float(planner_adjust["context_light_rerank_weight"])
            effective_context_qa_objective_weight_i = float(planner_adjust["context_qa_objective_weight"])
            effective_context_qa_modality_bias_i = float(planner_adjust["context_qa_modality_bias"])
            effective_context_k_i = max(
                int(effective_context_k_i),
                int(round(effective_qa_context_k + 2.0 * planner_multi_mass + max(0, query_complexity_level - 3))),
            )
            effective_retrieval_conflict_penalty_weight_i = float(
                max(
                    0.0,
                    float(args.retrieval_conflict_penalty_weight)
                    * (0.85 + 0.35 * float(planner_prior[1] + planner_prior[2]) + 0.20 * (1.0 - planner_confidence)),
                )
            )
            controller_planner_confidence.append(float(planner_confidence))
            controller_planner_entropy.append(float(planner_info.get("class_entropy", 0.0)))
            controller_planner_mix.append(float(planner_info.get("mix", args.planner_mix)))

        controller_query_modality_prior_mix.append(float(effective_query_modality_prior_mix_i))
        controller_context_active_threshold.append(float(effective_context_active_threshold_i))
        controller_context_anchor_dense_k.append(float(effective_context_anchor_dense_k_i))
        controller_context_anchor_uni_k.append(float(effective_context_anchor_uni_k_i))
        controller_context_redundancy_lambda.append(float(effective_context_redundancy_lambda_i))
        controller_context_conflict_penalty_weight.append(float(effective_context_conflict_penalty_weight_i))
        controller_retrieval_conflict_penalty_weight.append(float(effective_retrieval_conflict_penalty_weight_i))

        effective_context_k_trace.append(int(effective_context_k_i))
        effective_dense_pool_k_trace.append(int(effective_dense_pool_k_i))
        effective_candidate_expand_k_trace.append(int(effective_candidate_expand_k_i))

        if bool(args.query_modality_prior_adaptive):
            blended_router_prob_i, _, _, _ = blend_router_with_query_prior_adaptive(
                router_prob=router_prob_with_planner_i,
                query=q_texts[i],
                query_id=q_ids[i],
                prior_mix=float(effective_query_modality_prior_mix_i),
                router_entropy=float(router_entropy[i]),
                uncertainty_threshold=float(args.routing_uncertainty_threshold),
                entropy_scale=float(args.query_modality_prior_entropy_scale),
                disagreement_scale=float(args.query_modality_prior_disagreement_scale),
                mix_min=float(args.query_modality_prior_min),
                mix_max=float(args.query_modality_prior_max),
            )
        else:
            blended_router_prob_i, _, _, _ = blend_router_with_query_prior(
                router_prob=router_prob_with_planner_i,
                query=q_texts[i],
                query_id=q_ids[i],
                prior_mix=float(effective_query_modality_prior_mix_i),
            )
        rankings = build_rankings_for_query(
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
            table_tapas_cache,
            kg_gnn_cache,
            tapas_bundle,
            retrieve_topk=args.retrieve_topk,
            preserve_dense_top=args.preserve_dense_top,
            tessera_late_alpha=args.tessera_late_alpha,
            router_prob=blended_router_prob_i,
            query_modality_prior_mix=float(effective_query_modality_prior_mix_i),
            query_modality_prior_adaptive=bool(args.query_modality_prior_adaptive),
            query_modality_prior_entropy_scale=float(args.query_modality_prior_entropy_scale),
            query_modality_prior_disagreement_scale=float(args.query_modality_prior_disagreement_scale),
            query_modality_prior_min=float(args.query_modality_prior_min),
            query_modality_prior_max=float(args.query_modality_prior_max),
            router_entropy=float(router_entropy[i]),
            uncertainty_threshold=float(args.routing_uncertainty_threshold),
            pathmaxsim_weight=float(args.pathmaxsim_weight),
            pathmaxsim_kg_threshold=float(args.pathmaxsim_kg_threshold),
            table_cellmaxsim_weight=float(args.table_cellmaxsim_weight),
            table_cellmaxsim_top_cells=int(args.table_cellmaxsim_top_cells),
            innovation_scheme2=bool(args.innovation_scheme2),
            scheme2_cross_modal_weight=float(args.scheme2_cross_modal_weight),
            scheme2_token_maxsim_weight=float(args.scheme2_token_maxsim_weight),
            adapter_plus_mode=bool(args.adapter_plus_mode),
            adapter_official_lite=bool(args.adapter_official_lite),
            heavy_schemeb_mode=bool(args.schemeb_heavy_mode),
            heavy_table_encoder_weight=float(args.heavy_table_encoder_weight),
            heavy_kg_path_weight=float(args.heavy_kg_path_weight),
            heavy_token_late_weight=float(args.heavy_token_late_weight),
            heavy_query_max_tokens=int(args.heavy_query_max_tokens),
            heavy_table_max_cells=int(args.heavy_table_max_cells),
            heavy_token_doc_max_tokens=int(args.heavy_token_doc_max_tokens),
            heavy_table_backend=str(heavy_table_backend_effective),
            heavy_table_tapas_topn=int(args.heavy_table_tapas_topn),
            heavy_table_max_rows=int(args.heavy_table_max_rows),
            heavy_table_max_cols=int(args.heavy_table_max_cols),
            heavy_table_agg_cell_logit=float(args.heavy_table_agg_cell_logit),
            heavy_table_agg_row_logit=float(args.heavy_table_agg_row_logit),
            heavy_table_agg_col_logit=float(args.heavy_table_agg_col_logit),
            heavy_table_agg_temp=float(args.heavy_table_agg_temp),
            heavy_kg_backend=str(args.heavy_kg_backend),
            heavy_kg_gnn_topn=int(args.heavy_kg_gnn_topn),
            heavy_kg_max_hops=int(args.heavy_kg_max_hops),
            heavy_kg_max_paths=int(args.heavy_kg_max_paths),
            heavy_kg_contrastive_temp=float(args.heavy_kg_contrastive_temp),
            heavy_kg_hard_negative_mode=str(args.heavy_kg_hard_negative_mode),
            heavy_kg_hard_negative_topdocs=int(args.heavy_kg_hard_negative_topdocs),
            heavy_kg_hard_negative_max_paths=int(args.heavy_kg_hard_negative_max_paths),
            heavy_token_cross_modal_weight=float(args.heavy_token_cross_modal_weight),
            heavy_branch_candidate_expand_k=int(args.heavy_branch_candidate_expand_k),
            heavy_branch_candidate_table_weight=float(args.heavy_branch_candidate_table_weight),
            heavy_branch_candidate_kg_weight=float(args.heavy_branch_candidate_kg_weight),
            heavy_branch_candidate_max_total=int(args.heavy_branch_candidate_max_total),
            heavy_score_calibration=str(args.heavy_score_calibration),
            heavy_score_calibration_nonzero_only=bool(args.heavy_score_calibration_nonzero_only),
            qa_objective_retrieval_weight=float(args.qa_objective_retrieval_weight),
            qa_objective_targeted_only=bool(args.qa_objective_targeted_only),
            upo_lite_retrieval_weight=float(args.upo_lite_retrieval_weight),
            upo_lite_targeted_only=bool(args.upo_lite_targeted_only),
            retrieval_conflict_penalty_weight=float(effective_retrieval_conflict_penalty_weight_i),
            retrieval_conflict_targeted_only=bool(args.retrieval_conflict_targeted_only),
            retrieval_conflict_table_kg_only=bool(args.retrieval_conflict_table_kg_only),
            retrieval_conflict_risk_gating=bool(args.retrieval_conflict_risk_gating),
            retrieval_conflict_risk_low=float(args.retrieval_conflict_risk_low),
            retrieval_conflict_risk_high=float(args.retrieval_conflict_risk_high),
            retrieval_conflict_risk_probe_k=int(args.retrieval_conflict_risk_probe_k),
            retrieval_conflict_anchor_k=int(args.retrieval_conflict_anchor_k),
            retrieval_conflict_max_literals_per_doc=int(args.retrieval_conflict_max_literals_per_doc),
            retrieval_conflict_sensitive_target_scale=float(args.retrieval_conflict_sensitive_target_scale),
            conflict_bundle=conflict_bundle,
            selected_methods=set(selected_methods),
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
            debug_trace=schemeb_debug_trace,
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
            tessera_source_evidence_kg_verify_existing=bool(args.tessera_source_evidence_kg_verify_existing),
            tessera_source_evidence_kg_verify_existing_max_replacements=int(args.tessera_source_evidence_kg_verify_existing_max_replacements),
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
        if unihgkr_scores is not None:
            rankings["unihgkr_dense"] = topk_indices(unihgkr_scores[i], args.retrieve_topk).tolist()
        retrieval_latency.append(time.perf_counter() - t_rank0)

        if "dense_concat" in rankings and "tessera_rag" in rankings:
            dense10 = rankings["dense_concat"][:10]
            uni10 = rankings["tessera_rag"][:10]
            if dense10 == uni10:
                dense_uni_same_top10 += 1
            dense_uni_overlap_sum += len(set(dense10) & set(uni10)) / 10.0

        for m in selected_methods:
            if m in preloaded_preds and i < len(preloaded_preds[m]):
                idxs = rankings[m]
                top10_ids = [doc_ids[j] for j in idxs[:10]]
                per_method_top10[m].append(top10_ids)
                per_method_preds[m].append(preloaded_preds[m][i])
                if m in preloaded_context_docs and i < len(preloaded_context_docs[m]):
                    per_method_context_docs[m].append(preloaded_context_docs[m][i])
                else:
                    per_method_context_docs[m].append([])
                continue
            t0 = time.perf_counter()
            idxs = rankings[m]
            top10_ids = [doc_ids[j] for j in idxs[:10]]
            per_method_top10[m].append(top10_ids)
            if m == "tessera_submod" and submodular_t2g_packer is not None:
                ctx_idxs = submodular_t2g_packer(
                    candidate_idxs=idxs,
                    candidate_texts=[doc_texts[j] for j in idxs],
                    candidate_doc_ids=[doc_ids[j] for j in idxs],
                    query=q_texts[i],
                    router_prob=blended_router_prob_i,
                    router_entropy=float(router_entropy[i]),
                    k=effective_context_k_i,
                    dense_anchor_idxs=rankings.get("dense_concat", [])[: int(effective_context_anchor_dense_k_i)],
                    budget_mode="token_estimate",
                )
            elif m in {"tessera_rag", "tessera_submod", "ablation_no_redundancy_e2e", "ablation_no_pathmaxsim_e2e"}:
                method_redundancy_lambda = float(args.context_redundancy_lambda)
                method_hard_red_filter = True
                if m == "ablation_no_redundancy_e2e":
                    method_redundancy_lambda = 0.0
                    method_hard_red_filter = False
                dense_context_pool = rankings.get("dense_context_pool", rankings["dense_concat"])
                dense_pool_k = max(
                    int(effective_dense_pool_k_i),
                    int(effective_context_anchor_dense_k_i),
                    int(effective_context_candidate_expand_k_i),
                )
                ctx_idxs = select_modality_adaptive_context(
                    ranked_idxs=idxs,
                    dense_ranked_idxs=dense_context_pool[:dense_pool_k],
                    query=q_texts[i],
                    doc_ids=doc_ids,
                    doc_texts=doc_texts,
                    doc_tokens=doc_tokens,
                    doc_signal_tokens=doc_signal_tokens,
                    doc_numeric_literals=doc_numeric_literals,
                    router_prob=blended_router_prob_i,
                    router_entropy=float(router_entropy[i]),
                    k=effective_context_k_i,
                    active_threshold=float(effective_context_active_threshold_i),
                    anchor_dense_k=int(effective_context_anchor_dense_k_i),
                    anchor_uni_k=int(effective_context_anchor_uni_k_i),
                    context_candidate_expand_k=int(effective_context_candidate_expand_k_i),
                    redundancy_lambda=float(effective_context_redundancy_lambda_i if m != "ablation_no_redundancy_e2e" else 0.0),
                    enable_hard_redundancy_filter=method_hard_red_filter,
                    consistency_weight=float(args.context_consistency_weight),
                    context_conflict_penalty_weight=float(effective_context_conflict_penalty_weight_i),
                    context_conflict_targeted_only=bool(args.context_conflict_targeted_only),
                    context_conflict_table_kg_only=bool(args.context_conflict_table_kg_only),
                    context_conflict_risk_gating=bool(args.context_conflict_risk_gating),
                    context_conflict_risk_low=float(args.context_conflict_risk_low),
                    context_conflict_risk_high=float(args.context_conflict_risk_high),
                    context_conflict_risk_probe_k=int(args.context_conflict_risk_probe_k),
                    context_conflict_sensitive_target_scale=float(args.context_conflict_sensitive_target_scale),
                    context_conflict_max_literals_per_doc=int(args.context_conflict_max_literals_per_doc),
                    conflict_bundle=conflict_bundle,
                    context_number_table_quota_min=int(effective_context_number_table_quota_min_i),
                    context_subquery_coverage_weight=float(args.context_subquery_coverage_weight),
                    context_light_rerank_weight=float(effective_context_light_rerank_weight_i),
                    context_light_rerank_topn=int(args.context_light_rerank_topn),
                    context_light_rerank_targeted_only=bool(args.context_light_rerank_targeted_only),
                    context_strong_rerank_endpoint=str(args.context_strong_rerank_endpoint),
                    context_strong_rerank_topn=int(args.context_strong_rerank_topn),
                    context_strong_rerank_timeout=int(args.context_strong_rerank_timeout),
                    context_upo_lite_quota_mix=float(args.context_upo_lite_quota_mix),
                    context_upo_lite_rerank_bonus=float(args.context_upo_lite_rerank_bonus),
                    context_qa_objective_weight=float(effective_context_qa_objective_weight_i),
                    context_qa_modality_bias=float(effective_context_qa_modality_bias_i),
                    debug_trace=context_debug_trace,
                )
            else:
                ctx_idxs = idxs[:effective_context_k_i]
            if m == "tessera_rag" and bool(args.tessera_policy_context):
                if select_tessera_policy_context is None or TESSERAPolicyConfig is None:
                    raise RuntimeError("tessera_policy module is required for --tessera-policy-context")
                policy_cfg = TESSERAPolicyConfig(
                    candidate_pool_k=int(args.tessera_policy_pool_k),
                    dense_pool_k=int(args.tessera_policy_dense_pool_k),
                    target_weight=float(args.tessera_policy_target_weight),
                    coverage_weight=float(args.tessera_policy_coverage_weight),
                    diversity_weight=float(args.tessera_policy_diversity_weight),
                )
                ctx_idxs, policy_trace = select_tessera_policy_context(
                    query=q_texts[i],
                    base_ctx_idxs=ctx_idxs,
                    ranked_idxs=idxs,
                    dense_ranked_idxs=rankings.get("dense_context_pool", rankings.get("dense_concat", [])),
                    doc_ids=doc_ids,
                    doc_texts=doc_texts,
                    doc_tokens=doc_tokens,
                    doc_numeric_literals=doc_numeric_literals,
                    router_prob=blended_router_prob_i,
                    router_entropy=float(router_entropy[i]),
                    k=effective_context_k_i,
                    target_type=infer_qa_target_type(q_texts[i]),
                    upo_concept=infer_upo_lite_concept(q_texts[i]),
                    source_bucket_fn=source_bucket,
                    config=policy_cfg,
                )
                context_debug_trace.setdefault("tessera_policy_replaced_count", []).append(float(policy_trace.replaced_count))
                context_debug_trace.setdefault("tessera_policy_forced_hits", []).append(float(policy_trace.forced_hits))
                context_debug_trace.setdefault("tessera_policy_coverage", []).append(float(policy_trace.coverage))
            ctx = [doc_texts[j] for j in ctx_idxs]
            ctx_doc_ids = [doc_ids[j] for j in ctx_idxs]
            pred = reader_fn(q_texts[i], ctx)
            if (
                args.enable_tessera_answer_type_guard
                and is_llm_reader
                and m in {"tessera_rag", "tessera_submod", "ablation_no_redundancy_e2e", "ablation_no_pathmaxsim_e2e"}
            ):
                pred = apply_tessera_answer_type_guard(
                    q_texts[i],
                    pred,
                    ctx,
                    extractive_numeric_consensus=bool(args.extractive_numeric_consensus),
                )
            if (
                args.enable_tessera_answer_calibration
                and is_llm_reader
                and m in {"tessera_rag", "tessera_submod", "ablation_no_redundancy_e2e", "ablation_no_pathmaxsim_e2e"}
            ):
                pred = calibrate_tessera_answer(
                    q_texts[i],
                    pred,
                    ctx,
                    extractive_numeric_consensus=bool(args.extractive_numeric_consensus),
                )

            if (
                bool(args.tessera_no_evidence_retry)
                and is_llm_reader
                and m in {"tessera_rag", "tessera_submod", "ablation_no_redundancy_e2e", "ablation_no_pathmaxsim_e2e"}
            ):
                if (
                    is_no_evidence_answer is None
                    or select_tessera_retry_context is None
                    or TESSERARetryAgentConfig is None
                ):
                    raise RuntimeError("tessera_policy module is required for --tessera-no-evidence-retry")
                base_answer_support0 = answer_support_score(pred, ctx)
                no_evidence_answer = bool(is_no_evidence_answer(pred))
                low_support_answer = (
                    float(args.tessera_no_evidence_retry_low_support_threshold) >= 0.0
                    and base_answer_support0 < float(args.tessera_no_evidence_retry_low_support_threshold)
                )
                context_debug_trace.setdefault("tessera_no_evidence_retry_no_evidence", []).append(
                    float(no_evidence_answer)
                )
                context_debug_trace.setdefault("tessera_no_evidence_retry_base_support", []).append(
                    float(base_answer_support0)
                )
                if no_evidence_answer or low_support_answer:
                    no_evidence_retry_attempted += 1
                    target_type_retry = str(infer_qa_target_type(q_texts[i])).strip().lower()
                    retry_cfg = TESSERARetryAgentConfig(
                        candidate_pool_k=int(args.tessera_no_evidence_retry_pool_k),
                        dense_pool_k=int(args.tessera_no_evidence_retry_dense_pool_k),
                    )
                    retry_k = max(
                        int(effective_context_k_i),
                        int(args.tessera_no_evidence_retry_context_k),
                    )
                    retry_ctx_idxs, retry_agent_trace = select_tessera_retry_context(
                        query=q_texts[i],
                        current_answer=pred,
                        base_ctx_idxs=ctx_idxs,
                        ranked_idxs=idxs,
                        dense_ranked_idxs=rankings.get("dense_context_pool", rankings.get("dense_concat", [])),
                        doc_ids=doc_ids,
                        doc_texts=doc_texts,
                        doc_tokens=doc_tokens,
                        doc_numeric_literals=doc_numeric_literals,
                        router_prob=blended_router_prob_i,
                        router_entropy=float(router_entropy[i]),
                        k=retry_k,
                        target_type=target_type_retry,
                        upo_concept=infer_upo_lite_concept(q_texts[i]),
                        source_bucket_fn=source_bucket,
                        config=retry_cfg,
                    )
                    context_debug_trace.setdefault("tessera_no_evidence_retry_context_k", []).append(
                        float(len(retry_ctx_idxs))
                    )
                    context_debug_trace.setdefault("tessera_no_evidence_retry_coverage", []).append(
                        float(retry_agent_trace.coverage)
                    )
                    if retry_ctx_idxs and retry_ctx_idxs != ctx_idxs:
                        retry_ctx = [doc_texts[j] for j in retry_ctx_idxs]
                        retry_ctx_doc_ids = [doc_ids[j] for j in retry_ctx_idxs]
                        retry_pred = reader_fn(q_texts[i], retry_ctx)
                        if args.enable_tessera_answer_type_guard and is_llm_reader:
                            retry_pred = apply_tessera_answer_type_guard(
                                q_texts[i],
                                retry_pred,
                                retry_ctx,
                                extractive_numeric_consensus=bool(args.extractive_numeric_consensus),
                            )
                        if args.enable_tessera_answer_calibration and is_llm_reader:
                            retry_pred = calibrate_tessera_answer(
                                q_texts[i],
                                retry_pred,
                                retry_ctx,
                                extractive_numeric_consensus=bool(args.extractive_numeric_consensus),
                            )
                        retry_answer_support0 = answer_support_score(retry_pred, retry_ctx)
                        retry_no_evidence = bool(is_no_evidence_answer(retry_pred))
                        accept_retry0 = False
                        if no_evidence_answer:
                            accept_retry0 = (
                                not retry_no_evidence
                                and retry_answer_support0 >= float(args.tessera_no_evidence_retry_min_support)
                            )
                        if low_support_answer:
                            accept_retry0 = accept_retry0 or (
                                not retry_no_evidence
                                and retry_answer_support0 >= base_answer_support0 + float(args.tessera_no_evidence_retry_margin)
                            )
                        context_debug_trace.setdefault("tessera_no_evidence_retry_retry_support", []).append(
                            float(retry_answer_support0)
                        )
                        context_debug_trace.setdefault("tessera_no_evidence_retry_accept", []).append(
                            float(accept_retry0)
                        )
                        if accept_retry0:
                            pred = retry_pred
                            ctx_idxs = retry_ctx_idxs
                            ctx = retry_ctx
                            ctx_doc_ids = retry_ctx_doc_ids
                            no_evidence_retry_applied += 1

            if (
                bool(args.tessera_table_number_agent)
                and is_llm_reader
                and m in {"tessera_rag", "tessera_submod", "ablation_no_redundancy_e2e", "ablation_no_pathmaxsim_e2e"}
            ):
                pred, table_number_applied, table_number_debug = maybe_apply_tessera_table_number_agent(
                    q_texts[i],
                    pred,
                    ctx,
                    ctx_doc_ids,
                    args,
                )
                if table_number_debug.get("attempted", 0.0) > 0.0:
                    table_number_agent_attempted += 1
                    for tn_key, tn_value in table_number_debug.items():
                        context_debug_trace.setdefault(f"tessera_table_number_agent_{tn_key}", []).append(
                            float(tn_value)
                        )
                if table_number_applied:
                    table_number_agent_applied += 1

            retry_support_source = "answer_support_score"
            refine_support_source = "answer_support_score"
            if verifier_bundle is not None and verifier_feature_version >= 2:
                retry_support_source = "verifier_support_prob"
                refine_support_source = "verifier_support_prob"
            retry_acceptance_source = "hybrid_answer_support_or_verifier_support"

            verifier_support_prob = None
            conflict_probability = None
            retry_controls = None
            if verifier_bundle is not None and is_llm_reader and m in {"tessera_rag", "tessera_submod", "ablation_no_redundancy_e2e", "ablation_no_pathmaxsim_e2e"}:
                verifier_support_prob, _ = verifier_bundle.predict_support_probability(
                    q_texts[i],
                    pred,
                    ctx,
                    doc_ids=ctx_doc_ids,
                )
                if conflict_bundle is not None:
                    conflict_probability, _ = conflict_bundle.predict_conflict_probability(
                        q_texts[i],
                        ctx,
                        doc_ids=ctx_doc_ids,
                        table_kg_only=bool(args.context_conflict_table_kg_only),
                        probe_k=int(args.context_conflict_risk_probe_k),
                        max_literals_per_doc=int(args.context_conflict_max_literals_per_doc),
                    )
                retry_controls = verifier_bundle.derive_retry_controls(
                    query=q_texts[i],
                    answer=pred,
                    contexts=ctx,
                    doc_ids=ctx_doc_ids,
                    router_entropy=float(router_entropy[i]),
                    base_support_retry_threshold=float(args.tessera_support_retry_threshold),
                    base_support_retry_margin=float(args.tessera_support_retry_margin),
                    base_support_retry_mode=str(args.tessera_support_retry_mode),
                    base_consensus_support_threshold=float(args.tessera_consensus_refine_support_threshold),
                    conflict_probability=conflict_probability,
                )
                controller_verifier_support.append(float(verifier_support_prob))

            if (
                float(args.tessera_support_retry_threshold) >= 0.0
                and is_llm_reader
                and m in {"tessera_rag", "tessera_submod", "ablation_no_redundancy_e2e", "ablation_no_pathmaxsim_e2e"}
            ):
                target_type = str(infer_qa_target_type(q_texts[i])).strip().lower()
                allow_retry = True

                support_retry_threshold_eff = float(args.tessera_support_retry_threshold)
                support_retry_margin_eff = float(args.tessera_support_retry_margin)
                support_retry_mode_eff = str(args.tessera_support_retry_mode)
                if retry_controls is not None:
                    support_retry_threshold_eff = float(retry_controls["support_retry_threshold"])
                    support_retry_margin_eff = float(retry_controls["support_retry_margin"])
                    support_retry_mode_eff = str(retry_controls["support_retry_mode"])

                if bool(args.tessera_support_retry_targeted_only):
                    if target_type not in {"number", "year", "location"}:
                        allow_retry = False
                if allow_retry and int(args.tessera_support_retry_complexity_min) > 0:
                    if int(query_complexity_level) < int(args.tessera_support_retry_complexity_min):
                        allow_retry = False
                if allow_retry and float(args.tessera_support_retry_entropy_min) >= 0.0:
                    if float(router_entropy[i]) < float(args.tessera_support_retry_entropy_min):
                        allow_retry = False

                if allow_retry:
                    base_answer_support = answer_support_score(pred, ctx)
                    base_support = base_answer_support
                    if retry_support_source == "verifier_support_prob" and verifier_support_prob is not None:
                        base_support = float(verifier_support_prob)
                    if base_support < support_retry_threshold_eff:
                        support_retry_attempted += 1
                        retry_mode = str(support_retry_mode_eff).strip().lower()
                        if retry_mode == "evidence_chain":
                            retry_ctx_idxs = build_evidence_chain_retry_context(
                                query=q_texts[i],
                                current_answer=pred,
                                current_ctx_idxs=ctx_idxs,
                                dense_ranked_idxs=rankings.get("dense_concat", []),
                                doc_ids=doc_ids,
                                doc_tokens=doc_tokens,
                                k=effective_context_k_i,
                                pool_k=int(args.tessera_support_retry_pool_k),
                                answer_boost=float(args.tessera_support_retry_answer_boost),
                                target_type=target_type,
                            )
                        else:
                            retry_ctx_idxs = rankings.get("dense_concat", [])[:effective_context_k_i]
                        if retry_ctx_idxs and retry_ctx_idxs != ctx_idxs:
                            retry_ctx = [doc_texts[j] for j in retry_ctx_idxs]
                            retry_ctx_doc_ids = [doc_ids[j] for j in retry_ctx_idxs]
                            retry_pred = reader_fn(q_texts[i], retry_ctx)
                            if args.enable_tessera_answer_type_guard and is_llm_reader:
                                retry_pred = apply_tessera_answer_type_guard(
                                    q_texts[i],
                                    retry_pred,
                                    retry_ctx,
                                    extractive_numeric_consensus=bool(args.extractive_numeric_consensus),
                                )
                            if args.enable_tessera_answer_calibration and is_llm_reader:
                                retry_pred = calibrate_tessera_answer(
                                    q_texts[i],
                                    retry_pred,
                                    retry_ctx,
                                    extractive_numeric_consensus=bool(args.extractive_numeric_consensus),
                                )

                            retry_answer_support = answer_support_score(retry_pred, retry_ctx)
                            retry_support = retry_answer_support
                            if retry_support_source == "verifier_support_prob" and verifier_bundle is not None:
                                retry_support, _ = verifier_bundle.predict_support_probability(
                                    q_texts[i],
                                    retry_pred,
                                    retry_ctx,
                                    doc_ids=retry_ctx_doc_ids,
                                )
                            accept_retry = retry_answer_support >= base_answer_support + support_retry_margin_eff
                            if retry_support_source == "verifier_support_prob" and verifier_bundle is not None:
                                accept_retry = accept_retry or retry_support >= base_support + support_retry_margin_eff
                            if accept_retry:
                                pred = retry_pred
                                ctx_idxs = retry_ctx_idxs
                                ctx = retry_ctx
                                ctx_doc_ids = retry_ctx_doc_ids
                                support_retry_applied += 1

            if (
                args.enable_tessera_consensus_refine
                and is_llm_reader
                and m in {"tessera_rag", "tessera_submod", "ablation_no_redundancy_e2e", "ablation_no_pathmaxsim_e2e"}
            ):
                allow_refine = True
                consensus_support_threshold_eff = float(args.tessera_consensus_refine_support_threshold)
                if retry_controls is not None:
                    consensus_support_threshold_eff = float(retry_controls["consensus_support_threshold"])
                if int(args.tessera_consensus_refine_complexity_min) > 0:
                    if int(query_complexity_level) < int(args.tessera_consensus_refine_complexity_min):
                        allow_refine = False
                if allow_refine and float(args.tessera_consensus_refine_entropy_min) >= 0.0:
                    if float(router_entropy[i]) < float(args.tessera_consensus_refine_entropy_min):
                        allow_refine = False
                if allow_refine and consensus_support_threshold_eff >= 0.0:
                    current_support = answer_support_score(pred, ctx)
                    if refine_support_source == "verifier_support_prob" and verifier_support_prob is not None:
                        current_support = float(verifier_support_prob)
                    if current_support >= consensus_support_threshold_eff:
                        allow_refine = False

                if allow_refine:
                    pred, refine_attempted, refine_applied = refine_tessera_answer_with_consensus(
                        q_texts[i],
                        pred,
                        ctx,
                        extractive_numeric_consensus=bool(args.extractive_numeric_consensus),
                        min_gain=float(args.tessera_consensus_refine_min_gain),
                        targeted_only=bool(args.tessera_consensus_refine_targeted_only),
                    )
                    if refine_attempted:
                        consensus_refine_attempted += 1
                    if refine_applied:
                        consensus_refine_applied += 1

            per_method_preds[m].append(pred)
            per_method_context_docs[m].append(ctx_doc_ids)  # save selected context doc IDs
            per_method_latency[m].append(time.perf_counter() - t0)
            append_prediction_jsonl(args.out_dir / f"qa_predictions_{m}_test1286.jsonl", row, pred)
            append_context_docs_jsonl(args.out_dir / f"context_docs_{m}_test1286.jsonl", row, ctx_doc_ids)

        if args.include_oracle_measured_row:
            t_oracle0 = time.perf_counter()
            rel_items: list[tuple[str, float]] = []
            for chunk_id, label in row.get("relevant_chunks", {}).items():
                if chunk_id not in doc_id_to_idx:
                    continue
                try:
                    y = float(label)
                except Exception:
                    continue
                if y > 0:
                    rel_items.append((chunk_id, y))
            rel_items.sort(key=lambda x: (-x[1], x[0]))
            gold_doc_ids = [cid for cid, _ in rel_items]
            oracle_gold_top10.append(gold_doc_ids[:10])
            ctx_limit = max(1, int(effective_context_k_i))
            oracle_ctx = [doc_texts[doc_id_to_idx[cid]] for cid in gold_doc_ids[:ctx_limit]]
            oracle_pred = reader_fn(q_texts[i], oracle_ctx)
            oracle_gold_preds.append(oracle_pred)
            oracle_gold_latency.append(time.perf_counter() - t_oracle0)

        done_queries = i + 1
        now_ts = time.perf_counter()
        should_print_progress = False
        if progress_every > 0 and (done_queries % progress_every == 0 or done_queries == total_queries):
            should_print_progress = True
        if not should_print_progress and progress_min_seconds > 0.0 and (now_ts - last_progress_ts) >= progress_min_seconds:
            should_print_progress = True
        if should_print_progress:
            elapsed_s = now_ts - start_all
            avg_s = elapsed_s / max(1, done_queries)
            eta_s = max(0.0, (total_queries - done_queries) * avg_s)
            progress_pct = 100.0 * done_queries / max(1, total_queries)
            qps = done_queries / elapsed_s if elapsed_s > 0.0 else 0.0
            print(
                f"[progress] {done_queries}/{total_queries} ({progress_pct:.1f}%) "
                f"elapsed={format_duration(elapsed_s)} avg={avg_s:.2f}s/q "
                f"eta={format_duration(eta_s)} qps={qps:.3f}"
            )
            last_progress_ts = now_ts

    elapsed = time.perf_counter() - start_all
    print(f"[done] total_seconds={elapsed:.2f}")

    results = {
        "meta": {
            "queries": len(rows),
            "corpus": len(corpus),
            "split_file": str(args.split_file),
            "corpus_file": str(args.corpus_file),
            "progress_every": int(args.progress_every),
            "progress_min_seconds": float(args.progress_min_seconds),
            "qrels_positive_total": int(qrels_positive_total),
            "qrels_positive_in_corpus": int(qrels_positive_in_corpus),
            "qrels_coverage_in_corpus": float(qrels_coverage_in_corpus),
            "queries_with_missing_qrels_in_corpus": int(queries_with_missing_qrels_in_corpus),
            "likely_transductive_test_corpus": bool(
                qrels_coverage_in_corpus >= 0.99 and "test" in str(args.split_file).lower()
            ),
            "reader": args.reader,
            "ollama_model": args.ollama_model if args.reader == "ollama" else None,
            "openai_model": args.openai_model if args.reader == "openai" else None,
            "openai_base_url": args.openai_base_url if args.reader == "openai" else None,
            "openai_timeout": int(args.openai_timeout) if args.reader == "openai" else None,
            "openai_max_retries": int(args.openai_max_retries) if args.reader == "openai" else None,
            "openai_retry_backoff": float(args.openai_retry_backoff) if args.reader == "openai" else None,
            "openai_fail_soft": bool(args.openai_fail_soft) if args.reader == "openai" else None,
            "retrieve_topk": args.retrieve_topk,
            "qa_context_k": effective_qa_context_k,
            "qa_context_k_input": int(args.qa_context_k),
            "official_mmrag_mode": bool(args.official_mmrag_mode),
            "protocol_profile": "mmrag_official_main_aligned" if bool(args.official_mmrag_mode) else "custom",
            "preserve_dense_top": args.preserve_dense_top,
            "tessera_late_alpha": args.tessera_late_alpha,
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
            "tessera_source_evidence_kg_verify_existing": bool(args.tessera_source_evidence_kg_verify_existing),
            "tessera_source_evidence_kg_verify_existing_max_replacements": int(
                args.tessera_source_evidence_kg_verify_existing_max_replacements
            ),
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
            "routing_uncertainty_threshold": args.routing_uncertainty_threshold,
            "planner_model": str(args.planner_model) if args.planner_model is not None else None,
            "planner_mix": float(args.planner_mix),
            "verifier_model": str(args.verifier_model) if args.verifier_model is not None else None,
            "conflict_model": str(args.conflict_model) if args.conflict_model is not None else None,
            "controller_verifier_feature_version": int(verifier_feature_version) if verifier_bundle is not None else None,
            "controller_conflict_feature_version": int(conflict_feature_version) if conflict_bundle is not None else None,
            "controller_retry_support_source": retry_support_source,
            "controller_retry_acceptance_source": retry_acceptance_source,
            "controller_refine_support_source": refine_support_source,
            "controller_planner_loaded": bool(planner_bundle is not None),
            "controller_verifier_loaded": bool(verifier_bundle is not None),
            "controller_conflict_loaded": bool(conflict_bundle is not None),
            "controller_planner_confidence_mean": float(np.mean(controller_planner_confidence))
            if controller_planner_confidence
            else None,
            "controller_planner_entropy_mean": float(np.mean(controller_planner_entropy))
            if controller_planner_entropy
            else None,
            "controller_planner_mix_mean": float(np.mean(controller_planner_mix))
            if controller_planner_mix
            else None,
            "controller_verifier_support_mean": float(np.mean(controller_verifier_support))
            if controller_verifier_support
            else None,
            "controller_query_modality_prior_mix_mean": float(np.mean(controller_query_modality_prior_mix))
            if controller_query_modality_prior_mix
            else float(args.query_modality_prior_mix),
            "controller_context_active_threshold_mean": float(np.mean(controller_context_active_threshold))
            if controller_context_active_threshold
            else float(args.context_active_threshold),
            "controller_context_anchor_dense_k_mean": float(np.mean(controller_context_anchor_dense_k))
            if controller_context_anchor_dense_k
            else float(args.context_anchor_dense_k),
            "controller_context_anchor_uni_k_mean": float(np.mean(controller_context_anchor_uni_k))
            if controller_context_anchor_uni_k
            else float(args.context_anchor_uni_k),
            "controller_context_redundancy_lambda_mean": float(np.mean(controller_context_redundancy_lambda))
            if controller_context_redundancy_lambda
            else float(args.context_redundancy_lambda),
            "controller_context_conflict_penalty_weight_mean": float(np.mean(controller_context_conflict_penalty_weight))
            if controller_context_conflict_penalty_weight
            else float(args.context_conflict_penalty_weight),
            "controller_retrieval_conflict_penalty_weight_mean": float(np.mean(controller_retrieval_conflict_penalty_weight))
            if controller_retrieval_conflict_penalty_weight
            else float(args.retrieval_conflict_penalty_weight),
            "query_modality_prior_mix": float(args.query_modality_prior_mix),
            "query_modality_prior_adaptive": bool(args.query_modality_prior_adaptive),
            "query_modality_prior_entropy_scale": float(args.query_modality_prior_entropy_scale),
            "query_modality_prior_disagreement_scale": float(args.query_modality_prior_disagreement_scale),
            "query_modality_prior_min": float(args.query_modality_prior_min),
            "query_modality_prior_max": float(args.query_modality_prior_max),
            "upo_lite_retrieval_weight": float(args.upo_lite_retrieval_weight),
            "upo_lite_targeted_only": bool(args.upo_lite_targeted_only),
            "retrieval_conflict_penalty_weight": float(args.retrieval_conflict_penalty_weight),
            "retrieval_conflict_targeted_only": bool(args.retrieval_conflict_targeted_only),
            "retrieval_conflict_table_kg_only": bool(args.retrieval_conflict_table_kg_only),
            "retrieval_conflict_risk_gating": bool(args.retrieval_conflict_risk_gating),
            "retrieval_conflict_risk_low": float(args.retrieval_conflict_risk_low),
            "retrieval_conflict_risk_high": float(args.retrieval_conflict_risk_high),
            "retrieval_conflict_risk_probe_k": int(args.retrieval_conflict_risk_probe_k),
            "retrieval_conflict_anchor_k": int(args.retrieval_conflict_anchor_k),
            "retrieval_conflict_max_literals_per_doc": int(args.retrieval_conflict_max_literals_per_doc),
            "retrieval_conflict_sensitive_target_scale": float(args.retrieval_conflict_sensitive_target_scale),
            "context_router_prob_mode": "adaptive_blended"
            if bool(args.query_modality_prior_adaptive)
            else ("blended" if float(args.query_modality_prior_mix) > 1e-9 else "router_only"),
            "context_active_threshold": args.context_active_threshold,
            "context_anchor_dense_k": args.context_anchor_dense_k,
            "context_anchor_uni_k": args.context_anchor_uni_k,
            "context_dense_pool_k": int(args.context_dense_pool_k),
            "intent_complexity_aware_budgeting": bool(args.intent_complexity_aware_budgeting),
            "intent_complexity_context_k_simple": int(args.intent_complexity_context_k_simple),
            "intent_complexity_context_k_medium": int(args.intent_complexity_context_k_medium),
            "intent_complexity_context_k_complex": int(args.intent_complexity_context_k_complex),
            "intent_complexity_dense_pool_k_simple": int(args.intent_complexity_dense_pool_k_simple),
            "intent_complexity_dense_pool_k_medium": int(args.intent_complexity_dense_pool_k_medium),
            "intent_complexity_dense_pool_k_complex": int(args.intent_complexity_dense_pool_k_complex),
            "intent_complexity_candidate_expand_k_simple": int(args.intent_complexity_candidate_expand_k_simple),
            "intent_complexity_candidate_expand_k_medium": int(args.intent_complexity_candidate_expand_k_medium),
            "intent_complexity_candidate_expand_k_complex": int(args.intent_complexity_candidate_expand_k_complex),
            "intent_complexity_level_hist": {
                k: int(v)
                for k, v in sorted(intent_complexity_level_hist.items(), key=lambda kv: int(kv[0]))
            },
            "intent_complexity_tier_hist": {k: int(v) for k, v in sorted(intent_complexity_tier_hist.items())},
            "intent_type_hist": {k: int(v) for k, v in sorted(intent_type_hist.items())},
            "intent_complexity_effective_context_k_mean": float(np.mean(effective_context_k_trace))
            if effective_context_k_trace
            else float(effective_qa_context_k),
            "intent_complexity_effective_dense_pool_k_mean": float(np.mean(effective_dense_pool_k_trace))
            if effective_dense_pool_k_trace
            else float(int(args.context_dense_pool_k)),
            "intent_complexity_effective_candidate_expand_k_mean": float(np.mean(effective_candidate_expand_k_trace))
            if effective_candidate_expand_k_trace
            else float(int(args.context_candidate_expand_k)),
            "context_number_table_quota_min": int(args.context_number_table_quota_min),
            "context_subquery_coverage_weight": float(args.context_subquery_coverage_weight),
            "context_light_rerank_weight": float(args.context_light_rerank_weight),
            "context_light_rerank_topn": int(args.context_light_rerank_topn),
            "context_light_rerank_targeted_only": bool(args.context_light_rerank_targeted_only),
            "context_strong_rerank_endpoint": str(args.context_strong_rerank_endpoint),
            "context_strong_rerank_topn": int(args.context_strong_rerank_topn),
            "context_strong_rerank_timeout": int(args.context_strong_rerank_timeout),
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
            "tessera_policy_context": bool(args.tessera_policy_context),
            "tessera_policy_pool_k": int(args.tessera_policy_pool_k),
            "tessera_policy_dense_pool_k": int(args.tessera_policy_dense_pool_k),
            "tessera_policy_target_weight": float(args.tessera_policy_target_weight),
            "tessera_policy_coverage_weight": float(args.tessera_policy_coverage_weight),
            "tessera_policy_diversity_weight": float(args.tessera_policy_diversity_weight),
            "tessera_no_evidence_retry": bool(args.tessera_no_evidence_retry),
            "tessera_no_evidence_retry_context_k": int(args.tessera_no_evidence_retry_context_k),
            "tessera_no_evidence_retry_pool_k": int(args.tessera_no_evidence_retry_pool_k),
            "tessera_no_evidence_retry_dense_pool_k": int(args.tessera_no_evidence_retry_dense_pool_k),
            "tessera_no_evidence_retry_min_support": float(args.tessera_no_evidence_retry_min_support),
            "tessera_no_evidence_retry_margin": float(args.tessera_no_evidence_retry_margin),
            "tessera_no_evidence_retry_low_support_threshold": float(args.tessera_no_evidence_retry_low_support_threshold),
            "tessera_no_evidence_retry_attempted": int(no_evidence_retry_attempted),
            "tessera_no_evidence_retry_applied": int(no_evidence_retry_applied),
            "tessera_table_number_agent": bool(args.tessera_table_number_agent),
            "tessera_table_number_agent_min_score": float(args.tessera_table_number_agent_min_score),
            "tessera_table_number_agent_min_support": float(args.tessera_table_number_agent_min_support),
            "tessera_table_number_agent_margin": float(args.tessera_table_number_agent_margin),
            "tessera_table_number_agent_low_support_threshold": float(
                args.tessera_table_number_agent_low_support_threshold
            ),
            "tessera_table_number_agent_attempted": int(table_number_agent_attempted),
            "tessera_table_number_agent_applied": int(table_number_agent_applied),
            "context_upo_lite_quota_mix": float(args.context_upo_lite_quota_mix),
            "context_upo_lite_rerank_bonus": float(args.context_upo_lite_rerank_bonus),
            "context_candidate_expand_k": int(args.context_candidate_expand_k),
            "context_redundancy_lambda": args.context_redundancy_lambda,
            "context_qa_objective_weight": float(args.context_qa_objective_weight),
            "context_qa_modality_bias": float(args.context_qa_modality_bias),
            "context_conflict_penalty_weight": float(args.context_conflict_penalty_weight),
            "context_conflict_targeted_only": bool(args.context_conflict_targeted_only),
            "context_conflict_table_kg_only": bool(args.context_conflict_table_kg_only),
            "context_conflict_risk_gating": bool(args.context_conflict_risk_gating),
            "context_conflict_risk_low": float(args.context_conflict_risk_low),
            "context_conflict_risk_high": float(args.context_conflict_risk_high),
            "context_conflict_risk_probe_k": int(args.context_conflict_risk_probe_k),
            "context_conflict_sensitive_target_scale": float(args.context_conflict_sensitive_target_scale),
            "context_conflict_max_literals_per_doc": int(args.context_conflict_max_literals_per_doc),
            "extractive_numeric_consensus": bool(args.extractive_numeric_consensus),
            "pathmaxsim_weight": args.pathmaxsim_weight,
            "pathmaxsim_kg_threshold": args.pathmaxsim_kg_threshold,
            "innovation_scheme2": bool(args.innovation_scheme2),
            "scheme2_cross_modal_weight": float(args.scheme2_cross_modal_weight),
            "scheme2_token_maxsim_weight": float(args.scheme2_token_maxsim_weight),
            "schemeb_heavy_mode": bool(args.schemeb_heavy_mode),
            "heavy_table_encoder_weight": float(args.heavy_table_encoder_weight),
            "heavy_kg_path_weight": float(args.heavy_kg_path_weight),
            "heavy_token_late_weight": float(args.heavy_token_late_weight),
            "heavy_query_max_tokens": int(args.heavy_query_max_tokens),
            "heavy_table_max_cells": int(args.heavy_table_max_cells),
            "heavy_token_doc_max_tokens": int(args.heavy_token_doc_max_tokens),
            "heavy_table_backend": str(heavy_table_backend_effective),
            "heavy_table_tapas_model": str(args.heavy_table_tapas_model),
            "heavy_table_tapas_model_resolved": heavy_table_tapas_model_resolved,
            "heavy_table_tapas_enabled": bool(tapas_bundle is not None),
            "heavy_table_tapas_required": bool(args.heavy_table_tapas_required),
            "heavy_table_tapas_topn": int(args.heavy_table_tapas_topn),
            "heavy_table_max_rows": int(args.heavy_table_max_rows),
            "heavy_table_max_cols": int(args.heavy_table_max_cols),
            "heavy_table_agg_cell_logit": float(args.heavy_table_agg_cell_logit),
            "heavy_table_agg_row_logit": float(args.heavy_table_agg_row_logit),
            "heavy_table_agg_col_logit": float(args.heavy_table_agg_col_logit),
            "heavy_table_agg_temp": float(args.heavy_table_agg_temp),
            "heavy_kg_backend": str(args.heavy_kg_backend),
            "heavy_kg_gnn_topn": int(args.heavy_kg_gnn_topn),
            "heavy_kg_max_hops": int(args.heavy_kg_max_hops),
            "heavy_kg_max_paths": int(args.heavy_kg_max_paths),
            "heavy_kg_contrastive_temp": float(args.heavy_kg_contrastive_temp),
            "heavy_kg_hard_negative_mode": str(args.heavy_kg_hard_negative_mode),
            "heavy_kg_hard_negative_topdocs": int(args.heavy_kg_hard_negative_topdocs),
            "heavy_kg_hard_negative_max_paths": int(args.heavy_kg_hard_negative_max_paths),
            "heavy_token_cross_modal_weight": float(args.heavy_token_cross_modal_weight),
            "heavy_branch_candidate_expand_k": int(args.heavy_branch_candidate_expand_k),
            "heavy_branch_candidate_table_weight": float(args.heavy_branch_candidate_table_weight),
            "heavy_branch_candidate_kg_weight": float(args.heavy_branch_candidate_kg_weight),
            "heavy_branch_candidate_max_total": int(args.heavy_branch_candidate_max_total),
            "heavy_score_calibration": str(args.heavy_score_calibration),
            "heavy_score_calibration_nonzero_only": bool(args.heavy_score_calibration_nonzero_only),
            "heavy_remove_hard_caps": bool(args.heavy_remove_hard_caps),
            "qa_objective_retrieval_weight": float(args.qa_objective_retrieval_weight),
            "qa_objective_targeted_only": bool(args.qa_objective_targeted_only),
            "adapter_plus_mode": bool(args.adapter_plus_mode),
            "adapter_official_lite": bool(args.adapter_official_lite),
            "table_cellmaxsim_weight": args.table_cellmaxsim_weight,
            "table_cellmaxsim_top_cells": args.table_cellmaxsim_top_cells,
            "context_consistency_weight": args.context_consistency_weight,
            "enable_tessera_answer_type_guard": bool(args.enable_tessera_answer_type_guard),
            "enable_tessera_answer_calibration": bool(args.enable_tessera_answer_calibration),
            "tessera_support_retry_threshold": float(args.tessera_support_retry_threshold),
            "tessera_support_retry_margin": float(args.tessera_support_retry_margin),
            "tessera_support_retry_targeted_only": bool(args.tessera_support_retry_targeted_only),
            "tessera_support_retry_mode": str(args.tessera_support_retry_mode),
            "tessera_support_retry_pool_k": int(args.tessera_support_retry_pool_k),
            "tessera_support_retry_answer_boost": float(args.tessera_support_retry_answer_boost),
            "tessera_support_retry_complexity_min": int(args.tessera_support_retry_complexity_min),
            "tessera_support_retry_entropy_min": float(args.tessera_support_retry_entropy_min),
            "tessera_support_retry_attempted": int(support_retry_attempted),
            "tessera_support_retry_applied": int(support_retry_applied),
            "enable_tessera_consensus_refine": bool(args.enable_tessera_consensus_refine),
            "tessera_consensus_refine_min_gain": float(args.tessera_consensus_refine_min_gain),
            "tessera_consensus_refine_support_threshold": float(args.tessera_consensus_refine_support_threshold),
            "tessera_consensus_refine_complexity_min": int(args.tessera_consensus_refine_complexity_min),
            "tessera_consensus_refine_entropy_min": float(args.tessera_consensus_refine_entropy_min),
            "tessera_consensus_refine_targeted_only": bool(args.tessera_consensus_refine_targeted_only),
            "tessera_consensus_refine_attempted": int(consensus_refine_attempted),
            "tessera_consensus_refine_applied": int(consensus_refine_applied),
            "dense_vs_tessera_same_top10_ratio": float(dense_uni_same_top10 / max(1, len(rows))),
            "dense_vs_tessera_avg_top10_overlap": float(dense_uni_overlap_sum / max(1, len(rows))),
            "router_source": router_source,
            "router_model": str(router_model_path) if router_model_path is not None else None,
            "allow_heuristic_router_fallback": bool(args.allow_heuristic_router_fallback),
            "router_subset_acc_run": router_subset_acc_run,
            "router_micro_f1_run": router_micro_f1_run,
            "router_avg_entropy": float(np.mean(router_entropy)) if len(router_entropy) else 0.0,
            "router_uncertain_ratio": float(np.mean(router_entropy >= args.routing_uncertainty_threshold)) if len(router_entropy) else 0.0,
            "baseline_reproduction_protocol": (
                "official_lite"
                if bool(args.adapter_official_lite)
                else ("adapter_plus" if bool(args.adapter_plus_mode) else "adapter")
            ),
            "baseline_adapter_methods": ["CARP-Adapter", "TableRAG-Adapter", "QUASAR-Adapter"],
            "external_dense_baselines": ["UniHGKR-Base"] if "unihgkr_dense" in selected_methods else [],
            "methods_selected": selected_methods,
            "query_modality_distribution": build_query_modality_distribution(rows, source_bucket),
        },
        "methods": {},
        "table1c_rows": [],
    }

    if results["meta"]["dense_vs_tessera_same_top10_ratio"] >= 0.95:
        print(
            "[warn] Dense vs TESSERA top-10 rankings are almost identical. "
            "Check preserve-dense-top and fusion settings before reporting results."
        )

    router_acc_from_metrics = get_router_acc(args.router_metrics)

    for m in selected_methods:
        eval_result = evaluate_predictions(
            rows=rows,
            preds=per_method_preds[m],
            top10_lists=per_method_top10[m],
            exact_match_fn=exact_match,
            f1_score_fn=f1_score,
            mmrag_official_fn=mmrag_official_generation_score,
            source_bucket_fn=source_bucket,
        )

        qa_p95_ms = percentile95_ms(per_method_latency[m])
        retrieval_p95_ms = percentile95_ms(retrieval_latency)

        p95_ms = 0.0
        if per_method_latency[m] and retrieval_latency and len(per_method_latency[m]) == len(retrieval_latency):
            combined = [r + q for r, q in zip(retrieval_latency, per_method_latency[m])]
            p95_ms = percentile95_ms(combined)
        elif per_method_latency[m]:
            # Fallback additive approximation if latency arrays are not aligned.
            p95_ms = retrieval_p95_ms + qa_p95_ms
        elif m in previous_p95:
            p95_ms = float(previous_p95[m])

        pred_file = args.out_dir / f"qa_predictions_{m}_test1286.jsonl"
        write_predictions_jsonl(pred_file, rows, per_method_preds[m])
        # Save context doc IDs for evidence distribution analysis
        ctx_file = args.out_dir / f"context_docs_{m}_test1286.jsonl"
        with ctx_file.open("w", encoding="utf-8") as cf:
            for row_item, ctx_doc_ids in zip(rows, per_method_context_docs[m]):
                cf.write(json.dumps({"id": row_item.get("id"), "context_doc_ids": ctx_doc_ids}, ensure_ascii=False) + "\n")

        routing = None
        if m == "tessera_rag":
            routing = router_subset_acc_run

        method_result = {
            "method": METHOD_LABELS[m],
            "key": m,
            "exact_match": float(eval_result.get("exact_match", 0.0)),
            "f1": float(eval_result.get("f1", 0.0)),
            "mmrag_official_avg": float(eval_result.get("mmrag_official_avg", 0.0)),
            "recall@10": float(eval_result.get("recall@10", 0.0)),
            "routing_acc": routing,
            "p95_latency_ms": p95_ms,
            "qa_p95_latency_ms": qa_p95_ms,
            "retrieval_shared_p95_ms": retrieval_p95_ms,
            "pred_file": str(pred_file),
            "modality_coverage": METHOD_MODALITY_COVERAGE.get(m, "T+Tbl+G"),
            "slice_metrics": eval_result.get("slice_metrics", {}),
        }
        results["methods"][m] = method_result
        results["table1c_rows"].append(method_result)

    print(
        "[note] Baseline rows CARP/TableRAG/QUASAR are adapter reproductions under a unified protocol, "
        "not original official pipelines."
    )

    if args.include_oracle_row:
        oracle = {
            "method": "Oracle",
            "key": "oracle",
            "exact_match": 1.0,
            "f1": 1.0,
            "recall@10": 1.0,
            "routing_acc": 1.0,
            "p95_latency_ms": 0.0,
            "pred_file": None,
        }
        results["methods"]["oracle"] = oracle
        results["table1c_rows"].append(oracle)

    if args.include_oracle_measured_row:
        em_vals = []
        f1_vals = []
        r10_vals = []
        mmrag_official_vals = []
        preds_jsonl = []
        for row, pred, top10 in zip(rows, oracle_gold_preds, oracle_gold_top10):
            gold = str(row.get("answer", ""))
            em_vals.append(exact_match(pred, gold))
            f1_vals.append(f1_score(pred, gold))
            mmrag_official_vals.append(
                mmrag_official_generation_score(str(row.get("id", "")), row.get("answer", ""), pred)
            )
            rel = eval_positive_relevant_ids(row)
            inter = len(set(top10) & rel)
            denom = max(1, len(rel))
            r10_vals.append(inter / denom)
            preds_jsonl.append({"id": row.get("id"), "prediction": pred})

        oracle_pred_file = args.out_dir / "qa_predictions_oracle_gold_test1286.jsonl"
        with oracle_pred_file.open("w", encoding="utf-8") as f:
            for x in preds_jsonl:
                f.write(json.dumps(x, ensure_ascii=False) + "\n")

        oracle_measured = {
            "method": "Oracle (Gold Evidence, measured)",
            "key": "oracle_gold",
            "exact_match": float(np.mean(em_vals)) if em_vals else 0.0,
            "f1": float(np.mean(f1_vals)) if f1_vals else 0.0,
            "mmrag_official_avg": float(np.mean(mmrag_official_vals)) if mmrag_official_vals else 0.0,
            "recall@10": float(np.mean(r10_vals)) if r10_vals else 0.0,
            "routing_acc": None,
            "p95_latency_ms": percentile95_ms(oracle_gold_latency),
            "qa_p95_latency_ms": percentile95_ms(oracle_gold_latency),
            "retrieval_shared_p95_ms": 0.0,
            "pred_file": str(oracle_pred_file),
            "modality_coverage": "T+Tbl+G+Vec",
        }
        results["methods"]["oracle_gold"] = oracle_measured
        results["table1c_rows"].append(oracle_measured)

    if router_acc_from_metrics is not None:
        results["meta"]["router_subset_acc_external"] = float(router_acc_from_metrics)
    results["meta"]["retrieval_shared_p95_ms"] = percentile95_ms(retrieval_latency)

    if schemeb_debug_trace:
        summary = {}
        for k, vals in schemeb_debug_trace.items():
            if not vals:
                continue
            arr = np.asarray(vals, dtype=np.float32)
            summary[k] = {
                "mean": float(np.mean(arr)),
                "p50": float(np.percentile(arr, 50)),
                "p90": float(np.percentile(arr, 90)),
            }
        if summary:
            results["meta"]["schemeb_debug_summary"] = summary

    if context_debug_trace:
        summary = {}
        for k, vals in context_debug_trace.items():
            if not vals:
                continue
            arr = np.asarray(vals, dtype=np.float32)
            summary[k] = {
                "mean": float(np.mean(arr)),
                "p50": float(np.percentile(arr, 50)),
                "p90": float(np.percentile(arr, 90)),
            }
        if summary:
            results["meta"]["context_debug_summary"] = summary

    if use_heavy_schemeb and "schemeb_debug_summary" in results["meta"]:
        dbg = results["meta"]["schemeb_debug_summary"]
        wt = float(dbg.get("heavy_table_weight_eff", {}).get("mean", 0.0))
        wk = float(dbg.get("heavy_kg_path_weight_eff", {}).get("mean", 0.0))
        wl = float(dbg.get("heavy_token_late_weight_eff", {}).get("mean", 0.0))
        wc = float(dbg.get("heavy_token_cross_weight_eff", {}).get("mean", 0.0))
        if max(wt, wk, wl, wc) < 1e-4:
            print("[warn] schemeB heavy mode is enabled but effective heavy weights are near zero across queries.")

    if float(args.context_conflict_penalty_weight) > 0.0 and "context_debug_summary" in results["meta"]:
        cdbg = results["meta"]["context_debug_summary"]
        trig = float(cdbg.get("context_conflict_penalty_trigger_rate", {}).get("mean", 0.0))
        if trig < 0.05:
            print("[warn] context conflict penalty trigger rate is very low (<5%); consider revisiting conflict matching rules.")

    out_json = args.out_dir / "table1c_e2e_metrics.json"
    out_md = args.out_dir / "table1c_e2e_metrics.md"
    out_json.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    out_md.write_text(build_table1c_markdown(results), encoding="utf-8")

    print(f"[OK] metrics -> {out_json}")
    print(f"[OK] markdown -> {out_md}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
