from __future__ import annotations

import pickle
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from unifusion_exp.e2e.baselines import source_bucket
from unifusion_exp.e2e.conflict import estimate_context_conflict_risk, extract_numeric_literals, extract_signal_tokens
from unifusion_exp.e2e.metrics import normalize_answer
from unifusion_exp.e2e.objectives import (
    estimate_query_complexity_level,
    infer_query_intent_type,
    infer_qa_target_type,
    infer_upo_lite_concept,
)


MODALITY_NAMES = ("text", "table", "graph")
TOKEN_RE = re.compile(r"[a-z0-9]+")


def tokenize(text: str) -> list[str]:
    return TOKEN_RE.findall(str(text).lower())


def normalize_prob_vector(values: np.ndarray | list[float]) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float32).reshape(-1)
    if arr.size == 0:
        return np.asarray([], dtype=np.float32)
    arr = np.clip(arr, 1e-8, None)
    total = float(arr.sum())
    if total <= 0.0:
        return np.full(arr.shape, 1.0 / float(arr.size), dtype=np.float32)
    return (arr / total).astype(np.float32)


def normalized_entropy(probs: np.ndarray) -> float:
    arr = np.asarray(probs, dtype=np.float32).reshape(-1)
    if arr.size <= 1:
        return 0.0
    arr = np.clip(arr, 1e-8, 1.0)
    arr = arr / max(1e-8, float(arr.sum()))
    ent = float(-(arr * np.log(arr)).sum())
    return float(ent / max(1e-8, np.log(float(arr.size))))


def exact_label_from_multihot(labels: list[int] | tuple[int, ...] | np.ndarray) -> str:
    active = [name for name, flag in zip(MODALITY_NAMES, labels) if int(flag) > 0]
    if not active:
        return "unknown"
    return "+".join(active)


def multihot_from_exact_label(label: str) -> np.ndarray:
    parts = {part.strip().lower() for part in str(label).split("+") if part.strip()}
    return np.asarray([1.0 if name in parts else 0.0 for name in MODALITY_NAMES], dtype=np.float32)


def build_planner_input(query: str, query_id: str | None = None) -> str:
    query = str(query).strip()
    target = str(infer_qa_target_type(query)).strip().lower()
    intent = str(infer_query_intent_type(query)).strip().lower()
    complexity = int(estimate_query_complexity_level(query))
    concept = str(infer_upo_lite_concept(query)).strip().lower()
    qlen = len(tokenize(query))
    if qlen <= 6:
        qlen_bucket = "short"
    elif qlen <= 14:
        qlen_bucket = "medium"
    else:
        qlen_bucket = "long"
    parts = [query]
    if query_id:
        parts.append(f"__qidlen_{len(str(query_id))}")
    parts.extend(
        [
            f"__target_{target}",
            f"__intent_{intent}",
            f"__complexity_{complexity}",
            f"__upo_{concept}",
            f"__qlen_{qlen_bucket}",
        ]
    )
    return " ".join(parts).strip()


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
        for s in re.split(r"[\n\.\!\?;]+", c):
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

    length_penalty = max(0.0, (len(a_toks) - 8) * 0.03)
    return float(1.2 * has_exact + best_f - length_penalty)


def _jaccard(tokens_a: set[str], tokens_b: set[str]) -> float:
    if not tokens_a and not tokens_b:
        return 0.0
    return float(len(tokens_a & tokens_b) / max(1, len(tokens_a | tokens_b)))


def _as_token_set(text: str) -> set[str]:
    return set(tokenize(text))


CONFLICT_FEATURE_NAMES = [
    "query_token_count",
    "query_signal_token_count",
    "query_numeric_literal_count",
    "context_count",
    "context_char_count_sum",
    "context_token_count_sum",
    "context_token_count_mean",
    "context_token_count_max",
    "context_token_count_min",
    "query_context_union_overlap",
    "query_context_overlap_mean",
    "query_context_overlap_max",
    "context_pairwise_jaccard_mean",
    "context_pairwise_jaccard_max",
    "numeric_literal_total_count",
    "numeric_literal_unique_count",
    "numeric_literal_pair_disjoint_rate",
    "numeric_literal_pair_overlap_mean",
    "signal_token_total_count",
    "signal_token_unique_count",
    "signal_token_pair_overlap_mean",
    "signal_token_pair_overlap_max",
    "bucket_text_ratio",
    "bucket_table_ratio",
    "bucket_kg_ratio",
    "bucket_entropy",
    "unique_bucket_count",
    "target_number",
    "target_year",
    "target_location",
    "target_person",
    "target_entity",
    "target_boolean",
    "target_open",
    "complexity_norm",
    "query_numeric_presence",
    "heuristic_conflict_risk",
]


def _mean_or_zero(values: list[float]) -> float:
    return float(np.mean(values)) if values else 0.0


def _max_or_zero(values: list[float]) -> float:
    return float(np.max(values)) if values else 0.0


def _positive_class_probability(model: Any, features: np.ndarray) -> float:
    probs = np.asarray(model.predict_proba([features])[0], dtype=np.float32).reshape(-1)
    classes = np.asarray(getattr(model, "classes_", []))
    if probs.size == 1:
        if classes.size == 1 and str(classes[0]).strip().lower() in {"1", "true", "yes", "support", "conflict"}:
            return float(probs[0])
        return 0.0
    if classes.size == probs.size:
        class_list = classes.tolist()
        for idx, cls in enumerate(class_list):
            if str(cls).strip().lower() in {"1", "true", "yes", "support", "conflict"}:
                return float(probs[idx])
        if 1 in class_list:
            return float(probs[int(np.where(classes == 1)[0][0])])
    return float(probs[-1])


def build_conflict_feature_vector(
    query: str,
    contexts: list[str],
    doc_ids: list[str] | None = None,
    *,
    table_kg_only: bool = False,
    probe_k: int = 12,
    max_literals_per_doc: int = 0,
) -> np.ndarray:
    query = str(query)
    context_texts = [str(ctx) for ctx in contexts]
    if doc_ids is None or len(doc_ids) != len(context_texts):
        doc_ids = [f"ctx_{idx}" for idx in range(len(context_texts))]

    query_tokens = _as_token_set(query)
    query_signal_tokens = set(extract_signal_tokens(query))
    query_numeric_literals = set(extract_numeric_literals(query))
    context_tokens = [_as_token_set(ctx) for ctx in context_texts]
    context_signal_tokens = [set(extract_signal_tokens(ctx)) for ctx in context_texts]
    context_numeric_literals = [set(extract_numeric_literals(ctx)) for ctx in context_texts]
    context_token_counts = [len(tokens) for tokens in context_tokens]
    context_char_counts = [len(ctx) for ctx in context_texts]

    union_context_tokens: set[str] = set()
    union_signal_tokens: set[str] = set()
    union_numeric_literals: set[str] = set()
    for tokens in context_tokens:
        union_context_tokens.update(tokens)
    for tokens in context_signal_tokens:
        union_signal_tokens.update(tokens)
    for literals in context_numeric_literals:
        union_numeric_literals.update(literals)

    query_context_overlaps = [_jaccard(query_tokens, tokens) for tokens in context_tokens]
    pairwise_jaccards: list[float] = []
    numeric_pair_overlaps: list[float] = []
    signal_pair_overlaps: list[float] = []
    numeric_pair_disjoint = 0
    valid_numeric_pairs = 0
    for i in range(len(context_tokens)):
        for j in range(i + 1, len(context_tokens)):
            pairwise_jaccards.append(_jaccard(context_tokens[i], context_tokens[j]))

            nums_a = context_numeric_literals[i]
            nums_b = context_numeric_literals[j]
            if nums_a or nums_b:
                valid_numeric_pairs += 1
                numeric_pair_overlaps.append(_jaccard(nums_a, nums_b))
                if nums_a and nums_b and not (nums_a & nums_b):
                    numeric_pair_disjoint += 1

            signal_pair_overlaps.append(_jaccard(context_signal_tokens[i], context_signal_tokens[j]))

    bucket_counts = {"text": 0, "table": 0, "graph": 0}
    for doc_id in doc_ids:
        bucket = source_bucket(str(doc_id))
        if bucket in bucket_counts:
            bucket_counts[bucket] += 1

    context_count = len(context_texts)
    if context_count > 0:
        bucket_mass = np.asarray([bucket_counts["text"], bucket_counts["table"], bucket_counts["graph"]], dtype=np.float32)
        bucket_prob = normalize_prob_vector(bucket_mass)
        bucket_entropy = normalized_entropy(bucket_prob)
        unique_bucket_count = float(sum(1 for count in bucket_counts.values() if count > 0))
    else:
        bucket_prob = np.asarray([0.0, 0.0, 0.0], dtype=np.float32)
        bucket_entropy = 0.0
        unique_bucket_count = 0.0

    target_type = str(infer_qa_target_type(query)).strip().lower()
    target_vec = {
        "number": 0.0,
        "year": 0.0,
        "location": 0.0,
        "person": 0.0,
        "entity": 0.0,
        "boolean": 0.0,
        "open": 0.0,
    }
    if target_type in target_vec:
        target_vec[target_type] = 1.0
    else:
        target_vec["open"] = 1.0

    heuristic_conflict_risk = estimate_context_conflict_risk(
        query=query,
        contexts=context_texts,
        doc_ids=doc_ids,
        table_kg_only=bool(table_kg_only),
        probe_k=int(probe_k),
        max_literals_per_doc=int(max_literals_per_doc),
    )

    features = np.asarray(
        [
            float(len(query_tokens)),
            float(len(query_signal_tokens)),
            float(len(query_numeric_literals)),
            float(context_count),
            float(sum(context_char_counts)),
            float(sum(context_token_counts)),
            float(np.mean(context_token_counts)) if context_token_counts else 0.0,
            float(np.max(context_token_counts)) if context_token_counts else 0.0,
            float(np.min(context_token_counts)) if context_token_counts else 0.0,
            _jaccard(query_tokens, union_context_tokens),
            _mean_or_zero(query_context_overlaps),
            _max_or_zero(query_context_overlaps),
            _mean_or_zero(pairwise_jaccards),
            _max_or_zero(pairwise_jaccards),
            float(sum(len(nums) for nums in context_numeric_literals)),
            float(len(union_numeric_literals)),
            float(numeric_pair_disjoint / max(1, valid_numeric_pairs)),
            _mean_or_zero(numeric_pair_overlaps),
            float(sum(len(tokens) for tokens in context_signal_tokens)),
            float(len(union_signal_tokens)),
            _mean_or_zero(signal_pair_overlaps),
            _max_or_zero(signal_pair_overlaps),
            float(bucket_prob[0]),
            float(bucket_prob[1]),
            float(bucket_prob[2]),
            float(bucket_entropy),
            float(unique_bucket_count),
            float(target_vec["number"]),
            float(target_vec["year"]),
            float(target_vec["location"]),
            float(target_vec["person"]),
            float(target_vec["entity"]),
            float(target_vec["boolean"]),
            float(target_vec["open"]),
            float(int(estimate_query_complexity_level(query)) / 5.0),
            1.0 if query_numeric_literals else 0.0,
            float(heuristic_conflict_risk),
        ],
        dtype=np.float32,
    )
    if features.shape[0] != len(CONFLICT_FEATURE_NAMES):
        raise RuntimeError(
            f"conflict feature size mismatch: expected {len(CONFLICT_FEATURE_NAMES)}, got {features.shape[0]}"
        )
    return features


VERIFIER_FEATURE_NAMES = [
    "support_score",
    "answer_exact_contains",
    "query_token_count",
    "answer_token_count",
    "context_count",
    "context_char_count_sum",
    "context_token_count_sum",
    "context_token_count_mean",
    "context_token_count_max",
    "context_token_count_min",
    "query_answer_overlap",
    "query_context_union_overlap",
    "answer_context_union_overlap",
    "query_context_overlap_mean",
    "query_context_overlap_max",
    "answer_context_overlap_mean",
    "answer_context_overlap_max",
    "context_pairwise_jaccard_mean",
    "context_pairwise_jaccard_max",
    "number_answer",
    "year_answer",
    "boolean_answer",
    "target_number",
    "target_year",
    "target_location",
    "target_person",
    "target_entity",
    "target_boolean",
    "target_open",
    "complexity_norm",
    "bucket_text_ratio",
    "bucket_table_ratio",
    "bucket_kg_ratio",
    "conflict_risk",
]


def build_verifier_feature_vector(
    query: str,
    answer: str,
    contexts: list[str],
    doc_ids: list[str] | None = None,
    include_conflict_risk: bool = True,
) -> np.ndarray:
    query = str(query)
    answer = str(answer)
    query_tokens = _as_token_set(query)
    answer_norm = normalize_answer(answer)
    answer_tokens = _as_token_set(answer_norm)
    context_tokens = [_as_token_set(ctx) for ctx in contexts]
    context_token_counts = [len(tokens) for tokens in context_tokens]
    context_char_counts = [len(str(ctx)) for ctx in contexts]

    union_context_tokens: set[str] = set()
    for tokens in context_tokens:
        union_context_tokens.update(tokens)

    support_score = answer_support_score(answer, contexts)
    answer_exact_contains = 1.0 if any(answer_norm and answer_norm in normalize_answer(ctx) for ctx in contexts) else 0.0
    query_answer_overlap = _jaccard(query_tokens, answer_tokens)
    query_context_union_overlap = _jaccard(query_tokens, union_context_tokens)
    answer_context_union_overlap = _jaccard(answer_tokens, union_context_tokens)

    query_context_overlaps: list[float] = []
    answer_context_overlaps: list[float] = []
    for ctx_tokens in context_tokens:
        query_context_overlaps.append(_jaccard(query_tokens, ctx_tokens))
        answer_context_overlaps.append(_jaccard(answer_tokens, ctx_tokens))

    pairwise_jaccards: list[float] = []
    for i in range(len(context_tokens)):
        for j in range(i + 1, len(context_tokens)):
            pairwise_jaccards.append(_jaccard(context_tokens[i], context_tokens[j]))

    target_type = str(infer_qa_target_type(query)).strip().lower()
    complexity_norm = float(int(estimate_query_complexity_level(query)) / 5.0)
    answer_lower = answer_norm.lower()
    number_answer = 1.0 if re.search(r"\b\d{1,3}(?:,\d{3})*(?:\.\d+)?\b", answer_lower) else 0.0
    year_answer = 1.0 if re.search(r"\b(1[0-9]{3}|20[0-9]{2})\b", answer_lower) else 0.0
    boolean_answer = 1.0 if answer_lower in {"yes", "no", "true", "false"} else 0.0

    target_vec = {
        "number": 0.0,
        "year": 0.0,
        "location": 0.0,
        "person": 0.0,
        "entity": 0.0,
        "boolean": 0.0,
        "open": 0.0,
    }
    if target_type in target_vec:
        target_vec[target_type] = 1.0
    else:
        target_vec["open"] = 1.0

    bucket_counts = {"text": 0, "table": 0, "graph": 0}
    if doc_ids:
        for doc_id in doc_ids:
            bucket = source_bucket(str(doc_id))
            if bucket in bucket_counts:
                bucket_counts[bucket] += 1
    context_count = len(contexts)
    if context_count > 0:
        bucket_text_ratio = bucket_counts["text"] / context_count
        bucket_table_ratio = bucket_counts["table"] / context_count
        bucket_kg_ratio = bucket_counts["graph"] / context_count
    else:
        bucket_text_ratio = 0.0
        bucket_table_ratio = 0.0
        bucket_kg_ratio = 0.0

    conflict_risk = None
    if include_conflict_risk:
        conflict_risk = estimate_context_conflict_risk(
            query=query,
            contexts=contexts,
            doc_ids=doc_ids,
            table_kg_only=False,
            probe_k=min(12, max(2, context_count)),
            max_literals_per_doc=0,
        )

    features = np.asarray(
        [
            support_score,
            answer_exact_contains,
            float(len(query_tokens)),
            float(len(answer_tokens)),
            float(context_count),
            float(sum(context_char_counts)),
            float(sum(context_token_counts)),
            float(np.mean(context_token_counts)) if context_token_counts else 0.0,
            float(np.max(context_token_counts)) if context_token_counts else 0.0,
            float(np.min(context_token_counts)) if context_token_counts else 0.0,
            query_answer_overlap,
            query_context_union_overlap,
            answer_context_union_overlap,
            float(np.mean(query_context_overlaps)) if query_context_overlaps else 0.0,
            float(np.max(query_context_overlaps)) if query_context_overlaps else 0.0,
            float(np.mean(answer_context_overlaps)) if answer_context_overlaps else 0.0,
            float(np.max(answer_context_overlaps)) if answer_context_overlaps else 0.0,
            float(np.mean(pairwise_jaccards)) if pairwise_jaccards else 0.0,
            float(np.max(pairwise_jaccards)) if pairwise_jaccards else 0.0,
            number_answer,
            year_answer,
            boolean_answer,
            target_vec["number"],
            target_vec["year"],
            target_vec["location"],
            target_vec["person"],
            target_vec["entity"],
            target_vec["boolean"],
            target_vec["open"],
            complexity_norm,
            bucket_text_ratio,
            bucket_table_ratio,
            bucket_kg_ratio,
        ],
        dtype=np.float32,
    )
    if conflict_risk is not None:
        features = np.concatenate([features, np.asarray([conflict_risk], dtype=np.float32)])
    return features


def _resolve_bundle_path(path: str | Path, default_filename: str) -> Path:
    p = Path(path)
    if p.is_dir():
        return p / default_filename
    return p


def _load_pickle(path: Path):
    with path.open("rb") as handle:
        return pickle.load(handle)


def _save_pickle(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as handle:
        pickle.dump(obj, handle, protocol=pickle.HIGHEST_PROTOCOL)


def _patch_legacy_sklearn_estimator_compatibility(value: Any) -> None:
    seen: set[int] = set()

    def visit(item: Any) -> None:
        if item is None:
            return
        item_id = id(item)
        if item_id in seen:
            return
        seen.add(item_id)

        if item.__class__.__name__ == "LogisticRegression" and not hasattr(item, "multi_class"):
            setattr(item, "multi_class", "auto")

        steps = getattr(item, "steps", None)
        if steps:
            for _, step in steps:
                visit(step)

        named_steps = getattr(item, "named_steps", None)
        if named_steps:
            for step in named_steps.values():
                visit(step)

        estimators = getattr(item, "estimators", None)
        if estimators:
            for estimator in estimators:
                visit(estimator)

        for attr in ("estimator", "base_estimator", "classifier", "regressor"):
            visit(getattr(item, attr, None))

    visit(value)


def _normalize_label_probs(class_probs: np.ndarray, class_to_modality: np.ndarray) -> np.ndarray:
    mod = np.asarray(class_probs, dtype=np.float32) @ np.asarray(class_to_modality, dtype=np.float32)
    return normalize_prob_vector(mod)


def derive_packing_adjustments(
    *,
    base_query_modality_prior_mix: float,
    base_context_active_threshold: float,
    base_context_anchor_dense_k: int,
    base_context_anchor_uni_k: int,
    base_context_dense_pool_k: int,
    base_context_candidate_expand_k: int,
    base_context_redundancy_lambda: float,
    base_context_conflict_penalty_weight: float,
    base_context_number_table_quota_min: int,
    base_context_light_rerank_weight: float,
    base_context_qa_objective_weight: float,
    base_context_qa_modality_bias: float,
    planner_modality_prior: np.ndarray,
    planner_confidence: float,
    router_entropy: float,
    target_type: str,
    query_complexity_level: int,
) -> dict[str, float | int]:
    prior = normalize_prob_vector(planner_modality_prior)
    text_mass = float(prior[0]) if prior.size >= 1 else 0.0
    table_mass = float(prior[1]) if prior.size >= 2 else 0.0
    graph_mass = float(prior[2]) if prior.size >= 3 else 0.0
    multi_mass = float(max(0.0, 1.0 - float(np.max(prior)) if prior.size else 1.0))
    confidence = float(np.clip(planner_confidence, 0.0, 1.0))
    entropy = float(np.clip(router_entropy, 0.0, 1.0))
    complexity_boost = float(max(0, int(query_complexity_level) - 3))

    planner_mix = float(np.clip(base_query_modality_prior_mix * (0.60 + 0.40 * confidence) * (0.85 + 0.25 * entropy), 0.0, 0.75))
    active_threshold = float(
        np.clip(
            base_context_active_threshold - 0.05 * confidence + 0.03 * multi_mass + 0.02 * max(0.0, entropy - 0.5),
            0.20,
            0.70,
        )
    )
    anchor_dense_k = max(1, int(round(base_context_anchor_dense_k + (1 if entropy >= 0.70 else 0) - (1 if confidence >= 0.85 else 0))))
    anchor_uni_k = max(1, int(round(base_context_anchor_uni_k + (1 if multi_mass >= 0.35 else 0) + (1 if complexity_boost >= 1 else 0))))
    dense_pool_k = max(
        anchor_dense_k + 2,
        int(round(base_context_dense_pool_k * (1.0 + 0.20 * multi_mass + 0.10 * complexity_boost + 0.15 * (1.0 - confidence)))),
    )
    candidate_expand_k = max(
        0,
        int(round(base_context_candidate_expand_k + 2 * multi_mass + (2 if entropy >= 0.70 else 0) + complexity_boost)),
    )
    redundancy_lambda = float(max(0.0, base_context_redundancy_lambda * (0.70 + 0.40 * (1.0 - confidence))))
    conflict_penalty_weight = float(
        max(0.0, base_context_conflict_penalty_weight * (0.85 + 0.35 * (table_mass + graph_mass) + 0.20 * (1.0 - confidence)))
    )
    number_table_quota_min = int(base_context_number_table_quota_min)
    if target_type == "number" and table_mass >= 0.20:
        number_table_quota_min += 1
    if target_type == "year" and table_mass >= 0.15:
        number_table_quota_min += 1
    light_rerank_weight = float(max(0.0, base_context_light_rerank_weight * (0.85 + 0.25 * confidence + 0.20 * table_mass)))
    qa_objective_weight = float(max(0.0, base_context_qa_objective_weight * (0.75 + 0.25 * text_mass)))
    qa_modality_bias = float(
        max(
            0.0,
            base_context_qa_modality_bias
            * (
                0.85 + 0.35 * table_mass
                if target_type in {"number", "year", "boolean"}
                else 0.85 + 0.20 * text_mass
            ),
        )
    )

    return {
        "query_modality_prior_mix": planner_mix,
        "context_active_threshold": active_threshold,
        "context_anchor_dense_k": anchor_dense_k,
        "context_anchor_uni_k": anchor_uni_k,
        "context_dense_pool_k": dense_pool_k,
        "context_candidate_expand_k": candidate_expand_k,
        "context_redundancy_lambda": redundancy_lambda,
        "context_conflict_penalty_weight": conflict_penalty_weight,
        "context_number_table_quota_min": number_table_quota_min,
        "context_light_rerank_weight": light_rerank_weight,
        "context_qa_objective_weight": qa_objective_weight,
        "context_qa_modality_bias": qa_modality_bias,
    }


def derive_retry_controls(
    *,
    support_probability: float,
    target_type: str,
    router_entropy: float,
    base_support_retry_threshold: float,
    base_support_retry_margin: float,
    base_support_retry_mode: str,
    base_consensus_support_threshold: float,
    conflict_probability: float | None = None,
) -> dict[str, float | str]:
    support_probability = float(np.clip(support_probability, 0.0, 1.0))
    router_entropy = float(np.clip(router_entropy, 0.0, 1.0))
    conflict_probability = None if conflict_probability is None else float(np.clip(conflict_probability, 0.0, 1.0))
    threshold = float(base_support_retry_threshold)
    margin = float(base_support_retry_margin)
    mode = str(base_support_retry_mode)
    consensus_threshold = float(base_consensus_support_threshold)

    if support_probability < 0.35:
        threshold += 0.08
        margin *= 0.85
        mode = "evidence_chain"
        consensus_threshold = consensus_threshold if consensus_threshold < 0.0 else consensus_threshold + 0.05
    elif support_probability < 0.55:
        threshold += 0.04
        margin *= 0.95
        if router_entropy >= 0.65:
            mode = "evidence_chain"
        consensus_threshold = consensus_threshold if consensus_threshold < 0.0 else consensus_threshold + 0.03
    else:
        threshold -= 0.03
        margin *= 1.05
        consensus_threshold = consensus_threshold if consensus_threshold < 0.0 else consensus_threshold - 0.02

    if target_type in {"number", "year", "location"}:
        threshold += 0.03
        consensus_threshold = consensus_threshold if consensus_threshold < 0.0 else consensus_threshold + 0.02
    elif target_type in {"boolean"}:
        threshold += 0.02

    if router_entropy >= 0.75:
        threshold += 0.02
        if support_probability < 0.60:
            mode = "evidence_chain"

    if conflict_probability is not None:
        if conflict_probability >= 0.75:
            threshold += 0.05
            margin *= 0.90
            mode = "evidence_chain"
            consensus_threshold = consensus_threshold if consensus_threshold < 0.0 else consensus_threshold + 0.05
        elif conflict_probability >= 0.50:
            threshold += 0.03
            margin *= 0.95
            if support_probability < 0.70 or router_entropy >= 0.60:
                mode = "evidence_chain"
            consensus_threshold = consensus_threshold if consensus_threshold < 0.0 else consensus_threshold + 0.03
        elif conflict_probability <= 0.25:
            threshold -= 0.02
            margin *= 1.05
            consensus_threshold = consensus_threshold if consensus_threshold < 0.0 else consensus_threshold - 0.02

    threshold = float(np.clip(threshold, 0.0, 0.95))
    margin = float(np.clip(margin, 0.01, 0.50))
    if consensus_threshold >= 0.0:
        consensus_threshold = float(np.clip(consensus_threshold, 0.0, 0.95))

    return {
        "support_retry_threshold": threshold,
        "support_retry_margin": margin,
        "support_retry_mode": mode,
        "consensus_support_threshold": consensus_threshold,
    }


@dataclass(slots=True)
class PlannerBundle:
    model: Any
    class_names: list[str]
    class_to_modality: np.ndarray
    metadata: dict[str, Any]

    @classmethod
    def load(cls, path: str | Path) -> "PlannerBundle":
        resolved = _resolve_bundle_path(path, "planner_bundle.pkl")
        payload = _load_pickle(resolved)
        if isinstance(payload, cls):
            _patch_legacy_sklearn_estimator_compatibility(payload.model)
            return payload
        bundle = cls(**payload)
        _patch_legacy_sklearn_estimator_compatibility(bundle.model)
        return bundle

    def save(self, path: str | Path) -> Path:
        resolved = _resolve_bundle_path(path, "planner_bundle.pkl")
        _save_pickle(resolved, self)
        return resolved

    def predict(self, query: str, query_id: str | None = None) -> dict[str, Any]:
        text = build_planner_input(query, query_id=query_id)
        probs = np.asarray(self.model.predict_proba([text])[0], dtype=np.float32)
        probs = normalize_prob_vector(probs)
        class_idx = int(np.argmax(probs))
        modality_prior = _normalize_label_probs(probs, self.class_to_modality)
        return {
            "class_name": str(self.class_names[class_idx]),
            "class_probs": probs,
            "class_confidence": float(probs[class_idx]),
            "class_entropy": normalized_entropy(probs),
            "modality_prior": modality_prior,
        }

    def predict_modality_prior(self, query: str, query_id: str | None = None) -> np.ndarray:
        return np.asarray(self.predict(query, query_id=query_id)["modality_prior"], dtype=np.float32)

    def blend_router_prob(
        self,
        router_prob: np.ndarray,
        query: str,
        query_id: str | None = None,
        base_mix: float = 0.35,
        router_entropy: float = 0.0,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        pred = self.predict(query, query_id=query_id)
        confidence = float(pred["class_confidence"])
        mix = float(np.clip(base_mix * (0.60 + 0.40 * confidence) * (0.85 + 0.25 * float(router_entropy)), 0.0, 0.75))
        blended = normalize_prob_vector((1.0 - mix) * np.asarray(router_prob, dtype=np.float32) + mix * np.asarray(pred["modality_prior"], dtype=np.float32))
        return blended, {"mix": mix, **pred}

    def derive_packing_adjustments(
        self,
        *,
        base_query_modality_prior_mix: float,
        base_context_active_threshold: float,
        base_context_anchor_dense_k: int,
        base_context_anchor_uni_k: int,
        base_context_dense_pool_k: int,
        base_context_candidate_expand_k: int,
        base_context_redundancy_lambda: float,
        base_context_conflict_penalty_weight: float,
        base_context_number_table_quota_min: int,
        base_context_light_rerank_weight: float,
        base_context_qa_objective_weight: float,
        base_context_qa_modality_bias: float,
        query: str,
        query_id: str | None,
        router_entropy: float,
    ) -> dict[str, float | int]:
        pred = self.predict(query, query_id=query_id)
        target_type = str(infer_qa_target_type(query)).strip().lower()
        complexity_level = int(estimate_query_complexity_level(query))
        return derive_packing_adjustments(
            base_query_modality_prior_mix=base_query_modality_prior_mix,
            base_context_active_threshold=base_context_active_threshold,
            base_context_anchor_dense_k=base_context_anchor_dense_k,
            base_context_anchor_uni_k=base_context_anchor_uni_k,
            base_context_dense_pool_k=base_context_dense_pool_k,
            base_context_candidate_expand_k=base_context_candidate_expand_k,
            base_context_redundancy_lambda=base_context_redundancy_lambda,
            base_context_conflict_penalty_weight=base_context_conflict_penalty_weight,
            base_context_number_table_quota_min=base_context_number_table_quota_min,
            base_context_light_rerank_weight=base_context_light_rerank_weight,
            base_context_qa_objective_weight=base_context_qa_objective_weight,
            base_context_qa_modality_bias=base_context_qa_modality_bias,
            planner_modality_prior=np.asarray(pred["modality_prior"], dtype=np.float32),
            planner_confidence=float(pred["class_confidence"]),
            router_entropy=float(router_entropy),
            target_type=target_type,
            query_complexity_level=complexity_level,
        )


@dataclass(slots=True)
class VerifierBundle:
    model: Any
    feature_names: list[str]
    metadata: dict[str, Any]

    @classmethod
    def load(cls, path: str | Path) -> "VerifierBundle":
        resolved = _resolve_bundle_path(path, "verifier_bundle.pkl")
        payload = _load_pickle(resolved)
        if isinstance(payload, cls):
            _patch_legacy_sklearn_estimator_compatibility(payload.model)
            return payload
        bundle = cls(**payload)
        _patch_legacy_sklearn_estimator_compatibility(bundle.model)
        return bundle

    def save(self, path: str | Path) -> Path:
        resolved = _resolve_bundle_path(path, "verifier_bundle.pkl")
        _save_pickle(resolved, self)
        return resolved

    def predict_support_probability(
        self,
        query: str,
        answer: str,
        contexts: list[str],
        doc_ids: list[str] | None = None,
    ) -> tuple[float, np.ndarray]:
        feature_version = int(self.metadata.get("feature_version", 1) or 1)
        include_conflict_risk = feature_version >= 2 and len(self.feature_names) >= len(VERIFIER_FEATURE_NAMES)
        feats = build_verifier_feature_vector(
            query,
            answer,
            contexts,
            doc_ids=doc_ids,
            include_conflict_risk=include_conflict_risk,
        )
        prob = _positive_class_probability(self.model, feats)
        return prob, feats

    def derive_retry_controls(
        self,
        *,
        query: str,
        answer: str,
        contexts: list[str],
        doc_ids: list[str] | None,
        router_entropy: float,
        base_support_retry_threshold: float,
        base_support_retry_margin: float,
        base_support_retry_mode: str,
        base_consensus_support_threshold: float,
        conflict_probability: float | None = None,
    ) -> dict[str, float | str]:
        support_probability, _ = self.predict_support_probability(query, answer, contexts, doc_ids=doc_ids)
        target_type = str(infer_qa_target_type(query)).strip().lower()
        return derive_retry_controls(
            support_probability=support_probability,
            target_type=target_type,
            router_entropy=float(router_entropy),
            base_support_retry_threshold=base_support_retry_threshold,
            base_support_retry_margin=base_support_retry_margin,
            base_support_retry_mode=base_support_retry_mode,
            base_consensus_support_threshold=base_consensus_support_threshold,
            conflict_probability=conflict_probability,
        )


@dataclass(slots=True)
class ConflictBundle:
    model: Any
    feature_names: list[str]
    metadata: dict[str, Any]

    @classmethod
    def load(cls, path: str | Path) -> "ConflictBundle":
        resolved = _resolve_bundle_path(path, "conflict_bundle.pkl")
        payload = _load_pickle(resolved)
        if isinstance(payload, cls):
            _patch_legacy_sklearn_estimator_compatibility(payload.model)
            return payload
        bundle = cls(**payload)
        _patch_legacy_sklearn_estimator_compatibility(bundle.model)
        return bundle

    def save(self, path: str | Path) -> Path:
        resolved = _resolve_bundle_path(path, "conflict_bundle.pkl")
        _save_pickle(resolved, self)
        return resolved

    def predict_conflict_probability(
        self,
        query: str,
        contexts: list[str],
        doc_ids: list[str] | None = None,
        *,
        table_kg_only: bool = False,
        probe_k: int = 12,
        max_literals_per_doc: int = 0,
    ) -> tuple[float, np.ndarray]:
        feature_version = int(self.metadata.get("feature_version", 1) or 1)
        feats = build_conflict_feature_vector(
            query,
            contexts,
            doc_ids=doc_ids,
            table_kg_only=bool(table_kg_only),
            probe_k=int(probe_k),
            max_literals_per_doc=int(max_literals_per_doc),
        )
        if feature_version >= 1 and len(self.feature_names) != len(CONFLICT_FEATURE_NAMES):
            raise RuntimeError(
                f"conflict bundle feature mismatch: expected {len(CONFLICT_FEATURE_NAMES)}, got {len(self.feature_names)}"
            )
        prob = _positive_class_probability(self.model, feats)
        return prob, feats
