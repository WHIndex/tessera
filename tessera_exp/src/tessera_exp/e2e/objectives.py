from __future__ import annotations

import re

import numpy as np

TEXT_HINTS = {"why", "explain", "describe", "summary", "reason"}
TABLE_HINTS = {"table", "row", "column", "how many", "total", "capacity", "population", "rate", "percentage"}
KG_HINTS = {
    "relation",
    "parent",
    "subsidiary",
    "founded by",
    "owned",
    "acquired",
    "capital",
    "spouse",
    "who founded",
    "which company",
}

UPO_RELATION_HINTS = KG_HINTS | {
    "relationship",
    "related",
    "between",
    "member of",
    "part of",
    "ceo",
    "president",
    "chairman",
    "headquarters",
}

INTENT_COMPARISON_HINTS = {
    "compare",
    "difference",
    "different",
    "versus",
    "vs",
    "between",
}

INTENT_TEMPORAL_HINTS = {
    "before",
    "after",
    "earlier",
    "later",
    "during",
    "date",
    "year",
    "when",
}

INTENT_AGG_HINTS = {
    "total",
    "sum",
    "average",
    "ratio",
    "rate",
    "percentage",
    "how many",
    "how much",
    "number of",
}

COMPLEXITY_STRUCTURAL_PATTERNS = (
    r"\b(compare|difference|versus|between)\b",
    r"\b(before|after|earlier|later|during|what year|which year|date)\b",
    r"\b(total|sum|average|percentage|ratio|rate|how many|how much)\b",
    r"\b(both|either|neither|as well as|in addition to)\b",
)


def normalize_prob3(values: np.ndarray) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float32)
    arr = np.clip(arr, 1e-6, None)
    s = float(arr.sum())
    if s <= 1e-9:
        return np.asarray([1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0], dtype=np.float32)
    return (arr / s).astype(np.float32)


def infer_query_modality_prior(query: str, query_id: str | None = None) -> np.ndarray:
    q = str(query).lower()
    prior = np.asarray([0.34, 0.33, 0.33], dtype=np.float32)  # text, table, kg

    if query_id:
        prefix = str(query_id).split("_", 1)[0].lower()
        if prefix in {"tat", "ott"}:
            prior += np.asarray([-0.04, 0.14, -0.04], dtype=np.float32)
        elif prefix in {"cwq", "webqsp", "kg", "wikidata"}:
            prior += np.asarray([-0.03, -0.05, 0.15], dtype=np.float32)
        elif prefix in {"nq", "triviaqa", "hotpot", "squad", "newsqa"}:
            prior += np.asarray([0.12, -0.04, -0.04], dtype=np.float32)

    if any(x in q for x in TABLE_HINTS):
        prior += np.asarray([-0.05, 0.15, -0.03], dtype=np.float32)
    if any(x in q for x in KG_HINTS):
        prior += np.asarray([-0.04, -0.03, 0.15], dtype=np.float32)
    if any(x in q for x in TEXT_HINTS):
        prior += np.asarray([0.12, -0.05, -0.04], dtype=np.float32)

    if re.search(r"\b\d{1,3}(?:,\d{3})*(?:\.\d+)?\b", q):
        prior += np.asarray([0.0, 0.06, -0.01], dtype=np.float32)

    return normalize_prob3(prior)


def blend_router_with_query_prior(
    router_prob: np.ndarray,
    query: str,
    query_id: str | None,
    prior_mix: float,
) -> tuple[np.ndarray, np.ndarray, float, float]:
    p_router = normalize_prob3(np.asarray(router_prob, dtype=np.float32))
    mix = float(np.clip(prior_mix, 0.0, 1.0))
    p_prior = infer_query_modality_prior(query, query_id=query_id)
    disagreement = float(0.5 * np.sum(np.abs(p_router - p_prior)))
    if mix <= 1e-9:
        return p_router, p_prior, 0.0, disagreement
    p_blend = normalize_prob3((1.0 - mix) * p_router + mix * p_prior)
    return p_blend, p_prior, mix, disagreement


def blend_router_with_query_prior_adaptive(
    router_prob: np.ndarray,
    query: str,
    query_id: str | None,
    prior_mix: float,
    router_entropy: float,
    uncertainty_threshold: float,
    entropy_scale: float,
    disagreement_scale: float,
    mix_min: float,
    mix_max: float,
) -> tuple[np.ndarray, np.ndarray, float, float]:
    p_router = normalize_prob3(np.asarray(router_prob, dtype=np.float32))
    p_prior = infer_query_modality_prior(query, query_id=query_id)

    base = float(np.clip(prior_mix, 0.0, 1.0))
    disagreement = float(0.5 * np.sum(np.abs(p_router - p_prior)))

    th = float(np.clip(uncertainty_threshold, 0.0, 1.0))
    ent = float(np.clip(router_entropy, 0.0, 1.0))
    ent_factor = max(0.0, (ent - th) / max(1e-6, 1.0 - th))

    adaptive = (
        base
        + float(max(0.0, entropy_scale)) * ent_factor
        + float(max(0.0, disagreement_scale)) * disagreement
    )
    lo = float(np.clip(mix_min, 0.0, 1.0))
    hi = float(np.clip(mix_max, lo, 1.0))
    mix = float(np.clip(adaptive, lo, hi))

    if mix <= 1e-9:
        return p_router, p_prior, 0.0, disagreement
    p_blend = normalize_prob3((1.0 - mix) * p_router + mix * p_prior)
    return p_blend, p_prior, mix, disagreement


def infer_qa_target_type(query: str) -> str:
    q = str(query).strip().lower()
    if re.match(r"^(how many|how much|number of)\b", q):
        return "number"
    if re.match(r"^(when|what year|which year|in what year)\b", q):
        return "year"
    if re.match(r"^(who|which person|whose)\b", q):
        return "person"
    if re.match(r"^(where|which country|which city|in which)\b", q):
        return "location"
    if re.match(r"^(is|are|do|does|did|was|were|can|could|should)\b", q):
        return "boolean"
    if re.match(r"^(what|which)\b", q):
        return "entity"
    return "open"


def infer_query_intent_type(query: str) -> str:
    q = str(query).strip().lower()
    if any(x in q for x in INTENT_COMPARISON_HINTS):
        return "comparison"
    if any(x in q for x in INTENT_TEMPORAL_HINTS):
        return "temporal"
    if any(x in q for x in UPO_RELATION_HINTS):
        return "relation"
    if any(x in q for x in INTENT_AGG_HINTS):
        return "aggregation"

    target = infer_qa_target_type(q)
    if target == "boolean":
        return "boolean"
    if target in {"number", "year", "person", "location", "entity"}:
        return "factual"
    return "open"


def estimate_query_complexity_level(query: str) -> int:
    q_raw = str(query).strip()
    if not q_raw:
        return 1

    q = q_raw.lower()
    tokens = re.findall(r"[a-z0-9]+", q)
    token_count = len(tokens)
    score = 0

    if token_count >= 8:
        score += 1
    if token_count >= 14:
        score += 1
    if token_count >= 20:
        score += 1

    relation_hits = sum(1 for hint in UPO_RELATION_HINTS if hint in q)
    if relation_hits >= 1:
        score += 1
    if relation_hits >= 2:
        score += 1

    structural_hits = sum(1 for pat in COMPLEXITY_STRUCTURAL_PATTERNS if re.search(pat, q) is not None)
    score += min(2, structural_hits)

    conjunction_hits = len(re.findall(r"\b(and|or)\b", q))
    if conjunction_hits >= 2:
        score += 1
    if conjunction_hits >= 4:
        score += 1

    if re.search(r"\b(which|who|what)\b.*\bthat\b", q) is not None:
        score += 1

    if score <= 1:
        return 1
    if score <= 3:
        return 2
    if score <= 5:
        return 3
    if score <= 7:
        return 4
    return 5


def infer_upo_lite_concept(query: str) -> str:
    q = str(query).strip().lower()
    target = infer_qa_target_type(q)
    if target in {"number", "year", "location"}:
        return target

    has_relation_hint = any(x in q for x in UPO_RELATION_HINTS)
    if has_relation_hint:
        return "relation"

    if target in {"person", "entity"}:
        return "entity"
    if target == "boolean" and has_relation_hint:
        return "relation"
    return "open"


def infer_upo_lite_modality_prior(query: str, query_id: str | None = None) -> np.ndarray:
    concept = infer_upo_lite_concept(query)
    base = infer_query_modality_prior(query, query_id=query_id)

    concept_bias = {
        "number": np.asarray([-0.03, 0.12, -0.06], dtype=np.float32),
        "year": np.asarray([0.05, 0.07, -0.03], dtype=np.float32),
        "location": np.asarray([0.07, 0.06, -0.03], dtype=np.float32),
        "relation": np.asarray([0.04, -0.06, 0.14], dtype=np.float32),
        "entity": np.asarray([0.08, 0.01, 0.03], dtype=np.float32),
        "open": np.asarray([0.04, 0.00, -0.01], dtype=np.float32),
    }
    bias = concept_bias.get(concept, np.zeros(3, dtype=np.float32))
    return normalize_prob3(base + bias)


def qa_objective_retrieval_score(
    query: str,
    query_tokens: set[str],
    doc_text: str,
    doc_tokens: set[str],
    bucket: str,
) -> float:
    if not doc_tokens:
        return 0.0

    target = infer_qa_target_type(query)
    overlap = len(query_tokens & doc_tokens) / max(1, len(query_tokens)) if query_tokens else 0.0
    score = 0.25 * overlap

    lower = str(doc_text).lower()
    has_number = re.search(r"\b\d{1,3}(?:,\d{3})*(?:\.\d+)?\b", lower) is not None
    has_year = re.search(r"\b(1[0-9]{3}|20[0-9]{2})\b", lower) is not None
    has_yesno = bool(re.search(r"\b(yes|no|true|false)\b", lower))
    has_rel = any(x in lower for x in KG_HINTS)

    if target == "number":
        score += 0.45 if has_number else 0.0
        score += 0.10 if bucket == "table" else 0.0
    elif target == "year":
        score += 0.42 if has_year else 0.0
        score += 0.08 if bucket in {"table", "text"} else 0.0
    elif target == "person":
        score += 0.10 if bucket in {"text", "kg"} else 0.0
        score += 0.22 if has_rel else 0.0
    elif target == "location":
        score += 0.10 if bucket in {"text", "kg"} else 0.0
    elif target == "boolean":
        score += 0.30 if has_yesno else 0.0
    elif target == "entity":
        score += 0.08 if bucket in {"text", "kg", "table"} else 0.0
    else:
        score += 0.06 if bucket == "text" else 0.0

    if bucket == "kg" and has_rel:
        score += 0.08

    return float(np.clip(score, 0.0, 1.0))
