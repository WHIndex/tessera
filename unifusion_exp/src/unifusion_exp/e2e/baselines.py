from __future__ import annotations

import re
from collections import Counter
from typing import Callable

import numpy as np

TABLE_QUERY_HINTS = {
    "table",
    "row",
    "column",
    "capacity",
    "population",
    "revenue",
    "profit",
    "total",
    "how many",
    "which stadium",
}

CARP_BRIDGE_STOPWORDS = {
    "that",
    "with",
    "from",
    "this",
    "have",
    "were",
    "into",
    "about",
    "which",
    "their",
    "there",
    "after",
    "before",
    "between",
}

ROUTER_LABEL_TO_IDX = {"text": 0, "table": 1, "kg": 2}


def source_prefix(doc_id: str) -> str:
    if "_" in doc_id:
        return doc_id.split("_", 1)[0]
    if doc_id.startswith("m."):
        return "m"
    return doc_id


def source_bucket(doc_id: str) -> str:
    if doc_id.startswith("m.") or doc_id.startswith("/m/"):
        return "kg"
    p = source_prefix(doc_id)
    if p in {"tat", "ott"}:
        return "table"
    if p in {"nq", "triviaqa", "hotpot", "squad", "newsqa"}:
        return "text"
    if p in {"kg", "wikidata", "wd", "cwq", "webqsp"}:
        return "kg"
    return "text"


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


def compute_quasar_dynamic_quotas(retrieve_topk: int, text_p: float, table_p: float, kg_p: float) -> dict[str, int]:
    k = max(1, int(retrieve_topk))
    quotas = {"text": 2, "table": 2, "kg": 1}
    if k <= sum(quotas.values()):
        order = ["text", "table", "kg"]
        out = {"text": 0, "table": 0, "kg": 0}
        for b in order[:k]:
            out[b] = 1
        return out

    rem = k - sum(quotas.values())
    probs = np.asarray([max(1e-6, text_p), max(1e-6, table_p), max(1e-6, kg_p)], dtype=np.float64)
    probs = probs / probs.sum()
    buckets = ["text", "table", "kg"]

    alloc = np.floor(rem * probs).astype(np.int64)
    for bi, b in enumerate(buckets):
        quotas[b] += int(alloc[bi])
    used = int(np.sum(alloc))
    left = rem - used
    if left > 0:
        order = np.argsort(-probs)
        for t in range(left):
            quotas[buckets[int(order[t % len(order)])]] += 1

    return quotas


def table_intent_gate(query: str, table_prob: float) -> float:
    q = query.lower()
    lexical_hits = sum(1 for t in TABLE_QUERY_HINTS if t in q)
    gate = max(0.0, (float(table_prob) - 0.20) / 0.80)
    if lexical_hits >= 1:
        gate = max(gate, 0.45)
    if lexical_hits >= 2:
        gate = max(gate, 0.65)
    if re.search(r"\b\d{1,3}(?:,\d{3})*(?:\.\d+)?\b", q):
        gate = max(gate, 0.50)
    return min(1.0, gate)


def build_dense_concat_ranking(d_row: np.ndarray, retrieve_topk: int) -> tuple[list[int], np.ndarray]:
    dense_top400 = topk_indices(d_row, 400)
    return dense_top400[:retrieve_topk].tolist(), dense_top400


def build_naive_rag_ranking(
    d_row: np.ndarray,
    s_row: np.ndarray,
    retrieve_topk: int,
) -> tuple[list[int], np.ndarray]:
    sparse_top500 = topk_indices(s_row, 500)
    naive_candidates = sparse_top500[:500]
    naive_sorted = naive_candidates[np.argsort(-d_row[naive_candidates])]
    return naive_sorted[:retrieve_topk].tolist(), sparse_top500


def build_adapter_candidate_pool(
    d_row: np.ndarray,
    s_row: np.ndarray,
    dense_top400: np.ndarray,
    sparse_top500: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    candidate = np.unique(np.concatenate([dense_top400[:350], sparse_top500[:450]])).astype(np.int64)
    d_norm = normalize_scores(d_row[candidate])
    s_norm = normalize_scores(s_row[candidate])
    return candidate, d_norm, s_norm


def build_carp_ranking(
    query_tokens: set[str],
    candidate: np.ndarray,
    d_norm: np.ndarray,
    s_norm: np.ndarray,
    dense_top400: np.ndarray,
    doc_ids: list[str],
    doc_tokens: list[set[str]],
    adapter_plus_mode: bool,
    adapter_official_lite: bool,
    source_bucket_fn: Callable[[str], str],
    retrieve_topk: int,
) -> np.ndarray:
    seed_idx = dense_top400[:5]
    seed_terms = set()
    for j in seed_idx:
        seed_terms |= doc_tokens[int(j)]

    seed_overlap_vec = np.asarray(
        [len(seed_terms & doc_tokens[int(j)]) / max(1, len(seed_terms)) for j in candidate],
        dtype=np.float32,
    )
    q_overlap_vec = np.asarray(
        [len(query_tokens & doc_tokens[int(j)]) / max(1, len(query_tokens)) for j in candidate],
        dtype=np.float32,
    )

    bridge_overlap_vec = np.zeros_like(seed_overlap_vec)
    if adapter_plus_mode or adapter_official_lite:
        bridge_counter: Counter[str] = Counter()
        for j in dense_top400[:30].tolist():
            b = source_bucket_fn(doc_ids[int(j)])
            if b not in {"table", "kg"}:
                continue
            for t in doc_tokens[int(j)]:
                if len(t) < 4 or t in CARP_BRIDGE_STOPWORDS:
                    continue
                bridge_counter[t] += 1
        bridge_terms = {t for t, _ in bridge_counter.most_common(40)}
        bridge_overlap_vec = np.asarray(
            [len(bridge_terms & doc_tokens[int(j)]) / max(1, len(bridge_terms)) for j in candidate],
            dtype=np.float32,
        )

    if adapter_official_lite:
        carp_score = (
            0.58 * d_norm
            + 0.34 * s_norm
            + 0.12 * seed_overlap_vec
            + 0.10 * q_overlap_vec
            + 0.08 * bridge_overlap_vec
        )
    elif adapter_plus_mode:
        carp_score = 0.62 * d_norm + 0.38 * s_norm + 0.13 * seed_overlap_vec + 0.09 * q_overlap_vec
    else:
        carp_score = 0.62 * d_norm + 0.38 * s_norm + 0.12 * seed_overlap_vec + 0.08 * q_overlap_vec

    carp_rank = candidate[np.argsort(-carp_score)]
    return carp_rank[:retrieve_topk]


def build_tablerag_ranking(
    query_tokens: set[str],
    candidate: np.ndarray,
    d_norm: np.ndarray,
    s_norm: np.ndarray,
    doc_ids: list[str],
    doc_table_structs: list[dict[str, list[set[str]]] | None],
    table_gate: float,
    table_cellmaxsim_weight: float,
    table_cellmaxsim_top_cells: int,
    adapter_plus_mode: bool,
    adapter_official_lite: bool,
    source_bucket_fn: Callable[[str], str],
    cellmaxsim_like_score_fn: Callable[[set[str], dict[str, list[set[str]]] | None, int], float],
    table_schema_alignment_score_fn: Callable[[set[str], dict[str, list[set[str]]] | None], float],
    retrieve_topk: int,
) -> np.ndarray:
    gated_table_cell_weight = float(table_cellmaxsim_weight) * float(table_gate)

    table_bonus = np.asarray(
        [0.1 if source_bucket_fn(doc_ids[int(j)]) == "table" else 0.0 for j in candidate],
        dtype=np.float32,
    )
    table_cell_bonus = np.asarray(
        [
            cellmaxsim_like_score_fn(query_tokens, doc_table_structs[int(j)], top_cells=int(table_cellmaxsim_top_cells))
            if source_bucket_fn(doc_ids[int(j)]) == "table"
            else 0.0
            for j in candidate
        ],
        dtype=np.float32,
    )
    table_schema_bonus = np.asarray(
        [
            table_schema_alignment_score_fn(query_tokens, doc_table_structs[int(j)])
            if source_bucket_fn(doc_ids[int(j)]) == "table"
            else 0.0
            for j in candidate
        ],
        dtype=np.float32,
    )

    if adapter_official_lite:
        tab_score = (
            0.48 * d_norm
            + 0.30 * s_norm
            + (0.08 + 0.12 * table_gate) * table_bonus
            + (gated_table_cell_weight + 0.10 * table_gate) * table_cell_bonus
            + 0.08 * table_schema_bonus
        )
    elif adapter_plus_mode:
        tab_score = (
            0.50 * d_norm
            + 0.34 * s_norm
            + (0.10 + 0.10 * table_gate) * table_bonus
            + (gated_table_cell_weight + 0.08 * table_gate) * table_cell_bonus
        )
    else:
        tab_score = 0.55 * d_norm + 0.45 * s_norm + table_bonus + gated_table_cell_weight * table_cell_bonus

    tab_rank = candidate[np.argsort(-tab_score)]
    return tab_rank[:retrieve_topk]


def build_quasar_ranking(
    candidate: np.ndarray,
    d_norm: np.ndarray,
    s_norm: np.ndarray,
    doc_ids: list[str],
    text_p: float,
    table_p: float,
    kg_p: float,
    adapter_official_lite: bool,
    source_bucket_fn: Callable[[str], str],
    retrieve_topk: int,
) -> list[int]:
    qua_score = 0.65 * d_norm + 0.35 * s_norm
    qua_rank_all = candidate[np.argsort(-qua_score)]

    bucket_order = {"text": [], "table": [], "kg": []}
    for j in qua_rank_all.tolist():
        bucket_order[source_bucket_fn(doc_ids[int(j)])].append(int(j))

    merged = []
    if adapter_official_lite:
        dq = compute_quasar_dynamic_quotas(retrieve_topk, text_p=text_p, table_p=table_p, kg_p=kg_p)
        quotas = [("text", dq["text"]), ("table", dq["table"]), ("kg", dq["kg"])]
    else:
        quotas = [("text", 4), ("table", 4), ("kg", 2)]

    for b, qn in quotas:
        merged.extend(bucket_order[b][: int(qn)])

    if len(merged) < retrieve_topk:
        seen = set(merged)
        for j in qua_rank_all.tolist():
            if int(j) in seen:
                continue
            merged.append(int(j))
            if len(merged) >= retrieve_topk:
                break

    return merged[:retrieve_topk]
