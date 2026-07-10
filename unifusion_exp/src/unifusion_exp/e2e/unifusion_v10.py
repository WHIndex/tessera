from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Sequence

import numpy as np


TOKEN_RE = re.compile(r"[a-z0-9]+")
CHUNK_ID_RE = re.compile(r"^(.*?)([_:\-.])(\d+)$")

STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "by", "did", "do", "does",
    "for", "from", "how", "in", "is", "it", "many", "much", "of", "on",
    "or", "the", "that", "this", "to", "was", "were", "what", "when",
    "where", "which", "who", "whom", "whose", "will", "with",
}

COMPLEX_MARKERS = {
    "after", "before", "between", "following", "located", "sponsored",
    "whose", "while", "written", "appointed", "produced", "export",
    "import", "representative", "senator", "compared", "relationship",
}


@dataclass(frozen=True)
class V10RerankConfig:
    preserve_top: int = 1
    direct_preserve_top: int = 2
    reference_pool_k: int = 40
    candidate_pool_k: int = 120
    reference_weight: float = 0.54
    current_weight: float = 0.24
    base_weight: float = 0.10
    dense_weight: float = 0.07
    sparse_weight: float = 0.04
    probe_weight: float = 0.04
    slot_weight: float = 0.03
    diversity_weight: float = 0.012
    margin: float = 0.035
    relevance_floor: float = 0.18


@dataclass
class V10Trace:
    candidate_count: int = 0
    reference_count: int = 0
    preserve_count: int = 0
    restored_from_reference: int = 0
    accepted_new: int = 0
    rejected_new: int = 0
    changed_count: int = 0
    direct_need: float = 0.0
    complex_need: float = 0.0
    effective_margin: float = 0.0
    effective_relevance_floor: float = 0.0
    slot_coverage: float = 0.0
    reference_topk_overlap_before: float = 0.0
    reference_topk_overlap_after: float = 0.0


def _tokens(text: str | None) -> list[str]:
    return TOKEN_RE.findall(str(text or "").lower())


def _content_tokens(text: str | None) -> set[str]:
    return {tok for tok in _tokens(text) if len(tok) > 1 and tok not in STOPWORDS}


def _minmax_map(keys: Sequence[int], values: Sequence[float]) -> dict[int, float]:
    arr = np.asarray(values, dtype=np.float32).reshape(-1)
    if arr.size == 0:
        return {}
    lo = float(np.min(arr))
    hi = float(np.max(arr))
    if hi <= lo + 1e-9:
        return {int(k): 0.0 for k in keys}
    norm = ((arr - lo) / (hi - lo)).astype(np.float32)
    return {int(k): float(v) for k, v in zip(keys, norm)}


def _jaccard(a: set[str], b: set[str]) -> float:
    if not a or not b:
        return 0.0
    inter = len(a & b)
    return float(inter / max(1, len(a | b))) if inter else 0.0


def _direct_need(query_text: str, target_type: str) -> float:
    toks = _tokens(query_text)
    score = 0.0
    if len(toks) <= 8:
        score += 0.45
    elif len(toks) <= 12:
        score += 0.30
    if re.search(r"^(?:what|who|where|when|which|how many|how much|name)\b", str(query_text or "").lower()):
        score += 0.25
    if str(target_type or "").lower() in {"number", "year", "entity", "person", "location"}:
        score += 0.15
    score -= 0.10 * sum(1 for tok in toks if tok in COMPLEX_MARKERS)
    return float(np.clip(score, 0.0, 1.0))


def _query_probes(query_text: str, target_type: str, max_probes: int = 8) -> list[set[str]]:
    text = str(query_text or "").lower()
    probes: list[set[str]] = []
    seen: set[frozenset[str]] = set()

    def add(tokens: set[str]) -> None:
        toks = {tok for tok in tokens if len(tok) > 1 and tok not in STOPWORDS}
        if len(toks) < 2:
            return
        key = frozenset(toks)
        if key in seen:
            return
        seen.add(key)
        probes.append(toks)

    add(_content_tokens(text))
    for part in re.split(
        r"[,;\.?]|\band\b|\bthat\b|\bwhich\b|\bwho\b|\bwhose\b|\bwhere\b|\bwhen\b|\bwhile\b|\bbefore\b|\bafter\b|\bwith\b|\bby\b|\bfrom\b|\bfor\b",
        text,
    ):
        add(_content_tokens(part))
    toks = _tokens(text)
    for i, tok in enumerate(toks):
        if tok not in {"of", "in", "from", "by", "with", "for"}:
            continue
        add({x for x in toks[max(0, i - 5):i + 6] if x not in STOPWORDS})
    if str(target_type or "").lower() in {"person", "location", "entity"}:
        add({tok for tok in toks[-7:] if tok not in STOPWORDS})
    return probes[: max(1, int(max_probes))]


def _complex_need(query_text: str, probes: Sequence[set[str]]) -> float:
    text = str(query_text or "").lower()
    toks = _tokens(text)
    marker_hits = sum(1 for tok in toks if tok in COMPLEX_MARKERS)
    marker_hits += text.count(" whose ") + text.count(" which ") + text.count(" that ")
    marker_score = min(1.0, marker_hits / 3.0)
    probe_score = min(1.0, max(0, len(probes) - 1) / 5.0)
    return float(np.clip(0.55 * marker_score + 0.45 * probe_score, 0.0, 1.0))


def _probe_score(doc_tokens: set[str], probes: Sequence[set[str]]) -> float:
    if not doc_tokens or not probes:
        return 0.0
    return float(max((len(probe & doc_tokens) / max(1, len(probe)) for probe in probes), default=0.0))


def _slot_gain(doc_tokens: set[str], probes: Sequence[set[str]], covered: set[int]) -> tuple[float, set[int]]:
    if not doc_tokens or not probes:
        return 0.0, set()
    newly: set[int] = set()
    total = 0.0
    for idx, probe in enumerate(probes):
        if idx in covered:
            continue
        overlap = len(probe & doc_tokens) / max(1, len(probe))
        if overlap >= 0.40:
            newly.add(idx)
            total += overlap
    return float(total / max(1, len(probes))), newly


def _rrf_map(rankings: Sequence[Sequence[int]], pool_k: int, offset: float = 12.0) -> dict[int, float]:
    scores: dict[int, float] = {}
    if not rankings:
        return scores
    for ranking in rankings:
        for rank, raw_j in enumerate(list(ranking)[: max(1, int(pool_k))]):
            j = int(raw_j)
            scores[j] = scores.get(j, 0.0) + 1.0 / (offset + rank + 1.0)
    if not scores:
        return scores
    max_score = max(scores.values())
    if max_score <= 0.0:
        return {j: 0.0 for j in scores}
    return {j: float(v / max_score) for j, v in scores.items()}


def _dedup_valid(seq: Sequence[int], n_docs: int) -> list[int]:
    out: list[int] = []
    seen: set[int] = set()
    for raw_j in seq:
        j = int(raw_j)
        if j < 0 or j >= n_docs or j in seen:
            continue
        seen.add(j)
        out.append(j)
    return out


def apply_v10_conservative_gate(
    *,
    query_text: str,
    current_ranked_idxs: Sequence[int],
    reference_ranked_groups: Sequence[Sequence[int]],
    candidate_idxs: Sequence[int],
    candidate_scores: Sequence[float],
    dense_scores: np.ndarray,
    sparse_scores: np.ndarray,
    doc_ids: list[str],
    doc_tokens: Sequence[set[str]],
    target_type: str,
    k: int,
    config: V10RerankConfig | None = None,
) -> tuple[list[int], V10Trace]:
    cfg = config or V10RerankConfig()
    topk = max(1, int(k))
    n_docs = len(doc_ids)
    current = _dedup_valid(current_ranked_idxs, n_docs)
    references = [_dedup_valid(group, n_docs) for group in reference_ranked_groups if group]
    references = [group for group in references if group]
    if not current:
        return [], V10Trace()
    if not references:
        return current[:topk], V10Trace(candidate_count=len(current), changed_count=0)

    target = str(target_type or "").lower()
    probes = _query_probes(query_text, target)
    direct_need = _direct_need(query_text, target)
    complex_need = _complex_need(query_text, probes)
    ref_pool_k = max(topk, int(cfg.reference_pool_k))
    cand_pool_k = max(topk, int(cfg.candidate_pool_k))

    score_map = {int(j): float(score) for j, score in zip(candidate_idxs, candidate_scores)}
    pool: list[int] = []
    seen: set[int] = set()

    def add_many(seq: Sequence[int], limit: int) -> None:
        for raw_j in list(seq)[: max(1, int(limit))]:
            j = int(raw_j)
            if j < 0 or j >= n_docs or j in seen:
                continue
            seen.add(j)
            pool.append(j)

    add_many(current, cand_pool_k)
    for group in references:
        add_many(group, ref_pool_k)
    scored_candidates = [int(j) for j in candidate_idxs if 0 <= int(j) < n_docs]
    scored_candidates = sorted(scored_candidates, key=lambda j: score_map.get(j, 0.0), reverse=True)
    add_many(scored_candidates, cand_pool_k)
    if not pool:
        return current[:topk], V10Trace(candidate_count=0, changed_count=0)

    base_norm = _minmax_map(pool, [score_map.get(j, 0.0) for j in pool])
    dense_norm = _minmax_map(pool, [float(dense_scores[j]) for j in pool])
    sparse_norm = _minmax_map(pool, [float(sparse_scores[j]) for j in pool])
    probe_norm = _minmax_map(pool, [_probe_score(doc_tokens[j], probes) for j in pool])
    ref_rrf = _rrf_map(references, ref_pool_k)
    current_rrf = _rrf_map([current], cand_pool_k)
    ref_set = {j for group in references for j in group[:ref_pool_k]}
    reference_top = references[0][:topk]
    ref_weight = float(cfg.reference_weight) * (1.0 + 0.20 * direct_need)
    current_weight = float(cfg.current_weight) * (1.0 + 0.10 * direct_need)
    base_weight = float(cfg.base_weight) * (1.0 + 0.08 * direct_need)
    dense_weight = float(cfg.dense_weight) * (1.0 + 0.06 * direct_need)
    sparse_weight = float(cfg.sparse_weight) * (1.0 + 0.08 * max(direct_need, complex_need))
    probe_weight = float(cfg.probe_weight) * (1.0 + 0.35 * complex_need)
    effective_margin = max(0.0, float(cfg.margin) * (1.0 + 0.45 * direct_need - 0.25 * complex_need))
    effective_floor = max(0.0, float(cfg.relevance_floor) + 0.04 * direct_need - 0.03 * complex_need)

    def relevance(j: int) -> float:
        return (
            0.40 * float(base_norm.get(j, 0.0))
            + 0.28 * float(dense_norm.get(j, 0.0))
            + 0.18 * float(sparse_norm.get(j, 0.0))
            + 0.14 * float(probe_norm.get(j, 0.0))
        )

    def static_score(j: int) -> float:
        return (
            ref_weight * float(ref_rrf.get(j, 0.0))
            + current_weight * float(current_rrf.get(j, 0.0))
            + base_weight * float(base_norm.get(j, 0.0))
            + dense_weight * float(dense_norm.get(j, 0.0))
            + sparse_weight * float(sparse_norm.get(j, 0.0))
            + probe_weight * float(probe_norm.get(j, 0.0))
        )

    preserve = int(cfg.direct_preserve_top) if direct_need >= 0.55 else int(cfg.preserve_top)
    preserve = min(max(0, preserve), topk, len(reference_top))
    selected: list[int] = []
    selected_set: set[int] = set()
    covered_slots: set[int] = set()
    for j in reference_top[:preserve]:
        if j in selected_set:
            continue
        selected.append(j)
        selected_set.add(j)
        _, newly = _slot_gain(doc_tokens[j], probes, covered_slots)
        covered_slots.update(newly)

    rejected_new = 0
    accepted_new = 0
    remaining_ref_scores = [static_score(j) for j in reference_top if j not in selected_set]
    ref_bar = max(remaining_ref_scores) if remaining_ref_scores else 0.0

    while len(selected) < topk:
        remaining = [j for j in pool if j not in selected_set]
        if not remaining:
            break

        def dynamic_score(j: int) -> float:
            slot_bonus, _ = _slot_gain(doc_tokens[j], probes, covered_slots)
            redundancy = max((_jaccard(doc_tokens[j], doc_tokens[s]) for s in selected), default=0.0)
            return (
                static_score(j)
                + float(cfg.slot_weight) * (0.45 + complex_need) * slot_bonus
                - float(cfg.diversity_weight) * (1.0 - direct_need) * redundancy
            )

        ordered = sorted(remaining, key=dynamic_score, reverse=True)
        best = None
        for j in ordered:
            dyn = dynamic_score(j)
            low_rel = relevance(j) < effective_floor
            weak_new = j not in ref_set and dyn + effective_margin < ref_bar
            if low_rel and weak_new and len(ordered) > topk - len(selected):
                rejected_new += 1
                continue
            best = j
            break
        if best is None:
            best = ordered[0]
        if best not in ref_set:
            accepted_new += 1
        selected.append(best)
        selected_set.add(best)
        _, newly = _slot_gain(doc_tokens[best], probes, covered_slots)
        covered_slots.update(newly)

    if len(selected) < topk:
        for group in [reference_top, current, pool]:
            for j in group:
                if j in selected_set:
                    continue
                selected.append(j)
                selected_set.add(j)
                if len(selected) >= topk:
                    break
            if len(selected) >= topk:
                break

    out = selected[:topk]
    old = current[:topk]
    restored = sum(1 for j in out if j in ref_set and j not in set(old))
    before_overlap = len(set(old) & set(reference_top)) / max(1, topk)
    after_overlap = len(set(out) & set(reference_top)) / max(1, topk)
    trace = V10Trace(
        candidate_count=len(pool),
        reference_count=len(ref_set),
        preserve_count=preserve,
        restored_from_reference=restored,
        accepted_new=accepted_new,
        rejected_new=rejected_new,
        changed_count=sum(1 for a, b in zip(out, old) if a != b),
        direct_need=direct_need,
        complex_need=complex_need,
        effective_margin=float(effective_margin),
        effective_relevance_floor=float(effective_floor),
        slot_coverage=float(len(covered_slots) / max(1, len(probes))),
        reference_topk_overlap_before=float(before_overlap),
        reference_topk_overlap_after=float(after_overlap),
    )
    return out, trace
