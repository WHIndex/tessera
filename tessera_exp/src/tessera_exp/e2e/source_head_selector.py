from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Sequence

import numpy as np


TOKEN_RE = re.compile(r"[a-z0-9]+")
STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "by", "did", "do", "does",
    "for", "from", "how", "in", "is", "it", "many", "much", "of", "on",
    "or", "the", "that", "this", "to", "was", "were", "what", "when",
    "where", "which", "who", "whom", "whose", "will", "with",
}


@dataclass(frozen=True)
class SourceHeadSelectorConfig:
    topn: int = 5
    source_weight: float = 0.42
    same_query_weight: float = 0.16
    position_weight: float = 0.16
    reference_weight: float = 0.12
    lexical_weight: float = 0.10
    base_weight: float = 0.08
    dense_weight: float = 0.05
    sparse_weight: float = 0.04
    margin: float = 0.015
    off_source_margin: float = 0.04


@dataclass
class SourceHeadSelectorTrace:
    attempted: int = 0
    changed: int = 0
    top_family_aligned_before: int = 0
    top_family_aligned_after: int = 0
    source_candidate_count: int = 0
    selected_rank: int = 0
    selected_score: float = 0.0
    old_score: float = 0.0
    target_family_count: int = 0


def _tokens(text: str | None) -> list[str]:
    return TOKEN_RE.findall(str(text or "").lower())


def _content_tokens(text: str | None) -> set[str]:
    return {tok for tok in _tokens(text) if len(tok) > 1 and tok not in STOPWORDS}


def _family(value: str | None) -> str:
    raw = str(value or "")
    if raw.startswith("m.") or raw.startswith("g."):
        return "m"
    return raw.split("_", 1)[0].lower()


def _source_bucket(doc_id: str | None) -> str:
    raw = str(doc_id or "")
    if raw.startswith("m.") or raw.startswith("/m/") or raw.startswith("g."):
        return "kg"
    prefix = raw.split("_", 1)[0].lower() if "_" in raw else raw.lower()
    if prefix in {"ott", "tat"}:
        return "table"
    return "text"


def _target_families(query_id: str) -> set[str]:
    qfam = _family(query_id)
    if qfam in {"cwq", "webqsp"}:
        return {"m"}
    if qfam in {"ott", "tat", "nq", "triviaqa"}:
        return {qfam}
    return {qfam} if qfam else set()


def _same_query_doc(query_id: str, doc_id: str) -> bool:
    q = str(query_id or "")
    d = str(doc_id or "")
    return bool(q and (d == q or d.startswith(q + "_") or d.startswith(q + "-") or d.startswith(q + ":")))


def _minmax(values: Sequence[float]) -> list[float]:
    arr = np.asarray(values, dtype=np.float32).reshape(-1)
    if arr.size == 0:
        return []
    lo = float(np.min(arr))
    hi = float(np.max(arr))
    if hi <= lo + 1e-9:
        return [0.0 for _ in arr]
    return [float(v) for v in ((arr - lo) / (hi - lo)).tolist()]


def _reference_score(candidate: int, reference_groups: Sequence[Sequence[int]], topn: int) -> float:
    best = 0.0
    for group in reference_groups:
        for rank, raw_j in enumerate(list(group)[: max(1, int(topn))]):
            if int(raw_j) != int(candidate):
                continue
            best = max(best, 1.0 - rank / max(1, int(topn) - 1))
            break
    return float(best)


def apply_source_aware_head_selector(
    *,
    query_id: str,
    query_text: str,
    current_ranked_idxs: Sequence[int],
    reference_ranked_groups: Sequence[Sequence[int]],
    candidate_idxs: Sequence[int],
    candidate_scores: Sequence[float],
    dense_scores: np.ndarray,
    sparse_scores: np.ndarray,
    doc_ids: list[str],
    doc_tokens: Sequence[set[str]],
    config: SourceHeadSelectorConfig | None = None,
    allowed_top1_sources: set[str] | None = None,
) -> tuple[list[int], SourceHeadSelectorTrace]:
    cfg = config or SourceHeadSelectorConfig()
    ranked = [int(j) for j in current_ranked_idxs if 0 <= int(j) < len(doc_ids)]
    if len(ranked) <= 1:
        return ranked, SourceHeadSelectorTrace()
    topn = min(max(1, int(cfg.topn)), len(ranked))
    head_pool = ranked[:topn]
    target_families = _target_families(query_id)
    if not target_families:
        return ranked, SourceHeadSelectorTrace()

    q_tokens = _content_tokens(query_text)
    score_map = {int(j): float(score) for j, score in zip(candidate_idxs, candidate_scores)}
    base_norm = dict(zip(head_pool, _minmax([score_map.get(j, 0.0) for j in head_pool])))
    dense_norm = dict(zip(head_pool, _minmax([float(dense_scores[j]) for j in head_pool])))
    sparse_norm = dict(zip(head_pool, _minmax([float(sparse_scores[j]) for j in head_pool])))

    def aligned(j: int) -> bool:
        return _family(doc_ids[j]) in target_families

    source_count = sum(1 for j in head_pool if aligned(j))
    if source_count == 0:
        return ranked, SourceHeadSelectorTrace(
            attempted=1,
            source_candidate_count=0,
            top_family_aligned_before=int(aligned(ranked[0])),
            top_family_aligned_after=int(aligned(ranked[0])),
            target_family_count=len(target_families),
        )

    def lexical(j: int) -> float:
        if not q_tokens:
            return 0.0
        return float(len(q_tokens & doc_tokens[j]) / max(1, len(q_tokens)))

    def score(pos: int, j: int) -> float:
        position = 1.0 - pos / max(1, topn - 1)
        src = 1.0 if aligned(j) else 0.0
        same_query = 1.0 if _same_query_doc(query_id, doc_ids[j]) else 0.0
        ref = _reference_score(j, reference_ranked_groups, topn)
        return (
            float(cfg.source_weight) * src
            + float(cfg.same_query_weight) * same_query
            + float(cfg.position_weight) * position
            + float(cfg.reference_weight) * ref
            + float(cfg.lexical_weight) * lexical(j)
            + float(cfg.base_weight) * float(base_norm.get(j, 0.0))
            + float(cfg.dense_weight) * float(dense_norm.get(j, 0.0))
            + float(cfg.sparse_weight) * float(sparse_norm.get(j, 0.0))
        )

    scored = [(score(pos, j), pos, j) for pos, j in enumerate(head_pool)]
    old_score = scored[0][0]
    best_score, best_pos, best = max(scored, key=lambda item: (item[0], -item[1]))
    old_aligned = aligned(ranked[0])
    best_aligned = aligned(best)
    if allowed_top1_sources:
        allowed = {str(x).lower() for x in allowed_top1_sources if str(x).strip()}
        if allowed and _source_bucket(doc_ids[best]) not in allowed:
            return ranked, SourceHeadSelectorTrace(
                attempted=1,
                changed=0,
                top_family_aligned_before=int(old_aligned),
                top_family_aligned_after=int(old_aligned),
                source_candidate_count=source_count,
                selected_rank=int(best_pos + 1),
                selected_score=float(best_score),
                old_score=float(old_score),
                target_family_count=len(target_families),
            )
    required_margin = float(cfg.margin)
    if not old_aligned and best_aligned:
        required_margin = -float(cfg.off_source_margin)
    if best == ranked[0] or best_score < old_score + required_margin:
        return ranked, SourceHeadSelectorTrace(
            attempted=1,
            changed=0,
            top_family_aligned_before=int(old_aligned),
            top_family_aligned_after=int(old_aligned),
            source_candidate_count=source_count,
            selected_rank=1,
            selected_score=float(old_score),
            old_score=float(old_score),
            target_family_count=len(target_families),
        )

    reordered = [best] + [j for j in ranked if j != best]
    return reordered, SourceHeadSelectorTrace(
        attempted=1,
        changed=1,
        top_family_aligned_before=int(old_aligned),
        top_family_aligned_after=int(aligned(reordered[0])),
        source_candidate_count=source_count,
        selected_rank=int(best_pos + 1),
        selected_score=float(best_score),
        old_score=float(old_score),
        target_family_count=len(target_families),
    )
