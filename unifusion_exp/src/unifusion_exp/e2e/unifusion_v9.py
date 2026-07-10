from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Callable, Sequence

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
    "import", "representative", "senator",
}


@dataclass(frozen=True)
class V9CandidateConfig:
    dense_pool_k: int = 1200
    sparse_pool_k: int = 1800
    candidate_pool_k: int = 900
    graph_seed_k: int = 36
    graph_window: int = 1
    preserve_top: int = 0
    base_weight: float = 0.28
    dense_weight: float = 0.30
    sparse_weight: float = 0.20
    probe_weight: float = 0.16
    graph_weight: float = 0.08
    slot_weight: float = 0.08
    diversity_weight: float = 0.018
    modality_weight: float = 0.04


@dataclass
class V9Trace:
    input_candidate_count: int = 0
    output_candidate_count: int = 0
    dense_added: int = 0
    sparse_added: int = 0
    graph_added: int = 0
    probe_count: int = 0
    complex_need: float = 0.0
    direct_need: float = 0.0
    changed_count: int = 0
    slot_coverage: float = 0.0


def _tokens(text: str | None) -> list[str]:
    return TOKEN_RE.findall(str(text or "").lower())


def _content_tokens(text: str | None) -> set[str]:
    return {tok for tok in _tokens(text) if len(tok) > 1 and tok not in STOPWORDS}


def _minmax(values: Sequence[float]) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float32).reshape(-1)
    if arr.size == 0:
        return arr
    lo = float(np.min(arr))
    hi = float(np.max(arr))
    if hi <= lo + 1e-9:
        return np.zeros_like(arr, dtype=np.float32)
    return ((arr - lo) / (hi - lo)).astype(np.float32)


def _jaccard(a: set[str], b: set[str]) -> float:
    if not a or not b:
        return 0.0
    inter = len(a & b)
    return float(inter / max(1, len(a | b))) if inter else 0.0


def _overlap(a: set[str], b: set[str]) -> float:
    if not a or not b:
        return 0.0
    return float(len(a & b) / max(1, len(a)))


def _family_key(doc_id: str) -> str:
    m = CHUNK_ID_RE.match(str(doc_id))
    return m.group(1) if m else str(doc_id)


def _neighbor_ids(doc_id: str, window: int) -> list[str]:
    m = CHUNK_ID_RE.match(str(doc_id))
    if not m:
        return []
    base, sep, raw_idx = m.group(1), m.group(2), m.group(3)
    idx = int(raw_idx)
    width = len(raw_idx)
    out: list[str] = []
    for offset in range(-int(window), int(window) + 1):
        if offset == 0:
            continue
        nxt = idx + offset
        if nxt < 0:
            continue
        if width > 1 and raw_idx.startswith("0"):
            out.append(f"{base}{sep}{nxt:0{width}d}")
        else:
            out.append(f"{base}{sep}{nxt}")
    return out


def _topk(scores: np.ndarray, k: int) -> list[int]:
    kk = max(1, min(int(k), len(scores)))
    if len(scores) <= kk:
        return np.argsort(-scores).astype(np.int64).tolist()
    idx = np.argpartition(-scores, kth=kk - 1)[:kk]
    return idx[np.argsort(-scores[idx])].astype(np.int64).tolist()


def _direct_need(query_text: str, target_type: str) -> float:
    toks = _tokens(query_text)
    score = 0.0
    if len(toks) <= 8:
        score += 0.45
    elif len(toks) <= 12:
        score += 0.30
    if re.search(r"^(?:what|who|where|when|which|how many|how much|name)\b", query_text.lower()):
        score += 0.25
    if target_type in {"number", "year", "entity", "person", "location"}:
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
        if key not in seen:
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
    if target_type in {"person", "location", "entity"}:
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


def expand_v9_candidates(
    *,
    query_text: str,
    candidate_idxs: Sequence[int],
    candidate_base_scores: Sequence[float],
    dense_scores: np.ndarray,
    sparse_scores: np.ndarray,
    doc_ids: list[str],
    doc_id_to_idx: dict[str, int] | None,
    doc_tokens: Sequence[set[str]],
    router_prob: Sequence[float],
    target_type: str,
    source_bucket_fn: Callable[[str], str],
    config: V9CandidateConfig | None = None,
) -> tuple[np.ndarray, np.ndarray, V9Trace]:
    cfg = config or V9CandidateConfig()
    q_tokens = _content_tokens(query_text)
    target = str(target_type or "").lower()
    probes = _query_probes(query_text, target)
    direct_need = _direct_need(query_text, target)
    complex_need = _complex_need(query_text, probes)

    score_map = {
        int(j): float(score)
        for j, score in zip(candidate_idxs, candidate_base_scores)
        if 0 <= int(j) < len(doc_ids)
    }
    input_count = len(score_map)
    dense_added = 0
    sparse_added = 0
    graph_added = 0

    dense_top = _topk(dense_scores, int(cfg.dense_pool_k))
    sparse_top = _topk(sparse_scores, int(cfg.sparse_pool_k))
    for j in dense_top:
        if j not in score_map:
            score_map[j] = 0.0
            dense_added += 1
    for j in sparse_top:
        if j not in score_map:
            score_map[j] = 0.0
            sparse_added += 1

    graph_bonus: dict[int, float] = {}
    if doc_id_to_idx and int(cfg.graph_window) > 0:
        seed_scores = {
            int(j): float(score_map.get(int(j), 0.0)) + 0.25 * float(dense_scores[int(j)]) + 0.15 * float(sparse_scores[int(j)])
            for j in list(score_map)
        }
        seeds = sorted(seed_scores, key=lambda j: seed_scores[j], reverse=True)[: int(cfg.graph_seed_k)]
        for seed_rank, seed in enumerate(seeds):
            if seed < 0 or seed >= len(doc_ids):
                continue
            seed_bonus = 1.0 / float(seed_rank + 2)
            for nid in _neighbor_ids(doc_ids[seed], int(cfg.graph_window)):
                j = doc_id_to_idx.get(nid)
                if j is None or j < 0 or j >= len(doc_ids):
                    continue
                graph_bonus[j] = max(graph_bonus.get(j, 0.0), seed_bonus)
                if j not in score_map:
                    score_map[j] = 0.0
                    graph_added += 1

    pool = list(score_map)
    if not pool:
        return np.asarray([], dtype=np.int64), np.asarray([], dtype=np.float32), V9Trace()

    base_norm = _minmax([score_map[j] for j in pool])
    dense_norm = _minmax([float(dense_scores[j]) for j in pool])
    sparse_norm = _minmax([float(sparse_scores[j]) for j in pool])
    probe_norm = _minmax([_probe_score(doc_tokens[j], probes) for j in pool])
    graph_norm = _minmax([graph_bonus.get(j, 0.0) for j in pool])
    lexical = np.asarray([_overlap(q_tokens, doc_tokens[j]) for j in pool], dtype=np.float32)

    router = np.asarray(router_prob, dtype=np.float32).reshape(-1)
    if router.size < 3:
        router = np.full(3, 1.0 / 3.0, dtype=np.float32)
    bucket_prior = []
    for j in pool:
        bucket = source_bucket_fn(doc_ids[j])
        if bucket == "table":
            bucket_prior.append(float(router[1]))
        elif bucket == "kg":
            bucket_prior.append(float(router[2]))
        else:
            bucket_prior.append(float(router[0]))
    bucket_norm = _minmax(bucket_prior)

    base_w = float(cfg.base_weight) * (1.0 - 0.25 * direct_need)
    dense_w = float(cfg.dense_weight) * (1.0 + 0.20 * direct_need)
    sparse_w = float(cfg.sparse_weight) * (1.0 + 0.15 * max(direct_need, complex_need))
    probe_w = float(cfg.probe_weight) * (1.0 + 0.35 * complex_need)
    graph_w = float(cfg.graph_weight) * (1.0 + 0.25 * max(direct_need, complex_need))
    final = (
        base_w * base_norm
        + dense_w * dense_norm
        + sparse_w * sparse_norm
        + probe_w * probe_norm
        + graph_w * graph_norm
        + 0.05 * lexical
        + float(cfg.modality_weight) * bucket_norm
    )
    if len(pool) > int(cfg.candidate_pool_k):
        keep_pos = np.argsort(final)[::-1][: int(cfg.candidate_pool_k)]
        keep = [pool[int(pos)] for pos in keep_pos.tolist()]
        keep_scores = [float(final[int(pos)]) for pos in keep_pos.tolist()]
    else:
        order = np.argsort(final)[::-1]
        keep = [pool[int(pos)] for pos in order.tolist()]
        keep_scores = [float(final[int(pos)]) for pos in order.tolist()]

    trace = V9Trace(
        input_candidate_count=input_count,
        output_candidate_count=len(keep),
        dense_added=dense_added,
        sparse_added=sparse_added,
        graph_added=graph_added,
        probe_count=len(probes),
        complex_need=complex_need,
        direct_need=direct_need,
    )
    return np.asarray(keep, dtype=np.int64), np.asarray(keep_scores, dtype=np.float32), trace


def rerank_v9_local_evidence(
    *,
    query_text: str,
    current_ranked_idxs: Sequence[int],
    candidate_idxs: Sequence[int],
    candidate_scores: Sequence[float],
    dense_scores: np.ndarray,
    sparse_scores: np.ndarray,
    doc_ids: list[str],
    doc_tokens: Sequence[set[str]],
    target_type: str,
    k: int,
    config: V9CandidateConfig | None = None,
) -> tuple[list[int], V9Trace]:
    cfg = config or V9CandidateConfig()
    topk = max(1, int(k))
    candidates = [int(j) for j in candidate_idxs if 0 <= int(j) < len(doc_ids)]
    if not candidates:
        return [int(j) for j in current_ranked_idxs[:topk]], V9Trace()

    target = str(target_type or "").lower()
    probes = _query_probes(query_text, target)
    direct_need = _direct_need(query_text, target)
    complex_need = _complex_need(query_text, probes)
    score_map = {int(j): float(s) for j, s in zip(candidate_idxs, candidate_scores)}
    dense_norm = {j: v for j, v in zip(candidates, _minmax([float(dense_scores[j]) for j in candidates]))}
    sparse_norm = {j: v for j, v in zip(candidates, _minmax([float(sparse_scores[j]) for j in candidates]))}

    preserve = min(max(0, int(cfg.preserve_top)), topk, len(current_ranked_idxs))
    selected = [int(j) for j in current_ranked_idxs[:preserve] if int(j) in set(candidates)]
    selected_set = set(selected)
    covered_slots: set[int] = set()
    for j in selected:
        _, newly = _slot_gain(doc_tokens[j], probes, covered_slots)
        covered_slots.update(newly)

    families = {j: _family_key(doc_ids[j]) for j in candidates}
    while len(selected) < topk:
        remaining = [j for j in candidates if j not in selected_set]
        if not remaining:
            break
        selected_families = {families.get(j) for j in selected}

        def score(j: int) -> float:
            slot_bonus, _ = _slot_gain(doc_tokens[j], probes, covered_slots)
            redundancy = max((_jaccard(doc_tokens[j], doc_tokens[s]) for s in selected), default=0.0)
            same_family = 1.0 if families.get(j) in selected_families else 0.0
            return (
                score_map.get(j, 0.0)
                + float(cfg.slot_weight) * (0.45 + complex_need) * slot_bonus
                + 0.035 * float(dense_norm.get(j, 0.0))
                + 0.020 * float(sparse_norm.get(j, 0.0))
                + 0.020 * direct_need * same_family
                - float(cfg.diversity_weight) * (1.0 - direct_need) * redundancy
            )

        best = max(remaining, key=score)
        selected.append(best)
        selected_set.add(best)
        _, newly = _slot_gain(doc_tokens[best], probes, covered_slots)
        covered_slots.update(newly)

    if len(selected) < topk:
        for j in candidates:
            if j not in selected_set:
                selected.append(j)
                selected_set.add(j)
            if len(selected) >= topk:
                break

    old = [int(j) for j in current_ranked_idxs[:topk]]
    out = selected[:topk]
    trace = V9Trace(
        input_candidate_count=len(candidates),
        output_candidate_count=len(candidates),
        probe_count=len(probes),
        complex_need=complex_need,
        direct_need=direct_need,
        changed_count=sum(1 for a, b in zip(out, old) if a != b),
        slot_coverage=float(len(covered_slots) / max(1, len(probes))),
    )
    return out, trace
