from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Callable, Sequence

import numpy as np


TOKEN_RE = re.compile(r"[a-z0-9]+")
CHUNK_ID_RE = re.compile(r"^(.*?)([_:\-.])(\d+)$")

STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "by",
    "did",
    "do",
    "does",
    "for",
    "from",
    "how",
    "in",
    "is",
    "it",
    "many",
    "much",
    "of",
    "on",
    "or",
    "the",
    "that",
    "this",
    "to",
    "was",
    "were",
    "what",
    "when",
    "where",
    "which",
    "who",
    "whom",
    "whose",
    "will",
    "with",
}

COMPLEX_TERMS = {
    "after",
    "before",
    "between",
    "following",
    "located",
    "sponsored",
    "whose",
    "while",
    "written",
}

PLURAL_COVERAGE_TERMS = {
    "all",
    "both",
    "current",
    "episodes",
    "examples",
    "members",
    "senators",
    "several",
    "states",
    "types",
}


@dataclass(frozen=True)
class GraphEvidenceConfig:
    candidate_pool_k: int = 420
    dense_pool_k: int = 260
    sparse_pool_k: int = 220
    graph_seed_k: int = 18
    graph_window: int = 1
    preserve_top: int = 2
    trigger_threshold: float = 0.58
    base_weight: float = 0.28
    dense_weight: float = 0.34
    sparse_weight: float = 0.14
    probe_weight: float = 0.12
    graph_weight: float = 0.06
    slot_weight: float = 0.06
    sibling_weight: float = 0.04
    redundancy_weight: float = 0.012


@dataclass
class GraphEvidenceTrace:
    triggered: bool = False
    pool_size: int = 0
    graph_added: int = 0
    probe_count: int = 0
    coverage_need: float = 0.0
    complex_need: float = 0.0
    changed_count: int = 0


@dataclass
class GraphCandidateExpansionTrace:
    triggered: bool = False
    input_candidate_count: int = 0
    output_candidate_count: int = 0
    graph_added: int = 0
    boosted_existing: int = 0
    probe_count: int = 0
    coverage_need: float = 0.0
    complex_need: float = 0.0
    trigger_score: float = 0.0


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


def _rank_map(seq: Sequence[int]) -> dict[int, int]:
    return {int(j): pos for pos, j in enumerate(seq)}


def _rr(pos_map: dict[int, int], j: int) -> float:
    if int(j) not in pos_map:
        return 0.0
    return 1.0 / float(pos_map[int(j)] + 1)


def _jaccard(a: set[str], b: set[str]) -> float:
    if not a or not b:
        return 0.0
    inter = len(a & b)
    if inter == 0:
        return 0.0
    return float(inter / max(1, len(a | b)))


def _overlap(a: set[str], b: set[str]) -> float:
    if not a or not b:
        return 0.0
    return float(len(a & b) / max(1, len(a)))


def build_query_probes(query_text: str | None, target_type: str | None = None, max_probes: int = 7) -> list[set[str]]:
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

    full = _content_tokens(text)
    add(full)
    parts = re.split(
        r"[,;\.?]|\band\b|\bthat\b|\bwhich\b|\bwho\b|\bwhose\b|\bwhere\b|\bwhen\b|\bwhile\b|\bbefore\b|\bafter\b|\bwith\b|\bby\b|\bfrom\b|\bfor\b",
        text,
    )
    for part in parts:
        add(_content_tokens(part))

    toks = _tokens(text)
    for i, tok in enumerate(toks):
        if tok not in {"of", "in", "from", "by", "with", "for"}:
            continue
        left = {x for x in toks[max(0, i - 4):i] if x not in STOPWORDS}
        right = {x for x in toks[i + 1:i + 6] if x not in STOPWORDS}
        add(left | right)

    target = str(target_type or "").lower()
    if target in {"person", "location", "entity"}:
        tail = {tok for tok in toks[-6:] if tok not in STOPWORDS}
        add(tail)
    return probes[: max(1, int(max_probes))]


def _direct_factoid_score(query_text: str | None, target_type: str | None) -> float:
    text = str(query_text or "").strip().lower()
    toks = _tokens(text)
    if not toks:
        return 0.0
    score = 0.0
    if len(toks) <= 8:
        score += 0.45
    elif len(toks) <= 12:
        score += 0.32
    elif len(toks) <= 16:
        score += 0.18
    if re.search(r"^(?:what|who|where|when|which|how many|how much|name)\b", text):
        score += 0.28
    if str(target_type or "").lower() in {"entity", "person", "location", "year", "number"}:
        score += 0.12
    marker_hits = sum(1 for tok in toks if tok in COMPLEX_TERMS)
    marker_hits += text.count(" whose ") + text.count(" which ") + text.count(" that ")
    score -= 0.11 * marker_hits
    return float(np.clip(score, 0.0, 1.0))


def _dense_flatness(dense_scores: np.ndarray, dense_ranked_idxs: Sequence[int]) -> float:
    ranked = [int(j) for j in dense_ranked_idxs[:8] if 0 <= int(j) < len(dense_scores)]
    if len(ranked) < 5:
        return 0.0
    vals = np.asarray([float(dense_scores[j]) for j in ranked[:8]], dtype=np.float32)
    top = float(vals[0])
    fifth = float(vals[min(4, len(vals) - 1)])
    spread = max(1e-6, float(np.std(vals)) + abs(top) * 0.02)
    return float(np.clip(1.0 - ((top - fifth) / (spread * 4.0)), 0.0, 1.0))


def _coverage_need(query_text: str | None, target_type: str, dense_scores: np.ndarray, dense_ranked_idxs: Sequence[int]) -> float:
    text = str(query_text or "").lower()
    toks = _tokens(text)
    direct = _direct_factoid_score(text, target_type)
    flat = _dense_flatness(dense_scores, dense_ranked_idxs)
    plural = 1.0 if any(tok in PLURAL_COVERAGE_TERMS for tok in toks) else 0.0
    short = 1.0 if 0 < len(toks) <= 11 else 0.0
    target_bonus = 1.0 if str(target_type or "").lower() in {"entity", "person", "location", "year", "number"} else 0.0
    score = 0.50 * direct + 0.20 * flat + 0.12 * plural + 0.10 * short + 0.08 * target_bonus
    return float(np.clip(score, 0.0, 1.0))


def _complex_need(query_text: str | None, probes: Sequence[set[str]]) -> float:
    text = str(query_text or "").lower()
    toks = _tokens(text)
    markers = sum(1 for tok in toks if tok in COMPLEX_TERMS)
    markers += text.count(" whose ") + text.count(" which ") + text.count(" that ")
    marker_score = min(1.0, markers / 3.0)
    probe_score = min(1.0, max(0, len(probes) - 1) / 4.0)
    return float(np.clip(0.55 * marker_score + 0.45 * probe_score, 0.0, 1.0))


def _probe_score(doc_tokens: set[str], probes: Sequence[set[str]]) -> float:
    if not doc_tokens or not probes:
        return 0.0
    best = 0.0
    for probe in probes:
        if not probe:
            continue
        ov = len(probe & doc_tokens) / max(1, len(probe))
        if ov > best:
            best = ov
    return float(best)


def _slot_gain(doc_tokens: set[str], probes: Sequence[set[str]], covered: set[int]) -> tuple[float, set[int]]:
    if not probes or not doc_tokens:
        return 0.0, set()
    newly: set[int] = set()
    gain = 0.0
    for idx, probe in enumerate(probes):
        if idx in covered or not probe:
            continue
        ov = len(probe & doc_tokens) / max(1, len(probe))
        if ov >= 0.40:
            newly.add(idx)
            gain += ov
    return float(gain / max(1, len(probes))), newly


def _family_key(doc_id: str) -> str:
    m = CHUNK_ID_RE.match(str(doc_id))
    if not m:
        return str(doc_id)
    return m.group(1)


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
        out.append(f"{base}{sep}{nxt:0{width}d}" if width > 1 and raw_idx.startswith("0") else f"{base}{sep}{nxt}")
    return out


def expand_graph_evidence_candidates(
    *,
    query_text: str,
    current_ranked_idxs: list[int],
    candidate_idxs: Sequence[int],
    candidate_base_scores: Sequence[float],
    dense_scores: np.ndarray,
    sparse_scores: np.ndarray,
    dense_ranked_idxs: Sequence[int],
    sparse_ranked_idxs: Sequence[int],
    doc_ids: list[str],
    doc_id_to_idx: dict[str, int] | None,
    doc_tokens: Sequence[set[str]],
    target_type: str,
    config: GraphEvidenceConfig | None = None,
) -> tuple[np.ndarray, np.ndarray, GraphCandidateExpansionTrace]:
    cfg = config or GraphEvidenceConfig()
    cand = [int(j) for j in candidate_idxs if 0 <= int(j) < len(doc_ids)]
    base_scores = [float(s) for s in candidate_base_scores]
    score_map = {j: float(s) for j, s in zip(cand, base_scores)}
    input_count = len(score_map)

    probes = build_query_probes(query_text, target_type)
    coverage_need = _coverage_need(query_text, target_type, dense_scores, dense_ranked_idxs)
    complex_need = _complex_need(query_text, probes)
    trigger_score = max(coverage_need, 0.75 * complex_need)
    if trigger_score < float(cfg.trigger_threshold):
        trace = GraphCandidateExpansionTrace(
            triggered=False,
            input_candidate_count=input_count,
            output_candidate_count=input_count,
            probe_count=len(probes),
            coverage_need=coverage_need,
            complex_need=complex_need,
            trigger_score=trigger_score,
        )
        return np.asarray(cand, dtype=np.int64), np.asarray([score_map[j] for j in cand], dtype=np.float32), trace

    graph_bonus: dict[int, float] = {}
    graph_added = 0
    boosted_existing = 0
    if doc_id_to_idx and int(cfg.graph_window) > 0:
        seeds: list[int] = []
        seeds.extend([int(j) for j in current_ranked_idxs[: int(cfg.graph_seed_k)]])
        seeds.extend([int(j) for j in dense_ranked_idxs[: int(cfg.graph_seed_k)]])
        seeds.extend([int(j) for j in sparse_ranked_idxs[: max(1, int(cfg.graph_seed_k) // 2)]])
        for seed_rank, seed in enumerate(seeds):
            if seed < 0 or seed >= len(doc_ids):
                continue
            seed_score = score_map.get(seed, 0.0)
            seed_bonus = (1.0 / float(seed_rank + 2)) + 0.20 * max(0.0, seed_score)
            for nid in _neighbor_ids(doc_ids[seed], int(cfg.graph_window)):
                j = doc_id_to_idx.get(nid)
                if j is None or j < 0 or j >= len(doc_ids):
                    continue
                if j in graph_bonus:
                    graph_bonus[j] = max(graph_bonus[j], seed_bonus)
                else:
                    graph_bonus[j] = seed_bonus
                if j not in score_map:
                    score_map[j] = 0.0
                    graph_added += 1
                else:
                    boosted_existing += 1

    if not score_map:
        return np.asarray([], dtype=np.int64), np.asarray([], dtype=np.float32), GraphCandidateExpansionTrace()

    pool = list(score_map)
    dense_norm = _minmax([float(dense_scores[j]) for j in pool])
    sparse_norm = _minmax([float(sparse_scores[j]) for j in pool])
    probe_norm = _minmax([_probe_score(doc_tokens[j], probes) for j in pool])
    graph_norm = _minmax([graph_bonus.get(j, 0.0) for j in pool])
    lexical = np.asarray([_overlap(_content_tokens(query_text), doc_tokens[j]) for j in pool], dtype=np.float32)

    expanded_scores: dict[int, float] = {}
    for pos, j in enumerate(pool):
        prior = score_map.get(j, 0.0)
        expansion_bonus = (
            0.18 * coverage_need * float(dense_norm[pos])
            + 0.06 * coverage_need * float(sparse_norm[pos])
            + 0.10 * max(coverage_need, complex_need) * float(probe_norm[pos])
            + 0.08 * max(coverage_need, complex_need) * float(graph_norm[pos])
            + 0.03 * complex_need * float(lexical[pos])
        )
        expanded_scores[j] = float(prior + expansion_bonus)

    if len(expanded_scores) > int(cfg.candidate_pool_k):
        keep = sorted(expanded_scores, key=lambda j: expanded_scores[j], reverse=True)[: int(cfg.candidate_pool_k)]
    else:
        keep = sorted(expanded_scores, key=lambda j: expanded_scores[j], reverse=True)

    trace = GraphCandidateExpansionTrace(
        triggered=True,
        input_candidate_count=input_count,
        output_candidate_count=len(keep),
        graph_added=graph_added,
        boosted_existing=boosted_existing,
        probe_count=len(probes),
        coverage_need=coverage_need,
        complex_need=complex_need,
        trigger_score=trigger_score,
    )
    return (
        np.asarray(keep, dtype=np.int64),
        np.asarray([expanded_scores[j] for j in keep], dtype=np.float32),
        trace,
    )


def expand_and_rerank_graph_evidence(
    *,
    query_text: str,
    current_ranked_idxs: list[int],
    candidate_idxs: Sequence[int],
    candidate_base_scores: Sequence[float],
    dense_scores: np.ndarray,
    sparse_scores: np.ndarray,
    dense_ranked_idxs: Sequence[int],
    sparse_ranked_idxs: Sequence[int],
    doc_ids: list[str],
    doc_id_to_idx: dict[str, int] | None,
    doc_tokens: Sequence[set[str]],
    target_type: str,
    source_bucket_fn: Callable[[str], str],
    k: int,
    config: GraphEvidenceConfig | None = None,
) -> tuple[list[int], GraphEvidenceTrace]:
    cfg = config or GraphEvidenceConfig()
    topk = max(1, int(k))
    if not current_ranked_idxs:
        return [], GraphEvidenceTrace()

    probes = build_query_probes(query_text, target_type)
    coverage_need = _coverage_need(query_text, target_type, dense_scores, dense_ranked_idxs)
    complex_need = _complex_need(query_text, probes)
    trigger_score = max(coverage_need, 0.75 * complex_need)
    if trigger_score < float(cfg.trigger_threshold):
        return current_ranked_idxs[:topk], GraphEvidenceTrace(
            triggered=False,
            probe_count=len(probes),
            coverage_need=coverage_need,
            complex_need=complex_need,
        )

    base_score_map = {int(j): float(sc) for j, sc in zip(candidate_idxs, candidate_base_scores)}
    pool: list[int] = []
    seen: set[int] = set()
    graph_bonus: dict[int, float] = {}

    def push(seq: Sequence[int], limit: int) -> None:
        for raw in list(seq)[: max(0, int(limit))]:
            j = int(raw)
            if j in seen or j < 0 or j >= len(doc_ids):
                continue
            seen.add(j)
            pool.append(j)

    push(current_ranked_idxs, max(topk * 6, int(cfg.candidate_pool_k) // 4))
    base_sorted = sorted(base_score_map, key=lambda j: base_score_map.get(j, 0.0), reverse=True)
    push(base_sorted, int(cfg.candidate_pool_k))
    push(dense_ranked_idxs, int(cfg.dense_pool_k))
    push(sparse_ranked_idxs, int(cfg.sparse_pool_k))

    graph_added = 0
    if doc_id_to_idx and int(cfg.graph_window) > 0:
        seeds: list[int] = []
        seeds.extend([int(j) for j in current_ranked_idxs[: int(cfg.graph_seed_k)]])
        seeds.extend([int(j) for j in dense_ranked_idxs[: int(cfg.graph_seed_k)]])
        seeds.extend([int(j) for j in sparse_ranked_idxs[: max(1, int(cfg.graph_seed_k) // 2)]])
        for seed_rank, seed in enumerate(seeds):
            if seed < 0 or seed >= len(doc_ids):
                continue
            seed_bonus = 1.0 / float(seed_rank + 2)
            for nid in _neighbor_ids(doc_ids[seed], int(cfg.graph_window)):
                j = doc_id_to_idx.get(nid)
                if j is None or j < 0 or j >= len(doc_ids):
                    continue
                graph_bonus[j] = max(graph_bonus.get(j, 0.0), seed_bonus)
                if j in seen:
                    continue
                seen.add(j)
                pool.append(j)
                graph_added += 1

    if not pool:
        return current_ranked_idxs[:topk], GraphEvidenceTrace()

    if len(pool) > int(cfg.candidate_pool_k):
        pre_scores = []
        for j in pool:
            pre_scores.append(
                0.46 * float(dense_scores[j])
                + 0.20 * float(sparse_scores[j])
                + 0.24 * base_score_map.get(j, 0.0)
                + 0.10 * graph_bonus.get(j, 0.0)
            )
        keep_pos = np.argsort(np.asarray(pre_scores, dtype=np.float32))[::-1][: int(cfg.candidate_pool_k)]
        pool = [pool[int(pos)] for pos in keep_pos.tolist()]

    base_norm = _minmax([base_score_map.get(j, 0.0) for j in pool])
    dense_norm = _minmax([float(dense_scores[j]) for j in pool])
    sparse_norm = _minmax([float(sparse_scores[j]) for j in pool])
    probe_norm = _minmax([_probe_score(doc_tokens[j], probes) for j in pool])
    graph_norm = _minmax([graph_bonus.get(j, 0.0) for j in pool])
    lexical = np.asarray([_overlap(_content_tokens(query_text), doc_tokens[j]) for j in pool], dtype=np.float32)

    direct_boost = coverage_need
    complex_boost = complex_need
    base_w = float(cfg.base_weight) * (1.0 - 0.35 * direct_boost)
    dense_w = float(cfg.dense_weight) * (1.0 + 0.35 * direct_boost)
    sparse_w = float(cfg.sparse_weight) * (1.0 + 0.15 * direct_boost)
    probe_w = float(cfg.probe_weight) * (1.0 + 0.25 * complex_boost)
    graph_w = float(cfg.graph_weight) * (1.0 + 0.20 * max(direct_boost, complex_boost))
    lexical_w = 0.04 + 0.04 * complex_boost
    final = (
        base_w * base_norm
        + dense_w * dense_norm
        + sparse_w * sparse_norm
        + probe_w * probe_norm
        + graph_w * graph_norm
        + lexical_w * lexical
    )
    score_map = {j: float(sc) for j, sc in zip(pool, final)}
    dense_pos = _rank_map(dense_ranked_idxs)
    sparse_pos = _rank_map(sparse_ranked_idxs)
    family = {_j: _family_key(doc_ids[_j]) for _j in pool}

    preserve_raw = max(0, int(cfg.preserve_top))
    if coverage_need >= 0.72 and complex_need < 0.45:
        preserve_raw = min(preserve_raw, 1)
    elif coverage_need < 0.62 and complex_need < 0.75:
        preserve_raw = max(preserve_raw, 3)
    preserve_n = min(preserve_raw, len(current_ranked_idxs), topk)
    selected = [int(j) for j in current_ranked_idxs[:preserve_n]]
    selected_set = set(selected)
    covered_slots: set[int] = set()
    for j in selected:
        _, newly = _slot_gain(doc_tokens[j], probes, covered_slots)
        covered_slots.update(newly)

    while len(selected) < topk:
        remaining = [j for j in pool if j not in selected_set]
        if not remaining:
            break

        selected_families = {_family_key(doc_ids[j]) for j in selected}

        def score(j: int) -> float:
            slot_bonus, _ = _slot_gain(doc_tokens[j], probes, covered_slots)
            same_family = 1.0 if family.get(j) in selected_families else 0.0
            redundancy = max((_jaccard(doc_tokens[j], doc_tokens[s]) for s in selected), default=0.0)
            bucket = source_bucket_fn(doc_ids[j])
            bucket_bonus = 0.015 if bucket == "text" and coverage_need >= complex_need else 0.0
            return (
                score_map.get(j, 0.0)
                + float(cfg.slot_weight) * (0.35 + complex_boost) * slot_bonus
                + float(cfg.sibling_weight) * direct_boost * same_family
                + 0.025 * _rr(dense_pos, j)
                + 0.010 * _rr(sparse_pos, j)
                + bucket_bonus
                - float(cfg.redundancy_weight) * (1.0 - direct_boost) * redundancy
            )

        best = max(remaining, key=score)
        selected.append(best)
        selected_set.add(best)
        _, newly = _slot_gain(doc_tokens[best], probes, covered_slots)
        covered_slots.update(newly)

    if len(selected) < topk:
        for seq in (dense_ranked_idxs, sparse_ranked_idxs, current_ranked_idxs):
            for raw in seq:
                j = int(raw)
                if j in selected_set or j < 0 or j >= len(doc_ids):
                    continue
                selected.append(j)
                selected_set.add(j)
                if len(selected) >= topk:
                    break
            if len(selected) >= topk:
                break

    out = selected[:topk]
    old = [int(j) for j in current_ranked_idxs[:topk]]
    trace = GraphEvidenceTrace(
        triggered=True,
        pool_size=len(pool),
        graph_added=graph_added,
        probe_count=len(probes),
        coverage_need=coverage_need,
        complex_need=complex_need,
        changed_count=sum(1 for a, b in zip(out, old) if a != b),
    )
    return out, trace
