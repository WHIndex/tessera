from __future__ import annotations

from dataclasses import dataclass, field
import pickle
import re
from pathlib import Path
from typing import Callable, Sequence

import numpy as np

try:
    from unifusion_exp.e2e.pairwise_slot_verifier import PESV_FEATURE_NAMES
except Exception:  # pragma: no cover - optional verifier dependency
    PESV_FEATURE_NAMES = []


NUMBER_RE = re.compile(r"(?<!\w)[+-]?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?%?(?!\w)")
YEAR_RE = re.compile(r"\b(?:1[6-9]\d{2}|20\d{2}|21\d{2})\b")
CHUNK_ID_RE = re.compile(r"^(.*?)([_:\-.])(\d+)$")


SER_FEATURE_NAMES = [
    "base_score",
    "dense_score",
    "sparse_score",
    "base_rank_rr",
    "dense_rank_rr",
    "sparse_rank_rr",
    "query_doc_overlap",
    "doc_query_precision",
    "bucket_text",
    "bucket_table",
    "bucket_kg",
    "router_bucket_prob",
    "target_number",
    "target_year",
    "target_entity",
    "target_person",
    "target_location",
    "has_number",
    "has_year",
    "table_number_match",
    "kg_entity_match",
    "doc_token_count_log",
    "query_direct_factoid",
    "query_complex_need",
    "query_slot_count_norm",
    "query_content_token_count_log",
    "query_coverage_need",
    "dense_sparse_rank_agree",
    "base_dense_rank_agree",
    "dense_top5_flatness",
    "doc_probe_score",
    "doc_slot_coverage",
    "same_family_as_dense_top1",
    "same_family_as_dense_top5",
    "same_family_as_base_top5",
    "target_sensitive_bucket",
]


@dataclass(frozen=True)
class SERRankerConfig:
    preserve_top: int = 1
    candidate_pool_k: int = 180
    dense_pool_k: int = 120
    sparse_pool_k: int = 120
    blend_weight: float = 0.65
    diversity_weight: float = 0.02
    evidence_rescue_k: int = 0
    evidence_rescue_pool_k: int = 24
    evidence_preserve_top: int = 3
    evidence_redundancy_weight: float = 0.04
    evidence_min_gain: float = 0.03
    plan_adaptive: bool = False
    plan_dense_weight: float = 0.12
    plan_sparse_weight: float = 0.03
    plan_lexical_weight: float = 0.03
    plan_slot_weight: float = 0.04
    evidence_set_selection: bool = False
    evidence_set_preserve_top: int = 2
    evidence_set_pool_k: int = 220
    evidence_set_cardinality_threshold: float = 0.46
    evidence_set_learned_weight: float = 0.22
    evidence_set_base_weight: float = 0.30
    evidence_set_dense_weight: float = 0.18
    evidence_set_sparse_weight: float = 0.10
    evidence_set_probe_weight: float = 0.10
    evidence_set_slot_weight: float = 0.12
    evidence_set_anchor_weight: float = 0.16
    evidence_set_family_weight: float = 0.05
    evidence_set_redundancy_weight: float = 0.018


@dataclass
class SERRankerTrace:
    candidate_count: int = 0
    preserve_count: int = 0
    changed_count: int = 0
    mean_score: float = 0.0
    max_score: float = 0.0
    evidence_rescue_added: int = 0
    evidence_rescue_pool: int = 0
    plan_direct_score: float = 0.0
    plan_complex_score: float = 0.0
    plan_slot_count: int = 0
    evidence_set_enabled: bool = False
    evidence_set_cardinality_need: float = 0.0
    evidence_set_slot_coverage: float = 0.0
    evidence_set_family_count: int = 0
    evidence_set_anchor_hits: int = 0


@dataclass(frozen=True)
class FinalEvidenceComposerConfig:
    topk: int = 5
    preserve_top: int = 1
    candidate_pool_k: int = 120
    dense_pool_k: int = 80
    sparse_pool_k: int = 80
    max_replacements: int = 1
    min_candidate_score: float = 0.62
    replacement_margin: float = 0.08
    min_query_overlap: float = 0.0
    protect_kg: bool = True
    source_need_threshold: float = 0.45
    source_need_weight: float = 0.035
    dense_rr_weight: float = 0.025
    sparse_rr_weight: float = 0.015
    lexical_weight: float = 0.020
    redundancy_weight: float = 0.025
    replacement_verifier_threshold: float = 0.70
    replacement_verifier_margin: float = 0.0


@dataclass
class FinalEvidenceComposerTrace:
    active: bool = False
    candidate_count: int = 0
    preserve_count: int = 0
    changed_count: int = 0
    replacement_count: int = 0
    rejected_count: int = 0
    mean_candidate_score: float = 0.0
    max_candidate_score: float = 0.0
    min_replaced_score: float = 0.0
    max_inserted_score: float = 0.0
    verifier_active: int = 0
    verifier_scored: int = 0
    verifier_accepted: int = 0
    verifier_rejected: int = 0
    verifier_max_score: float = 0.0


@dataclass
class SERRankerBundle:
    model: object
    feature_names: list[str] = field(default_factory=lambda: list(SER_FEATURE_NAMES))
    config: SERRankerConfig = field(default_factory=SERRankerConfig)
    meta: dict = field(default_factory=dict)
    source_models: dict[str, object] = field(default_factory=dict)
    source_blend_weight: float = 0.0

    def _align_features(self, features: np.ndarray) -> np.ndarray:
        arr = np.asarray(features, dtype=np.float32)
        if arr.ndim == 1:
            arr = arr.reshape(1, -1)
        expected = len(self.feature_names) if self.feature_names else len(SER_FEATURE_NAMES)
        if arr.shape[1] == expected:
            return arr
        if arr.shape[1] > expected:
            return arr[:, :expected]
        pad = np.zeros((arr.shape[0], expected - arr.shape[1]), dtype=np.float32)
        return np.hstack([arr, pad]).astype(np.float32)

    def score(self, features: np.ndarray) -> np.ndarray:
        if features.size == 0:
            return np.zeros((0,), dtype=np.float32)
        features = self._align_features(features)
        return self._score_model(self.model, features)

    def _score_model(self, model: object, features: np.ndarray) -> np.ndarray:
        if hasattr(model, "predict_proba"):
            probs = np.asarray(model.predict_proba(features), dtype=np.float32)
            if probs.ndim == 2 and probs.shape[1] >= 2:
                return probs[:, -1].astype(np.float32)
            return probs.reshape(-1).astype(np.float32)
        if hasattr(model, "decision_function"):
            raw = np.asarray(model.decision_function(features), dtype=np.float32).reshape(-1)
            return (1.0 / (1.0 + np.exp(-raw))).astype(np.float32)
        raw = np.asarray(model.predict(features), dtype=np.float32).reshape(-1)
        return raw.astype(np.float32)

    def score_by_source(self, features: np.ndarray, source_buckets: Sequence[str]) -> np.ndarray:
        global_scores = self.score(features)
        if not self.source_models or float(self.source_blend_weight) <= 0.0 or global_scores.size == 0:
            return global_scores
        features = self._align_features(features)
        blend = float(np.clip(self.source_blend_weight, 0.0, 1.0))
        out = global_scores.copy()
        buckets = [str(x) for x in source_buckets]
        for bucket, model in self.source_models.items():
            idx = [i for i, raw in enumerate(buckets) if raw == bucket]
            if not idx:
                continue
            idx_arr = np.asarray(idx, dtype=np.int64)
            source_scores = self._score_model(model, features[idx_arr])
            out[idx_arr] = (1.0 - blend) * out[idx_arr] + blend * source_scores
        return out.astype(np.float32)


def save_ser_ranker_bundle(bundle: SERRankerBundle, path: Path | str) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("wb") as f:
        pickle.dump(bundle, f)


def load_ser_ranker_bundle(path: Path | str) -> SERRankerBundle:
    with Path(path).open("rb") as f:
        obj = pickle.load(f)
    if isinstance(obj, SERRankerBundle):
        if not hasattr(obj, "source_models"):
            obj.source_models = {}
        if not hasattr(obj, "source_blend_weight"):
            obj.source_blend_weight = 0.0
        return obj
    if isinstance(obj, dict) and "model" in obj:
        return SERRankerBundle(
            model=obj["model"],
            feature_names=list(obj.get("feature_names", SER_FEATURE_NAMES)),
            config=obj.get("config", SERRankerConfig()),
            meta=dict(obj.get("meta", {})),
            source_models=dict(obj.get("source_models", {}) or {}),
            source_blend_weight=float(obj.get("source_blend_weight", 0.0) or 0.0),
        )
    raise TypeError(f"Unsupported SER ranker bundle type: {type(obj)!r}")


def minmax(values: Sequence[float]) -> np.ndarray:
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


def _overlap(query_tokens: set[str], doc_tokens: set[str]) -> float:
    if not query_tokens or not doc_tokens:
        return 0.0
    return float(len(query_tokens & doc_tokens) / max(1, len(query_tokens)))


def _jaccard(a: set[str], b: set[str]) -> float:
    if not a or not b:
        return 0.0
    return float(len(a & b) / max(1, len(a | b)))


QUERY_TOKEN_RE = re.compile(r"[a-z0-9]+")
DIRECT_START_RE = re.compile(
    r"^(?:what|who|where|when|which|how many|how much|name|the)\b",
    re.IGNORECASE,
)
COMPLEX_MARKERS = {
    "whose",
    "which",
    "that",
    "where",
    "while",
    "before",
    "after",
    "between",
    "following",
    "follows",
    "follow",
    "sponsored",
    "written",
    "wrote",
    "located",
    "born",
}
SLOT_STOPWORDS = {
    "the",
    "a",
    "an",
    "of",
    "in",
    "on",
    "for",
    "to",
    "by",
    "with",
    "and",
    "or",
    "is",
    "are",
    "was",
    "were",
    "did",
    "does",
    "do",
    "what",
    "who",
    "where",
    "when",
    "which",
    "how",
    "many",
    "much",
}


def _query_token_list(query_text: str | None) -> list[str]:
    return QUERY_TOKEN_RE.findall(str(query_text or "").lower())


def _estimate_direct_factoid_score(query_text: str | None, target_type: str) -> float:
    text = str(query_text or "").strip().lower()
    toks = _query_token_list(text)
    if not toks:
        return 0.0
    score = 0.0
    if len(toks) <= 8:
        score += 0.45
    elif len(toks) <= 12:
        score += 0.30
    elif len(toks) <= 16:
        score += 0.12
    if DIRECT_START_RE.search(text):
        score += 0.25
    if str(target_type or "").lower() in {"entity", "person", "location", "year", "number"}:
        score += 0.10
    marker_hits = sum(1 for tok in toks if tok in COMPLEX_MARKERS)
    marker_hits += text.count(" by the ") + text.count(" who ") + text.count(" that ")
    marker_hits += text.count(" whose ") + text.count(" which ")
    score -= 0.13 * marker_hits
    if "," in text or ";" in text:
        score -= 0.08
    return float(np.clip(score, 0.0, 1.0))


def _decompose_query_slots(query_text: str | None) -> list[set[str]]:
    text = str(query_text or "").lower()
    parts = re.split(
        r"[,;\.?]|\bwho\b|\bwhose\b|\bwhich\b|\bthat\b|\bwhere\b|\bwhen\b|\bwhile\b|\bbefore\b|\bafter\b|\bby\b|\bwith\b",
        text,
    )
    slots: list[set[str]] = []
    seen: set[frozenset[str]] = set()
    for part in parts:
        toks = {t for t in QUERY_TOKEN_RE.findall(part) if len(t) > 1 and t not in SLOT_STOPWORDS}
        if len(toks) < 2:
            continue
        key = frozenset(toks)
        if key in seen:
            continue
        seen.add(key)
        slots.append(toks)
    if len(slots) <= 1:
        toks = {t for t in QUERY_TOKEN_RE.findall(text) if len(t) > 1 and t not in SLOT_STOPWORDS}
        return [toks] if toks else []
    return slots[:8]


def _slot_gain(doc_tokens: set[str], slots: list[set[str]], covered: set[int]) -> tuple[float, set[int]]:
    if not slots or not doc_tokens:
        return 0.0, set()
    newly: set[int] = set()
    gain = 0.0
    for i, slot in enumerate(slots):
        if i in covered or not slot:
            continue
        overlap = len(slot & doc_tokens) / max(1, len(slot))
        if overlap >= 0.40:
            newly.add(i)
            gain += overlap
    return float(gain / max(1, len(slots))), newly


def _content_query_tokens(query_text: str | None) -> set[str]:
    return {t for t in _query_token_list(query_text) if len(t) > 1 and t not in SLOT_STOPWORDS}


def _estimate_complex_need(query_text: str | None, slots: Sequence[set[str]], direct_score: float) -> float:
    text = str(query_text or "").lower()
    toks = _query_token_list(text)
    markers = sum(1 for tok in toks if tok in COMPLEX_MARKERS)
    markers += text.count(" by the ") + text.count(" who ") + text.count(" that ")
    markers += text.count(" whose ") + text.count(" which ")
    marker_score = min(1.0, markers / 3.0)
    slot_score = min(1.0, max(0, len(slots) - 1) / 4.0)
    return float(np.clip(0.55 * marker_score + 0.30 * slot_score + 0.15 * (1.0 - direct_score), 0.0, 1.0))


def _estimate_coverage_need(query_text: str | None, target_type: str, slots: Sequence[set[str]], direct_score: float) -> float:
    toks = _query_token_list(query_text)
    plural_like = any(tok.endswith("s") for tok in toks if len(tok) > 4) or any(
        tok in {"all", "both", "members", "states", "episodes", "teams", "players", "examples"}
        for tok in toks
    )
    target_bonus = str(target_type or "").lower() in {"entity", "person", "location", "year", "number"}
    score = (
        0.38 * direct_score
        + 0.20 * min(1.0, len(slots) / 4.0)
        + 0.18 * (1.0 if plural_like else 0.0)
        + 0.14 * (1.0 if 0 < len(toks) <= 12 else 0.0)
        + 0.10 * (1.0 if target_bonus else 0.0)
    )
    return float(np.clip(score, 0.0, 1.0))


def _dense_top_flatness(dense_scores: np.ndarray, dense_ranked_idxs: Sequence[int]) -> float:
    ranked = [int(j) for j in dense_ranked_idxs[:8] if 0 <= int(j) < len(dense_scores)]
    if len(ranked) < 5:
        return 0.0
    vals = np.asarray([float(dense_scores[j]) for j in ranked], dtype=np.float32)
    spread = max(1e-6, float(np.std(vals)) + abs(float(vals[0])) * 0.02)
    return float(np.clip(1.0 - ((float(vals[0]) - float(vals[4])) / (spread * 4.0)), 0.0, 1.0))


def _rank_agreement(pos_a: dict[int, int], pos_b: dict[int, int], j: int, missing_rank: int) -> float:
    a = pos_a.get(int(j), int(missing_rank))
    b = pos_b.get(int(j), int(missing_rank))
    denom = max(1.0, float(missing_rank))
    return float(np.clip(1.0 - (abs(float(a) - float(b)) / denom), 0.0, 1.0))


def _probe_score(doc_tokens: set[str], slots: Sequence[set[str]]) -> float:
    if not doc_tokens or not slots:
        return 0.0
    best = 0.0
    for slot in slots:
        if not slot:
            continue
        best = max(best, len(slot & doc_tokens) / max(1, len(slot)))
    return float(best)


def _slot_coverage(doc_tokens: set[str], slots: Sequence[set[str]]) -> float:
    if not doc_tokens or not slots:
        return 0.0
    covered = 0
    for slot in slots:
        if not slot:
            continue
        if len(slot & doc_tokens) / max(1, len(slot)) >= 0.40:
            covered += 1
    return float(covered / max(1, len(slots)))


def _family_key(doc_id: str) -> str:
    m = CHUNK_ID_RE.match(str(doc_id))
    if not m:
        return str(doc_id)
    return m.group(1)


def _estimate_evidence_cardinality_need(
    *,
    query_text: str | None,
    target_type: str,
    query_slots: Sequence[set[str]],
    dense_scores: np.ndarray,
    dense_ranked_idxs: Sequence[int],
) -> float:
    direct = _estimate_direct_factoid_score(query_text, target_type)
    complex_need = _estimate_complex_need(query_text, query_slots, direct)
    coverage_need = _estimate_coverage_need(query_text, target_type, query_slots, direct)
    flatness = _dense_top_flatness(dense_scores, dense_ranked_idxs)
    slot_need = min(1.0, max(0, len(query_slots) - 1) / 4.0)
    text = str(query_text or "").lower()
    plural = 1.0 if any(
        term in text
        for term in (
            " all ",
            " both ",
            " members ",
            " states ",
            " teams ",
            " examples ",
            " episodes ",
            " purchases ",
            " books ",
        )
    ) else 0.0
    target_bonus = 1.0 if str(target_type or "").lower() in {"entity", "person", "location", "open"} else 0.0
    score = (
        0.30 * complex_need
        + 0.22 * coverage_need
        + 0.18 * slot_need
        + 0.14 * flatness
        + 0.10 * plural
        + 0.06 * target_bonus
    )
    return float(np.clip(score, 0.0, 1.0))


def _select_evidence_set(
    *,
    pool: list[int],
    current_ranked_idxs: list[int],
    base_scores: Sequence[float],
    learned_scores: np.ndarray,
    dense_scores: np.ndarray,
    sparse_scores: np.ndarray,
    dense_ranked_idxs: Sequence[int],
    sparse_ranked_idxs: Sequence[int],
    doc_ids: list[str],
    doc_tokens: list[set[str]],
    query_tokens: set[str],
    query_text: str | None,
    target_type: str,
    topk: int,
    config: SERRankerConfig,
) -> tuple[list[int], dict[str, float]]:
    if not pool:
        return current_ranked_idxs[:topk], {
            "cardinality_need": 0.0,
            "slot_coverage": 0.0,
            "family_count": 0.0,
            "anchor_hits": 0.0,
        }

    query_slots = _decompose_query_slots(query_text)
    if not query_slots and query_tokens:
        query_slots = [set(query_tokens)]
    cardinality_need = _estimate_evidence_cardinality_need(
        query_text=query_text,
        target_type=target_type,
        query_slots=query_slots,
        dense_scores=dense_scores,
        dense_ranked_idxs=dense_ranked_idxs,
    )
    pool_k = max(topk, int(getattr(config, "evidence_set_pool_k", 220)))
    base_norm = minmax(base_scores)
    dense_norm = minmax([float(dense_scores[j]) for j in pool])
    sparse_norm = minmax([float(sparse_scores[j]) for j in pool])
    learned_norm = minmax(learned_scores)
    probe_norm = minmax([_probe_score(doc_tokens[j], query_slots) for j in pool])
    lexical_norm = minmax([_overlap(query_tokens, doc_tokens[j]) for j in pool])

    base_pos = _rank_map(current_ranked_idxs)
    dense_pos = _rank_map(dense_ranked_idxs)
    sparse_pos = _rank_map(sparse_ranked_idxs)
    anchor_docs = {
        int(j)
        for seq in (current_ranked_idxs[:12], dense_ranked_idxs[:12], sparse_ranked_idxs[:12])
        for j in seq
        if 0 <= int(j) < len(doc_ids)
    }
    anchor_families = {_family_key(doc_ids[j]) for j in anchor_docs}

    learned_w = float(getattr(config, "evidence_set_learned_weight", 0.22)) * (1.0 - 0.45 * cardinality_need)
    base_w = float(getattr(config, "evidence_set_base_weight", 0.30)) * (1.0 + 0.35 * cardinality_need)
    dense_w = float(getattr(config, "evidence_set_dense_weight", 0.18)) * (1.0 + 0.20 * cardinality_need)
    sparse_w = float(getattr(config, "evidence_set_sparse_weight", 0.10)) * (1.0 + 0.10 * cardinality_need)
    probe_w = float(getattr(config, "evidence_set_probe_weight", 0.10)) * (1.0 + 0.30 * cardinality_need)
    lexical_w = 0.05 * cardinality_need
    evidence_score = (
        learned_w * learned_norm
        + base_w * base_norm
        + dense_w * dense_norm
        + sparse_w * sparse_norm
        + probe_w * probe_norm
        + lexical_w * lexical_norm
    )
    evidence_score_map = {j: float(sc) for j, sc in zip(pool, evidence_score)}

    if len(pool) > pool_k:
        keep = set(sorted(pool, key=lambda j: evidence_score_map.get(j, 0.0), reverse=True)[:pool_k])
        keep.update(int(j) for j in current_ranked_idxs[: max(topk * 4, 20)] if 0 <= int(j) < len(doc_ids))
        keep.update(int(j) for j in dense_ranked_idxs[: max(topk * 4, 20)] if 0 <= int(j) < len(doc_ids))
        keep.update(int(j) for j in sparse_ranked_idxs[: max(topk * 3, 15)] if 0 <= int(j) < len(doc_ids))
        pool = [j for j in pool if j in keep]

    threshold = float(getattr(config, "evidence_set_cardinality_threshold", 0.46))
    preserve_raw = int(getattr(config, "preserve_top", 1))
    if cardinality_need >= threshold:
        preserve_raw = max(preserve_raw, int(getattr(config, "evidence_set_preserve_top", 2)))
    preserve_n = min(max(0, preserve_raw), len(current_ranked_idxs), topk)
    selected = [int(j) for j in current_ranked_idxs[:preserve_n] if 0 <= int(j) < len(doc_ids)]
    selected_set = set(selected)
    covered_slots: set[int] = set()
    for j in selected:
        _, newly = _slot_gain(doc_tokens[j], query_slots, covered_slots)
        covered_slots.update(newly)

    while len(selected) < topk:
        remaining = [j for j in pool if j not in selected_set]
        if not remaining:
            break
        selected_families = {_family_key(doc_ids[j]) for j in selected}

        def score(j: int) -> float:
            slot_gain, _ = _slot_gain(doc_tokens[j], query_slots, covered_slots)
            family = _family_key(doc_ids[j])
            same_anchor_family = 1.0 if family in anchor_families else 0.0
            new_family = 1.0 if family not in selected_families else 0.0
            anchor_rr = max(_rr(base_pos, j), 0.72 * _rr(dense_pos, j), 0.55 * _rr(sparse_pos, j))
            redundancy = max((_jaccard(doc_tokens[j], doc_tokens[s]) for s in selected), default=0.0)
            sibling_bonus = 0.0
            if selected_families and family in selected_families:
                sibling_bonus = 0.5 * same_anchor_family
            family_bonus = float(getattr(config, "evidence_set_family_weight", 0.05)) * (
                same_anchor_family + sibling_bonus + 0.35 * cardinality_need * new_family
            )
            return (
                evidence_score_map.get(j, 0.0)
                + float(getattr(config, "evidence_set_slot_weight", 0.12)) * (0.25 + cardinality_need) * slot_gain
                + float(getattr(config, "evidence_set_anchor_weight", 0.16)) * (0.20 + cardinality_need) * anchor_rr
                + family_bonus
                - float(getattr(config, "evidence_set_redundancy_weight", 0.018)) * (1.0 - 0.45 * cardinality_need) * redundancy
            )

        best = max(remaining, key=score)
        selected.append(best)
        selected_set.add(best)
        _, newly = _slot_gain(doc_tokens[best], query_slots, covered_slots)
        covered_slots.update(newly)

    if len(selected) < topk:
        for seq in (current_ranked_idxs, dense_ranked_idxs, sparse_ranked_idxs, pool):
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
    slot_coverage = 0.0
    if query_slots:
        covered = 0
        for slot in query_slots:
            if any((len(slot & doc_tokens[j]) / max(1, len(slot))) >= 0.40 for j in out):
                covered += 1
        slot_coverage = covered / max(1, len(query_slots))
    family_count = len({_family_key(doc_ids[j]) for j in out})
    anchor_hits = sum(1 for j in out if j in anchor_docs)
    return out, {
        "cardinality_need": float(cardinality_need),
        "slot_coverage": float(slot_coverage),
        "family_count": float(family_count),
        "anchor_hits": float(anchor_hits),
    }


def _numeric_flags_for_doc(
    j: int,
    doc_numeric_literals: list[set[str]],
    doc_texts: list[str] | None,
) -> tuple[float, float]:
    literals: set[str] = set()
    if 0 <= int(j) < len(doc_numeric_literals):
        literals = doc_numeric_literals[int(j)]
    if literals:
        has_year = any(YEAR_RE.search(x) is not None for x in literals)
        return 1.0, 1.0 if has_year else 0.0

    if doc_texts is None or int(j) < 0 or int(j) >= len(doc_texts):
        return 0.0, 0.0
    text = doc_texts[int(j)] or ""
    has_num = NUMBER_RE.search(text) is not None
    has_year = YEAR_RE.search(text) is not None
    return 1.0 if has_num else 0.0, 1.0 if has_year else 0.0


def router_prob_for_bucket(router_prob: Sequence[float], bucket: str) -> float:
    vals = list(router_prob)
    while len(vals) < 3:
        vals.append(1.0 / 3.0)
    if bucket == "text":
        return float(vals[0])
    if bucket == "table":
        return float(vals[1])
    if bucket == "kg":
        return float(vals[2])
    return 0.0


def build_ser_feature_matrix(
    *,
    query_tokens: set[str],
    candidate_idxs: Sequence[int],
    candidate_base_scores: Sequence[float],
    dense_scores: np.ndarray,
    sparse_scores: np.ndarray,
    base_ranked_idxs: Sequence[int],
    dense_ranked_idxs: Sequence[int],
    sparse_ranked_idxs: Sequence[int],
    doc_ids: list[str],
    doc_tokens: list[set[str]],
    doc_numeric_literals: list[set[str]],
    router_prob: Sequence[float],
    target_type: str,
    source_bucket_fn: Callable[[str], str],
    query_text: str | None = None,
    doc_texts: list[str] | None = None,
) -> np.ndarray:
    cand = [int(j) for j in candidate_idxs]
    base_norm = minmax(candidate_base_scores)
    d_norm = minmax([float(dense_scores[j]) for j in cand])
    s_norm = minmax([float(sparse_scores[j]) for j in cand])
    base_pos = _rank_map(base_ranked_idxs)
    dense_pos = _rank_map(dense_ranked_idxs)
    sparse_pos = _rank_map(sparse_ranked_idxs)
    target = str(target_type or "").strip().lower()
    query_for_features = query_text if query_text is not None else " ".join(sorted(query_tokens))
    query_content = _content_query_tokens(query_for_features)
    query_slots = _decompose_query_slots(query_for_features)
    if not query_slots and query_content:
        query_slots = [query_content]
    query_direct = _estimate_direct_factoid_score(query_for_features, target)
    query_complex = _estimate_complex_need(query_for_features, query_slots, query_direct)
    query_coverage = _estimate_coverage_need(query_for_features, target, query_slots, query_direct)
    query_slot_count_norm = min(1.0, len(query_slots) / 6.0)
    query_content_count_log = float(np.log1p(len(query_content)) / 4.0)
    dense_top5_flatness = _dense_top_flatness(dense_scores, dense_ranked_idxs)
    missing_rank = max(1, len(base_ranked_idxs), len(dense_ranked_idxs), len(sparse_ranked_idxs))
    dense_top1_family = ""
    if dense_ranked_idxs:
        d0 = int(dense_ranked_idxs[0])
        if 0 <= d0 < len(doc_ids):
            dense_top1_family = _family_key(doc_ids[d0])
    dense_top5_families = {
        _family_key(doc_ids[int(j)])
        for j in dense_ranked_idxs[:5]
        if 0 <= int(j) < len(doc_ids)
    }
    base_top5_families = {
        _family_key(doc_ids[int(j)])
        for j in base_ranked_idxs[:5]
        if 0 <= int(j) < len(doc_ids)
    }

    rows: list[list[float]] = []
    for idx, j in enumerate(cand):
        bucket = source_bucket_fn(doc_ids[j])
        toks = doc_tokens[j]
        inter = len(query_tokens & toks)
        q_overlap = inter / max(1, len(query_tokens)) if query_tokens else 0.0
        precision = inter / max(1, len(toks)) if toks else 0.0
        has_num, has_year = _numeric_flags_for_doc(j, doc_numeric_literals, doc_texts)
        target_number = 1.0 if target == "number" else 0.0
        target_year = 1.0 if target == "year" else 0.0
        target_entity = 1.0 if target == "entity" else 0.0
        target_person = 1.0 if target == "person" else 0.0
        target_location = 1.0 if target == "location" else 0.0
        table_number = 1.0 if bucket == "table" and target in {"number", "year"} and (has_num or has_year) else 0.0
        kg_entity = 1.0 if bucket == "kg" and target in {"entity", "person", "location"} else 0.0
        family = _family_key(doc_ids[j])
        target_sensitive_bucket = max(
            table_number,
            kg_entity,
            1.0 if bucket == "text" and target in {"entity", "person", "location", "year"} else 0.0,
        )
        rows.append(
            [
                float(base_norm[idx]) if idx < len(base_norm) else 0.0,
                float(d_norm[idx]) if idx < len(d_norm) else 0.0,
                float(s_norm[idx]) if idx < len(s_norm) else 0.0,
                _rr(base_pos, j),
                _rr(dense_pos, j),
                _rr(sparse_pos, j),
                float(q_overlap),
                float(precision),
                1.0 if bucket == "text" else 0.0,
                1.0 if bucket == "table" else 0.0,
                1.0 if bucket == "kg" else 0.0,
                router_prob_for_bucket(router_prob, bucket),
                target_number,
                target_year,
                target_entity,
                target_person,
                target_location,
                has_num,
                has_year,
                table_number,
                kg_entity,
                float(np.log1p(len(toks)) / 8.0),
                float(query_direct),
                float(query_complex),
                float(query_slot_count_norm),
                float(query_content_count_log),
                float(query_coverage),
                _rank_agreement(dense_pos, sparse_pos, j, missing_rank),
                _rank_agreement(base_pos, dense_pos, j, missing_rank),
                float(dense_top5_flatness),
                _probe_score(toks, query_slots),
                _slot_coverage(toks, query_slots),
                1.0 if family and family == dense_top1_family else 0.0,
                1.0 if family and family in dense_top5_families else 0.0,
                1.0 if family and family in base_top5_families else 0.0,
                float(target_sensitive_bucket),
            ]
        )
    return np.asarray(rows, dtype=np.float32)


def rerank_with_ser(
    *,
    query_tokens: set[str],
    current_ranked_idxs: list[int],
    candidate_idxs: Sequence[int],
    candidate_base_scores: Sequence[float],
    dense_scores: np.ndarray,
    sparse_scores: np.ndarray,
    dense_ranked_idxs: list[int],
    sparse_ranked_idxs: list[int],
    doc_ids: list[str],
    doc_tokens: list[set[str]],
    doc_numeric_literals: list[set[str]],
    router_prob: Sequence[float],
    target_type: str,
    source_bucket_fn: Callable[[str], str],
    k: int,
    bundle: SERRankerBundle,
    config: SERRankerConfig | None = None,
    doc_texts: list[str] | None = None,
    query_text: str | None = None,
) -> tuple[list[int], SERRankerTrace]:
    cfg = config or bundle.config or SERRankerConfig()
    topk = max(1, int(k))
    preserve_n = min(max(0, int(cfg.preserve_top)), len(current_ranked_idxs), topk)
    preserve = [int(j) for j in current_ranked_idxs[:preserve_n]]

    base_score_map = {int(j): float(sc) for j, sc in zip(candidate_idxs, candidate_base_scores)}
    dense_pos = _rank_map(dense_ranked_idxs)
    sparse_pos = _rank_map(sparse_ranked_idxs)
    pool: list[int] = []
    seen: set[int] = set()

    def push(seq: Sequence[int], limit: int) -> None:
        for raw in list(seq)[: max(0, int(limit))]:
            j = int(raw)
            if j in seen or j < 0 or j >= len(doc_ids):
                continue
            seen.add(j)
            pool.append(j)

    push(current_ranked_idxs, max(topk * 4, int(cfg.candidate_pool_k)))
    base_sorted = sorted([int(j) for j in candidate_idxs], key=lambda j: base_score_map.get(j, 0.0), reverse=True)
    push(base_sorted, int(cfg.candidate_pool_k))
    push(dense_ranked_idxs, int(cfg.dense_pool_k))
    push(sparse_ranked_idxs, int(cfg.sparse_pool_k))

    if not pool:
        return current_ranked_idxs[:topk], SERRankerTrace()

    base_scores = [base_score_map.get(j, 0.0) for j in pool]
    feats = build_ser_feature_matrix(
        query_tokens=query_tokens,
        candidate_idxs=pool,
        candidate_base_scores=base_scores,
        dense_scores=dense_scores,
        sparse_scores=sparse_scores,
        base_ranked_idxs=current_ranked_idxs,
        dense_ranked_idxs=dense_ranked_idxs,
        sparse_ranked_idxs=sparse_ranked_idxs,
        doc_ids=doc_ids,
        doc_tokens=doc_tokens,
        doc_numeric_literals=doc_numeric_literals,
        doc_texts=doc_texts,
        router_prob=router_prob,
        target_type=target_type,
        source_bucket_fn=source_bucket_fn,
        query_text=query_text,
    )
    source_buckets = [source_bucket_fn(doc_ids[j]) for j in pool]
    learned = bundle.score_by_source(feats, source_buckets)
    base_norm = minmax(base_scores)
    blend = float(np.clip(cfg.blend_weight, 0.0, 1.0))
    final = (1.0 - blend) * base_norm + blend * learned
    plan_direct = 0.0
    plan_complex = 0.0
    query_slots: list[set[str]] = []
    if bool(getattr(cfg, "plan_adaptive", False)):
        dense_norm = minmax([float(dense_scores[j]) for j in pool])
        sparse_norm = minmax([float(sparse_scores[j]) for j in pool])
        lexical_norm = np.asarray([_overlap(query_tokens, doc_tokens[j]) for j in pool], dtype=np.float32)
        plan_direct = _estimate_direct_factoid_score(query_text, target_type)
        query_slots = _decompose_query_slots(query_text)
        plan_complex = float(np.clip((1.0 - plan_direct) * min(1.0, len(query_slots) / 3.0), 0.0, 1.0))
        final = (
            final
            + plan_direct * float(getattr(cfg, "plan_dense_weight", 0.12)) * dense_norm
            + plan_direct * float(getattr(cfg, "plan_sparse_weight", 0.03)) * sparse_norm
            + plan_direct * float(getattr(cfg, "plan_lexical_weight", 0.03)) * lexical_norm
        )
    score_map = {j: float(sc) for j, sc in zip(pool, final)}

    evidence_set_stats: dict[str, float] = {}
    if bool(getattr(cfg, "evidence_set_selection", False)):
        selected, evidence_set_stats = _select_evidence_set(
            pool=pool,
            current_ranked_idxs=current_ranked_idxs,
            base_scores=base_scores,
            learned_scores=learned,
            dense_scores=dense_scores,
            sparse_scores=sparse_scores,
            dense_ranked_idxs=dense_ranked_idxs,
            sparse_ranked_idxs=sparse_ranked_idxs,
            doc_ids=doc_ids,
            doc_tokens=doc_tokens,
            query_tokens=query_tokens,
            query_text=query_text,
            target_type=target_type,
            topk=topk,
            config=cfg,
        )
    else:
        selected = preserve[:]
        selected_set = set(selected)
        covered_slots: set[int] = set()
        if query_slots:
            for j in selected:
                _, newly = _slot_gain(doc_tokens[j], query_slots, covered_slots)
                covered_slots.update(newly)
        while len(selected) < topk:
            remaining = [j for j in pool if j not in selected_set]
            if not remaining:
                break

            def score(j: int) -> float:
                diversity_pen = 0.0
                if selected and float(cfg.diversity_weight) > 0.0:
                    diversity_pen = max(_jaccard(doc_tokens[j], doc_tokens[s]) for s in selected)
                slot_bonus = 0.0
                if query_slots and plan_complex > 0.0:
                    gain, _ = _slot_gain(doc_tokens[j], query_slots, covered_slots)
                    slot_bonus = float(getattr(cfg, "plan_slot_weight", 0.04)) * plan_complex * gain
                return float(score_map.get(j, 0.0) + slot_bonus - float(cfg.diversity_weight) * diversity_pen)

            best = max(remaining, key=score)
            selected.append(best)
            selected_set.add(best)
            if query_slots:
                _, newly = _slot_gain(doc_tokens[best], query_slots, covered_slots)
                covered_slots.update(newly)

    selected_set = set(selected)
    if len(selected) < topk:
        for j in current_ranked_idxs + dense_ranked_idxs + sparse_ranked_idxs:
            jj = int(j)
            if jj in selected_set:
                continue
            selected.append(jj)
            selected_set.add(jj)
            if len(selected) >= topk:
                break

    evidence_rescue_added = 0
    evidence_rescue_pool = 0
    rescue_k = min(max(0, int(getattr(cfg, "evidence_rescue_k", 0))), topk)
    if rescue_k > 0 and selected:
        keep_n = min(max(0, int(getattr(cfg, "evidence_preserve_top", 3))), len(selected), topk)
        replace_slots = max(0, min(rescue_k, topk - keep_n))
        rescue_pool_limit = max(replace_slots, int(getattr(cfg, "evidence_rescue_pool_k", 24)))
        dense_pool = [
            int(j)
            for j in dense_ranked_idxs[:rescue_pool_limit]
            if 0 <= int(j) < len(doc_ids) and int(j) not in selected_set
        ]
        evidence_rescue_pool = len(dense_pool)
        if replace_slots > 0 and dense_pool:
            dense_pool_base = minmax([base_score_map.get(j, 0.0) for j in dense_pool])
            base_norm_map = {j: float(sc) for j, sc in zip(dense_pool, dense_pool_base)}
            rescue_scores: list[tuple[float, int]] = []
            for j in dense_pool:
                if j not in score_map:
                    continue
                redundancy = max((_jaccard(doc_tokens[j], doc_tokens[s]) for s in selected[:keep_n]), default=0.0)
                rescue_score = (
                    0.56 * score_map.get(j, 0.0)
                    + 0.20 * _rr(dense_pos, j)
                    + 0.10 * _rr(sparse_pos, j)
                    + 0.08 * base_norm_map.get(j, 0.0)
                    + 0.06 * _overlap(query_tokens, doc_tokens[j])
                    - float(getattr(cfg, "evidence_redundancy_weight", 0.04)) * redundancy
                )
                rescue_scores.append((float(rescue_score), j))
            rescue_scores.sort(key=lambda item: item[0], reverse=True)
            replaceable = [
                (score_map.get(j, 0.0), pos, j)
                for pos, j in enumerate(selected[keep_n:], start=keep_n)
            ]
            replaceable.sort(key=lambda item: item[0])
            min_gain = float(getattr(cfg, "evidence_min_gain", 0.03))
            rescue_docs: list[int] = []
            for cand_score, cand in rescue_scores:
                if len(rescue_docs) >= min(len(replaceable), replace_slots):
                    break
                tail_score = float(replaceable[len(rescue_docs)][0])
                if cand_score <= tail_score + min_gain:
                    continue
                rescue_docs.append(cand)
            if rescue_docs:
                rebuilt: list[int] = []
                rebuilt_seen: set[int] = set()
                for j in selected[:keep_n]:
                    if j not in rebuilt_seen:
                        rebuilt.append(j)
                        rebuilt_seen.add(j)
                for j in rescue_docs:
                    if j not in rebuilt_seen:
                        rebuilt.append(j)
                        rebuilt_seen.add(j)
                        evidence_rescue_added += 1
                removed_tail = {j for _, _, j in replaceable[: len(rescue_docs)]}
                for j in selected[keep_n:] + dense_ranked_idxs + sparse_ranked_idxs + current_ranked_idxs:
                    jj = int(j)
                    if jj in removed_tail or jj in rebuilt_seen or jj < 0 or jj >= len(doc_ids):
                        continue
                    rebuilt.append(jj)
                    rebuilt_seen.add(jj)
                    if len(rebuilt) >= topk:
                        break
                selected = rebuilt[:topk]

    out = selected[:topk]
    old = [int(j) for j in current_ranked_idxs[:topk]]
    trace = SERRankerTrace(
        candidate_count=len(pool),
        preserve_count=preserve_n,
        changed_count=sum(1 for a, b in zip(out, old) if a != b),
        mean_score=float(np.mean(learned)) if learned.size else 0.0,
        max_score=float(np.max(learned)) if learned.size else 0.0,
        evidence_rescue_added=int(evidence_rescue_added),
        evidence_rescue_pool=int(evidence_rescue_pool),
        plan_direct_score=float(plan_direct),
        plan_complex_score=float(plan_complex),
        plan_slot_count=int(len(query_slots)),
        evidence_set_enabled=bool(getattr(cfg, "evidence_set_selection", False)),
        evidence_set_cardinality_need=float(evidence_set_stats.get("cardinality_need", 0.0)),
        evidence_set_slot_coverage=float(evidence_set_stats.get("slot_coverage", 0.0)),
        evidence_set_family_count=int(evidence_set_stats.get("family_count", 0.0)),
        evidence_set_anchor_hits=int(evidence_set_stats.get("anchor_hits", 0.0)),
    )
    return out, trace


def _source_need(source_budget: dict | None, bucket: str) -> float:
    if not source_budget:
        return 0.0
    keys = [f"need_{bucket}", f"top1_prob_{bucket}"]
    vals = []
    for key in keys:
        try:
            vals.append(float(source_budget.get(key, 0.0)))
        except Exception:
            vals.append(0.0)
    return max(vals or [0.0])


def _kg_sensitive_final(query_id: str, source_budget: dict | None, router_prob: Sequence[float]) -> bool:
    family = str(query_id or "").split("_", 1)[0].lower()
    if family in {"cwq", "webqsp"}:
        return True
    kg_need = _source_need(source_budget, "kg")
    try:
        kg_router = float(list(router_prob)[2])
    except Exception:
        kg_router = 0.0
    return max(kg_need, kg_router) >= 0.45


def _doc_stem_for_replacement(doc_id: str) -> str:
    raw = str(doc_id or "")
    if "_" not in raw:
        return raw
    head, tail = raw.rsplit("_", 1)
    return head if tail.isdigit() else raw


def _query_family_for_replacement(query_id: str) -> str:
    return str(query_id or "").split("_", 1)[0].lower()


def _target_bucket_for_replacement(query_id: str, source_budget: dict | None, router_prob: Sequence[float]) -> str:
    family = _query_family_for_replacement(query_id)
    if family in {"cwq", "webqsp"}:
        return "kg"
    if family in {"ott", "tat"}:
        return "table"
    if family in {"nq", "triviaqa"}:
        return "text"
    budget_top = str((source_budget or {}).get("top1_source", "")).strip().lower()
    if budget_top in {"text", "table", "kg"}:
        return budget_top
    labels = ["text", "table", "kg"]
    try:
        arr = list(router_prob)
        return labels[int(np.argmax(arr[:3]))]
    except Exception:
        return "text"


def _safe_doc_tokens(doc_tokens: Sequence[set[str]], idx: int) -> set[str]:
    try:
        if 0 <= int(idx) < len(doc_tokens):
            return set(doc_tokens[int(idx)])
    except Exception:
        pass
    return set()


def _final_replacement_feature_vector(
    *,
    query_id: str,
    query_tokens: set[str],
    candidate: int,
    old: int,
    old_slot: int,
    current_top: Sequence[int],
    pool_rank: dict[int, int],
    dense_rank: dict[int, int],
    sparse_rank: dict[int, int],
    learned_map: dict[int, float],
    score_map: dict[int, float],
    source_budget: dict | None,
    router_prob: Sequence[float],
    doc_ids: list[str],
    doc_tokens: Sequence[set[str]],
    doc_numeric_literals: Sequence[set[str]],
    source_bucket_fn: Callable[[str], str],
) -> list[float]:
    names = list(PESV_FEATURE_NAMES) if PESV_FEATURE_NAMES else []
    if not names:
        names = [
            "candidate_score",
            "old_score",
            "score_delta",
            "candidate_reference",
            "old_reference",
            "reference_delta",
            "candidate_dense",
            "old_dense",
            "dense_delta",
            "candidate_tail",
            "old_tail",
            "tail_delta",
            "candidate_sibling",
            "old_sibling",
            "sibling_delta",
            "candidate_source",
            "old_source",
            "source_delta",
            "candidate_lexical",
            "old_lexical",
            "lexical_delta",
            "candidate_family",
            "old_family",
            "family_delta",
            "candidate_redundancy",
            "old_redundancy",
            "redundancy_delta",
            "candidate_rank_rr",
            "old_rank_rr",
            "rank_rr_delta",
            "old_slot_norm",
        ]

    qfam = _query_family_for_replacement(query_id)
    target_bucket = _target_bucket_for_replacement(query_id, source_budget, router_prob)
    anchors = {
        _doc_stem_for_replacement(doc_ids[j])
        for j in list(current_top[: max(1, min(3, len(current_top)))])
        if 0 <= int(j) < len(doc_ids) and int(j) not in {int(candidate), int(old)}
    }
    if query_id:
        anchors.add(str(query_id))
    anchor_tokens: set[str] = set()
    for j in current_top[: max(1, min(3, len(current_top)))]:
        if int(j) not in {int(candidate), int(old)}:
            anchor_tokens.update(_safe_doc_tokens(doc_tokens, int(j)))
    q_numeric = {tok for tok in query_tokens if any(ch.isdigit() for ch in tok)}

    max_pool_rank = max(pool_rank.values(), default=0) + 1

    def parts(j: int) -> dict[str, float]:
        j = int(j)
        bucket = source_bucket_fn(doc_ids[j])
        stem = _doc_stem_for_replacement(doc_ids[j])
        toks = _safe_doc_tokens(doc_tokens, j)
        rank_pos = int(pool_rank.get(j, max_pool_rank))
        dense_pos = int(dense_rank.get(j, max_pool_rank))
        sparse_pos = int(sparse_rank.get(j, max_pool_rank))
        dense_rr = 1.0 / float(dense_pos + 1) if j in dense_rank else 0.0
        sparse_rr = 1.0 / float(sparse_pos + 1) if j in sparse_rank else 0.0
        tail = 0.0
        if rank_pos >= 5:
            tail = 1.0 - min(1.0, (rank_pos - 5) / max(1.0, float(max_pool_rank - 5)))
        reference = float(learned_map.get(j, 0.0))
        score = float(score_map.get(j, reference))
        lexical = _overlap(query_tokens, toks)
        family_hit = 1.0 if ((qfam in {"cwq", "webqsp"} and bucket == "kg") or stem == str(query_id) or qfam in stem) else 0.0
        redundancy = 0.0
        comparison = [int(x) for x in current_top if int(x) != j]
        if comparison:
            redundancy = max((_jaccard(toks, _safe_doc_tokens(doc_tokens, x)) for x in comparison), default=0.0)
        numeric = 0.0
        try:
            numeric = len(set(doc_numeric_literals[j]) & q_numeric) / max(1, len(q_numeric)) if q_numeric else 0.0
        except Exception:
            numeric = 0.0
        anchor_overlap = len(toks & anchor_tokens) if anchor_tokens else 0
        query_overlap = len(toks & query_tokens) if query_tokens else 0
        query_anchor_terms = query_tokens & anchor_tokens if query_tokens and anchor_tokens else set()
        novel_query_terms = (toks & query_tokens) - query_anchor_terms if query_tokens else set()
        return {
            "score": score,
            "reference": reference,
            "dense": dense_rr,
            "sparse": sparse_rr,
            "tail": tail,
            "sibling": 1.0 if stem in anchors else 0.0,
            "source": 1.0 if bucket == target_bucket else 0.0,
            "lexical": lexical,
            "family": family_hit,
            "redundancy": redundancy,
            "rank_rr": 1.0 / float(rank_pos + 1),
            "support_count": float(
                int(reference >= 0.5)
                + int(dense_rr > 0.0)
                + int(sparse_rr > 0.0)
                + int(tail > 0.0)
                + int(stem in anchors)
                + int(bucket == target_bucket)
                + int(lexical >= 0.08)
            ),
            "bucket_text": 1.0 if bucket == "text" else 0.0,
            "bucket_table": 1.0 if bucket == "table" else 0.0,
            "bucket_kg": 1.0 if bucket == "kg" else 0.0,
            "query_jaccard": _jaccard(toks, query_tokens),
            "query_coverage": float(query_overlap / max(1, len(query_tokens))) if query_tokens else 0.0,
            "query_overlap_count": float(query_overlap),
            "numeric_overlap": float(numeric),
            "anchor_jaccard": _jaccard(toks, anchor_tokens),
            "anchor_overlap_count": float(anchor_overlap),
            "anchor_novelty": float(len(novel_query_terms) / max(1, len(query_tokens))) if query_tokens else 0.0,
            "len_log": float(np.log1p(len(toks))),
        }

    cand = parts(candidate)
    oldp = parts(old)
    vals = {
        "candidate_score": cand["score"],
        "old_score": oldp["score"],
        "score_delta": cand["score"] - oldp["score"],
        "candidate_reference": cand["reference"],
        "old_reference": oldp["reference"],
        "reference_delta": cand["reference"] - oldp["reference"],
        "candidate_dense": cand["dense"],
        "old_dense": oldp["dense"],
        "dense_delta": cand["dense"] - oldp["dense"],
        "candidate_tail": cand["tail"],
        "old_tail": oldp["tail"],
        "tail_delta": cand["tail"] - oldp["tail"],
        "candidate_sibling": cand["sibling"],
        "old_sibling": oldp["sibling"],
        "sibling_delta": cand["sibling"] - oldp["sibling"],
        "candidate_source": cand["source"],
        "old_source": oldp["source"],
        "source_delta": cand["source"] - oldp["source"],
        "candidate_lexical": cand["lexical"],
        "old_lexical": oldp["lexical"],
        "lexical_delta": cand["lexical"] - oldp["lexical"],
        "candidate_family": cand["family"],
        "old_family": oldp["family"],
        "family_delta": cand["family"] - oldp["family"],
        "candidate_redundancy": cand["redundancy"],
        "old_redundancy": oldp["redundancy"],
        "redundancy_delta": cand["redundancy"] - oldp["redundancy"],
        "candidate_rank_rr": cand["rank_rr"],
        "old_rank_rr": oldp["rank_rr"],
        "rank_rr_delta": cand["rank_rr"] - oldp["rank_rr"],
        "old_slot_norm": float(old_slot) / 4.0,
        "candidate_support_count": cand["support_count"],
        "old_support_count": oldp["support_count"],
        "support_delta": cand["support_count"] - oldp["support_count"],
        "qfam_cwq": 1.0 if qfam == "cwq" else 0.0,
        "qfam_nq": 1.0 if qfam == "nq" else 0.0,
        "qfam_ott": 1.0 if qfam == "ott" else 0.0,
        "qfam_tat": 1.0 if qfam == "tat" else 0.0,
        "qfam_triviaqa": 1.0 if qfam == "triviaqa" else 0.0,
        "qfam_webqsp": 1.0 if qfam == "webqsp" else 0.0,
        "candidate_bucket_text": cand["bucket_text"],
        "candidate_bucket_table": cand["bucket_table"],
        "candidate_bucket_kg": cand["bucket_kg"],
        "old_bucket_text": oldp["bucket_text"],
        "old_bucket_table": oldp["bucket_table"],
        "old_bucket_kg": oldp["bucket_kg"],
        "candidate_query_jaccard": cand["query_jaccard"],
        "old_query_jaccard": oldp["query_jaccard"],
        "query_jaccard_delta": cand["query_jaccard"] - oldp["query_jaccard"],
        "candidate_query_coverage": cand["query_coverage"],
        "old_query_coverage": oldp["query_coverage"],
        "query_coverage_delta": cand["query_coverage"] - oldp["query_coverage"],
        "candidate_query_overlap_count": cand["query_overlap_count"],
        "old_query_overlap_count": oldp["query_overlap_count"],
        "query_overlap_count_delta": cand["query_overlap_count"] - oldp["query_overlap_count"],
        "candidate_numeric_overlap": cand["numeric_overlap"],
        "old_numeric_overlap": oldp["numeric_overlap"],
        "numeric_overlap_delta": cand["numeric_overlap"] - oldp["numeric_overlap"],
        "candidate_anchor_jaccard": cand["anchor_jaccard"],
        "old_anchor_jaccard": oldp["anchor_jaccard"],
        "anchor_jaccard_delta": cand["anchor_jaccard"] - oldp["anchor_jaccard"],
        "candidate_anchor_overlap_count": cand["anchor_overlap_count"],
        "old_anchor_overlap_count": oldp["anchor_overlap_count"],
        "anchor_overlap_count_delta": cand["anchor_overlap_count"] - oldp["anchor_overlap_count"],
        "candidate_anchor_novelty": cand["anchor_novelty"],
        "old_anchor_novelty": oldp["anchor_novelty"],
        "anchor_novelty_delta": cand["anchor_novelty"] - oldp["anchor_novelty"],
        "candidate_len_log": cand["len_log"],
        "old_len_log": oldp["len_log"],
        "len_log_delta": cand["len_log"] - oldp["len_log"],
    }
    return [float(vals.get(name, 0.0)) for name in names]


def compose_final_with_ser(
    *,
    query_id: str,
    query_text: str,
    query_tokens: set[str],
    current_ranked_idxs: list[int],
    candidate_idxs: Sequence[int],
    candidate_base_scores: Sequence[float],
    dense_scores: np.ndarray,
    sparse_scores: np.ndarray,
    dense_ranked_idxs: list[int],
    sparse_ranked_idxs: list[int],
    doc_ids: list[str],
    doc_tokens: list[set[str]],
    doc_numeric_literals: list[set[str]],
    router_prob: Sequence[float],
    target_type: str,
    source_bucket_fn: Callable[[str], str],
    bundle: SERRankerBundle,
    config: FinalEvidenceComposerConfig | None = None,
    source_budget: dict | None = None,
    doc_texts: list[str] | None = None,
    replacement_verifier_bundle: object | None = None,
) -> tuple[list[int], FinalEvidenceComposerTrace]:
    cfg = config or FinalEvidenceComposerConfig()
    topk = max(1, int(cfg.topk))
    if not current_ranked_idxs:
        return [], FinalEvidenceComposerTrace(active=False)
    current = []
    seen_current: set[int] = set()
    for raw in current_ranked_idxs:
        j = int(raw)
        if j in seen_current or j < 0 or j >= len(doc_ids):
            continue
        current.append(j)
        seen_current.add(j)
    if len(current) < topk:
        for seq in (dense_ranked_idxs, sparse_ranked_idxs, candidate_idxs):
            for raw in seq:
                j = int(raw)
                if j in seen_current or j < 0 or j >= len(doc_ids):
                    continue
                current.append(j)
                seen_current.add(j)
                if len(current) >= topk:
                    break
            if len(current) >= topk:
                break
    if len(current) <= 1:
        return current, FinalEvidenceComposerTrace(active=False)

    base_score_map = {int(j): float(sc) for j, sc in zip(candidate_idxs, candidate_base_scores)}
    dense_pos = _rank_map(dense_ranked_idxs)
    sparse_pos = _rank_map(sparse_ranked_idxs)
    pool: list[int] = []
    seen: set[int] = set()

    def push(seq: Sequence[int], limit: int) -> None:
        for raw in list(seq)[: max(0, int(limit))]:
            j = int(raw)
            if j in seen or j < 0 or j >= len(doc_ids):
                continue
            seen.add(j)
            pool.append(j)

    push(current, max(topk * 2, topk))
    base_sorted = sorted([int(j) for j in candidate_idxs], key=lambda j: base_score_map.get(j, 0.0), reverse=True)
    push(base_sorted, int(cfg.candidate_pool_k))
    push(dense_ranked_idxs, int(cfg.dense_pool_k))
    push(sparse_ranked_idxs, int(cfg.sparse_pool_k))
    if not pool:
        return current, FinalEvidenceComposerTrace(active=False)

    base_scores = [base_score_map.get(j, 0.0) for j in pool]
    feats = build_ser_feature_matrix(
        query_tokens=query_tokens,
        candidate_idxs=pool,
        candidate_base_scores=base_scores,
        dense_scores=dense_scores,
        sparse_scores=sparse_scores,
        base_ranked_idxs=current,
        dense_ranked_idxs=dense_ranked_idxs,
        sparse_ranked_idxs=sparse_ranked_idxs,
        doc_ids=doc_ids,
        doc_tokens=doc_tokens,
        doc_numeric_literals=doc_numeric_literals,
        doc_texts=doc_texts,
        router_prob=router_prob,
        target_type=target_type,
        source_bucket_fn=source_bucket_fn,
        query_text=query_text,
    )
    buckets = [source_bucket_fn(doc_ids[j]) for j in pool]
    learned = bundle.score_by_source(feats, buckets)
    dense_rr = np.asarray([_rr(dense_pos, j) for j in pool], dtype=np.float32)
    sparse_rr = np.asarray([_rr(sparse_pos, j) for j in pool], dtype=np.float32)
    lexical = np.asarray([_overlap(query_tokens, doc_tokens[j]) for j in pool], dtype=np.float32)
    source_need = np.asarray([_source_need(source_budget, source_bucket_fn(doc_ids[j])) for j in pool], dtype=np.float32)
    adjusted = (
        learned
        + float(cfg.dense_rr_weight) * dense_rr
        + float(cfg.sparse_rr_weight) * sparse_rr
        + float(cfg.lexical_weight) * lexical
        + float(cfg.source_need_weight) * (source_need >= float(cfg.source_need_threshold)).astype(np.float32)
    )
    score_map = {j: float(sc) for j, sc in zip(pool, adjusted)}
    learned_map = {j: float(sc) for j, sc in zip(pool, learned)}

    selected = list(current[:topk])
    selected_set = set(selected)
    preserve_n = min(max(0, int(cfg.preserve_top)), len(selected), topk)
    replaceable = []
    kg_sensitive = _kg_sensitive_final(query_id, source_budget, router_prob)
    for pos, j in enumerate(selected[preserve_n:], start=preserve_n):
        replaceable.append((score_map.get(j, 0.0), pos, j))
    replaceable.sort(key=lambda item: item[0])

    candidate_rows: list[tuple[float, int]] = []
    rejected = 0
    for j in pool:
        if j in selected_set:
            continue
        learned_score = learned_map.get(j, 0.0)
        if learned_score < float(cfg.min_candidate_score):
            rejected += 1
            continue
        if float(cfg.min_query_overlap) > 0.0 and _overlap(query_tokens, doc_tokens[j]) < float(cfg.min_query_overlap):
            rejected += 1
            continue
        redundancy = max((_jaccard(doc_tokens[j], doc_tokens[s]) for s in selected[:topk]), default=0.0)
        candidate_score = score_map.get(j, 0.0) - float(cfg.redundancy_weight) * redundancy
        candidate_rows.append((float(candidate_score), j))
    candidate_rows.sort(key=lambda item: item[0], reverse=True)

    replacements: list[tuple[int, int, float, float]] = []
    used_candidates: set[int] = set()
    max_replacements = min(max(0, int(cfg.max_replacements)), len(replaceable))
    pool_rank = {int(j): pos for pos, j in enumerate(pool)}
    verifier_active = bool(replacement_verifier_bundle is not None and hasattr(replacement_verifier_bundle, "score"))
    verifier_scored = 0
    verifier_accepted = 0
    verifier_rejected = 0
    verifier_max_score = 0.0
    if verifier_active:
        meta = getattr(replacement_verifier_bundle, "metadata", {}) or {}
        enabled_families = {
            str(x).strip().lower()
            for x in list(meta.get("enabled_families", []) or [])
            if str(x).strip()
        }
        qfam = _query_family_for_replacement(query_id)
        if enabled_families and qfam not in enabled_families:
            verifier_active = False
    for cand_score, cand in candidate_rows:
        if len(replacements) >= max_replacements:
            break
        if cand in used_candidates:
            continue
        if len(replacements) >= len(replaceable):
            break
        old_score, pos, old = replaceable[len(replacements)]
        if cand_score <= old_score + float(cfg.replacement_margin):
            continue
        if bool(cfg.protect_kg) and kg_sensitive and source_bucket_fn(doc_ids[old]) == "kg" and source_bucket_fn(doc_ids[cand]) != "kg":
            continue
        if verifier_active:
            feat = np.asarray(
                _final_replacement_feature_vector(
                    query_id=query_id,
                    query_tokens=query_tokens,
                    candidate=int(cand),
                    old=int(old),
                    old_slot=int(pos),
                    current_top=selected,
                    pool_rank=pool_rank,
                    dense_rank=dense_pos,
                    sparse_rank=sparse_pos,
                    learned_map=learned_map,
                    score_map=score_map,
                    source_budget=source_budget,
                    router_prob=router_prob,
                    doc_ids=doc_ids,
                    doc_tokens=doc_tokens,
                    doc_numeric_literals=doc_numeric_literals,
                    source_bucket_fn=source_bucket_fn,
                ),
                dtype=np.float32,
            ).reshape(1, -1)
            verifier_score = float(replacement_verifier_bundle.score(feat)[0])
            verifier_scored += 1
            verifier_max_score = max(verifier_max_score, verifier_score)
            if verifier_score + 1e-9 < float(cfg.replacement_verifier_threshold) + float(cfg.replacement_verifier_margin):
                verifier_rejected += 1
                continue
            verifier_accepted += 1
        replacements.append((pos, cand, float(old_score), float(cand_score)))
        used_candidates.add(cand)

    if not replacements:
        return current, FinalEvidenceComposerTrace(
            active=True,
            candidate_count=len(pool),
            preserve_count=preserve_n,
            changed_count=0,
            replacement_count=0,
            rejected_count=int(rejected),
            mean_candidate_score=float(np.mean(learned)) if learned.size else 0.0,
            max_candidate_score=float(np.max(learned)) if learned.size else 0.0,
            verifier_active=int(verifier_active),
            verifier_scored=int(verifier_scored),
            verifier_accepted=int(verifier_accepted),
            verifier_rejected=int(verifier_rejected),
            verifier_max_score=float(verifier_max_score),
        )

    removed = {selected[pos] for pos, _, _, _ in replacements}
    for pos, cand, _, _ in replacements:
        selected[pos] = cand
    rebuilt: list[int] = []
    seen_rebuilt: set[int] = set()
    for seq in (selected, current, dense_ranked_idxs, sparse_ranked_idxs, candidate_idxs):
        for raw in seq:
            j = int(raw)
            if j in removed or j in seen_rebuilt or j < 0 or j >= len(doc_ids):
                continue
            rebuilt.append(j)
            seen_rebuilt.add(j)
            if len(rebuilt) >= len(current):
                break
        if len(rebuilt) >= len(current):
            break
    old_top = current[:topk]
    new_top = rebuilt[:topk]
    return rebuilt, FinalEvidenceComposerTrace(
        active=True,
        candidate_count=len(pool),
        preserve_count=preserve_n,
        changed_count=sum(1 for a, b in zip(old_top, new_top) if a != b),
        replacement_count=len(replacements),
        rejected_count=int(rejected),
        mean_candidate_score=float(np.mean(learned)) if learned.size else 0.0,
        max_candidate_score=float(np.max(learned)) if learned.size else 0.0,
        min_replaced_score=float(min(x[2] for x in replacements)),
        max_inserted_score=float(max(x[3] for x in replacements)),
        verifier_active=int(verifier_active),
        verifier_scored=int(verifier_scored),
        verifier_accepted=int(verifier_accepted),
        verifier_rejected=int(verifier_rejected),
        verifier_max_score=float(verifier_max_score),
    )
