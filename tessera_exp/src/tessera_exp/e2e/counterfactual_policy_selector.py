from __future__ import annotations

from dataclasses import dataclass, field
import math
import pickle
from pathlib import Path
from typing import Sequence

import numpy as np

try:
    from tessera_exp.e2e.source_action_policy import (
        SOURCE_LABELS,
        build_policy_features,
        source_bucket,
    )
except Exception:  # pragma: no cover - optional during partial imports
    SOURCE_LABELS = ["text", "table", "kg"]

    def build_policy_features(query_text: str, query_id: str, ranked_doc_ids: Sequence[str], trace: dict | None = None):
        return np.zeros((0,), dtype=np.float32)

    def source_bucket(doc_id: str) -> str:
        raw = str(doc_id or "")
        if raw.startswith("m.") or raw.startswith("/m/") or raw.startswith("g."):
            return "kg"
        prefix = raw.split("_", 1)[0].lower() if "_" in raw else raw.lower()
        if prefix in {"ott", "tat"}:
            return "table"
        return "text"


POLICY_LABELS = [
    "keep_current",
    "family_source_top1",
    "family_source_top5",
    "family_source_frontload",
    "kg_relation_guard",
    "kg_entity_relation_pack",
    "kg_sibling_relation_pack",
    "text_evidence_pack",
    "text_same_stem_pack",
    "table_bridge_pack",
    "table_same_stem_bridge",
    "balanced_2source",
    "balanced_3source",
    "source_sibling_bridge",
]

FAMILY_LABELS = ["cwq", "nq", "ott", "tat", "triviaqa", "webqsp", "other"]

RANK_STAT_NAMES = [
    "cand_top1_text",
    "cand_top1_table",
    "cand_top1_kg",
    "cand_top5_text_count",
    "cand_top5_table_count",
    "cand_top5_kg_count",
    "cand_top10_text_count",
    "cand_top10_table_count",
    "cand_top10_kg_count",
    "cand_top5_source_count",
    "cand_top10_source_count",
    "cand_top5_overlap_base",
    "cand_top10_overlap_base",
    "cand_changed_top1",
    "cand_changed_top5_count",
    "delta_top5_text_count",
    "delta_top5_table_count",
    "delta_top5_kg_count",
    "delta_top10_text_count",
    "delta_top10_table_count",
    "delta_top10_kg_count",
]

POLICY_FEATURE_NAMES = [f"policy_{name}" for name in POLICY_LABELS]
FAMILY_FEATURE_NAMES = [f"family_{name}" for name in FAMILY_LABELS]
FEATURE_NAMES = (
    [f"base_{i}" for i in range(128)]
    + FAMILY_FEATURE_NAMES
    + POLICY_FEATURE_NAMES
    + RANK_STAT_NAMES
)


def query_family(query_id: str) -> str:
    family = str(query_id or "").split("_", 1)[0].lower()
    return family if family in FAMILY_LABELS else "other"


def expected_source_for_family(family: str) -> str:
    if family in {"cwq", "webqsp"}:
        return "kg"
    if family in {"ott", "tat"}:
        return "table"
    return "text"


def expected_count_for_family(family: str) -> int:
    if family in {"cwq", "webqsp"}:
        return 2
    if family in {"ott", "tat"}:
        return 3
    return 3


def _dedupe(items: Sequence[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for item in items:
        doc_id = str(item)
        if doc_id in seen:
            continue
        seen.add(doc_id)
        out.append(doc_id)
    return out


def _move_to_front(items: Sequence[str], pos: int) -> list[str]:
    out = list(items)
    if pos <= 0 or pos >= len(out):
        return out
    chosen = out[pos]
    return [chosen] + out[:pos] + out[pos + 1 :]


def doc_stem(doc_id: str) -> str:
    raw = str(doc_id or "")
    if "_" not in raw:
        return raw
    prefix, rest = raw.split("_", 1)
    if prefix in {"m.", "/m/", "g."}:
        return raw.rsplit("_", 1)[0]
    pieces = raw.rsplit("_", 1)
    if len(pieces) == 2 and pieces[1].isdigit():
        return pieces[0]
    return raw


def frontload_source(
    ranked_doc_ids: Sequence[str],
    source: str,
    *,
    pool_k: int = 30,
    max_front: int = 3,
    preserve_top1: bool = False,
) -> list[str]:
    out = _dedupe(ranked_doc_ids)
    if not out:
        return out
    pool_k = max(1, int(pool_k))
    max_front = max(1, int(max_front))
    pool = list(out[:pool_k])
    rest = list(out[pool_k:])
    protected = [pool[0]] if preserve_top1 and pool else []
    protected_set = set(protected)
    chosen = [doc_id for doc_id in pool if doc_id not in protected_set and source_bucket(doc_id) == source][:max_front]
    others = [doc_id for doc_id in pool if doc_id not in protected_set and doc_id not in set(chosen)]
    return _dedupe(protected + chosen + others + rest)


def stem_cluster_pack(
    ranked_doc_ids: Sequence[str],
    *,
    source: str | None = None,
    pool_k: int = 30,
    max_front: int = 4,
    preserve_top1: bool = True,
) -> list[str]:
    out = _dedupe(ranked_doc_ids)
    if not out:
        return out
    pool_k = max(1, int(pool_k))
    pool = list(out[:pool_k])
    rest = list(out[pool_k:])
    groups: dict[str, list[str]] = {}
    for doc_id in pool:
        if source is not None and source_bucket(doc_id) != source:
            continue
        stem = doc_stem(doc_id)
        groups.setdefault(stem, []).append(doc_id)
    if not groups:
        return out
    # Prefer clusters with multiple chunks; break ties by earliest occurrence.
    def group_key(item: tuple[str, list[str]]) -> tuple[int, int]:
        stem, docs = item
        first_pos = min(pool.index(doc_id) for doc_id in docs)
        multi_bonus = 1 if len(docs) >= 2 else 0
        return (multi_bonus * 100 + len(docs), -first_pos)

    _, cluster = max(groups.items(), key=group_key)
    cluster = cluster[: max(1, int(max_front))]
    protected = [pool[0]] if preserve_top1 and pool and pool[0] not in cluster else []
    protected_set = set(protected)
    cluster = [doc_id for doc_id in cluster if doc_id not in protected_set]
    cluster_set = set(cluster)
    others = [doc_id for doc_id in pool if doc_id not in protected_set and doc_id not in cluster_set]
    return _dedupe(protected + cluster + others + rest)


def promote_source_top1(ranked_doc_ids: Sequence[str], source: str, pool_k: int = 20) -> list[str]:
    out = _dedupe(ranked_doc_ids)
    for pos, doc_id in enumerate(out[: max(1, int(pool_k))]):
        if source_bucket(doc_id) == source:
            return _move_to_front(out, pos)
    return out


def require_source_topk(
    ranked_doc_ids: Sequence[str],
    *,
    source: str,
    min_count: int,
    topk: int = 5,
    pool_k: int = 30,
    preserve_top1: bool = True,
) -> list[str]:
    out = _dedupe(ranked_doc_ids)
    topk = max(1, int(topk))
    pool_k = max(topk, int(pool_k))
    min_count = max(0, int(min_count))
    if min_count <= 0:
        return out
    top = list(out[:topk])
    tail = list(out[topk:])
    current = [doc_id for doc_id in top if source_bucket(doc_id) == source]
    if len(current) >= min_count:
        return out
    source_pool = [
        doc_id
        for doc_id in out[topk:pool_k]
        if source_bucket(doc_id) == source and doc_id not in set(top)
    ]
    protected = {top[0]} if preserve_top1 and top else set()
    for doc_id in source_pool[: max(0, min_count - len(current))]:
        replace_pos = None
        for pos in range(topk - 1, -1, -1):
            if top[pos] in protected:
                continue
            if source_bucket(top[pos]) != source:
                replace_pos = pos
                break
        if replace_pos is None:
            break
        old = top[replace_pos]
        top[replace_pos] = doc_id
        tail = [x for x in tail if x != doc_id]
        tail.insert(0, old)
    return _dedupe(top + tail)


def require_sources_topk(
    ranked_doc_ids: Sequence[str],
    *,
    sources: Sequence[str],
    topk: int = 5,
    pool_k: int = 30,
) -> list[str]:
    out = _dedupe(ranked_doc_ids)
    for source in sources:
        if any(source_bucket(doc_id) == source for doc_id in out[:topk]):
            continue
        out = require_source_topk(out, source=source, min_count=1, topk=topk, pool_k=pool_k)
    return out


def apply_policy(
    policy: str,
    ranked_doc_ids: Sequence[str],
    *,
    query_id: str,
    topk: int = 5,
    pool_k: int = 30,
) -> list[str]:
    policy = str(policy or "keep_current")
    base = _dedupe(ranked_doc_ids)
    family = query_family(query_id)
    expected = expected_source_for_family(family)
    if policy == "keep_current":
        return base
    if policy == "family_source_top1":
        return promote_source_top1(base, expected, pool_k=pool_k)
    if policy == "family_source_top5":
        return require_source_topk(
            base,
            source=expected,
            min_count=expected_count_for_family(family),
            topk=topk,
            pool_k=pool_k,
        )
    if policy == "family_source_frontload":
        return frontload_source(base, expected, pool_k=pool_k, max_front=expected_count_for_family(family), preserve_top1=False)
    if policy == "kg_relation_guard":
        return require_source_topk(base, source="kg", min_count=2, topk=topk, pool_k=pool_k)
    if policy == "kg_entity_relation_pack":
        kg_front = frontload_source(base, "kg", pool_k=pool_k, max_front=3, preserve_top1=False)
        return require_source_topk(kg_front, source="kg", min_count=2, topk=topk, pool_k=pool_k, preserve_top1=False)
    if policy == "kg_sibling_relation_pack":
        kg_cluster = stem_cluster_pack(base, source="kg", pool_k=pool_k, max_front=3, preserve_top1=False)
        return require_source_topk(kg_cluster, source="kg", min_count=2, topk=topk, pool_k=pool_k, preserve_top1=False)
    if policy == "text_evidence_pack":
        return require_source_topk(base, source="text", min_count=3, topk=topk, pool_k=pool_k)
    if policy == "text_same_stem_pack":
        text_cluster = stem_cluster_pack(base, source="text", pool_k=pool_k, max_front=4, preserve_top1=False)
        return require_source_topk(text_cluster, source="text", min_count=3, topk=topk, pool_k=pool_k, preserve_top1=False)
    if policy == "table_bridge_pack":
        return require_source_topk(base, source="table", min_count=3, topk=topk, pool_k=pool_k)
    if policy == "table_same_stem_bridge":
        table_cluster = stem_cluster_pack(base, source="table", pool_k=pool_k, max_front=4, preserve_top1=True)
        return require_source_topk(table_cluster, source="table", min_count=3, topk=topk, pool_k=pool_k)
    if policy == "balanced_2source":
        return require_sources_topk(base, sources=[expected, "text", "table", "kg"], topk=topk, pool_k=pool_k)
    if policy == "balanced_3source":
        return require_sources_topk(base, sources=["text", "table", "kg"], topk=topk, pool_k=pool_k)
    if policy == "source_sibling_bridge":
        clustered = stem_cluster_pack(base, source=expected, pool_k=pool_k, max_front=4, preserve_top1=True)
        return require_sources_topk(clustered, sources=[expected, "text", "table", "kg"], topk=topk, pool_k=pool_k)
    return base


def candidate_rankings(
    ranked_doc_ids: Sequence[str],
    *,
    query_id: str,
    topk: int = 5,
    pool_k: int = 30,
    policy_labels: Sequence[str] = POLICY_LABELS,
) -> dict[str, list[str]]:
    return {
        str(policy): apply_policy(str(policy), ranked_doc_ids, query_id=query_id, topk=topk, pool_k=pool_k)
        for policy in policy_labels
    }


def _source_counts(doc_ids: Sequence[str], topn: int) -> dict[str, int]:
    counts = {source: 0 for source in SOURCE_LABELS}
    for doc_id in list(doc_ids)[: max(0, int(topn))]:
        bucket = source_bucket(str(doc_id))
        if bucket in counts:
            counts[bucket] += 1
    return counts


def _rank_stats(candidate: Sequence[str], base: Sequence[str]) -> list[float]:
    cand = list(candidate)
    base_list = list(base)
    top1 = source_bucket(cand[0]) if cand else ""
    cand5 = _source_counts(cand, 5)
    cand10 = _source_counts(cand, 10)
    base5 = _source_counts(base_list, 5)
    base10 = _source_counts(base_list, 10)
    top5 = set(cand[:5])
    top10 = set(cand[:10])
    base_top5 = set(base_list[:5])
    base_top10 = set(base_list[:10])
    changed_top5 = sum(1 for idx in range(min(5, len(cand), len(base_list))) if cand[idx] != base_list[idx])
    return [
        float(top1 == "text"),
        float(top1 == "table"),
        float(top1 == "kg"),
        float(cand5.get("text", 0)),
        float(cand5.get("table", 0)),
        float(cand5.get("kg", 0)),
        float(cand10.get("text", 0)),
        float(cand10.get("table", 0)),
        float(cand10.get("kg", 0)),
        float(sum(1 for source in SOURCE_LABELS if cand5.get(source, 0) > 0)),
        float(sum(1 for source in SOURCE_LABELS if cand10.get(source, 0) > 0)),
        float(len(top5 & base_top5) / max(1, len(top5 | base_top5))),
        float(len(top10 & base_top10) / max(1, len(top10 | base_top10))),
        float(bool(cand and base_list and cand[0] != base_list[0])),
        float(changed_top5),
        float(cand5.get("text", 0) - base5.get("text", 0)),
        float(cand5.get("table", 0) - base5.get("table", 0)),
        float(cand5.get("kg", 0) - base5.get("kg", 0)),
        float(cand10.get("text", 0) - base10.get("text", 0)),
        float(cand10.get("table", 0) - base10.get("table", 0)),
        float(cand10.get("kg", 0) - base10.get("kg", 0)),
    ]


def _one_hot(value: str, labels: Sequence[str]) -> list[float]:
    return [float(str(value) == str(label)) for label in labels]


def build_selector_features(
    *,
    query_text: str,
    query_id: str,
    base_ranked_doc_ids: Sequence[str],
    candidate_ranked_doc_ids: Sequence[str],
    policy: str,
    trace: dict | None = None,
    policy_labels: Sequence[str] | None = None,
) -> np.ndarray:
    base_features = np.asarray(
        build_policy_features(
            query_text=query_text,
            query_id=query_id,
            ranked_doc_ids=base_ranked_doc_ids,
            trace=trace,
        ),
        dtype=np.float32,
    ).reshape(-1)
    # Keep a stable upper bound so old/new source_budget features can coexist.
    if base_features.size >= 128:
        base_part = base_features[:128]
    else:
        base_part = np.pad(base_features, (0, 128 - base_features.size), mode="constant")
    labels = list(policy_labels) if policy_labels is not None else list(POLICY_LABELS)
    values = (
        list(base_part.astype(float))
        + _one_hot(query_family(query_id), FAMILY_LABELS)
        + _one_hot(policy, labels)
        + _rank_stats(candidate_ranked_doc_ids, base_ranked_doc_ids)
    )
    return np.asarray(values, dtype=np.float32)


def dcg(grades: Sequence[float]) -> float:
    return float(sum((2.0 ** float(grade) - 1.0) / math.log2(rank + 2.0) for rank, grade in enumerate(grades)))


def retrieval_metrics(ranked_doc_ids: Sequence[str], qrels: dict[str, float], *, topk: int = 5) -> dict[str, float]:
    rel = {str(doc_id) for doc_id, grade in qrels.items() if float(grade) > 0.0}
    top = list(ranked_doc_ids)[:topk]
    grades = [float(qrels.get(doc_id, 0.0)) for doc_id in top]
    ideal = sorted([float(v) for v in qrels.values() if float(v) > 0.0], reverse=True)[:topk]
    idcg = dcg(ideal)
    ndcg5 = dcg(grades) / idcg if idcg > 0.0 else 0.0
    ndcg1 = 1.0 if top and top[0] in rel else 0.0
    hits = 0
    ap_sum = 0.0
    for rank, doc_id in enumerate(top, start=1):
        if doc_id in rel:
            hits += 1
            ap_sum += hits / rank
    map5 = ap_sum / len(rel) if rel else 0.0
    return {
        "ndcg1": float(ndcg1),
        "ndcg5": float(ndcg5),
        "map5": float(map5),
        "hits5": float(hits),
        "anyhit5": float(hits > 0),
    }


def retrieval_utility(
    ranked_doc_ids: Sequence[str],
    qrels: dict[str, float],
    *,
    topk: int = 5,
    ndcg1_weight: float = 0.28,
    ndcg5_weight: float = 0.32,
    map5_weight: float = 0.28,
    hits5_weight: float = 0.12,
) -> float:
    metrics = retrieval_metrics(ranked_doc_ids, qrels, topk=topk)
    rel_count = max(1.0, float(sum(1 for grade in qrels.values() if float(grade) > 0.0)))
    hits_norm = min(1.0, metrics["hits5"] / min(float(topk), rel_count))
    return float(
        ndcg1_weight * metrics["ndcg1"]
        + ndcg5_weight * metrics["ndcg5"]
        + map5_weight * metrics["map5"]
        + hits5_weight * hits_norm
    )


@dataclass
class PolicySelection:
    policy: str
    predicted_utility: float
    default_utility: float
    margin: float
    policy_scores: dict[str, float] = field(default_factory=dict)
    positive_probs: dict[str, float] = field(default_factory=dict)
    switched: bool = False


@dataclass
class CounterfactualPolicySelectorBundle:
    model: object
    risk_model: object | None = None
    feature_names: list[str] = field(default_factory=lambda: list(FEATURE_NAMES))
    policy_labels: list[str] = field(default_factory=lambda: list(POLICY_LABELS))
    metadata: dict = field(default_factory=dict)
    default_policy: str = "keep_current"
    topk: int = 5
    pool_k: int = 30
    switch_margin: float = 0.0
    positive_prob_threshold: float = 0.0
    policy_switch_margins: dict[str, float] = field(default_factory=dict)
    family_switch_margins: dict[str, float] = field(default_factory=dict)
    family_policy_thresholds: dict[str, dict[str, float]] = field(default_factory=dict)
    policy_positive_thresholds: dict[str, float] = field(default_factory=dict)
    family_policy_positive_thresholds: dict[str, dict[str, float]] = field(default_factory=dict)

    def _align_features(self, features: np.ndarray) -> np.ndarray:
        arr = np.asarray(features, dtype=np.float32)
        if arr.ndim == 1:
            arr = arr.reshape(1, -1)
        expected = len(self.feature_names)
        if arr.shape[1] == expected:
            return arr
        if arr.shape[1] > expected:
            return arr[:, :expected]
        pad = np.zeros((arr.shape[0], expected - arr.shape[1]), dtype=np.float32)
        return np.hstack([arr, pad]).astype(np.float32)

    def score_candidates(
        self,
        *,
        query_text: str,
        query_id: str,
        base_ranked_doc_ids: Sequence[str],
        trace: dict | None = None,
    ) -> tuple[dict[str, list[str]], dict[str, float], dict[str, float]]:
        candidates = candidate_rankings(
            base_ranked_doc_ids,
            query_id=query_id,
            topk=int(self.topk),
            pool_k=int(self.pool_k),
            policy_labels=self.policy_labels,
        )
        features = []
        labels = []
        for policy, ranked in candidates.items():
            labels.append(policy)
            features.append(
                build_selector_features(
                    query_text=query_text,
                    query_id=query_id,
                    base_ranked_doc_ids=base_ranked_doc_ids,
                    candidate_ranked_doc_ids=ranked,
                    policy=policy,
                    trace=trace,
                    policy_labels=self.policy_labels,
                )
            )
        x = self._align_features(np.vstack(features))
        preds = np.asarray(self.model.predict(x), dtype=np.float32).reshape(-1)
        scores = {label: float(score) for label, score in zip(labels, preds.tolist())}
        score_mode = str((self.metadata or {}).get("score_mode", (self.metadata or {}).get("target_mode", "utility")))
        if score_mode == "gain":
            default_policy = self.default_policy if self.default_policy in candidates else "keep_current"
            scores[default_policy] = 0.0
        positive_probs = {label: 1.0 for label in labels}
        if getattr(self, "risk_model", None) is not None:
            if hasattr(self.risk_model, "predict_proba"):
                probs = np.asarray(self.risk_model.predict_proba(x), dtype=np.float32)
                if probs.ndim == 2 and probs.shape[1] >= 2:
                    positive = probs[:, 1]
                else:
                    positive = probs.reshape(-1)
            else:
                positive = np.asarray(self.risk_model.predict(x), dtype=np.float32).reshape(-1)
            positive_probs = {label: float(score) for label, score in zip(labels, positive.tolist())}
            default_policy = self.default_policy if self.default_policy in candidates else "keep_current"
            positive_probs[default_policy] = 1.0
        return candidates, scores, positive_probs

    def _required_margin(self, *, family: str, policy: str, fallback_margin: float) -> float:
        required = float(fallback_margin)
        policy_margins = getattr(self, "policy_switch_margins", {}) or {}
        family_margins = getattr(self, "family_switch_margins", {}) or {}
        family_policy_thresholds = getattr(self, "family_policy_thresholds", {}) or {}
        if str(policy) in policy_margins:
            required = max(required, float(policy_margins[str(policy)]))
        if str(family) in family_margins:
            required = max(required, float(family_margins[str(family)]))
        family_thresholds = family_policy_thresholds.get(str(family), {}) or {}
        if str(policy) in family_thresholds:
            required = max(required, float(family_thresholds[str(policy)]))
        return required

    def _required_positive_prob(self, *, family: str, policy: str) -> float:
        required = float(getattr(self, "positive_prob_threshold", 0.0) or 0.0)
        policy_thresholds = getattr(self, "policy_positive_thresholds", {}) or {}
        family_policy_thresholds = getattr(self, "family_policy_positive_thresholds", {}) or {}
        if str(policy) in policy_thresholds:
            required = max(required, float(policy_thresholds[str(policy)]))
        family_thresholds = family_policy_thresholds.get(str(family), {}) or {}
        if str(policy) in family_thresholds:
            required = max(required, float(family_thresholds[str(policy)]))
        return required

    def select(
        self,
        *,
        query_text: str,
        query_id: str,
        base_ranked_doc_ids: Sequence[str],
        trace: dict | None = None,
        switch_margin: float | None = None,
    ) -> tuple[list[str], PolicySelection]:
        candidates, scores, positive_probs = self.score_candidates(
            query_text=query_text,
            query_id=query_id,
            base_ranked_doc_ids=base_ranked_doc_ids,
            trace=trace,
        )
        default_policy = self.default_policy if self.default_policy in candidates else "keep_current"
        default_score = float(scores.get(default_policy, 0.0))
        best_policy = max(scores, key=lambda key: (scores[key], -self.policy_labels.index(key) if key in self.policy_labels else 0))
        best_score = float(scores.get(best_policy, default_score))
        fallback_margin = float(self.switch_margin if switch_margin is None else switch_margin)
        family = query_family(query_id)
        eligible: list[tuple[float, float, int, str]] = []
        for policy, score in scores.items():
            if policy == default_policy:
                continue
            diff = float(score) - default_score
            required_margin = self._required_margin(
                family=family,
                policy=policy,
                fallback_margin=fallback_margin,
            )
            required_prob = self._required_positive_prob(family=family, policy=policy)
            if diff >= required_margin and float(positive_probs.get(policy, 1.0)) >= required_prob:
                policy_idx = self.policy_labels.index(policy) if policy in self.policy_labels else len(self.policy_labels)
                eligible.append((float(score), diff, -policy_idx, policy))
        if eligible:
            _, _, _, chosen = max(eligible)
            switched = True
        else:
            chosen = default_policy
            switched = False
        return candidates[chosen], PolicySelection(
            policy=chosen,
            predicted_utility=float(scores.get(chosen, default_score)),
            default_utility=default_score,
            margin=float(best_score - default_score),
            policy_scores=scores,
            positive_probs=positive_probs,
            switched=bool(switched),
        )


def save_counterfactual_policy_selector_bundle(bundle: CounterfactualPolicySelectorBundle, path: Path | str) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("wb") as f:
        pickle.dump(bundle, f)


def _normalize_bundle(bundle: CounterfactualPolicySelectorBundle) -> CounterfactualPolicySelectorBundle:
    if not hasattr(bundle, "risk_model"):
        bundle.risk_model = None
    if not hasattr(bundle, "positive_prob_threshold"):
        bundle.positive_prob_threshold = 0.0
    if not hasattr(bundle, "policy_switch_margins"):
        bundle.policy_switch_margins = {}
    if not hasattr(bundle, "family_switch_margins"):
        bundle.family_switch_margins = {}
    if not hasattr(bundle, "family_policy_thresholds"):
        bundle.family_policy_thresholds = {}
    if not hasattr(bundle, "policy_positive_thresholds"):
        bundle.policy_positive_thresholds = {}
    if not hasattr(bundle, "family_policy_positive_thresholds"):
        bundle.family_policy_positive_thresholds = {}
    return bundle


def load_counterfactual_policy_selector_bundle(path: Path | str) -> CounterfactualPolicySelectorBundle:
    with Path(path).open("rb") as f:
        obj = pickle.load(f)
    if isinstance(obj, CounterfactualPolicySelectorBundle):
        return _normalize_bundle(obj)
    if isinstance(obj, dict) and "model" in obj:
        return CounterfactualPolicySelectorBundle(
            model=obj["model"],
            risk_model=obj.get("risk_model"),
            feature_names=list(obj.get("feature_names", FEATURE_NAMES)),
            policy_labels=list(obj.get("policy_labels", POLICY_LABELS)),
            metadata=dict(obj.get("metadata", obj.get("meta", {}))),
            default_policy=str(obj.get("default_policy", "keep_current")),
            topk=int(obj.get("topk", 5)),
            pool_k=int(obj.get("pool_k", 30)),
            switch_margin=float(obj.get("switch_margin", 0.0)),
            positive_prob_threshold=float(obj.get("positive_prob_threshold", 0.0)),
            policy_switch_margins={str(k): float(v) for k, v in (obj.get("policy_switch_margins", {}) or {}).items()},
            family_switch_margins={str(k): float(v) for k, v in (obj.get("family_switch_margins", {}) or {}).items()},
            family_policy_thresholds={
                str(family): {str(policy): float(value) for policy, value in (thresholds or {}).items()}
                for family, thresholds in (obj.get("family_policy_thresholds", {}) or {}).items()
            },
            policy_positive_thresholds={str(k): float(v) for k, v in (obj.get("policy_positive_thresholds", {}) or {}).items()},
            family_policy_positive_thresholds={
                str(family): {str(policy): float(value) for policy, value in (thresholds or {}).items()}
                for family, thresholds in (obj.get("family_policy_positive_thresholds", {}) or {}).items()
            },
        )
    raise TypeError(f"Unsupported counterfactual policy selector bundle type: {type(obj)!r}")
