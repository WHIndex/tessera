from __future__ import annotations

from dataclasses import dataclass, field
import math
import pickle
from pathlib import Path
from typing import Sequence

import numpy as np

try:
    from tessera_exp.e2e.source_budgeter import (
        SOURCE_BUDGET_FEATURE_NAMES,
        SOURCE_LABELS,
        build_source_budget_features,
    )
except Exception:  # pragma: no cover - optional during partial imports
    SOURCE_LABELS = ["text", "table", "kg"]
    SOURCE_BUDGET_FEATURE_NAMES = []

    def build_source_budget_features(query_text: str, query_id: str = "") -> np.ndarray:
        return np.zeros((0,), dtype=np.float32)


ACTION_LABELS = [
    "keep_current",
    "promote_text_top1",
    "promote_table_top1",
    "promote_kg_top1",
    "suppress_kg_top5",
    "require_kg_top5_1",
    "require_kg_top5_2",
    "require_text_top5_3",
    "require_table_top5_3",
]

TRACE_FEATURE_KEYS = [
    "router_text_prob_blend",
    "router_table_prob_blend",
    "router_kg_prob_blend",
    "tessera_source_budgeter_top1_prob_text",
    "tessera_source_budgeter_top1_prob_table",
    "tessera_source_budgeter_top1_prob_kg",
    "tessera_source_budgeter_need_prob_text",
    "tessera_source_budgeter_need_prob_table",
    "tessera_source_budgeter_need_prob_kg",
    "tessera_source_evidence_kg_guard_active",
    "tessera_source_evidence_kg_guard_recovered",
    "tessera_source_evidence_kg_guard_effective_min_kg",
    "tessera_source_evidence_kg_count_after",
    "tessera_source_head_changed",
    "tessera_source_head_top1_changed",
]

RANK_FEATURE_NAMES = [
    "rank_top1_text",
    "rank_top1_table",
    "rank_top1_kg",
    "rank_top5_text_count",
    "rank_top5_table_count",
    "rank_top5_kg_count",
    "rank_top10_text_count",
    "rank_top10_table_count",
    "rank_top10_kg_count",
    "rank_top5_source_count",
    "rank_top10_source_count",
    "rank_has_text_in_top10",
    "rank_has_table_in_top10",
    "rank_has_kg_in_top10",
]

FEATURE_NAMES = list(SOURCE_BUDGET_FEATURE_NAMES) + list(RANK_FEATURE_NAMES) + list(TRACE_FEATURE_KEYS)
ACTION_FEATURE_NAMES = [f"action_{label}" for label in ACTION_LABELS]
UTILITY_FEATURE_NAMES = list(FEATURE_NAMES) + list(ACTION_FEATURE_NAMES)


def source_bucket(doc_id: str) -> str:
    raw = str(doc_id or "")
    if raw.startswith("m.") or raw.startswith("/m/") or raw.startswith("g."):
        return "kg"
    prefix = raw.split("_", 1)[0].lower() if "_" in raw else raw.lower()
    if prefix in {"ott", "tat"}:
        return "table"
    return "text"


def _trace_value(trace: dict | None, key: str) -> float:
    if not trace:
        return 0.0
    raw = trace.get(key, 0.0)
    if isinstance(raw, (list, tuple)):
        raw = raw[-1] if raw else 0.0
    try:
        return float(raw)
    except Exception:
        return 0.0


def _source_counts(doc_ids: Sequence[str], topn: int) -> dict[str, int]:
    counts = {source: 0 for source in SOURCE_LABELS}
    for doc_id in list(doc_ids)[: max(0, int(topn))]:
        bucket = source_bucket(str(doc_id))
        if bucket in counts:
            counts[bucket] += 1
    return counts


def build_policy_features(
    *,
    query_text: str,
    query_id: str,
    ranked_doc_ids: Sequence[str],
    trace: dict | None = None,
) -> np.ndarray:
    query_features = np.asarray(build_source_budget_features(query_text, query_id), dtype=np.float32).reshape(-1)
    top1_bucket = source_bucket(str(ranked_doc_ids[0])) if ranked_doc_ids else ""
    top5_counts = _source_counts(ranked_doc_ids, 5)
    top10_counts = _source_counts(ranked_doc_ids, 10)
    rank_values = [
        float(top1_bucket == "text"),
        float(top1_bucket == "table"),
        float(top1_bucket == "kg"),
        float(top5_counts["text"]),
        float(top5_counts["table"]),
        float(top5_counts["kg"]),
        float(top10_counts["text"]),
        float(top10_counts["table"]),
        float(top10_counts["kg"]),
        float(sum(1 for v in top5_counts.values() if v > 0)),
        float(sum(1 for v in top10_counts.values() if v > 0)),
        float(top10_counts["text"] > 0),
        float(top10_counts["table"] > 0),
        float(top10_counts["kg"] > 0),
    ]
    trace_values = [_trace_value(trace, key) for key in TRACE_FEATURE_KEYS]
    return np.asarray(list(query_features) + rank_values + trace_values, dtype=np.float32)


def build_action_utility_features(
    base_features: np.ndarray,
    action: str,
    action_labels: Sequence[str] = ACTION_LABELS,
) -> np.ndarray:
    base = np.asarray(base_features, dtype=np.float32).reshape(-1)
    one_hot = np.zeros((len(action_labels),), dtype=np.float32)
    if action in action_labels:
        one_hot[list(action_labels).index(action)] = 1.0
    return np.concatenate([base, one_hot]).astype(np.float32)


def _query_family(query_id: str) -> str:
    return str(query_id or "").split("_", 1)[0].lower()


def kg_sensitive_query(query_id: str, ranked_doc_ids: Sequence[str], trace: dict | None = None) -> bool:
    family = _query_family(query_id)
    if family in {"cwq", "webqsp"}:
        return True
    top5_kg = sum(1 for doc_id in list(ranked_doc_ids)[:5] if source_bucket(str(doc_id)) == "kg")
    if top5_kg > 0:
        return True
    kg_signals = [
        _trace_value(trace, "router_kg_prob_blend"),
        _trace_value(trace, "tessera_source_budgeter_top1_prob_kg"),
        _trace_value(trace, "tessera_source_budgeter_need_prob_kg"),
    ]
    return max(kg_signals or [0.0]) >= 0.45


def source_action_allowed(
    action: str,
    *,
    query_id: str,
    ranked_doc_ids: Sequence[str],
    trace: dict | None = None,
    protect_kg: bool = True,
) -> bool:
    action = str(action or "keep_current")
    if action == "keep_current":
        return True
    if bool(protect_kg) and action == "suppress_kg_top5":
        return not kg_sensitive_query(query_id, ranked_doc_ids, trace)
    return True


def _move_to_front(items: list[str], pos: int) -> list[str]:
    if pos <= 0 or pos >= len(items):
        return items
    chosen = items[pos]
    return [chosen] + items[:pos] + items[pos + 1 :]


def _promote_source_top1(ranked_doc_ids: Sequence[str], target_source: str, pool_k: int) -> list[str]:
    out = list(ranked_doc_ids)
    for pos, doc_id in enumerate(out[: max(1, int(pool_k))]):
        if source_bucket(doc_id) == target_source:
            return _move_to_front(out, pos)
    return out


def _suppress_source_topk(ranked_doc_ids: Sequence[str], source: str, topk: int, pool_k: int) -> list[str]:
    pool = list(ranked_doc_ids[: max(topk, int(pool_k))])
    rest = list(ranked_doc_ids[max(topk, int(pool_k)) :])
    preferred = [doc_id for doc_id in pool if source_bucket(doc_id) != source]
    suppressed = [doc_id for doc_id in pool if source_bucket(doc_id) == source]
    return preferred + suppressed + rest


def _require_source_topk(
    ranked_doc_ids: Sequence[str],
    *,
    source: str,
    min_count: int,
    topk: int,
    pool_k: int,
    preserve_top1: bool = True,
) -> list[str]:
    out = list(ranked_doc_ids)
    topk = max(1, int(topk))
    pool_k = max(topk, int(pool_k))
    min_count = max(0, int(min_count))
    if min_count <= 0:
        return out
    current = [doc_id for doc_id in out[:topk] if source_bucket(doc_id) == source]
    if len(current) >= min_count:
        return out
    source_pool = [
        doc_id
        for doc_id in out[topk:pool_k]
        if source_bucket(doc_id) == source and doc_id not in current
    ]
    if not source_pool:
        return out
    protected = {out[0]} if preserve_top1 and out else set()
    top = list(out[:topk])
    tail = list(out[topk:])
    need = min_count - len(current)
    for doc_id in source_pool[:need]:
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
    seen: set[str] = set()
    deduped: list[str] = []
    for doc_id in top + tail:
        if doc_id in seen:
            continue
        seen.add(doc_id)
        deduped.append(doc_id)
    return deduped


def apply_source_action_to_doc_ids(
    action: str,
    ranked_doc_ids: Sequence[str],
    *,
    topk: int = 5,
    pool_k: int = 10,
) -> list[str]:
    action = str(action or "keep_current")
    if action == "promote_text_top1":
        return _promote_source_top1(ranked_doc_ids, "text", pool_k)
    if action == "promote_table_top1":
        return _promote_source_top1(ranked_doc_ids, "table", pool_k)
    if action == "promote_kg_top1":
        return _promote_source_top1(ranked_doc_ids, "kg", pool_k)
    if action == "suppress_kg_top5":
        return _suppress_source_topk(ranked_doc_ids, "kg", topk, pool_k)
    if action == "require_kg_top5_1":
        return _require_source_topk(ranked_doc_ids, source="kg", min_count=1, topk=topk, pool_k=pool_k)
    if action == "require_kg_top5_2":
        return _require_source_topk(ranked_doc_ids, source="kg", min_count=2, topk=topk, pool_k=pool_k)
    if action == "require_text_top5_3":
        return _require_source_topk(ranked_doc_ids, source="text", min_count=3, topk=topk, pool_k=pool_k)
    if action == "require_table_top5_3":
        return _require_source_topk(ranked_doc_ids, source="table", min_count=3, topk=topk, pool_k=pool_k)
    return list(ranked_doc_ids)


def apply_source_action_to_ranked_idxs(
    action: str,
    ranked_idxs: Sequence[int],
    doc_ids: Sequence[str],
    *,
    topk: int = 5,
    pool_k: int = 10,
) -> list[int]:
    idxs = [int(j) for j in ranked_idxs if 0 <= int(j) < len(doc_ids)]
    ranked_doc_ids = [str(doc_ids[j]) for j in idxs]
    reranked_doc_ids = apply_source_action_to_doc_ids(action, ranked_doc_ids, topk=topk, pool_k=pool_k)
    pos_by_id: dict[str, list[int]] = {}
    for j, doc_id in zip(idxs, ranked_doc_ids):
        pos_by_id.setdefault(doc_id, []).append(j)
    out: list[int] = []
    for doc_id in reranked_doc_ids:
        bucket = pos_by_id.get(doc_id, [])
        if bucket:
            out.append(bucket.pop(0))
    seen = set(out)
    out.extend(j for j in idxs if j not in seen)
    return out[: len(idxs)]


def dcg(grades: Sequence[float]) -> float:
    return float(sum((2.0 ** float(grade) - 1.0) / math.log2(rank + 2.0) for rank, grade in enumerate(grades)))


def retrieval_score(
    ranked_doc_ids: Sequence[str],
    qrels: dict[str, float],
    *,
    topk: int = 5,
) -> float:
    rel = {doc_id for doc_id, grade in qrels.items() if float(grade) > 0.0}
    grades = [float(qrels.get(doc_id, 0.0)) for doc_id in list(ranked_doc_ids)[:topk]]
    ideal = sorted([float(v) for v in qrels.values() if float(v) > 0.0], reverse=True)[:topk]
    idcg = dcg(ideal)
    ndcg5 = dcg(grades) / idcg if idcg > 0.0 else 0.0
    ndcg1 = 1.0 if ranked_doc_ids and ranked_doc_ids[0] in rel else 0.0
    hits = 0
    ap_sum = 0.0
    for rank, doc_id in enumerate(list(ranked_doc_ids)[:topk], start=1):
        if doc_id in rel:
            hits += 1
            ap_sum += hits / rank
    map5 = ap_sum / len(rel) if rel else 0.0
    return float(0.55 * ndcg5 + 0.35 * map5 + 0.10 * ndcg1)


@dataclass
class SourceActionPrediction:
    action: str
    confidence: float
    action_probs: dict[str, float] = field(default_factory=dict)


@dataclass
class SourceActionPolicyBundle:
    model: object
    feature_names: list[str] = field(default_factory=lambda: list(FEATURE_NAMES))
    action_labels: list[str] = field(default_factory=lambda: list(ACTION_LABELS))
    metadata: dict = field(default_factory=dict)

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

    def predict(
        self,
        *,
        query_text: str,
        query_id: str,
        ranked_doc_ids: Sequence[str],
        trace: dict | None = None,
    ) -> SourceActionPrediction:
        x = self._align_features(
            build_policy_features(query_text=query_text, query_id=query_id, ranked_doc_ids=ranked_doc_ids, trace=trace)
        )
        probs: dict[str, float] = {}
        if hasattr(self.model, "predict_proba"):
            raw = np.asarray(self.model.predict_proba(x), dtype=np.float32)
            classes = [str(c) for c in getattr(self.model, "classes_", [])]
            if raw.ndim == 2 and raw.shape[0] == 1:
                for cls, prob in zip(classes, raw[0].tolist()):
                    probs[cls] = float(prob)
        if probs:
            action = max(probs, key=probs.get)
            confidence = float(probs.get(action, 0.0))
        else:
            action = str(self.model.predict(x)[0])
            confidence = 1.0
            probs = {label: (1.0 if label == action else 0.0) for label in self.action_labels}
        return SourceActionPrediction(action=action, confidence=confidence, action_probs=probs)


@dataclass
class SourceUtilityGateBundle:
    model: object
    feature_names: list[str] = field(default_factory=lambda: list(FEATURE_NAMES))
    utility_feature_names: list[str] = field(default_factory=lambda: list(UTILITY_FEATURE_NAMES))
    action_labels: list[str] = field(default_factory=lambda: list(ACTION_LABELS))
    action_thresholds: dict[str, float] = field(default_factory=dict)
    action_stats: dict[str, dict] = field(default_factory=dict)
    metadata: dict = field(default_factory=dict)
    min_gain: float = 0.01
    protect_kg: bool = True

    def _align_base_features(self, features: np.ndarray) -> np.ndarray:
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

    def _align_utility_features(self, features: np.ndarray) -> np.ndarray:
        arr = np.asarray(features, dtype=np.float32)
        if arr.ndim == 1:
            arr = arr.reshape(1, -1)
        expected = len(self.utility_feature_names)
        if arr.shape[1] == expected:
            return arr
        if arr.shape[1] > expected:
            return arr[:, :expected]
        pad = np.zeros((arr.shape[0], expected - arr.shape[1]), dtype=np.float32)
        return np.hstack([arr, pad]).astype(np.float32)

    def predict(
        self,
        *,
        query_text: str,
        query_id: str,
        ranked_doc_ids: Sequence[str],
        trace: dict | None = None,
    ) -> SourceActionPrediction:
        base = self._align_base_features(
            build_policy_features(query_text=query_text, query_id=query_id, ranked_doc_ids=ranked_doc_ids, trace=trace)
        )[0]
        utilities: dict[str, float] = {"keep_current": 0.0}
        best_action = "keep_current"
        best_gain = 0.0
        for action in self.action_labels:
            if action == "keep_current":
                continue
            if not source_action_allowed(
                action,
                query_id=query_id,
                ranked_doc_ids=ranked_doc_ids,
                trace=trace,
                protect_kg=bool(self.protect_kg),
            ):
                utilities[action] = float("-inf")
                continue
            threshold = float(self.action_thresholds.get(action, float("inf")))
            if not np.isfinite(threshold):
                utilities[action] = float("-inf")
                continue
            x = self._align_utility_features(
                build_action_utility_features(base, action, action_labels=self.action_labels)
            )
            predicted_gain = float(np.asarray(self.model.predict(x)).reshape(-1)[0])
            utilities[action] = predicted_gain
            if predicted_gain >= max(float(self.min_gain), threshold) and predicted_gain > best_gain:
                best_action = action
                best_gain = predicted_gain
        return SourceActionPrediction(action=best_action, confidence=max(0.0, float(best_gain)), action_probs=utilities)


def save_source_action_policy_bundle(bundle: SourceActionPolicyBundle, path: Path | str) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("wb") as f:
        pickle.dump(bundle, f)


def load_source_action_policy_bundle(path: Path | str) -> SourceActionPolicyBundle:
    with Path(path).open("rb") as f:
        obj = pickle.load(f)
    if isinstance(obj, SourceActionPolicyBundle):
        return obj
    if isinstance(obj, SourceUtilityGateBundle):
        return obj
    if isinstance(obj, dict) and "model" in obj:
        if obj.get("bundle_type") == "source_utility_gate":
            return SourceUtilityGateBundle(
                model=obj["model"],
                feature_names=list(obj.get("feature_names", FEATURE_NAMES)),
                utility_feature_names=list(obj.get("utility_feature_names", UTILITY_FEATURE_NAMES)),
                action_labels=list(obj.get("action_labels", ACTION_LABELS)),
                action_thresholds=dict(obj.get("action_thresholds", {})),
                action_stats=dict(obj.get("action_stats", {})),
                metadata=dict(obj.get("metadata", obj.get("meta", {}))),
                min_gain=float(obj.get("min_gain", 0.01)),
                protect_kg=bool(obj.get("protect_kg", True)),
            )
        return SourceActionPolicyBundle(
            model=obj["model"],
            feature_names=list(obj.get("feature_names", FEATURE_NAMES)),
            action_labels=list(obj.get("action_labels", ACTION_LABELS)),
            metadata=dict(obj.get("metadata", obj.get("meta", {}))),
        )
    raise TypeError(f"Unsupported source action policy bundle type: {type(obj)!r}")
