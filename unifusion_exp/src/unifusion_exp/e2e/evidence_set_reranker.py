from __future__ import annotations

from dataclasses import dataclass, field, replace
import math
import pickle
import re
from pathlib import Path
from typing import Any, Sequence

import numpy as np

try:
    from unifusion_exp.e2e.counterfactual_policy_selector import (
        FAMILY_LABELS,
        doc_stem,
        expected_source_for_family,
        query_family,
        source_bucket,
    )
except Exception:  # pragma: no cover - keeps this module importable in partial installs
    FAMILY_LABELS = ["cwq", "nq", "ott", "tat", "triviaqa", "webqsp", "other"]

    def source_bucket(doc_id: str) -> str:
        raw = str(doc_id or "")
        if raw.startswith("m.") or raw.startswith("/m/") or raw.startswith("g."):
            return "kg"
        prefix = raw.split("_", 1)[0].lower() if "_" in raw else raw.lower()
        if prefix in {"ott", "tat"}:
            return "table"
        return "text"

    def query_family(query_id: str) -> str:
        family = str(query_id or "").split("_", 1)[0].lower()
        return family if family in FAMILY_LABELS else "other"

    def expected_source_for_family(family: str) -> str:
        if family in {"cwq", "webqsp"}:
            return "kg"
        if family in {"ott", "tat"}:
            return "table"
        return "text"

    def doc_stem(doc_id: str) -> str:
        raw = str(doc_id or "")
        pieces = raw.rsplit("_", 1)
        return pieces[0] if len(pieces) == 2 and pieces[1].isdigit() else raw


SOURCE_LABELS = ["text", "table", "kg"]
TOKEN_RE = re.compile(r"[a-z0-9]+")
NUMBER_RE = re.compile(r"(?<!\w)[+-]?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?%?(?!\w)")
YEAR_RE = re.compile(r"\b(?:1[6-9]\d{2}|20\d{2}|21\d{2})\b")
STOPWORDS = {
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
    "be",
    "been",
    "being",
    "do",
    "does",
    "did",
    "what",
    "who",
    "where",
    "when",
    "which",
    "how",
    "many",
    "much",
    "whose",
    "that",
    "this",
    "these",
    "those",
}


ESR_FEATURE_NAMES = [
    "rank_rr",
    "rank_norm",
    "is_top1",
    "is_top3",
    "is_top5",
    "is_top10",
    "source_text",
    "source_table",
    "source_kg",
    "source_expected",
    "source_count_top5_norm",
    "source_count_top10_norm",
    "family_cwq",
    "family_nq",
    "family_ott",
    "family_tat",
    "family_triviaqa",
    "family_webqsp",
    "family_other",
    "family_expected_source_count_top5_norm",
    "same_stem_as_top1",
    "same_source_as_top1",
    "query_token_count_log",
    "doc_token_count_log",
    "token_overlap_log",
    "token_jaccard",
    "query_token_coverage",
    "doc_token_precision",
    "doc_id_query_overlap",
    "query_number_count_log",
    "doc_number_count_log",
    "number_overlap",
    "query_year_count_log",
    "doc_year_count_log",
    "year_overlap",
    "router_bucket_prob",
    "router_expected_prob",
    "query_prior_bucket",
    "query_prior_expected",
    "query_prior_disagreement",
    "source_budget_top1_prob",
    "source_budget_need_prob",
    "source_budget_avg_prob",
    "evidence_source_count_trace",
    "uni_candidate_size_log",
    "heavy_table_weight_eff",
    "heavy_kg_path_weight_eff",
    "heavy_token_late_weight_eff",
    "heavy_token_cross_weight_eff",
    "qa_objective_weight_eff",
    "upo_lite_retrieval_weight_eff",
    "ser_score_trace",
    "ceps_positive_prob_trace",
    "kgcv_score_trace",
    "erv_score_trace",
]


@dataclass(frozen=True)
class EvidenceSetRerankerConfig:
    pool_k: int = 10
    topk: int = 5
    preserve_top: int = 0
    blend_original_weight: float = 0.10
    top1_switch_margin: float = 0.02
    min_model_score: float = -1.0
    anchor_guard_enabled: bool = False
    anchor_guard_topk: int = 5
    anchor_guard_max_restores: int = 1
    anchor_guard_min_model_score: float = -1.0


@dataclass
class EvidenceSetRerankerResult:
    ranked_doc_ids: list[str]
    changed_count: int
    switched_top1: bool
    pool_size: int
    model_scores: dict[str, float] = field(default_factory=dict)
    final_scores: dict[str, float] = field(default_factory=dict)
    anchor_guard_anchors: list[str] = field(default_factory=list)
    anchor_guard_restored: int = 0


@dataclass
class EvidenceSetRerankerBundle:
    model: object
    feature_names: list[str] = field(default_factory=lambda: list(ESR_FEATURE_NAMES))
    config: EvidenceSetRerankerConfig = field(default_factory=EvidenceSetRerankerConfig)
    metadata: dict[str, Any] = field(default_factory=dict)

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

    def score(self, features: np.ndarray) -> np.ndarray:
        if features.size == 0:
            return np.zeros((0,), dtype=np.float32)
        features = self._align_features(features)
        if hasattr(self.model, "predict_proba"):
            probs = np.asarray(self.model.predict_proba(features), dtype=np.float32)
            if probs.ndim == 2 and probs.shape[1] >= 2:
                return probs[:, -1].astype(np.float32)
            return probs.reshape(-1).astype(np.float32)
        raw = np.asarray(self.model.predict(features), dtype=np.float32).reshape(-1)
        return raw.astype(np.float32)

    def with_config(self, **kwargs: Any) -> "EvidenceSetRerankerBundle":
        config = _coerce_config(self.config)
        return EvidenceSetRerankerBundle(
            model=self.model,
            feature_names=list(self.feature_names),
            config=replace(config, **kwargs),
            metadata=dict(self.metadata),
        )

    def rerank(
        self,
        *,
        query_text: str,
        query_id: str,
        ranked_doc_ids: Sequence[str],
        trace: dict[str, Any] | None = None,
        corpus_texts: dict[str, str] | None = None,
        pool_k: int | None = None,
        top1_switch_margin: float | None = None,
        blend_original_weight: float | None = None,
        preserve_top: int | None = None,
    ) -> EvidenceSetRerankerResult:
        base = _dedupe([str(x) for x in ranked_doc_ids])
        if not base:
            return EvidenceSetRerankerResult([], 0, False, 0)
        config = _coerce_config(self.config)
        pool_k_val = max(1, int(config.pool_k if pool_k is None else pool_k))
        preserve = max(0, int(config.preserve_top if preserve_top is None else preserve_top))
        margin = float(config.top1_switch_margin if top1_switch_margin is None else top1_switch_margin)
        blend = float(config.blend_original_weight if blend_original_weight is None else blend_original_weight)
        pool = base[: min(pool_k_val, len(base))]
        protected = pool[: min(preserve, len(pool))]
        candidates = pool[min(preserve, len(pool)) :]
        if not candidates:
            return EvidenceSetRerankerResult(base, 0, False, len(pool))

        feature_rows = [
            build_evidence_features(
                query_text=query_text,
                query_id=query_id,
                doc_id=doc_id,
                doc_text=(corpus_texts or {}).get(doc_id, ""),
                ranked_doc_ids=base,
                rank_position=base.index(doc_id),
                trace=trace or {},
            )
            for doc_id in candidates
        ]
        x = np.vstack(feature_rows).astype(np.float32)
        model_scores = self.score(x)
        final_scores = []
        for doc_id, model_score in zip(candidates, model_scores):
            rr = 1.0 / float(base.index(doc_id) + 1)
            final_scores.append(float(model_score) + blend * rr)

        score_by_doc = {doc_id: float(score) for doc_id, score in zip(candidates, model_scores)}
        final_by_doc = {doc_id: float(score) for doc_id, score in zip(candidates, final_scores)}
        ordered = [
            doc_id
            for doc_id, _ in sorted(
                zip(candidates, final_scores),
                key=lambda item: (item[1], -base.index(item[0])),
                reverse=True,
            )
        ]
        if float(config.min_model_score) > -1.0:
            high = [doc_id for doc_id in ordered if score_by_doc.get(doc_id, 0.0) >= float(config.min_model_score)]
            low = [doc_id for doc_id in ordered if doc_id not in set(high)]
            ordered = high + low

        if preserve <= 0 and ordered and ordered[0] != base[0] and base[0] in final_by_doc:
            proposed = ordered[0]
            advantage = float(final_by_doc.get(proposed, -1e9) - final_by_doc.get(base[0], -1e9))
            if advantage < margin:
                ordered = [base[0]] + [doc_id for doc_id in ordered if doc_id != base[0]]

        new_pool = _dedupe(protected + ordered)
        ranked = _dedupe(new_pool + base[len(pool) :])
        anchors: list[str] = []
        restored = 0
        if bool(config.anchor_guard_enabled):
            ranked, anchors, restored = apply_anchor_guard(
                query_id=query_id,
                base_ranked_doc_ids=base,
                proposed_ranked_doc_ids=ranked,
                model_scores=score_by_doc,
                final_scores=final_by_doc,
                topk=int(config.topk),
                preserve_top=preserve,
                guard_topk=int(config.anchor_guard_topk),
                max_restores=int(config.anchor_guard_max_restores),
                min_model_score=float(config.anchor_guard_min_model_score),
            )
        changed = sum(1 for old, new in zip(base[: config.topk], ranked[: config.topk]) if old != new)
        switched_top1 = bool(base and ranked and base[0] != ranked[0])
        return EvidenceSetRerankerResult(
            ranked_doc_ids=ranked,
            changed_count=int(changed),
            switched_top1=switched_top1,
            pool_size=len(pool),
            model_scores=score_by_doc,
            final_scores=final_by_doc,
            anchor_guard_anchors=anchors,
            anchor_guard_restored=int(restored),
        )


def save_evidence_set_reranker_bundle(bundle: EvidenceSetRerankerBundle, path: Path | str) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("wb") as f:
        pickle.dump(bundle, f)


def load_evidence_set_reranker_bundle(path: Path | str) -> EvidenceSetRerankerBundle:
    with Path(path).open("rb") as f:
        obj = pickle.load(f)
    if isinstance(obj, EvidenceSetRerankerBundle):
        obj.config = _coerce_config(obj.config)
        return obj
    if isinstance(obj, dict) and "model" in obj:
        return EvidenceSetRerankerBundle(
            model=obj["model"],
            feature_names=list(obj.get("feature_names", ESR_FEATURE_NAMES)),
            config=_coerce_config(obj.get("config", EvidenceSetRerankerConfig())),
            metadata=dict(obj.get("metadata", {})),
        )
    raise TypeError(f"Unsupported evidence-set reranker bundle type: {type(obj)!r}")


def _coerce_config(config: object | None) -> EvidenceSetRerankerConfig:
    defaults = EvidenceSetRerankerConfig()
    if config is None:
        return defaults
    values: dict[str, Any] = {}
    for name in defaults.__dataclass_fields__:
        values[name] = getattr(config, name, getattr(defaults, name))
    return EvidenceSetRerankerConfig(**values)


def apply_anchor_guard(
    *,
    query_id: str,
    base_ranked_doc_ids: Sequence[str],
    proposed_ranked_doc_ids: Sequence[str],
    model_scores: dict[str, float],
    final_scores: dict[str, float],
    topk: int,
    preserve_top: int,
    guard_topk: int,
    max_restores: int,
    min_model_score: float,
) -> tuple[list[str], list[str], int]:
    base = _dedupe([str(x) for x in base_ranked_doc_ids])
    proposed = _dedupe([str(x) for x in proposed_ranked_doc_ids])
    topk = max(1, int(topk))
    preserve_top = max(0, int(preserve_top))
    guard_topk = max(preserve_top + 1, int(guard_topk))
    max_restores = max(0, int(max_restores))
    if not base or not proposed or max_restores <= 0:
        return proposed, [], 0

    family = query_family(str(query_id))
    expected = expected_source_for_family(family)
    head = base[0]
    head_source = source_bucket(head)
    head_stem = doc_stem(head)
    anchors: list[str] = []
    for pos, doc_id in enumerate(base[preserve_top : min(len(base), guard_topk)], start=preserve_top):
        if doc_id not in proposed:
            continue
        score = float(model_scores.get(doc_id, final_scores.get(doc_id, 0.0)))
        if float(min_model_score) > -1.0 and score < float(min_model_score):
            continue
        bucket = source_bucket(doc_id)
        stem = doc_stem(doc_id)
        protect = False
        if family in {"ott", "tat"}:
            protect = bucket == "table" and pos < min(guard_topk, 3)
        elif family in {"cwq", "webqsp"}:
            protect = bucket == "kg" and pos < min(guard_topk, 3)
        elif family in {"nq", "triviaqa"}:
            protect = stem == head_stem
        else:
            protect = bucket == expected or stem == head_stem
        if stem == head_stem and pos < guard_topk:
            protect = True
        if family in {"ott", "tat", "cwq", "webqsp"} and bucket == expected and pos < min(guard_topk, 3):
            protect = True
        if protect:
            anchors.append(doc_id)

    anchors = _dedupe(anchors)
    if not anchors:
        return proposed, [], 0

    top = proposed[:topk]
    missing = [doc_id for doc_id in anchors if doc_id not in top]
    if not missing:
        return proposed, anchors, 0

    protected = set(base[:preserve_top])
    anchor_set = set(anchors)
    restored = 0
    out = list(proposed)
    base_top = set(base[:topk])
    for anchor in missing:
        if restored >= max_restores or anchor not in out:
            break
        replace_idx = None
        for idx in range(min(topk, len(out)) - 1, preserve_top - 1, -1):
            candidate = out[idx]
            if candidate in protected or candidate in anchor_set:
                continue
            if candidate not in base_top:
                replace_idx = idx
                break
            if replace_idx is None:
                replace_idx = idx
        if replace_idx is None:
            continue
        displaced = out[replace_idx]
        without = [doc_id for doc_id in out if doc_id not in {anchor, displaced}]
        prefix = without[:replace_idx]
        suffix = without[replace_idx:]
        out = _dedupe(prefix + [anchor] + suffix + [displaced])
        restored += 1
    return out, anchors, int(restored)


def build_evidence_features(
    *,
    query_text: str,
    query_id: str,
    doc_id: str,
    doc_text: str,
    ranked_doc_ids: Sequence[str],
    rank_position: int,
    trace: dict[str, Any] | None = None,
) -> np.ndarray:
    trace = trace or {}
    ranked = [str(x) for x in ranked_doc_ids]
    family = query_family(str(query_id))
    expected = expected_source_for_family(family)
    bucket = source_bucket(str(doc_id))
    top1 = ranked[0] if ranked else ""
    q_tokens = content_tokens(query_text)
    d_text = doc_text or doc_id.replace("_", " ")
    d_tokens = content_tokens(d_text)
    id_tokens = content_tokens(doc_id.replace("_", " "))
    q_nums = numbers(query_text)
    d_nums = numbers(d_text)
    q_years = years(query_text)
    d_years = years(d_text)
    overlap = q_tokens & d_tokens
    id_overlap = q_tokens & id_tokens
    source_counts_top5 = _source_counts(ranked[:5])
    source_counts_top10 = _source_counts(ranked[:10])
    expected_top5 = sum(1 for item in ranked[:5] if source_bucket(item) == expected)
    source_count5 = source_counts_top5.get(bucket, 0)
    source_count10 = source_counts_top10.get(bucket, 0)

    row: list[float] = [
        1.0 / float(rank_position + 1),
        float(rank_position) / max(1.0, float(len(ranked) - 1)),
        float(rank_position == 0),
        float(rank_position < 3),
        float(rank_position < 5),
        float(rank_position < 10),
        float(bucket == "text"),
        float(bucket == "table"),
        float(bucket == "kg"),
        float(bucket == expected),
        float(source_count5) / 5.0,
        float(source_count10) / 10.0,
    ]
    row.extend(float(family == label) for label in FAMILY_LABELS)
    row.extend(
        [
            float(expected_top5) / 5.0,
            float(doc_stem(doc_id) == doc_stem(top1)) if top1 else 0.0,
            float(bucket == source_bucket(top1)) if top1 else 0.0,
            _log_count(len(q_tokens)),
            _log_count(len(d_tokens)),
            _log_count(len(overlap)),
            _jaccard(q_tokens, d_tokens),
            float(len(overlap)) / max(1.0, float(len(q_tokens))),
            float(len(overlap)) / max(1.0, float(len(d_tokens))),
            float(len(id_overlap)) / max(1.0, float(len(q_tokens))),
            _log_count(len(q_nums)),
            _log_count(len(d_nums)),
            _overlap_ratio(q_nums, d_nums),
            _log_count(len(q_years)),
            _log_count(len(d_years)),
            _overlap_ratio(q_years, d_years),
            trace_source_value(trace, "router", bucket),
            trace_source_value(trace, "router", expected),
            trace_source_value(trace, "query_prior", bucket),
            trace_source_value(trace, "query_prior", expected),
            trace_float(trace, "query_prior_disagreement"),
            trace_source_value(trace, "unifusion_source_budgeter_top1_prob", bucket),
            trace_source_value(trace, "unifusion_source_budgeter_need_prob", bucket),
            _avg(
                [
                    trace_source_value(trace, "unifusion_source_budgeter_top1_prob", bucket),
                    trace_source_value(trace, "unifusion_source_budgeter_need_prob", bucket),
                    trace_source_value(trace, "unifusion_source_budgeter_add_prob", bucket),
                ]
            ),
            trace_source_value(trace, "unifusion_source_evidence", bucket),
            _log_count(trace_float(trace, "uni_candidate_size")),
            trace_float(trace, "heavy_table_weight_eff"),
            trace_float(trace, "heavy_kg_path_weight_eff"),
            trace_float(trace, "heavy_token_late_weight_eff"),
            trace_float(trace, "heavy_token_cross_weight_eff"),
            trace_float(trace, "qa_objective_weight_eff"),
            trace_float(trace, "upo_lite_retrieval_weight_eff"),
            _max_prefixed(trace, ("unifusion_ser_", "ser_")),
            _max_prefixed(trace, ("unifusion_ceps_positive", "unifusion_ceps_predicted")),
            _max_prefixed(trace, ("unifusion_kgcv_", "kgcv_")),
            _max_prefixed(trace, ("unifusion_erv_", "erv_")),
        ]
    )
    return np.asarray(row, dtype=np.float32)


def content_tokens(text: str | None) -> set[str]:
    return {
        tok
        for tok in TOKEN_RE.findall(str(text or "").lower())
        if len(tok) > 1 and tok not in STOPWORDS
    }


def numbers(text: str | None) -> set[str]:
    return {x.replace(",", "") for x in NUMBER_RE.findall(str(text or "").lower())}


def years(text: str | None) -> set[str]:
    return set(YEAR_RE.findall(str(text or "")))


def trace_float(trace: dict[str, Any], key: str, default: float = 0.0) -> float:
    try:
        val = trace.get(key, default)
        if val is None:
            return float(default)
        return float(val)
    except Exception:
        return float(default)


def trace_source_value(trace: dict[str, Any], prefix: str, source: str) -> float:
    keys = [
        f"{prefix}_{source}",
        f"{prefix}_{source}_prob",
        f"{prefix}_{source}_prob_blend",
        f"{prefix}_prob_{source}",
    ]
    for key in keys:
        if key in trace:
            return trace_float(trace, key)
    return 0.0


def _max_prefixed(trace: dict[str, Any], prefixes: Sequence[str]) -> float:
    best = 0.0
    for key, raw in trace.items():
        if not any(str(key).startswith(prefix) for prefix in prefixes):
            continue
        try:
            best = max(best, float(raw))
        except Exception:
            continue
    return float(best)


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


def _source_counts(items: Sequence[str]) -> dict[str, int]:
    counts = {label: 0 for label in SOURCE_LABELS}
    for item in items:
        bucket = source_bucket(str(item))
        counts[bucket] = counts.get(bucket, 0) + 1
    return counts


def _log_count(value: float | int) -> float:
    return float(math.log1p(max(0.0, float(value))))


def _jaccard(a: set[str], b: set[str]) -> float:
    if not a or not b:
        return 0.0
    return float(len(a & b) / max(1, len(a | b)))


def _overlap_ratio(a: set[str], b: set[str]) -> float:
    if not a or not b:
        return 0.0
    return float(len(a & b) / max(1, len(a)))


def _avg(values: Sequence[float]) -> float:
    vals = [float(v) for v in values if not math.isnan(float(v))]
    return float(sum(vals) / max(1, len(vals)))
