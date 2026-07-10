from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Callable, Sequence

import numpy as np

try:
    from unifusion_exp.e2e.pairwise_slot_verifier import PESV_FEATURE_NAMES
except Exception:  # pragma: no cover - optional at runtime
    PESV_FEATURE_NAMES = []


TOKEN_RE = re.compile(r"[a-z0-9]+")
STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "by", "did", "do", "does",
    "for", "from", "how", "in", "is", "it", "many", "much", "of", "on",
    "or", "the", "that", "this", "to", "was", "were", "what", "when",
    "where", "which", "who", "whom", "whose", "will", "with",
}


@dataclass(frozen=True)
class SourceEvidenceFusionConfig:
    topk: int = 5
    candidate_pool_k: int = 80
    preserve_top: int = 1
    base_weight: float = 0.34
    dense_weight: float = 0.16
    sparse_weight: float = 0.08
    reference_weight: float = 0.14
    lexical_weight: float = 0.10
    modality_prior_weight: float = 0.12
    source_balance_weight: float = 0.10
    target_family_weight: float = 0.08
    diversity_weight: float = 0.025
    replacement_margin: float = 0.01
    min_candidate_score: float = 0.08
    dense_guard: bool = False
    dense_guard_topn: int = 5
    dense_guard_prefixes: str = ""
    dense_guard_weight: float = 0.22
    dense_rank_weight: float = 0.10
    current_rank_weight: float = 0.06
    source_balance_prefixes: str = ""
    max_changed_slots: int = 0
    slot_acceptance_guard: bool = False
    slot_acceptance_prefixes: str = ""
    slot_acceptance_margin: float = 0.02
    budget_composer: bool = False
    budget_prefixes: str = ""
    budget_candidate_pool_k: int = 180
    budget_start_slot: int = 4
    budget_max_selected: int = 2
    budget_score_weight: float = 0.10
    budget_sibling_weight: float = 0.16
    budget_source_quota_weight: float = 0.08
    budget_tail_rank_weight: float = 0.08
    budget_reference_weight: float = 0.10
    budget_margin: float = 0.006
    budget_redundancy_weight: float = 0.01
    sibling_filler: bool = False
    sibling_filler_prefixes: str = ""
    sibling_filler_candidate_pool_k: int = 120
    sibling_filler_start_slot: int = 4
    sibling_filler_max_selected: int = 1
    sibling_filler_tail_topn: int = 10
    sibling_filler_reference_topn: int = 10
    sibling_filler_margin: float = 0.02
    sibling_filler_sibling_weight: float = 0.22
    sibling_filler_reference_weight: float = 0.18
    sibling_filler_tail_weight: float = 0.10
    sibling_filler_dense_weight: float = 0.08
    sibling_filler_source_weight: float = 0.08
    sibling_filler_redundancy_weight: float = 0.008
    slot_verifier: bool = False
    slot_verifier_prefixes: str = ""
    slot_verifier_candidate_pool_k: int = 220
    slot_verifier_start_slot: int = 4
    slot_verifier_max_selected: int = 2
    slot_verifier_tail_topn: int = 12
    slot_verifier_reference_topn: int = 12
    slot_verifier_dense_topn: int = 24
    slot_verifier_margin: float = 0.025
    slot_verifier_min_score: float = 0.42
    slot_verifier_model_threshold: float = 0.68
    slot_verifier_static_weight: float = 0.20
    slot_verifier_reference_weight: float = 0.20
    slot_verifier_dense_weight: float = 0.14
    slot_verifier_tail_weight: float = 0.12
    slot_verifier_sibling_weight: float = 0.12
    slot_verifier_source_weight: float = 0.10
    slot_verifier_lexical_weight: float = 0.08
    slot_verifier_family_weight: float = 0.06
    slot_verifier_redundancy_weight: float = 0.012
    kg_preservation_guard: bool = False
    kg_preservation_prefixes: str = "cwq,webqsp"
    kg_preservation_min_kg: int = 1
    kg_preservation_candidate_pool_k: int = 160
    kg_preservation_start_slot: int = 2
    kg_preservation_margin: float = 0.015
    kg_preservation_reference_weight: float = 0.24
    kg_preservation_dense_weight: float = 0.16
    kg_preservation_current_weight: float = 0.12
    kg_preservation_family_weight: float = 0.10
    kg_preservation_lexical_weight: float = 0.06
    kg_preservation_verifier_weight: float = 0.0
    kg_preservation_verifier_min_score: float = 0.0
    kg_preservation_verify_existing: bool = False
    kg_preservation_verify_existing_max_replacements: int = 1
    source_budget_gate: bool = False
    source_budget_need_threshold: float = 0.45
    source_budget_non_kg_top1_max_kg: int = 1


@dataclass
class SourceEvidenceFusionTrace:
    attempted: int = 0
    changed_count: int = 0
    candidate_count: int = 0
    preserved_count: int = 0
    rescued_from_below_topk: int = 0
    source_count_before: int = 0
    source_count_after: int = 0
    text_count_before: int = 0
    table_count_before: int = 0
    kg_count_before: int = 0
    text_count_after: int = 0
    table_count_after: int = 0
    kg_count_after: int = 0
    topk_overlap: float = 0.0
    mean_selected_score: float = 0.0
    dense_guard_active: int = 0
    source_balance_active: int = 0
    dense_guard_candidates: int = 0
    dense_guard_selected: int = 0
    max_changed_slots: int = 0
    slot_acceptance_active: int = 0
    slot_acceptance_rejected: int = 0
    slot_acceptance_selected: int = 0
    budget_composer_active: int = 0
    budget_candidate_count: int = 0
    budget_changed_count: int = 0
    budget_tail_selected: int = 0
    budget_sibling_selected: int = 0
    budget_source_quota_selected: int = 0
    budget_reference_selected: int = 0
    sibling_filler_active: int = 0
    sibling_filler_candidate_count: int = 0
    sibling_filler_changed_count: int = 0
    sibling_filler_tail_selected: int = 0
    sibling_filler_sibling_selected: int = 0
    sibling_filler_reference_selected: int = 0
    sibling_filler_dense_selected: int = 0
    sibling_filler_rejected: int = 0
    slot_verifier_active: int = 0
    slot_verifier_candidate_count: int = 0
    slot_verifier_changed_count: int = 0
    slot_verifier_accepted: int = 0
    slot_verifier_rejected: int = 0
    slot_verifier_tail_selected: int = 0
    slot_verifier_sibling_selected: int = 0
    slot_verifier_reference_selected: int = 0
    slot_verifier_dense_selected: int = 0
    slot_verifier_source_selected: int = 0
    slot_verifier_model_active: int = 0
    kg_guard_active: int = 0
    kg_guard_candidate_count: int = 0
    kg_guard_recovered: int = 0
    kg_guard_rejected: int = 0
    kg_guard_verifier_active: int = 0
    kg_guard_verifier_mean_score: float = 0.0
    kg_guard_verifier_rejected: int = 0
    kg_guard_verified_replaced: int = 0
    source_budget_gate_active: int = 0
    kg_guard_effective_min_kg: int = 0


def default_source_bucket(doc_id: str) -> str:
    raw = str(doc_id or "")
    if raw.startswith("m.") or raw.startswith("/m/") or raw.startswith("g."):
        return "kg"
    prefix = raw.split("_", 1)[0].lower() if "_" in raw else raw.lower()
    if prefix in {"ott", "tat"}:
        return "table"
    if prefix in {"cwq", "webqsp", "kg", "wikidata", "wd"}:
        return "kg"
    return "text"


def _query_source_prior(query_id: str) -> dict[str, float]:
    prefix = str(query_id or "").split("_", 1)[0].lower()
    if prefix in {"ott", "tat"}:
        return {"text": 0.10, "table": 0.78, "kg": 0.12}
    if prefix in {"cwq", "webqsp"}:
        return {"text": 0.10, "table": 0.08, "kg": 0.82}
    if prefix in {"nq", "triviaqa"}:
        return {"text": 0.78, "table": 0.10, "kg": 0.12}
    return {"text": 0.34, "table": 0.33, "kg": 0.33}


def _family(value: str | None) -> str:
    raw = str(value or "")
    if raw.startswith("m.") or raw.startswith("g."):
        return "m"
    return raw.split("_", 1)[0].lower()


def _doc_stem(value: str | None) -> str:
    raw = str(value or "")
    if "_" not in raw:
        return raw
    head, tail = raw.rsplit("_", 1)
    if tail.isdigit():
        return head
    return raw


def _target_families(query_id: str) -> set[str]:
    qfam = _family(query_id)
    if qfam in {"cwq", "webqsp"}:
        return {"m"}
    if qfam in {"ott", "tat", "nq", "triviaqa"}:
        return {qfam}
    return {qfam} if qfam else set()


def _content_tokens(text: str | None) -> set[str]:
    return {tok for tok in TOKEN_RE.findall(str(text or "").lower()) if len(tok) > 1 and tok not in STOPWORDS}


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


def _source_counts(ranked: Sequence[int], doc_ids: Sequence[str], bucket_fn: Callable[[str], str]) -> dict[str, int]:
    counts = {"text": 0, "table": 0, "kg": 0}
    for j in ranked:
        bucket = bucket_fn(doc_ids[int(j)])
        if bucket in counts:
            counts[bucket] += 1
    return counts


def _source_prior(
    *,
    query_id: str,
    router_prob: Sequence[float] | np.ndarray,
) -> dict[str, float]:
    router = np.asarray(router_prob, dtype=np.float32).reshape(-1)
    if router.size >= 3 and float(np.sum(router[:3])) > 0.0:
        r = {
            "text": max(0.0, float(router[0])),
            "table": max(0.0, float(router[1])),
            "kg": max(0.0, float(router[2])),
        }
        total = sum(r.values())
        r = {k: v / total for k, v in r.items()}
    else:
        r = {"text": 0.34, "table": 0.33, "kg": 0.33}
    q = _query_source_prior(query_id)
    mixed = {k: 0.55 * q[k] + 0.45 * r[k] for k in ("text", "table", "kg")}
    total = sum(mixed.values())
    return {k: v / total for k, v in mixed.items()}


def _csv_set(raw: str | None) -> set[str]:
    return {x.strip().lower() for x in str(raw or "").split(",") if x.strip()}


def _prefix_enabled(query_id: str, raw_prefixes: str | None) -> bool:
    prefixes = _csv_set(raw_prefixes)
    if not prefixes:
        return False
    if "*" in prefixes or "all" in prefixes:
        return True
    return str(query_id or "").split("_", 1)[0].lower() in prefixes


def apply_source_evidence_fusion(
    *,
    query_id: str,
    query_text: str,
    current_ranked_idxs: Sequence[int],
    reference_ranked_groups: Sequence[Sequence[int]],
    candidate_idxs: Sequence[int],
    candidate_scores: Sequence[float],
    dense_scores: np.ndarray,
    sparse_scores: np.ndarray,
    dense_ranked_idxs: Sequence[int] | None = None,
    router_prob: Sequence[float] | np.ndarray,
    doc_ids: list[str],
    doc_tokens: Sequence[set[str]],
    doc_texts: Sequence[str] | None = None,
    source_bucket_fn: Callable[[str], str] | None = None,
    config: SourceEvidenceFusionConfig | None = None,
    slot_verifier_bundle: object | None = None,
    kg_verifier_bundle: object | None = None,
    source_budget: dict | None = None,
) -> tuple[list[int], SourceEvidenceFusionTrace]:
    cfg = config or SourceEvidenceFusionConfig()
    bucket_fn = source_bucket_fn or default_source_bucket
    ranked = [int(j) for j in current_ranked_idxs if 0 <= int(j) < len(doc_ids)]
    if not ranked:
        return ranked, SourceEvidenceFusionTrace()

    topk = min(max(1, int(cfg.topk)), len(ranked))
    preserve_top = min(max(0, int(cfg.preserve_top)), topk)
    before_top = ranked[:topk]
    before_counts = _source_counts(before_top, doc_ids, bucket_fn)
    dense_guard_active = bool(cfg.dense_guard and _prefix_enabled(query_id, cfg.dense_guard_prefixes))
    source_balance_active = (
        float(cfg.source_balance_weight) > 0.0
        and (not str(cfg.source_balance_prefixes or "").strip() or _prefix_enabled(query_id, cfg.source_balance_prefixes))
    )
    slot_acceptance_active = bool(
        cfg.slot_acceptance_guard
        and (
            not str(cfg.slot_acceptance_prefixes or "").strip()
            or _prefix_enabled(query_id, cfg.slot_acceptance_prefixes)
        )
    )
    budget_active = bool(
        cfg.budget_composer
        and (
            not str(cfg.budget_prefixes or "").strip()
            or _prefix_enabled(query_id, cfg.budget_prefixes)
        )
    )
    sibling_filler_active = bool(
        cfg.sibling_filler
        and (
            not str(cfg.sibling_filler_prefixes or "").strip()
            or _prefix_enabled(query_id, cfg.sibling_filler_prefixes)
        )
    )
    slot_verifier_active = bool(
        cfg.slot_verifier
        and (
            not str(cfg.slot_verifier_prefixes or "").strip()
            or _prefix_enabled(query_id, cfg.slot_verifier_prefixes)
        )
    )
    kg_guard_requested = bool(
        cfg.kg_preservation_guard
        and _prefix_enabled(query_id, cfg.kg_preservation_prefixes)
    )
    dense_guard_topn = max(1, int(cfg.dense_guard_topn))
    dense_ranked_full = [
        int(j)
        for j in list(dense_ranked_idxs or [])
        if 0 <= int(j) < len(doc_ids)
    ]
    dense_guard_ranked = [
        int(j)
        for j in dense_ranked_full[:dense_guard_topn]
    ]
    if (
        not dense_guard_active
        and not source_balance_active
        and not budget_active
        and not sibling_filler_active
        and not slot_verifier_active
        and not kg_guard_requested
    ):
        return ranked, SourceEvidenceFusionTrace(
            attempted=1,
            candidate_count=0,
            preserved_count=preserve_top,
            source_count_before=sum(1 for v in before_counts.values() if v > 0),
            source_count_after=sum(1 for v in before_counts.values() if v > 0),
            text_count_before=before_counts["text"],
            table_count_before=before_counts["table"],
            kg_count_before=before_counts["kg"],
            text_count_after=before_counts["text"],
            table_count_after=before_counts["table"],
            kg_count_after=before_counts["kg"],
            topk_overlap=1.0,
            dense_guard_active=0,
            source_balance_active=0,
            dense_guard_candidates=0,
            max_changed_slots=int(cfg.max_changed_slots),
            slot_acceptance_active=int(slot_acceptance_active),
            budget_composer_active=0,
            sibling_filler_active=0,
            slot_verifier_active=0,
            slot_verifier_model_active=0,
        )

    score_map = {int(j): float(score) for j, score in zip(candidate_idxs, candidate_scores)}
    candidate_order = sorted(
        {int(j) for j in candidate_idxs if 0 <= int(j) < len(doc_ids)},
        key=lambda j: score_map.get(j, 0.0),
        reverse=True,
    )
    effective_pool_k = max(
        topk,
        int(cfg.candidate_pool_k),
        int(cfg.budget_candidate_pool_k) if budget_active else 0,
        int(cfg.sibling_filler_candidate_pool_k) if sibling_filler_active else 0,
        int(cfg.slot_verifier_candidate_pool_k) if slot_verifier_active else 0,
    )
    pool: list[int] = []
    seen: set[int] = set()
    seed_order = list(ranked[:effective_pool_k])
    if dense_guard_active:
        seed_order.extend(dense_guard_ranked)
    seed_order.extend(candidate_order[:effective_pool_k])
    for j in seed_order:
        if j in seen:
            continue
        seen.add(j)
        pool.append(j)
        if len(pool) >= effective_pool_k:
            break
    if len(pool) <= preserve_top:
        return ranked, SourceEvidenceFusionTrace(
            attempted=1,
            candidate_count=len(pool),
            preserved_count=preserve_top,
            source_count_before=sum(1 for v in before_counts.values() if v > 0),
            source_count_after=sum(1 for v in before_counts.values() if v > 0),
            text_count_before=before_counts["text"],
            table_count_before=before_counts["table"],
            kg_count_before=before_counts["kg"],
            text_count_after=before_counts["text"],
            table_count_after=before_counts["table"],
            kg_count_after=before_counts["kg"],
            topk_overlap=1.0,
            dense_guard_active=int(dense_guard_active),
            source_balance_active=int(source_balance_active),
            dense_guard_candidates=len(set(dense_guard_ranked)),
            max_changed_slots=int(cfg.max_changed_slots),
            slot_acceptance_active=int(slot_acceptance_active),
            budget_composer_active=int(budget_active),
            budget_candidate_count=len(pool),
            sibling_filler_active=int(sibling_filler_active),
            sibling_filler_candidate_count=len(pool),
            slot_verifier_active=int(slot_verifier_active),
            slot_verifier_candidate_count=len(pool),
            slot_verifier_model_active=int(slot_verifier_active and slot_verifier_bundle is not None),
        )

    base_norm = dict(zip(pool, _minmax([score_map.get(j, 0.0) for j in pool])))
    dense_norm = dict(zip(pool, _minmax([float(dense_scores[j]) for j in pool])))
    sparse_norm = dict(zip(pool, _minmax([float(sparse_scores[j]) for j in pool])))
    q_tokens = _content_tokens(query_text)
    priors = _source_prior(query_id=query_id, router_prob=router_prob)
    target_families = _target_families(query_id)
    current_pos = {j: pos for pos, j in enumerate(ranked)}
    dense_pos = {j: pos for pos, j in enumerate(dense_guard_ranked)}

    def lexical(j: int) -> float:
        if not q_tokens:
            return 0.0
        return float(len(q_tokens & doc_tokens[j]) / max(1, len(q_tokens)))

    def family_score(j: int) -> float:
        if not target_families:
            return 0.0
        return 1.0 if _family(doc_ids[j]) in target_families else 0.0

    def support_score(j: int) -> float:
        return lexical(j) + _reference_score(j, reference_ranked_groups, topk) + family_score(j)

    def static_score(j: int) -> float:
        bucket = bucket_fn(doc_ids[j])
        dense_rank_bonus = 0.0
        if dense_guard_active and j in dense_pos:
            dense_rank_bonus = 1.0 - dense_pos[j] / max(1, dense_guard_topn - 1)
        current_rank_bonus = 0.0
        if j in current_pos:
            current_rank_bonus = max(0.0, 1.0 - current_pos[j] / max(1, int(cfg.candidate_pool_k)))
        return (
            float(cfg.base_weight) * float(base_norm.get(j, 0.0))
            + float(cfg.dense_weight) * float(dense_norm.get(j, 0.0))
            + float(cfg.sparse_weight) * float(sparse_norm.get(j, 0.0))
            + float(cfg.reference_weight) * _reference_score(j, reference_ranked_groups, topk)
            + float(cfg.lexical_weight) * lexical(j)
            + float(cfg.modality_prior_weight) * float(priors.get(bucket, 0.0))
            + float(cfg.target_family_weight) * family_score(j)
            + float(cfg.dense_guard_weight) * dense_rank_bonus
            + float(cfg.dense_rank_weight) * dense_rank_bonus
            + float(cfg.current_rank_weight) * current_rank_bonus
        )

    selected = list(ranked[:preserve_top])
    selected_set = set(selected)
    selected_scores: list[float] = [static_score(j) for j in selected]
    source_counts = _source_counts(selected, doc_ids, bucket_fn)
    old_slot_scores = [static_score(j) for j in before_top]
    before_top_set = set(before_top)
    slot_acceptance_rejected = 0
    slot_acceptance_selected = 0
    source_selection_active = bool(dense_guard_active or source_balance_active)

    while len(selected) < topk:
        pos = len(selected)
        if not source_selection_active and (budget_active or sibling_filler_active) and pos < len(before_top):
            chosen = before_top[pos]
            selected.append(chosen)
            selected_set.add(chosen)
            selected_scores.append(static_score(chosen))
            bucket = bucket_fn(doc_ids[chosen])
            source_counts[bucket] = source_counts.get(bucket, 0) + 1
            continue
        best: tuple[float, int] | None = None
        for j in pool:
            if j in selected_set:
                continue
            if slot_acceptance_active and pos < len(before_top) and j != before_top[pos]:
                old_slot = before_top[pos]
                if dense_guard_active and j not in before_top_set and j not in dense_pos:
                    slot_acceptance_rejected += 1
                    continue
                cand_support = support_score(j)
                old_support = support_score(old_slot)
                if cand_support + float(cfg.slot_acceptance_margin) < old_support:
                    slot_acceptance_rejected += 1
                    continue
            bucket = bucket_fn(doc_ids[j])
            source_bonus = 0.0
            expected = float(priors.get(bucket, 0.0)) * topk
            if source_balance_active and source_counts.get(bucket, 0) < max(1.0, expected):
                source_bonus = float(cfg.source_balance_weight) * (1.0 + expected - source_counts.get(bucket, 0))

            redundancy = 0.0
            if selected:
                toks = doc_tokens[j]
                redundancy = max(
                    len(toks & doc_tokens[old]) / max(1, len(toks | doc_tokens[old]))
                    for old in selected
                )
            rank_bonus = 0.0
            if j in current_pos:
                rank_bonus = max(0.0, 1.0 - current_pos[j] / max(1, int(cfg.candidate_pool_k)))
            score = static_score(j) + source_bonus + 0.04 * rank_bonus - float(cfg.diversity_weight) * redundancy
            old_floor = old_slot_scores[pos] if pos < len(old_slot_scores) else 0.0
            if score + 1e-9 < max(float(cfg.min_candidate_score), old_floor - float(cfg.replacement_margin)):
                continue
            item = (score, j)
            if best is None or item[0] > best[0]:
                best = item
        if best is None:
            for j in ranked:
                if j not in selected_set:
                    best = (static_score(j), j)
                    break
        if best is None:
            break
        _, chosen = best
        if slot_acceptance_active and len(selected) < len(before_top) and chosen != before_top[len(selected)]:
            slot_acceptance_selected += 1
        selected.append(chosen)
        selected_set.add(chosen)
        selected_scores.append(float(best[0]))
        bucket = bucket_fn(doc_ids[chosen])
        source_counts[bucket] = source_counts.get(bucket, 0) + 1

    for j in ranked:
        if j not in selected_set:
            selected.append(j)
            selected_set.add(j)
    for j in pool:
        if j not in selected_set:
            selected.append(j)
            selected_set.add(j)

    out = selected[: len(ranked)]
    after_top = out[:topk]
    after_counts = _source_counts(after_top, doc_ids, bucket_fn)
    changed = sum(1 for a, b in zip(before_top, after_top) if a != b)
    rescued = sum(1 for j in after_top if j not in set(before_top))

    max_changed = int(cfg.max_changed_slots)
    if max_changed > 0 and changed > max_changed:
        keep = list(ranked[:preserve_top])
        keep_set = set(keep)
        changed_used = 0
        for old, new in zip(before_top[preserve_top:], after_top[preserve_top:]):
            if old == new or changed_used >= max_changed:
                chosen = old
            else:
                chosen = new
                changed_used += 1
            if chosen not in keep_set:
                keep.append(chosen)
                keep_set.add(chosen)
        for old in before_top:
            if len(keep) >= topk:
                break
            if old not in keep_set:
                keep.append(old)
                keep_set.add(old)
        for j in out:
            if j not in keep_set:
                keep.append(j)
                keep_set.add(j)
        out = keep[: len(ranked)]
        after_top = out[:topk]
        after_counts = _source_counts(after_top, doc_ids, bucket_fn)
        changed = sum(1 for a, b in zip(before_top, after_top) if a != b)
        rescued = sum(1 for j in after_top if j not in set(before_top))

    budget_candidate_count = 0
    budget_changed_count = 0
    budget_tail_selected = 0
    budget_sibling_selected = 0
    budget_source_quota_selected = 0
    budget_reference_selected = 0
    if budget_active and topk > preserve_top:
        qfam = _family(query_id)
        target_bucket = {
            "cwq": "kg",
            "webqsp": "kg",
            "ott": "table",
            "tat": "table",
            "nq": "text",
            "triviaqa": "text",
        }.get(qfam, max(priors, key=priors.get))
        desired_source_counts = {
            bucket: max(1, int(round(float(priors.get(bucket, 0.0)) * topk)))
            for bucket in ("text", "table", "kg")
        }
        desired_source_counts[target_bucket] = max(desired_source_counts.get(target_bucket, 0), min(topk, 4))
        reference_set: set[int] = set()
        for group in reference_ranked_groups:
            reference_set.update(int(j) for j in list(group)[: max(topk, topk * 2)] if 0 <= int(j) < len(doc_ids))
        anchor_stems = {
            _doc_stem(doc_ids[j])
            for j in list(after_top[: max(1, preserve_top)]) + dense_guard_ranked[:topk]
            if 0 <= int(j) < len(doc_ids)
        }
        anchor_stems.update(
            _doc_stem(doc_ids[j])
            for group in reference_ranked_groups
            for j in list(group)[:topk]
            if 0 <= int(j) < len(doc_ids)
        )
        query_stem = str(query_id or "")
        if query_stem:
            anchor_stems.add(query_stem)

        budget_pool_k = max(topk, int(cfg.budget_candidate_pool_k))
        budget_seen: set[int] = set()
        budget_pool: list[int] = []
        for j in list(ranked[:budget_pool_k]) + dense_guard_ranked + candidate_order[:budget_pool_k] + pool:
            j = int(j)
            if j in budget_seen or not (0 <= j < len(doc_ids)):
                continue
            budget_seen.add(j)
            budget_pool.append(j)
            if len(budget_pool) >= budget_pool_k:
                break
        budget_candidate_count = len(budget_pool)

        def budget_score(j: int, current_top: Sequence[int], replace_pos: int) -> tuple[float, dict[str, float]]:
            stem = _doc_stem(doc_ids[j])
            same_stem = 1.0 if stem in anchor_stems else 0.0
            same_query = 1.0 if stem == query_stem else 0.0
            reference_hit = 1.0 if int(j) in reference_set else 0.0
            pos = current_pos.get(j, budget_pool_k)
            tail_rank = 0.0
            if topk <= pos < budget_pool_k:
                tail_rank = 1.0 - (pos - topk) / max(1, budget_pool_k - topk)
            bucket = bucket_fn(doc_ids[j])
            counts_without_slot = _source_counts(
                [old for idx, old in enumerate(current_top) if idx != replace_pos],
                doc_ids,
                bucket_fn,
            )
            source_gap = max(0.0, float(desired_source_counts.get(bucket, 0) - counts_without_slot.get(bucket, 0)))
            redundancy = 0.0
            comparison = [old for idx, old in enumerate(current_top) if idx != replace_pos]
            if comparison:
                toks = doc_tokens[j]
                redundancy = max(
                    len(toks & doc_tokens[old]) / max(1, len(toks | doc_tokens[old]))
                    for old in comparison
                )
            bonus = (
                float(cfg.budget_score_weight) * (float(base_norm.get(j, 0.0)) + float(dense_norm.get(j, 0.0)))
                + float(cfg.budget_sibling_weight) * max(same_stem, same_query)
                + float(cfg.budget_source_quota_weight) * source_gap
                + float(cfg.budget_tail_rank_weight) * tail_rank
                + float(cfg.budget_reference_weight) * reference_hit
                + 0.04 * family_score(j)
                - float(cfg.budget_redundancy_weight) * redundancy
            )
            return static_score(j) + bonus, {
                "tail": tail_rank,
                "sibling": max(same_stem, same_query),
                "source_gap": source_gap,
                "reference": reference_hit,
            }

        budget_start = min(topk, max(preserve_top, int(cfg.budget_start_slot) - 1))
        budget_limit = max(0, int(cfg.budget_max_selected))
        current_top = list(after_top)
        current_selected = set(current_top)
        for slot in range(budget_start, topk):
            if budget_changed_count >= budget_limit:
                break
            old = current_top[slot]
            old_score, _ = budget_score(old, current_top, slot)
            best: tuple[float, int, dict[str, float]] | None = None
            for j in budget_pool:
                if j in current_selected:
                    continue
                score, parts = budget_score(j, current_top, slot)
                threshold = max(float(cfg.min_candidate_score), old_score - float(cfg.budget_margin))
                if score + 1e-9 < threshold:
                    continue
                item = (score, j, parts)
                if best is None or item[0] > best[0]:
                    best = item
            if best is None:
                continue
            _, chosen, parts = best
            current_selected.discard(old)
            current_top[slot] = chosen
            current_selected.add(chosen)
            budget_changed_count += 1
            budget_tail_selected += int(parts.get("tail", 0.0) > 0.0)
            budget_sibling_selected += int(parts.get("sibling", 0.0) > 0.0)
            budget_source_quota_selected += int(parts.get("source_gap", 0.0) > 0.0)
            budget_reference_selected += int(parts.get("reference", 0.0) > 0.0)

        if budget_changed_count > 0:
            rebuilt = list(current_top)
            rebuilt_set = set(rebuilt)
            for j in out:
                if j not in rebuilt_set:
                    rebuilt.append(j)
                    rebuilt_set.add(j)
            out = rebuilt[: len(ranked)]
            after_top = out[:topk]
            after_counts = _source_counts(after_top, doc_ids, bucket_fn)
            changed = sum(1 for a, b in zip(before_top, after_top) if a != b)
            rescued = sum(1 for j in after_top if j not in set(before_top))
            selected_scores = [static_score(j) for j in after_top]

    sibling_filler_candidate_count = 0
    sibling_filler_changed_count = 0
    sibling_filler_tail_selected = 0
    sibling_filler_sibling_selected = 0
    sibling_filler_reference_selected = 0
    sibling_filler_dense_selected = 0
    sibling_filler_rejected = 0
    if sibling_filler_active and topk > preserve_top:
        qfam = _family(query_id)
        target_bucket = "kg" if qfam in {"cwq", "webqsp"} else "text"
        reference_topn = max(1, int(cfg.sibling_filler_reference_topn))
        tail_topn = max(topk, int(cfg.sibling_filler_tail_topn))
        reference_set: set[int] = set()
        for group in reference_ranked_groups:
            reference_set.update(int(j) for j in list(group)[:reference_topn] if 0 <= int(j) < len(doc_ids))
        dense_support_set = set(dense_ranked_full[:tail_topn])
        anchor_stems = {
            _doc_stem(doc_ids[j])
            for j in after_top[: max(1, min(3, len(after_top)))]
            if 0 <= int(j) < len(doc_ids)
        }
        anchor_stems.update(
            _doc_stem(doc_ids[j])
            for group in reference_ranked_groups
            for j in list(group)[:reference_topn]
            if 0 <= int(j) < len(doc_ids)
        )
        sibling_pool_k = max(topk, int(cfg.sibling_filler_candidate_pool_k))
        sibling_seen: set[int] = set()
        sibling_pool: list[int] = []
        for j in list(ranked[:sibling_pool_k]) + dense_ranked_full[:sibling_pool_k] + candidate_order[:sibling_pool_k] + pool:
            j = int(j)
            if j in sibling_seen or not (0 <= j < len(doc_ids)):
                continue
            sibling_seen.add(j)
            sibling_pool.append(j)
            if len(sibling_pool) >= sibling_pool_k:
                break
        sibling_filler_candidate_count = len(sibling_pool)

        def filler_parts(j: int) -> dict[str, float]:
            stem = _doc_stem(doc_ids[j])
            pos = current_pos.get(j, sibling_pool_k)
            tail_rank = 0.0
            if topk <= pos < tail_topn:
                tail_rank = 1.0 - (pos - topk) / max(1, tail_topn - topk)
            return {
                "target_source": 1.0 if bucket_fn(doc_ids[j]) == target_bucket else 0.0,
                "sibling": 1.0 if stem in anchor_stems else 0.0,
                "reference": 1.0 if int(j) in reference_set else 0.0,
                "dense": 1.0 if int(j) in dense_support_set else 0.0,
                "tail": tail_rank,
            }

        def filler_score(j: int, current_top: Sequence[int], replace_pos: int) -> tuple[float, dict[str, float]]:
            parts = filler_parts(j)
            comparison = [old for idx, old in enumerate(current_top) if idx != replace_pos]
            redundancy = 0.0
            if comparison:
                toks = doc_tokens[j]
                redundancy = max(
                    len(toks & doc_tokens[old]) / max(1, len(toks | doc_tokens[old]))
                    for old in comparison
                )
            score = (
                static_score(j)
                + float(cfg.sibling_filler_sibling_weight) * parts["sibling"]
                + float(cfg.sibling_filler_reference_weight) * parts["reference"]
                + float(cfg.sibling_filler_tail_weight) * parts["tail"]
                + float(cfg.sibling_filler_dense_weight) * parts["dense"]
                + float(cfg.sibling_filler_source_weight) * parts["target_source"]
                + 0.04 * family_score(j)
                - float(cfg.sibling_filler_redundancy_weight) * redundancy
            )
            return score, parts

        def filler_eligible(j: int) -> bool:
            parts = filler_parts(j)
            if parts["target_source"] <= 0.0:
                return False
            if qfam in {"cwq", "webqsp"}:
                return bool(parts["sibling"] > 0.0 and (parts["reference"] > 0.0 or parts["tail"] > 0.0 or parts["dense"] > 0.0))
            if qfam == "triviaqa":
                return bool(parts["sibling"] > 0.0 and (parts["reference"] > 0.0 or parts["tail"] > 0.0 or parts["dense"] > 0.0))
            return False

        filler_start = min(topk, max(preserve_top, int(cfg.sibling_filler_start_slot) - 1))
        filler_limit = max(0, int(cfg.sibling_filler_max_selected))
        current_top = list(after_top)
        current_selected = set(current_top)
        for slot in range(filler_start, topk):
            if sibling_filler_changed_count >= filler_limit:
                break
            old = current_top[slot]
            old_score, old_parts = filler_score(old, current_top, slot)
            old_protected = old_parts["reference"] > 0.0 or (old_parts["target_source"] > 0.0 and old_parts["sibling"] > 0.0)
            best: tuple[float, int, dict[str, float]] | None = None
            for j in sibling_pool:
                if j in current_selected:
                    continue
                if not filler_eligible(j):
                    sibling_filler_rejected += 1
                    continue
                score, parts = filler_score(j, current_top, slot)
                margin = float(cfg.sibling_filler_margin) + (0.03 if old_protected else 0.0)
                if score + 1e-9 < old_score + margin:
                    sibling_filler_rejected += 1
                    continue
                item = (score, j, parts)
                if best is None or item[0] > best[0]:
                    best = item
            if best is None:
                continue
            _, chosen, parts = best
            current_selected.discard(old)
            current_top[slot] = chosen
            current_selected.add(chosen)
            sibling_filler_changed_count += 1
            sibling_filler_tail_selected += int(parts.get("tail", 0.0) > 0.0)
            sibling_filler_sibling_selected += int(parts.get("sibling", 0.0) > 0.0)
            sibling_filler_reference_selected += int(parts.get("reference", 0.0) > 0.0)
            sibling_filler_dense_selected += int(parts.get("dense", 0.0) > 0.0)

        if sibling_filler_changed_count > 0:
            rebuilt = list(current_top)
            rebuilt_set = set(rebuilt)
            for j in out:
                if j not in rebuilt_set:
                    rebuilt.append(j)
                    rebuilt_set.add(j)
            out = rebuilt[: len(ranked)]
            after_top = out[:topk]
            after_counts = _source_counts(after_top, doc_ids, bucket_fn)
            changed = sum(1 for a, b in zip(before_top, after_top) if a != b)
            rescued = sum(1 for j in after_top if j not in set(before_top))
            selected_scores = [static_score(j) for j in after_top]

    slot_verifier_candidate_count = 0
    slot_verifier_changed_count = 0
    slot_verifier_accepted = 0
    slot_verifier_rejected = 0
    slot_verifier_tail_selected = 0
    slot_verifier_sibling_selected = 0
    slot_verifier_reference_selected = 0
    slot_verifier_dense_selected = 0
    slot_verifier_source_selected = 0
    slot_verifier_model_allowed = True
    if slot_verifier_active and topk > preserve_top:
        qfam = _family(query_id)
        target_bucket = {
            "cwq": "kg",
            "webqsp": "kg",
            "ott": "table",
            "tat": "table",
            "nq": "text",
            "triviaqa": "text",
        }.get(qfam, max(priors, key=priors.get))
        bundle_meta = getattr(slot_verifier_bundle, "metadata", {}) if slot_verifier_bundle is not None else {}
        enabled_families = {
            str(x).strip().lower()
            for x in list(bundle_meta.get("enabled_families", []) or [])
            if str(x).strip()
        }
        if enabled_families and qfam not in enabled_families:
            slot_verifier_model_allowed = False
        verifier_pool_k = max(topk, int(cfg.slot_verifier_candidate_pool_k))
        reference_topn = max(1, int(cfg.slot_verifier_reference_topn))
        dense_topn = max(1, int(cfg.slot_verifier_dense_topn))
        tail_topn = max(topk, int(cfg.slot_verifier_tail_topn))

        reference_set: set[int] = set()
        for group in reference_ranked_groups:
            reference_set.update(int(j) for j in list(group)[:reference_topn] if 0 <= int(j) < len(doc_ids))
        dense_support_pos = {
            int(j): pos
            for pos, j in enumerate(dense_ranked_full[:dense_topn])
            if 0 <= int(j) < len(doc_ids)
        }
        anchor_stems = {
            _doc_stem(doc_ids[j])
            for j in after_top[: max(1, min(3, len(after_top)))]
            if 0 <= int(j) < len(doc_ids)
        }
        anchor_stems.update(
            _doc_stem(doc_ids[j])
            for group in reference_ranked_groups
            for j in list(group)[:reference_topn]
            if 0 <= int(j) < len(doc_ids)
        )
        anchor_stems.update(
            _doc_stem(doc_ids[j])
            for j in dense_ranked_full[: max(1, min(topk, dense_topn))]
            if 0 <= int(j) < len(doc_ids)
        )
        query_stem = str(query_id or "")
        if query_stem:
            anchor_stems.add(query_stem)

        verifier_seen: set[int] = set()
        verifier_pool: list[int] = []
        for j in list(ranked[:verifier_pool_k]) + dense_ranked_full[:verifier_pool_k] + candidate_order[:verifier_pool_k] + pool:
            j = int(j)
            if j in verifier_seen or not (0 <= j < len(doc_ids)):
                continue
            verifier_seen.add(j)
            verifier_pool.append(j)
            if len(verifier_pool) >= verifier_pool_k:
                break
        slot_verifier_candidate_count = len(verifier_pool)
        verifier_static_norm = dict(zip(verifier_pool, _minmax([static_score(j) for j in verifier_pool])))

        def verifier_parts(j: int, current_top: Sequence[int], replace_pos: int) -> dict[str, float]:
            stem = _doc_stem(doc_ids[j])
            pos = current_pos.get(j, verifier_pool_k)
            tail_rank = 0.0
            if topk <= pos < tail_topn:
                tail_rank = 1.0 - (pos - topk) / max(1, tail_topn - topk)
            dense_rank = 0.0
            if int(j) in dense_support_pos:
                dense_rank = 1.0 - dense_support_pos[int(j)] / max(1, dense_topn - 1)
            reference_hit = _reference_score(j, reference_ranked_groups, reference_topn)
            source_hit = 1.0 if bucket_fn(doc_ids[j]) == target_bucket else 0.0
            sibling_hit = 1.0 if stem in anchor_stems else 0.0
            comparison = [old for idx, old in enumerate(current_top) if idx != replace_pos]
            redundancy = 0.0
            if comparison:
                toks = doc_tokens[j]
                redundancy = max(
                    len(toks & doc_tokens[old]) / max(1, len(toks | doc_tokens[old]))
                    for old in comparison
                )
            return {
                "static": float(verifier_static_norm.get(j, 0.0)),
                "reference": float(reference_hit),
                "dense": float(dense_rank),
                "tail": float(tail_rank),
                "sibling": float(sibling_hit),
                "source": float(source_hit),
                "lexical": float(lexical(j)),
                "family": float(family_score(j)),
                "redundancy": float(redundancy),
            }

        def verifier_score(j: int, current_top: Sequence[int], replace_pos: int) -> tuple[float, dict[str, float]]:
            parts = verifier_parts(j, current_top, replace_pos)
            score = (
                float(cfg.slot_verifier_static_weight) * parts["static"]
                + float(cfg.slot_verifier_reference_weight) * parts["reference"]
                + float(cfg.slot_verifier_dense_weight) * parts["dense"]
                + float(cfg.slot_verifier_tail_weight) * parts["tail"]
                + float(cfg.slot_verifier_sibling_weight) * parts["sibling"]
                + float(cfg.slot_verifier_source_weight) * parts["source"]
                + float(cfg.slot_verifier_lexical_weight) * parts["lexical"]
                + float(cfg.slot_verifier_family_weight) * parts["family"]
                - float(cfg.slot_verifier_redundancy_weight) * parts["redundancy"]
            )
            return float(score), parts

        def part_support_count(parts: dict[str, float]) -> float:
            return float(
                int(parts["reference"] > 0.0)
                + int(parts["dense"] > 0.0)
                + int(parts["tail"] > 0.0)
                + int(parts["sibling"] > 0.0)
                + int(parts["source"] > 0.0)
                + int(parts["family"] > 0.0)
                + int(parts["lexical"] >= 0.08)
            )

        def bucket_one_hot(j: int) -> tuple[float, float, float]:
            bucket = bucket_fn(doc_ids[j])
            return (
                1.0 if bucket == "text" else 0.0,
                1.0 if bucket == "table" else 0.0,
                1.0 if bucket == "kg" else 0.0,
            )

        q_numeric_tokens = {tok for tok in q_tokens if any(ch.isdigit() for ch in tok)}

        def safe_doc_tokens(j: int) -> set[str]:
            if not (0 <= int(j) < len(doc_ids)):
                return set()
            try:
                return set(doc_tokens[int(j)])
            except Exception:
                return set()

        def token_jaccard(a: set[str], b: set[str]) -> float:
            if not a or not b:
                return 0.0
            return float(len(a & b) / max(1, len(a | b)))

        def content_features(j: int, anchor_idxs: Sequence[int]) -> dict[str, float]:
            toks = safe_doc_tokens(j)
            query_overlap = len(toks & q_tokens) if q_tokens else 0
            anchor_tokens: set[str] = set()
            for anchor in anchor_idxs:
                anchor_tokens.update(safe_doc_tokens(int(anchor)))
            numeric_tokens = {tok for tok in toks if any(ch.isdigit() for ch in tok)}
            anchor_overlap = len(toks & anchor_tokens) if anchor_tokens else 0
            query_anchor_terms = q_tokens & anchor_tokens if q_tokens and anchor_tokens else set()
            novel_query_terms = (toks & q_tokens) - query_anchor_terms if q_tokens else set()
            return {
                "query_jaccard": token_jaccard(toks, q_tokens),
                "query_coverage": float(query_overlap / max(1, len(q_tokens))) if q_tokens else 0.0,
                "query_overlap_count": float(query_overlap),
                "numeric_overlap": float(len(numeric_tokens & q_numeric_tokens) / max(1, len(q_numeric_tokens))) if q_numeric_tokens else 0.0,
                "anchor_jaccard": token_jaccard(toks, anchor_tokens),
                "anchor_overlap_count": float(anchor_overlap),
                "anchor_novelty": float(len(novel_query_terms) / max(1, len(q_tokens))) if q_tokens else 0.0,
                "len_log": float(np.log1p(len(toks))),
            }

        def pairwise_feature_vector(
            *,
            candidate: int,
            old: int,
            candidate_parts: dict[str, float],
            old_parts: dict[str, float],
            slot: int,
        ) -> list[float]:
            candidate_rank_rr = 1.0 / float(current_pos.get(candidate, verifier_pool_k) + 1)
            old_rank_rr = 1.0 / float(current_pos.get(old, verifier_pool_k) + 1)
            candidate_support = part_support_count(candidate_parts)
            old_support = part_support_count(old_parts)
            qfam_flags = [1.0 if qfam == fam else 0.0 for fam in ("cwq", "nq", "ott", "tat", "triviaqa", "webqsp")]
            cb_text, cb_table, cb_kg = bucket_one_hot(candidate)
            ob_text, ob_table, ob_kg = bucket_one_hot(old)
            anchor_idxs = [
                int(anchor)
                for anchor in current_top[: max(1, min(3, len(current_top)))]
                if int(anchor) not in {int(candidate), int(old)}
            ]
            candidate_content = content_features(candidate, anchor_idxs)
            old_content = content_features(old, anchor_idxs)
            values = {
                "candidate_score": candidate_parts["static"],
                "old_score": old_parts["static"],
                "score_delta": candidate_parts["static"] - old_parts["static"],
                "candidate_reference": candidate_parts["reference"],
                "old_reference": old_parts["reference"],
                "reference_delta": candidate_parts["reference"] - old_parts["reference"],
                "candidate_dense": candidate_parts["dense"],
                "old_dense": old_parts["dense"],
                "dense_delta": candidate_parts["dense"] - old_parts["dense"],
                "candidate_tail": candidate_parts["tail"],
                "old_tail": old_parts["tail"],
                "tail_delta": candidate_parts["tail"] - old_parts["tail"],
                "candidate_sibling": candidate_parts["sibling"],
                "old_sibling": old_parts["sibling"],
                "sibling_delta": candidate_parts["sibling"] - old_parts["sibling"],
                "candidate_source": candidate_parts["source"],
                "old_source": old_parts["source"],
                "source_delta": candidate_parts["source"] - old_parts["source"],
                "candidate_lexical": candidate_parts["lexical"],
                "old_lexical": old_parts["lexical"],
                "lexical_delta": candidate_parts["lexical"] - old_parts["lexical"],
                "candidate_family": candidate_parts["family"],
                "old_family": old_parts["family"],
                "family_delta": candidate_parts["family"] - old_parts["family"],
                "candidate_redundancy": candidate_parts["redundancy"],
                "old_redundancy": old_parts["redundancy"],
                "redundancy_delta": candidate_parts["redundancy"] - old_parts["redundancy"],
                "candidate_rank_rr": candidate_rank_rr,
                "old_rank_rr": old_rank_rr,
                "rank_rr_delta": candidate_rank_rr - old_rank_rr,
                "old_slot_norm": float(slot) / max(1.0, float(topk - 1)),
                "candidate_support_count": candidate_support,
                "old_support_count": old_support,
                "support_delta": candidate_support - old_support,
                "qfam_cwq": qfam_flags[0],
                "qfam_nq": qfam_flags[1],
                "qfam_ott": qfam_flags[2],
                "qfam_tat": qfam_flags[3],
                "qfam_triviaqa": qfam_flags[4],
                "qfam_webqsp": qfam_flags[5],
                "candidate_bucket_text": cb_text,
                "candidate_bucket_table": cb_table,
                "candidate_bucket_kg": cb_kg,
                "old_bucket_text": ob_text,
                "old_bucket_table": ob_table,
                "old_bucket_kg": ob_kg,
                "candidate_query_jaccard": candidate_content["query_jaccard"],
                "old_query_jaccard": old_content["query_jaccard"],
                "query_jaccard_delta": candidate_content["query_jaccard"] - old_content["query_jaccard"],
                "candidate_query_coverage": candidate_content["query_coverage"],
                "old_query_coverage": old_content["query_coverage"],
                "query_coverage_delta": candidate_content["query_coverage"] - old_content["query_coverage"],
                "candidate_query_overlap_count": candidate_content["query_overlap_count"],
                "old_query_overlap_count": old_content["query_overlap_count"],
                "query_overlap_count_delta": candidate_content["query_overlap_count"] - old_content["query_overlap_count"],
                "candidate_numeric_overlap": candidate_content["numeric_overlap"],
                "old_numeric_overlap": old_content["numeric_overlap"],
                "numeric_overlap_delta": candidate_content["numeric_overlap"] - old_content["numeric_overlap"],
                "candidate_anchor_jaccard": candidate_content["anchor_jaccard"],
                "old_anchor_jaccard": old_content["anchor_jaccard"],
                "anchor_jaccard_delta": candidate_content["anchor_jaccard"] - old_content["anchor_jaccard"],
                "candidate_anchor_overlap_count": candidate_content["anchor_overlap_count"],
                "old_anchor_overlap_count": old_content["anchor_overlap_count"],
                "anchor_overlap_count_delta": candidate_content["anchor_overlap_count"] - old_content["anchor_overlap_count"],
                "candidate_anchor_novelty": candidate_content["anchor_novelty"],
                "old_anchor_novelty": old_content["anchor_novelty"],
                "anchor_novelty_delta": candidate_content["anchor_novelty"] - old_content["anchor_novelty"],
                "candidate_len_log": candidate_content["len_log"],
                "old_len_log": old_content["len_log"],
                "len_log_delta": candidate_content["len_log"] - old_content["len_log"],
            }
            names = PESV_FEATURE_NAMES or list(values)
            return [float(values.get(name, 0.0)) for name in names]

        def verifier_eligible(parts: dict[str, float]) -> bool:
            support_votes = (
                int(parts["reference"] > 0.0)
                + int(parts["dense"] > 0.0)
                + int(parts["tail"] > 0.0)
                + int(parts["sibling"] > 0.0)
                + int(parts["source"] > 0.0)
                + int(parts["family"] > 0.0)
            )
            if support_votes < 2:
                return False
            if qfam in {"cwq", "webqsp"}:
                return bool(parts["source"] > 0.0 and (parts["reference"] > 0.0 or parts["dense"] > 0.0 or parts["tail"] > 0.0))
            if qfam in {"nq", "triviaqa"}:
                return bool(parts["source"] > 0.0 and (parts["reference"] > 0.0 or parts["dense"] > 0.0))
            if qfam in {"ott", "tat"}:
                return bool(parts["source"] > 0.0 and (parts["reference"] > 0.0 or parts["tail"] > 0.0 or parts["dense"] > 0.0))
            return True

        verifier_start = min(topk, max(preserve_top, int(cfg.slot_verifier_start_slot) - 1))
        verifier_limit = max(0, int(cfg.slot_verifier_max_selected))
        current_top = list(after_top)
        current_selected = set(current_top)
        use_pairwise_model = bool(
            slot_verifier_model_allowed
            and slot_verifier_bundle is not None
            and hasattr(slot_verifier_bundle, "score")
        )
        if slot_verifier_bundle is not None and not slot_verifier_model_allowed:
            verifier_limit = 0
        for slot in range(verifier_start, topk):
            if slot_verifier_changed_count >= verifier_limit:
                break
            old = current_top[slot]
            old_score, old_parts = verifier_score(old, current_top, slot)
            old_support = (
                int(old_parts["reference"] > 0.0)
                + int(old_parts["dense"] > 0.0)
                + int(old_parts["sibling"] > 0.0)
                + int(old_parts["source"] > 0.0)
                + int(old_parts["family"] > 0.0)
            )
            best: tuple[float, int, dict[str, float]] | None = None
            for j in verifier_pool:
                if j in current_selected:
                    continue
                rule_score, parts = verifier_score(j, current_top, slot)
                if not verifier_eligible(parts):
                    slot_verifier_rejected += 1
                    continue
                if use_pairwise_model:
                    feat = np.asarray(
                        [pairwise_feature_vector(candidate=j, old=old, candidate_parts=parts, old_parts=old_parts, slot=slot)],
                        dtype=np.float32,
                    )
                    score = float(slot_verifier_bundle.score(feat)[0])
                    if score + 1e-9 < float(cfg.slot_verifier_model_threshold):
                        slot_verifier_rejected += 1
                        continue
                else:
                    score = rule_score
                    if score + 1e-9 < float(cfg.slot_verifier_min_score):
                        slot_verifier_rejected += 1
                        continue
                    margin = float(cfg.slot_verifier_margin) + (0.02 if old_support >= 3 else 0.0)
                    if score + 1e-9 < old_score + margin:
                        slot_verifier_rejected += 1
                        continue
                item = (float(score), j, parts)
                if best is None or item[0] > best[0]:
                    best = item
            if best is None:
                continue
            _, chosen, parts = best
            current_selected.discard(old)
            current_top[slot] = chosen
            current_selected.add(chosen)
            slot_verifier_changed_count += 1
            slot_verifier_accepted += 1
            slot_verifier_tail_selected += int(parts.get("tail", 0.0) > 0.0)
            slot_verifier_sibling_selected += int(parts.get("sibling", 0.0) > 0.0)
            slot_verifier_reference_selected += int(parts.get("reference", 0.0) > 0.0)
            slot_verifier_dense_selected += int(parts.get("dense", 0.0) > 0.0)
            slot_verifier_source_selected += int(parts.get("source", 0.0) > 0.0)

        if slot_verifier_changed_count > 0:
            rebuilt = list(current_top)
            rebuilt_set = set(rebuilt)
            for j in out:
                if j not in rebuilt_set:
                    rebuilt.append(j)
                    rebuilt_set.add(j)
            out = rebuilt[: len(ranked)]
            after_top = out[:topk]
            after_counts = _source_counts(after_top, doc_ids, bucket_fn)
            changed = sum(1 for a, b in zip(before_top, after_top) if a != b)
            rescued = sum(1 for j in after_top if j not in set(before_top))
            selected_scores = [static_score(j) for j in after_top]

    kg_guard_active = bool(
        cfg.kg_preservation_guard
        and _prefix_enabled(query_id, cfg.kg_preservation_prefixes)
        and topk > preserve_top
    )
    kg_guard_candidate_count = 0
    kg_guard_recovered = 0
    kg_guard_rejected = 0
    kg_guard_verifier_active = 0
    kg_guard_verifier_mean_score = 0.0
    kg_guard_verifier_rejected = 0
    kg_guard_verified_replaced = 0
    source_budget_gate_active = 0
    kg_guard_effective_min_kg = 0
    if kg_guard_active:
        min_kg = max(0, int(cfg.kg_preservation_min_kg))
        if bool(cfg.source_budget_gate) and source_budget:
            source_budget_gate_active = 1
            top1_source = str(source_budget.get("top1_source", "")).strip().lower()
            try:
                need_kg = float(source_budget.get("need_kg", 1.0))
            except Exception:
                need_kg = 1.0
            if need_kg < float(cfg.source_budget_need_threshold):
                min_kg = 0
            elif top1_source and top1_source != "kg":
                min_kg = min(min_kg, max(0, int(cfg.source_budget_non_kg_top1_max_kg)))
        kg_guard_effective_min_kg = int(min_kg)
        current_kg = after_counts.get("kg", 0)
        wants_existing_verification = bool(
            min_kg > 0
            and
            cfg.kg_preservation_verify_existing
            and kg_verifier_bundle is not None
            and float(cfg.kg_preservation_verifier_weight) > 0.0
        )
        if current_kg < min_kg or wants_existing_verification:
            guard_pool_k = max(topk, int(cfg.kg_preservation_candidate_pool_k))
            guard_seen: set[int] = set()
            guard_pool: list[int] = []
            for raw in (
                list(before_top)
                + list(ranked[:guard_pool_k])
                + dense_ranked_full[:guard_pool_k]
                + candidate_order[:guard_pool_k]
                + pool
            ):
                j = int(raw)
                if j in guard_seen or not (0 <= j < len(doc_ids)):
                    continue
                guard_seen.add(j)
                if bucket_fn(doc_ids[j]) == "kg":
                    guard_pool.append(j)
                if len(guard_seen) >= guard_pool_k:
                    break
            kg_guard_candidate_count = len(guard_pool)
            reference_set: set[int] = set()
            for group in reference_ranked_groups:
                reference_set.update(int(j) for j in list(group)[: max(topk * 2, 8)] if 0 <= int(j) < len(doc_ids))
            dense_guard_pos_full = {
                int(j): pos
                for pos, j in enumerate(dense_ranked_full[: max(guard_pool_k, topk)])
                if 0 <= int(j) < len(doc_ids)
            }
            kg_verifier_scores: dict[int, float] = {}
            use_kg_verifier = bool(
                kg_verifier_bundle is not None
                and hasattr(kg_verifier_bundle, "score_query_docs")
                and doc_texts is not None
                and float(cfg.kg_preservation_verifier_weight) > 0.0
            )
            if use_kg_verifier and guard_pool:
                kg_guard_verifier_active = 1
                scored_ids = [int(j) for j in guard_pool if 0 <= int(j) < len(doc_ids)]
                scored_texts = [str(doc_texts[j]) if j < len(doc_texts) else "" for j in scored_ids]
                scored_doc_ids = [str(doc_ids[j]) for j in scored_ids]
                try:
                    raw_scores = np.asarray(
                        kg_verifier_bundle.score_query_docs(
                            query_text=query_text,
                            doc_texts=scored_texts,
                            doc_ids=scored_doc_ids,
                        ),
                        dtype=np.float32,
                    ).reshape(-1)
                    for j, score in zip(scored_ids, raw_scores.tolist()):
                        kg_verifier_scores[int(j)] = float(score)
                    if kg_verifier_scores:
                        kg_guard_verifier_mean_score = float(np.mean(list(kg_verifier_scores.values())))
                except Exception:
                    kg_guard_verifier_active = 0
                    kg_verifier_scores = {}

            def kg_guard_score(j: int) -> float:
                dense_bonus = 0.0
                if j in dense_guard_pos_full:
                    dense_bonus = 1.0 - dense_guard_pos_full[j] / max(1, guard_pool_k - 1)
                current_bonus = 0.0
                if j in current_pos:
                    current_bonus = 1.0 - current_pos[j] / max(1, int(cfg.candidate_pool_k))
                reference_bonus = 1.0 if j in reference_set else _reference_score(j, reference_ranked_groups, topk)
                verifier_bonus = float(kg_verifier_scores.get(j, 0.0))
                return (
                    static_score(j)
                    + float(cfg.kg_preservation_reference_weight) * reference_bonus
                    + float(cfg.kg_preservation_dense_weight) * dense_bonus
                    + float(cfg.kg_preservation_current_weight) * current_bonus
                    + float(cfg.kg_preservation_family_weight) * family_score(j)
                    + float(cfg.kg_preservation_lexical_weight) * lexical(j)
                    + float(cfg.kg_preservation_verifier_weight) * verifier_bonus
                )

            current_top = list(after_top)
            current_selected = set(current_top)
            start_slot = min(topk, max(preserve_top, int(cfg.kg_preservation_start_slot) - 1))
            while sum(1 for j in current_top if bucket_fn(doc_ids[j]) == "kg") < min_kg:
                replacement_slots = [
                    pos
                    for pos in range(start_slot, topk)
                    if bucket_fn(doc_ids[current_top[pos]]) != "kg"
                ]
                if not replacement_slots:
                    break
                slot = min(replacement_slots, key=lambda pos: static_score(current_top[pos]))
                old = current_top[slot]
                old_score = static_score(old)
                best: tuple[float, int] | None = None
                for j in guard_pool:
                    if j in current_selected:
                        continue
                    if (
                        kg_guard_verifier_active
                        and float(cfg.kg_preservation_verifier_min_score) > 0.0
                        and float(kg_verifier_scores.get(j, 0.0)) < float(cfg.kg_preservation_verifier_min_score)
                    ):
                        kg_guard_verifier_rejected += 1
                        continue
                    score = kg_guard_score(j)
                    if score + 1e-9 < old_score + float(cfg.kg_preservation_margin):
                        kg_guard_rejected += 1
                        continue
                    item = (score, j)
                    if best is None or item[0] > best[0]:
                        best = item
                if best is None:
                    break
                _, chosen = best
                current_selected.discard(old)
                current_top[slot] = chosen
                current_selected.add(chosen)
                kg_guard_recovered += 1

            verify_limit = max(0, int(cfg.kg_preservation_verify_existing_max_replacements))
            while (
                kg_guard_verifier_active
                and bool(cfg.kg_preservation_verify_existing)
                and kg_guard_verified_replaced < verify_limit
            ):
                kg_slots = [
                    pos
                    for pos in range(start_slot, topk)
                    if bucket_fn(doc_ids[current_top[pos]]) == "kg"
                ]
                if not kg_slots:
                    break
                slot = min(
                    kg_slots,
                    key=lambda pos: (
                        float(kg_verifier_scores.get(current_top[pos], 0.0)),
                        static_score(current_top[pos]),
                    ),
                )
                old = current_top[slot]
                old_score = kg_guard_score(old)
                old_verifier = float(kg_verifier_scores.get(old, 0.0))
                best: tuple[float, int] | None = None
                for j in guard_pool:
                    if j in current_selected:
                        continue
                    cand_verifier = float(kg_verifier_scores.get(j, 0.0))
                    if float(cfg.kg_preservation_verifier_min_score) > 0.0 and cand_verifier < float(
                        cfg.kg_preservation_verifier_min_score
                    ):
                        kg_guard_verifier_rejected += 1
                        continue
                    if cand_verifier + 1e-9 < old_verifier + 0.02:
                        kg_guard_verifier_rejected += 1
                        continue
                    score = kg_guard_score(j)
                    if score + 1e-9 < old_score + float(cfg.kg_preservation_margin):
                        kg_guard_rejected += 1
                        continue
                    item = (score, j)
                    if best is None or item[0] > best[0]:
                        best = item
                if best is None:
                    break
                _, chosen = best
                current_selected.discard(old)
                current_top[slot] = chosen
                current_selected.add(chosen)
                kg_guard_verified_replaced += 1

            if kg_guard_recovered > 0 or kg_guard_verified_replaced > 0:
                rebuilt = list(current_top)
                rebuilt_set = set(rebuilt)
                for j in out:
                    if j not in rebuilt_set:
                        rebuilt.append(j)
                        rebuilt_set.add(j)
                out = rebuilt[: len(ranked)]
                after_top = out[:topk]
                after_counts = _source_counts(after_top, doc_ids, bucket_fn)
                changed = sum(1 for a, b in zip(before_top, after_top) if a != b)
                rescued = sum(1 for j in after_top if j not in set(before_top))
                selected_scores = [static_score(j) for j in after_top]

    overlap = len(set(before_top) & set(after_top)) / max(1, topk)
    dense_selected = 0
    if dense_guard_active:
        dense_selected = sum(1 for j in after_top if j in dense_pos and j not in set(before_top))

    return out, SourceEvidenceFusionTrace(
        attempted=1,
        changed_count=int(changed),
        candidate_count=len(pool),
        preserved_count=preserve_top,
        rescued_from_below_topk=int(rescued),
        source_count_before=sum(1 for v in before_counts.values() if v > 0),
        source_count_after=sum(1 for v in after_counts.values() if v > 0),
        text_count_before=before_counts["text"],
        table_count_before=before_counts["table"],
        kg_count_before=before_counts["kg"],
        text_count_after=after_counts["text"],
        table_count_after=after_counts["table"],
        kg_count_after=after_counts["kg"],
        topk_overlap=float(overlap),
        mean_selected_score=float(np.mean(selected_scores)) if selected_scores else 0.0,
        dense_guard_active=int(dense_guard_active),
        source_balance_active=int(source_balance_active),
        dense_guard_candidates=len(set(dense_guard_ranked)),
        dense_guard_selected=int(dense_selected),
        max_changed_slots=int(cfg.max_changed_slots),
        slot_acceptance_active=int(slot_acceptance_active),
        slot_acceptance_rejected=int(slot_acceptance_rejected),
        slot_acceptance_selected=int(slot_acceptance_selected),
        budget_composer_active=int(budget_active),
        budget_candidate_count=int(budget_candidate_count),
        budget_changed_count=int(budget_changed_count),
        budget_tail_selected=int(budget_tail_selected),
        budget_sibling_selected=int(budget_sibling_selected),
        budget_source_quota_selected=int(budget_source_quota_selected),
        budget_reference_selected=int(budget_reference_selected),
        sibling_filler_active=int(sibling_filler_active),
        sibling_filler_candidate_count=int(sibling_filler_candidate_count),
        sibling_filler_changed_count=int(sibling_filler_changed_count),
        sibling_filler_tail_selected=int(sibling_filler_tail_selected),
        sibling_filler_sibling_selected=int(sibling_filler_sibling_selected),
        sibling_filler_reference_selected=int(sibling_filler_reference_selected),
        sibling_filler_dense_selected=int(sibling_filler_dense_selected),
        sibling_filler_rejected=int(sibling_filler_rejected),
        slot_verifier_active=int(slot_verifier_active),
        slot_verifier_candidate_count=int(slot_verifier_candidate_count),
        slot_verifier_changed_count=int(slot_verifier_changed_count),
        slot_verifier_accepted=int(slot_verifier_accepted),
        slot_verifier_rejected=int(slot_verifier_rejected),
        slot_verifier_tail_selected=int(slot_verifier_tail_selected),
        slot_verifier_sibling_selected=int(slot_verifier_sibling_selected),
        slot_verifier_reference_selected=int(slot_verifier_reference_selected),
        slot_verifier_dense_selected=int(slot_verifier_dense_selected),
        slot_verifier_source_selected=int(slot_verifier_source_selected),
        slot_verifier_model_active=int(slot_verifier_active and slot_verifier_bundle is not None and slot_verifier_model_allowed),
        kg_guard_active=int(kg_guard_active),
        kg_guard_candidate_count=int(kg_guard_candidate_count),
        kg_guard_recovered=int(kg_guard_recovered),
        kg_guard_rejected=int(kg_guard_rejected),
        kg_guard_verifier_active=int(kg_guard_verifier_active),
        kg_guard_verifier_mean_score=float(kg_guard_verifier_mean_score),
        kg_guard_verifier_rejected=int(kg_guard_verifier_rejected),
        kg_guard_verified_replaced=int(kg_guard_verified_replaced),
        source_budget_gate_active=int(source_budget_gate_active),
        kg_guard_effective_min_kg=int(kg_guard_effective_min_kg),
    )
