from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
import re
from typing import Callable, Sequence


BUCKETS = ("text", "table", "kg")
TOKEN_RE = re.compile(r"[A-Za-z0-9]+")
CHUNK_ID_RE = re.compile(r"^(.+)_([0-9]+)$")
YEAR_RE = re.compile(r"\b(?:1[5-9]\d{2}|20\d{2}|2100)\b")
NUMBER_RE = re.compile(r"\b\d{1,3}(?:,\d{3})*(?:\.\d+)?%?\b")
STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "by", "for", "from", "has", "have",
    "in", "is", "it", "its", "of", "on", "or", "that", "the", "to", "was", "were",
    "what", "which", "who", "whom", "whose", "when", "where", "why", "how", "many",
    "much", "did", "does", "do", "with", "than", "then", "there", "this", "these",
    "those", "into", "about", "after", "before", "between", "both", "also",
}
NO_EVIDENCE_RE = re.compile(
    r"\b("
    r"no evidence|no specific|not specified|not provided|does not provide|"
    r"doesn't provide|insufficient|cannot determine|can't determine|"
    r"not enough information|no information|no evidence found"
    r")\b",
    re.I,
)


@dataclass(frozen=True)
class QueryEvidenceProfile:
    target_type: str
    upo_concept: str
    router_entropy: float
    desired_modalities: dict[str, float]
    force_modalities: tuple[str, ...] = ()
    multi_hop: bool = False


@dataclass(frozen=True)
class UnifusionPolicyConfig:
    candidate_pool_k: int = 80
    dense_pool_k: int = 40
    rank_weight: float = 0.36
    lexical_weight: float = 0.24
    modality_weight: float = 0.18
    target_weight: float = 0.18
    coverage_weight: float = 0.16
    diversity_weight: float = 0.08
    base_keep_bonus: float = 0.03
    missing_modality_bonus: float = 0.08
    coverage_threshold: float = 0.35


@dataclass(frozen=True)
class UnifusionRetrievalAgentConfig:
    candidate_pool_k: int = 120
    dense_pool_k: int = 80
    sparse_pool_k: int = 80
    preserve_top: int = 1
    base_weight: float = 0.50
    dense_weight: float = 0.16
    sparse_weight: float = 0.10
    lexical_weight: float = 0.08
    modality_weight: float = 0.06
    target_weight: float = 0.10
    coverage_weight: float = 0.06
    diversity_weight: float = 0.04
    coverage_threshold: float = 0.30
    dense_rescue_k: int = 0
    dense_rescue_pool_k: int = 12
    sibling_seed_k: int = 0
    sibling_window: int = 0
    sibling_weight: float = 0.0


@dataclass(frozen=True)
class UnifusionRetryAgentConfig:
    candidate_pool_k: int = 120
    dense_pool_k: int = 120
    rank_weight: float = 0.30
    dense_weight: float = 0.18
    lexical_weight: float = 0.20
    answer_weight: float = 0.06
    modality_weight: float = 0.12
    target_weight: float = 0.18
    coverage_weight: float = 0.12
    diversity_weight: float = 0.04
    missing_modality_bonus: float = 0.10
    table_number_bonus: float = 0.12
    coverage_threshold: float = 0.30


@dataclass(frozen=True)
class UnifusionMoERetrievalConfig:
    candidate_pool_k: int = 260
    preserve_top: int = 1
    prf_seed_k: int = 6
    prf_dense_seed_k: int = 6
    prf_sparse_seed_k: int = 6
    prf_max_terms: int = 48
    sibling_seed_k: int = 6
    sibling_window: int = 1
    sibling_weight: float = 0.03
    base_weight: float = 0.42
    dense_weight: float = 0.13
    sparse_weight: float = 0.10
    lexical_weight: float = 0.10
    prf_weight: float = 0.12
    modality_weight: float = 0.08
    target_weight: float = 0.12
    coverage_weight: float = 0.08
    diversity_weight: float = 0.05
    table_dense_scale: float = 0.35
    table_target_boost: float = 0.08
    kg_target_boost: float = 0.05
    uncertainty_coverage_boost: float = 0.05
    coverage_threshold: float = 0.30


@dataclass(frozen=True)
class UnifusionTableNumberAgentConfig:
    min_score: float = 0.34
    max_units_per_context: int = 80
    query_overlap_weight: float = 0.52
    table_weight: float = 0.18
    rank_weight: float = 0.08
    target_weight: float = 0.12
    header_weight: float = 0.06
    query_literal_penalty: float = 0.22


@dataclass
class PolicyTrace:
    profile: QueryEvidenceProfile
    selected_buckets: list[str] = field(default_factory=list)
    forced_hits: int = 0
    replaced_count: int = 0
    coverage: float = 0.0


@dataclass
class RetrievalAgentTrace:
    profile: QueryEvidenceProfile
    selected_buckets: list[str] = field(default_factory=list)
    preserve_count: int = 0
    dense_added: int = 0
    sparse_added: int = 0
    dense_rescue_added: int = 0
    sibling_added: int = 0
    forced_hits: int = 0
    coverage: float = 0.0


@dataclass
class RetryAgentTrace:
    profile: QueryEvidenceProfile
    selected_buckets: list[str] = field(default_factory=list)
    forced_hits: int = 0
    coverage: float = 0.0
    no_evidence_answer: bool = False


@dataclass
class MoERetrievalTrace:
    profile: QueryEvidenceProfile
    selected_buckets: list[str] = field(default_factory=list)
    table_like: bool = False
    kg_like: bool = False
    prf_terms: int = 0
    sibling_added: int = 0
    coverage: float = 0.0


@dataclass
class TableNumberAgentTrace:
    target_type: str = ""
    attempted: bool = False
    accepted: bool = False
    candidate_count: int = 0
    best_score: float = 0.0
    best_bucket: str = ""
    raw_no_evidence: bool = False
    raw_has_literal: bool = False
    forced_by_low_support: bool = False


def normalize3(values: Sequence[float]) -> dict[str, float]:
    vals = [max(1e-6, float(v)) for v in list(values)[:3]]
    while len(vals) < 3:
        vals.append(1.0)
    total = sum(vals) or 1.0
    return {b: float(v / total) for b, v in zip(BUCKETS, vals)}


def tokenize_list(text: str) -> list[str]:
    return [m.group(0).lower() for m in TOKEN_RE.finditer(str(text))]


def sibling_chunk_indices(
    doc_id: str,
    doc_id_to_idx: dict[str, int] | None,
    window: int,
) -> list[int]:
    if not doc_id_to_idx or int(window) <= 0:
        return []
    m = CHUNK_ID_RE.match(str(doc_id))
    if not m:
        return []
    prefix = m.group(1)
    pos = int(m.group(2))
    out: list[int] = []
    for delta in range(-int(window), int(window) + 1):
        if delta == 0:
            continue
        cand = f"{prefix}_{pos + delta}"
        if cand in doc_id_to_idx:
            out.append(int(doc_id_to_idx[cand]))
    return out


def query_content_tokens(text: str) -> set[str]:
    return {t for t in tokenize_list(text) if len(t) > 1 and t not in STOPWORDS}


def build_prf_token_weights(
    *,
    query: str,
    ranked_idxs: list[int],
    dense_ranked_idxs: list[int],
    sparse_ranked_idxs: list[int],
    doc_tokens: list[set[str]],
    config: UnifusionMoERetrievalConfig,
) -> dict[str, float]:
    q_tokens = query_content_tokens(query)
    counts: Counter[str] = Counter()

    def add_seed(seq: list[int], limit: int, source_weight: float) -> None:
        for pos, j in enumerate(seq[: max(0, int(limit))]):
            if j < 0 or j >= len(doc_tokens):
                continue
            rank_weight = float(source_weight) / float(pos + 1)
            for tok in doc_tokens[j]:
                if len(tok) <= 2 or tok in STOPWORDS:
                    continue
                if tok in q_tokens:
                    counts[tok] += 2.0 * rank_weight
                else:
                    counts[tok] += 0.45 * rank_weight

    add_seed(ranked_idxs, int(config.prf_seed_k), 1.00)
    add_seed(dense_ranked_idxs, int(config.prf_dense_seed_k), 0.65)
    add_seed(sparse_ranked_idxs, int(config.prf_sparse_seed_k), 0.55)
    if not counts:
        return {}
    top = counts.most_common(max(1, int(config.prf_max_terms)))
    max_val = float(top[0][1]) if top else 1.0
    return {tok: float(val / max(1e-9, max_val)) for tok, val in top}


def is_no_evidence_answer(answer: str) -> bool:
    raw = str(answer or "").strip()
    if not raw:
        return True
    return NO_EVIDENCE_RE.search(raw) is not None


def lite_normalize(text: str) -> str:
    toks = tokenize_list(text)
    return " ".join(toks)


def answer_has_target_literal(answer: str, target_type: str) -> bool:
    target = str(target_type or "").lower()
    raw = str(answer or "")
    if target == "year":
        return YEAR_RE.search(raw) is not None
    if target == "number":
        norm = lite_normalize(raw)
        number_words = {
            "none", "no", "zero", "one", "two", "three", "four", "five", "six",
            "seven", "eight", "nine", "ten", "eleven", "twelve", "dozen",
        }
        if any(tok in number_words for tok in norm.split()):
            return True
        return NUMBER_RE.search(raw) is not None
    return bool(raw.strip())


def evidence_units(text: str, max_units: int) -> list[str]:
    raw = str(text or "")
    units: list[str] = []
    seen: set[str] = set()

    def add(unit: str) -> None:
        u = re.sub(r"\s+", " ", str(unit or "").strip())
        if len(u) < 3:
            return
        key = u.lower()
        if key in seen:
            return
        seen.add(key)
        units.append(u)

    for line in raw.splitlines():
        line = line.strip()
        if not line:
            continue
        add(line)
        if "|" in line or "\t" in line:
            for cell in re.split(r"\s*\|\s*|\t+", line):
                add(cell)
    for sent in re.split(r"[\.\!\?;]+", raw):
        add(sent)
        if len(units) >= max_units:
            break
    return units[: max(1, int(max_units))]


def table_like_score(text: str, bucket: str) -> float:
    raw = str(text or "")
    score = 0.0
    if str(bucket) == "table":
        score += 0.75
    if "|" in raw or "\t" in raw:
        score += 0.35
    if raw.count(":") >= 2 or raw.count(",") >= 4:
        score += 0.12
    if len(NUMBER_RE.findall(raw)) >= 2:
        score += 0.15
    return float(min(1.0, score))


def select_unifusion_table_number_answer(
    *,
    query: str,
    current_answer: str,
    contexts: list[str],
    doc_ids: list[str] | None,
    target_type: str,
    source_bucket_fn: Callable[[str], str] | None = None,
    force_trigger: bool = False,
    config: UnifusionTableNumberAgentConfig | None = None,
) -> tuple[str, TableNumberAgentTrace]:
    """Extract a table/number-backed short answer for weak UniFusion reader outputs."""

    cfg = config or UnifusionTableNumberAgentConfig()
    target = str(target_type or "").strip().lower()
    trace = TableNumberAgentTrace(target_type=target)
    if target not in {"number", "year"}:
        return current_answer, trace
    trace.attempted = True
    trace.raw_no_evidence = is_no_evidence_answer(current_answer)
    trace.raw_has_literal = answer_has_target_literal(current_answer, target)
    trace.forced_by_low_support = bool(force_trigger)

    q_tokens = query_content_tokens(query)
    query_literals = {lite_normalize(m.group(0)) for m in NUMBER_RE.finditer(str(query))}
    query_years = {lite_normalize(m.group(0)) for m in YEAR_RE.finditer(str(query))}
    literal_re = YEAR_RE if target == "year" else NUMBER_RE
    candidates: list[tuple[str, float, str]] = []

    for ctx_pos, ctx in enumerate(contexts):
        doc_id = ""
        if doc_ids and ctx_pos < len(doc_ids):
            doc_id = str(doc_ids[ctx_pos])
        bucket = source_bucket_fn(doc_id) if source_bucket_fn is not None and doc_id else ""
        units = evidence_units(ctx, int(cfg.max_units_per_context))
        for unit in units:
            matches = list(literal_re.finditer(unit))
            if not matches:
                continue
            unit_tokens = query_content_tokens(unit)
            overlap = len(q_tokens & unit_tokens) / max(1, len(q_tokens)) if q_tokens else 0.0
            table_score = table_like_score(unit, bucket)
            header_hit = 1.0 if len(q_tokens & unit_tokens) >= 2 else 0.0
            rank_signal = 1.0 / float(ctx_pos + 1)
            for match in matches:
                literal = match.group(0).strip()
                literal_norm = lite_normalize(literal)
                if not literal_norm:
                    continue
                literal_in_query = literal_norm in query_literals
                if target == "year":
                    literal_in_query = literal_in_query or literal_norm in query_years
                score = (
                    float(cfg.query_overlap_weight) * overlap
                    + float(cfg.table_weight) * table_score
                    + float(cfg.rank_weight) * rank_signal
                    + float(cfg.target_weight)
                    + float(cfg.header_weight) * header_hit
                )
                if literal_in_query:
                    score -= float(cfg.query_literal_penalty)
                if target == "number" and YEAR_RE.fullmatch(literal):
                    # Years are often constraints in numeric questions, not the numeric answer.
                    score -= 0.10
                candidates.append((literal, float(score), str(bucket)))

    trace.candidate_count = len(candidates)
    if not candidates:
        return current_answer, trace
    candidates.sort(key=lambda item: (item[1], len(item[0])), reverse=True)
    best_literal, best_score, best_bucket = candidates[0]
    trace.best_score = float(best_score)
    trace.best_bucket = best_bucket

    trigger = trace.raw_no_evidence or bool(force_trigger) or not trace.raw_has_literal
    if not trigger:
        return current_answer, trace
    if best_score < float(cfg.min_score):
        return current_answer, trace
    if lite_normalize(best_literal) == lite_normalize(current_answer):
        return current_answer, trace

    trace.accepted = True
    return best_literal, trace


def decompose_segments(query: str) -> list[set[str]]:
    raw = str(query)
    parts = re.split(r"\b(?:and|or|versus|vs|between|compared with|as well as)\b|[,;:]", raw, flags=re.I)
    segments: list[set[str]] = []
    for part in parts:
        toks = {t for t in tokenize_list(part) if len(t) > 1}
        if toks:
            segments.append(toks)
    if not segments:
        toks = {t for t in tokenize_list(raw) if len(t) > 1}
        if toks:
            segments.append(toks)
    return segments


def jaccard(a: set[str], b: set[str]) -> float:
    if not a or not b:
        return 0.0
    return float(len(a & b) / max(1, len(a | b)))


def build_query_evidence_profile(
    query: str,
    router_prob: Sequence[float],
    router_entropy: float,
    target_type: str,
    upo_concept: str,
) -> QueryEvidenceProfile:
    desired = normalize3(router_prob)
    target = str(target_type or "open").lower()
    upo = str(upo_concept or "open").lower()
    q = str(query).lower()
    multi_hop = bool(
        re.search(r"\b(compare|between|both|and|versus|vs|before|after|difference|total|sum|average)\b", q)
    )

    def add(bucket: str, amount: float) -> None:
        desired[bucket] = desired.get(bucket, 0.0) + float(amount)

    forced: list[str] = []
    if target in {"number", "year"}:
        add("table", 0.20)
        add("text", 0.08)
        if target == "number":
            forced.append("table")
    elif target in {"person", "entity", "location"}:
        add("kg", 0.13)
        add("text", 0.08)
        if upo == "relation":
            forced.append("kg")
    elif target == "boolean":
        add("text", 0.14)

    if upo == "number":
        add("table", 0.15)
    elif upo == "relation":
        add("kg", 0.15)
    elif upo in {"entity", "open"}:
        add("text", 0.05)

    if multi_hop:
        top2 = sorted(BUCKETS, key=lambda b: desired.get(b, 0.0), reverse=True)[:2]
        for b in top2:
            if b not in forced:
                forced.append(b)

    if float(router_entropy) >= 0.70:
        # High uncertainty: avoid letting one router class monopolize context.
        for b in BUCKETS:
            desired[b] = 0.75 * desired.get(b, 0.0) + 0.25 / 3.0

    norm = normalize3([desired["text"], desired["table"], desired["kg"]])
    return QueryEvidenceProfile(
        target_type=target,
        upo_concept=upo,
        router_entropy=float(router_entropy),
        desired_modalities=norm,
        force_modalities=tuple(dict.fromkeys(forced)),
        multi_hop=multi_hop,
    )


def target_support_score(
    profile: QueryEvidenceProfile,
    bucket: str,
    doc_text: str,
    doc_tokens: set[str],
    numeric_literals: set[str],
) -> float:
    target = profile.target_type
    text = str(doc_text)
    score = 0.0
    has_num = bool(numeric_literals) or NUMBER_RE.search(text) is not None
    has_year = YEAR_RE.search(text) is not None

    if target == "number":
        if has_num:
            score += 0.55
        if bucket == "table":
            score += 0.35
    elif target == "year":
        if has_year:
            score += 0.55
        if bucket in {"table", "text"}:
            score += 0.20
    elif target in {"person", "entity", "location"}:
        if bucket == "kg":
            score += 0.35
        if bucket == "text":
            score += 0.15
        if len(doc_tokens) >= 8:
            score += 0.05
    elif target == "boolean":
        if bucket == "text":
            score += 0.30

    if profile.upo_concept == "relation" and bucket == "kg":
        score += 0.20
    return float(min(1.0, score))


def coverage_gain(doc_tokens: set[str], segments: list[set[str]], covered: set[int], threshold: float) -> tuple[float, set[int]]:
    if not segments:
        return 0.0, set()
    newly: set[int] = set()
    for idx, seg in enumerate(segments):
        if idx in covered:
            continue
        overlap = len(seg & doc_tokens) / max(1, len(seg))
        if overlap >= threshold:
            newly.add(idx)
    return float(len(newly) / max(1, len(segments))), newly


def select_unifusion_policy_context(
    *,
    query: str,
    base_ctx_idxs: list[int],
    ranked_idxs: list[int],
    dense_ranked_idxs: list[int],
    doc_ids: list[str],
    doc_texts: list[str],
    doc_tokens: list[set[str]],
    doc_numeric_literals: list[set[str]],
    router_prob: Sequence[float],
    router_entropy: float,
    k: int,
    target_type: str,
    upo_concept: str,
    source_bucket_fn: Callable[[str], str],
    config: UnifusionPolicyConfig | None = None,
) -> tuple[list[int], PolicyTrace]:
    cfg = config or UnifusionPolicyConfig()
    topk = max(1, int(k))
    profile = build_query_evidence_profile(
        query=query,
        router_prob=router_prob,
        router_entropy=router_entropy,
        target_type=target_type,
        upo_concept=upo_concept,
    )

    ranked = [int(j) for j in ranked_idxs if 0 <= int(j) < len(doc_ids)]
    dense = [int(j) for j in dense_ranked_idxs if 0 <= int(j) < len(doc_ids)]
    base = [int(j) for j in base_ctx_idxs if 0 <= int(j) < len(doc_ids)]

    candidates: list[int] = []
    seen: set[int] = set()

    def push(seq: list[int], limit: int) -> None:
        for j in seq[: max(0, int(limit))]:
            if j in seen:
                continue
            seen.add(j)
            candidates.append(j)

    push(base, len(base))
    push(ranked, max(topk * 4, int(cfg.candidate_pool_k)))
    push(dense, max(topk * 2, int(cfg.dense_pool_k)))
    if not candidates:
        return ranked[:topk], PolicyTrace(profile=profile)

    rank_pos = {j: p for p, j in enumerate(ranked)}
    dense_pos = {j: p for p, j in enumerate(dense)}
    base_set = set(base)
    q_tokens = {t for t in tokenize_list(query) if len(t) > 1}
    segments = decompose_segments(query)

    selected: list[int] = []
    selected_set: set[int] = set()
    covered_segments: set[int] = set()

    def score_doc(j: int, missing_forced: set[str]) -> float:
        bucket = source_bucket_fn(doc_ids[j])
        uni_signal = 1.0 / float(rank_pos[j] + 1) if j in rank_pos else 0.0
        dense_signal = 1.0 / float(dense_pos[j] + 1) if j in dense_pos else 0.0
        rank_signal = 0.64 * uni_signal + 0.36 * dense_signal
        lexical = len(q_tokens & doc_tokens[j]) / max(1, len(q_tokens)) if q_tokens else 0.0
        cov, _ = coverage_gain(doc_tokens[j], segments, covered_segments, float(cfg.coverage_threshold))
        support = target_support_score(profile, bucket, doc_texts[j], doc_tokens[j], doc_numeric_literals[j])
        modality = profile.desired_modalities.get(bucket, 0.0)
        missing_bonus = float(cfg.missing_modality_bonus) if bucket in missing_forced else 0.0
        base_bonus = float(cfg.base_keep_bonus) if j in base_set else 0.0
        diversity_pen = 0.0
        if selected:
            diversity_pen = max(jaccard(doc_tokens[j], doc_tokens[x]) for x in selected)
        return float(
            cfg.rank_weight * rank_signal
            + cfg.lexical_weight * lexical
            + cfg.modality_weight * modality
            + cfg.target_weight * support
            + cfg.coverage_weight * cov
            + missing_bonus
            + base_bonus
            - cfg.diversity_weight * diversity_pen
        )

    def add_doc(j: int) -> bool:
        if j in selected_set:
            return False
        selected.append(j)
        selected_set.add(j)
        _, newly = coverage_gain(doc_tokens[j], segments, covered_segments, float(cfg.coverage_threshold))
        covered_segments.update(newly)
        return True

    forced = set(profile.force_modalities)
    for bucket in sorted(forced, key=lambda b: profile.desired_modalities.get(b, 0.0), reverse=True):
        if len(selected) >= topk:
            break
        pool = [j for j in candidates if source_bucket_fn(doc_ids[j]) == bucket]
        if not pool:
            continue
        best = max(pool, key=lambda j: score_doc(j, forced))
        if add_doc(best):
            forced.discard(bucket)

    while len(selected) < topk:
        remaining = [j for j in candidates if j not in selected_set]
        if not remaining:
            break
        selected_buckets = {source_bucket_fn(doc_ids[j]) for j in selected}
        missing_forced = set(profile.force_modalities) - selected_buckets
        best = max(remaining, key=lambda j: score_doc(j, missing_forced))
        add_doc(best)

    if len(selected) < topk:
        for j in base + ranked + dense:
            if len(selected) >= topk:
                break
            add_doc(j)

    buckets = [source_bucket_fn(doc_ids[j]) for j in selected[:topk]]
    trace = PolicyTrace(
        profile=profile,
        selected_buckets=buckets,
        forced_hits=len(set(profile.force_modalities) & set(buckets)),
        replaced_count=len([j for j in selected[:topk] if j not in base_set]),
        coverage=float(len(covered_segments) / max(1, len(segments))) if segments else 0.0,
    )
    return selected[:topk], trace


def select_unifusion_retry_context(
    *,
    query: str,
    current_answer: str,
    base_ctx_idxs: list[int],
    ranked_idxs: list[int],
    dense_ranked_idxs: list[int],
    doc_ids: list[str],
    doc_texts: list[str],
    doc_tokens: list[set[str]],
    doc_numeric_literals: list[set[str]],
    router_prob: Sequence[float],
    router_entropy: float,
    k: int,
    target_type: str,
    upo_concept: str,
    source_bucket_fn: Callable[[str], str],
    config: UnifusionRetryAgentConfig | None = None,
) -> tuple[list[int], RetryAgentTrace]:
    cfg = config or UnifusionRetryAgentConfig()
    topk = max(1, int(k))
    no_evidence = is_no_evidence_answer(current_answer)
    profile = build_query_evidence_profile(
        query=query,
        router_prob=router_prob,
        router_entropy=router_entropy,
        target_type=target_type,
        upo_concept=upo_concept,
    )

    ranked = [int(j) for j in ranked_idxs if 0 <= int(j) < len(doc_ids)]
    dense = [int(j) for j in dense_ranked_idxs if 0 <= int(j) < len(doc_ids)]
    base = [int(j) for j in base_ctx_idxs if 0 <= int(j) < len(doc_ids)]

    candidates: list[int] = []
    seen: set[int] = set()

    def push(seq: Sequence[int], limit: int) -> None:
        for j in list(seq)[: max(0, int(limit))]:
            jj = int(j)
            if jj in seen:
                continue
            if jj < 0 or jj >= len(doc_ids):
                continue
            seen.add(jj)
            candidates.append(jj)

    push(base, len(base))
    push(ranked, max(topk * 6, int(cfg.candidate_pool_k)))
    push(dense, max(topk * 6, int(cfg.dense_pool_k)))
    if not candidates:
        return ranked[:topk], RetryAgentTrace(profile=profile, no_evidence_answer=no_evidence)

    rank_pos = {j: p for p, j in enumerate(ranked)}
    dense_pos = {j: p for p, j in enumerate(dense)}
    base_set = set(base)
    q_tokens = {t for t in tokenize_list(query) if len(t) > 1}
    answer_tokens = set()
    if not no_evidence:
        answer_tokens = {t for t in tokenize_list(current_answer) if len(t) > 1}
    segments = decompose_segments(query)

    selected: list[int] = []
    selected_set: set[int] = set()
    covered_segments: set[int] = set()

    def add_doc(j: int) -> bool:
        if j in selected_set:
            return False
        selected.append(j)
        selected_set.add(j)
        _, newly = coverage_gain(doc_tokens[j], segments, covered_segments, float(cfg.coverage_threshold))
        covered_segments.update(newly)
        return True

    def rank_signal(pos_map: dict[int, int], j: int) -> float:
        if j not in pos_map:
            return 0.0
        return 1.0 / float(pos_map[j] + 1)

    def score_doc(j: int, missing_forced: set[str]) -> float:
        bucket = source_bucket_fn(doc_ids[j])
        q_overlap = len(q_tokens & doc_tokens[j]) / max(1, len(q_tokens)) if q_tokens else 0.0
        a_overlap = len(answer_tokens & doc_tokens[j]) / max(1, len(answer_tokens)) if answer_tokens else 0.0
        cov, _ = coverage_gain(doc_tokens[j], segments, covered_segments, float(cfg.coverage_threshold))
        support = target_support_score(profile, bucket, doc_texts[j], doc_tokens[j], doc_numeric_literals[j])
        modality = profile.desired_modalities.get(bucket, 0.0)
        forced_bonus = float(cfg.missing_modality_bonus) if bucket in missing_forced else 0.0
        table_bonus = 0.0
        if profile.target_type in {"number", "year"} and bucket == "table":
            table_bonus = float(cfg.table_number_bonus)
        base_bonus = 0.02 if j in base_set else 0.0
        diversity_pen = 0.0
        if selected:
            diversity_pen = max(jaccard(doc_tokens[j], doc_tokens[x]) for x in selected)
        return float(
            cfg.rank_weight * rank_signal(rank_pos, j)
            + cfg.dense_weight * rank_signal(dense_pos, j)
            + cfg.lexical_weight * q_overlap
            + cfg.answer_weight * a_overlap
            + cfg.modality_weight * modality
            + cfg.target_weight * support
            + cfg.coverage_weight * cov
            + forced_bonus
            + table_bonus
            + base_bonus
            - cfg.diversity_weight * diversity_pen
        )

    forced = set(profile.force_modalities)
    if profile.target_type in {"number", "year"}:
        forced.add("table")
    for bucket in sorted(forced, key=lambda b: profile.desired_modalities.get(b, 0.0), reverse=True):
        if len(selected) >= topk:
            break
        pool = [j for j in candidates if source_bucket_fn(doc_ids[j]) == bucket]
        if not pool:
            continue
        best = max(pool, key=lambda j: score_doc(j, forced))
        if add_doc(best):
            forced.discard(bucket)

    while len(selected) < topk:
        remaining = [j for j in candidates if j not in selected_set]
        if not remaining:
            break
        selected_buckets = {source_bucket_fn(doc_ids[j]) for j in selected}
        missing_forced = set(profile.force_modalities) - selected_buckets
        if profile.target_type in {"number", "year"} and "table" not in selected_buckets:
            missing_forced.add("table")
        best = max(remaining, key=lambda j: score_doc(j, missing_forced))
        add_doc(best)

    if len(selected) < topk:
        for j in base + ranked + dense:
            if len(selected) >= topk:
                break
            add_doc(j)

    out = selected[:topk]
    buckets = [source_bucket_fn(doc_ids[j]) for j in out]
    trace = RetryAgentTrace(
        profile=profile,
        selected_buckets=buckets,
        forced_hits=len(set(profile.force_modalities) & set(buckets)),
        coverage=float(len(covered_segments) / max(1, len(segments))) if segments else 0.0,
        no_evidence_answer=no_evidence,
    )
    return out, trace


def rerank_unifusion_retrieval(
    *,
    query: str,
    current_ranked_idxs: list[int],
    candidate_idxs: Sequence[int],
    candidate_scores: Sequence[float],
    dense_ranked_idxs: list[int],
    sparse_ranked_idxs: list[int],
    doc_ids: list[str],
    doc_id_to_idx: dict[str, int] | None,
    doc_texts: list[str],
    doc_tokens: list[set[str]],
    doc_numeric_literals: list[set[str]],
    router_prob: Sequence[float],
    router_entropy: float,
    k: int,
    target_type: str,
    upo_concept: str,
    source_bucket_fn: Callable[[str], str],
    config: UnifusionRetrievalAgentConfig | None = None,
) -> tuple[list[int], RetrievalAgentTrace]:
    """Multi-agent UniFusion retrieval reranker.

    The agents are lightweight scoring views rather than LLM calls: the base
    agent preserves UniFusion rank evidence, dense/sparse agents add recall,
    and target/coverage/diversity agents improve the top-k evidence set.
    """
    cfg = config or UnifusionRetrievalAgentConfig()
    topk = max(1, int(k))
    profile = build_query_evidence_profile(
        query=query,
        router_prob=router_prob,
        router_entropy=router_entropy,
        target_type=target_type,
        upo_concept=upo_concept,
    )

    ranked = [int(j) for j in current_ranked_idxs if 0 <= int(j) < len(doc_ids)]
    dense = [int(j) for j in dense_ranked_idxs if 0 <= int(j) < len(doc_ids)]
    sparse = [int(j) for j in sparse_ranked_idxs if 0 <= int(j) < len(doc_ids)]

    raw_score: dict[int, float] = {}
    for j, sc in zip(candidate_idxs, candidate_scores):
        jj = int(j)
        if 0 <= jj < len(doc_ids):
            raw_score[jj] = max(raw_score.get(jj, float("-inf")), float(sc))

    if raw_score:
        vals = list(raw_score.values())
        lo = min(vals)
        hi = max(vals)
        span = max(1e-9, hi - lo)
        base_score = {j: (sc - lo) / span for j, sc in raw_score.items()}
        base_ranked = sorted(raw_score, key=lambda j: raw_score[j], reverse=True)
    else:
        base_score = {}
        base_ranked = []

    candidates: list[int] = []
    seen: set[int] = set()

    def push(seq: Sequence[int], limit: int) -> None:
        for j in list(seq)[: max(0, int(limit))]:
            jj = int(j)
            if jj in seen or not (0 <= jj < len(doc_ids)):
                continue
            seen.add(jj)
            candidates.append(jj)

    push(ranked, max(topk * 4, int(cfg.candidate_pool_k) // 3))
    push(base_ranked, max(topk * 6, int(cfg.candidate_pool_k)))
    push(dense, max(topk * 4, int(cfg.dense_pool_k)))
    push(sparse, max(topk * 4, int(cfg.sparse_pool_k)))
    sibling_set: set[int] = set()
    if int(cfg.sibling_window) > 0 and int(cfg.sibling_seed_k) > 0:
        seeds = ranked[: int(cfg.sibling_seed_k)] + dense[: int(cfg.sibling_seed_k)]
        sibling_candidates: list[int] = []
        for seed in seeds:
            for sib in sibling_chunk_indices(doc_ids[int(seed)], doc_id_to_idx, int(cfg.sibling_window)):
                sibling_set.add(int(sib))
                sibling_candidates.append(int(sib))
        push(sibling_candidates, max(topk * 4, int(cfg.sibling_seed_k) * max(1, int(cfg.sibling_window)) * 2))
    if not candidates:
        return ranked[:topk], RetrievalAgentTrace(profile=profile)

    rank_pos = {j: p for p, j in enumerate(ranked)}
    dense_pos = {j: p for p, j in enumerate(dense)}
    sparse_pos = {j: p for p, j in enumerate(sparse)}
    q_tokens = {t for t in tokenize_list(query) if len(t) > 1}
    segments = decompose_segments(query)

    selected: list[int] = []
    selected_set: set[int] = set()
    covered_segments: set[int] = set()

    def add_doc(j: int) -> bool:
        if j in selected_set:
            return False
        selected.append(j)
        selected_set.add(j)
        _, newly = coverage_gain(doc_tokens[j], segments, covered_segments, float(cfg.coverage_threshold))
        covered_segments.update(newly)
        return True

    preserve_n = min(max(0, int(cfg.preserve_top)), topk)
    for j in ranked[:preserve_n]:
        add_doc(j)

    dense_rescue_added = 0
    rescue_limit = min(max(0, int(cfg.dense_rescue_k)), max(0, topk - len(selected)))
    if rescue_limit > 0:
        rescue_pool = dense[: max(rescue_limit, int(cfg.dense_rescue_pool_k))]
        for j in rescue_pool:
            if dense_rescue_added >= rescue_limit or len(selected) >= topk:
                break
            if add_doc(j):
                dense_rescue_added += 1

    def rank_signal(pos_map: dict[int, int], j: int) -> float:
        if j not in pos_map:
            return 0.0
        return 1.0 / float(pos_map[j] + 1)

    def score_doc(j: int, missing_forced: set[str]) -> float:
        bucket = source_bucket_fn(doc_ids[j])
        base_signal = base_score.get(j, rank_signal(rank_pos, j))
        dense_signal = rank_signal(dense_pos, j)
        sparse_signal = rank_signal(sparse_pos, j)
        lexical = len(q_tokens & doc_tokens[j]) / max(1, len(q_tokens)) if q_tokens else 0.0
        cov, _ = coverage_gain(doc_tokens[j], segments, covered_segments, float(cfg.coverage_threshold))
        support = target_support_score(profile, bucket, doc_texts[j], doc_tokens[j], doc_numeric_literals[j])
        modality = profile.desired_modalities.get(bucket, 0.0)
        forced_bonus = 0.05 if bucket in missing_forced else 0.0
        diversity_pen = 0.0
        if selected:
            diversity_pen = max(jaccard(doc_tokens[j], doc_tokens[x]) for x in selected)
        sibling_bonus = float(cfg.sibling_weight) if j in sibling_set else 0.0
        return float(
            cfg.base_weight * base_signal
            + cfg.dense_weight * dense_signal
            + cfg.sparse_weight * sparse_signal
            + cfg.lexical_weight * lexical
            + cfg.modality_weight * modality
            + cfg.target_weight * support
            + cfg.coverage_weight * cov
            + forced_bonus
            + sibling_bonus
            - cfg.diversity_weight * diversity_pen
        )

    while len(selected) < topk:
        remaining = [j for j in candidates if j not in selected_set]
        if not remaining:
            break
        selected_buckets = {source_bucket_fn(doc_ids[j]) for j in selected}
        missing_forced = set(profile.force_modalities) - selected_buckets
        best = max(remaining, key=lambda j: score_doc(j, missing_forced))
        add_doc(best)

    if len(selected) < topk:
        for j in ranked + dense + sparse + base_ranked:
            if len(selected) >= topk:
                break
            add_doc(j)

    out = selected[:topk]
    out_set = set(out)
    buckets = [source_bucket_fn(doc_ids[j]) for j in out]
    trace = RetrievalAgentTrace(
        profile=profile,
        selected_buckets=buckets,
        preserve_count=len([j for j in out[:preserve_n] if j in ranked[:preserve_n]]),
        dense_added=len([j for j in out_set if j in dense_pos and j not in rank_pos]),
        sparse_added=len([j for j in out_set if j in sparse_pos and j not in rank_pos]),
        dense_rescue_added=int(dense_rescue_added),
        sibling_added=len([j for j in out_set if j in sibling_set]),
        forced_hits=len(set(profile.force_modalities) & set(buckets)),
        coverage=float(len(covered_segments) / max(1, len(segments))) if segments else 0.0,
    )
    return out, trace


def rerank_unifusion_moe_retrieval(
    *,
    query: str,
    current_ranked_idxs: list[int],
    candidate_idxs: Sequence[int],
    candidate_scores: Sequence[float],
    dense_ranked_idxs: list[int],
    sparse_ranked_idxs: list[int],
    doc_ids: list[str],
    doc_id_to_idx: dict[str, int] | None,
    doc_texts: list[str],
    doc_tokens: list[set[str]],
    doc_numeric_literals: list[set[str]],
    router_prob: Sequence[float],
    router_entropy: float,
    k: int,
    target_type: str,
    upo_concept: str,
    source_bucket_fn: Callable[[str], str],
    config: UnifusionMoERetrievalConfig | None = None,
) -> tuple[list[int], MoERetrievalTrace]:
    cfg = config or UnifusionMoERetrievalConfig()
    topk = max(1, int(k))
    profile = build_query_evidence_profile(
        query=query,
        router_prob=router_prob,
        router_entropy=router_entropy,
        target_type=target_type,
        upo_concept=upo_concept,
    )

    ranked = [int(j) for j in current_ranked_idxs if 0 <= int(j) < len(doc_ids)]
    dense = [int(j) for j in dense_ranked_idxs if 0 <= int(j) < len(doc_ids)]
    sparse = [int(j) for j in sparse_ranked_idxs if 0 <= int(j) < len(doc_ids)]
    target = profile.target_type
    q_low = str(query).lower()
    table_like = bool(
        target in {"number", "year"}
        or profile.desired_modalities.get("table", 0.0) >= 0.38
        or re.search(r"\b(population|capacity|number|total|percent|percentage|average|rank|date|year|how many)\b", q_low)
    )
    kg_like = bool(
        profile.upo_concept == "relation"
        or profile.desired_modalities.get("kg", 0.0) >= 0.38
        or re.search(r"\b(parent|subsidiary|spouse|founder|located|owned|member of|capital of)\b", q_low)
    )

    raw_score: dict[int, float] = {}
    for j, sc in zip(candidate_idxs, candidate_scores):
        jj = int(j)
        if 0 <= jj < len(doc_ids):
            raw_score[jj] = max(raw_score.get(jj, float("-inf")), float(sc))
    if raw_score:
        vals = list(raw_score.values())
        lo = min(vals)
        hi = max(vals)
        span = max(1e-9, hi - lo)
        base_score = {j: (sc - lo) / span for j, sc in raw_score.items()}
        base_ranked = sorted(raw_score, key=lambda j: raw_score[j], reverse=True)
    else:
        base_score = {}
        base_ranked = []

    candidates: list[int] = []
    seen: set[int] = set()

    def push(seq: Sequence[int], limit: int) -> None:
        for j in list(seq)[: max(0, int(limit))]:
            jj = int(j)
            if jj in seen or not (0 <= jj < len(doc_ids)):
                continue
            seen.add(jj)
            candidates.append(jj)

    pool_k = max(topk * 8, int(cfg.candidate_pool_k))
    push(ranked, pool_k)
    push(base_ranked, pool_k)
    push(dense, pool_k // (3 if table_like else 2))
    push(sparse, pool_k // 2)

    sibling_set: set[int] = set()
    if int(cfg.sibling_window) > 0 and int(cfg.sibling_seed_k) > 0 and not table_like:
        sibling_candidates: list[int] = []
        for seed in ranked[: int(cfg.sibling_seed_k)] + dense[: int(cfg.sibling_seed_k)]:
            for sib in sibling_chunk_indices(doc_ids[int(seed)], doc_id_to_idx, int(cfg.sibling_window)):
                sibling_set.add(int(sib))
                sibling_candidates.append(int(sib))
        push(sibling_candidates, max(topk * 4, int(cfg.sibling_seed_k) * 2))

    if not candidates:
        return ranked[:topk], MoERetrievalTrace(profile=profile, table_like=table_like, kg_like=kg_like)

    prf_weights = build_prf_token_weights(
        query=query,
        ranked_idxs=ranked,
        dense_ranked_idxs=dense,
        sparse_ranked_idxs=sparse,
        doc_tokens=doc_tokens,
        config=cfg,
    )
    prf_norm = sum(prf_weights.values()) or 1.0
    rank_pos = {j: p for p, j in enumerate(ranked)}
    dense_pos = {j: p for p, j in enumerate(dense)}
    sparse_pos = {j: p for p, j in enumerate(sparse)}
    q_tokens = query_content_tokens(query)
    segments = decompose_segments(query)

    weights = {
        "base": float(cfg.base_weight),
        "dense": float(cfg.dense_weight),
        "sparse": float(cfg.sparse_weight),
        "lexical": float(cfg.lexical_weight),
        "prf": float(cfg.prf_weight),
        "modality": float(cfg.modality_weight),
        "target": float(cfg.target_weight),
        "coverage": float(cfg.coverage_weight),
        "diversity": float(cfg.diversity_weight),
    }
    sibling_weight = float(cfg.sibling_weight)
    if table_like:
        weights["dense"] *= float(cfg.table_dense_scale)
        weights["target"] += float(cfg.table_target_boost)
        weights["modality"] += 0.03
        sibling_weight = 0.0
    if kg_like:
        weights["target"] += float(cfg.kg_target_boost)
        weights["modality"] += 0.03
    if float(router_entropy) >= 0.70:
        weights["coverage"] += float(cfg.uncertainty_coverage_boost)
        weights["diversity"] += 0.02

    selected: list[int] = []
    selected_set: set[int] = set()
    covered_segments: set[int] = set()

    def rank_signal(pos_map: dict[int, int], j: int) -> float:
        if j not in pos_map:
            return 0.0
        return 1.0 / float(pos_map[j] + 1)

    def add_doc(j: int) -> bool:
        if j in selected_set:
            return False
        selected.append(j)
        selected_set.add(j)
        _, newly = coverage_gain(doc_tokens[j], segments, covered_segments, float(cfg.coverage_threshold))
        covered_segments.update(newly)
        return True

    def score_doc(j: int, missing_forced: set[str]) -> float:
        bucket = source_bucket_fn(doc_ids[j])
        lexical = len(q_tokens & doc_tokens[j]) / max(1, len(q_tokens)) if q_tokens else 0.0
        prf = sum(prf_weights.get(tok, 0.0) for tok in doc_tokens[j]) / max(1e-9, prf_norm)
        cov, _ = coverage_gain(doc_tokens[j], segments, covered_segments, float(cfg.coverage_threshold))
        support = target_support_score(profile, bucket, doc_texts[j], doc_tokens[j], doc_numeric_literals[j])
        modality = profile.desired_modalities.get(bucket, 0.0)
        forced_bonus = 0.06 if bucket in missing_forced else 0.0
        sibling_bonus = sibling_weight if j in sibling_set else 0.0
        diversity_pen = 0.0
        if selected:
            diversity_pen = max(jaccard(doc_tokens[j], doc_tokens[x]) for x in selected)
        return float(
            weights["base"] * base_score.get(j, rank_signal(rank_pos, j))
            + weights["dense"] * rank_signal(dense_pos, j)
            + weights["sparse"] * rank_signal(sparse_pos, j)
            + weights["lexical"] * lexical
            + weights["prf"] * prf
            + weights["modality"] * modality
            + weights["target"] * support
            + weights["coverage"] * cov
            + forced_bonus
            + sibling_bonus
            - weights["diversity"] * diversity_pen
        )

    preserve_n = min(max(0, int(cfg.preserve_top)), topk)
    for j in ranked[:preserve_n]:
        add_doc(j)

    forced = set(profile.force_modalities)
    if table_like:
        forced.add("table")
    if kg_like:
        forced.add("kg")
    for bucket in sorted(forced, key=lambda b: profile.desired_modalities.get(b, 0.0), reverse=True):
        if len(selected) >= topk:
            break
        bucket_pool = [j for j in candidates if j not in selected_set and source_bucket_fn(doc_ids[j]) == bucket]
        if not bucket_pool:
            continue
        add_doc(max(bucket_pool, key=lambda j: score_doc(j, forced)))

    while len(selected) < topk:
        remaining = [j for j in candidates if j not in selected_set]
        if not remaining:
            break
        selected_buckets = {source_bucket_fn(doc_ids[j]) for j in selected}
        missing_forced = set(profile.force_modalities) - selected_buckets
        if table_like and "table" not in selected_buckets:
            missing_forced.add("table")
        if kg_like and "kg" not in selected_buckets:
            missing_forced.add("kg")
        add_doc(max(remaining, key=lambda j: score_doc(j, missing_forced)))

    if len(selected) < topk:
        for j in ranked + dense + sparse:
            if len(selected) >= topk:
                break
            add_doc(j)

    out = selected[:topk]
    buckets = [source_bucket_fn(doc_ids[j]) for j in out]
    return out, MoERetrievalTrace(
        profile=profile,
        selected_buckets=buckets,
        table_like=table_like,
        kg_like=kg_like,
        prf_terms=len(prf_weights),
        sibling_added=len([j for j in set(out) if j in sibling_set]),
        coverage=float(len(covered_segments) / max(1, len(segments))) if segments else 0.0,
    )
