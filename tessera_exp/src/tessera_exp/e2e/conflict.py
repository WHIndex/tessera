from __future__ import annotations

import re
from typing import Iterable

import numpy as np

from tessera_exp.e2e.baselines import source_bucket


TOKEN_RE = re.compile(r"[a-z0-9]+")
NUMERIC_LITERAL_RE = re.compile(
    r"\b\d{1,3}(?:,\d{3})*(?:\.\d+)?(?:\s*(?:million|billion|thousand))?\b",
    flags=re.IGNORECASE,
)
YEAR_LITERAL_RE = re.compile(r"\b(?:1[0-9]{3}|20[0-9]{2})\b")

RELATION_HINT_TERMS = [
    "parent",
    "subsidiary",
    "competitor",
    "founded",
    "founded by",
    "owned",
    "acquired",
    "headquarter",
    "located",
    "spouse",
    "capital",
]
RELATION_HINT_ATOMS = {
    "parent",
    "subsidiary",
    "competitor",
    "founded",
    "owned",
    "acquired",
    "headquarter",
    "located",
    "spouse",
    "capital",
}
CONSISTENCY_KEYWORDS = {
    "revenue",
    "profit",
    "loss",
    "capacity",
    "population",
    "bankrupt",
    "winner",
    "champion",
    "price",
    "score",
    "year",
}


def tokenize(text: str) -> set[str]:
    return set(TOKEN_RE.findall(str(text).lower()))


def normalize_numeric_literal(raw: str) -> str:
    value = str(raw).strip().lower().replace(",", "")
    return re.sub(r"\s+", " ", value)


def extract_numeric_literals(text: str) -> set[str]:
    out: set[str] = set()
    for match in NUMERIC_LITERAL_RE.finditer(str(text)):
        literal = normalize_numeric_literal(match.group(0))
        if literal:
            out.add(literal)
    return out


def extract_signal_tokens(text: str) -> set[str]:
    lower = str(text).lower()
    out = {kw for kw in CONSISTENCY_KEYWORDS if kw in lower}
    out |= {kw for kw in RELATION_HINT_TERMS if kw in lower}
    out |= extract_numeric_literals(text)
    return out


def _context_pairs(indices: Iterable[int]) -> list[tuple[int, int]]:
    items = list(indices)
    out: list[tuple[int, int]] = []
    for i in range(len(items)):
        for j in range(i + 1, len(items)):
            out.append((items[i], items[j]))
    return out


def estimate_context_conflict_risk(
    query: str,
    contexts: list[str],
    doc_ids: list[str] | None = None,
    *,
    table_kg_only: bool = False,
    probe_k: int = 12,
    max_literals_per_doc: int = 0,
) -> float:
    if not contexts:
        return 0.0

    query_tokens = tokenize(query)
    context_tokens = [tokenize(ctx) for ctx in contexts]
    context_signal_tokens = [extract_signal_tokens(ctx) for ctx in contexts]
    context_numeric_literals = [extract_numeric_literals(ctx) for ctx in contexts]

    if doc_ids is None or len(doc_ids) != len(contexts):
        doc_ids = [f"ctx_{i}" for i in range(len(contexts))]

    probe = list(range(min(len(contexts), max(2, int(probe_k)))))
    valid_pairs = 0
    conflict_pairs = 0

    for a, b in _context_pairs(probe):
        nums_a = context_numeric_literals[a]
        nums_b = context_numeric_literals[b]
        if not nums_a or not nums_b:
            continue
        if int(max_literals_per_doc) > 0:
            if len(nums_a) > int(max_literals_per_doc) or len(nums_b) > int(max_literals_per_doc):
                continue

        qov_a = len(query_tokens & context_tokens[a]) / max(1, len(query_tokens)) if query_tokens else 0.0
        qov_b = len(query_tokens & context_tokens[b]) / max(1, len(query_tokens)) if query_tokens else 0.0
        if qov_a <= 0.0 or qov_b <= 0.0:
            continue

        bucket_a = source_bucket(str(doc_ids[a]))
        bucket_b = source_bucket(str(doc_ids[b]))
        if bucket_a == bucket_b:
            continue
        if bool(table_kg_only) and {bucket_a, bucket_b} != {"table", "kg"}:
            continue

        kw_a = {x for x in context_signal_tokens[a] if (x in CONSISTENCY_KEYWORDS or x in RELATION_HINT_ATOMS)}
        kw_b = {x for x in context_signal_tokens[b] if (x in CONSISTENCY_KEYWORDS or x in RELATION_HINT_ATOMS)}
        kw_overlap = kw_a & kw_b
        if not kw_overlap and (query_tokens & context_tokens[a] & context_tokens[b]) == set():
            continue

        valid_pairs += 1
        if (nums_a & nums_b) == set():
            conflict_pairs += 1

    if valid_pairs <= 0:
        return 0.0
    return float(conflict_pairs / valid_pairs)