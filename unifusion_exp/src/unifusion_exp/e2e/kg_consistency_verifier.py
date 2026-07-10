from __future__ import annotations

from dataclasses import dataclass, field
import pickle
from pathlib import Path
import re
from typing import Sequence

import numpy as np


TOKEN_RE = re.compile(r"[a-z0-9]+")
TRIPLE_RE = re.compile(r"([^-\n]{1,120}?)\s+--\s+([^-\n]{1,100}?)\s+-->\s+([^-\n]{1,180})")

STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "by", "did", "do", "does",
    "for", "from", "how", "in", "is", "it", "many", "much", "of", "on",
    "or", "the", "that", "this", "to", "was", "were", "what", "when",
    "where", "which", "who", "whom", "whose", "will", "with",
}

QUESTION_RELATION_HINTS = {
    "where": {"place", "location", "country", "city", "county", "state", "capital", "birth", "death", "buried"},
    "when": {"date", "year", "time", "start", "end", "release", "founded", "born", "died"},
    "who": {"person", "people", "actor", "author", "director", "leader", "minister", "president", "player"},
    "whose": {"person", "people", "owner", "parent", "spouse", "child", "author"},
    "which": {"type", "name", "genre", "country", "city", "team", "film", "album", "school"},
    "how": {"number", "count", "population", "amount", "total", "many", "much"},
}

KG_CONSISTENCY_FEATURE_NAMES = [
    "query_len_log",
    "doc_len_log",
    "triple_count_log",
    "query_doc_jaccard",
    "query_doc_recall",
    "query_doc_overlap_log",
    "best_triple_query_recall",
    "best_triple_query_jaccard",
    "best_relation_query_recall",
    "best_relation_query_jaccard",
    "best_subject_query_recall",
    "best_subject_query_jaccard",
    "best_object_query_recall",
    "best_object_query_jaccard",
    "best_entity_query_recall",
    "best_relation_hint_match",
    "description_relation_rate",
    "query_answer_type_place",
    "query_answer_type_time",
    "query_answer_type_person",
    "query_answer_type_number",
    "doc_has_date",
    "doc_has_number",
    "doc_id_is_mid",
]


def tokenize(text: str | None) -> set[str]:
    return {tok for tok in TOKEN_RE.findall(str(text or "").lower()) if len(tok) > 1 and tok not in STOPWORDS}


def _jaccard(a: set[str], b: set[str]) -> float:
    if not a or not b:
        return 0.0
    return float(len(a & b) / max(1, len(a | b)))


def _recall(a: set[str], b: set[str]) -> float:
    if not a:
        return 0.0
    return float(len(a & b) / max(1, len(a)))


def _parse_triples(text: str | None) -> list[tuple[set[str], set[str], set[str]]]:
    out: list[tuple[set[str], set[str], set[str]]] = []
    for subj, rel, obj in TRIPLE_RE.findall(str(text or "")):
        s = tokenize(subj)
        r = tokenize(rel)
        o = tokenize(obj)
        if s or r or o:
            out.append((s, r, o))
        if len(out) >= 64:
            break
    return out


def _question_hint(query_tokens: set[str], relation_tokens: set[str]) -> float:
    best = 0.0
    for qword, hints in QUESTION_RELATION_HINTS.items():
        if qword not in query_tokens:
            continue
        if relation_tokens & hints:
            best = 1.0
            break
        if any(any(hint in rel_tok or rel_tok in hint for rel_tok in relation_tokens) for hint in hints):
            best = max(best, 0.6)
    return float(best)


def _answer_type_flags(query_tokens: set[str]) -> dict[str, float]:
    return {
        "place": float(bool(query_tokens & {"where", "country", "city", "county", "state", "place", "location"})),
        "time": float(bool(query_tokens & {"when", "date", "year", "season"})),
        "person": float(bool(query_tokens & {"who", "whose", "person", "player", "actor", "author", "president"})),
        "number": float(bool(query_tokens & {"how", "many", "much", "number", "total"})),
    }


def build_kg_consistency_features(
    *,
    query_text: str,
    doc_text: str,
    doc_id: str = "",
) -> np.ndarray:
    q = tokenize(query_text)
    d = tokenize(doc_text)
    triples = _parse_triples(doc_text)
    flags = _answer_type_flags(q)
    best_triple_recall = 0.0
    best_triple_jaccard = 0.0
    best_relation_recall = 0.0
    best_relation_jaccard = 0.0
    best_subject_recall = 0.0
    best_subject_jaccard = 0.0
    best_object_recall = 0.0
    best_object_jaccard = 0.0
    best_entity_recall = 0.0
    best_hint = 0.0
    desc_count = 0
    for subj, rel, obj in triples:
        triple_tokens = set(subj) | set(rel) | set(obj)
        entity_tokens = set(subj) | set(obj)
        best_triple_recall = max(best_triple_recall, _recall(q, triple_tokens))
        best_triple_jaccard = max(best_triple_jaccard, _jaccard(q, triple_tokens))
        best_relation_recall = max(best_relation_recall, _recall(q, rel))
        best_relation_jaccard = max(best_relation_jaccard, _jaccard(q, rel))
        best_subject_recall = max(best_subject_recall, _recall(q, subj))
        best_subject_jaccard = max(best_subject_jaccard, _jaccard(q, subj))
        best_object_recall = max(best_object_recall, _recall(q, obj))
        best_object_jaccard = max(best_object_jaccard, _jaccard(q, obj))
        best_entity_recall = max(best_entity_recall, _recall(q, entity_tokens))
        best_hint = max(best_hint, _question_hint(q, rel))
        desc_count += int("description" in rel)

    doc_has_date = float(bool(re.search(r"\b(1[5-9]\d{2}|20\d{2})\b", str(doc_text or ""))))
    doc_has_number = float(bool(re.search(r"\d", str(doc_text or ""))))
    values = [
        float(np.log1p(len(q))),
        float(np.log1p(len(d))),
        float(np.log1p(len(triples))),
        _jaccard(q, d),
        _recall(q, d),
        float(np.log1p(len(q & d))),
        best_triple_recall,
        best_triple_jaccard,
        best_relation_recall,
        best_relation_jaccard,
        best_subject_recall,
        best_subject_jaccard,
        best_object_recall,
        best_object_jaccard,
        best_entity_recall,
        best_hint,
        float(desc_count / max(1, len(triples))),
        flags["place"],
        flags["time"],
        flags["person"],
        flags["number"],
        doc_has_date,
        doc_has_number,
        float(str(doc_id or "").startswith("m.")),
    ]
    return np.asarray(values, dtype=np.float32)


def build_kg_consistency_matrix(
    *,
    query_text: str,
    doc_texts: Sequence[str],
    doc_ids: Sequence[str] | None = None,
) -> np.ndarray:
    ids = list(doc_ids or ["" for _ in doc_texts])
    feats = [
        build_kg_consistency_features(query_text=query_text, doc_text=str(text), doc_id=str(ids[i] if i < len(ids) else ""))
        for i, text in enumerate(doc_texts)
    ]
    if not feats:
        return np.zeros((0, len(KG_CONSISTENCY_FEATURE_NAMES)), dtype=np.float32)
    return np.vstack(feats).astype(np.float32)


@dataclass
class KGConsistencyBundle:
    model: object
    feature_names: list[str] = field(default_factory=lambda: list(KG_CONSISTENCY_FEATURE_NAMES))
    metadata: dict = field(default_factory=dict)

    def _align_features(self, features: np.ndarray) -> np.ndarray:
        arr = np.asarray(features, dtype=np.float32)
        if arr.ndim == 1:
            arr = arr.reshape(1, -1)
        expected = len(self.feature_names) if self.feature_names else len(KG_CONSISTENCY_FEATURE_NAMES)
        if arr.shape[1] == expected:
            return arr
        if arr.shape[1] > expected:
            return arr[:, :expected]
        pad = np.zeros((arr.shape[0], expected - arr.shape[1]), dtype=np.float32)
        return np.hstack([arr, pad]).astype(np.float32)

    def score_features(self, features: np.ndarray) -> np.ndarray:
        if features.size == 0:
            return np.zeros((0,), dtype=np.float32)
        x = self._align_features(features)
        if hasattr(self.model, "predict_proba"):
            probs = np.asarray(self.model.predict_proba(x), dtype=np.float32)
            if probs.ndim == 2 and probs.shape[1] >= 2:
                classes = np.asarray(getattr(self.model, "classes_", []))
                if classes.size == probs.shape[1] and 1 in classes.tolist():
                    idx = int(np.where(classes == 1)[0][0])
                    return probs[:, idx].astype(np.float32)
                return probs[:, -1].astype(np.float32)
            return probs.reshape(-1).astype(np.float32)
        if hasattr(self.model, "decision_function"):
            raw = np.asarray(self.model.decision_function(x), dtype=np.float32).reshape(-1)
            return (1.0 / (1.0 + np.exp(-raw))).astype(np.float32)
        return np.asarray(self.model.predict(x), dtype=np.float32).reshape(-1).astype(np.float32)

    def score_query_docs(
        self,
        *,
        query_text: str,
        doc_texts: Sequence[str],
        doc_ids: Sequence[str] | None = None,
    ) -> np.ndarray:
        feats = build_kg_consistency_matrix(query_text=query_text, doc_texts=doc_texts, doc_ids=doc_ids)
        return self.score_features(feats)


def save_kg_consistency_bundle(bundle: KGConsistencyBundle, path: Path | str) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("wb") as f:
        pickle.dump(bundle, f)


def load_kg_consistency_bundle(path: Path | str) -> KGConsistencyBundle:
    with Path(path).open("rb") as f:
        obj = pickle.load(f)
    if isinstance(obj, KGConsistencyBundle):
        return obj
    if isinstance(obj, dict) and "model" in obj:
        return KGConsistencyBundle(
            model=obj["model"],
            feature_names=list(obj.get("feature_names", KG_CONSISTENCY_FEATURE_NAMES)),
            metadata=dict(obj.get("metadata", obj.get("meta", {}))),
        )
    raise TypeError(f"Unsupported KG consistency bundle type: {type(obj)!r}")
