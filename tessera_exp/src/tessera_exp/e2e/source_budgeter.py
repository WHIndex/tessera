from __future__ import annotations

from dataclasses import dataclass, field
import pickle
from pathlib import Path
import re
from typing import Sequence

import numpy as np


TOKEN_RE = re.compile(r"[a-z0-9]+")

SOURCE_LABELS = ["text", "table", "kg"]

SOURCE_BUDGET_FEATURE_NAMES = [
    "q_len_log",
    "has_who",
    "has_where",
    "has_when",
    "has_which",
    "has_how",
    "has_number_word",
    "has_year_token",
    "has_table_cue",
    "has_kg_relation_cue",
    "has_entity_relation_cue",
    "has_text_factoid_cue",
    "qfam_cwq",
    "qfam_webqsp",
    "qfam_nq",
    "qfam_triviaqa",
    "qfam_ott",
    "qfam_tat",
]

TABLE_CUES = {
    "how", "many", "much", "number", "total", "rank", "population", "area", "length",
    "score", "season", "team", "club", "tournament", "league", "championship",
}
KG_RELATION_CUES = {
    "spouse", "parent", "child", "children", "born", "died", "buried", "located",
    "capital", "country", "language", "religion", "genre", "director", "author",
    "founded", "leader", "minister", "president", "bordering", "contains",
}
TEXT_FACTOID_CUES = {
    "released", "premiered", "episode", "series", "album", "song", "movie", "film",
    "novel", "book", "life", "career", "finale",
}


def _tokens(text: str | None) -> list[str]:
    return TOKEN_RE.findall(str(text or "").lower())


def _family(query_id: str | None) -> str:
    return str(query_id or "").split("_", 1)[0].lower()


def build_source_budget_features(query_text: str, query_id: str = "") -> np.ndarray:
    toks = _tokens(query_text)
    tok_set = set(toks)
    qfam = _family(query_id)
    values = [
        float(np.log1p(len(toks))),
        float("who" in tok_set),
        float("where" in tok_set),
        float("when" in tok_set),
        float("which" in tok_set),
        float("how" in tok_set),
        float(bool(tok_set & {"many", "much", "number", "total", "count"})),
        float(any(re.fullmatch(r"(1[5-9]\d{2}|20\d{2})", tok) for tok in toks)),
        float(bool(tok_set & TABLE_CUES)),
        float(bool(tok_set & KG_RELATION_CUES)),
        float(bool(tok_set & {"of", "whose", "from", "by", "in"} and tok_set & KG_RELATION_CUES)),
        float(bool(tok_set & TEXT_FACTOID_CUES)),
        float(qfam == "cwq"),
        float(qfam == "webqsp"),
        float(qfam == "nq"),
        float(qfam == "triviaqa"),
        float(qfam == "ott"),
        float(qfam == "tat"),
    ]
    return np.asarray(values, dtype=np.float32)


@dataclass
class SourceBudgetPrediction:
    top1_source: str
    top1_probs: dict[str, float]
    need_probs: dict[str, float]

    def as_trace(self) -> dict[str, float]:
        out: dict[str, float] = {}
        for source in SOURCE_LABELS:
            out[f"top1_prob_{source}"] = float(self.top1_probs.get(source, 0.0))
            out[f"need_prob_{source}"] = float(self.need_probs.get(source, 0.0))
        out["top1_source_index"] = float(SOURCE_LABELS.index(self.top1_source) if self.top1_source in SOURCE_LABELS else -1)
        return out


@dataclass
class SourceBudgeterBundle:
    top1_model: object
    need_models: dict[str, object] = field(default_factory=dict)
    feature_names: list[str] = field(default_factory=lambda: list(SOURCE_BUDGET_FEATURE_NAMES))
    metadata: dict = field(default_factory=dict)

    def _align_features(self, features: np.ndarray) -> np.ndarray:
        arr = np.asarray(features, dtype=np.float32)
        if arr.ndim == 1:
            arr = arr.reshape(1, -1)
        expected = len(self.feature_names) if self.feature_names else len(SOURCE_BUDGET_FEATURE_NAMES)
        if arr.shape[1] == expected:
            return arr
        if arr.shape[1] > expected:
            return arr[:, :expected]
        pad = np.zeros((arr.shape[0], expected - arr.shape[1]), dtype=np.float32)
        return np.hstack([arr, pad]).astype(np.float32)

    @staticmethod
    def _positive_proba(model: object, x: np.ndarray) -> np.ndarray:
        if hasattr(model, "predict_proba"):
            probs = np.asarray(model.predict_proba(x), dtype=np.float32)
            if probs.ndim == 2 and probs.shape[1] >= 2:
                classes = np.asarray(getattr(model, "classes_", []))
                if classes.size == probs.shape[1] and 1 in classes.tolist():
                    idx = int(np.where(classes == 1)[0][0])
                    return probs[:, idx].astype(np.float32)
                return probs[:, -1].astype(np.float32)
        if hasattr(model, "decision_function"):
            raw = np.asarray(model.decision_function(x), dtype=np.float32).reshape(-1)
            return (1.0 / (1.0 + np.exp(-raw))).astype(np.float32)
        return np.asarray(model.predict(x), dtype=np.float32).reshape(-1).astype(np.float32)

    def predict(self, query_text: str, query_id: str = "") -> SourceBudgetPrediction:
        x = self._align_features(build_source_budget_features(query_text, query_id))
        top1_probs: dict[str, float] = {}
        if hasattr(self.top1_model, "predict_proba"):
            probs = np.asarray(self.top1_model.predict_proba(x), dtype=np.float32)
            classes = [str(c) for c in getattr(self.top1_model, "classes_", [])]
            if probs.ndim == 2 and probs.shape[0] == 1:
                for cls, prob in zip(classes, probs[0].tolist()):
                    if cls in SOURCE_LABELS:
                        top1_probs[cls] = float(prob)
        if not top1_probs:
            pred = str(self.top1_model.predict(x)[0])
            top1_probs = {source: (1.0 if source == pred else 0.0) for source in SOURCE_LABELS}
        for source in SOURCE_LABELS:
            top1_probs.setdefault(source, 0.0)
        top1_source = max(SOURCE_LABELS, key=lambda source: top1_probs.get(source, 0.0))

        need_probs: dict[str, float] = {}
        for source in SOURCE_LABELS:
            model = self.need_models.get(source)
            need_probs[source] = float(self._positive_proba(model, x)[0]) if model is not None else 0.0
        return SourceBudgetPrediction(top1_source=top1_source, top1_probs=top1_probs, need_probs=need_probs)


def save_source_budgeter_bundle(bundle: SourceBudgeterBundle, path: Path | str) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("wb") as f:
        pickle.dump(bundle, f)


def load_source_budgeter_bundle(path: Path | str) -> SourceBudgeterBundle:
    with Path(path).open("rb") as f:
        obj = pickle.load(f)
    if isinstance(obj, SourceBudgeterBundle):
        return obj
    if isinstance(obj, dict) and "top1_model" in obj:
        return SourceBudgeterBundle(
            top1_model=obj["top1_model"],
            need_models=dict(obj.get("need_models", {})),
            feature_names=list(obj.get("feature_names", SOURCE_BUDGET_FEATURE_NAMES)),
            metadata=dict(obj.get("metadata", obj.get("meta", {}))),
        )
    raise TypeError(f"Unsupported source budgeter bundle type: {type(obj)!r}")
