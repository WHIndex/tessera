from __future__ import annotations

from dataclasses import dataclass, field
import pickle
from pathlib import Path

import numpy as np


PESV_FEATURE_NAMES = [
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
    "candidate_support_count",
    "old_support_count",
    "support_delta",
    "qfam_cwq",
    "qfam_nq",
    "qfam_ott",
    "qfam_tat",
    "qfam_triviaqa",
    "qfam_webqsp",
    "candidate_bucket_text",
    "candidate_bucket_table",
    "candidate_bucket_kg",
    "old_bucket_text",
    "old_bucket_table",
    "old_bucket_kg",
    "candidate_query_jaccard",
    "old_query_jaccard",
    "query_jaccard_delta",
    "candidate_query_coverage",
    "old_query_coverage",
    "query_coverage_delta",
    "candidate_query_overlap_count",
    "old_query_overlap_count",
    "query_overlap_count_delta",
    "candidate_numeric_overlap",
    "old_numeric_overlap",
    "numeric_overlap_delta",
    "candidate_anchor_jaccard",
    "old_anchor_jaccard",
    "anchor_jaccard_delta",
    "candidate_anchor_overlap_count",
    "old_anchor_overlap_count",
    "anchor_overlap_count_delta",
    "candidate_anchor_novelty",
    "old_anchor_novelty",
    "anchor_novelty_delta",
    "candidate_len_log",
    "old_len_log",
    "len_log_delta",
]


@dataclass
class PairwiseSlotVerifierBundle:
    model: object
    feature_names: list[str] = field(default_factory=lambda: list(PESV_FEATURE_NAMES))
    metadata: dict = field(default_factory=dict)

    def _align_features(self, features: np.ndarray) -> np.ndarray:
        arr = np.asarray(features, dtype=np.float32)
        if arr.ndim == 1:
            arr = arr.reshape(1, -1)
        expected = len(self.feature_names) if self.feature_names else len(PESV_FEATURE_NAMES)
        if arr.shape[1] == expected:
            return arr
        if arr.shape[1] > expected:
            return arr[:, :expected]
        pad = np.zeros((arr.shape[0], expected - arr.shape[1]), dtype=np.float32)
        return np.hstack([arr, pad]).astype(np.float32)

    def score(self, features: np.ndarray) -> np.ndarray:
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
        raw = np.asarray(self.model.predict(x), dtype=np.float32).reshape(-1)
        return raw.astype(np.float32)


def save_pairwise_slot_verifier_bundle(bundle: PairwiseSlotVerifierBundle, path: Path | str) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("wb") as f:
        pickle.dump(bundle, f)


def load_pairwise_slot_verifier_bundle(path: Path | str) -> PairwiseSlotVerifierBundle:
    with Path(path).open("rb") as f:
        obj = pickle.load(f)
    if isinstance(obj, PairwiseSlotVerifierBundle):
        return obj
    if isinstance(obj, dict) and "model" in obj:
        return PairwiseSlotVerifierBundle(
            model=obj["model"],
            feature_names=list(obj.get("feature_names", PESV_FEATURE_NAMES)),
            metadata=dict(obj.get("metadata", obj.get("meta", {}))),
        )
    raise TypeError(f"Unsupported pairwise slot verifier bundle type: {type(obj)!r}")
