#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import sys

import numpy as np
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
EVAL = ROOT / "scripts/eval"
for p in (SRC, EVAL):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import run_e2e_table1c as e2e  # noqa: E402
from tessera_exp.e2e.graph_evidence import GraphEvidenceConfig, expand_graph_evidence_candidates  # noqa: E402
from tessera_exp.e2e.objectives import infer_query_modality_prior, infer_qa_target_type  # noqa: E402
from tessera_exp.e2e.ser_ranker import (  # noqa: E402
    SERRankerBundle,
    SERRankerConfig,
    SER_FEATURE_NAMES,
    build_ser_feature_matrix,
    minmax,
    save_ser_ranker_bundle,
)


CHUNK_ID_RE = re.compile(r"^(.*?)([_:\-.])(\d+)$")
QUERY_TOKEN_RE = re.compile(r"[a-z0-9]+")
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
SLOT_SPLIT_RE = re.compile(
    r"[,;\.?]|\bwho\b|\bwhose\b|\bwhich\b|\bthat\b|\bwhere\b|\bwhen\b|\bwhile\b|\bbefore\b|\bafter\b|\bby\b|\bwith\b",
    re.IGNORECASE,
)


def is_test_like_path(path: Path) -> bool:
    raw = str(path).lower()
    name = path.name.lower()
    return "test" in name or "/test" in raw or "\\test" in raw


def split_id_set(rows: list[dict]) -> set[str]:
    return {str(row.get("id", "")).strip() for row in rows if str(row.get("id", "")).strip()}


def make_cache_key(ids: list[str], max_items: int = 2048) -> str:
    if not ids:
        return "empty"
    if len(ids) <= max_items:
        sampled = ids
    else:
        step = max(1, len(ids) // max_items)
        sampled = ids[::step][:max_items]
    payload = "|".join(sampled) + f"|n={len(ids)}"
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()[:12]


def positive_ids(row: dict, doc_id_to_idx: dict[str, int]) -> set[int]:
    out: set[int] = set()
    for doc_id, label in row.get("relevant_chunks", {}).items():
        if doc_id not in doc_id_to_idx:
            continue
        try:
            if float(label) > 0:
                out.add(int(doc_id_to_idx[str(doc_id)]))
        except Exception:
            continue
    return out


def answer_support_score(answer: str | None, doc_text: str | None) -> float:
    ans = e2e.normalize_answer(str(answer or ""))
    if not ans:
        return 0.0
    text = e2e.normalize_answer(str(doc_text or ""))
    if not text:
        return 0.0
    if ans in text:
        return 1.0
    ans_tokens = ans.split()
    text_tokens = set(text.split())
    if not ans_tokens or not text_tokens:
        return 0.0
    overlap = sum(1 for tok in ans_tokens if tok in text_tokens)
    recall = overlap / max(1, len(ans_tokens))
    precision = overlap / max(1, len(text_tokens))
    f1 = 0.0 if recall + precision <= 0.0 else (2.0 * recall * precision) / (recall + precision)
    numeric_ans = {tok for tok in ans_tokens if any(ch.isdigit() for ch in tok)}
    numeric_hit = 1.0 if numeric_ans and numeric_ans.issubset(text_tokens) else 0.0
    return float(max(f1, 0.80 * recall, numeric_hit))


def doc_family_key(doc_id: str) -> str:
    m = CHUNK_ID_RE.match(str(doc_id))
    if not m:
        return str(doc_id)
    return m.group(1)


def decompose_query_slots(query: str | None) -> list[set[str]]:
    text = str(query or "").lower()
    slots: list[set[str]] = []
    seen: set[frozenset[str]] = set()
    for part in SLOT_SPLIT_RE.split(text):
        toks = {tok for tok in QUERY_TOKEN_RE.findall(part) if len(tok) > 1 and tok not in SLOT_STOPWORDS}
        if len(toks) < 2:
            continue
        key = frozenset(toks)
        if key in seen:
            continue
        seen.add(key)
        slots.append(toks)
    if slots:
        return slots[:8]
    toks = {tok for tok in QUERY_TOKEN_RE.findall(text) if len(tok) > 1 and tok not in SLOT_STOPWORDS}
    return [toks] if toks else []


def slot_coverage_score(doc_tokens: set[str], slots: list[set[str]]) -> float:
    if not doc_tokens or not slots:
        return 0.0
    best = 0.0
    covered = 0
    for slot in slots:
        if not slot:
            continue
        overlap = len(slot & doc_tokens) / max(1, len(slot))
        best = max(best, overlap)
        if overlap >= 0.40:
            covered += 1
    covered_score = covered / max(1, len(slots))
    return float(max(best, covered_score))


def apply_coverage_aware_weights(
    *,
    sample_weights: np.ndarray,
    labels: np.ndarray,
    candidate: list[int],
    base_scores: np.ndarray,
    query: str,
    doc_ids: list[str],
    doc_tokens: list[set[str]],
    coverage_positive_weight: float,
    coverage_source_weight: float,
    coverage_family_weight: float,
    coverage_slot_weight: float,
    coverage_hard_negative_weight: float,
    coverage_max_positive_weight: float,
) -> tuple[np.ndarray, dict[str, float]]:
    weights = np.asarray(sample_weights, dtype=np.float32).copy()
    labels_arr = np.asarray(labels, dtype=np.int64)
    pos_idx = np.where(labels_arr == 1)[0]
    neg_idx = np.where(labels_arr == 0)[0]
    stats = {
        "coverage_positive_weight_sum": 0.0,
        "coverage_positive_count": int(pos_idx.size),
        "coverage_unique_positive_sources_sum": 0.0,
        "coverage_unique_positive_families_sum": 0.0,
        "coverage_hard_negative_count": 0.0,
    }
    if pos_idx.size <= 0:
        return weights, stats

    pos_sources = [e2e.source_bucket(doc_ids[candidate[int(i)]]) for i in pos_idx]
    pos_families = [doc_family_key(doc_ids[candidate[int(i)]]) for i in pos_idx]
    source_counts = {src: pos_sources.count(src) for src in set(pos_sources)}
    family_counts = {fam: pos_families.count(fam) for fam in set(pos_families)}
    stats["coverage_unique_positive_sources_sum"] = float(len(source_counts))
    stats["coverage_unique_positive_families_sum"] = float(len(family_counts))

    slots = decompose_query_slots(query)
    multi_positive_need = min(1.0, max(0, int(pos_idx.size) - 1) / 4.0)
    base_positive_boost = float(coverage_positive_weight) * (0.50 + 0.50 * multi_positive_need)
    for i in pos_idx:
        j = candidate[int(i)]
        src = e2e.source_bucket(doc_ids[j])
        fam = doc_family_key(doc_ids[j])
        source_bonus = float(coverage_source_weight) / np.sqrt(max(1.0, float(source_counts.get(src, 1))))
        family_bonus = float(coverage_family_weight) / np.sqrt(max(1.0, float(family_counts.get(fam, 1))))
        slot_bonus = float(coverage_slot_weight) * slot_coverage_score(doc_tokens[j], slots)
        positive_weight = 1.0 + base_positive_boost + source_bonus + family_bonus + slot_bonus
        weights[int(i)] *= float(min(float(coverage_max_positive_weight), positive_weight))

    if neg_idx.size > 0 and float(coverage_hard_negative_weight) > 0.0:
        neg_scores = np.asarray([float(base_scores[int(i)]) for i in neg_idx], dtype=np.float32)
        if neg_scores.size:
            threshold = float(np.quantile(neg_scores, 0.80))
            hard_mask = neg_scores >= threshold
            hard_neg_idx = neg_idx[hard_mask]
            weights[hard_neg_idx] *= 1.0 + float(coverage_hard_negative_weight)
            stats["coverage_hard_negative_count"] = float(hard_neg_idx.size)

    stats["coverage_positive_weight_sum"] = float(np.sum(weights[pos_idx]))
    return weights.astype(np.float32), stats


def build_sparse_backend(doc_texts: list[str], query_texts: list[str], max_features: int):
    from sklearn.feature_extraction.text import TfidfVectorizer

    vec = TfidfVectorizer(stop_words="english", ngram_range=(1, 2), max_features=max_features, min_df=2)
    c_mat = vec.fit_transform(doc_texts).astype(np.float32).tocsr()
    q_mat = vec.transform(query_texts).astype(np.float32).tocsr()
    return q_mat, c_mat


def topk(arr: np.ndarray, k: int) -> np.ndarray:
    return e2e.topk_indices(arr, int(k))


def approximate_base_scores(
    *,
    query: str,
    query_id: str,
    candidate: list[int],
    d_row: np.ndarray,
    s_row: np.ndarray,
    doc_ids: list[str],
    doc_tokens: list[set[str]],
    q_tokens: set[str],
) -> np.ndarray:
    if not candidate:
        return np.zeros((0,), dtype=np.float32)
    d_norm = minmax([float(d_row[j]) for j in candidate])
    s_norm = minmax([float(s_row[j]) for j in candidate])
    prior = infer_query_modality_prior(query, query_id=query_id)
    vals: list[float] = []
    for pos, j in enumerate(candidate):
        bucket = e2e.source_bucket(doc_ids[j])
        if bucket == "text":
            bucket_prob = float(prior[0])
        elif bucket == "table":
            bucket_prob = float(prior[1])
        else:
            bucket_prob = float(prior[2])
        overlap = len(q_tokens & doc_tokens[j]) / max(1, len(q_tokens)) if q_tokens else 0.0
        vals.append(float(0.60 * d_norm[pos] + 0.24 * s_norm[pos] + 0.10 * overlap + 0.06 * bucket_prob))
    return np.asarray(vals, dtype=np.float32)


def build_dataset(
    *,
    rows: list[dict],
    qv: np.ndarray,
    cv: np.ndarray,
    q_sparse,
    c_sparse,
    doc_ids: list[str],
    doc_id_to_idx: dict[str, int],
    doc_texts: list[str],
    doc_tokens: list[set[str]],
    doc_numeric_literals: list[set[str]],
    dense_pool_k: int,
    sparse_pool_k: int,
    candidate_pool_k: int,
    negatives_per_query: int,
    gee_candidate_expansion: bool = False,
    gee_config: GraphEvidenceConfig | None = None,
    answer_aware_supervision: bool = False,
    answer_positive_weight: float = 1.0,
    answer_negative_discount: float = 0.35,
    answer_pseudo_positive_threshold: float = 1.01,
    answer_min_negative_weight: float = 0.25,
    coverage_aware_supervision: bool = False,
    coverage_positive_weight: float = 0.70,
    coverage_source_weight: float = 0.25,
    coverage_family_weight: float = 0.18,
    coverage_slot_weight: float = 0.20,
    coverage_hard_negative_weight: float = 0.10,
    coverage_max_positive_weight: float = 2.40,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict]:
    x_parts: list[np.ndarray] = []
    y_parts: list[np.ndarray] = []
    w_parts: list[np.ndarray] = []
    source_parts: list[np.ndarray] = []
    stats = {
        "queries": 0,
        "examples": 0,
        "positives": 0,
        "positive_in_pool": 0,
        "answer_aware_supervision": int(bool(answer_aware_supervision)),
        "answer_support_positive_sum": 0.0,
        "answer_support_negative_sum": 0.0,
        "answer_support_positive_count": 0,
        "answer_support_negative_count": 0,
        "answer_pseudo_positives": 0,
        "sample_weight_sum": 0.0,
        "coverage_aware_supervision": int(bool(coverage_aware_supervision)),
        "coverage_positive_weight_sum": 0.0,
        "coverage_positive_count": 0,
        "coverage_unique_positive_sources_sum": 0.0,
        "coverage_unique_positive_families_sum": 0.0,
        "coverage_hard_negative_count": 0,
        "gee_enabled": int(bool(gee_candidate_expansion)),
        "gee_triggered": 0,
        "gee_graph_added": 0,
        "gee_boosted_existing": 0,
        "gee_input_candidates": 0,
        "gee_output_candidates": 0,
    }

    for qi, row in enumerate(rows):
        query = str(row.get("query", ""))
        query_id = str(row.get("id", f"q_{qi}"))
        answer = str(row.get("answer", ""))
        q_tokens = e2e.tokenize(query)
        pos = positive_ids(row, doc_id_to_idx)
        if not pos:
            continue

        d_row = np.asarray(qv[qi] @ cv.T, dtype=np.float32).reshape(-1)
        s_row = np.asarray((q_sparse[qi] @ c_sparse.T).toarray(), dtype=np.float32).reshape(-1)
        dense_rank = topk(d_row, dense_pool_k).tolist()
        sparse_rank = topk(s_row, sparse_pool_k).tolist()
        candidate: list[int] = []
        seen: set[int] = set()
        for j in dense_rank + sparse_rank + sorted(pos):
            jj = int(j)
            if jj in seen:
                continue
            seen.add(jj)
            candidate.append(jj)
            if len(candidate) >= max(candidate_pool_k, len(pos)):
                break
        for j in sorted(pos):
            if j not in seen:
                seen.add(j)
                candidate.append(j)

        base_scores = approximate_base_scores(
            query=query,
            query_id=query_id,
            candidate=candidate,
            d_row=d_row,
            s_row=s_row,
            doc_ids=doc_ids,
            doc_tokens=doc_tokens,
            q_tokens=q_tokens,
        )
        base_ranked = [j for j, _ in sorted(zip(candidate, base_scores.tolist()), key=lambda item: item[1], reverse=True)]
        if bool(gee_candidate_expansion):
            target_type = infer_qa_target_type(query)
            candidate_arr, base_scores_arr, gee_trace = expand_graph_evidence_candidates(
                query_text=query,
                current_ranked_idxs=base_ranked[: max(10, min(40, len(base_ranked)))],
                candidate_idxs=candidate,
                candidate_base_scores=base_scores,
                dense_scores=d_row,
                sparse_scores=s_row,
                dense_ranked_idxs=dense_rank,
                sparse_ranked_idxs=sparse_rank,
                doc_ids=doc_ids,
                doc_id_to_idx=doc_id_to_idx,
                doc_tokens=doc_tokens,
                target_type=target_type,
                config=gee_config,
            )
            candidate = [int(j) for j in candidate_arr.tolist()]
            base_scores = np.asarray(base_scores_arr, dtype=np.float32).reshape(-1)
            stats["gee_triggered"] += int(bool(getattr(gee_trace, "triggered", False)))
            stats["gee_graph_added"] += int(getattr(gee_trace, "graph_added", 0))
            stats["gee_boosted_existing"] += int(getattr(gee_trace, "boosted_existing", 0))
            stats["gee_input_candidates"] += int(getattr(gee_trace, "input_candidate_count", 0))
            stats["gee_output_candidates"] += int(getattr(gee_trace, "output_candidate_count", len(candidate)))

        score_map: dict[int, float] = {}
        ordered_candidate: list[int] = []
        for j, sc in zip(candidate, base_scores.tolist()):
            jj = int(j)
            if jj in score_map:
                score_map[jj] = max(score_map[jj], float(sc))
                continue
            score_map[jj] = float(sc)
            ordered_candidate.append(jj)
        missing_pos = [int(j) for j in sorted(pos) if int(j) not in score_map]
        if missing_pos:
            missing_scores = approximate_base_scores(
                query=query,
                query_id=query_id,
                candidate=missing_pos,
                d_row=d_row,
                s_row=s_row,
                doc_ids=doc_ids,
                doc_tokens=doc_tokens,
                q_tokens=q_tokens,
            )
            for j, sc in zip(missing_pos, missing_scores.tolist()):
                score_map[int(j)] = float(sc)
                ordered_candidate.append(int(j))
        candidate = ordered_candidate
        base_scores = np.asarray([score_map[j] for j in candidate], dtype=np.float32)
        labels_all = np.asarray([1 if j in pos else 0 for j in candidate], dtype=np.int64)
        support_all = np.asarray(
            [answer_support_score(answer, doc_texts[j]) for j in candidate],
            dtype=np.float32,
        )
        sample_weights = np.ones_like(support_all, dtype=np.float32)
        if bool(coverage_aware_supervision):
            sample_weights, coverage_stats = apply_coverage_aware_weights(
                sample_weights=sample_weights,
                labels=labels_all,
                candidate=candidate,
                base_scores=base_scores,
                query=query,
                doc_ids=doc_ids,
                doc_tokens=doc_tokens,
                coverage_positive_weight=float(coverage_positive_weight),
                coverage_source_weight=float(coverage_source_weight),
                coverage_family_weight=float(coverage_family_weight),
                coverage_slot_weight=float(coverage_slot_weight),
                coverage_hard_negative_weight=float(coverage_hard_negative_weight),
                coverage_max_positive_weight=float(coverage_max_positive_weight),
            )
            stats["coverage_positive_weight_sum"] += float(coverage_stats["coverage_positive_weight_sum"])
            stats["coverage_positive_count"] += int(coverage_stats["coverage_positive_count"])
            stats["coverage_unique_positive_sources_sum"] += float(
                coverage_stats["coverage_unique_positive_sources_sum"]
            )
            stats["coverage_unique_positive_families_sum"] += float(
                coverage_stats["coverage_unique_positive_families_sum"]
            )
            stats["coverage_hard_negative_count"] += int(coverage_stats["coverage_hard_negative_count"])
        if bool(answer_aware_supervision):
            pseudo_mask = (labels_all == 0) & (support_all >= float(answer_pseudo_positive_threshold))
            if np.any(pseudo_mask):
                labels_all[pseudo_mask] = 1
                stats["answer_pseudo_positives"] += int(np.sum(pseudo_mask))
            pos_mask = labels_all == 1
            neg_mask = labels_all == 0
            sample_weights[pos_mask] = 1.0 + float(answer_positive_weight) * support_all[pos_mask]
            sample_weights[neg_mask] = np.maximum(
                float(answer_min_negative_weight),
                1.0 - float(answer_negative_discount) * support_all[neg_mask],
            )
            stats["answer_support_positive_sum"] += float(np.sum(support_all[pos_mask]))
            stats["answer_support_negative_sum"] += float(np.sum(support_all[neg_mask]))
            stats["answer_support_positive_count"] += int(np.sum(pos_mask))
            stats["answer_support_negative_count"] += int(np.sum(neg_mask))
        if labels_all.sum() <= 0:
            continue
        neg_idx = np.where(labels_all == 0)[0]
        pos_idx = np.where(labels_all == 1)[0]
        if neg_idx.size > max(1, negatives_per_query):
            # Keep hard negatives near the top according to dense/sparse approximation.
            hard_neg = neg_idx[np.argsort(-base_scores[neg_idx])[: int(negatives_per_query)]]
            keep_idx = np.unique(np.concatenate([pos_idx, hard_neg])).astype(np.int64)
            candidate = [candidate[int(i)] for i in keep_idx.tolist()]
            labels_all = labels_all[keep_idx]
            sample_weights = sample_weights[keep_idx]
            base_scores = base_scores[keep_idx]

        base_ranked = [j for j, _ in sorted(zip(candidate, base_scores.tolist()), key=lambda item: item[1], reverse=True)]
        feats = build_ser_feature_matrix(
            query_tokens=q_tokens,
            candidate_idxs=candidate,
            candidate_base_scores=base_scores,
            dense_scores=d_row,
            sparse_scores=s_row,
            base_ranked_idxs=base_ranked,
            dense_ranked_idxs=dense_rank,
            sparse_ranked_idxs=sparse_rank,
            doc_ids=doc_ids,
            doc_tokens=doc_tokens,
            doc_numeric_literals=doc_numeric_literals,
            doc_texts=doc_texts,
            router_prob=infer_query_modality_prior(query, query_id=query_id),
            target_type=infer_qa_target_type(query),
            source_bucket_fn=e2e.source_bucket,
            query_text=query,
        )
        x_parts.append(feats)
        y_parts.append(labels_all.astype(np.int64))
        w_parts.append(sample_weights.astype(np.float32))
        source_parts.append(np.asarray([e2e.source_bucket(doc_ids[j]) for j in candidate], dtype=object))
        stats["queries"] += 1
        stats["examples"] += int(labels_all.size)
        stats["positives"] += int(labels_all.sum())
        stats["positive_in_pool"] += int(len(pos & set(candidate)))
        stats["sample_weight_sum"] += float(np.sum(sample_weights))

    if not x_parts:
        return (
            np.zeros((0, len(SER_FEATURE_NAMES)), dtype=np.float32),
            np.zeros((0,), dtype=np.int64),
            np.zeros((0,), dtype=np.float32),
            np.zeros((0,), dtype=object),
            stats,
        )
    return (
        np.vstack(x_parts).astype(np.float32),
        np.concatenate(y_parts).astype(np.int64),
        np.concatenate(w_parts).astype(np.float32),
        np.concatenate(source_parts),
        stats,
    )


def make_ser_model(seed: int) -> Pipeline:
    return Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            (
                "clf",
                SGDClassifier(
                    loss="log_loss",
                    alpha=1e-4,
                    penalty="l2",
                    class_weight="balanced",
                    max_iter=1000,
                    tol=1e-4,
                    random_state=int(seed),
                ),
            ),
        ]
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Train TESSERA-SER supervised evidence-set reranker")
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--train-file", type=Path, required=True)
    parser.add_argument("--dev-file", type=Path, required=True)
    parser.add_argument("--corpus-file", type=Path, required=True)
    parser.add_argument("--out-bundle", type=Path, required=True)
    parser.add_argument("--out-metrics", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path, default=ROOT / "artifacts/retrieval")
    parser.add_argument("--max-train", type=int, default=0)
    parser.add_argument("--max-dev", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--sparse-max-features", type=int, default=200000)
    parser.add_argument("--dense-pool-k", type=int, default=180)
    parser.add_argument("--sparse-pool-k", type=int, default=180)
    parser.add_argument("--candidate-pool-k", type=int, default=260)
    parser.add_argument("--negatives-per-query", type=int, default=96)
    parser.add_argument("--preserve-top", type=int, default=1)
    parser.add_argument("--blend-weight", type=float, default=0.65)
    parser.add_argument("--diversity-weight", type=float, default=0.02)
    parser.add_argument("--gee-candidate-expansion", action="store_true")
    parser.add_argument("--gee-candidate-pool-k", type=int, default=640)
    parser.add_argument("--gee-dense-pool-k", type=int, default=320)
    parser.add_argument("--gee-sparse-pool-k", type=int, default=280)
    parser.add_argument("--gee-graph-seed-k", type=int, default=32)
    parser.add_argument("--gee-graph-window", type=int, default=1)
    parser.add_argument("--gee-preserve-top", type=int, default=2)
    parser.add_argument("--gee-trigger-threshold", type=float, default=0.50)
    parser.add_argument("--gee-base-weight", type=float, default=0.28)
    parser.add_argument("--gee-dense-weight", type=float, default=0.34)
    parser.add_argument("--gee-sparse-weight", type=float, default=0.14)
    parser.add_argument("--gee-probe-weight", type=float, default=0.12)
    parser.add_argument("--gee-graph-weight", type=float, default=0.06)
    parser.add_argument("--gee-slot-weight", type=float, default=0.06)
    parser.add_argument("--gee-sibling-weight", type=float, default=0.04)
    parser.add_argument("--gee-redundancy-weight", type=float, default=0.012)
    parser.add_argument("--answer-aware-supervision", action="store_true")
    parser.add_argument("--answer-positive-weight", type=float, default=1.0)
    parser.add_argument("--answer-negative-discount", type=float, default=0.35)
    parser.add_argument("--answer-pseudo-positive-threshold", type=float, default=1.01)
    parser.add_argument("--answer-min-negative-weight", type=float, default=0.25)
    parser.add_argument("--coverage-aware-supervision", action="store_true")
    parser.add_argument("--coverage-positive-weight", type=float, default=0.70)
    parser.add_argument("--coverage-source-weight", type=float, default=0.25)
    parser.add_argument("--coverage-family-weight", type=float, default=0.18)
    parser.add_argument("--coverage-slot-weight", type=float, default=0.20)
    parser.add_argument("--coverage-hard-negative-weight", type=float, default=0.10)
    parser.add_argument("--coverage-max-positive-weight", type=float, default=2.40)
    parser.add_argument("--source-conditional-heads", action="store_true")
    parser.add_argument("--source-head-blend-weight", type=float, default=0.35)
    parser.add_argument("--source-head-min-positives", type=int, default=256)
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument(
        "--allow-test-split-training",
        action="store_true",
        help="Allow paths that look like test splits. Keep this off for paper-clean training.",
    )
    args = parser.parse_args()

    allow_test_training = bool(args.allow_test_split_training) or os.environ.get("TESSERA_ALLOW_TEST_SPLIT_TRAINING", "0") == "1"
    if not allow_test_training:
        bad_paths = [
            ("train-file", args.train_file),
            ("dev-file", args.dev_file),
        ]
        for label, path in bad_paths:
            if is_test_like_path(Path(path)):
                raise ValueError(
                    f"{label} looks like a test split ({path}). "
                    "SER training must use train/dev only. "
                    "Set TESSERA_ALLOW_TEST_SPLIT_TRAINING=1 only for debugging, never for paper results."
                )

    if int(args.negatives_per_query) < 1:
        raise ValueError("negatives-per-query must be >= 1")
    if int(args.candidate_pool_k) < 1:
        raise ValueError("candidate-pool-k must be >= 1")
    for name in ("gee_candidate_pool_k", "gee_dense_pool_k", "gee_sparse_pool_k", "gee_graph_seed_k"):
        if int(getattr(args, name)) < 1:
            raise ValueError(f"{name.replace('_', '-')} must be >= 1")
    for name in ("gee_graph_window", "gee_preserve_top"):
        if int(getattr(args, name)) < 0:
            raise ValueError(f"{name.replace('_', '-')} must be >= 0")
    args.cache_dir.mkdir(parents=True, exist_ok=True)

    print(f"[stage] loading train/dev: {args.train_file} {args.dev_file}", flush=True)
    train_rows = json.loads(args.train_file.read_text(encoding="utf-8"))
    dev_rows = json.loads(args.dev_file.read_text(encoding="utf-8"))
    if int(args.max_train) > 0:
        train_rows = train_rows[: int(args.max_train)]
    if int(args.max_dev) > 0:
        dev_rows = dev_rows[: int(args.max_dev)]
    train_ids = split_id_set(train_rows)
    dev_ids = split_id_set(dev_rows)
    train_dev_overlap = sorted(train_ids & dev_ids)
    if train_dev_overlap:
        raise ValueError(
            f"train/dev split overlap detected: {len(train_dev_overlap)} queries. "
            f"Examples: {train_dev_overlap[:5]}"
        )
    print(f"[stage] loading training corpus: {args.corpus_file}", flush=True)
    corpus = json.loads(args.corpus_file.read_text(encoding="utf-8"))
    doc_ids = [str(row["id"]) for row in corpus]
    doc_texts = [str(row.get("text", "")) for row in corpus]
    doc_id_to_idx = {doc_id: idx for idx, doc_id in enumerate(doc_ids)}
    print(f"[stage] corpus docs={len(doc_texts)} train={len(train_rows)} dev={len(dev_rows)}", flush=True)
    doc_tokens = [e2e.tokenize(text) for text in doc_texts]
    doc_numeric_literals = [e2e.extract_numeric_literals(text) for text in doc_texts]

    all_q = [str(r.get("query", "")) for r in train_rows + dev_rows]
    all_q_ids = [str(r.get("id", f"q_{i}")) for i, r in enumerate(train_rows + dev_rows)]
    doc_key = make_cache_key(doc_ids)
    q_key = make_cache_key(all_q_ids)
    print("[stage] loading dense encoder", flush=True)
    tokenizer, model, device, resolved = e2e.load_e5(args.model_dir)
    pooling = e2e.detect_st_pooling_mode(resolved)
    model_key = hashlib.sha1(str(resolved).encode("utf-8")).hexdigest()[:12]
    q_cache = args.cache_dir / f"ser_dense_query_{model_key}_{pooling}_{len(all_q)}_{q_key}.npy"
    c_cache = args.cache_dir / f"dense_corpus_{model_key}_{pooling}_{len(doc_texts)}_{doc_key}.npy"
    if q_cache.exists() and np.load(q_cache, mmap_mode="r").shape[0] == len(all_q):
        qv_all = np.load(q_cache)
    else:
        qv_all = e2e.encode_texts(all_q, tokenizer, model, device, batch_size=int(args.batch_size), pooling_mode=pooling)
        np.save(q_cache, qv_all)
    if c_cache.exists() and np.load(c_cache, mmap_mode="r").shape[0] == len(doc_texts):
        cv = np.load(c_cache, mmap_mode="r")
    else:
        cv_arr = e2e.encode_texts(doc_texts, tokenizer, model, device, batch_size=int(args.batch_size), pooling_mode=pooling)
        np.save(c_cache, cv_arr)
        cv = np.load(c_cache, mmap_mode="r")

    print("[stage] building sparse training backend", flush=True)
    q_sparse, c_sparse = build_sparse_backend(doc_texts, all_q, max_features=int(args.sparse_max_features))
    train_qv = qv_all[: len(train_rows)]
    dev_qv = qv_all[len(train_rows):]
    train_sparse = q_sparse[: len(train_rows)]
    dev_sparse = q_sparse[len(train_rows):]
    gee_config = GraphEvidenceConfig(
        candidate_pool_k=int(args.gee_candidate_pool_k),
        dense_pool_k=int(args.gee_dense_pool_k),
        sparse_pool_k=int(args.gee_sparse_pool_k),
        graph_seed_k=int(args.gee_graph_seed_k),
        graph_window=int(args.gee_graph_window),
        preserve_top=int(args.gee_preserve_top),
        trigger_threshold=float(args.gee_trigger_threshold),
        base_weight=float(args.gee_base_weight),
        dense_weight=float(args.gee_dense_weight),
        sparse_weight=float(args.gee_sparse_weight),
        probe_weight=float(args.gee_probe_weight),
        graph_weight=float(args.gee_graph_weight),
        slot_weight=float(args.gee_slot_weight),
        sibling_weight=float(args.gee_sibling_weight),
        redundancy_weight=float(args.gee_redundancy_weight),
    )

    print("[stage] building SER train examples", flush=True)
    x_train, y_train, w_train, train_sources, train_stats = build_dataset(
        rows=train_rows,
        qv=train_qv,
        cv=cv,
        q_sparse=train_sparse,
        c_sparse=c_sparse,
        doc_ids=doc_ids,
        doc_id_to_idx=doc_id_to_idx,
        doc_texts=doc_texts,
        doc_tokens=doc_tokens,
        doc_numeric_literals=doc_numeric_literals,
        dense_pool_k=int(args.dense_pool_k),
        sparse_pool_k=int(args.sparse_pool_k),
        candidate_pool_k=int(args.candidate_pool_k),
        negatives_per_query=int(args.negatives_per_query),
        gee_candidate_expansion=bool(args.gee_candidate_expansion),
        gee_config=gee_config,
        answer_aware_supervision=bool(args.answer_aware_supervision),
        answer_positive_weight=float(args.answer_positive_weight),
        answer_negative_discount=float(args.answer_negative_discount),
        answer_pseudo_positive_threshold=float(args.answer_pseudo_positive_threshold),
        answer_min_negative_weight=float(args.answer_min_negative_weight),
        coverage_aware_supervision=bool(args.coverage_aware_supervision),
        coverage_positive_weight=float(args.coverage_positive_weight),
        coverage_source_weight=float(args.coverage_source_weight),
        coverage_family_weight=float(args.coverage_family_weight),
        coverage_slot_weight=float(args.coverage_slot_weight),
        coverage_hard_negative_weight=float(args.coverage_hard_negative_weight),
        coverage_max_positive_weight=float(args.coverage_max_positive_weight),
    )
    print("[stage] building SER dev examples", flush=True)
    x_dev, y_dev, w_dev, dev_sources, dev_stats = build_dataset(
        rows=dev_rows,
        qv=dev_qv,
        cv=cv,
        q_sparse=dev_sparse,
        c_sparse=c_sparse,
        doc_ids=doc_ids,
        doc_id_to_idx=doc_id_to_idx,
        doc_texts=doc_texts,
        doc_tokens=doc_tokens,
        doc_numeric_literals=doc_numeric_literals,
        dense_pool_k=int(args.dense_pool_k),
        sparse_pool_k=int(args.sparse_pool_k),
        candidate_pool_k=int(args.candidate_pool_k),
        negatives_per_query=int(args.negatives_per_query),
        gee_candidate_expansion=bool(args.gee_candidate_expansion),
        gee_config=gee_config,
        answer_aware_supervision=bool(args.answer_aware_supervision),
        answer_positive_weight=float(args.answer_positive_weight),
        answer_negative_discount=float(args.answer_negative_discount),
        answer_pseudo_positive_threshold=float(args.answer_pseudo_positive_threshold),
        answer_min_negative_weight=float(args.answer_min_negative_weight),
        coverage_aware_supervision=bool(args.coverage_aware_supervision),
        coverage_positive_weight=float(args.coverage_positive_weight),
        coverage_source_weight=float(args.coverage_source_weight),
        coverage_family_weight=float(args.coverage_family_weight),
        coverage_slot_weight=float(args.coverage_slot_weight),
        coverage_hard_negative_weight=float(args.coverage_hard_negative_weight),
        coverage_max_positive_weight=float(args.coverage_max_positive_weight),
    )

    if x_train.size == 0 or len(set(y_train.tolist())) < 2:
        raise RuntimeError("SER training data is empty or has a single class")
    print(f"[stage] fitting SER model examples={x_train.shape[0]} positives={int(y_train.sum())}", flush=True)
    model_pipe = make_ser_model(seed=int(args.seed))
    weighted_supervision = bool(args.answer_aware_supervision) or bool(args.coverage_aware_supervision)
    fit_kwargs = {}
    if weighted_supervision:
        fit_kwargs["clf__sample_weight"] = w_train
    model_pipe.fit(x_train, y_train, **fit_kwargs)

    source_models: dict[str, object] = {}
    source_head_metrics: dict[str, dict] = {}
    if bool(args.source_conditional_heads):
        print("[stage] fitting source-conditional SER heads", flush=True)
        for offset, source in enumerate(("text", "table", "kg"), start=1):
            train_mask = np.asarray(train_sources == source, dtype=bool)
            dev_mask = np.asarray(dev_sources == source, dtype=bool)
            train_positive = int(np.sum(y_train[train_mask])) if np.any(train_mask) else 0
            train_total = int(np.sum(train_mask))
            source_head_metrics[source] = {
                "train_examples": train_total,
                "train_positives": train_positive,
                "enabled": False,
            }
            if train_total <= 0 or train_positive < int(args.source_head_min_positives):
                continue
            if len(set(y_train[train_mask].tolist())) < 2:
                continue
            source_model = make_ser_model(seed=int(args.seed) + offset)
            source_fit_kwargs = {}
            if weighted_supervision:
                source_fit_kwargs["clf__sample_weight"] = w_train[train_mask]
            source_model.fit(x_train[train_mask], y_train[train_mask], **source_fit_kwargs)
            source_models[source] = source_model
            source_head_metrics[source]["enabled"] = True
            if np.any(dev_mask):
                source_dev_prob = source_model.predict_proba(x_dev[dev_mask])[:, 1]
                source_head_metrics[source]["dev_examples"] = int(np.sum(dev_mask))
                source_head_metrics[source]["dev_positives"] = int(np.sum(y_dev[dev_mask]))
                source_head_metrics[source]["dev_average_precision"] = float(
                    average_precision_score(y_dev[dev_mask], source_dev_prob)
                )
                source_head_metrics[source]["dev_roc_auc"] = (
                    float(roc_auc_score(y_dev[dev_mask], source_dev_prob))
                    if len(set(y_dev[dev_mask].tolist())) > 1
                    else None
                )

    dev_prob = model_pipe.predict_proba(x_dev)[:, 1] if x_dev.size else np.zeros((0,), dtype=np.float32)
    weighted_dev_ap = None
    weighted_dev_auc = None
    if weighted_supervision and x_dev.size:
        weighted_dev_ap = float(average_precision_score(y_dev, dev_prob, sample_weight=w_dev))
        weighted_dev_auc = (
            float(roc_auc_score(y_dev, dev_prob, sample_weight=w_dev))
            if len(set(y_dev.tolist())) > 1
            else None
        )
    if bool(args.source_conditional_heads) and bool(args.coverage_aware_supervision) and bool(args.answer_aware_supervision):
        formulation = (
            "source-conditional coverage-preserving answer-aware evidence utility learning "
            "over unified text/table/KG chunks"
        )
    elif bool(args.coverage_aware_supervision) and bool(args.answer_aware_supervision):
        formulation = "coverage-preserving answer-aware evidence utility learning over unified text/table/KG chunks"
    elif bool(args.coverage_aware_supervision):
        formulation = "coverage-preserving evidence utility learning over unified text/table/KG chunks"
    elif bool(args.answer_aware_supervision):
        formulation = "answer-aware supervised evidence utility learning over unified text/table/KG chunks"
    else:
        formulation = "supervised evidence utility learning over unified text/table/KG chunks"
    metrics = {
        "method_name": (
            "Source-Conditional Evidence Utility Model"
            if bool(args.source_conditional_heads)
            else "Coverage-Aware Source Evidence Utility Model"
        ),
        "method_formulation": formulation,
        "train": train_stats,
        "dev": dev_stats,
        "feature_names": SER_FEATURE_NAMES,
        "model_dir": str(resolved),
        "split_guard": {
            "train_file": str(args.train_file),
            "dev_file": str(args.dev_file),
            "corpus_file": str(args.corpus_file),
            "train_queries": int(len(train_ids)),
            "dev_queries": int(len(dev_ids)),
            "train_dev_overlap": int(len(train_dev_overlap)),
            "test_like_paths_allowed": bool(allow_test_training),
        },
        "dev_average_precision": float(average_precision_score(y_dev, dev_prob)) if x_dev.size else None,
        "dev_roc_auc": float(roc_auc_score(y_dev, dev_prob)) if x_dev.size and len(set(y_dev.tolist())) > 1 else None,
        "dev_weighted_average_precision": weighted_dev_ap,
        "dev_weighted_roc_auc": weighted_dev_auc,
        "source_head_metrics": source_head_metrics,
        "config": {
            "candidate_pool_k": int(args.candidate_pool_k),
            "dense_pool_k": int(args.dense_pool_k),
            "sparse_pool_k": int(args.sparse_pool_k),
            "negatives_per_query": int(args.negatives_per_query),
            "preserve_top": int(args.preserve_top),
            "blend_weight": float(args.blend_weight),
            "diversity_weight": float(args.diversity_weight),
            "gee_candidate_expansion": bool(args.gee_candidate_expansion),
            "gee_candidate_pool_k": int(args.gee_candidate_pool_k),
            "gee_dense_pool_k": int(args.gee_dense_pool_k),
            "gee_sparse_pool_k": int(args.gee_sparse_pool_k),
            "gee_graph_seed_k": int(args.gee_graph_seed_k),
            "gee_graph_window": int(args.gee_graph_window),
            "gee_preserve_top": int(args.gee_preserve_top),
            "gee_trigger_threshold": float(args.gee_trigger_threshold),
            "gee_base_weight": float(args.gee_base_weight),
            "gee_dense_weight": float(args.gee_dense_weight),
            "gee_sparse_weight": float(args.gee_sparse_weight),
            "gee_probe_weight": float(args.gee_probe_weight),
            "gee_graph_weight": float(args.gee_graph_weight),
            "gee_slot_weight": float(args.gee_slot_weight),
            "gee_sibling_weight": float(args.gee_sibling_weight),
            "gee_redundancy_weight": float(args.gee_redundancy_weight),
            "answer_aware_supervision": bool(args.answer_aware_supervision),
            "answer_positive_weight": float(args.answer_positive_weight),
            "answer_negative_discount": float(args.answer_negative_discount),
            "answer_pseudo_positive_threshold": float(args.answer_pseudo_positive_threshold),
            "answer_min_negative_weight": float(args.answer_min_negative_weight),
            "coverage_aware_supervision": bool(args.coverage_aware_supervision),
            "coverage_positive_weight": float(args.coverage_positive_weight),
            "coverage_source_weight": float(args.coverage_source_weight),
            "coverage_family_weight": float(args.coverage_family_weight),
            "coverage_slot_weight": float(args.coverage_slot_weight),
            "coverage_hard_negative_weight": float(args.coverage_hard_negative_weight),
            "coverage_max_positive_weight": float(args.coverage_max_positive_weight),
            "source_conditional_heads": bool(args.source_conditional_heads),
            "source_head_blend_weight": float(args.source_head_blend_weight),
            "source_head_min_positives": int(args.source_head_min_positives),
        },
    }
    bundle = SERRankerBundle(
        model=model_pipe,
        feature_names=list(SER_FEATURE_NAMES),
        config=SERRankerConfig(
            preserve_top=int(args.preserve_top),
            candidate_pool_k=int(args.candidate_pool_k),
            dense_pool_k=int(args.dense_pool_k),
            sparse_pool_k=int(args.sparse_pool_k),
            blend_weight=float(args.blend_weight),
            diversity_weight=float(args.diversity_weight),
        ),
        meta=metrics,
        source_models=source_models,
        source_blend_weight=float(args.source_head_blend_weight) if bool(args.source_conditional_heads) else 0.0,
    )
    save_ser_ranker_bundle(bundle, args.out_bundle)
    args.out_metrics.parent.mkdir(parents=True, exist_ok=True)
    args.out_metrics.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(metrics, ensure_ascii=False, indent=2))
    print(f"[OK] bundle -> {args.out_bundle}")
    print(f"[OK] metrics -> {args.out_metrics}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
