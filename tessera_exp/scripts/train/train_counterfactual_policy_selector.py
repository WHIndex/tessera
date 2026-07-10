#!/usr/bin/env python3
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
import math
import os
from pathlib import Path
import sys

import numpy as np
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from tessera_exp.e2e.counterfactual_policy_selector import (  # noqa: E402
    CounterfactualPolicySelectorBundle,
    FEATURE_NAMES,
    POLICY_LABELS,
    build_selector_features,
    candidate_rankings,
    query_family,
    retrieval_utility,
    save_counterfactual_policy_selector_bundle,
)


def is_test_like_path(path: Path) -> bool:
    raw = str(path).lower()
    return "test" in path.name.lower() or "/test" in raw or "\\test" in raw


def iter_jsonl(path: Path, max_examples: int = 0) -> list[dict]:
    rows: list[dict] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            rows.append(json.loads(line))
            if int(max_examples) > 0 and len(rows) >= int(max_examples):
                break
    return rows


def qrels_for_row(row: dict) -> dict[str, float]:
    out: dict[str, float] = {}
    for doc_id, raw in (row.get("qrels", {}) or {}).items():
        try:
            val = float(raw)
        except Exception:
            continue
        if val > 0.0:
            out[str(doc_id)] = val
    return out


def ranking_for_row(row: dict, method: str) -> list[str]:
    rankings = row.get("rankings", {}) or {}
    return [str(x) for x in rankings.get(method, [])]


def make_dataset(
    rows: list[dict],
    *,
    method: str,
    topk: int,
    pool_k: int,
    policy_labels: list[str],
    weights: dict[str, float],
    target_mode: str = "utility",
) -> tuple[np.ndarray, np.ndarray, dict]:
    xs: list[np.ndarray] = []
    ys: list[float] = []
    query_count = 0
    best_counter: Counter[str] = Counter()
    positive_gain_counter: Counter[str] = Counter()
    family_best: dict[str, Counter[str]] = defaultdict(Counter)
    family_positive_gain: dict[str, Counter[str]] = defaultdict(Counter)
    oracle_gain_sum = 0.0
    default_utility_sum = 0.0
    for row in rows:
        base = ranking_for_row(row, method)
        qrels = qrels_for_row(row)
        if len(base) < max(1, int(topk)) or not qrels:
            continue
        query_count += 1
        family = query_family(str(row.get("query_id", "")))
        candidates = candidate_rankings(
            base,
            query_id=str(row.get("query_id", "")),
            topk=int(topk),
            pool_k=int(pool_k),
            policy_labels=policy_labels,
        )
        utilities: dict[str, float] = {}
        for policy, ranked in candidates.items():
            utility = retrieval_utility(
                ranked,
                qrels,
                topk=int(topk),
                ndcg1_weight=float(weights["ndcg1"]),
                ndcg5_weight=float(weights["ndcg5"]),
                map5_weight=float(weights["map5"]),
                hits5_weight=float(weights["hits5"]),
            )
            utilities[policy] = float(utility)
        default_utility = float(utilities.get("keep_current", 0.0))
        for policy, ranked in candidates.items():
            xs.append(
                build_selector_features(
                    query_text=str(row.get("query", "")),
                    query_id=str(row.get("query_id", "")),
                    base_ranked_doc_ids=base,
                    candidate_ranked_doc_ids=ranked,
                    policy=policy,
                    trace=row.get("trace", {}),
                )
            )
            if str(target_mode) == "gain":
                ys.append(float(utilities.get(policy, 0.0) - default_utility))
            else:
                ys.append(float(utilities.get(policy, 0.0)))
            if policy != "keep_current" and float(utilities.get(policy, 0.0) - default_utility) > 1e-12:
                positive_gain_counter[policy] += 1
                family_positive_gain[family][policy] += 1
        best_policy = max(policy_labels, key=lambda label: (utilities.get(label, 0.0), -policy_labels.index(label)))
        best_counter[best_policy] += 1
        family_best[family][best_policy] += 1
        default_utility_sum += default_utility
        oracle_gain_sum += max(0.0, float(utilities.get(best_policy, 0.0)) - default_utility)
    stats = {
        "queries": int(query_count),
        "examples": int(len(xs)),
        "target_mode": str(target_mode),
        "best_policy_counts": dict(best_counter),
        "best_policy_by_family": {family: dict(counter) for family, counter in family_best.items()},
        "positive_gain_policy_counts": dict(positive_gain_counter),
        "positive_gain_policy_by_family": {family: dict(counter) for family, counter in family_positive_gain.items()},
        "default_mean_utility": float(default_utility_sum / max(1, query_count)),
        "oracle_mean_positive_gain": float(oracle_gain_sum / max(1, query_count)),
    }
    if not xs:
        return np.zeros((0, len(FEATURE_NAMES)), dtype=np.float32), np.zeros((0,), dtype=np.float32), stats
    return np.vstack(xs).astype(np.float32), np.asarray(ys, dtype=np.float32), stats


def select_with_model(
    row: dict,
    *,
    model: object,
    method: str,
    topk: int,
    pool_k: int,
    policy_labels: list[str],
    switch_margin: float,
    weights: dict[str, float],
    score_mode: str = "utility",
) -> tuple[str, float, float, float]:
    base = ranking_for_row(row, method)
    qrels = qrels_for_row(row)
    candidates = candidate_rankings(
        base,
        query_id=str(row.get("query_id", "")),
        topk=int(topk),
        pool_k=int(pool_k),
        policy_labels=policy_labels,
    )
    pred_scores: dict[str, float] = {}
    true_utilities: dict[str, float] = {}
    for policy, ranked in candidates.items():
        feat = build_selector_features(
            query_text=str(row.get("query", "")),
            query_id=str(row.get("query_id", "")),
            base_ranked_doc_ids=base,
            candidate_ranked_doc_ids=ranked,
            policy=policy,
            trace=row.get("trace", {}),
        ).reshape(1, -1)
        pred_scores[policy] = float(np.asarray(model.predict(feat)).reshape(-1)[0])
        true_utilities[policy] = retrieval_utility(
            ranked,
            qrels,
            topk=int(topk),
            ndcg1_weight=float(weights["ndcg1"]),
            ndcg5_weight=float(weights["ndcg5"]),
            map5_weight=float(weights["map5"]),
            hits5_weight=float(weights["hits5"]),
        )
    if str(score_mode) == "gain":
        pred_scores["keep_current"] = 0.0
    default_pred = float(pred_scores.get("keep_current", 0.0))
    best_policy = max(policy_labels, key=lambda label: (pred_scores.get(label, 0.0), -policy_labels.index(label)))
    if best_policy != "keep_current" and pred_scores[best_policy] - default_pred >= float(switch_margin):
        chosen = best_policy
    else:
        chosen = "keep_current"
    return (
        chosen,
        float(true_utilities.get(chosen, 0.0)),
        float(true_utilities.get("keep_current", 0.0)),
        float(max(true_utilities.values()) if true_utilities else 0.0),
    )


def evaluate_selector(
    rows: list[dict],
    *,
    model: object,
    method: str,
    topk: int,
    pool_k: int,
    policy_labels: list[str],
    switch_margin: float,
    weights: dict[str, float],
    score_mode: str = "utility",
) -> dict:
    chosen_counter: Counter[str] = Counter()
    family_counter: dict[str, Counter[str]] = defaultdict(Counter)
    chosen_utilities = []
    default_utilities = []
    oracle_utilities = []
    positive_switch = 0
    negative_switch = 0
    switched = 0
    usable = 0
    for row in rows:
        if not qrels_for_row(row) or len(ranking_for_row(row, method)) < max(1, int(topk)):
            continue
        usable += 1
        policy, chosen_u, default_u, oracle_u = select_with_model(
            row,
            model=model,
            method=method,
            topk=topk,
            pool_k=pool_k,
            policy_labels=policy_labels,
            switch_margin=switch_margin,
            weights=weights,
            score_mode=score_mode,
        )
        chosen_counter[policy] += 1
        family_counter[query_family(str(row.get("query_id", "")))][policy] += 1
        chosen_utilities.append(chosen_u)
        default_utilities.append(default_u)
        oracle_utilities.append(oracle_u)
        if policy != "keep_current":
            switched += 1
            positive_switch += int(chosen_u > default_u + 1e-12)
            negative_switch += int(chosen_u < default_u - 1e-12)
    mean_chosen = float(np.mean(chosen_utilities)) if chosen_utilities else 0.0
    mean_default = float(np.mean(default_utilities)) if default_utilities else 0.0
    mean_oracle = float(np.mean(oracle_utilities)) if oracle_utilities else 0.0
    return {
        "queries": int(usable),
        "switch_margin": float(switch_margin),
        "chosen_mean_utility": mean_chosen,
        "default_mean_utility": mean_default,
        "oracle_mean_utility": mean_oracle,
        "mean_gain_vs_default": float(mean_chosen - mean_default),
        "oracle_gap": float(mean_oracle - mean_chosen),
        "switched": int(switched),
        "positive_switch": int(positive_switch),
        "negative_switch": int(negative_switch),
        "chosen_policy_counts": dict(chosen_counter),
        "chosen_policy_by_family": {family: dict(counter) for family, counter in family_counter.items()},
    }


def precompute_eval_records(
    rows: list[dict],
    *,
    model: object,
    risk_model: object | None = None,
    method: str,
    topk: int,
    pool_k: int,
    policy_labels: list[str],
    weights: dict[str, float],
    score_mode: str = "utility",
) -> list[dict]:
    records: list[dict] = []
    for row in rows:
        base = ranking_for_row(row, method)
        qrels = qrels_for_row(row)
        if len(base) < max(1, int(topk)) or not qrels:
            continue
        candidates = candidate_rankings(
            base,
            query_id=str(row.get("query_id", "")),
            topk=int(topk),
            pool_k=int(pool_k),
            policy_labels=policy_labels,
        )
        features = []
        true_utilities = []
        for policy in policy_labels:
            ranked = candidates[policy]
            features.append(
                build_selector_features(
                    query_text=str(row.get("query", "")),
                    query_id=str(row.get("query_id", "")),
                    base_ranked_doc_ids=base,
                    candidate_ranked_doc_ids=ranked,
                    policy=policy,
                    trace=row.get("trace", {}),
                )
            )
            true_utilities.append(
                retrieval_utility(
                    ranked,
                    qrels,
                    topk=int(topk),
                    ndcg1_weight=float(weights["ndcg1"]),
                    ndcg5_weight=float(weights["ndcg5"]),
                    map5_weight=float(weights["map5"]),
                    hits5_weight=float(weights["hits5"]),
                )
            )
        pred_scores = np.asarray(model.predict(np.vstack(features).astype(np.float32)), dtype=np.float32).reshape(-1)
        if str(score_mode) == "gain":
            default_idx = policy_labels.index("keep_current") if "keep_current" in policy_labels else 0
            pred_scores[default_idx] = 0.0
        positive_probs = np.ones((len(policy_labels),), dtype=np.float32)
        if risk_model is not None:
            feature_matrix = np.vstack(features).astype(np.float32)
            if hasattr(risk_model, "predict_proba"):
                probs = np.asarray(risk_model.predict_proba(feature_matrix), dtype=np.float32)
                if probs.ndim == 2 and probs.shape[1] >= 2:
                    positive_probs = probs[:, 1].reshape(-1)
                else:
                    positive_probs = probs.reshape(-1)
            else:
                positive_probs = np.asarray(risk_model.predict(feature_matrix), dtype=np.float32).reshape(-1)
            default_idx = policy_labels.index("keep_current") if "keep_current" in policy_labels else 0
            positive_probs[default_idx] = 1.0
        records.append(
            {
                "family": query_family(str(row.get("query_id", ""))),
                "pred_scores": pred_scores,
                "positive_probs": positive_probs,
                "true_utilities": np.asarray(true_utilities, dtype=np.float32),
            }
        )
    return records


def evaluate_selector_records(
    records: list[dict],
    *,
    policy_labels: list[str],
    switch_margin: float,
) -> dict:
    chosen_counter: Counter[str] = Counter()
    family_counter: dict[str, Counter[str]] = defaultdict(Counter)
    chosen_utilities = []
    default_utilities = []
    oracle_utilities = []
    positive_switch = 0
    negative_switch = 0
    switched = 0
    default_idx = policy_labels.index("keep_current") if "keep_current" in policy_labels else 0
    for record in records:
        pred = np.asarray(record["pred_scores"], dtype=np.float32)
        true = np.asarray(record["true_utilities"], dtype=np.float32)
        best_idx = int(np.argmax(pred))
        default_pred = float(pred[default_idx])
        if best_idx != default_idx and float(pred[best_idx]) - default_pred >= float(switch_margin):
            chosen_idx = best_idx
        else:
            chosen_idx = default_idx
        policy = policy_labels[chosen_idx]
        chosen_u = float(true[chosen_idx])
        default_u = float(true[default_idx])
        oracle_u = float(np.max(true))
        family = str(record.get("family", "other"))
        chosen_counter[policy] += 1
        family_counter[family][policy] += 1
        chosen_utilities.append(chosen_u)
        default_utilities.append(default_u)
        oracle_utilities.append(oracle_u)
        if chosen_idx != default_idx:
            switched += 1
            positive_switch += int(chosen_u > default_u + 1e-12)
            negative_switch += int(chosen_u < default_u - 1e-12)
    mean_chosen = float(np.mean(chosen_utilities)) if chosen_utilities else 0.0
    mean_default = float(np.mean(default_utilities)) if default_utilities else 0.0
    mean_oracle = float(np.mean(oracle_utilities)) if oracle_utilities else 0.0
    return {
        "queries": int(len(records)),
        "switch_margin": float(switch_margin),
        "chosen_mean_utility": mean_chosen,
        "default_mean_utility": mean_default,
        "oracle_mean_utility": mean_oracle,
        "mean_gain_vs_default": float(mean_chosen - mean_default),
        "oracle_gap": float(mean_oracle - mean_chosen),
        "switched": int(switched),
        "positive_switch": int(positive_switch),
        "negative_switch": int(negative_switch),
        "chosen_policy_counts": dict(chosen_counter),
        "chosen_policy_by_family": {family: dict(counter) for family, counter in family_counter.items()},
    }


def _required_margin_for_record(
    *,
    family: str,
    policy: str,
    fallback_margin: float,
    policy_switch_margins: dict[str, float] | None = None,
    family_switch_margins: dict[str, float] | None = None,
    family_policy_thresholds: dict[str, dict[str, float]] | None = None,
) -> float:
    required = float(fallback_margin)
    if policy_switch_margins and policy in policy_switch_margins:
        required = max(required, float(policy_switch_margins[policy]))
    if family_switch_margins and family in family_switch_margins:
        required = max(required, float(family_switch_margins[family]))
    if family_policy_thresholds:
        family_thresholds = family_policy_thresholds.get(family, {}) or {}
        if policy in family_thresholds:
            required = max(required, float(family_thresholds[policy]))
    return required


def calibrated_choice_index(
    record: dict,
    *,
    policy_labels: list[str],
    switch_margin: float,
    positive_prob_threshold: float = 0.0,
    policy_switch_margins: dict[str, float] | None = None,
    family_switch_margins: dict[str, float] | None = None,
    family_policy_thresholds: dict[str, dict[str, float]] | None = None,
) -> int:
    pred = np.asarray(record["pred_scores"], dtype=np.float32)
    positive_probs = np.asarray(record.get("positive_probs", np.ones_like(pred)), dtype=np.float32)
    default_idx = policy_labels.index("keep_current") if "keep_current" in policy_labels else 0
    default_pred = float(pred[default_idx])
    family = str(record.get("family", "other"))
    eligible: list[tuple[float, float, int, int]] = []
    for idx, policy in enumerate(policy_labels):
        if idx == default_idx:
            continue
        margin = float(pred[idx]) - default_pred
        required = _required_margin_for_record(
            family=family,
            policy=policy,
            fallback_margin=float(switch_margin),
            policy_switch_margins=policy_switch_margins,
            family_switch_margins=family_switch_margins,
            family_policy_thresholds=family_policy_thresholds,
        )
        if margin >= required:
            if float(positive_probs[idx]) >= float(positive_prob_threshold):
                eligible.append((float(pred[idx]), margin, -idx, idx))
    if not eligible:
        return default_idx
    return int(max(eligible)[-1])


def evaluate_selector_records_calibrated(
    records: list[dict],
    *,
    policy_labels: list[str],
    switch_margin: float,
    positive_prob_threshold: float = 0.0,
    policy_switch_margins: dict[str, float] | None = None,
    family_switch_margins: dict[str, float] | None = None,
    family_policy_thresholds: dict[str, dict[str, float]] | None = None,
) -> dict:
    chosen_counter: Counter[str] = Counter()
    family_counter: dict[str, Counter[str]] = defaultdict(Counter)
    chosen_utilities = []
    default_utilities = []
    oracle_utilities = []
    positive_switch = 0
    negative_switch = 0
    switched = 0
    default_idx = policy_labels.index("keep_current") if "keep_current" in policy_labels else 0
    for record in records:
        true = np.asarray(record["true_utilities"], dtype=np.float32)
        chosen_idx = calibrated_choice_index(
            record,
            policy_labels=policy_labels,
            switch_margin=float(switch_margin),
            positive_prob_threshold=float(positive_prob_threshold),
            policy_switch_margins=policy_switch_margins,
            family_switch_margins=family_switch_margins,
            family_policy_thresholds=family_policy_thresholds,
        )
        policy = policy_labels[chosen_idx]
        chosen_u = float(true[chosen_idx])
        default_u = float(true[default_idx])
        oracle_u = float(np.max(true))
        family = str(record.get("family", "other"))
        chosen_counter[policy] += 1
        family_counter[family][policy] += 1
        chosen_utilities.append(chosen_u)
        default_utilities.append(default_u)
        oracle_utilities.append(oracle_u)
        if chosen_idx != default_idx:
            switched += 1
            positive_switch += int(chosen_u > default_u + 1e-12)
            negative_switch += int(chosen_u < default_u - 1e-12)
    mean_chosen = float(np.mean(chosen_utilities)) if chosen_utilities else 0.0
    mean_default = float(np.mean(default_utilities)) if default_utilities else 0.0
    mean_oracle = float(np.mean(oracle_utilities)) if oracle_utilities else 0.0
    return {
        "queries": int(len(records)),
        "switch_margin": float(switch_margin),
        "positive_prob_threshold": float(positive_prob_threshold),
        "chosen_mean_utility": mean_chosen,
        "default_mean_utility": mean_default,
        "oracle_mean_utility": mean_oracle,
        "mean_gain_vs_default": float(mean_chosen - mean_default),
        "oracle_gap": float(mean_oracle - mean_chosen),
        "switched": int(switched),
        "positive_switch": int(positive_switch),
        "negative_switch": int(negative_switch),
        "chosen_policy_counts": dict(chosen_counter),
        "chosen_policy_by_family": {family: dict(counter) for family, counter in family_counter.items()},
    }


def _best_predicted_nondefault_index(pred: np.ndarray, default_idx: int) -> int:
    best_idx = default_idx
    best_score = -float("inf")
    for idx, score in enumerate(pred):
        if idx == default_idx:
            continue
        if float(score) > best_score:
            best_idx = idx
            best_score = float(score)
    return int(best_idx)


def tune_family_policy_thresholds(
    records: list[dict],
    *,
    policy_labels: list[str],
    base_margin: float,
    max_margin: float,
    min_switches: int,
    min_gain_per_switch: float,
    max_negative_rate: float,
    safety_margin: float,
) -> tuple[dict[str, dict[str, float]], dict]:
    default_idx = policy_labels.index("keep_current") if "keep_current" in policy_labels else 0
    families = sorted({str(record.get("family", "other")) for record in records})
    disabled_threshold = 1.0e9
    thresholds: dict[str, dict[str, float]] = {
        family: {policy: disabled_threshold for policy in policy_labels if policy != "keep_current"}
        for family in families
    }
    stats: dict[str, dict[str, dict]] = {}
    base_margin = float(base_margin)
    max_margin = max(float(max_margin), base_margin)
    for family in families:
        family_records = [record for record in records if str(record.get("family", "other")) == family]
        stats[family] = {}
        for policy_idx, policy in enumerate(policy_labels):
            if policy_idx == default_idx:
                continue
            events: list[tuple[float, float]] = []
            for record in family_records:
                pred = np.asarray(record["pred_scores"], dtype=np.float32)
                true = np.asarray(record["true_utilities"], dtype=np.float32)
                if _best_predicted_nondefault_index(pred, default_idx) != policy_idx:
                    continue
                margin = float(pred[policy_idx]) - float(pred[default_idx])
                if margin < base_margin:
                    continue
                gain = float(true[policy_idx]) - float(true[default_idx])
                events.append((margin, gain))
            if not events:
                stats[family][policy] = {"enabled": False, "reason": "no_predicted_best_events"}
                continue
            margins = np.asarray([m for m, _ in events], dtype=np.float32)
            quantiles = np.quantile(margins, np.linspace(0.0, 1.0, num=11)).tolist()
            grid = sorted({round(base_margin, 4), round(max_margin, 4), *[round(float(x), 4) for x in quantiles]})
            best: dict | None = None
            for threshold in grid:
                selected = [(m, gain) for m, gain in events if m >= float(threshold)]
                if len(selected) < int(min_switches):
                    continue
                gains = [gain for _, gain in selected]
                gain_sum = float(np.sum(gains))
                positive = int(sum(gain > 1e-12 for gain in gains))
                negative = int(sum(gain < -1e-12 for gain in gains))
                nonzero = positive + negative
                negative_rate = float(negative / max(1, nonzero))
                gain_per_switch = float(gain_sum / max(1, len(selected)))
                if gain_sum <= 0.0:
                    continue
                if gain_per_switch < float(min_gain_per_switch):
                    continue
                if negative_rate > float(max_negative_rate):
                    continue
                candidate = {
                    "threshold": float(threshold),
                    "selected": int(len(selected)),
                    "positive": int(positive),
                    "negative": int(negative),
                    "gain_sum": gain_sum,
                    "gain_per_switch": gain_per_switch,
                    "negative_rate": negative_rate,
                }
                current_key = (
                    candidate["gain_sum"],
                    candidate["gain_per_switch"],
                    -candidate["negative"],
                    -candidate["selected"],
                    candidate["threshold"],
                )
                best_key = (
                    best["gain_sum"],
                    best["gain_per_switch"],
                    -best["negative"],
                    -best["selected"],
                    best["threshold"],
                ) if best else None
                if best is None or current_key > best_key:
                    best = candidate
            if best is None:
                stats[family][policy] = {
                    "enabled": False,
                    "reason": "failed_risk_gain_filter",
                    "candidate_events": int(len(events)),
                }
                continue
            calibrated_threshold = min(disabled_threshold, max(base_margin, float(best["threshold"]) + float(safety_margin)))
            thresholds[family][policy] = float(calibrated_threshold)
            stats[family][policy] = {
                "enabled": True,
                "candidate_events": int(len(events)),
                "calibrated_threshold": float(calibrated_threshold),
                **best,
            }
    return thresholds, stats


def tune_margin_from_records(
    records: list[dict],
    *,
    policy_labels: list[str],
    max_margin: float,
) -> tuple[float, dict]:
    candidates = [round(x, 4) for x in np.linspace(0.0, float(max_margin), num=31)]
    best_margin = 0.0
    best_eval: dict | None = None
    for margin in candidates:
        metrics = evaluate_selector_records(records, policy_labels=policy_labels, switch_margin=float(margin))
        if best_eval is None:
            best_margin = float(margin)
            best_eval = metrics
            continue
        current_key = (
            metrics["mean_gain_vs_default"],
            -metrics["negative_switch"],
            -metrics["switched"],
        )
        best_key = (
            best_eval["mean_gain_vs_default"],
            -best_eval["negative_switch"],
            -best_eval["switched"],
        )
        if current_key > best_key:
            best_margin = float(margin)
            best_eval = metrics
    return best_margin, dict(best_eval or {})


def tune_margin_and_positive_prob_from_records(
    records: list[dict],
    *,
    policy_labels: list[str],
    max_margin: float,
    min_positive_prob: float,
    max_positive_prob: float,
    negative_penalty: float,
) -> tuple[float, float, dict]:
    margins = [round(x, 4) for x in np.linspace(0.0, float(max_margin), num=31)]
    probs = [round(x, 4) for x in np.linspace(float(min_positive_prob), float(max_positive_prob), num=17)]
    if 0.0 not in probs and float(min_positive_prob) <= 0.0:
        probs.insert(0, 0.0)
    best_margin = 0.0
    best_prob = float(min_positive_prob)
    best_eval: dict | None = None
    best_score: tuple[float, float, int, int] | None = None
    for margin in margins:
        for prob in probs:
            metrics = evaluate_selector_records_calibrated(
                records,
                policy_labels=policy_labels,
                switch_margin=float(margin),
                positive_prob_threshold=float(prob),
            )
            adjusted_gain = float(metrics["mean_gain_vs_default"]) - float(negative_penalty) * (
                float(metrics["negative_switch"]) / max(1.0, float(metrics["queries"]))
            )
            score = (
                adjusted_gain,
                float(metrics["mean_gain_vs_default"]),
                -int(metrics["negative_switch"]),
                -int(metrics["switched"]),
            )
            if best_score is None or score > best_score:
                best_score = score
                best_margin = float(margin)
                best_prob = float(prob)
                best_eval = metrics
                best_eval["risk_adjusted_gain"] = float(adjusted_gain)
                best_eval["risk_negative_penalty"] = float(negative_penalty)
    return best_margin, best_prob, dict(best_eval or {})


def tune_margin(
    rows: list[dict],
    *,
    model: object,
    method: str,
    topk: int,
    pool_k: int,
    policy_labels: list[str],
    weights: dict[str, float],
    max_margin: float,
) -> tuple[float, dict]:
    candidates = [round(x, 4) for x in np.linspace(0.0, float(max_margin), num=31)]
    best_margin = 0.0
    best_eval: dict | None = None
    for margin in candidates:
        metrics = evaluate_selector(
            rows,
            model=model,
            method=method,
            topk=topk,
            pool_k=pool_k,
            policy_labels=policy_labels,
            switch_margin=float(margin),
            weights=weights,
        )
        if best_eval is None:
            best_margin = float(margin)
            best_eval = metrics
            continue
        current_key = (
            metrics["mean_gain_vs_default"],
            -metrics["negative_switch"],
            -metrics["switched"],
        )
        best_key = (
            best_eval["mean_gain_vs_default"],
            -best_eval["negative_switch"],
            -best_eval["switched"],
        )
        if current_key > best_key:
            best_margin = float(margin)
            best_eval = metrics
    return best_margin, dict(best_eval or {})


def make_model(seed: int, model_type: str = "ridge") -> object:
    model_type = str(model_type or "ridge").lower()
    if model_type == "hist_gbdt":
        return HistGradientBoostingRegressor(
            loss="squared_error",
            learning_rate=0.045,
            max_iter=180,
            max_leaf_nodes=15,
            min_samples_leaf=12,
            l2_regularization=0.02,
            random_state=int(seed),
        )
    if model_type == "ridge":
        return Pipeline(
            steps=[
                ("scaler", StandardScaler()),
                ("regressor", Ridge(alpha=1.0)),
            ]
        )
    raise ValueError(f"Unknown CEPS model_type={model_type!r}; expected ridge or hist_gbdt")


def make_risk_model(seed: int) -> HistGradientBoostingClassifier:
    return HistGradientBoostingClassifier(
        loss="log_loss",
        learning_rate=0.045,
        max_iter=160,
        max_leaf_nodes=15,
        min_samples_leaf=12,
        l2_regularization=0.02,
        random_state=int(seed) + 17,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Train Counterfactual Evidence Policy Selector (CEPS).")
    parser.add_argument("--train-rankings-jsonl", type=Path, required=True)
    parser.add_argument("--dev-rankings-jsonl", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--run-id", type=str, default="tessera_ceps_v1")
    parser.add_argument("--method", type=str, default="tessera_rag")
    parser.add_argument("--topk", type=int, default=5)
    parser.add_argument("--pool-k", type=int, default=30)
    parser.add_argument("--policies", type=str, default=",".join(POLICY_LABELS))
    parser.add_argument("--ndcg1-weight", type=float, default=0.28)
    parser.add_argument("--ndcg5-weight", type=float, default=0.32)
    parser.add_argument("--map5-weight", type=float, default=0.28)
    parser.add_argument("--hits5-weight", type=float, default=0.12)
    parser.add_argument("--target-mode", choices=["utility", "gain"], default="utility")
    parser.add_argument("--model-type", choices=["ridge", "hist_gbdt"], default="ridge")
    parser.add_argument("--use-positive-risk-model", action="store_true")
    parser.add_argument("--positive-label-eps", type=float, default=1e-8)
    parser.add_argument("--min-positive-prob", type=float, default=0.50)
    parser.add_argument("--max-positive-prob", type=float, default=0.95)
    parser.add_argument("--risk-negative-penalty", type=float, default=0.02)
    parser.add_argument("--max-train", type=int, default=0)
    parser.add_argument("--max-dev", type=int, default=0)
    parser.add_argument("--max-margin", type=float, default=0.08)
    parser.add_argument("--calibrate-family-policies", action="store_true")
    parser.add_argument("--calibration-min-switches", type=int, default=1)
    parser.add_argument("--calibration-min-gain-per-switch", type=float, default=0.001)
    parser.add_argument("--calibration-max-negative-rate", type=float, default=0.0)
    parser.add_argument("--calibration-safety-margin", type=float, default=0.004)
    parser.add_argument("--seed", type=int, default=41)
    parser.add_argument("--allow-test-split-training", action="store_true")
    args = parser.parse_args()

    allow_test_training = bool(args.allow_test_split_training) or os.environ.get("TESSERA_ALLOW_TEST_SPLIT_TRAINING", "0") == "1"
    if not allow_test_training:
        for label, path in (("train-rankings-jsonl", args.train_rankings_jsonl), ("dev-rankings-jsonl", args.dev_rankings_jsonl)):
            if is_test_like_path(Path(path)):
                raise ValueError(f"{label} looks like a test split ({path}). CEPS must use train/dev only.")

    policy_labels = [x.strip() for x in str(args.policies).split(",") if x.strip()]
    unknown = [label for label in policy_labels if label not in POLICY_LABELS]
    if unknown:
        raise ValueError(f"Unknown CEPS policies: {unknown}; allowed={POLICY_LABELS}")
    if "keep_current" not in policy_labels:
        policy_labels.insert(0, "keep_current")
    weights = {
        "ndcg1": float(args.ndcg1_weight),
        "ndcg5": float(args.ndcg5_weight),
        "map5": float(args.map5_weight),
        "hits5": float(args.hits5_weight),
    }
    weight_sum = sum(weights.values())
    if weight_sum <= 0.0:
        raise ValueError("utility weights must sum to a positive value")
    weights = {key: val / weight_sum for key, val in weights.items()}

    train_rows = iter_jsonl(args.train_rankings_jsonl, int(args.max_train))
    dev_rows = iter_jsonl(args.dev_rankings_jsonl, int(args.max_dev))
    train_ids = {str(row.get("query_id", "")) for row in train_rows if str(row.get("query_id", ""))}
    dev_ids = {str(row.get("query_id", "")) for row in dev_rows if str(row.get("query_id", ""))}
    overlap = sorted(train_ids & dev_ids)
    if overlap:
        raise ValueError(f"train/dev ranking overlap detected: {len(overlap)} examples: {overlap[:5]}")

    x_train, y_train, train_stats = make_dataset(
        train_rows,
        method=str(args.method),
        topk=int(args.topk),
        pool_k=int(args.pool_k),
        policy_labels=policy_labels,
        weights=weights,
        target_mode=str(args.target_mode),
    )
    x_dev, y_dev, dev_stats = make_dataset(
        dev_rows,
        method=str(args.method),
        topk=int(args.topk),
        pool_k=int(args.pool_k),
        policy_labels=policy_labels,
        weights=weights,
        target_mode=str(args.target_mode),
    )
    if x_train.size == 0 or x_dev.size == 0:
        raise RuntimeError("CEPS train/dev data is empty")

    print(f"[stage] train examples={x_train.shape[0]} dev examples={x_dev.shape[0]}", flush=True)
    model = make_model(int(args.seed), str(args.model_type))
    model.fit(x_train, y_train)
    risk_model = None
    if bool(args.use_positive_risk_model):
        if str(args.target_mode) != "gain":
            raise ValueError("--use-positive-risk-model requires --target-mode gain")
        y_risk = (np.asarray(y_train, dtype=np.float32) > float(args.positive_label_eps)).astype(np.int32)
        if len(set(y_risk.tolist())) < 2:
            print("[warn] risk model disabled because train labels have one class", flush=True)
        else:
            risk_model = make_risk_model(int(args.seed))
            risk_model.fit(x_train, y_risk)
    dev_pred = np.asarray(model.predict(x_dev), dtype=np.float32)
    dev_eval_records = precompute_eval_records(
        dev_rows,
        model=model,
        risk_model=risk_model,
        method=str(args.method),
        topk=int(args.topk),
        pool_k=int(args.pool_k),
        policy_labels=policy_labels,
        weights=weights,
        score_mode=str(args.target_mode),
    )
    recommended_positive_prob_threshold = 0.0
    if risk_model is not None:
        recommended_margin, recommended_positive_prob_threshold, dev_policy_eval = tune_margin_and_positive_prob_from_records(
            dev_eval_records,
            policy_labels=policy_labels,
            max_margin=float(args.max_margin),
            min_positive_prob=float(args.min_positive_prob),
            max_positive_prob=float(args.max_positive_prob),
            negative_penalty=float(args.risk_negative_penalty),
        )
    else:
        recommended_margin, dev_policy_eval = tune_margin_from_records(
            dev_eval_records,
            policy_labels=policy_labels,
            max_margin=float(args.max_margin),
        )
    family_policy_thresholds: dict[str, dict[str, float]] = {}
    family_policy_calibration: dict = {}
    dev_policy_eval_calibrated: dict = dict(dev_policy_eval)
    if bool(args.calibrate_family_policies):
        family_policy_thresholds, family_policy_calibration = tune_family_policy_thresholds(
            dev_eval_records,
            policy_labels=policy_labels,
            base_margin=float(recommended_margin),
            max_margin=float(args.max_margin),
            min_switches=int(args.calibration_min_switches),
            min_gain_per_switch=float(args.calibration_min_gain_per_switch),
            max_negative_rate=float(args.calibration_max_negative_rate),
            safety_margin=float(args.calibration_safety_margin),
        )
        dev_policy_eval_calibrated = evaluate_selector_records_calibrated(
            dev_eval_records,
            policy_labels=policy_labels,
            switch_margin=float(recommended_margin),
            positive_prob_threshold=float(recommended_positive_prob_threshold),
            family_policy_thresholds=family_policy_thresholds,
        )

    metrics = {
        "method_name": "Counterfactual Evidence Policy Selector",
        "method_formulation": (
            "learns U(policy | query, current evidence set) from train/dev counterfactual rankings "
            "and switches policy only when predicted utility improves over keep_current; optional "
            "family-policy calibration treats each source-action as a risk-controlled decision; "
            "gain target learns delta utility over keep_current directly"
        ),
        "score_mode": str(args.target_mode),
        "model_type": str(args.model_type),
        "risk_model_type": "hist_gbdt_classifier" if risk_model is not None else None,
        "run_id": str(args.run_id),
        "feature_names": FEATURE_NAMES,
        "policy_labels": policy_labels,
        "recommended_switch_margin": float(recommended_margin),
        "recommended_positive_prob_threshold": float(recommended_positive_prob_threshold),
        "family_policy_thresholds": family_policy_thresholds,
        "family_policy_calibration": family_policy_calibration,
        "utility_weights": weights,
        "split_guard": {
            "train_rankings_jsonl": str(args.train_rankings_jsonl),
            "dev_rankings_jsonl": str(args.dev_rankings_jsonl),
            "train_queries": int(len(train_ids)),
            "dev_queries": int(len(dev_ids)),
            "train_dev_overlap": int(len(overlap)),
            "test_like_paths_allowed": bool(allow_test_training),
        },
        "train": train_stats,
        "dev": dev_stats,
        "dev_regression": {
            "mae": float(mean_absolute_error(y_dev, dev_pred)),
            "r2": float(r2_score(y_dev, dev_pred)),
        },
        "dev_policy_eval": dev_policy_eval,
        "dev_policy_eval_calibrated": dev_policy_eval_calibrated,
        "config": {
            "method": str(args.method),
            "topk": int(args.topk),
            "pool_k": int(args.pool_k),
            "target_mode": str(args.target_mode),
            "model_type": str(args.model_type),
            "use_positive_risk_model": bool(args.use_positive_risk_model),
            "positive_label_eps": float(args.positive_label_eps),
            "min_positive_prob": float(args.min_positive_prob),
            "max_positive_prob": float(args.max_positive_prob),
            "risk_negative_penalty": float(args.risk_negative_penalty),
            "max_margin": float(args.max_margin),
            "calibrate_family_policies": bool(args.calibrate_family_policies),
            "calibration_min_switches": int(args.calibration_min_switches),
            "calibration_min_gain_per_switch": float(args.calibration_min_gain_per_switch),
            "calibration_max_negative_rate": float(args.calibration_max_negative_rate),
            "calibration_safety_margin": float(args.calibration_safety_margin),
            "seed": int(args.seed),
        },
    }

    bundle = CounterfactualPolicySelectorBundle(
        model=model,
        risk_model=risk_model,
        feature_names=list(FEATURE_NAMES),
        policy_labels=policy_labels,
        metadata=metrics,
        default_policy="keep_current",
        topk=int(args.topk),
        pool_k=int(args.pool_k),
        switch_margin=float(recommended_margin),
        positive_prob_threshold=float(recommended_positive_prob_threshold),
        family_policy_thresholds=family_policy_thresholds,
    )
    args.out_dir.mkdir(parents=True, exist_ok=True)
    bundle_path = args.out_dir / "counterfactual_policy_selector.pkl"
    metrics_path = args.out_dir / f"{args.run_id}_metrics.json"
    save_counterfactual_policy_selector_bundle(bundle, bundle_path)
    metrics_path.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(metrics, ensure_ascii=False, indent=2))
    print(f"[OK] bundle -> {bundle_path}")
    print(f"[OK] metrics -> {metrics_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
