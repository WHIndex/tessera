#!/usr/bin/env python3
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
import os
from pathlib import Path
import sys

import numpy as np
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from tessera_exp.e2e.source_action_policy import (  # noqa: E402
    ACTION_LABELS,
    FEATURE_NAMES,
    UTILITY_FEATURE_NAMES,
    SourceUtilityGateBundle,
    apply_source_action_to_doc_ids,
    build_action_utility_features,
    build_policy_features,
    retrieval_score,
    save_source_action_policy_bundle,
    source_action_allowed,
)


def is_test_like_path(path: Path) -> bool:
    raw = str(path).lower()
    return "test" in path.name.lower() or "/test" in raw or "\\test" in raw


def iter_jsonl(path: Path, max_examples: int = 0) -> list[dict]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            rows.append(json.loads(line))
            if int(max_examples) > 0 and len(rows) >= int(max_examples):
                break
    return rows


def qrels_for_row(row: dict) -> dict[str, float]:
    out = {}
    for doc_id, raw in (row.get("qrels", {}) or {}).items():
        try:
            val = float(raw)
        except Exception:
            continue
        if val > 0.0:
            out[str(doc_id)] = val
    return out


def ranking_for_row(row: dict, method: str) -> list[str]:
    return [str(x) for x in (row.get("rankings", {}) or {}).get(method, [])]


def action_deltas_for_row(row: dict, *, method: str, topk: int, pool_k: int) -> tuple[list[str], dict[str, float]]:
    ranked = ranking_for_row(row, method)
    qrels = qrels_for_row(row)
    base_score = retrieval_score(ranked, qrels, topk=topk)
    deltas = {}
    for action in ACTION_LABELS:
        acted = apply_source_action_to_doc_ids(action, ranked, topk=topk, pool_k=pool_k)
        deltas[action] = retrieval_score(acted, qrels, topk=topk) - base_score
    return ranked, deltas


def build_matrix(
    rows: list[dict],
    *,
    method: str,
    topk: int,
    pool_k: int,
    hard_disabled: set[str],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
    xs = []
    ys = []
    weights = []
    oracle_counts = Counter()
    oracle_gain_sum = 0.0
    usable = 0
    for row in rows:
        ranked = ranking_for_row(row, method)
        qrels = qrels_for_row(row)
        if len(ranked) < max(1, int(topk)) or not qrels:
            continue
        _, deltas = action_deltas_for_row(row, method=method, topk=topk, pool_k=pool_k)
        base_features = build_policy_features(
            query_text=str(row.get("query", "")),
            query_id=str(row.get("query_id", "")),
            ranked_doc_ids=ranked,
            trace=row.get("trace", {}),
        )
        best_action = max(ACTION_LABELS, key=lambda action: (deltas[action], -ACTION_LABELS.index(action)))
        best_gain = max(0.0, float(deltas.get(best_action, 0.0)))
        if best_gain <= 0.0:
            best_action = "keep_current"
        oracle_counts[best_action] += 1
        oracle_gain_sum += best_gain
        usable += 1
        for action in ACTION_LABELS:
            if action == "keep_current" or action in hard_disabled:
                continue
            xs.append(build_action_utility_features(base_features, action, ACTION_LABELS))
            delta = float(deltas[action])
            ys.append(delta)
            weights.append(1.0 + 10.0 * abs(delta) + (4.0 if delta > 0.0 else 0.0))
    stats = {
        "usable": int(usable),
        "oracle_action_counts": dict(oracle_counts),
        "oracle_mean_positive_gain": float(oracle_gain_sum / max(1, usable)),
        "utility_examples": int(len(xs)),
    }
    if not xs:
        return (
            np.zeros((0, len(UTILITY_FEATURE_NAMES)), dtype=np.float32),
            np.zeros((0,), dtype=np.float32),
            np.zeros((0,), dtype=np.float32),
            stats,
        )
    return np.vstack(xs).astype(np.float32), np.asarray(ys, dtype=np.float32), np.asarray(weights, dtype=np.float32), stats


def calibrate_thresholds(
    rows: list[dict],
    *,
    model: object,
    method: str,
    topk: int,
    pool_k: int,
    min_pred_gain: float,
    min_dev_support: int,
    min_precision: float,
    min_mean_gain: float,
    hard_disabled: set[str],
    protect_kg: bool,
) -> tuple[dict[str, float], dict[str, dict]]:
    by_action: dict[str, list[tuple[float, float]]] = defaultdict(list)
    for row in rows:
        ranked = ranking_for_row(row, method)
        if len(ranked) < max(1, int(topk)) or not qrels_for_row(row):
            continue
        _, deltas = action_deltas_for_row(row, method=method, topk=topk, pool_k=pool_k)
        base_features = build_policy_features(
            query_text=str(row.get("query", "")),
            query_id=str(row.get("query_id", "")),
            ranked_doc_ids=ranked,
            trace=row.get("trace", {}),
        )
        for action in ACTION_LABELS:
            if action == "keep_current" or action in hard_disabled:
                continue
            if not source_action_allowed(
                action,
                query_id=str(row.get("query_id", "")),
                ranked_doc_ids=ranked,
                trace=row.get("trace", {}),
                protect_kg=protect_kg,
            ):
                continue
            x = build_action_utility_features(base_features, action, ACTION_LABELS).reshape(1, -1)
            pred = float(np.asarray(model.predict(x)).reshape(-1)[0])
            by_action[action].append((pred, float(deltas[action])))

    thresholds: dict[str, float] = {}
    stats: dict[str, dict] = {}
    for action in ACTION_LABELS:
        if action == "keep_current":
            continue
        pairs = by_action.get(action, [])
        if not pairs:
            thresholds[action] = float("inf")
            stats[action] = {"enabled": False, "reason": "no_dev_candidates", "candidates": 0}
            continue
        preds = np.asarray([p for p, _ in pairs], dtype=np.float32)
        deltas = np.asarray([d for _, d in pairs], dtype=np.float32)
        candidate_thresholds = sorted(
            set([float(min_pred_gain)] + [float(np.quantile(preds, q)) for q in np.linspace(0.50, 0.98, 13)]),
            reverse=True,
        )
        best = None
        for threshold in candidate_thresholds:
            selected = preds >= max(float(min_pred_gain), float(threshold))
            support = int(selected.sum())
            if support < int(min_dev_support):
                continue
            selected_delta = deltas[selected]
            positive = int((selected_delta > 1e-9).sum())
            negative = int((selected_delta < -1e-9).sum())
            precision = positive / max(1, support)
            mean_gain = float(selected_delta.mean())
            total_gain = float(selected_delta.sum())
            if precision < float(min_precision) or mean_gain < float(min_mean_gain) or total_gain <= 0.0:
                continue
            score = total_gain + 0.25 * mean_gain - 0.02 * negative
            if best is None or score > best["score"]:
                best = {
                    "score": float(score),
                    "threshold": float(max(float(min_pred_gain), float(threshold))),
                    "support": support,
                    "precision": float(precision),
                    "mean_gain": mean_gain,
                    "total_gain": total_gain,
                    "positive": positive,
                    "negative": negative,
                }
        if best is None:
            thresholds[action] = float("inf")
            stats[action] = {
                "enabled": False,
                "reason": "failed_dev_safety",
                "candidates": int(len(pairs)),
                "mean_actual_delta": float(deltas.mean()),
                "mean_predicted_delta": float(preds.mean()),
            }
        else:
            thresholds[action] = float(best["threshold"])
            stats[action] = {"enabled": True, "candidates": int(len(pairs)), **best}
    return thresholds, stats


def evaluate_bundle(rows: list[dict], *, bundle: SourceUtilityGateBundle, method: str, topk: int, pool_k: int) -> dict:
    applied = 0
    positive = 0
    negative = 0
    gain_sum = 0.0
    action_counts = Counter()
    for row in rows:
        ranked = ranking_for_row(row, method)
        qrels = qrels_for_row(row)
        if len(ranked) < max(1, int(topk)) or not qrels:
            continue
        pred = bundle.predict(
            query_text=str(row.get("query", "")),
            query_id=str(row.get("query_id", "")),
            ranked_doc_ids=ranked,
            trace=row.get("trace", {}),
        )
        if pred.action == "keep_current":
            continue
        acted = apply_source_action_to_doc_ids(pred.action, ranked, topk=topk, pool_k=pool_k)
        delta = retrieval_score(acted, qrels, topk=topk) - retrieval_score(ranked, qrels, topk=topk)
        applied += 1
        gain_sum += float(delta)
        positive += int(delta > 1e-9)
        negative += int(delta < -1e-9)
        action_counts[pred.action] += 1
    return {
        "applied": int(applied),
        "mean_gain_all_queries": float(gain_sum / max(1, len(rows))),
        "mean_gain_applied": float(gain_sum / max(1, applied)),
        "positive_applied": int(positive),
        "negative_applied": int(negative),
        "action_counts": dict(action_counts),
    }


def json_safe(obj):
    if isinstance(obj, dict):
        return {str(k): json_safe(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [json_safe(v) for v in obj]
    if isinstance(obj, tuple):
        return [json_safe(v) for v in obj]
    if isinstance(obj, float):
        return obj if np.isfinite(obj) else None
    return obj


def main() -> int:
    parser = argparse.ArgumentParser(description="Train conservative source utility gate from train/dev rankings.")
    parser.add_argument("--train-rankings-jsonl", type=Path, required=True)
    parser.add_argument("--dev-rankings-jsonl", type=Path, required=True)
    parser.add_argument("--out-bundle", type=Path, required=True)
    parser.add_argument("--out-metrics", type=Path, required=True)
    parser.add_argument("--method", type=str, default="tessera_rag")
    parser.add_argument("--topk", type=int, default=5)
    parser.add_argument("--pool-k", type=int, default=10)
    parser.add_argument("--min-pred-gain", type=float, default=0.012)
    parser.add_argument("--min-dev-support", type=int, default=8)
    parser.add_argument("--min-precision", type=float, default=0.62)
    parser.add_argument("--min-mean-gain", type=float, default=0.004)
    parser.add_argument("--hard-disabled-actions", type=str, default="suppress_kg_top5")
    parser.add_argument("--no-protect-kg", action="store_true")
    parser.add_argument("--max-train", type=int, default=0)
    parser.add_argument("--max-dev", type=int, default=0)
    parser.add_argument("--seed", type=int, default=43)
    parser.add_argument("--allow-test-split-training", action="store_true")
    args = parser.parse_args()

    allow_test_training = bool(args.allow_test_split_training) or os.environ.get("TESSERA_ALLOW_TEST_SPLIT_TRAINING", "0") == "1"
    if not allow_test_training:
        for label, path in (("train-rankings-jsonl", args.train_rankings_jsonl), ("dev-rankings-jsonl", args.dev_rankings_jsonl)):
            if is_test_like_path(Path(path)):
                raise ValueError(f"{label} looks like a test split ({path}). Source utility gate must use train/dev only.")

    hard_disabled = {x.strip() for x in str(args.hard_disabled_actions).split(",") if x.strip()}
    train_rows = iter_jsonl(args.train_rankings_jsonl, int(args.max_train))
    dev_rows = iter_jsonl(args.dev_rankings_jsonl, int(args.max_dev))
    train_ids = {str(row.get("query_id", "")) for row in train_rows if str(row.get("query_id", ""))}
    dev_ids = {str(row.get("query_id", "")) for row in dev_rows if str(row.get("query_id", ""))}
    overlap = sorted(train_ids & dev_ids)
    if overlap:
        raise ValueError(f"train/dev ranking overlap detected: {len(overlap)} examples: {overlap[:5]}")

    x_train, y_train, sample_weight, train_stats = build_matrix(
        train_rows,
        method=str(args.method),
        topk=int(args.topk),
        pool_k=int(args.pool_k),
        hard_disabled=hard_disabled,
    )
    x_dev, y_dev, _, dev_stats = build_matrix(
        dev_rows,
        method=str(args.method),
        topk=int(args.topk),
        pool_k=int(args.pool_k),
        hard_disabled=hard_disabled,
    )
    if x_train.size == 0 or x_dev.size == 0:
        raise RuntimeError("source utility gate data is empty")

    print(f"[stage] utility train examples={x_train.shape[0]} dev examples={x_dev.shape[0]}", flush=True)
    model = GradientBoostingRegressor(
        random_state=int(args.seed),
        n_estimators=220,
        learning_rate=0.045,
        max_depth=3,
        subsample=0.85,
    )
    model.fit(x_train, y_train, sample_weight=sample_weight)
    dev_pred = model.predict(x_dev)
    thresholds, action_stats = calibrate_thresholds(
        dev_rows,
        model=model,
        method=str(args.method),
        topk=int(args.topk),
        pool_k=int(args.pool_k),
        min_pred_gain=float(args.min_pred_gain),
        min_dev_support=int(args.min_dev_support),
        min_precision=float(args.min_precision),
        min_mean_gain=float(args.min_mean_gain),
        hard_disabled=hard_disabled,
        protect_kg=not bool(args.no_protect_kg),
    )
    metadata = {
        "method_name": "Conservative Source Utility Gate",
        "method_formulation": "predicts per-action counterfactual retrieval utility and only executes actions that pass dev-set safety calibration",
        "feature_names": FEATURE_NAMES,
        "utility_feature_names": UTILITY_FEATURE_NAMES,
        "action_labels": ACTION_LABELS,
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
        "dev_regression_mae": float(mean_absolute_error(y_dev, dev_pred)),
        "action_thresholds": thresholds,
        "action_stats": action_stats,
        "config": {
            "method": str(args.method),
            "topk": int(args.topk),
            "pool_k": int(args.pool_k),
            "min_pred_gain": float(args.min_pred_gain),
            "min_dev_support": int(args.min_dev_support),
            "min_precision": float(args.min_precision),
            "min_mean_gain": float(args.min_mean_gain),
            "hard_disabled_actions": sorted(hard_disabled),
            "protect_kg": not bool(args.no_protect_kg),
            "seed": int(args.seed),
        },
    }
    bundle = SourceUtilityGateBundle(
        model=model,
        feature_names=list(FEATURE_NAMES),
        utility_feature_names=list(UTILITY_FEATURE_NAMES),
        action_labels=list(ACTION_LABELS),
        action_thresholds=thresholds,
        action_stats=action_stats,
        metadata=metadata,
        min_gain=float(args.min_pred_gain),
        protect_kg=not bool(args.no_protect_kg),
    )
    metadata["dev_policy_eval"] = evaluate_bundle(
        dev_rows,
        bundle=bundle,
        method=str(args.method),
        topk=int(args.topk),
        pool_k=int(args.pool_k),
    )
    save_source_action_policy_bundle(bundle, args.out_bundle)
    args.out_metrics.parent.mkdir(parents=True, exist_ok=True)
    metrics_json = json_safe(metadata)
    args.out_metrics.write_text(json.dumps(metrics_json, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(metrics_json, ensure_ascii=False, indent=2))
    print(f"[OK] bundle -> {args.out_bundle}")
    print(f"[OK] metrics -> {args.out_metrics}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
