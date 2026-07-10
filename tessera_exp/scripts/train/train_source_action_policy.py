#!/usr/bin/env python3
from __future__ import annotations

import argparse
from collections import Counter
import json
import os
from pathlib import Path
import sys

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from tessera_exp.e2e.source_action_policy import (  # noqa: E402
    ACTION_LABELS,
    FEATURE_NAMES,
    SourceActionPolicyBundle,
    apply_source_action_to_doc_ids,
    build_policy_features,
    retrieval_score,
    save_source_action_policy_bundle,
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
    rankings = row.get("rankings", {}) or {}
    return [str(x) for x in rankings.get(method, [])]


def best_action_for_row(
    row: dict,
    *,
    method: str,
    topk: int,
    pool_k: int,
    min_gain: float,
) -> tuple[str, float, dict[str, float]]:
    ranked = ranking_for_row(row, method)
    qrels = qrels_for_row(row)
    base_score = retrieval_score(ranked, qrels, topk=topk)
    gains: dict[str, float] = {}
    for action in ACTION_LABELS:
        acted = apply_source_action_to_doc_ids(action, ranked, topk=topk, pool_k=pool_k)
        gains[action] = retrieval_score(acted, qrels, topk=topk) - base_score
    best = max(ACTION_LABELS, key=lambda action: (gains[action], -ACTION_LABELS.index(action)))
    best_gain = float(gains[best])
    if best == "keep_current" or best_gain < float(min_gain):
        return "keep_current", best_gain, gains
    return best, best_gain, gains


def build_dataset(
    rows: list[dict],
    *,
    method: str,
    topk: int,
    pool_k: int,
    min_gain: float,
) -> tuple[np.ndarray, np.ndarray, dict]:
    xs = []
    ys = []
    action_counter = Counter()
    oracle_gain_sum = 0.0
    usable = 0
    for row in rows:
        ranked = ranking_for_row(row, method)
        if len(ranked) < max(1, int(topk)) or not qrels_for_row(row):
            continue
        action, gain, _ = best_action_for_row(
            row,
            method=method,
            topk=int(topk),
            pool_k=int(pool_k),
            min_gain=float(min_gain),
        )
        feat = build_policy_features(
            query_text=str(row.get("query", "")),
            query_id=str(row.get("query_id", "")),
            ranked_doc_ids=ranked,
            trace=row.get("trace", {}),
        )
        xs.append(feat)
        ys.append(action)
        action_counter[action] += 1
        oracle_gain_sum += max(0.0, float(gain))
        usable += 1
    stats = {
        "usable": int(usable),
        "action_counts": dict(action_counter),
        "oracle_mean_positive_gain": float(oracle_gain_sum / max(1, usable)),
    }
    if not xs:
        return np.zeros((0, len(FEATURE_NAMES)), dtype=np.float32), np.zeros((0,), dtype=object), stats
    return np.vstack(xs).astype(np.float32), np.asarray(ys, dtype=object), stats


def evaluate_policy(
    rows: list[dict],
    *,
    model: object,
    method: str,
    topk: int,
    pool_k: int,
    min_prob: float,
) -> dict:
    applied = 0
    gain_sum = 0.0
    positive = 0
    negative = 0
    action_counts = Counter()
    for row in rows:
        ranked = ranking_for_row(row, method)
        qrels = qrels_for_row(row)
        if len(ranked) < max(1, int(topk)) or not qrels:
            continue
        feat = build_policy_features(
            query_text=str(row.get("query", "")),
            query_id=str(row.get("query_id", "")),
            ranked_doc_ids=ranked,
            trace=row.get("trace", {}),
        ).reshape(1, -1)
        action = str(model.predict(feat)[0])
        conf = 1.0
        if hasattr(model, "predict_proba"):
            probs = model.predict_proba(feat)[0]
            classes = [str(x) for x in getattr(model, "classes_", [])]
            if action in classes:
                conf = float(probs[classes.index(action)])
        if action == "keep_current" or conf < float(min_prob):
            continue
        base = retrieval_score(ranked, qrels, topk=topk)
        acted = apply_source_action_to_doc_ids(action, ranked, topk=topk, pool_k=pool_k)
        delta = retrieval_score(acted, qrels, topk=topk) - base
        applied += 1
        gain_sum += float(delta)
        positive += int(delta > 1e-9)
        negative += int(delta < -1e-9)
        action_counts[action] += 1
    return {
        "applied": int(applied),
        "mean_gain_all_queries": float(gain_sum / max(1, len(rows))),
        "mean_gain_applied": float(gain_sum / max(1, applied)),
        "positive_applied": int(positive),
        "negative_applied": int(negative),
        "action_counts": dict(action_counts),
    }


def make_classifier(seed: int) -> Pipeline:
    return Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            (
                "clf",
                LogisticRegression(
                    class_weight="balanced",
                    max_iter=1500,
                    random_state=int(seed),
                    solver="lbfgs",
                ),
            ),
        ]
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Train counterfactual source-action policy from train/dev rankings.")
    parser.add_argument("--train-rankings-jsonl", type=Path, required=True)
    parser.add_argument("--dev-rankings-jsonl", type=Path, required=True)
    parser.add_argument("--out-bundle", type=Path, required=True)
    parser.add_argument("--out-metrics", type=Path, required=True)
    parser.add_argument("--method", type=str, default="tessera_rag")
    parser.add_argument("--topk", type=int, default=5)
    parser.add_argument("--pool-k", type=int, default=10)
    parser.add_argument("--min-gain", type=float, default=0.003)
    parser.add_argument("--min-prob", type=float, default=0.42)
    parser.add_argument("--max-train", type=int, default=0)
    parser.add_argument("--max-dev", type=int, default=0)
    parser.add_argument("--seed", type=int, default=41)
    parser.add_argument("--allow-test-split-training", action="store_true")
    args = parser.parse_args()

    allow_test_training = bool(args.allow_test_split_training) or os.environ.get("TESSERA_ALLOW_TEST_SPLIT_TRAINING", "0") == "1"
    if not allow_test_training:
        for label, path in (("train-rankings-jsonl", args.train_rankings_jsonl), ("dev-rankings-jsonl", args.dev_rankings_jsonl)):
            if is_test_like_path(Path(path)):
                raise ValueError(f"{label} looks like a test split ({path}). Source-action policy must use train/dev only.")

    train_rows = iter_jsonl(args.train_rankings_jsonl, int(args.max_train))
    dev_rows = iter_jsonl(args.dev_rankings_jsonl, int(args.max_dev))
    train_ids = {str(row.get("query_id", "")) for row in train_rows if str(row.get("query_id", ""))}
    dev_ids = {str(row.get("query_id", "")) for row in dev_rows if str(row.get("query_id", ""))}
    overlap = sorted(train_ids & dev_ids)
    if overlap:
        raise ValueError(f"train/dev ranking overlap detected: {len(overlap)} examples: {overlap[:5]}")

    x_train, y_train, train_stats = build_dataset(
        train_rows,
        method=str(args.method),
        topk=int(args.topk),
        pool_k=int(args.pool_k),
        min_gain=float(args.min_gain),
    )
    x_dev, y_dev, dev_stats = build_dataset(
        dev_rows,
        method=str(args.method),
        topk=int(args.topk),
        pool_k=int(args.pool_k),
        min_gain=float(args.min_gain),
    )
    if x_train.size == 0 or x_dev.size == 0:
        raise RuntimeError("source-action policy data is empty")
    if len(set(y_train.tolist())) < 2:
        raise RuntimeError("source-action policy train labels have a single class")

    print(f"[stage] train examples={x_train.shape[0]} dev examples={x_dev.shape[0]}", flush=True)
    model = make_classifier(int(args.seed))
    model.fit(x_train, y_train)
    dev_pred = model.predict(x_dev)
    policy_eval = evaluate_policy(
        dev_rows,
        model=model,
        method=str(args.method),
        topk=int(args.topk),
        pool_k=int(args.pool_k),
        min_prob=float(args.min_prob),
    )
    metrics = {
        "method_name": "Counterfactual Source-Action Policy",
        "method_formulation": "learns which source-level retrieval action maximizes counterfactual retrieval utility from train/dev rankings",
        "feature_names": FEATURE_NAMES,
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
        "dev_action_accuracy": float(accuracy_score(y_dev, dev_pred)),
        "dev_policy_eval": policy_eval,
        "config": {
            "method": str(args.method),
            "topk": int(args.topk),
            "pool_k": int(args.pool_k),
            "min_gain": float(args.min_gain),
            "min_prob": float(args.min_prob),
            "seed": int(args.seed),
        },
    }
    bundle = SourceActionPolicyBundle(
        model=model,
        feature_names=list(FEATURE_NAMES),
        action_labels=list(ACTION_LABELS),
        metadata=metrics,
    )
    save_source_action_policy_bundle(bundle, args.out_bundle)
    args.out_metrics.parent.mkdir(parents=True, exist_ok=True)
    args.out_metrics.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(metrics, ensure_ascii=False, indent=2))
    print(f"[OK] bundle -> {args.out_bundle}")
    print(f"[OK] metrics -> {args.out_metrics}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
