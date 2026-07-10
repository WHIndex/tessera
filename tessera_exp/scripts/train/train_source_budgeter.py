#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, average_precision_score, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from tessera_exp.e2e.source_budgeter import (  # noqa: E402
    SOURCE_BUDGET_FEATURE_NAMES,
    SOURCE_LABELS,
    SourceBudgeterBundle,
    build_source_budget_features,
    save_source_budgeter_bundle,
)


def is_test_like_path(path: Path) -> bool:
    raw = str(path).lower()
    name = path.name.lower()
    return "test" in name or "/test" in raw or "\\test" in raw


def split_id_set(rows: list[dict]) -> set[str]:
    return {str(row.get("id", "")).strip() for row in rows if str(row.get("id", "")).strip()}


def source_bucket(doc_id: str) -> str:
    raw = str(doc_id or "")
    if raw.startswith("m.") or raw.startswith("/m/") or raw.startswith("g."):
        return "kg"
    prefix = raw.split("_", 1)[0].lower() if "_" in raw else raw.lower()
    if prefix in {"ott", "tat"}:
        return "table"
    return "text"


def row_targets(row: dict) -> tuple[str, dict[str, int], dict[str, float]]:
    gains = {source: 0.0 for source in SOURCE_LABELS}
    counts = {source: 0 for source in SOURCE_LABELS}
    for doc_id, raw_label in (row.get("relevant_chunks", {}) or {}).items():
        try:
            grade = float(raw_label)
        except Exception:
            continue
        if grade <= 0.0:
            continue
        source = source_bucket(str(doc_id))
        counts[source] += 1
        gains[source] += grade
    if sum(counts.values()) <= 0:
        return "text", counts, gains
    top1 = max(SOURCE_LABELS, key=lambda source: (gains[source], counts[source]))
    return top1, counts, gains


def build_dataset(rows: list[dict]) -> tuple[np.ndarray, np.ndarray, dict[str, np.ndarray], dict]:
    xs: list[np.ndarray] = []
    top1: list[str] = []
    need = {source: [] for source in SOURCE_LABELS}
    stats = {
        "queries": 0,
        "top1_source_counts": {source: 0 for source in SOURCE_LABELS},
        "need_source_counts": {source: 0 for source in SOURCE_LABELS},
    }
    for row in rows:
        target, counts, _ = row_targets(row)
        if sum(counts.values()) <= 0:
            continue
        xs.append(build_source_budget_features(str(row.get("query", "")), str(row.get("id", ""))))
        top1.append(target)
        stats["queries"] += 1
        stats["top1_source_counts"][target] += 1
        for source in SOURCE_LABELS:
            val = int(counts[source] > 0)
            need[source].append(val)
            stats["need_source_counts"][source] += val
    if not xs:
        return (
            np.zeros((0, len(SOURCE_BUDGET_FEATURE_NAMES)), dtype=np.float32),
            np.zeros((0,), dtype=object),
            {source: np.zeros((0,), dtype=np.int64) for source in SOURCE_LABELS},
            stats,
        )
    return (
        np.vstack(xs).astype(np.float32),
        np.asarray(top1, dtype=object),
        {source: np.asarray(vals, dtype=np.int64) for source, vals in need.items()},
        stats,
    )


def make_classifier(seed: int) -> Pipeline:
    return Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            (
                "clf",
                LogisticRegression(
                    class_weight="balanced",
                    max_iter=1000,
                    random_state=int(seed),
                    solver="lbfgs",
                ),
            ),
        ]
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Train query-adaptive source budgeter")
    parser.add_argument("--train-file", type=Path, required=True)
    parser.add_argument("--dev-file", type=Path, required=True)
    parser.add_argument("--out-bundle", type=Path, required=True)
    parser.add_argument("--out-metrics", type=Path, required=True)
    parser.add_argument("--max-train", type=int, default=0)
    parser.add_argument("--max-dev", type=int, default=0)
    parser.add_argument("--seed", type=int, default=31)
    parser.add_argument("--allow-test-split-training", action="store_true")
    args = parser.parse_args()

    allow_test_training = bool(args.allow_test_split_training) or os.environ.get("TESSERA_ALLOW_TEST_SPLIT_TRAINING", "0") == "1"
    if not allow_test_training:
        for label, path in (("train-file", args.train_file), ("dev-file", args.dev_file)):
            if is_test_like_path(Path(path)):
                raise ValueError(f"{label} looks like a test split ({path}). Source budgeter must use train/dev only.")

    print(f"[stage] loading train/dev: {args.train_file} {args.dev_file}", flush=True)
    train_rows = json.loads(args.train_file.read_text(encoding="utf-8"))
    dev_rows = json.loads(args.dev_file.read_text(encoding="utf-8"))
    if int(args.max_train) > 0:
        train_rows = train_rows[: int(args.max_train)]
    if int(args.max_dev) > 0:
        dev_rows = dev_rows[: int(args.max_dev)]
    train_ids = split_id_set(train_rows)
    dev_ids = split_id_set(dev_rows)
    overlap = sorted(train_ids & dev_ids)
    if overlap:
        raise ValueError(f"train/dev split overlap detected: {len(overlap)} examples: {overlap[:5]}")

    x_train, y_top_train, y_need_train, train_stats = build_dataset(train_rows)
    x_dev, y_top_dev, y_need_dev, dev_stats = build_dataset(dev_rows)
    if x_train.size == 0 or x_dev.size == 0:
        raise RuntimeError("source budgeter data is empty")
    if len(set(y_top_train.tolist())) < 2:
        raise RuntimeError("source budgeter top1 data has a single class")

    top1_model = make_classifier(seed=int(args.seed))
    print(f"[stage] fitting top1 source model examples={x_train.shape[0]}", flush=True)
    top1_model.fit(x_train, y_top_train)

    need_models = {}
    need_metrics = {}
    for offset, source in enumerate(SOURCE_LABELS, start=1):
        y_train = y_need_train[source]
        y_dev = y_need_dev[source]
        need_metrics[source] = {
            "train_positive": int(np.sum(y_train)),
            "dev_positive": int(np.sum(y_dev)),
            "enabled": False,
        }
        if len(set(y_train.tolist())) < 2:
            continue
        model = make_classifier(seed=int(args.seed) + offset)
        model.fit(x_train, y_train)
        need_models[source] = model
        prob = model.predict_proba(x_dev)[:, 1]
        need_metrics[source].update(
            {
                "enabled": True,
                "dev_average_precision": float(average_precision_score(y_dev, prob)),
                "dev_roc_auc": float(roc_auc_score(y_dev, prob)) if len(set(y_dev.tolist())) > 1 else None,
            }
        )

    top_pred = top1_model.predict(x_dev)
    metrics = {
        "method_name": "Query-Adaptive Source Budgeter",
        "method_formulation": "predict query-level top1 source and text/table/KG evidence needs from train/dev qrel source distributions",
        "feature_names": SOURCE_BUDGET_FEATURE_NAMES,
        "train": train_stats,
        "dev": dev_stats,
        "split_guard": {
            "train_file": str(args.train_file),
            "dev_file": str(args.dev_file),
            "train_queries": int(len(train_ids)),
            "dev_queries": int(len(dev_ids)),
            "train_dev_overlap": int(len(overlap)),
            "test_like_paths_allowed": bool(allow_test_training),
        },
        "dev_top1_accuracy": float(accuracy_score(y_top_dev, top_pred)),
        "need_metrics": need_metrics,
        "config": {"seed": int(args.seed)},
    }
    bundle = SourceBudgeterBundle(
        top1_model=top1_model,
        need_models=need_models,
        feature_names=list(SOURCE_BUDGET_FEATURE_NAMES),
        metadata=metrics,
    )
    save_source_budgeter_bundle(bundle, args.out_bundle)
    args.out_metrics.parent.mkdir(parents=True, exist_ok=True)
    args.out_metrics.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(metrics, ensure_ascii=False, indent=2))
    print(f"[OK] bundle -> {args.out_bundle}")
    print(f"[OK] metrics -> {args.out_metrics}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
