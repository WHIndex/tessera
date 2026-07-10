#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score

from tessera_exp.utils import read_json, write_json, ensure_dir


def load_xy(path: Path):
    rows = read_json(path)
    x = [r.get("query", "") for r in rows]
    y = np.array([r.get("labels_multihot", [0, 0, 0]) for r in rows], dtype=np.int64)
    return x, y, rows


def subset_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(np.all(y_true == y_pred, axis=1)))


def normalized_entropy(probs: np.ndarray) -> np.ndarray:
    eps = 1e-7
    probs = np.clip(probs, eps, 1 - eps)
    ent = -(probs * np.log(probs) + (1 - probs) * np.log(1 - probs))
    ent = np.mean(ent, axis=1)
    return ent / np.log(2.0)


def main() -> int:
    parser = argparse.ArgumentParser(description="Train router baseline (fast sklearn mode)")
    parser.add_argument("--train-file", type=Path, required=True)
    parser.add_argument("--val-file", type=Path, required=True)
    parser.add_argument("--test-file", type=Path, default=None)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--max-train", type=int, default=None)
    parser.add_argument("--max-val", type=int, default=None)
    parser.add_argument("--max-test", type=int, default=None)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--run-id", type=str, default="router_fast")
    args = parser.parse_args()

    ensure_dir(args.out_dir)

    x_train, y_train, train_rows = load_xy(args.train_file)
    x_val, y_val, val_rows = load_xy(args.val_file)

    if args.max_train is not None:
        x_train = x_train[: args.max_train]
        y_train = y_train[: args.max_train]
    if args.max_val is not None:
        x_val = x_val[: args.max_val]
        y_val = y_val[: args.max_val]

    print(f"[train] train_rows={len(x_train)} val_rows={len(x_val)}")

    vectorizer = TfidfVectorizer(ngram_range=(1, 2), max_features=50000)
    x_train_t = vectorizer.fit_transform(x_train)
    x_val_t = vectorizer.transform(x_val)

    models = []
    train_label_pos_rate = []
    for idx in range(y_train.shape[1]):
        y_col = y_train[:, idx]
        pos_rate = float(np.mean(y_col))
        train_label_pos_rate.append(pos_rate)

        uniq = np.unique(y_col)
        if uniq.shape[0] < 2:
            models.append({"kind": "constant", "value": int(uniq[0])})
            continue

        lr = LogisticRegression(max_iter=200, n_jobs=max(1, (os.cpu_count() or 4) - 1))
        lr.fit(x_train_t, y_col)
        models.append({"kind": "logreg", "model": lr})

    val_probs_list = []
    for model_info in models:
        if model_info["kind"] == "constant":
            p = float(model_info["value"])
            val_probs_list.append(np.full((x_val_t.shape[0],), p, dtype=np.float32))
        else:
            val_probs_list.append(model_info["model"].predict_proba(x_val_t)[:, 1])
    val_probs = np.stack(val_probs_list, axis=1)
    y_val_pred = (val_probs >= args.threshold).astype(np.int64)

    # Ensure at least one modality is selected.
    empty_mask = np.sum(y_val_pred, axis=1) == 0
    if np.any(empty_mask):
        top_idx = np.argmax(val_probs[empty_mask], axis=1)
        y_val_pred[empty_mask] = 0
        y_val_pred[empty_mask, top_idx] = 1

    val_micro_f1 = float(f1_score(y_val, y_val_pred, average="micro", zero_division=0))
    val_subset_acc = subset_accuracy(y_val, y_val_pred)
    val_entropy = normalized_entropy(val_probs)

    metrics = {
        "run_id": args.run_id,
        "train_rows": len(x_train),
        "val_rows": len(x_val),
        "threshold": args.threshold,
        "train_label_pos_rate": train_label_pos_rate,
        "val_micro_f1": val_micro_f1,
        "val_subset_acc": val_subset_acc,
        "val_entropy_mean": float(np.mean(val_entropy)),
        "val_entropy_p95": float(np.percentile(val_entropy, 95)),
    }

    if args.test_file is not None and args.test_file.exists():
        x_test, y_test, _ = load_xy(args.test_file)
        if args.max_test is not None:
            x_test = x_test[: args.max_test]
            y_test = y_test[: args.max_test]

        x_test_t = vectorizer.transform(x_test)
        test_probs_list = []
        for model_info in models:
            if model_info["kind"] == "constant":
                p = float(model_info["value"])
                test_probs_list.append(np.full((x_test_t.shape[0],), p, dtype=np.float32))
            else:
                test_probs_list.append(model_info["model"].predict_proba(x_test_t)[:, 1])
        test_probs = np.stack(test_probs_list, axis=1)
        y_test_pred = (test_probs >= args.threshold).astype(np.int64)
        empty_mask = np.sum(y_test_pred, axis=1) == 0
        if np.any(empty_mask):
            top_idx = np.argmax(test_probs[empty_mask], axis=1)
            y_test_pred[empty_mask] = 0
            y_test_pred[empty_mask, top_idx] = 1

        metrics.update(
            {
                "test_rows": len(x_test),
                "test_micro_f1": float(f1_score(y_test, y_test_pred, average="micro", zero_division=0)),
                "test_subset_acc": subset_accuracy(y_test, y_test_pred),
            }
        )

    metrics_path = args.out_dir / f"{args.run_id}_metrics.json"
    write_json(metrics_path, metrics)
    print(f"[OK] metrics -> {metrics_path}")
    print(metrics)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
