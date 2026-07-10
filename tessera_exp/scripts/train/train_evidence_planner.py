#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
from collections import Counter
from pathlib import Path
import sys

import numpy as np
from sklearn.dummy import DummyClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.pipeline import Pipeline

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from tessera_exp.e2e.controller import (  # noqa: E402
    PlannerBundle,
    build_planner_input,
    exact_label_from_multihot,
    multihot_from_exact_label,
)
from tessera_exp.utils import (  # noqa: E402
    infer_modalities_from_dataset_score,
    infer_modalities_from_relevant_chunks,
    modality_multihot,
    read_json,
    write_json,
    ensure_dir,
)


def infer_multihot_label(row: dict) -> list[int]:
    labels = row.get("labels_multihot")
    if isinstance(labels, list) and labels:
        return [int(x) for x in labels[:3]]

    labels_text = row.get("labels")
    if isinstance(labels_text, list) and labels_text:
        return modality_multihot(labels_text)

    dataset_score = row.get("dataset_score")
    if isinstance(dataset_score, dict) and dataset_score:
        labels = infer_modalities_from_dataset_score(dataset_score)
        if labels:
            return modality_multihot(labels)

    relevant_chunks = row.get("relevant_chunks")
    if isinstance(relevant_chunks, dict) and relevant_chunks:
        labels = infer_modalities_from_relevant_chunks(relevant_chunks)
        if labels:
            return modality_multihot(labels)

    return [0, 0, 0]


def load_samples(path: Path, max_rows: int | None = None) -> tuple[list[str], list[str]]:
    rows = read_json(path)
    if max_rows is not None:
        rows = rows[:max_rows]

    texts: list[str] = []
    labels: list[str] = []
    for row in rows:
        multihot = infer_multihot_label(row)
        label = exact_label_from_multihot(multihot)
        if label == "unknown":
            continue
        texts.append(build_planner_input(row.get("query", ""), query_id=row.get("id")))
        labels.append(label)
    return texts, labels


def top2_accuracy(y_true: np.ndarray, prob: np.ndarray, class_names: list[str]) -> float:
    if prob.size == 0:
        return 0.0
    order = np.argsort(-prob, axis=1)
    top2 = order[:, : min(2, prob.shape[1])]
    hits = 0
    for i, true_idx in enumerate(y_true):
        if int(true_idx) in set(int(j) for j in top2[i]):
            hits += 1
    return float(hits / max(1, len(y_true)))


def make_model(x_train: list[str], y_train: list[str]) -> Pipeline:
    if len(set(y_train)) < 2:
        model = Pipeline(
            steps=[
                ("vectorizer", TfidfVectorizer(ngram_range=(1, 2), max_features=50000)),
                ("clf", DummyClassifier(strategy="most_frequent")),
            ]
        )
        model.fit(x_train, y_train)
        return model

    vectorizer = TfidfVectorizer(ngram_range=(1, 2), max_features=100000)
    classifier = LogisticRegression(
        max_iter=1000,
        solver="saga",
        class_weight="balanced",
        n_jobs=max(1, (os.cpu_count() or 4) - 1),
    )
    model = Pipeline(
        steps=[
            ("vectorizer", vectorizer),
            ("clf", classifier),
        ]
    )
    model.fit(x_train, y_train)
    return model


def evaluate_split(model: Pipeline, x: list[str], y: list[str]) -> dict[str, float]:
    if not x:
        return {"exact_acc": 0.0, "macro_f1": 0.0, "top2_acc": 0.0}
    probs = np.asarray(model.predict_proba(x), dtype=np.float32)
    preds = np.asarray(model.predict(x))
    classes = list(model.named_steps["clf"].classes_)
    class_to_idx = {name: idx for idx, name in enumerate(classes)}
    y_idx = np.asarray([class_to_idx.get(label, 0) for label in y], dtype=np.int64)
    pred_idx = np.asarray([class_to_idx.get(label, 0) for label in preds], dtype=np.int64)
    return {
        "exact_acc": float(accuracy_score(y_idx, pred_idx)),
        "macro_f1": float(f1_score(y_idx, pred_idx, average="macro", zero_division=0)),
        "top2_acc": float(top2_accuracy(y_idx, probs, classes)),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Train a learned evidence planner")
    parser.add_argument("--train-file", type=Path, required=True)
    parser.add_argument("--val-file", type=Path, required=True)
    parser.add_argument("--test-file", type=Path, default=None)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--run-id", type=str, default="evidence_planner")
    parser.add_argument("--max-train", type=int, default=None)
    parser.add_argument("--max-val", type=int, default=None)
    parser.add_argument("--max-test", type=int, default=None)
    args = parser.parse_args()

    ensure_dir(args.out_dir)

    x_train, y_train = load_samples(args.train_file, max_rows=args.max_train)
    x_val, y_val = load_samples(args.val_file, max_rows=args.max_val)

    print(f"[planner] train_rows={len(x_train)} val_rows={len(x_val)}")
    train_distribution = Counter(y_train)
    print(f"[planner] train_distribution={dict(train_distribution)}")

    model = make_model(x_train, y_train)
    val_metrics = evaluate_split(model, x_val, y_val)

    metrics = {
        "run_id": args.run_id,
        "train_rows": len(x_train),
        "val_rows": len(x_val),
        "train_distribution": dict(train_distribution),
        "val_exact_acc": val_metrics["exact_acc"],
        "val_macro_f1": val_metrics["macro_f1"],
        "val_top2_acc": val_metrics["top2_acc"],
    }

    class_names = list(model.named_steps["clf"].classes_)
    class_to_modality = np.stack([multihot_from_exact_label(name) for name in class_names], axis=0)
    bundle = PlannerBundle(
        model=model,
        class_names=class_names,
        class_to_modality=class_to_modality,
        metadata={
            "run_id": args.run_id,
            "train_rows": len(x_train),
            "val_rows": len(x_val),
            "feature_source": "query_text_plus_metadata_tokens",
            "classes": class_names,
        },
    )

    if args.test_file is not None and args.test_file.exists():
        x_test, y_test = load_samples(args.test_file, max_rows=args.max_test)
        test_metrics = evaluate_split(model, x_test, y_test)
        metrics.update(
            {
                "test_rows": len(x_test),
                "test_exact_acc": test_metrics["exact_acc"],
                "test_macro_f1": test_metrics["macro_f1"],
                "test_top2_acc": test_metrics["top2_acc"],
            }
        )

    bundle_path = bundle.save(args.out_dir)
    metrics_path = args.out_dir / f"{args.run_id}_metrics.json"
    write_json(metrics_path, metrics)

    print(metrics)
    print(f"[OK] bundle -> {bundle_path}")
    print(f"[OK] metrics -> {metrics_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
