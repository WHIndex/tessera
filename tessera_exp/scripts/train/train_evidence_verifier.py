#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import random
from collections import Counter, defaultdict
from pathlib import Path
import sys

import numpy as np
from sklearn.calibration import CalibratedClassifierCV
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, average_precision_score, brier_score_loss, f1_score, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.frozen import FrozenEstimator

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from tessera_exp.e2e.controller import (  # noqa: E402
    VerifierBundle,
    answer_support_score,
    build_verifier_feature_vector,
    exact_label_from_multihot,
    VERIFIER_FEATURE_NAMES,
)
from tessera_exp.e2e.baselines import source_bucket  # noqa: E402
from tessera_exp.e2e.metrics import normalize_answer  # noqa: E402
from tessera_exp.utils import ensure_dir, infer_modalities_from_dataset_score, infer_modalities_from_relevant_chunks, modality_multihot, read_json, write_json  # noqa: E402


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


def load_corpus_map(corpus_file: Path) -> tuple[dict[str, str], dict[str, list[str]]]:
    corpus = json.loads(corpus_file.read_text(encoding="utf-8"))
    corpus_map: dict[str, str] = {}
    bucket_map: dict[str, list[str]] = defaultdict(list)
    for row in corpus:
        doc_id = str(row.get("id", ""))
        if not doc_id:
            continue
        corpus_map[doc_id] = str(row.get("text", ""))
        bucket_map[source_bucket(doc_id)].append(doc_id)
    return corpus_map, bucket_map


def choose_other_answer(answer_pool: list[str], current_answer: str, rng: random.Random) -> str | None:
    candidates = [a for a in answer_pool if normalize_answer(a) and normalize_answer(a) != normalize_answer(current_answer)]
    if not candidates:
        return None
    return rng.choice(candidates)


def sample_negative_ids(
    positive_ids: list[str],
    bucket_map: dict[str, list[str]],
    all_doc_ids: list[str],
    rng: random.Random,
) -> list[str]:
    positive_set = set(positive_ids)
    selected: list[str] = []
    seen: set[str] = set()
    for pos_id in positive_ids:
        bucket = source_bucket(pos_id)
        pool = [doc_id for doc_id in bucket_map.get(bucket, []) if doc_id not in positive_set and doc_id not in seen]
        if not pool:
            pool = [doc_id for doc_id in all_doc_ids if doc_id not in positive_set and doc_id not in seen]
        if not pool:
            continue
        chosen = rng.choice(pool)
        seen.add(chosen)
        selected.append(chosen)
    if not selected:
        pool = [doc_id for doc_id in all_doc_ids if doc_id not in positive_set]
        if pool:
            selected = [rng.choice(pool)]
    return selected


def build_examples(
    rows: list[dict],
    corpus_map: dict[str, str],
    bucket_map: dict[str, list[str]],
    *,
    negatives_per_query: int,
    support_k: int,
    seed: int,
) -> tuple[list[np.ndarray], list[int], list[dict[str, object]]]:
    rng = random.Random(seed)
    all_doc_ids = list(corpus_map.keys())
    answer_pool = [str(row.get("answer", "")) for row in rows if str(row.get("answer", "")).strip()]

    features: list[np.ndarray] = []
    labels: list[int] = []
    meta: list[dict[str, object]] = []

    for row in rows:
        query = str(row.get("query", ""))
        answer = str(row.get("answer", ""))
        rel_items: list[str] = []
        for chunk_id, label in row.get("relevant_chunks", {}).items():
            if chunk_id not in corpus_map:
                continue
            try:
                if float(label) > 0:
                    rel_items.append(str(chunk_id))
            except Exception:
                continue
        if not rel_items:
            continue

        pos_ids = rel_items[: max(1, int(support_k))]
        pos_ctx = [corpus_map[cid] for cid in pos_ids if cid in corpus_map]
        if not pos_ctx:
            continue

        # Positive pack.
        features.append(build_verifier_feature_vector(query, answer, pos_ctx, doc_ids=pos_ids))
        labels.append(1)
        meta.append({"label": 1, "kind": "gold", "query_id": row.get("id"), "doc_ids": pos_ids})

        negatives_added = 0

        # Negative 1: random same-size pack from the corpus.
        if negatives_added < negatives_per_query:
            neg_ids = sample_negative_ids(pos_ids, bucket_map, all_doc_ids, rng)
            neg_ctx = [corpus_map[cid] for cid in neg_ids if cid in corpus_map]
            if neg_ctx:
                features.append(build_verifier_feature_vector(query, answer, neg_ctx, doc_ids=neg_ids))
                labels.append(0)
                meta.append({"label": 0, "kind": "random_neg", "query_id": row.get("id"), "doc_ids": neg_ids})
                negatives_added += 1

        # Negative 2: same evidence, swapped answer.
        if negatives_added < negatives_per_query:
            swapped = choose_other_answer(answer_pool, answer, rng)
            if swapped:
                features.append(build_verifier_feature_vector(query, swapped, pos_ctx, doc_ids=pos_ids))
                labels.append(0)
                meta.append({"label": 0, "kind": "answer_swap", "query_id": row.get("id"), "doc_ids": pos_ids})
                negatives_added += 1

        # Negative 3: mixed pack, if we still need more negatives.
        if negatives_added < negatives_per_query:
            neg_ids = sample_negative_ids(pos_ids, bucket_map, all_doc_ids, rng)
            mixed_ids = list(pos_ids[:-1]) + neg_ids[:1]
            mixed_ids = list(dict.fromkeys(mixed_ids))
            mixed_ctx = [corpus_map[cid] for cid in mixed_ids if cid in corpus_map]
            if mixed_ctx:
                features.append(build_verifier_feature_vector(query, answer, mixed_ctx, doc_ids=mixed_ids))
                labels.append(0)
                meta.append({"label": 0, "kind": "mixed_neg", "query_id": row.get("id"), "doc_ids": mixed_ids})
                negatives_added += 1

        while negatives_added < negatives_per_query:
            neg_ids = sample_negative_ids(pos_ids, bucket_map, all_doc_ids, rng)
            neg_ctx = [corpus_map[cid] for cid in neg_ids if cid in corpus_map]
            if not neg_ctx:
                break
            features.append(build_verifier_feature_vector(query, answer, neg_ctx, doc_ids=neg_ids))
            labels.append(0)
            meta.append({"label": 0, "kind": f"random_neg_{negatives_added}", "query_id": row.get("id"), "doc_ids": neg_ids})
            negatives_added += 1

    return features, labels, meta


def fit_verifier(x_train: np.ndarray, y_train: np.ndarray) -> Pipeline:
    if x_train.size == 0 or len(set(int(x) for x in y_train.tolist())) < 2:
        model = DummyClassifier(strategy="most_frequent")
        if x_train.size == 0:
            model.fit(np.zeros((1, len(VERIFIER_FEATURE_NAMES)), dtype=np.float32), np.asarray([0], dtype=np.int64))
        else:
            model.fit(x_train, y_train)
        return model

    clf = LogisticRegression(
        max_iter=1000,
        solver="lbfgs",
        class_weight="balanced",
    )
    model = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            ("clf", clf),
        ]
    )
    model.fit(x_train, y_train)
    return model


def calibrate_verifier(model: Pipeline, x_val: np.ndarray, y_val: np.ndarray, calibration_method: str) -> tuple[Pipeline, bool]:
    if x_val.size == 0 or len(set(int(x) for x in y_val.tolist())) < 2:
        return model, False
    class_counts = Counter(int(x) for x in y_val.tolist())
    min_class_count = min(class_counts.values()) if class_counts else 0
    if min_class_count < 2:
        return model, False
    cv_splits = max(2, min(5, int(x_val.shape[0]), int(min_class_count)))
    calibrated = CalibratedClassifierCV(estimator=FrozenEstimator(model), method=calibration_method, cv=cv_splits)
    calibrated.fit(x_val, y_val)
    return calibrated, True


def positive_class_probability(model: Pipeline, x: np.ndarray) -> np.ndarray:
    prob = np.asarray(model.predict_proba(x), dtype=np.float32)
    if prob.ndim == 1:
        return prob.astype(np.float32)
    if prob.shape[1] == 1:
        classes = np.asarray(getattr(model, "classes_", []))
        if classes.size == 1 and str(classes[0]).strip().lower() in {"1", "true", "yes", "support"}:
            return prob[:, 0]
        return np.zeros((prob.shape[0],), dtype=np.float32)

    classes = np.asarray(getattr(model, "classes_", []))
    if classes.size == prob.shape[1]:
        class_list = classes.tolist()
        if 1 in class_list:
            return prob[:, int(np.where(classes == 1)[0][0])]
        for idx, cls in enumerate(class_list):
            if str(cls).strip().lower() in {"1", "true", "yes", "support"}:
                return prob[:, idx]
    return prob[:, -1]


def expected_calibration_error(y_true: np.ndarray, prob: np.ndarray, n_bins: int = 10) -> float:
    if y_true.size == 0:
        return 0.0
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    total = float(y_true.shape[0])
    ece = 0.0
    for idx in range(n_bins):
        left = bins[idx]
        right = bins[idx + 1]
        if idx == n_bins - 1:
            mask = (prob >= left) & (prob <= right)
        else:
            mask = (prob >= left) & (prob < right)
        if not np.any(mask):
            continue
        acc = float(np.mean(y_true[mask]))
        conf = float(np.mean(prob[mask]))
        ece += (float(np.sum(mask)) / total) * abs(acc - conf)
    return float(ece)


def evaluate(model: Pipeline, x: np.ndarray, y: np.ndarray) -> dict[str, float]:
    if x.size == 0:
        return {"accuracy": 0.0, "f1": 0.0, "roc_auc": 0.0, "avg_precision": 0.0, "brier": 0.0, "ece": 0.0}
    prob = positive_class_probability(model, x)
    pred = (prob >= 0.5).astype(np.int64)
    out = {
        "accuracy": float(accuracy_score(y, pred)),
        "f1": float(f1_score(y, pred, zero_division=0)),
        "brier": float(brier_score_loss(y, prob)),
        "ece": float(expected_calibration_error(y, prob)),
    }
    try:
        out["roc_auc"] = float(roc_auc_score(y, prob))
    except Exception:
        out["roc_auc"] = 0.0
    try:
        out["avg_precision"] = float(average_precision_score(y, prob))
    except Exception:
        out["avg_precision"] = 0.0
    return out


def load_rows(path: Path, max_rows: int | None = None) -> list[dict]:
    rows = read_json(path)
    if max_rows is not None:
        rows = rows[:max_rows]
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description="Train a learned evidence verifier")
    parser.add_argument("--train-file", type=Path, required=True)
    parser.add_argument("--val-file", type=Path, required=True)
    parser.add_argument("--test-file", type=Path, default=None)
    parser.add_argument("--corpus-file", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--run-id", type=str, default="evidence_verifier")
    parser.add_argument("--max-train", type=int, default=None)
    parser.add_argument("--max-val", type=int, default=None)
    parser.add_argument("--max-test", type=int, default=None)
    parser.add_argument("--support-k", type=int, default=6)
    parser.add_argument("--negatives-per-query", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--calibration-method",
        type=str,
        default="sigmoid",
        choices=("sigmoid", "isotonic"),
        help="Calibration method applied on the validation split after the base verifier is fit.",
    )
    args = parser.parse_args()

    ensure_dir(args.out_dir)

    train_rows = load_rows(args.train_file, max_rows=args.max_train)
    val_rows = load_rows(args.val_file, max_rows=args.max_val)
    corpus_map, bucket_map = load_corpus_map(args.corpus_file)

    x_train_list, y_train_list, train_meta = build_examples(
        train_rows,
        corpus_map,
        bucket_map,
        negatives_per_query=int(args.negatives_per_query),
        support_k=int(args.support_k),
        seed=int(args.seed),
    )
    x_val_list, y_val_list, val_meta = build_examples(
        val_rows,
        corpus_map,
        bucket_map,
        negatives_per_query=int(args.negatives_per_query),
        support_k=int(args.support_k),
        seed=int(args.seed) + 1,
    )

    x_train = np.asarray(x_train_list, dtype=np.float32)
    y_train = np.asarray(y_train_list, dtype=np.int64)
    x_val = np.asarray(x_val_list, dtype=np.float32)
    y_val = np.asarray(y_val_list, dtype=np.int64)

    print(f"[verifier] train_examples={len(x_train)} val_examples={len(x_val)}")
    print(f"[verifier] train_label_distribution={dict(Counter(int(x) for x in y_train.tolist()))}")

    model = fit_verifier(x_train, y_train)
    model, calibrated = calibrate_verifier(model, x_val, y_val, calibration_method=str(args.calibration_method))
    val_metrics = evaluate(model, x_val, y_val)

    metrics = {
        "run_id": args.run_id,
        "train_rows": len(train_rows),
        "val_rows": len(val_rows),
        "train_examples": len(x_train),
        "val_examples": len(x_val),
        "support_k": int(args.support_k),
        "negatives_per_query": int(args.negatives_per_query),
        "calibration_method": str(args.calibration_method),
        "calibrated": bool(calibrated),
        "val_accuracy": val_metrics["accuracy"],
        "val_f1": val_metrics["f1"],
        "val_roc_auc": val_metrics["roc_auc"],
        "val_avg_precision": val_metrics["avg_precision"],
        "val_brier": val_metrics["brier"],
        "val_ece": val_metrics["ece"],
    }

    if args.test_file is not None and args.test_file.exists():
        test_rows = load_rows(args.test_file, max_rows=args.max_test)
        x_test_list, y_test_list, test_meta = build_examples(
            test_rows,
            corpus_map,
            bucket_map,
            negatives_per_query=int(args.negatives_per_query),
            support_k=int(args.support_k),
            seed=int(args.seed) + 2,
        )
        x_test = np.asarray(x_test_list, dtype=np.float32)
        y_test = np.asarray(y_test_list, dtype=np.int64)
        test_metrics = evaluate(model, x_test, y_test)
        metrics.update(
            {
                "test_rows": len(test_rows),
                "test_examples": len(x_test),
                "test_accuracy": test_metrics["accuracy"],
                "test_f1": test_metrics["f1"],
                "test_roc_auc": test_metrics["roc_auc"],
                "test_avg_precision": test_metrics["avg_precision"],
                "test_brier": test_metrics["brier"],
                "test_ece": test_metrics["ece"],
            }
        )
    else:
        test_meta = []

    bundle = VerifierBundle(
        model=model,
        feature_names=list(VERIFIER_FEATURE_NAMES),
        metadata={
            "run_id": args.run_id,
            "support_k": int(args.support_k),
            "negatives_per_query": int(args.negatives_per_query),
            "train_examples": len(x_train),
            "val_examples": len(x_val),
            "feature_source": "query_answer_context_pack_features",
            "feature_version": 3,
            "calibration_method": str(args.calibration_method),
            "calibrated": bool(calibrated),
            "train_meta_preview": train_meta[:3],
            "val_meta_preview": val_meta[:3],
        },
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
