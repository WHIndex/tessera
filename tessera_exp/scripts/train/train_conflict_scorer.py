#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
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
    CONFLICT_FEATURE_NAMES,
    ConflictBundle,
    build_conflict_feature_vector,
)
from tessera_exp.e2e.metrics import normalize_answer  # noqa: E402
from tessera_exp.utils import ensure_dir, write_json  # noqa: E402


def parse_bool(value: object) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool):
        return int(value)
    text = str(value).strip().lower()
    if text in {"1", "true", "yes", "y"}:
        return 1
    if text in {"0", "false", "no", "n"}:
        return 0
    return None


def safe_float(value: object, default: float | None = None) -> float | None:
    try:
        return float(value)
    except Exception:
        return default


def load_rows(path: Path, max_rows: int | None = None) -> list[dict]:
    if not path.exists():
        raise FileNotFoundError(path)

    suffix = path.suffix.lower()
    if suffix == ".csv":
        with path.open("r", encoding="utf-8", newline="") as handle:
            rows = [dict(item) for item in csv.DictReader(handle)]
        return rows[:max_rows] if max_rows is not None else rows

    text = path.read_text(encoding="utf-8").strip()
    if not text:
        return []

    rows: list[dict]
    if suffix == ".jsonl":
        rows = []
        for line in text.splitlines():
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    else:
        try:
            payload = json.loads(text)
        except json.JSONDecodeError:
            rows = []
            for line in text.splitlines():
                line = line.strip()
                if not line:
                    continue
                rows.append(json.loads(line))
        else:
            if isinstance(payload, list):
                rows = [dict(item) for item in payload]
            elif isinstance(payload, dict):
                if isinstance(payload.get("rows"), list):
                    rows = [dict(item) for item in payload["rows"]]
                elif isinstance(payload.get("data"), list):
                    rows = [dict(item) for item in payload["data"]]
                else:
                    rows = [dict(payload)]
            else:
                rows = []

    return rows[:max_rows] if max_rows is not None else rows


def add_context(
    contexts: list[str],
    doc_ids: list[str],
    seen: set[tuple[str, str]],
    text: object,
    doc_id: object | None,
) -> None:
    snippet = str(text or "").strip()
    if not snippet:
        return
    doc_id_text = str(doc_id or f"ctx_{len(contexts)}").strip()
    key = (doc_id_text, snippet)
    if key in seen:
        return
    seen.add(key)
    contexts.append(snippet)
    doc_ids.append(doc_id_text)


def collect_contexts(row: dict) -> tuple[list[str], list[str]]:
    contexts: list[str] = []
    doc_ids: list[str] = []
    seen: set[tuple[str, str]] = set()

    for key in ("contexts", "context_texts", "selected_contexts", "retrieved_contexts", "evidence_contexts"):
        value = row.get(key)
        if not isinstance(value, list):
            continue
        for item in value:
            if isinstance(item, dict):
                add_context(
                    contexts,
                    doc_ids,
                    seen,
                    item.get("text") or item.get("snippet") or item.get("content") or item.get("context"),
                    item.get("doc_id") or item.get("id") or item.get("chunk_id") or item.get("chunkId"),
                )
            else:
                add_context(contexts, doc_ids, seen, item, None)

    for text_key, doc_key in (
        ("table_anchor_snippet", "table_anchor_chunk_id"),
        ("kg_anchor_snippet", "kg_anchor_chunk_id"),
        ("text_anchor_snippet", "text_anchor_chunk_id"),
        ("context_snippet", "context_chunk_id"),
    ):
        add_context(contexts, doc_ids, seen, row.get(text_key), row.get(doc_key))

    return contexts, doc_ids


def resolve_label(row: dict, label_threshold: float) -> tuple[int | None, str]:
    explicit = parse_bool(row.get("is_conflict"))
    if explicit is None:
        explicit = parse_bool(row.get("conflict_flag"))
    if explicit is not None:
        return int(explicit), "annotated"

    risk_score = safe_float(row.get("risk_score"), default=None)
    if risk_score is not None:
        return int(float(risk_score) >= float(label_threshold)), "risk_proxy"

    dense_pred = normalize_answer(str(row.get("dense_prediction", "")))
    uni_pred = normalize_answer(str(row.get("tessera_prediction", "")))
    if dense_pred or uni_pred:
        return int(dense_pred != uni_pred), "prediction_disagreement"

    return None, "missing"


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


def positive_class_probability(model: object, x: np.ndarray) -> np.ndarray:
    prob = np.asarray(model.predict_proba(x), dtype=np.float32)
    if prob.ndim == 1:
        return prob.astype(np.float32)
    if prob.shape[1] == 1:
        classes = np.asarray(getattr(model, "classes_", []))
        if classes.size == 1 and str(classes[0]).strip().lower() in {"1", "true", "yes", "conflict"}:
            return prob[:, 0]
        return np.zeros((prob.shape[0],), dtype=np.float32)

    classes = np.asarray(getattr(model, "classes_", []))
    if classes.size == prob.shape[1]:
        class_list = classes.tolist()
        if 1 in class_list:
            return prob[:, int(np.where(classes == 1)[0][0])]
        for idx, cls in enumerate(class_list):
            if str(cls).strip().lower() in {"1", "true", "yes", "conflict"}:
                return prob[:, idx]
    return prob[:, -1]


def build_examples(rows: list[dict], label_threshold: float) -> tuple[list[np.ndarray], list[int], list[dict[str, object]], Counter[str]]:
    features: list[np.ndarray] = []
    labels: list[int] = []
    meta: list[dict[str, object]] = []
    label_sources: Counter[str] = Counter()

    for row in rows:
        contexts, doc_ids = collect_contexts(row)
        if not contexts:
            continue

        label, source = resolve_label(row, label_threshold)
        if label is None:
            continue

        label_sources[source] += 1
        features.append(
            build_conflict_feature_vector(
                query=str(row.get("query", "")),
                contexts=contexts,
                doc_ids=doc_ids,
                table_kg_only=False,
                probe_k=min(12, max(2, len(contexts))),
                max_literals_per_doc=0,
            )
        )
        labels.append(int(label))
        meta.append(
            {
                "query_id": row.get("id"),
                "label": int(label),
                "label_source": source,
                "doc_ids": doc_ids,
                "context_count": len(contexts),
            }
        )

    return features, labels, meta, label_sources


def fit_model(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
    *,
    calibration_method: str,
) -> tuple[object, bool]:
    if x_train.size == 0 or len(set(int(x) for x in y_train.tolist())) < 2:
        model = DummyClassifier(strategy="most_frequent")
        if x_train.size == 0:
            model.fit(np.zeros((1, len(CONFLICT_FEATURE_NAMES)), dtype=np.float32), np.asarray([0], dtype=np.int64))
        else:
            model.fit(x_train, y_train)
        return model, False

    base = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            (
                "clf",
                LogisticRegression(
                    max_iter=1000,
                    solver="lbfgs",
                    class_weight="balanced",
                ),
            ),
        ]
    )
    base.fit(x_train, y_train)

    if x_val.size > 0 and len(set(int(x) for x in y_val.tolist())) >= 2:
        class_counts = Counter(int(x) for x in y_val.tolist())
        min_class_count = min(class_counts.values()) if class_counts else 0
        if min_class_count < 2:
            return base, False
        cv_splits = max(2, min(5, int(x_val.shape[0]), int(min_class_count)))
        calibrated = CalibratedClassifierCV(estimator=FrozenEstimator(base), method=calibration_method, cv=cv_splits)
        calibrated.fit(x_val, y_val)
        return calibrated, True

    return base, False


def evaluate(model: object, x: np.ndarray, y: np.ndarray) -> dict[str, float]:
    if x.size == 0 or y.size == 0:
        return {
            "accuracy": 0.0,
            "f1": 0.0,
            "roc_auc": 0.0,
            "avg_precision": 0.0,
            "brier": 0.0,
            "ece": 0.0,
        }

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


def main() -> int:
    parser = argparse.ArgumentParser(description="Train a learned conflict scorer")
    parser.add_argument("--train-file", type=Path, required=True)
    parser.add_argument("--val-file", type=Path, required=True)
    parser.add_argument("--test-file", type=Path, default=None)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--run-id", type=str, default="conflict_scorer")
    parser.add_argument("--max-train", type=int, default=None)
    parser.add_argument("--max-val", type=int, default=None)
    parser.add_argument("--max-test", type=int, default=None)
    parser.add_argument(
        "--label-threshold",
        type=float,
        default=None,
        help="Proxy threshold used when explicit annotations are missing. Defaults to the training median risk_score.",
    )
    parser.add_argument(
        "--calibration-method",
        type=str,
        default="sigmoid",
        choices=("sigmoid", "isotonic"),
        help="Calibration method applied on the validation split after fitting the base model.",
    )
    args = parser.parse_args()

    ensure_dir(args.out_dir)

    train_rows = load_rows(args.train_file, max_rows=args.max_train)
    val_rows = load_rows(args.val_file, max_rows=args.max_val)
    test_rows = load_rows(args.test_file, max_rows=args.max_test) if args.test_file is not None else []

    if args.label_threshold is not None:
        label_threshold = float(args.label_threshold)
    else:
        train_risks = [float(risk) for risk in (safe_float(row.get("risk_score"), default=None) for row in train_rows) if risk is not None]
        label_threshold = float(np.median(train_risks)) if train_risks else 0.5

    x_train_list, y_train_list, train_meta, train_sources = build_examples(train_rows, label_threshold)
    x_val_list, y_val_list, val_meta, val_sources = build_examples(val_rows, label_threshold)
    x_test_list, y_test_list, test_meta, test_sources = build_examples(test_rows, label_threshold) if test_rows else ([], [], [], Counter())

    x_train = np.asarray(x_train_list, dtype=np.float32)
    y_train = np.asarray(y_train_list, dtype=np.int64)
    x_val = np.asarray(x_val_list, dtype=np.float32)
    y_val = np.asarray(y_val_list, dtype=np.int64)
    x_test = np.asarray(x_test_list, dtype=np.float32)
    y_test = np.asarray(y_test_list, dtype=np.int64)

    print(f"[conflict] train_examples={len(x_train)} val_examples={len(x_val)}")
    print(f"[conflict] train_label_distribution={dict(Counter(int(x) for x in y_train.tolist()))}")
    print(f"[conflict] train_label_sources={dict(train_sources)}")

    model, calibrated = fit_model(
        x_train,
        y_train,
        x_val,
        y_val,
        calibration_method=str(args.calibration_method),
    )

    val_metrics = evaluate(model, x_val, y_val)
    test_metrics = evaluate(model, x_test, y_test) if x_test.size > 0 else None

    metrics = {
        "run_id": args.run_id,
        "train_rows": len(train_rows),
        "val_rows": len(val_rows),
        "test_rows": len(test_rows),
        "train_examples": len(x_train),
        "val_examples": len(x_val),
        "test_examples": len(x_test),
        "label_threshold": float(label_threshold),
        "calibration_method": str(args.calibration_method),
        "calibrated": bool(calibrated),
        "train_label_pos_rate": float(np.mean(y_train)) if y_train.size else 0.0,
        "val_label_pos_rate": float(np.mean(y_val)) if y_val.size else 0.0,
        "train_label_sources": dict(train_sources),
        "val_label_sources": dict(val_sources),
        "test_label_sources": dict(test_sources),
        "val_accuracy": val_metrics["accuracy"],
        "val_f1": val_metrics["f1"],
        "val_roc_auc": val_metrics["roc_auc"],
        "val_avg_precision": val_metrics["avg_precision"],
        "val_brier": val_metrics["brier"],
        "val_ece": val_metrics["ece"],
    }
    if test_metrics is not None:
        metrics.update(
            {
                "test_accuracy": test_metrics["accuracy"],
                "test_f1": test_metrics["f1"],
                "test_roc_auc": test_metrics["roc_auc"],
                "test_avg_precision": test_metrics["avg_precision"],
                "test_brier": test_metrics["brier"],
                "test_ece": test_metrics["ece"],
            }
        )

    bundle = ConflictBundle(
        model=model,
        feature_names=list(CONFLICT_FEATURE_NAMES),
        metadata={
            "run_id": args.run_id,
            "feature_source": "query_context_conflict_pack_features",
            "feature_version": 1,
            "label_threshold": float(label_threshold),
            "calibration_method": str(args.calibration_method),
            "calibrated": bool(calibrated),
            "train_rows": len(train_rows),
            "val_rows": len(val_rows),
            "test_rows": len(test_rows),
            "train_examples": len(x_train),
            "val_examples": len(x_val),
            "test_examples": len(x_test),
            "train_meta_preview": train_meta[:3],
            "val_meta_preview": val_meta[:3],
            "test_meta_preview": test_meta[:3],
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
