#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys

import numpy as np
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from unifusion_exp.e2e.kg_consistency_verifier import (  # noqa: E402
    KGConsistencyBundle,
    KG_CONSISTENCY_FEATURE_NAMES,
    build_kg_consistency_features,
    save_kg_consistency_bundle,
)


def is_test_like_path(path: Path) -> bool:
    raw = str(path).lower()
    name = path.name.lower()
    return "test" in name or "/test" in raw or "\\test" in raw


def split_id_set(rows: list[dict]) -> set[str]:
    return {str(row.get("id", "")).strip() for row in rows if str(row.get("id", "")).strip()}


def is_kg_doc_id(doc_id: str) -> bool:
    raw = str(doc_id or "")
    return raw.startswith("m.") or raw.startswith("/m/") or raw.startswith("g.")


def build_doc_maps(corpus: list[dict]) -> tuple[dict[str, str], dict[str, list[str]]]:
    doc_text_by_id: dict[str, str] = {}
    by_mid: dict[str, list[str]] = {}
    for row in corpus:
        did = str(row.get("id", ""))
        if not did:
            continue
        doc_text_by_id[did] = str(row.get("text", ""))
        if is_kg_doc_id(did):
            mid = did.rsplit("_", 1)[0] if "_" in did else did
            by_mid.setdefault(mid, []).append(did)
    return doc_text_by_id, by_mid


def build_examples(
    *,
    rows: list[dict],
    doc_text_by_id: dict[str, str],
    by_mid: dict[str, list[str]],
    max_same_mid_negatives: int,
) -> tuple[np.ndarray, np.ndarray, dict]:
    x_parts: list[np.ndarray] = []
    y_parts: list[int] = []
    stats = {
        "queries": 0,
        "examples": 0,
        "positives": 0,
        "negatives": 0,
        "same_mid_negatives": 0,
        "kg_positive_queries": 0,
        "kg_negative_queries": 0,
    }
    for row in rows:
        query = str(row.get("query", ""))
        rels = row.get("relevant_chunks", {}) or {}
        kg_items: list[tuple[str, int]] = []
        for raw_doc_id, raw_label in rels.items():
            doc_id = str(raw_doc_id)
            if not is_kg_doc_id(doc_id) or doc_id not in doc_text_by_id:
                continue
            try:
                label = 1 if float(raw_label) > 0 else 0
            except Exception:
                label = 0
            kg_items.append((doc_id, label))
        if not kg_items:
            continue
        stats["queries"] += 1
        stats["kg_positive_queries"] += int(any(label > 0 for _, label in kg_items))
        stats["kg_negative_queries"] += int(any(label <= 0 for _, label in kg_items))

        seen_doc_ids = {doc_id for doc_id, _ in kg_items}
        positives = [doc_id for doc_id, label in kg_items if label > 0]
        same_mid_added = 0
        for pos_doc_id in positives:
            mid = pos_doc_id.rsplit("_", 1)[0] if "_" in pos_doc_id else pos_doc_id
            for cand_id in by_mid.get(mid, []):
                if cand_id in seen_doc_ids:
                    continue
                kg_items.append((cand_id, 0))
                seen_doc_ids.add(cand_id)
                same_mid_added += 1
                if same_mid_added >= int(max_same_mid_negatives):
                    break
            if same_mid_added >= int(max_same_mid_negatives):
                break
        stats["same_mid_negatives"] += int(same_mid_added)

        for doc_id, label in kg_items:
            x_parts.append(
                build_kg_consistency_features(
                    query_text=query,
                    doc_text=doc_text_by_id.get(doc_id, ""),
                    doc_id=doc_id,
                )
            )
            y_parts.append(int(label))

    if not x_parts:
        return np.zeros((0, len(KG_CONSISTENCY_FEATURE_NAMES)), dtype=np.float32), np.zeros((0,), dtype=np.int64), stats
    x = np.vstack(x_parts).astype(np.float32)
    y = np.asarray(y_parts, dtype=np.int64)
    stats["examples"] = int(y.size)
    stats["positives"] = int(np.sum(y))
    stats["negatives"] = int(y.size - np.sum(y))
    return x, y, stats


def make_model(seed: int) -> Pipeline:
    return Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            (
                "clf",
                SGDClassifier(
                    loss="log_loss",
                    alpha=3e-5,
                    penalty="elasticnet",
                    l1_ratio=0.08,
                    class_weight="balanced",
                    max_iter=1500,
                    tol=1e-4,
                    random_state=int(seed),
                ),
            ),
        ]
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Train KG entity-relation consistency verifier")
    parser.add_argument("--train-file", type=Path, required=True)
    parser.add_argument("--dev-file", type=Path, required=True)
    parser.add_argument("--corpus-file", type=Path, required=True)
    parser.add_argument("--out-bundle", type=Path, required=True)
    parser.add_argument("--out-metrics", type=Path, required=True)
    parser.add_argument("--max-train", type=int, default=0)
    parser.add_argument("--max-dev", type=int, default=0)
    parser.add_argument("--max-same-mid-negatives", type=int, default=8)
    parser.add_argument("--seed", type=int, default=23)
    parser.add_argument("--allow-test-split-training", action="store_true")
    args = parser.parse_args()

    allow_test_training = bool(args.allow_test_split_training) or os.environ.get("UNIFUSION_ALLOW_TEST_SPLIT_TRAINING", "0") == "1"
    if not allow_test_training:
        for label, path in (("train-file", args.train_file), ("dev-file", args.dev_file)):
            if is_test_like_path(Path(path)):
                raise ValueError(
                    f"{label} looks like a test split ({path}). "
                    "KG verifier training must use train/dev only."
                )
    if int(args.max_same_mid_negatives) < 0:
        raise ValueError("max-same-mid-negatives must be >= 0")

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

    print(f"[stage] loading corpus: {args.corpus_file}", flush=True)
    corpus = json.loads(args.corpus_file.read_text(encoding="utf-8"))
    doc_text_by_id, by_mid = build_doc_maps(corpus)
    print(f"[stage] corpus docs={len(doc_text_by_id)} kg_mids={len(by_mid)}", flush=True)

    print("[stage] building KG train examples", flush=True)
    x_train, y_train, train_stats = build_examples(
        rows=train_rows,
        doc_text_by_id=doc_text_by_id,
        by_mid=by_mid,
        max_same_mid_negatives=int(args.max_same_mid_negatives),
    )
    print("[stage] building KG dev examples", flush=True)
    x_dev, y_dev, dev_stats = build_examples(
        rows=dev_rows,
        doc_text_by_id=doc_text_by_id,
        by_mid=by_mid,
        max_same_mid_negatives=int(args.max_same_mid_negatives),
    )
    if x_train.size == 0 or len(set(y_train.tolist())) < 2:
        raise RuntimeError("KG verifier training data is empty or has a single class")
    if x_dev.size == 0:
        raise RuntimeError("KG verifier dev data is empty")

    print(f"[stage] fitting KG verifier examples={x_train.shape[0]} positives={int(y_train.sum())}", flush=True)
    model = make_model(seed=int(args.seed))
    model.fit(x_train, y_train)
    dev_prob = model.predict_proba(x_dev)[:, 1]
    metrics = {
        "method_name": "Entity-Relation Consistency Verifier",
        "method_formulation": "P(y=1 | query, KG evidence) learned from train/dev KG qrels with same-entity hard negatives",
        "feature_names": KG_CONSISTENCY_FEATURE_NAMES,
        "train": train_stats,
        "dev": dev_stats,
        "split_guard": {
            "train_file": str(args.train_file),
            "dev_file": str(args.dev_file),
            "corpus_file": str(args.corpus_file),
            "train_queries": int(len(train_ids)),
            "dev_queries": int(len(dev_ids)),
            "train_dev_overlap": int(len(overlap)),
            "test_like_paths_allowed": bool(allow_test_training),
        },
        "dev_average_precision": float(average_precision_score(y_dev, dev_prob)),
        "dev_roc_auc": float(roc_auc_score(y_dev, dev_prob)) if len(set(y_dev.tolist())) > 1 else None,
        "config": {
            "max_same_mid_negatives": int(args.max_same_mid_negatives),
            "seed": int(args.seed),
        },
    }
    bundle = KGConsistencyBundle(model=model, feature_names=list(KG_CONSISTENCY_FEATURE_NAMES), metadata=metrics)
    save_kg_consistency_bundle(bundle, args.out_bundle)
    args.out_metrics.parent.mkdir(parents=True, exist_ok=True)
    args.out_metrics.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(metrics, ensure_ascii=False, indent=2))
    print(f"[OK] bundle -> {args.out_bundle}")
    print(f"[OK] metrics -> {args.out_metrics}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
