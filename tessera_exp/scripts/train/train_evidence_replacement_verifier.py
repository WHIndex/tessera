#!/usr/bin/env python3
from __future__ import annotations

import argparse
from collections import Counter
import json
import os
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from tessera_exp.e2e.pairwise_slot_verifier import (  # noqa: E402
    PESV_FEATURE_NAMES,
    PairwiseSlotVerifierBundle,
    save_pairwise_slot_verifier_bundle,
)

from train_pairwise_slot_verifier import (  # noqa: E402
    choose_enabled_families,
    collect_doc_ids,
    evaluate,
    evaluate_by_family,
    fit,
    load_doc_token_map,
    read_examples,
    split_by_query,
    tune,
)


def looks_test_like(path: Path) -> bool:
    raw = str(path).lower()
    return "test" in path.name.lower() or "/test" in raw or "\\test" in raw


def parse_slots(raw: str) -> list[int]:
    slots: list[int] = []
    for part in str(raw or "").split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            left, right = part.split("-", 1)
            start, end = int(left), int(right)
            slots.extend(range(min(start, end), max(start, end) + 1))
        else:
            slots.append(int(part))
    out = sorted({slot for slot in slots if slot >= 1})
    if not out:
        raise ValueError("old-slots must contain at least one 1-based slot")
    return out


def read_multislot_examples(
    path: Path,
    *,
    old_slots: list[int],
    candidate_start: int,
    candidate_end: int,
    min_gain: float,
    doc_token_map: dict[str, set[str]],
) -> tuple[np.ndarray, np.ndarray, list[dict]]:
    xs: list[np.ndarray] = []
    ys: list[np.ndarray] = []
    metas: list[dict] = []
    for old_slot in old_slots:
        x, y, meta = read_examples(
            path,
            old_slot,
            candidate_start,
            candidate_end,
            min_gain,
            doc_token_map,
        )
        for row in meta:
            row["old_slot"] = int(old_slot)
        if x.size:
            xs.append(x)
            ys.append(y)
            metas.extend(meta)
    if not xs:
        return (
            np.zeros((0, len(PESV_FEATURE_NAMES)), dtype=np.float32),
            np.zeros((0,), dtype=np.int64),
            [],
        )
    return np.vstack(xs).astype(np.float32), np.concatenate(ys).astype(np.int64), metas


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Train Evidence Replacement Verifier from train/dev TESSERA rankings_debug traces."
    )
    parser.add_argument("--train-debug-jsonl", type=Path, required=True)
    parser.add_argument("--val-debug-jsonl", type=Path, default=None)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--run-id", default="tessera_erv_v1")
    parser.add_argument("--old-slots", default="2,3,4,5")
    parser.add_argument("--candidate-start", type=int, default=6)
    parser.add_argument("--candidate-end", type=int, default=25)
    parser.add_argument("--min-gain", type=float, default=0.003)
    parser.add_argument("--min-precision", type=float, default=0.72)
    parser.add_argument("--corpus-file", type=Path, default=None)
    parser.add_argument("--min-family-precision", type=float, default=0.52)
    parser.add_argument("--min-family-predictions", type=int, default=2)
    parser.add_argument("--enable-families", default="")
    parser.add_argument("--val-ratio", type=float, default=0.20)
    parser.add_argument("--seed", type=int, default=29)
    parser.add_argument("--allow-test-split-training", action="store_true")
    args = parser.parse_args()

    allow_test_training = bool(args.allow_test_split_training) or os.environ.get(
        "TESSERA_ALLOW_TEST_SPLIT_TRAINING", "0"
    ) == "1"
    if not allow_test_training:
        for label, path in (("train-debug-jsonl", args.train_debug_jsonl), ("val-debug-jsonl", args.val_debug_jsonl)):
            if path is not None and looks_test_like(Path(path)):
                raise ValueError(f"{label} looks like a test split ({path}); ERV must use train/dev traces only.")

    old_slots = parse_slots(str(args.old_slots))
    args.out_dir.mkdir(parents=True, exist_ok=True)

    needed_doc_ids = collect_doc_ids(args.train_debug_jsonl, args.candidate_end)
    if args.val_debug_jsonl is not None:
        needed_doc_ids.update(collect_doc_ids(args.val_debug_jsonl, args.candidate_end))
    doc_token_map = load_doc_token_map(args.corpus_file, needed_doc_ids)
    print(
        f"[stage] content tokens loaded: {len(doc_token_map)}/{len(needed_doc_ids)} docs "
        f"from {args.corpus_file if args.corpus_file is not None else 'no corpus'}",
        file=sys.stderr,
        flush=True,
    )

    x_train, y_train, train_meta = read_multislot_examples(
        args.train_debug_jsonl,
        old_slots=old_slots,
        candidate_start=int(args.candidate_start),
        candidate_end=int(args.candidate_end),
        min_gain=float(args.min_gain),
        doc_token_map=doc_token_map,
    )
    if args.val_debug_jsonl is not None:
        x_val, y_val, val_meta = read_multislot_examples(
            args.val_debug_jsonl,
            old_slots=old_slots,
            candidate_start=int(args.candidate_start),
            candidate_end=int(args.candidate_end),
            min_gain=float(args.min_gain),
            doc_token_map=doc_token_map,
        )
    else:
        x_train, y_train, train_meta, x_val, y_val, val_meta = split_by_query(
            x_train,
            y_train,
            train_meta,
            val_ratio=float(args.val_ratio),
            seed=int(args.seed),
        )

    train_queries = {str(row.get("query_id", "")) for row in train_meta if str(row.get("query_id", ""))}
    val_queries = {str(row.get("query_id", "")) for row in val_meta if str(row.get("query_id", ""))}
    overlap = sorted(train_queries & val_queries)
    if overlap:
        raise ValueError(f"train/val query overlap detected: {len(overlap)} examples: {overlap[:5]}")
    if x_train.size == 0 or x_val.size == 0:
        raise RuntimeError("ERV train/val data is empty")

    model = fit(x_train, y_train)
    threshold, val_metrics = tune(model, x_val, y_val, float(args.min_precision))
    train_metrics = evaluate(model, x_train, y_train, threshold)
    val_family_metrics = evaluate_by_family(model, x_val, y_val, val_meta, threshold)
    train_family_metrics = evaluate_by_family(model, x_train, y_train, train_meta, threshold)
    enabled_families = choose_enabled_families(
        val_family_metrics,
        min_precision=float(args.min_family_precision),
        min_predictions=int(args.min_family_predictions),
        override=str(args.enable_families or ""),
    )

    metadata = {
        "run_id": args.run_id,
        "method_name": "Evidence Replacement Verifier",
        "method_formulation": (
            "P(replace old evidence with candidate improves truncated retrieval utility | "
            "query, source, rank, lexical, family and redundancy features)"
        ),
        "feature_version": 1,
        "recommended_threshold": float(threshold),
        "enabled_families": enabled_families,
        "old_slots": [int(x) for x in old_slots],
        "candidate_start": int(args.candidate_start),
        "candidate_end": int(args.candidate_end),
        "min_gain": float(args.min_gain),
        "train_debug_jsonl": str(args.train_debug_jsonl),
        "val_debug_jsonl": str(args.val_debug_jsonl) if args.val_debug_jsonl is not None else None,
        "corpus_file": str(args.corpus_file) if args.corpus_file is not None else None,
        "needed_doc_ids": int(len(needed_doc_ids)),
        "loaded_doc_tokens": int(len(doc_token_map)),
        "train_examples": int(len(y_train)),
        "val_examples": int(len(y_val)),
        "train_queries": int(len(train_queries)),
        "val_queries": int(len(val_queries)),
        "train_dev_overlap": int(len(overlap)),
        "train_label_distribution": dict(Counter(int(v) for v in y_train.tolist())),
        "val_label_distribution": dict(Counter(int(v) for v in y_val.tolist())),
        "family_gate_policy": {
            "min_family_precision": float(args.min_family_precision),
            "min_family_predictions": int(args.min_family_predictions),
            "override": str(args.enable_families or ""),
        },
    }
    bundle = PairwiseSlotVerifierBundle(
        model=model,
        feature_names=list(PESV_FEATURE_NAMES),
        metadata=metadata,
    )
    bundle_path = args.out_dir / "evidence_replacement_verifier.pkl"
    save_pairwise_slot_verifier_bundle(bundle, bundle_path)
    metrics = {
        "run_id": args.run_id,
        "bundle": str(bundle_path),
        "recommended_threshold": float(threshold),
        "enabled_families": enabled_families,
        "train": train_metrics,
        "val": val_metrics,
        "train_by_family": train_family_metrics,
        "val_by_family": val_family_metrics,
        **metadata,
    }
    metrics_path = args.out_dir / f"{args.run_id}_metrics.json"
    metrics_path.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(metrics, ensure_ascii=False, indent=2))
    print(f"[OK] bundle -> {bundle_path}")
    print(f"[OK] metrics -> {metrics_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
