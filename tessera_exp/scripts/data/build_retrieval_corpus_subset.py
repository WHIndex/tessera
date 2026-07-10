#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
import json
import random

import ijson


def load_split(path: Path):
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def collect_relevant_ids(rows):
    ids = set()
    for r in rows:
        rc = r.get("relevant_chunks", {})
        for chunk_id, label in rc.items():
            try:
                if float(label) > 0:
                    ids.add(chunk_id)
            except Exception:
                continue
    return ids


def main() -> int:
    parser = argparse.ArgumentParser(description="Build subset corpus for retrieval experiments")
    parser.add_argument("--mmrag-root", type=Path, required=True)
    parser.add_argument("--processed-docs", type=Path, required=True)
    parser.add_argument("--out-file", type=Path, required=True)
    parser.add_argument(
        "--mode",
        type=str,
        choices=["qrel_augmented", "random_only"],
        default="qrel_augmented",
        help="qrel_augmented: include positive qrels + random negatives; random_only: random corpus without qrel injection",
    )
    parser.add_argument("--extra-negatives", type=int, default=50000)
    parser.add_argument(
        "--relevant-splits",
        type=str,
        default="dev",
        help="Comma-separated splits used to collect positive qrels: train,dev,test (default: dev)",
    )
    parser.add_argument(
        "--target-size",
        type=int,
        default=0,
        help="Only for random_only mode. Number of chunks to sample into output corpus.",
    )
    parser.add_argument(
        "--meta-file",
        type=Path,
        default=None,
        help="Optional json file to save construction metadata and counts",
    )
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)

    selected = [s.strip().lower() for s in args.relevant_splits.split(",") if s.strip()]
    needed_ids = set()
    if args.mode == "qrel_augmented":
        split_rows = {
            "train": load_split(args.mmrag_root / "mmrag_train.json"),
            "dev": load_split(args.mmrag_root / "mmrag_dev.json"),
            "test": load_split(args.mmrag_root / "mmrag_test.json"),
        }
        for s in selected:
            if s not in split_rows:
                raise ValueError(f"Invalid split in --relevant-splits: {s}")
        for s in selected:
            needed_ids |= collect_relevant_ids(split_rows[s])
        print(f"[stage] mode=qrel_augmented selected_splits={selected} unique_positive_qrels={len(needed_ids)}")
    else:
        if args.target_size <= 0:
            raise ValueError("--target-size must be > 0 when --mode=random_only")
        print(f"[stage] mode=random_only target_size={args.target_size}")

    kept = []
    negatives = []
    total_docs = 0

    with args.processed_docs.open("rb") as f:
        for obj in ijson.items(f, "item"):
            total_docs += 1
            doc_id = obj.get("id")
            text = obj.get("text", "")
            if not doc_id or not text:
                continue

            if args.mode == "qrel_augmented" and doc_id in needed_ids:
                kept.append({"id": doc_id, "text": text})
                continue

            limit = args.extra_negatives if args.mode == "qrel_augmented" else args.target_size
            if len(negatives) < limit:
                negatives.append({"id": doc_id, "text": text})
            else:
                # reservoir-ish replacement for diversity
                j = random.randint(0, len(kept) + len(negatives))
                if j < len(negatives):
                    negatives[j] = {"id": doc_id, "text": text}

    corpus = kept + negatives if args.mode == "qrel_augmented" else negatives
    args.out_file.parent.mkdir(parents=True, exist_ok=True)
    args.out_file.write_text(json.dumps(corpus, ensure_ascii=False), encoding="utf-8")

    meta = {
        "mode": args.mode,
        "selected_splits": selected,
        "total_docs_scanned": total_docs,
        "positive_qrels_unique": len(needed_ids),
        "kept_relevant": len(kept),
        "negatives": len(negatives),
        "total": len(corpus),
        "seed": args.seed,
    }

    print(f"[OK] kept relevant={len(kept)}, negatives={len(negatives)}, total={len(corpus)}")
    print(f"[OK] saved -> {args.out_file}")
    if args.meta_file is not None:
        args.meta_file.parent.mkdir(parents=True, exist_ok=True)
        args.meta_file.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"[OK] meta -> {args.meta_file}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
