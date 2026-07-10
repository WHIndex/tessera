#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from tessera_exp.utils import (
    ensure_dir,
    infer_modalities_from_dataset_score,
    infer_modalities_from_relevant_chunks,
    modality_multihot,
    read_json,
    write_json,
)


def convert_split(input_path: Path, output_path: Path, max_samples: int | None = None) -> dict:
    rows = read_json(input_path)
    if max_samples is not None:
        rows = rows[:max_samples]

    converted = []
    for row in rows:
        relevant_chunks = row.get("relevant_chunks", {})
        dataset_score = row.get("dataset_score", {})
        labels = infer_modalities_from_dataset_score(dataset_score) if isinstance(dataset_score, dict) else []
        if not labels:
            labels = infer_modalities_from_relevant_chunks(relevant_chunks)
        converted.append(
            {
                "id": row.get("id"),
                "query": row.get("query", ""),
                "answer": row.get("answer", ""),
                "labels": labels,
                "labels_multihot": modality_multihot(labels),
                "dataset_score": dataset_score,
            }
        )

    ensure_dir(output_path.parent)
    write_json(output_path, converted)

    combo_count: dict[str, int] = {}
    for item in converted:
        key = "+".join(item["labels"]) if item["labels"] else "none"
        combo_count[key] = combo_count.get(key, 0) + 1

    return {
        "input": str(input_path),
        "output": str(output_path),
        "rows": len(converted),
        "label_combo_distribution": combo_count,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Build router dataset from mmRAG split files")
    parser.add_argument("--mmrag-root", type=Path, required=True, help="Path to mmRAG_ds directory")
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--max-train", type=int, default=None)
    parser.add_argument("--max-val", type=int, default=None)
    parser.add_argument("--max-test", type=int, default=None)
    args = parser.parse_args()

    split_map = {
        "train": (args.mmrag_root / "mmrag_train.json", args.max_train),
        "val": (args.mmrag_root / "mmrag_dev.json", args.max_val),
        "test": (args.mmrag_root / "mmrag_test.json", args.max_test),
    }

    summary = {}
    for split, (input_path, max_samples) in split_map.items():
        if not input_path.exists():
            raise FileNotFoundError(f"Missing split file: {input_path}")
        output_path = args.out_dir / f"router_{split}.json"
        summary[split] = convert_split(input_path, output_path, max_samples=max_samples)
        print(f"[OK] {split}: {summary[split]['rows']} -> {output_path}")

    summary_path = args.out_dir / "router_dataset_summary.json"
    write_json(summary_path, summary)
    print(f"[OK] summary: {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
