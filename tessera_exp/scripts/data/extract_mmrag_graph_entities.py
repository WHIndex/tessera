#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from tessera_exp.utils import read_json, write_json, ensure_dir


def collect_graph_entities(rows, max_samples: int | None = None) -> set[str]:
    if max_samples is not None:
        rows = rows[:max_samples]

    entities: set[str] = set()
    for row in rows:
        relevant_chunks = row.get("relevant_chunks", {})
        for chunk_id, label in relevant_chunks.items():
            try:
                if float(label) <= 0:
                    continue
            except Exception:
                continue
            if chunk_id.startswith("m.") or chunk_id.startswith("g."):
                entity_id = chunk_id.rsplit("_", 1)[0]
                entities.add(entity_id)
    return entities


def main() -> int:
    parser = argparse.ArgumentParser(description="Extract graph entity IDs from mmRAG relevant_chunks")
    parser.add_argument("--mmrag-root", type=Path, required=True)
    parser.add_argument("--out-file", type=Path, required=True)
    parser.add_argument("--max-train", type=int, default=None)
    parser.add_argument("--max-dev", type=int, default=None)
    parser.add_argument("--max-test", type=int, default=None)
    args = parser.parse_args()

    split_files = {
        "train": (args.mmrag_root / "mmrag_train.json", args.max_train),
        "dev": (args.mmrag_root / "mmrag_dev.json", args.max_dev),
        "test": (args.mmrag_root / "mmrag_test.json", args.max_test),
    }

    all_entities: set[str] = set()
    split_counts = {}

    for split, (path, max_samples) in split_files.items():
        if not path.exists():
            raise FileNotFoundError(f"Missing file: {path}")
        rows = read_json(path)
        entities = collect_graph_entities(rows, max_samples=max_samples)
        split_counts[split] = len(entities)
        all_entities.update(entities)
        print(f"[OK] {split}: unique_graph_entities={len(entities)}")

    out = {
        "total_unique_entities": len(all_entities),
        "split_unique_entities": split_counts,
        "entity_ids": sorted(all_entities),
    }
    ensure_dir(args.out_file.parent)
    write_json(args.out_file, out)
    print(f"[OK] saved -> {args.out_file}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
