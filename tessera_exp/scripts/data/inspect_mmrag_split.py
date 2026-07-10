#!/usr/bin/env python3
"""检查 mmRAG 数据划分与 JSONL 可读性。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable


def count_jsonl(path: Path) -> tuple[int, int]:
    count = 0
    bad = 0
    if not path.exists():
        return count, -1

    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                json.loads(line)
                count += 1
            except json.JSONDecodeError:
                bad += 1
    return count, bad


def count_json(path: Path) -> tuple[int, str]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as e:
        return -1, f"json decode error: {e}"

    if isinstance(data, list):
        return len(data), "list"
    if isinstance(data, dict):
        # mmRAG 不同版本可能是 dict 包裹样本列表
        for key in ["data", "items", "samples", "examples"]:
            value = data.get(key)
            if isinstance(value, list):
                return len(value), f"dict[{key}]"
        return 1, "dict"
    return 1, type(data).__name__


def candidate_files(root: Path, split: str) -> Iterable[Path]:
    names = [
        f"{split}.jsonl",
        f"mmrag_{split}.jsonl",
        f"mmrag_{split}.json",
        f"{split}.json",
    ]

    # mmRAG 官方常用 dev 命名
    if split == "val":
        names.extend(["dev.jsonl", "mmrag_dev.jsonl", "dev.json", "mmrag_dev.json"])

    for n in names:
        p = root / n
        if p.exists():
            yield p


def main() -> int:
    parser = argparse.ArgumentParser(description="Inspect mmRAG split files")
    parser.add_argument("--root", type=Path, required=True, help="mmRAG_ds directory")
    args = parser.parse_args()

    root = args.root
    if not root.exists():
        print(f"[ERR] root not found: {root}")
        return 1

    print(f"[inspect] root={root}")

    found_any = False
    for split in ["train", "val", "test"]:
        found = list(candidate_files(root, split))
        if not found:
            print(f"[WARN] split={split}: no candidate file found")
            continue

        found_any = True
        for fp in found:
            if fp.suffix == ".jsonl":
                count, bad = count_jsonl(fp)
                status = "OK" if bad == 0 else "WARN"
                print(f"[{status}] {fp.name}: rows={count}, bad_lines={bad}")
            else:
                count, kind = count_json(fp)
                status = "OK" if count >= 0 else "WARN"
                print(f"[{status}] {fp.name}: rows={count}, type={kind}")

    if not found_any:
        print("[ERR] no split files discovered")
        return 2

    print("[inspect] done")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
