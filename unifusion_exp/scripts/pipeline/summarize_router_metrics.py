#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
import json
import csv


def main() -> int:
    parser = argparse.ArgumentParser(description="Summarize router metric JSON files into CSV")
    parser.add_argument("--metrics-dir", type=Path, required=True)
    parser.add_argument("--out-csv", type=Path, required=True)
    args = parser.parse_args()

    files = sorted(args.metrics_dir.glob("*_metrics.json"))
    if not files:
        raise FileNotFoundError(f"No metric files in {args.metrics_dir}")

    rows = []
    for fp in files:
        data = json.loads(fp.read_text(encoding="utf-8"))
        rows.append(
            {
                "run_id": data.get("run_id"),
                "threshold": data.get("threshold"),
                "val_micro_f1": data.get("val_micro_f1"),
                "val_subset_acc": data.get("val_subset_acc"),
                "test_micro_f1": data.get("test_micro_f1"),
                "test_subset_acc": data.get("test_subset_acc"),
            }
        )

    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    with args.out_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["run_id", "threshold", "val_micro_f1", "val_subset_acc", "test_micro_f1", "test_subset_acc"],
        )
        writer.writeheader()
        writer.writerows(rows)

    print(f"[OK] wrote {len(rows)} rows -> {args.out_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
