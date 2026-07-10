#!/usr/bin/env python3
"""检查 TESSERA 实验关键资源是否就绪。"""

from __future__ import annotations

import argparse
import os
from pathlib import Path


def check_path(label: str, path: str) -> tuple[bool, str, str]:
    p = Path(path)
    if p.exists():
        kind = "dir" if p.is_dir() else "file"
        return True, label, f"{kind}: {p}"
    return False, label, f"missing: {p}"


def main() -> int:
    parser = argparse.ArgumentParser(description="Check required resources for TESSERA preflight")
    parser.add_argument("--strict", action="store_true", help="如果有缺失则返回非零")
    args = parser.parse_args()

    required = {
        "RESOURCE_ROOT": os.getenv("RESOURCE_ROOT", ""),
        "MMRAG_ROOT": os.getenv("MMRAG_ROOT", ""),
        "MMRAG_DATA_ROOT": os.getenv("MMRAG_DATA_ROOT", ""),
        "CARP_ROOT": os.getenv("CARP_ROOT", ""),
        "QUASAR_ROOT": os.getenv("QUASAR_ROOT", ""),
        "TABLERAG_ROOT": os.getenv("TABLERAG_ROOT", ""),
        "SIMKGC_ROOT": os.getenv("SIMKGC_ROOT", ""),
    }

    optional = {
        "E5_MODEL_DIR": os.getenv("E5_MODEL_DIR", ""),
        "DEBERTA_MODEL_DIR": os.getenv("DEBERTA_MODEL_DIR", ""),
        "WIKIDATA5M_ROOT": os.getenv("WIKIDATA5M_ROOT", ""),
    }

    missing_required = 0

    print("[check_resources] Required")
    for key, value in required.items():
        if not value:
            print(f"[ERR] {key}: env not set")
            missing_required += 1
            continue
        ok, label, msg = check_path(key, value)
        prefix = "[OK]" if ok else "[ERR]"
        print(f"{prefix} {label}: {msg}")
        if not ok:
            missing_required += 1

    print("\n[check_resources] Optional")
    for key, value in optional.items():
        if not value:
            print(f"[WARN] {key}: env not set")
            continue
        ok, label, msg = check_path(key, value)
        prefix = "[OK]" if ok else "[WARN]"
        print(f"{prefix} {label}: {msg}")

    if missing_required > 0:
        print(f"\n[summary] missing required: {missing_required}")
        return 2 if args.strict else 0

    print("\n[summary] all required resources exist")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
