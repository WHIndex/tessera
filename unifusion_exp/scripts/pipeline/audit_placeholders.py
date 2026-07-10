#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
import re


TOKENS = ["PREDICTED", "TODO", "TBD", "placeholder", "占位"]
DEFAULT_EXCLUDES = ["artifacts/**", "**/__pycache__/**", "scripts/pipeline/audit_placeholders.py"]


def scan_file(path: Path):
    text = path.read_text(encoding="utf-8", errors="ignore")
    counts = {}
    for t in TOKENS:
        counts[t] = len(re.findall(re.escape(t), text, flags=re.IGNORECASE))
    total = sum(counts.values())
    return total, counts


def main() -> int:
    parser = argparse.ArgumentParser(description="Audit placeholder tokens in docs/config/scripts")
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--out-json", type=Path, default=Path("artifacts/results/placeholder_audit_v1.json"))
    parser.add_argument("--out-md", type=Path, default=Path("artifacts/results/placeholder_audit_v1.md"))
    parser.add_argument("--exclude", type=str, nargs="*", default=DEFAULT_EXCLUDES)
    args = parser.parse_args()

    files = []
    for pat in ["docs/**/*.md", "README.md", "scripts/**/*.py", "configs/**/*"]:
        files.extend(args.root.glob(pat))

    items = []
    aggregate = {t: 0 for t in TOKENS}
    excludes = [str(args.root / p) for p in args.exclude]

    for fp in sorted(set(files)):
        if fp.is_dir():
            continue
        fp_str = str(fp)
        if any(fp.match(p.replace(str(args.root) + "/", "")) for p in excludes):
            continue
        total, counts = scan_file(fp)
        if total == 0:
            continue
        for t in TOKENS:
            aggregate[t] += counts[t]
        items.append({"path": str(fp), "total": total, "counts": counts})

    items.sort(key=lambda x: x["total"], reverse=True)

    out = {
        "tokens": TOKENS,
        "aggregate": aggregate,
        "files_with_placeholders": len(items),
        "top_files": items[:50],
    }

    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")

    md = [
        "# Placeholder Audit",
        "",
        "## Aggregate",
        "",
    ]
    for t in TOKENS:
        md.append(f"- {t}: {aggregate[t]}")

    md.append("")
    md.append("## Top Files")
    md.append("")
    md.append("| File | Total | PREDICTED | TODO | TBD | placeholder | 占位 |")
    md.append("|---|---:|---:|---:|---:|---:|---:|")
    for it in items[:50]:
        c = it["counts"]
        md.append(
            f"| {it['path']} | {it['total']} | {c['PREDICTED']} | {c['TODO']} | {c['TBD']} | {c['placeholder']} | {c['占位']} |"
        )

    args.out_md.write_text("\n".join(md) + "\n", encoding="utf-8")
    print(f"[OK] json -> {args.out_json}")
    print(f"[OK] md -> {args.out_md}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
