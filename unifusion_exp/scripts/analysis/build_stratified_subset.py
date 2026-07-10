#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import random
import re
from collections import Counter, defaultdict
from pathlib import Path


def source_prefix(doc_id: str) -> str:
    x = str(doc_id)
    if "_" in x:
        return x.split("_", 1)[0]
    if x.startswith("m."):
        return "m"
    return x


def source_bucket(doc_id: str) -> str:
    x = str(doc_id)
    if x.startswith("m.") or x.startswith("/m/"):
        return "kg"
    p = source_prefix(x)
    if p in {"tat", "ott"}:
        return "table"
    if p in {"nq", "triviaqa", "hotpot", "squad", "newsqa"}:
        return "text"
    if p in {"kg", "wikidata", "wd", "cwq", "webqsp"}:
        return "kg"
    return "text"


def qrel_modality(row: dict) -> str:
    marks = {"text": 0, "table": 0, "kg": 0}
    for chunk_id, label in (row.get("relevant_chunks") or {}).items():
        try:
            if float(label) <= 0:
                continue
        except Exception:
            continue
        marks[source_bucket(str(chunk_id))] = 1

    s = int(sum(marks.values()))
    if s == 0:
        return "text_only"
    if s >= 2:
        return "multi_modal"
    if marks["table"] == 1:
        return "table_only"
    if marks["kg"] == 1:
        return "kg_only"
    return "text_only"


def infer_qa_target_type(query: str) -> str:
    q = str(query).strip().lower()
    if re.match(r"^(how many|how much|number of)\b", q):
        return "number"
    if re.match(r"^(when|what year|which year|in what year)\b", q):
        return "year"
    if re.match(r"^(who|which person|whose)\b", q):
        return "person"
    if re.match(r"^(where|which country|which city|in which)\b", q):
        return "location"
    if re.match(r"^(is|are|do|does|did|was|were|can|could|should)\b", q):
        return "boolean"
    if re.match(r"^(what|which)\b", q):
        return "entity"
    return "open"


def build_key(row: dict, stratify: str) -> str:
    m = qrel_modality(row)
    if stratify == "modality":
        return m
    t = infer_qa_target_type(str(row.get("query", "")))
    return f"{m}::{t}"


def proportional_allocate(
    total_needed: int,
    sizes: dict[str, int],
    min_per_stratum: int,
) -> dict[str, int]:
    keys = sorted(sizes.keys())
    total = int(sum(sizes.values()))
    if total_needed <= 0 or total <= 0:
        return {k: 0 for k in keys}

    alloc = {k: 0 for k in keys}

    # Optional floor to avoid empty strata in subset.
    if min_per_stratum > 0:
        for k in keys:
            if sizes[k] > 0 and total_needed > 0:
                take = min(min_per_stratum, sizes[k])
                alloc[k] = take
                total_needed -= take
        if total_needed < 0:
            # If min-per-stratum over-allocated, trim from largest strata allocations.
            overflow = -total_needed
            ordered = sorted(keys, key=lambda x: alloc[x], reverse=True)
            for k in ordered:
                if overflow <= 0:
                    break
                can_drop = max(0, alloc[k])
                drop = min(can_drop, overflow)
                alloc[k] -= drop
                overflow -= drop
            total_needed = 0

    remain_cap = {k: max(0, sizes[k] - alloc[k]) for k in keys}
    remain_total = int(sum(remain_cap.values()))
    if total_needed <= 0 or remain_total <= 0:
        return alloc

    raw = {k: total_needed * (remain_cap[k] / remain_total) for k in keys}
    floor_take = {k: int(math.floor(raw[k])) for k in keys}

    for k in keys:
        take = min(floor_take[k], remain_cap[k])
        alloc[k] += take

    used = int(sum(floor_take.values()))
    left = max(0, total_needed - used)
    # Largest remainder with capacity check.
    remainders = sorted(keys, key=lambda x: (raw[x] - floor_take[x]), reverse=True)
    idx = 0
    while left > 0 and idx < len(remainders) * 3:
        k = remainders[idx % len(remainders)]
        if alloc[k] < sizes[k]:
            alloc[k] += 1
            left -= 1
        idx += 1

    return alloc


def ratio(counter: Counter, n: int) -> dict[str, float]:
    if n <= 0:
        return {k: 0.0 for k in counter.keys()}
    return {k: float(counter[k] / n) for k in sorted(counter.keys())}


def main() -> int:
    parser = argparse.ArgumentParser(description="Build a stratified subset from mmRAG split JSON")
    parser.add_argument("--split-file", type=Path, required=True)
    parser.add_argument("--out-file", type=Path, required=True)
    parser.add_argument("--report-file", type=Path, default=None)
    parser.add_argument("--size", type=int, required=True)
    parser.add_argument("--seed", type=int, default=20260403)
    parser.add_argument("--stratify", choices=["modality", "modality_target"], default="modality")
    parser.add_argument("--min-per-stratum", type=int, default=1)
    args = parser.parse_args()

    rows = json.loads(args.split_file.read_text(encoding="utf-8"))
    n_total = len(rows)
    if n_total == 0:
        raise SystemExit("Empty split file")

    size = int(args.size)
    if size <= 0 or size > n_total:
        raise SystemExit(f"Invalid --size={size}; should be in [1, {n_total}]")

    strata: dict[str, list[int]] = defaultdict(list)
    for i, row in enumerate(rows):
        key = build_key(row, args.stratify)
        strata[key].append(i)

    strata_sizes = {k: len(v) for k, v in strata.items()}
    alloc = proportional_allocate(
        total_needed=size,
        sizes=strata_sizes,
        min_per_stratum=max(0, int(args.min_per_stratum)),
    )

    rng = random.Random(int(args.seed))
    picked: list[int] = []
    for k, idxs in strata.items():
        take = min(int(alloc.get(k, 0)), len(idxs))
        if take <= 0:
            continue
        picked.extend(rng.sample(idxs, take))

    if len(picked) < size:
        left = size - len(picked)
        pool = [i for i in range(n_total) if i not in set(picked)]
        picked.extend(rng.sample(pool, left))
    elif len(picked) > size:
        picked = rng.sample(picked, size)

    picked = sorted(picked)
    subset = [rows[i] for i in picked]

    args.out_file.parent.mkdir(parents=True, exist_ok=True)
    args.out_file.write_text(json.dumps(subset, ensure_ascii=False, indent=2), encoding="utf-8")

    full_mod = Counter(qrel_modality(r) for r in rows)
    sub_mod = Counter(qrel_modality(r) for r in subset)
    full_tgt = Counter(infer_qa_target_type(str(r.get("query", ""))) for r in rows)
    sub_tgt = Counter(infer_qa_target_type(str(r.get("query", ""))) for r in subset)

    report = {
        "meta": {
            "input_size": n_total,
            "subset_size": len(subset),
            "seed": int(args.seed),
            "stratify": str(args.stratify),
            "min_per_stratum": int(args.min_per_stratum),
            "out_file": str(args.out_file),
        },
        "strata_sizes": {k: int(v) for k, v in sorted(strata_sizes.items())},
        "allocated": {k: int(v) for k, v in sorted(alloc.items())},
        "distribution": {
            "full_modality_ratio": ratio(full_mod, n_total),
            "subset_modality_ratio": ratio(sub_mod, len(subset)),
            "full_target_ratio": ratio(full_tgt, n_total),
            "subset_target_ratio": ratio(sub_tgt, len(subset)),
        },
    }

    if args.report_file is not None:
        args.report_file.parent.mkdir(parents=True, exist_ok=True)
        args.report_file.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"[OK] subset -> {args.out_file}")
    if args.report_file is not None:
        print(f"[OK] report -> {args.report_file}")
    print("[dist] full modality:", report["distribution"]["full_modality_ratio"])
    print("[dist] sub  modality:", report["distribution"]["subset_modality_ratio"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
