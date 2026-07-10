#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
import string
from collections import Counter, defaultdict
from pathlib import Path


NUMERIC_QUERY_HINTS = [
    "how many",
    "how much",
    "percentage",
    "percent",
    "rate",
    "growth",
    "capacity",
    "population",
    "revenue",
    "average",
    "sum",
    "total",
    "difference",
    "ratio",
    "what year",
    "which year",
    "when",
]

ALIAS_QUERY_HINTS = [
    "also known as",
    "aka",
    "real name",
    "stage name",
    "nickname",
    "born as",
]


def normalize_answer(s: str) -> str:
    s = s.lower()
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    s = "".join(ch for ch in s if ch not in set(string.punctuation))
    s = " ".join(s.split())
    return s


def exact_match(pred: str, gold: str) -> bool:
    return normalize_answer(pred) == normalize_answer(gold)


def load_rows(path: Path) -> dict[str, dict]:
    rows = json.loads(path.read_text(encoding="utf-8"))
    out: dict[str, dict] = {}
    for row in rows:
        qid = row.get("id")
        if qid is None:
            continue
        out[str(qid)] = row
    return out


def load_prediction_jsonl(path: Path) -> dict[str, str]:
    out: dict[str, str] = {}
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            qid = row.get("id")
            if qid is None:
                continue
            out[str(qid)] = str(row.get("prediction", row.get("pred", row.get("answer", ""))))
    return out


def infer_unifusion_error_type(row: dict) -> str:
    query = str(row.get("query", "")).lower()
    dataset_score = row.get("dataset_score", {}) or {}
    positive_modalities = 0
    for v in dataset_score.values():
        if isinstance(v, (int, float)) and v > 0:
            positive_modalities += 1

    if any(k in query for k in NUMERIC_QUERY_HINTS):
        return "numeric_reasoning"
    if any(k in query for k in ALIAS_QUERY_HINTS):
        return "alias_disambiguation"
    if positive_modalities >= 2:
        return "cross_modal_conflict"
    return "other_unifusion_errors"


def main() -> int:
    parser = argparse.ArgumentParser(description="Build reproducible error typology for UniFusion vs Dense predictions")
    parser.add_argument("--gold-file", type=Path, required=True)
    parser.add_argument("--dense-pred-file", type=Path, required=True)
    parser.add_argument("--unifusion-pred-file", type=Path, required=True)
    parser.add_argument("--out-json", type=Path, required=True)
    parser.add_argument("--out-md", type=Path, required=True)
    parser.add_argument("--examples-per-type", type=int, default=3)
    args = parser.parse_args()

    rows_by_id = load_rows(args.gold_file)
    dense_pred = load_prediction_jsonl(args.dense_pred_file)
    uni_pred = load_prediction_jsonl(args.unifusion_pred_file)

    ids = sorted(set(rows_by_id.keys()) & set(dense_pred.keys()) & set(uni_pred.keys()))
    if not ids:
        raise SystemExit("No overlapping ids among gold/dense/unifusion predictions")

    outcome_counts = Counter()
    uni_error_type_counts = Counter()
    uni_error_examples: dict[str, list[dict]] = defaultdict(list)

    for qid in ids:
        row = rows_by_id[qid]
        gold = str(row.get("answer", ""))
        dense = dense_pred[qid]
        uni = uni_pred[qid]

        dense_ok = exact_match(dense, gold)
        uni_ok = exact_match(uni, gold)

        if dense_ok and uni_ok:
            outcome = "both_correct"
        elif (not dense_ok) and uni_ok:
            outcome = "uni_correct_dense_wrong"
        elif dense_ok and (not uni_ok):
            outcome = "dense_correct_uni_wrong"
        else:
            outcome = "both_wrong"
        outcome_counts[outcome] += 1

        if not uni_ok:
            err_type = infer_unifusion_error_type(row)
            uni_error_type_counts[err_type] += 1
            if len(uni_error_examples[err_type]) < int(args.examples_per_type):
                uni_error_examples[err_type].append(
                    {
                        "id": qid,
                        "query": str(row.get("query", "")),
                        "gold": gold,
                        "dense_prediction": dense,
                        "unifusion_prediction": uni,
                    }
                )

    total = len(ids)
    uni_error_total = outcome_counts["dense_correct_uni_wrong"] + outcome_counts["both_wrong"]

    out = {
        "meta": {
            "evaluated": total,
            "gold_total": len(rows_by_id),
            "dense_pred_total": len(dense_pred),
            "unifusion_pred_total": len(uni_pred),
            "coverage": float(total / max(1, len(rows_by_id))),
            "examples_per_type": int(args.examples_per_type),
            "error_type_rule": {
                "numeric_reasoning": "query contains numeric/computation hint keywords",
                "alias_disambiguation": "query contains alias/disambiguation hint keywords",
                "cross_modal_conflict": "dataset_score has >=2 positive modality sources",
                "other_unifusion_errors": "none of the above",
            },
        },
        "outcome_counts": dict(outcome_counts),
        "outcome_ratios": {
            k: float(v / max(1, total)) for k, v in outcome_counts.items()
        },
        "unifusion_error_total": int(uni_error_total),
        "unifusion_error_type_counts": dict(uni_error_type_counts),
        "unifusion_error_type_ratios": {
            k: float(v / max(1, uni_error_total)) for k, v in uni_error_type_counts.items()
        },
        "examples": dict(uni_error_examples),
    }

    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")

    md = [
        "# UniFusion Error Typology",
        "",
        f"- evaluated: {out['meta']['evaluated']}",
        f"- coverage: {out['meta']['coverage']:.4f}",
        f"- unifusion_error_total: {out['unifusion_error_total']}",
        "",
        "## Outcome Summary",
        "",
        "| Outcome | Count | Ratio |",
        "|---|---:|---:|",
    ]

    for k in ["both_correct", "uni_correct_dense_wrong", "dense_correct_uni_wrong", "both_wrong"]:
        md.append(
            f"| {k} | {out['outcome_counts'].get(k, 0)} | {out['outcome_ratios'].get(k, 0.0):.4f} |"
        )

    md.extend(
        [
            "",
            "## UniFusion Error Type Breakdown",
            "",
            "| Error Type | Count | Ratio (among UniFusion errors) |",
            "|---|---:|---:|",
        ]
    )

    for k in ["cross_modal_conflict", "numeric_reasoning", "alias_disambiguation", "other_unifusion_errors"]:
        md.append(
            f"| {k} | {out['unifusion_error_type_counts'].get(k, 0)} | {out['unifusion_error_type_ratios'].get(k, 0.0):.4f} |"
        )

    for k in ["cross_modal_conflict", "numeric_reasoning", "alias_disambiguation", "other_unifusion_errors"]:
        examples = out["examples"].get(k, [])
        if not examples:
            continue
        md.extend(["", f"## Examples: {k}", ""])
        for ex in examples:
            md.append(f"- id: {ex['id']}")
            md.append(f"  - query: {ex['query']}")
            md.append(f"  - gold: {ex['gold']}")
            md.append(f"  - dense: {ex['dense_prediction']}")
            md.append(f"  - unifusion: {ex['unifusion_prediction']}")

    args.out_md.parent.mkdir(parents=True, exist_ok=True)
    args.out_md.write_text("\n".join(md) + "\n", encoding="utf-8")

    print(f"[OK] json -> {args.out_json}")
    print(f"[OK] markdown -> {args.out_md}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
