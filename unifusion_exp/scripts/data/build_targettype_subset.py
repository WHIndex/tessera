#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from unifusion_exp.e2e.objectives import infer_qa_target_type
from unifusion_exp.utils import ensure_dir, infer_modalities_from_relevant_chunks, read_json, write_json


RISK_KEYWORDS = {
    "how many",
    "how much",
    "total",
    "capacity",
    "population",
    "revenue",
    "profit",
    "loss",
    "rate",
    "gdp",
    "inflation",
    "largest",
    "smallest",
    "before",
    "after",
    "between",
    "compared",
    "versus",
    "vs",
}


def normalize_answer(text: str) -> str:
    text = str(text).lower()
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    text = "".join(ch for ch in text if ch not in set("!\"#$%&'()*+,-./:;<=>?@[\\]^_`{|}~"))
    text = " ".join(text.split())
    return text


def exact_match(pred: str, gold: str) -> float:
    return 1.0 if normalize_answer(pred) == normalize_answer(gold) else 0.0


def source_bucket(doc_id: str) -> str:
    if doc_id.startswith("m.") or doc_id.startswith("/m/") or doc_id.startswith("g."):
        return "kg"
    if "_" in doc_id:
        prefix = doc_id.split("_", 1)[0]
    else:
        prefix = doc_id
    if prefix in {"tat", "ott"}:
        return "table"
    if prefix in {"nq", "triviaqa", "hotpot", "squad", "newsqa"}:
        return "text"
    if prefix in {"kg", "wikidata", "wd"}:
        return "kg"
    return "text"


def load_pred_map(path: Path | None) -> dict[str, str]:
    if path is None or (not path.exists()):
        return {}
    out: dict[str, str] = {}
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            qid = str(row.get("id", ""))
            if not qid:
                continue
            out[qid] = str(row.get("prediction", row.get("pred", row.get("answer", ""))))
    return out


def collect_positive_chunks(row: dict) -> dict[str, list[str]]:
    out = {"text": [], "table": [], "kg": []}
    for chunk_id, label in row.get("relevant_chunks", {}).items():
        try:
            if float(label) <= 0:
                continue
        except Exception:
            continue
        out[source_bucket(str(chunk_id))].append(str(chunk_id))
    return out


def keyword_hits(query: str) -> int:
    q = query.lower()
    hits = sum(1 for kw in RISK_KEYWORDS if kw in q)
    if re.search(r"\b\d{1,3}(?:,\d{3})*(?:\.\d+)?\b", q):
        hits += 1
    return hits


def build_score(row: dict, dense_pred: str, uni_pred: str) -> float:
    chunks = collect_positive_chunks(row)
    has_text = bool(chunks["text"])
    has_table = bool(chunks["table"])
    has_kg = bool(chunks["kg"])

    score = 0.0
    if has_table and has_kg:
        score += 2.0
    if has_text and has_table:
        score += 1.0
    if has_text and has_kg:
        score += 1.0

    hits = keyword_hits(str(row.get("query", "")))
    score += min(1.5, 0.25 * hits)

    gold = str(row.get("answer", ""))
    dense_em = exact_match(dense_pred, gold)
    uni_em = exact_match(uni_pred, gold)
    if normalize_answer(dense_pred) != normalize_answer(uni_pred):
        score += 0.8
    if dense_em != uni_em:
        score += 0.6
    if dense_em < 1.0 and uni_em < 1.0:
        score += 0.6
    if re.search(r"\d", gold):
        score += 0.4

    return score


def main() -> int:
    parser = argparse.ArgumentParser(description="Build a target-type-focused mmRAG split")
    parser.add_argument("--split-file", type=Path, required=True)
    parser.add_argument("--dense-pred-file", type=Path, default=None)
    parser.add_argument("--unifusion-pred-file", type=Path, default=None)
    parser.add_argument("--target-types", type=str, required=True, help="Comma-separated target types, e.g. number,year")
    parser.add_argument("--target-size", type=int, default=0, help="Optional hard cap after scoring; 0 keeps all matched rows")
    parser.add_argument("--max-queries", type=int, default=1286)
    parser.add_argument("--out-split-file", type=Path, required=True)
    parser.add_argument("--out-summary", type=Path, required=True)
    args = parser.parse_args()

    target_types = {x.strip().lower() for x in args.target_types.split(",") if x.strip()}
    if not target_types:
        raise ValueError("--target-types must contain at least one target type")

    rows = read_json(args.split_file)[: args.max_queries]
    dense_map = load_pred_map(args.dense_pred_file)
    uni_map = load_pred_map(args.unifusion_pred_file)

    scored: list[tuple[float, str, dict, dict[str, list[str]], str, str, str]] = []
    for row in rows:
        qid = str(row.get("id", ""))
        if not qid:
            continue
        target_type = infer_qa_target_type(str(row.get("query", "")))
        if target_type not in target_types:
            continue
        chunks = collect_positive_chunks(row)
        dense_pred = dense_map.get(qid, "")
        uni_pred = uni_map.get(qid, "")
        score = build_score(row, dense_pred, uni_pred)
        scored.append((score, qid, row, chunks, dense_pred, uni_pred, target_type))

    scored.sort(key=lambda item: item[0], reverse=True)
    if args.target_size > 0:
        scored = scored[: args.target_size]

    subset_rows = []
    target_counter = Counter()
    combo_counter = Counter()
    for rank, (score, qid, row, chunks, dense_pred, uni_pred, target_type) in enumerate(scored, start=1):
        combo = "+".join(infer_modalities_from_relevant_chunks(row.get("relevant_chunks", {}))) or "none"
        target_counter[target_type] += 1
        combo_counter[combo] += 1
        new_row = dict(row)
        new_row["target_type"] = target_type
        new_row["selection_rank"] = rank
        new_row["selection_score"] = round(float(score), 4)
        new_row["modality_combo"] = combo
        new_row["dense_prediction"] = dense_pred
        new_row["unifusion_prediction"] = uni_pred
        new_row["dense_em"] = exact_match(dense_pred, str(row.get("answer", "")))
        new_row["unifusion_em"] = exact_match(uni_pred, str(row.get("answer", "")))
        new_row["positive_text_chunk_ids"] = chunks["text"]
        new_row["positive_table_chunk_ids"] = chunks["table"]
        new_row["positive_kg_chunk_ids"] = chunks["kg"]
        subset_rows.append(new_row)

    ensure_dir(args.out_split_file.parent)
    write_json(args.out_split_file, subset_rows)

    summary = {
        "split_file": str(args.split_file),
        "dense_pred_file": str(args.dense_pred_file) if args.dense_pred_file is not None else None,
        "unifusion_pred_file": str(args.unifusion_pred_file) if args.unifusion_pred_file is not None else None,
        "target_types": sorted(target_types),
        "target_size": int(args.target_size),
        "selected": len(subset_rows),
        "target_type_distribution": dict(target_counter),
        "combo_distribution": dict(combo_counter),
        "out_split_file": str(args.out_split_file),
    }

    ensure_dir(args.out_summary.parent)
    write_json(args.out_summary, summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
