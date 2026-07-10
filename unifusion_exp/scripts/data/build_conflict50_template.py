#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

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


def load_corpus_snippets(corpus_file: Path, needed_ids: set[str], max_chars: int) -> dict[str, str]:
    if not corpus_file.exists() or not needed_ids:
        return {}
    try:
        rows = read_json(corpus_file)
    except Exception:
        return {}

    out: dict[str, str] = {}
    for row in rows:
        cid = str(row.get("id", ""))
        if cid not in needed_ids:
            continue
        text = str(row.get("text", ""))
        text = " ".join(text.split())
        out[cid] = text[:max_chars]
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description="Build Conflict-50 annotation template from mmRAG test split")
    parser.add_argument("--split-file", type=Path, required=True)
    parser.add_argument("--dense-pred-file", type=Path, default=None)
    parser.add_argument("--unifusion-pred-file", type=Path, default=None)
    parser.add_argument("--corpus-file", type=Path, default=None)
    parser.add_argument("--target-size", type=int, default=50)
    parser.add_argument("--max-queries", type=int, default=1286)
    parser.add_argument("--max-corpus-snippet-chars", type=int, default=320)
    parser.add_argument("--out-jsonl", type=Path, required=True)
    parser.add_argument("--out-csv", type=Path, required=True)
    parser.add_argument("--out-summary", type=Path, required=True)
    args = parser.parse_args()

    rows = read_json(args.split_file)[: args.max_queries]
    dense_map = load_pred_map(args.dense_pred_file)
    uni_map = load_pred_map(args.unifusion_pred_file)

    scored = []
    for row in rows:
        qid = str(row.get("id", ""))
        if not qid:
            continue
        chunks = collect_positive_chunks(row)
        labels = infer_modalities_from_relevant_chunks(row.get("relevant_chunks", {}))
        dense_pred = dense_map.get(qid, "")
        uni_pred = uni_map.get(qid, "")
        risk = build_score(row, dense_pred, uni_pred)
        combo = "+".join(labels) if labels else "none"
        scored.append((risk, combo, row, chunks, dense_pred, uni_pred))

    scored.sort(key=lambda x: x[0], reverse=True)

    selected = []
    combo_count: dict[str, int] = {}
    cap_per_combo = max(8, args.target_size // 2)
    for item in scored:
        if len(selected) >= args.target_size:
            break
        combo = item[1]
        if combo_count.get(combo, 0) >= cap_per_combo:
            continue
        selected.append(item)
        combo_count[combo] = combo_count.get(combo, 0) + 1

    if len(selected) < args.target_size:
        selected_ids = {str(x[2].get("id", "")) for x in selected}
        for item in scored:
            if len(selected) >= args.target_size:
                break
            qid = str(item[2].get("id", ""))
            if qid in selected_ids:
                continue
            selected.append(item)
            selected_ids.add(qid)

    needed_ids: set[str] = set()
    for _, _, _, chunks, _, _ in selected:
        for key in ("table", "kg"):
            if chunks[key]:
                needed_ids.add(chunks[key][0])

    snippet_map = {}
    if args.corpus_file is not None:
        snippet_map = load_corpus_snippets(args.corpus_file, needed_ids, args.max_corpus_snippet_chars)

    records = []
    for rank, (risk, combo, row, chunks, dense_pred, uni_pred) in enumerate(selected, start=1):
        qid = str(row.get("id", ""))
        gold = str(row.get("answer", ""))
        table_anchor = chunks["table"][0] if chunks["table"] else ""
        kg_anchor = chunks["kg"][0] if chunks["kg"] else ""
        rec = {
            "rank": rank,
            "id": qid,
            "query": str(row.get("query", "")),
            "gold_answer": gold,
            "modality_combo": combo,
            "risk_score": round(float(risk), 4),
            "positive_table_chunk_ids": chunks["table"],
            "positive_kg_chunk_ids": chunks["kg"],
            "positive_text_chunk_ids": chunks["text"],
            "table_anchor_chunk_id": table_anchor,
            "kg_anchor_chunk_id": kg_anchor,
            "table_anchor_snippet": snippet_map.get(table_anchor, ""),
            "kg_anchor_snippet": snippet_map.get(kg_anchor, ""),
            "dense_prediction": dense_pred,
            "unifusion_prediction": uni_pred,
            "dense_em": exact_match(dense_pred, gold),
            "unifusion_em": exact_match(uni_pred, gold),
            "is_conflict": "",
            "preferred_evidence_side": "",
            "annotator": "",
            "annotation_notes": "",
        }
        records.append(rec)

    ensure_dir(args.out_jsonl.parent)
    with args.out_jsonl.open("w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    ensure_dir(args.out_csv.parent)
    if records:
        fieldnames = list(records[0].keys())
    else:
        fieldnames = [
            "rank",
            "id",
            "query",
            "gold_answer",
            "modality_combo",
            "risk_score",
            "is_conflict",
            "preferred_evidence_side",
            "annotator",
            "annotation_notes",
        ]
    with args.out_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(records)

    summary = {
        "split_file": str(args.split_file),
        "target_size": int(args.target_size),
        "selected": len(records),
        "combo_distribution": combo_count,
        "dense_pred_file": str(args.dense_pred_file) if args.dense_pred_file else None,
        "unifusion_pred_file": str(args.unifusion_pred_file) if args.unifusion_pred_file else None,
        "corpus_file": str(args.corpus_file) if args.corpus_file else None,
        "out_jsonl": str(args.out_jsonl),
        "out_csv": str(args.out_csv),
    }
    write_json(args.out_summary, summary)

    print(f"[OK] conflict template jsonl -> {args.out_jsonl}")
    print(f"[OK] conflict template csv -> {args.out_csv}")
    print(f"[OK] summary -> {args.out_summary}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
