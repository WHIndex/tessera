#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import random
import re
from pathlib import Path


def safe_id(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]", "_", text)


def norm_text(text: str) -> str:
    return re.sub(r"\s+", " ", text.lower()).strip()


def table_to_text(obj: dict, max_rows: int, max_cols: int) -> str:
    title = str(obj.get("title", ""))
    section_title = str(obj.get("section_title", ""))
    section_text = str(obj.get("section_text", ""))
    intro = str(obj.get("intro", ""))

    headers = []
    for h in obj.get("header", []):
        if isinstance(h, list) and h:
            headers.append(str(h[0]))
        else:
            headers.append(str(h))
    header_text = " | ".join(headers[:max_cols])

    row_lines = []
    for row in obj.get("data", [])[:max_rows]:
        cells = []
        for c in row[:max_cols]:
            if isinstance(c, list) and c:
                cells.append(str(c[0]))
            else:
                cells.append(str(c))
        row_lines.append(" ; ".join(cells))

    return "\n".join(
        [
            f"title: {title}",
            f"section_title: {section_title}",
            f"section_text: {section_text}",
            f"intro: {intro}",
            f"header: {header_text}",
            "rows:",
            "\n".join(row_lines),
        ]
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--hybridqa-split", type=Path, required=True)
    parser.add_argument("--tables-dir", type=Path, required=True)
    parser.add_argument("--request-dir", type=Path, required=True)
    parser.add_argument("--out-split", type=Path, required=True)
    parser.add_argument("--out-corpus", type=Path, required=True)
    parser.add_argument("--out-stats", type=Path, required=True)
    parser.add_argument("--max-queries", type=int, default=300)
    parser.add_argument("--distractor-tables", type=int, default=3000)
    parser.add_argument("--max-passages-per-table", type=int, default=15)
    parser.add_argument("--max-table-rows", type=int, default=30)
    parser.add_argument("--max-table-cols", type=int, default=8)
    parser.add_argument("--seed", type=int, default=20260329)
    args = parser.parse_args()

    random.seed(args.seed)

    data = json.loads(args.hybridqa_split.read_text(encoding="utf-8"))
    data = data[: args.max_queries]

    all_table_files = sorted(args.tables_dir.glob("*.json"))
    all_table_ids = [p.stem for p in all_table_files]
    all_table_set = set(all_table_ids)

    query_table_ids = []
    rows = []
    for x in data:
        qid = str(x.get("question_id", ""))
        q = str(x.get("question", ""))
        table_id = str(x.get("table_id", ""))
        ans = x.get("answer-text", "")
        if isinstance(ans, list):
            ans = ans[0] if ans else ""
        ans = str(ans)
        if table_id not in all_table_set:
            continue
        query_table_ids.append(table_id)
        rows.append({"id": qid, "query": q, "table_id": table_id, "answer": ans})

    query_table_set = set(query_table_ids)

    distractor_pool = [tid for tid in all_table_ids if tid not in query_table_set]
    k = min(args.distractor_tables, len(distractor_pool))
    distractor_ids = random.sample(distractor_pool, k)

    selected_table_ids = sorted(query_table_set.union(distractor_ids))

    corpus = []
    table_doc_map = {}
    table_passages = {}

    for tid in selected_table_ids:
        p = args.tables_dir / f"{tid}.json"
        try:
            obj = json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            continue
        doc_id = f"tat_hq_{safe_id(tid)}"
        text = table_to_text(obj, args.max_table_rows, args.max_table_cols)
        corpus.append({"id": doc_id, "text": text})
        table_doc_map[tid] = doc_id

    for tid in sorted(query_table_set):
        rp = args.request_dir / f"{tid}.json"
        if not rp.exists():
            table_passages[tid] = []
            continue
        try:
            robj = json.loads(rp.read_text(encoding="utf-8"))
        except Exception:
            table_passages[tid] = []
            continue
        if not isinstance(robj, dict):
            table_passages[tid] = []
            continue
        items = list(robj.items())[: args.max_passages_per_table]
        pids = []
        for i, (link, txt) in enumerate(items):
            pid = f"nq_hq_{safe_id(tid)}_{i}"
            text = f"link: {link}\npassage: {str(txt)}"
            corpus.append({"id": pid, "text": text})
            pids.append((pid, str(txt)))
        table_passages[tid] = pids

    split_rows = []
    pos_counts = []
    for r in rows:
        tid = r["table_id"]
        rel = {}
        tdoc = table_doc_map.get(tid)
        if tdoc is not None:
            rel[tdoc] = 1.0

        ans = norm_text(r["answer"])
        if ans:
            for pid, ptxt in table_passages.get(tid, []):
                if ans in norm_text(ptxt):
                    rel[pid] = 1.0

        split_rows.append(
            {
                "id": f"hybridqa_{r['id']}",
                "query": r["query"],
                "answer": r["answer"],
                "relevant_chunks": rel,
            }
        )
        pos_counts.append(len(rel))

    args.out_split.parent.mkdir(parents=True, exist_ok=True)
    args.out_corpus.parent.mkdir(parents=True, exist_ok=True)
    args.out_stats.parent.mkdir(parents=True, exist_ok=True)

    args.out_split.write_text(json.dumps(split_rows, ensure_ascii=False, indent=2), encoding="utf-8")
    args.out_corpus.write_text(json.dumps(corpus, ensure_ascii=False), encoding="utf-8")

    stats = {
        "queries_requested": args.max_queries,
        "queries_built": len(split_rows),
        "unique_query_tables": len(query_table_set),
        "distractor_tables": len(distractor_ids),
        "corpus_docs": len(corpus),
        "avg_positive_per_query": (sum(pos_counts) / len(pos_counts)) if pos_counts else 0.0,
        "max_positive_per_query": max(pos_counts) if pos_counts else 0,
        "min_positive_per_query": min(pos_counts) if pos_counts else 0,
    }
    args.out_stats.write_text(json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8")

    print(json.dumps(stats, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
