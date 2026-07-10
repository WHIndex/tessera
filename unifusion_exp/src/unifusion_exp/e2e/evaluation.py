from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Callable

import numpy as np


def positive_relevant_ids(row: dict) -> set[str]:
    rel = set()
    for chunk_id, label in row.get("relevant_chunks", {}).items():
        try:
            if float(label) > 0:
                rel.add(chunk_id)
        except Exception:
            continue
    return rel


def query_modality_label(row: dict, source_bucket_fn: Callable[[str], str]) -> str:
    mods = set()
    for chunk_id in positive_relevant_ids(row):
        mods.add(source_bucket_fn(chunk_id))
    if not mods:
        return "unknown"
    if len(mods) == 1:
        m = next(iter(mods))
        return f"{m}_only"
    return "multi_modal"


def build_query_modality_distribution(rows: list[dict], source_bucket_fn: Callable[[str], str]) -> dict:
    counter = defaultdict(int)
    for row in rows:
        counter[query_modality_label(row, source_bucket_fn)] += 1

    n = max(1, len(rows))
    out = {}
    for k in ["text_only", "table_only", "kg_only", "multi_modal", "unknown"]:
        c = int(counter.get(k, 0))
        out[k] = {"count": c, "ratio": float(c / n)}
    return out


def write_predictions_jsonl(path: Path, rows: list[dict], preds: list[str]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for row, pred in zip(rows, preds):
            f.write(json.dumps({"id": row.get("id"), "prediction": pred}, ensure_ascii=False) + "\n")


def evaluate_predictions(
    rows: list[dict],
    preds: list[str],
    top10_lists: list[list[str]],
    exact_match_fn: Callable[[str, str], float],
    f1_score_fn: Callable[[str, str], float],
    mmrag_official_fn: Callable[[str, object, str], float],
    source_bucket_fn: Callable[[str], str],
) -> dict:
    em_vals = []
    f1_vals = []
    r10_vals = []
    mm_vals = []

    slices: dict[str, dict[str, list[float]]] = defaultdict(
        lambda: {"em": [], "f1": [], "r10": [], "mm": []}
    )

    for row, pred, top10 in zip(rows, preds, top10_lists):
        gold = str(row.get("answer", ""))
        em = float(exact_match_fn(pred, gold))
        f1 = float(f1_score_fn(pred, gold))
        mm = float(mmrag_official_fn(str(row.get("id", "")), row.get("answer", ""), pred))

        rel = positive_relevant_ids(row)
        inter = len(set(top10) & rel)
        r10 = float(inter / max(1, len(rel)))

        em_vals.append(em)
        f1_vals.append(f1)
        r10_vals.append(r10)
        mm_vals.append(mm)

        label = query_modality_label(row, source_bucket_fn)
        slices[label]["em"].append(em)
        slices[label]["f1"].append(f1)
        slices[label]["r10"].append(r10)
        slices[label]["mm"].append(mm)

    slice_metrics = {}
    for k in ["text_only", "table_only", "kg_only", "multi_modal", "unknown"]:
        vals = slices.get(k, {"em": [], "f1": [], "r10": [], "mm": []})
        n = len(vals["em"])
        slice_metrics[k] = {
            "count": int(n),
            "exact_match": float(np.mean(vals["em"])) if n else 0.0,
            "f1": float(np.mean(vals["f1"])) if n else 0.0,
            "recall@10": float(np.mean(vals["r10"])) if n else 0.0,
            "mmrag_official_avg": float(np.mean(vals["mm"])) if n else 0.0,
        }

    return {
        "exact_match": float(np.mean(em_vals)) if em_vals else 0.0,
        "f1": float(np.mean(f1_vals)) if f1_vals else 0.0,
        "mmrag_official_avg": float(np.mean(mm_vals)) if mm_vals else 0.0,
        "recall@10": float(np.mean(r10_vals)) if r10_vals else 0.0,
        "slice_metrics": slice_metrics,
    }
