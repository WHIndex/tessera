#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer


def resolve_model_dir(path: str) -> Path:
    p = Path(path)
    if (p / "config.json").exists():
        return p
    snapshots = p / "snapshots"
    if snapshots.exists():
        cands = sorted([x for x in snapshots.iterdir() if x.is_dir()])
        for cand in reversed(cands):
            if (cand / "config.json").exists():
                return cand
        if cands:
            return cands[-1]
    raise FileNotFoundError(path)


def load_json(path: Path):
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def mean_pool(last_hidden_state, attention_mask):
    mask = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
    return torch.sum(last_hidden_state * mask, dim=1) / torch.clamp(mask.sum(dim=1), min=1e-9)


def encode_texts(texts, tokenizer, model, batch_size=64, device="cuda"):
    vecs = []
    model.eval()
    with torch.no_grad():
        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            inputs = tokenizer(batch, padding=True, truncation=True, max_length=512, return_tensors="pt")
            inputs = {k: v.to(device) for k, v in inputs.items()}
            outputs = model(**inputs)
            emb = mean_pool(outputs.last_hidden_state, inputs["attention_mask"])
            emb = torch.nn.functional.normalize(emb, p=2, dim=1)
            vecs.append(emb.cpu().numpy())
            if (i // batch_size) % 20 == 0:
                print(f"[encode] {min(i+batch_size, len(texts))}/{len(texts)}")
    return np.concatenate(vecs, axis=0)


def positive_relevant_ids(row: dict) -> set[str]:
    rel = set()
    for chunk_id, label in row.get("relevant_chunks", {}).items():
        try:
            if float(label) > 0:
                rel.add(chunk_id)
        except Exception:
            continue
    return rel


def topk_indices(scores: np.ndarray, k: int) -> np.ndarray:
    if len(scores) <= k:
        return np.argsort(-scores)
    idx = np.argpartition(-scores, kth=k - 1)[:k]
    return idx[np.argsort(-scores[idx])]


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate dense retrieval on subset corpus")
    parser.add_argument("--model-dir", type=str, required=True)
    parser.add_argument("--split-file", type=Path, required=True)
    parser.add_argument("--corpus-file", type=Path, required=True)
    parser.add_argument("--out-file", type=Path, required=True)
    parser.add_argument("--detail-file", type=Path, default=None)
    parser.add_argument("--max-queries", type=int, default=300)
    parser.add_argument("--batch-size", type=int, default=64)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_dir = resolve_model_dir(args.model_dir)
    tokenizer = AutoTokenizer.from_pretrained(str(model_dir))
    model = AutoModel.from_pretrained(str(model_dir)).to(device)

    rows = load_json(args.split_file)
    rows = rows[: args.max_queries]
    corpus = load_json(args.corpus_file)

    q_texts = [r.get("query", "") for r in rows]
    c_ids = [d["id"] for d in corpus]
    c_texts = [d["text"] for d in corpus]

    print(f"[stage] queries={len(q_texts)} corpus={len(c_texts)} device={device}")
    qv = encode_texts(q_texts, tokenizer, model, batch_size=args.batch_size, device=device)
    cv = encode_texts(c_texts, tokenizer, model, batch_size=args.batch_size, device=device)

    scores = qv @ cv.T

    ks = [5, 10, 20]
    any_hit = {k: [] for k in ks}
    recall = {k: [] for k in ks}
    precision = {k: [] for k in ks}
    rel_count = []
    rel_in_corpus_count = []
    id_set = set(c_ids)

    for qi, row in enumerate(rows):
        rel = positive_relevant_ids(row)
        rel_count.append(len(rel))
        rel_in_corpus = len(rel & id_set)
        rel_in_corpus_count.append(rel_in_corpus)
        denom = max(1, rel_in_corpus)

        for k in ks:
            idx = topk_indices(scores[qi], k)
            pred = {c_ids[j] for j in idx}
            inter = len(pred & rel)
            any_hit[k].append(1 if inter > 0 else 0)
            recall[k].append(inter / denom)
            precision[k].append(inter / k)

    metrics = {
        "queries": len(q_texts),
        "corpus": len(c_texts),
        "avg_positive_qrels": float(np.mean(rel_count)) if rel_count else 0.0,
        "qrels_coverage_in_corpus": float(np.sum(rel_in_corpus_count) / max(1, np.sum(rel_count))),
        "queries_without_positive_qrels": int(sum(1 for x in rel_count if x == 0)),
        "any_hit@5": float(np.mean(any_hit[5])) if any_hit[5] else 0.0,
        "any_hit@10": float(np.mean(any_hit[10])) if any_hit[10] else 0.0,
        "any_hit@20": float(np.mean(any_hit[20])) if any_hit[20] else 0.0,
        "recall@5": float(np.mean(recall[5])) if recall[5] else 0.0,
        "recall@10": float(np.mean(recall[10])) if recall[10] else 0.0,
        "recall@20": float(np.mean(recall[20])) if recall[20] else 0.0,
        "precision@5": float(np.mean(precision[5])) if precision[5] else 0.0,
        "precision@10": float(np.mean(precision[10])) if precision[10] else 0.0,
        "precision@20": float(np.mean(precision[20])) if precision[20] else 0.0,
    }

    args.out_file.parent.mkdir(parents=True, exist_ok=True)
    args.out_file.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")

    if args.detail_file is not None:
        detail = {
            "queries": len(rows),
            "query_ids": [r.get("id", f"q_{i}") for i, r in enumerate(rows)],
            "rel_count": rel_count,
            "rel_in_corpus_count": rel_in_corpus_count,
            "any_hit@5": any_hit[5],
            "any_hit@10": any_hit[10],
            "any_hit@20": any_hit[20],
            "hit@5": any_hit[5],
            "hit@10": any_hit[10],
            "hit@20": any_hit[20],
            "recall@5": recall[5],
            "recall@10": recall[10],
            "recall@20": recall[20],
            "precision@5": precision[5],
            "precision@10": precision[10],
            "precision@20": precision[20],
        }
        args.detail_file.parent.mkdir(parents=True, exist_ok=True)
        args.detail_file.write_text(json.dumps(detail, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"[OK] detail -> {args.detail_file}")

    print(json.dumps(metrics, ensure_ascii=False, indent=2))
    print(f"[OK] saved -> {args.out_file}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
