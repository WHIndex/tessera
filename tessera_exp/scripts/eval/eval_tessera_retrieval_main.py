#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import importlib
import json
from pathlib import Path
import re
import sys

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

mod = importlib.import_module("tessera_exp.utils.e5_embed")
load_e5 = mod.load_e5
encode_texts = mod.encode_texts


def tokenize(text: str) -> set[str]:
    return set(re.findall(r"[a-z0-9]+", text.lower()))


def topk_indices(scores: np.ndarray, k: int) -> np.ndarray:
    if len(scores) <= k:
        return np.argsort(-scores)
    idx = np.argpartition(-scores, kth=k - 1)[:k]
    return idx[np.argsort(-scores[idx])]


def source_prefix(doc_id: str) -> str:
    if "_" not in doc_id:
        return doc_id
    return doc_id.rsplit("_", 1)[0]


def positive_relevant_ids(row: dict) -> set[str]:
    rel = set()
    for chunk_id, label in row.get("relevant_chunks", {}).items():
        try:
            if float(label) > 0:
                rel.add(chunk_id)
        except Exception:
            continue
    return rel


def pred_from_score_matrix(scores: np.ndarray, doc_ids: list[str], topk: int) -> list[list[str]]:
    preds = []
    for i in range(scores.shape[0]):
        idx = topk_indices(scores[i], topk)
        preds.append([doc_ids[j] for j in idx])
    return preds


def build_sparse_scores(corpus_texts: list[str], query_texts: list[str], max_features: int):
    vec = TfidfVectorizer(stop_words="english", ngram_range=(1, 2), max_features=max_features, min_df=2)
    c_mat = vec.fit_transform(corpus_texts)
    q_mat = vec.transform(query_texts)
    return (q_mat @ c_mat.T).toarray().astype(np.float32)


def normalize_scores(values: np.ndarray) -> np.ndarray:
    if values.size == 0:
        return values
    lo = float(values.min())
    hi = float(values.max())
    if hi - lo < 1e-9:
        return np.zeros_like(values)
    return (values - lo) / (hi - lo)


def make_cache_key(ids: list[str], max_items: int = 2048) -> str:
    if not ids:
        return "empty"
    if len(ids) <= max_items:
        sampled = ids
    else:
        step = max(1, len(ids) // max_items)
        sampled = ids[::step][:max_items]
    payload = "|".join(sampled) + f"|n={len(ids)}"
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()[:12]


def build_main_preds(
    rows,
    query_texts,
    query_tokens,
    doc_ids,
    corpus_texts,
    dense_scores,
    sparse_scores,
    topk,
    candidate_k,
    enable_late_interaction: bool,
    enable_uncertainty_gating: bool,
    enable_redundancy_detection: bool,
    late_alpha: float,
    conf_threshold: float,
    anchor_dense_top: int,
    max_per_source: int,
    preserve_dense_top: int,
):
    corpus_tokens = [tokenize(t) for t in corpus_texts]
    preds = []
    diagnostics = []

    for qi, _ in enumerate(rows):
        d_idx = topk_indices(dense_scores[qi], candidate_k)
        s_idx = topk_indices(sparse_scores[qi], candidate_k)

        d_rank = {j: r + 1 for r, j in enumerate(d_idx)}
        s_rank = {j: r + 1 for r, j in enumerate(s_idx)}

        all_idx = list(set(d_idx.tolist() + s_idx.tolist()))

        d_top = float(dense_scores[qi][d_idx[0]]) if len(d_idx) else 0.0
        d_10 = float(dense_scores[qi][d_idx[min(9, len(d_idx) - 1)]]) if len(d_idx) else 0.0
        conf = d_top - d_10

        if enable_uncertainty_gating and conf < conf_threshold:
            w_dense = 1.0
            w_sparse = 1.3
        else:
            w_dense = 1.2
            w_sparse = 0.8

        qtok = query_tokens[qi]

        d_vals = np.asarray([dense_scores[qi][j] for j in all_idx], dtype=np.float32)
        s_vals = np.asarray([sparse_scores[qi][j] for j in all_idx], dtype=np.float32)
        d_norm = normalize_scores(d_vals)
        s_norm = normalize_scores(s_vals)
        dense_norm_by_idx = {j: float(v) for j, v in zip(all_idx, d_norm)}
        sparse_norm_by_idx = {j: float(v) for j, v in zip(all_idx, s_norm)}

        scored = []
        for j in all_idx:
            sc = w_dense * dense_norm_by_idx.get(j, 0.0) + w_sparse * sparse_norm_by_idx.get(j, 0.0)

            if enable_late_interaction:
                overlap = len(qtok & corpus_tokens[j]) / max(1, len(qtok))
                sc += late_alpha * overlap

            scored.append((j, sc))

        scored.sort(key=lambda x: x[1], reverse=True)

        if not enable_redundancy_detection:
            sel_idx = [j for j, _ in scored[:topk]]
            sel = [doc_ids[j] for j in sel_idx]
            preds.append(sel)

            dense_topk_idx = topk_indices(dense_scores[qi], topk)
            dense_topk_set = {doc_ids[j] for j in dense_topk_idx}
            sel_set = set(sel)
            dset = set(d_idx.tolist())
            sset = set(s_idx.tolist())
            sparse_only = len([j for j in sel_idx if (j in sset and j not in dset)])
            diagnostics.append(
                {
                    "candidate_union_size": len(all_idx),
                    "main_dense_overlap_at_k": len(sel_set & dense_topk_set) / max(1, topk),
                    "main_new_over_dense_at_k": len(sel_set - dense_topk_set),
                    "selected_sparse_only": sparse_only,
                }
            )
            continue

        seen_count: dict[str, int] = {}
        sel = []
        sel_idx = []

        # Preserve a dense prefix as a conservative guardrail against regression.
        for j in d_idx[: min(preserve_dense_top, topk)]:
            did = doc_ids[j]
            if did in sel:
                continue
            sel.append(did)
            sel_idx.append(j)
            sid = source_prefix(did)
            seen_count[sid] = seen_count.get(sid, 0) + 1
            if len(sel) >= topk:
                break

        # Then preserve additional strong dense anchors before diversification.
        for j in d_idx[preserve_dense_top : preserve_dense_top + anchor_dense_top]:
            did = doc_ids[j]
            if did in sel:
                continue
            sel.append(did)
            sel_idx.append(j)
            sid = source_prefix(did)
            seen_count[sid] = seen_count.get(sid, 0) + 1
            if len(sel) >= topk:
                break

        for j, _ in scored:
            sid = source_prefix(doc_ids[j])
            if seen_count.get(sid, 0) >= max_per_source:
                continue
            seen_count[sid] = seen_count.get(sid, 0) + 1
            sel.append(doc_ids[j])
            sel_idx.append(j)
            if len(sel) >= topk:
                break

        if len(sel) < topk:
            for j, _ in scored:
                if doc_ids[j] in sel:
                    continue
                sel.append(doc_ids[j])
                sel_idx.append(j)
                if len(sel) >= topk:
                    break

        preds.append(sel)
        dense_topk_idx = topk_indices(dense_scores[qi], topk)
        dense_topk_set = {doc_ids[j] for j in dense_topk_idx}
        sel_set = set(sel)
        dset = set(d_idx.tolist())
        sset = set(s_idx.tolist())
        sparse_only = len([j for j in sel_idx if (j in sset and j not in dset)])
        diagnostics.append(
            {
                "candidate_union_size": len(all_idx),
                "main_dense_overlap_at_k": len(sel_set & dense_topk_set) / max(1, topk),
                "main_new_over_dense_at_k": len(sel_set - dense_topk_set),
                "selected_sparse_only": sparse_only,
            }
        )

    return preds, diagnostics


def method_metrics(rows, preds):
    ks = [5, 10, 20]
    any_hit = {k: [] for k in ks}
    recall = {k: [] for k in ks}
    precision = {k: [] for k in ks}
    rel_count = []

    for r, p in zip(rows, preds):
        rel = positive_relevant_ids(r)
        rel_count.append(len(rel))
        denom = max(1, len(rel))
        for k in ks:
            topk = set(p[:k])
            inter = len(topk & rel)
            any_hit[k].append(1 if inter > 0 else 0)
            recall[k].append(inter / denom)
            precision[k].append(inter / k)

    summary = {
        "avg_positive_qrels": float(np.mean(rel_count)) if rel_count else 0.0,
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
    detail = {
        "rel_count": rel_count,
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
    return summary, detail


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate TESSERA main method vs baselines and ablations")
    parser.add_argument("--model-dir", type=str, required=True)
    parser.add_argument("--split-file", type=Path, required=True)
    parser.add_argument("--corpus-file", type=Path, required=True)
    parser.add_argument("--out-file", type=Path, required=True)
    parser.add_argument("--detail-file", type=Path, required=True)
    parser.add_argument("--max-queries", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--topk", type=int, default=20)
    parser.add_argument("--candidate-k", type=int, default=400)
    parser.add_argument("--sparse-max-features", type=int, default=200000)
    parser.add_argument("--late-alpha", type=float, default=0.08)
    parser.add_argument("--conf-threshold", type=float, default=0.02)
    parser.add_argument("--anchor-dense-top", type=int, default=3)
    parser.add_argument("--max-per-source", type=int, default=3)
    parser.add_argument("--preserve-dense-top", type=int, default=10)
    parser.add_argument("--cache-dir", type=Path, default=Path("artifacts/retrieval"))
    parser.add_argument("--save-predictions", action="store_true")
    args = parser.parse_args()

    rows = json.loads(args.split_file.read_text(encoding="utf-8"))[: args.max_queries]
    corpus = json.loads(args.corpus_file.read_text(encoding="utf-8"))

    q_texts = [r.get("query", "") for r in rows]
    q_tokens = [tokenize(x) for x in q_texts]
    q_ids = [r.get("id", f"q_{i}") for i, r in enumerate(rows)]
    doc_ids = [d["id"] for d in corpus]
    c_texts = [d.get("text", "") for d in corpus]

    args.cache_dir.mkdir(parents=True, exist_ok=True)
    q_key = make_cache_key(q_ids)
    c_key = make_cache_key(doc_ids)
    sparse_cache = args.cache_dir / f"tfidf_scores_{len(q_texts)}x{len(c_texts)}_{q_key}_{c_key}.npy"

    tokenizer, model, device, resolved = load_e5(args.model_dir)
    print(f"[stage] model={resolved} device={device} queries={len(q_texts)} corpus={len(c_texts)}")

    model_key = hashlib.sha1(str(resolved).encode("utf-8")).hexdigest()[:12]
    q_cache = args.cache_dir / f"e5_query_{model_key}_{len(q_texts)}_{q_key}.npy"
    c_cache = args.cache_dir / f"e5_corpus_{model_key}_{len(c_texts)}_{c_key}.npy"

    if q_cache.exists() and np.load(q_cache, mmap_mode="r").shape[0] == len(q_texts):
        qv = np.load(q_cache)
    else:
        qv = encode_texts(q_texts, tokenizer, model, device, batch_size=args.batch_size)
        np.save(q_cache, qv)

    if c_cache.exists() and np.load(c_cache, mmap_mode="r").shape[0] == len(c_texts):
        cv = np.load(c_cache)
    else:
        cv = encode_texts(c_texts, tokenizer, model, device, batch_size=args.batch_size)
        np.save(c_cache, cv)

    dense_scores = qv @ cv.T

    if sparse_cache.exists():
        sparse_scores = np.load(sparse_cache)
        if sparse_scores.shape != (len(q_texts), len(c_texts)):
            sparse_scores = build_sparse_scores(c_texts, q_texts, max_features=args.sparse_max_features)
            np.save(sparse_cache, sparse_scores)
    else:
        sparse_scores = build_sparse_scores(c_texts, q_texts, max_features=args.sparse_max_features)
        np.save(sparse_cache, sparse_scores)

    pred_dense = pred_from_score_matrix(dense_scores, doc_ids, args.topk)
    pred_sparse = pred_from_score_matrix(sparse_scores, doc_ids, args.topk)

    pred_main, main_diag = build_main_preds(
        rows,
        q_texts,
        q_tokens,
        doc_ids,
        c_texts,
        dense_scores,
        sparse_scores,
        args.topk,
        args.candidate_k,
        enable_late_interaction=True,
        enable_uncertainty_gating=True,
        enable_redundancy_detection=True,
        late_alpha=args.late_alpha,
        conf_threshold=args.conf_threshold,
        anchor_dense_top=args.anchor_dense_top,
        max_per_source=args.max_per_source,
        preserve_dense_top=args.preserve_dense_top,
    )

    pred_ab_no_late, _ = build_main_preds(
        rows,
        q_texts,
        q_tokens,
        doc_ids,
        c_texts,
        dense_scores,
        sparse_scores,
        args.topk,
        args.candidate_k,
        enable_late_interaction=False,
        enable_uncertainty_gating=True,
        enable_redundancy_detection=True,
        late_alpha=args.late_alpha,
        conf_threshold=args.conf_threshold,
        anchor_dense_top=args.anchor_dense_top,
        max_per_source=args.max_per_source,
        preserve_dense_top=args.preserve_dense_top,
    )

    pred_ab_no_unc, _ = build_main_preds(
        rows,
        q_texts,
        q_tokens,
        doc_ids,
        c_texts,
        dense_scores,
        sparse_scores,
        args.topk,
        args.candidate_k,
        enable_late_interaction=True,
        enable_uncertainty_gating=False,
        enable_redundancy_detection=True,
        late_alpha=args.late_alpha,
        conf_threshold=args.conf_threshold,
        anchor_dense_top=args.anchor_dense_top,
        max_per_source=args.max_per_source,
        preserve_dense_top=args.preserve_dense_top,
    )

    pred_ab_no_red, _ = build_main_preds(
        rows,
        q_texts,
        q_tokens,
        doc_ids,
        c_texts,
        dense_scores,
        sparse_scores,
        args.topk,
        args.candidate_k,
        enable_late_interaction=True,
        enable_uncertainty_gating=True,
        enable_redundancy_detection=False,
        late_alpha=args.late_alpha,
        conf_threshold=args.conf_threshold,
        anchor_dense_top=args.anchor_dense_top,
        max_per_source=args.max_per_source,
        preserve_dense_top=args.preserve_dense_top,
    )

    methods = {
        "baseline_dense": pred_dense,
        "baseline_sparse_tfidf": pred_sparse,
        "main_tessera": pred_main,
        "ablation_no_late_interaction": pred_ab_no_late,
        "ablation_no_uncertainty_gating": pred_ab_no_unc,
        "ablation_no_redundancy_detection": pred_ab_no_red,
    }

    metrics = {}
    details = {"queries": len(rows), "query_ids": [r.get("id", f"q_{i}") for i, r in enumerate(rows)], "methods": {}}
    for name, pred in methods.items():
        m, d = method_metrics(rows, pred)
        metrics[name] = m
        details["methods"][name] = d

    out = {
        "queries": len(rows),
        "corpus": len(corpus),
        "methods": metrics,
        "main_method": "main_tessera",
    }

    if main_diag:
        overlap = [x["main_dense_overlap_at_k"] for x in main_diag]
        added = [x["main_new_over_dense_at_k"] for x in main_diag]
        sparse_only = [x["selected_sparse_only"] for x in main_diag]
        cand_union = [x["candidate_union_size"] for x in main_diag]
        out_diag = {
            "avg_main_dense_overlap_at_k": float(np.mean(overlap)),
            "avg_main_new_over_dense_at_k": float(np.mean(added)),
            "avg_selected_sparse_only": float(np.mean(sparse_only)),
            "avg_candidate_union_size": float(np.mean(cand_union)),
        }
        out["main_diagnostics"] = out_diag
        details["main_diagnostics"] = {
            "summary": out_diag,
            "per_query": main_diag,
        }

    if args.save_predictions:
        details["predictions"] = methods

    args.out_file.parent.mkdir(parents=True, exist_ok=True)
    args.out_file.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    args.detail_file.parent.mkdir(parents=True, exist_ok=True)
    args.detail_file.write_text(json.dumps(details, ensure_ascii=False, indent=2), encoding="utf-8")

    print(json.dumps(out, ensure_ascii=False, indent=2))
    print(f"[OK] saved -> {args.out_file}")
    print(f"[OK] detail -> {args.detail_file}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
