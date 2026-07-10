#!/usr/bin/env python3
"""
TESSERA-RAG V4: 保守融合策略

核心改进:
1. 不降低 dense 权重（保持 0.6+），仅调整 sparse 作为补充
2. 使用 Router 置信度进行软加权
3. 高置信度时才应用模态特定权重
"""
from __future__ import annotations

import argparse
import hashlib
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

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from tessera_exp.utils.e5_embed import load_e5, encode_texts


def topk_indices(scores: np.ndarray, k: int) -> np.ndarray:
    if len(scores) <= k:
        return np.argsort(-scores)
    idx = np.argpartition(-scores, kth=k - 1)[:k]
    return idx[np.argsort(-scores[idx])]


def source_prefix(doc_id: str) -> str:
    if "_" not in doc_id:
        return doc_id
    return doc_id.rsplit("_", 1)[0]


def source_bucket(doc_id: str) -> str:
    if doc_id.startswith("m.") or doc_id.startswith("/m/"):
        return "kg"
    p = doc_id.split("_", 1)[0] if "_" in doc_id else doc_id
    if p in {"tat", "ott"}:
        return "table"
    if p in {"nq", "triviaqa", "hotpot", "squad", "newsqa"}:
        return "text"
    if p in {"kg", "wikidata", "wd"}:
        return "kg"
    return "text"


def positive_relevant_ids(row: dict) -> set[str]:
    rel = set()
    for chunk_id, label in row.get("relevant_chunks", {}).items():
        try:
            if float(label) > 0:
                rel.add(chunk_id)
        except Exception:
            continue
    return rel


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


class RouterInference:
    """Router inference with confidence scores."""
    
    def __init__(self, model_path: Path, device: str = "cuda" if torch.cuda.is_available() else "cpu"):
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(str(model_path))
        self.model = AutoModelForSequenceClassification.from_pretrained(str(model_path))
        self.model.to(device)
        self.model.eval()
        
    @torch.no_grad()
    def predict_with_confidence(self, queries: list[str], threshold: float = 0.5):
        inputs = self.tokenizer(
            queries, padding=True, truncation=True, max_length=256, return_tensors="pt"
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        outputs = self.model(**inputs)
        logits = outputs.logits.cpu().numpy()
        
        probs = 1.0 / (1.0 + np.exp(-logits))
        preds = (probs >= threshold).astype(np.int64)
        
        empty_mask = np.sum(preds, axis=1) == 0
        if np.any(empty_mask):
            top_idx = np.argmax(probs[empty_mask], axis=1)
            preds[empty_mask] = 0
            preds[empty_mask, top_idx] = 1
        
        # Confidence as max predicted probability
        confidences = np.max(probs * preds + (1 - preds) * (1 - probs), axis=1)
        
        return preds, confidences


def get_conservative_weights(has_text: int, has_table: int, has_kg: int, 
                             router_conf: float, dense_score: float) -> dict:
    """
    Conservative fusion: always keep dense dominant, use sparse as supplement.
    """
    # Base: dense dominant
    w_dense = 0.7
    w_sparse = 0.3
    
    # Only adjust if router is confident
    if router_conf > 0.8:
        if has_table and not has_text:
            # Table query: slightly boost sparse
            w_dense = 0.65
            w_sparse = 0.45
        elif has_text and not has_table:
            # Text query: keep dense high
            w_dense = 0.75
            w_sparse = 0.25
    
    # If dense score is very high, trust it more
    if dense_score > 0.8:
        w_dense = min(0.85, w_dense + 0.1)
        w_sparse = max(0.15, w_sparse - 0.1)
    
    return {"dense": w_dense, "sparse": w_sparse}


def build_v4_preds(rows, query_texts, doc_ids, corpus_texts, dense_scores, sparse_scores,
                   router: RouterInference, topk: int = 20,
                   use_router: bool = True, late_alpha: float = 0.1,
                   preserve_dense_top: int = 0, max_per_source: int = 5,
                   policy_switch_conf: float = 0.6,
                   policy_dense_w: float = 0.72,
                   policy_sparse_w: float = 0.28,
                   policy_late_w: float = 0.10,
                   policy_modality_w: float = 0.10):
    """Build predictions with conservative fusion."""
    preds = []
    diagnostics = []
    corpus_tokens = [set(re.findall(r"[a-z0-9]+", t.lower())) for t in corpus_texts]
    
    if use_router:
        print("[Router] Predicting modalities...")
        route_preds, router_confs = router.predict_with_confidence(query_texts)
        print(f"[Router] Text={route_preds[:,0].sum()}, Table={route_preds[:,1].sum()}, KG={route_preds[:,2].sum()}")
    else:
        # Strict no-router semantics: disable modality-specific branching.
        route_preds = np.zeros((len(rows), 3), dtype=np.int64)
        route_preds[:, 0] = 1
        router_confs = np.zeros(len(rows), dtype=np.float32)
    
    for qi in range(len(rows)):
        has_text = route_preds[qi][0]
        has_table = route_preds[qi][1]
        has_kg = route_preds[qi][2]
        router_conf = router_confs[qi]
        
        # Get candidates
        d_idx = topk_indices(dense_scores[qi], 200)
        s_idx = topk_indices(sparse_scores[qi], 200)
        all_idx = list(set(d_idx.tolist() + s_idx.tolist()))
        
        # Normalize
        d_norm = normalize_scores(dense_scores[qi])
        s_norm = normalize_scores(sparse_scores[qi])
        
        q_tokens = set(re.findall(r"[a-z0-9]+", query_texts[qi].lower()))

        # Score with conservative fusion + query-adaptive policy switching.
        # A branch: dense-focused safe ranker (good for text-dominant queries).
        # B branch: modality-aware fusion ranker (better for table/kg-heavy queries).
        scored = []
        scored_dense_focus = []
        for j in all_idx:
            d_score = d_norm[j]
            weights = get_conservative_weights(has_text, has_table, has_kg, router_conf, d_score)
            
            # Base fusion
            score = weights["dense"] * d_score + weights["sparse"] * s_norm[j]
            overlap = 0.0
            
            # Late interaction: simple token overlap bonus
            if late_alpha > 0:
                d_tokens = corpus_tokens[j]
                if q_tokens and d_tokens:
                    overlap = len(q_tokens & d_tokens) / max(1, len(q_tokens))
                    score += late_alpha * overlap

            b = source_bucket(doc_ids[j])
            if b == "table":
                modality_target = float(has_table)
            elif b == "kg":
                modality_target = float(has_kg)
            else:
                modality_target = float(has_text)

            policy_score = (
                float(policy_dense_w) * d_score
                + float(policy_sparse_w) * s_norm[j]
                + float(policy_late_w) * overlap
                + float(policy_modality_w) * modality_target
            )

            dense_focus_score = 0.84 * d_score + 0.16 * s_norm[j]
            if q_tokens:
                d_tokens = corpus_tokens[j]
                if d_tokens:
                    dense_focus_score += 0.03 * (len(q_tokens & d_tokens) / max(1, len(q_tokens)))
            
            scored.append((j, 0.4 * score + 0.6 * policy_score))
            scored_dense_focus.append((j, dense_focus_score))
        
        scored.sort(key=lambda x: x[1], reverse=True)
        scored_dense_focus.sort(key=lambda x: x[1], reverse=True)
        use_modality_branch = bool((has_table or has_kg) and (router_conf >= policy_switch_conf))
        scored_chosen = scored if use_modality_branch else scored_dense_focus
        sel_idx: list[int] = []
        sel_set: set[int] = set()
        seen_count: dict[str, int] = {}

        # Guard dense head to avoid top-k regression while allowing tail diversification.
        guard_k = min(max(0, preserve_dense_top), topk)
        for j in d_idx[:guard_k]:
            if j in sel_set:
                continue
            sel_idx.append(j)
            sel_set.add(j)
            sid = source_prefix(doc_ids[j])
            seen_count[sid] = seen_count.get(sid, 0) + 1

        for j, _ in scored_chosen:
            if j in sel_set:
                continue
            sid = source_prefix(doc_ids[j])
            if seen_count.get(sid, 0) >= max_per_source:
                continue
            sel_idx.append(j)
            sel_set.add(j)
            seen_count[sid] = seen_count.get(sid, 0) + 1
            if len(sel_idx) >= topk:
                break

        if len(sel_idx) < topk:
            for j, _ in scored_chosen:
                if j in sel_set:
                    continue
                sel_idx.append(j)
                sel_set.add(j)
                if len(sel_idx) >= topk:
                    break

        sel = [doc_ids[j] for j in sel_idx[:topk]]
        preds.append(sel)
        
        # Diagnostics
        dense_topk = [doc_ids[j] for j in d_idx[:topk]]
        overlap = len(set(sel) & set(dense_topk))
        diagnostics.append({
            "query_idx": qi,
            "routing": {"text": int(has_text), "table": int(has_table), "kg": int(has_kg)},
            "router_conf": float(router_conf),
            "branch": "modality_fusion" if use_modality_branch else "dense_focus",
            "dense_overlap": overlap,
            "dense_overlap_ratio": float(overlap / max(1, topk)),
            "main_new_over_dense": int(max(0, topk - overlap)),
        })
    
    return preds, diagnostics


def method_metrics(rows, preds):
    ks = [5, 10, 20]
    recall = {k: [] for k in ks}
    hit = {k: [] for k in ks}
    
    for r, p in zip(rows, preds):
        rel = positive_relevant_ids(r)
        denom = max(1, len(rel))
        for k in ks:
            inter = len(set(p[:k]) & rel)
            recall[k].append(inter / denom)
            hit[k].append(1 if inter > 0 else 0)
    
    summary = {
        "recall@5": float(np.mean(recall[5])),
        "recall@10": float(np.mean(recall[10])),
        "recall@20": float(np.mean(recall[20])),
    }
    detail = {
        "recall@5": recall[5],
        "recall@10": recall[10],
        "recall@20": recall[20],
        "hit@5": hit[5],
        "hit@10": hit[10],
        "hit@20": hit[20],
        "any_hit@5": hit[5],
        "any_hit@10": hit[10],
        "any_hit@20": hit[20],
    }
    return summary, detail


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", type=str, required=True)
    parser.add_argument("--router-model", type=Path, required=True)
    parser.add_argument("--split-file", type=Path, required=True)
    parser.add_argument("--corpus-file", type=Path, required=True)
    parser.add_argument("--out-file", type=Path, required=True)
    parser.add_argument("--detail-file", type=Path, required=True)
    parser.add_argument("--max-queries", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--topk", type=int, default=20)
    parser.add_argument("--late-alpha", type=float, default=0.1)
    parser.add_argument("--preserve-dense-top", type=int, default=8)
    parser.add_argument("--max-per-source", type=int, default=4)
    parser.add_argument("--policy-switch-conf", type=float, default=0.6)
    parser.add_argument("--policy-dense-w", type=float, default=0.72)
    parser.add_argument("--policy-sparse-w", type=float, default=0.28)
    parser.add_argument("--policy-late-w", type=float, default=0.10)
    parser.add_argument("--policy-modality-w", type=float, default=0.10)
    parser.add_argument("--no-router", action="store_true")
    parser.add_argument("--cache-dir", type=Path, default=Path("artifacts/retrieval"))
    args = parser.parse_args()

    print("=" * 60)
    print("TESSERA-RAG V4: Conservative Fusion")
    print("=" * 60)
    
    # Load data
    rows = json.loads(args.split_file.read_text(encoding="utf-8"))[:args.max_queries]
    corpus = json.loads(args.corpus_file.read_text(encoding="utf-8"))
    
    q_texts = [r.get("query", "") for r in rows]
    q_ids = [r.get("id", f"q_{i}") for i, r in enumerate(rows)]
    doc_ids = [d["id"] for d in corpus]
    c_texts = [d.get("text", "") for d in corpus]
    
    print(f"[Data] Queries: {len(rows)}, Corpus: {len(corpus)}")
    
    # Cache setup
    args.cache_dir.mkdir(parents=True, exist_ok=True)
    q_key = make_cache_key(q_ids)
    c_key = make_cache_key(doc_ids)
    sparse_cache = args.cache_dir / f"tfidf_scores_{len(q_texts)}x{len(c_texts)}_{q_key}_{c_key}.npy"
    
    # Load or encode
    tokenizer, model, device, resolved = load_e5(args.model_dir)
    print(f"[E5] Device: {device}")
    model_key = hashlib.sha1(str(resolved).encode("utf-8")).hexdigest()[:12]
    q_cache = args.cache_dir / f"e5_query_{model_key}_{len(q_texts)}_{q_key}.npy"
    c_cache = args.cache_dir / f"e5_corpus_{model_key}_{len(c_texts)}_{c_key}.npy"
    
    if q_cache.exists():
        print(f"[Cache] Queries: {q_cache}")
        qv = np.load(q_cache)
    else:
        print("[Encode] Encoding queries...")
        qv = encode_texts(q_texts, tokenizer, model, device, batch_size=args.batch_size)
        np.save(q_cache, qv)
    
    if c_cache.exists():
        print(f"[Cache] Corpus: {c_cache}")
        cv = np.load(c_cache)
    else:
        c_texts = [d.get("text", "") for d in corpus]
        print("[Encode] Encoding corpus...")
        cv = encode_texts(c_texts, tokenizer, model, device, batch_size=args.batch_size)
        np.save(c_cache, cv)
    
    dense_scores = qv @ cv.T
    
    if sparse_cache.exists():
        print(f"[Cache] Sparse: {sparse_cache}")
        sparse_scores = np.load(sparse_cache)
    else:
        print("[Compute] Sparse scores...")
        from sklearn.feature_extraction.text import TfidfVectorizer
        c_texts = [d.get("text", "") for d in corpus]
        vec = TfidfVectorizer(stop_words="english", ngram_range=(1, 2), max_features=200000, min_df=2)
        c_mat = vec.fit_transform(c_texts)
        q_mat = vec.transform(q_texts)
        sparse_scores = (q_mat @ c_mat.T).toarray().astype(np.float32)
        np.save(sparse_cache, sparse_scores)
    
    # Load Router
    router = None
    if not args.no_router:
        print(f"[Router] Loading...")
        router = RouterInference(args.router_model, device=device)
    
    # Build predictions
    print("[Eval] Building predictions...")
    
    # Dense baseline
    pred_dense = []
    for i in range(len(rows)):
        idx = topk_indices(dense_scores[i], args.topk)
        pred_dense.append([doc_ids[j] for j in idx])
    
    # V4 main
    pred_main, main_diag = build_v4_preds(
        rows, q_texts, doc_ids, c_texts, dense_scores, sparse_scores,
        router,
        topk=args.topk,
        use_router=not args.no_router,
        late_alpha=args.late_alpha,
        preserve_dense_top=args.preserve_dense_top,
        max_per_source=args.max_per_source,
        policy_switch_conf=args.policy_switch_conf,
        policy_dense_w=args.policy_dense_w,
        policy_sparse_w=args.policy_sparse_w,
        policy_late_w=args.policy_late_w,
        policy_modality_w=args.policy_modality_w,
    )
    
    # Metrics
    dense_summary, dense_detail = method_metrics(rows, pred_dense)
    v4_summary, v4_detail = method_metrics(rows, pred_main)
    methods = {
        "baseline_dense": dense_summary,
        "tessera_v4": v4_summary,
        "main_tessera": v4_summary,
    }
    
    print("\n" + "=" * 60)
    print("Results:")
    print("=" * 60)
    for name, m in methods.items():
        print(f"{name:20s}: R@5={m['recall@5']:.4f} R@10={m['recall@10']:.4f} R@20={m['recall@20']:.4f}")
    
    # Save
    out = {
        "queries": len(rows),
        "corpus": len(corpus),
        "use_router": not args.no_router,
        "methods": methods,
        "diagnostics": {
            "avg_dense_overlap": float(np.mean([d["dense_overlap"] for d in main_diag])),
            "avg_main_dense_overlap_at_k": float(np.mean([d["dense_overlap_ratio"] for d in main_diag])),
            "avg_main_new_over_dense_at_k": float(np.mean([d["main_new_over_dense"] for d in main_diag])),
        }
    }
    
    args.out_file.parent.mkdir(parents=True, exist_ok=True)
    args.out_file.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    
    detail_out = {
        "queries": len(rows),
        "query_ids": q_ids,
        "methods": {
            "baseline_dense": dense_detail,
            "tessera_v4": v4_detail,
            "main_tessera": v4_detail,
        },
        "main_diagnostics": {
            "summary": {
                "avg_main_dense_overlap_at_k": float(np.mean([d["dense_overlap_ratio"] for d in main_diag])),
                "avg_main_new_over_dense_at_k": float(np.mean([d["main_new_over_dense"] for d in main_diag])),
            },
            "per_query": main_diag,
        },
    }
    args.detail_file.parent.mkdir(parents=True, exist_ok=True)
    args.detail_file.write_text(json.dumps(detail_out, ensure_ascii=False, indent=2), encoding="utf-8")
    
    print(f"\n[OK] Results: {args.out_file}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
