#!/usr/bin/env python3
"""
UniFusion-RAG V2 Simplified: 先测试 Router + Late Interaction (无 KG)
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re
import sys
from typing import Dict, List, Set, Tuple

import numpy as np
import torch
from sklearn.feature_extraction.text import TfidfVectorizer

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from unifusion_exp.utils.e5_embed import load_e5, encode_texts


def tokenize(text: str) -> Set[str]:
    return set(re.findall(r"[a-z0-9]+", text.lower()))


def topk_indices(scores: np.ndarray, k: int) -> np.ndarray:
    if len(scores) <= k:
        return np.argsort(-scores)
    idx = np.argpartition(-scores, kth=k - 1)[:k]
    return idx[np.argsort(-scores[idx])]


def positive_relevant_ids(row: dict) -> Set[str]:
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


class RouterInference:
    """Inference wrapper for trained DeBERTa Router."""
    
    def __init__(self, model_path: Path, device: str = "cuda" if torch.cuda.is_available() else "cpu"):
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
        
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(str(model_path))
        self.model = AutoModelForSequenceClassification.from_pretrained(str(model_path))
        self.model.to(device)
        self.model.eval()
        
    @torch.no_grad()
    def predict(self, queries: list[str], threshold: float = 0.5) -> np.ndarray:
        """Predict modality labels [text, table, kg] for queries."""
        inputs = self.tokenizer(
            queries,
            padding=True,
            truncation=True,
            max_length=256,
            return_tensors="pt"
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
            
        return preds


def compute_late_interaction_score(
    query_text: str,
    doc_text: str,
    alpha: float = 0.1
) -> float:
    """Compute late interaction bonus using token overlap."""
    query_tokens = tokenize(query_text)
    doc_tokens = tokenize(doc_text)
    
    if not query_tokens or not doc_tokens:
        return 0.0
    
    # Jaccard similarity
    intersection = len(query_tokens & doc_tokens)
    union = len(query_tokens | doc_tokens)
    return alpha * (intersection / union if union > 0 else 0.0)


def build_enhanced_preds_v2(
    rows: list[dict],
    query_texts: list[str],
    doc_ids: list[str],
    corpus_texts: list[str],
    dense_scores: np.ndarray,
    sparse_scores: np.ndarray,
    router: RouterInference,
    topk: int = 20,
    late_alpha: float = 0.15,
    use_router: bool = True,
) -> Tuple[list[list[str]], list[dict]]:
    """Build predictions with Router and Late Interaction."""
    preds = []
    diagnostics = []
    
    # Get routing decisions
    if use_router:
        print("[Router] Predicting modalities...")
        route_preds = router.predict(query_texts)
        print(f"[Router] Done. Distribution: Text={route_preds[:,0].sum()}, Table={route_preds[:,1].sum()}, KG={route_preds[:,2].sum()}")
    else:
        route_preds = np.ones((len(rows), 3), dtype=np.int64)
    
    for qi, row in enumerate(rows):
        has_text = route_preds[qi][0]
        has_table = route_preds[qi][1]
        
        # Get candidates from dense + sparse
        d_idx = topk_indices(dense_scores[qi], 200)
        s_idx = topk_indices(sparse_scores[qi], 200)
        all_idx = list(set(d_idx.tolist() + s_idx.tolist()))
        
        # Score candidates
        scored = []
        for j in all_idx:
            # Base score: weighted combination of normalized dense and sparse
            d_norm = (dense_scores[qi][j] - dense_scores[qi].min()) / (dense_scores[qi].max() - dense_scores[qi].min() + 1e-9)
            s_norm = (sparse_scores[qi][j] - sparse_scores[qi].min()) / (sparse_scores[qi].max() - sparse_scores[qi].min() + 1e-9)
            base_score = 0.6 * d_norm + 0.4 * s_norm
            
            # Late interaction bonus
            li_bonus = 0.0
            if has_text or has_table:
                li_bonus = compute_late_interaction_score(
                    query_texts[qi], corpus_texts[j], alpha=late_alpha
                )
            
            final_score = base_score + li_bonus
            scored.append((j, final_score))
        
        # Sort and select top-k
        scored.sort(key=lambda x: x[1], reverse=True)
        sel = [doc_ids[j] for j, _ in scored[:topk]]
        preds.append(sel)
        
        # Diagnostics
        dense_topk = [doc_ids[j] for j in d_idx[:topk]]
        overlap = len(set(sel) & set(dense_topk))
        diagnostics.append({
            "query_idx": qi,
            "routing": {"text": int(has_text), "table": int(has_table), "kg": int(route_preds[qi][2])},
            "dense_overlap": overlap,
            "candidate_pool": len(all_idx),
        })
    
    return preds, diagnostics


def method_metrics(rows: list[dict], preds: list[list[str]]) -> Tuple[dict, dict]:
    """Compute evaluation metrics."""
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
        "recall@5": recall[5],
        "recall@10": recall[10],
        "recall@20": recall[20],
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
    parser.add_argument("--sparse-max-features", type=int, default=200000)
    parser.add_argument("--late-alpha", type=float, default=0.15)
    parser.add_argument("--no-router", action="store_true")
    parser.add_argument("--cache-dir", type=Path, default=Path("artifacts/retrieval"))
    args = parser.parse_args()

    print("=" * 60)
    print("UniFusion-RAG V2 Simple Evaluation")
    print("=" * 60)
    
    # Load data
    rows = json.loads(args.split_file.read_text(encoding="utf-8"))[:args.max_queries]
    corpus = json.loads(args.corpus_file.read_text(encoding="utf-8"))
    
    q_texts = [r.get("query", "") for r in rows]
    q_ids = [r.get("id", f"q_{i}") for i, r in enumerate(rows)]
    doc_ids = [d["id"] for d in corpus]
    c_texts = [d.get("text", "") for d in corpus]
    
    print(f"[Data] Queries: {len(rows)}, Corpus: {len(corpus)}")
    
    # Setup cache
    args.cache_dir.mkdir(parents=True, exist_ok=True)
    q_key = make_cache_key(q_ids)
    c_key = make_cache_key(doc_ids)
    sparse_cache = args.cache_dir / f"tfidf_scores_{len(q_texts)}x{len(c_texts)}_{q_key}_{c_key}.npy"
    
    # Load E5
    print("[E5] Loading model...")
    tokenizer, model, device, resolved = load_e5(args.model_dir)
    print(f"[E5] Device: {device}")
    model_key = hashlib.sha1(str(resolved).encode("utf-8")).hexdigest()[:12]
    q_cache = args.cache_dir / f"e5_query_{model_key}_{len(q_texts)}_{q_key}.npy"
    c_cache = args.cache_dir / f"e5_corpus_{model_key}_{len(c_texts)}_{c_key}.npy"
    
    # Encode queries
    if q_cache.exists():
        print(f"[Cache] Loading query embeddings from {q_cache}")
        qv = np.load(q_cache)
    else:
        print("[Encode] Encoding queries...")
        qv = encode_texts(q_texts, tokenizer, model, device, batch_size=args.batch_size)
        np.save(q_cache, qv)
    
    # Encode corpus
    if c_cache.exists():
        print(f"[Cache] Loading corpus embeddings from {c_cache}")
        cv = np.load(c_cache)
    else:
        print("[Encode] Encoding corpus (this may take a while)...")
        cv = encode_texts(c_texts, tokenizer, model, device, batch_size=args.batch_size)
        np.save(c_cache, cv)
    
    # Compute scores
    print("[Compute] Computing dense scores...")
    dense_scores = qv @ cv.T
    
    if sparse_cache.exists():
        print(f"[Cache] Loading sparse scores from {sparse_cache}")
        sparse_scores = np.load(sparse_cache)
    else:
        print("[Compute] Computing sparse scores...")
        sparse_scores = build_sparse_scores(c_texts, q_texts, max_features=args.sparse_max_features)
        np.save(sparse_cache, sparse_scores)
    
    # Load Router
    router = None
    if not args.no_router:
        print(f"[Router] Loading from {args.router_model}")
        router = RouterInference(args.router_model, device=device)
    
    # Build predictions
    print("[Eval] Building baseline predictions...")
    pred_dense = pred_from_score_matrix(dense_scores, doc_ids, args.topk)
    pred_sparse = pred_from_score_matrix(sparse_scores, doc_ids, args.topk)
    
    print("[Eval] Building enhanced predictions...")
    pred_main, main_diag = build_enhanced_preds_v2(
        rows, q_texts, doc_ids, c_texts,
        dense_scores, sparse_scores,
        router,
        topk=args.topk,
        late_alpha=args.late_alpha,
        use_router=not args.no_router,
    )
    
    # Compute metrics
    methods = {
        "baseline_dense": pred_dense,
        "baseline_sparse_tfidf": pred_sparse,
        "unifusion_v2_simple": pred_main,
    }
    
    metrics = {}
    for name, pred in methods.items():
        m, _ = method_metrics(rows, pred)
        metrics[name] = m
        print(f"[Result] {name}:")
        print(f"         R@5={m['recall@5']:.4f}, R@10={m['recall@10']:.4f}, R@20={m['recall@20']:.4f}")
    
    # Save results
    out = {
        "queries": len(rows),
        "corpus": len(corpus),
        "config": {"late_alpha": args.late_alpha, "use_router": not args.no_router},
        "methods": metrics,
        "diagnostics": {
            "avg_dense_overlap": float(np.mean([d["dense_overlap"] for d in main_diag])),
        }
    }
    
    args.out_file.parent.mkdir(parents=True, exist_ok=True)
    args.out_file.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    
    args.detail_file.parent.mkdir(parents=True, exist_ok=True)
    args.detail_file.write_text(json.dumps(main_diag, ensure_ascii=False, indent=2), encoding="utf-8")
    
    print(f"[OK] Results saved to {args.out_file}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
