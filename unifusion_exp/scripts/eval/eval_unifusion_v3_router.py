#!/usr/bin/env python3
"""
UniFusion-RAG V3: 真正集成 Router 进行模态感知的检索

核心改进:
1. 使用 Router 预测每个查询需要的模态
2. 根据模态预测调整 dense/sparse 权重
3. 使用 Router 置信度进行动态融合
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
from unifusion_exp.utils.e5_embed import load_e5, encode_texts


def tokenize(text: str) -> set[str]:
    return set(re.findall(r"[a-z0-9]+", text.lower()))


def topk_indices(scores: np.ndarray, k: int) -> np.ndarray:
    if len(scores) <= k:
        return np.argsort(-scores)
    idx = np.argpartition(-scores, kth=k - 1)[:k]
    return idx[np.argsort(-scores[idx])]


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
        """
        Returns:
            preds: binary matrix (n_queries, 3) for [text, table, kg]
            confidences: max probability for each query
        """
        inputs = self.tokenizer(
            queries, padding=True, truncation=True, max_length=256, return_tensors="pt"
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        outputs = self.model(**inputs)
        logits = outputs.logits.cpu().numpy()
        
        # Sigmoid for multi-label
        probs = 1.0 / (1.0 + np.exp(-logits))
        preds = (probs >= threshold).astype(np.int64)
        
        # Handle empty predictions
        empty_mask = np.sum(preds, axis=1) == 0
        if np.any(empty_mask):
            top_idx = np.argmax(probs[empty_mask], axis=1)
            preds[empty_mask] = 0
            preds[empty_mask, top_idx] = 1
        
        # Confidence as max predicted probability
        confidences = np.max(probs * preds + (1 - preds) * (1 - probs), axis=1)
        
        return preds, confidences


def get_modality_weights(has_text: int, has_table: int, has_kg: int, router_conf: float) -> dict:
    """
    Get fusion weights based on predicted modalities and router confidence.
    """
    # Base weights
    w_dense = 0.6
    w_sparse = 0.4
    
    # Adjust based on modalities
    if has_table and not has_text:
        # Table-heavy queries benefit more from sparse (keyword) matching
        w_dense = 0.5
        w_sparse = 0.6
    elif has_kg and not has_text:
        # KG queries need different handling
        w_dense = 0.5
        w_sparse = 0.5
    elif has_text and has_table:
        # Hybrid queries
        w_dense = 0.55
        w_sparse = 0.55
    
    # Adjust based on router confidence
    # Low confidence -> rely more on dense (safer)
    if router_conf < 0.7:
        w_dense = min(0.8, w_dense + 0.1)
        w_sparse = max(0.2, w_sparse - 0.1)
    
    return {"dense": w_dense, "sparse": w_sparse}


def build_v3_preds(
    rows,
    query_texts,
    doc_ids,
    corpus_texts,
    dense_scores,
    sparse_scores,
    router: RouterInference,
    topk: int = 20,
    candidate_k: int = 200,
    use_router: bool = True,
):
    """Build predictions with Router-guided fusion."""
    preds = []
    diagnostics = []
    
    # Get router predictions
    if use_router:
        print("[Router] Predicting modalities...")
        route_preds, router_confs = router.predict_with_confidence(query_texts)
        print(f"[Router] Done. Text={route_preds[:,0].sum()}, Table={route_preds[:,1].sum()}, KG={route_preds[:,2].sum()}")
    else:
        route_preds = np.ones((len(rows), 3), dtype=np.int64)
        router_confs = np.ones(len(rows)) * 0.8
    
    for qi, row in enumerate(rows):
        has_text = route_preds[qi][0]
        has_table = route_preds[qi][1]
        has_kg = route_preds[qi][2]
        router_conf = router_confs[qi]
        
        # Get candidate pool
        d_idx = topk_indices(dense_scores[qi], candidate_k)
        s_idx = topk_indices(sparse_scores[qi], candidate_k)
        all_idx = list(set(d_idx.tolist() + s_idx.tolist()))
        
        # Get modality-aware weights
        weights = get_modality_weights(has_text, has_table, has_kg, router_conf)
        
        # Score candidates with modality-aware fusion
        scored = []
        d_norm = normalize_scores(dense_scores[qi])
        s_norm = normalize_scores(sparse_scores[qi])
        
        for j in all_idx:
            score = weights["dense"] * d_norm[j] + weights["sparse"] * s_norm[j]
            scored.append((j, score))
        
        # Sort and select
        scored.sort(key=lambda x: x[1], reverse=True)
        sel = [doc_ids[j] for j, _ in scored[:topk]]
        preds.append(sel)
        
        # Diagnostics
        dense_topk = [doc_ids[j] for j in d_idx[:topk]]
        overlap = len(set(sel) & set(dense_topk))
        diagnostics.append({
            "query_idx": qi,
            "routing": {"text": int(has_text), "table": int(has_table), "kg": int(has_kg)},
            "router_conf": float(router_conf),
            "weights": weights,
            "dense_overlap": overlap,
            "candidate_pool": len(all_idx),
        })
    
    return preds, diagnostics


def method_metrics(rows, preds):
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
            topk_set = set(p[:k])
            inter = len(topk_set & rel)
            any_hit[k].append(1 if inter > 0 else 0)
            recall[k].append(inter / denom)
            precision[k].append(inter / k)

    summary = {
        "avg_positive_qrels": float(np.mean(rel_count)) if rel_count else 0.0,
        "any_hit@5": float(np.mean(any_hit[5])),
        "any_hit@10": float(np.mean(any_hit[10])),
        "any_hit@20": float(np.mean(any_hit[20])),
        "recall@5": float(np.mean(recall[5])),
        "recall@10": float(np.mean(recall[10])),
        "recall@20": float(np.mean(recall[20])),
        "precision@5": float(np.mean(precision[5])),
        "precision@10": float(np.mean(precision[10])),
        "precision@20": float(np.mean(precision[20])),
    }
    return summary


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
    parser.add_argument("--no-router", action="store_true")
    parser.add_argument("--cache-dir", type=Path, default=Path("artifacts/retrieval"))
    args = parser.parse_args()

    print("=" * 60)
    print("UniFusion-RAG V3: Router-Guided Retrieval")
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
        print(f"[Cache] Loading queries from {q_cache}")
        qv = np.load(q_cache)
    else:
        print("[Encode] Encoding queries...")
        qv = encode_texts(q_texts, tokenizer, model, device, batch_size=args.batch_size)
        np.save(q_cache, qv)
    
    if c_cache.exists():
        print(f"[Cache] Loading corpus from {c_cache}")
        cv = np.load(c_cache)
    else:
        print("[Encode] Encoding corpus...")
        cv = encode_texts(c_texts, tokenizer, model, device, batch_size=args.batch_size)
        np.save(c_cache, cv)
    
    print("[Compute] Computing scores...")
    dense_scores = qv @ cv.T
    
    if sparse_cache.exists():
        print(f"[Cache] Loading sparse from {sparse_cache}")
        sparse_scores = np.load(sparse_cache)
    else:
        print("[Compute] Computing sparse scores...")
        sparse_scores = build_sparse_scores(c_texts, q_texts, max_features=200000)
        np.save(sparse_cache, sparse_scores)
    
    # Load Router
    router = None
    if not args.no_router:
        print(f"[Router] Loading from {args.router_model}")
        router = RouterInference(args.router_model, device=device)
    
    # Build predictions
    print("[Eval] Building predictions...")
    pred_dense = pred_from_score_matrix(dense_scores, doc_ids, args.topk)
    pred_sparse = pred_from_score_matrix(sparse_scores, doc_ids, args.topk)
    
    pred_main, main_diag = build_v3_preds(
        rows, q_texts, doc_ids, c_texts,
        dense_scores, sparse_scores,
        router,
        topk=args.topk,
        use_router=not args.no_router,
    )
    
    # Compute metrics
    methods = {
        "baseline_dense": method_metrics(rows, pred_dense),
        "baseline_sparse": method_metrics(rows, pred_sparse),
        "unifusion_v3": method_metrics(rows, pred_main),
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
            "avg_router_conf": float(np.mean([d["router_conf"] for d in main_diag])),
        }
    }
    
    args.out_file.parent.mkdir(parents=True, exist_ok=True)
    args.out_file.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    
    args.detail_file.parent.mkdir(parents=True, exist_ok=True)
    args.detail_file.write_text(json.dumps(main_diag, ensure_ascii=False, indent=2), encoding="utf-8")
    
    print(f"\n[OK] Results: {args.out_file}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
