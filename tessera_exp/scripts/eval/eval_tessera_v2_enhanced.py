#!/usr/bin/env python3
"""
TESSERA-RAG V2 Enhanced: 真正集成 Router + KG + Late Interaction 的检索评测

注意：该脚本为历史实验脚本（experimental），不作为论文主结果口径。
论文主线请使用 scripts/eval/run_e2e_table1c.py 与 scripts/eval/eval_tessera_retrieval_main.py。

主要改进:
1. 使用训练好的 DeBERTa Router 进行模态感知路由
2. 集成 Neo4j KG 进行路径级检索
3. 实现更精细的表格-文本 Late Interaction
4. 使用实体链接进行跨模态对齐
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
from neo4j import GraphDatabase

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from tessera_exp.utils.e5_embed import load_e5, encode_texts


# =============================================================================
# Utility Functions
# =============================================================================

def tokenize(text: str) -> Set[str]:
    """Simple tokenization for overlap computation."""
    return set(re.findall(r"[a-z0-9]+", text.lower()))


def topk_indices(scores: np.ndarray, k: int) -> np.ndarray:
    """Get top-k indices efficiently."""
    if len(scores) <= k:
        return np.argsort(-scores)
    idx = np.argpartition(-scores, kth=k - 1)[:k]
    return idx[np.argsort(-scores[idx])]


def source_prefix(doc_id: str) -> str:
    """Extract source prefix from doc_id (e.g., 'nq_123' -> 'nq')."""
    if "_" not in doc_id:
        return doc_id
    return doc_id.rsplit("_", 1)[0]


def positive_relevant_ids(row: dict) -> Set[str]:
    """Extract positive relevant chunk IDs from a query row."""
    rel = set()
    for chunk_id, label in row.get("relevant_chunks", {}).items():
        try:
            if float(label) > 0:
                rel.add(chunk_id)
        except Exception:
            continue
    return rel


def pred_from_score_matrix(scores: np.ndarray, doc_ids: List[str], topk: int) -> List[List[str]]:
    """Convert score matrix to top-k predictions."""
    preds = []
    for i in range(scores.shape[0]):
        idx = topk_indices(scores[i], topk)
        preds.append([doc_ids[j] for j in idx])
    return preds


def build_sparse_scores(corpus_texts: List[str], query_texts: List[str], max_features: int):
    """Build TF-IDF sparse scores."""
    vec = TfidfVectorizer(stop_words="english", ngram_range=(1, 2), max_features=max_features, min_df=2)
    c_mat = vec.fit_transform(corpus_texts)
    q_mat = vec.transform(query_texts)
    return (q_mat @ c_mat.T).toarray().astype(np.float32)


def normalize_scores(values: np.ndarray) -> np.ndarray:
    """Min-max normalize scores to [0, 1]."""
    if values.size == 0:
        return values
    lo = float(values.min())
    hi = float(values.max())
    if hi - lo < 1e-9:
        return np.zeros_like(values)
    return (values - lo) / (hi - lo)


def make_cache_key(ids: List[str], max_items: int = 2048) -> str:
    """Create cache key from IDs."""
    if not ids:
        return "empty"
    if len(ids) <= max_items:
        sampled = ids
    else:
        step = max(1, len(ids) // max_items)
        sampled = ids[::step][:max_items]
    payload = "|".join(sampled) + f"|n={len(ids)}"
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()[:12]


# =============================================================================
# Router Integration
# =============================================================================

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
    def predict(self, queries: List[str], threshold: float = 0.5) -> np.ndarray:
        """
        Predict modality labels for queries.
        Returns: binary matrix of shape (n_queries, 3) for [text, table, graph]
        """
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
        
        # Sigmoid activation for multi-label
        probs = 1.0 / (1.0 + np.exp(-logits))
        preds = (probs >= threshold).astype(np.int64)
        
        # Handle empty predictions by taking argmax
        empty_mask = np.sum(preds, axis=1) == 0
        if np.any(empty_mask):
            top_idx = np.argmax(probs[empty_mask], axis=1)
            preds[empty_mask] = 0
            preds[empty_mask, top_idx] = 1
            
        return preds


# =============================================================================
# Knowledge Graph Retrieval
# =============================================================================

class KGRetriever:
    """Neo4j-based knowledge graph retriever with entity linking."""
    
    def __init__(self, uri: str = "bolt://127.0.0.1:7687", user: str = "neo4j", password: str = "password"):
        self.driver = GraphDatabase.driver(uri, auth=(user, password))
        
    def close(self):
        self.driver.close()
        
    def extract_entities_from_query(self, query: str) -> List[str]:
        """Simple entity extraction from query (can be improved with NER)."""
        # For now, extract capitalized phrases and quoted text
        entities = []
        
        # Extract quoted text
        quoted = re.findall(r'"([^"]+)"', query)
        entities.extend(quoted)
        
        # Extract capitalized phrases (potential entities)
        words = query.split()
        current = []
        for word in words:
            clean = re.sub(r'[^\w]', '', word)
            if clean and clean[0].isupper():
                current.append(clean)
            else:
                if len(current) >= 1:
                    entities.append(" ".join(current))
                current = []
        if len(current) >= 1:
            entities.append(" ".join(current))
            
        return list(set(entities))
    
    def get_2hop_neighbors(self, entity_id: str) -> List[Dict]:
        """Get 2-hop neighbors from Neo4j."""
        # Normalize entity ID
        if entity_id.startswith("m."):
            entity_id = "/m/" + entity_id[2:]
        elif entity_id.startswith("g."):
            entity_id = "/g/" + entity_id[2:]
            
        with self.driver.session() as session:
            result = session.run(
                """
                MATCH (s:Entity {id: $id})-[r1:REL]->(n1:Entity)
                OPTIONAL MATCH (n1)-[r2:REL]->(n2:Entity)
                RETURN s.id as source, r1.rel as rel1, n1.id as neighbor1,
                       r2.rel as rel2, n2.id as neighbor2
                LIMIT 50
                """,
                id=entity_id
            )
            paths = []
            for record in result:
                path = {
                    "source": record["source"],
                    "rel1": record["rel1"],
                    "neighbor1": record["neighbor1"],
                    "rel2": record["rel2"],
                    "neighbor2": record["neighbor2"]
                }
                paths.append(path)
            return paths
    
    def compute_path_score(self, query_emb: np.ndarray, paths: List[Dict]) -> float:
        """
        Compute PathMaxSim score.
        For now, return simple path count as proxy.
        Can be enhanced with TransE embedding similarity.
        """
        if not paths:
            return 0.0
        # Number of unique 2-hop paths as relevance signal
        unique_paths = set()
        for p in paths:
            path_key = f"{p['rel1']}->{p['neighbor1']}"
            if p['neighbor2']:
                path_key += f"->{p['rel2']}->{p['neighbor2']}"
            unique_paths.add(path_key)
        return min(1.0, len(unique_paths) / 10.0)  # Normalize


# =============================================================================
# Enhanced Late Interaction
# =============================================================================

def compute_cellmaxsim_score(
    query_text: str,
    doc_text: str,
    query_emb: np.ndarray,
    doc_emb: np.ndarray,
    alpha: float = 0.1
) -> float:
    """
    Compute CellMaxSim-like score for table/document.
    Combines semantic similarity with structure-aware token overlap.
    """
    # Base semantic similarity
    semantic_sim = np.dot(query_emb, doc_emb)
    
    # Structure-aware overlap (simplified CellMaxSim)
    query_tokens = tokenize(query_text)
    doc_tokens = tokenize(doc_text)
    
    # Jaccard similarity as structural signal
    if query_tokens and doc_tokens:
        jaccard = len(query_tokens & doc_tokens) / len(query_tokens | doc_tokens)
    else:
        jaccard = 0.0
    
    # Combined score
    return semantic_sim + alpha * jaccard


def compute_enhanced_fusion_score(
    query_idx: int,
    doc_idx: int,
    dense_scores: np.ndarray,
    sparse_scores: np.ndarray,
    query_texts: List[str],
    corpus_texts: List[str],
    query_embs: np.ndarray,
    corpus_embs: np.ndarray,
    modality_flags: Dict[str, int],
    late_alpha: float = 0.15,
    kg_alpha: float = 0.1
) -> float:
    """
    Compute enhanced fusion score with Late Interaction.
    
    modality_flags: dict with keys 'has_text', 'has_table', 'has_kg'
    """
    # Base dense and sparse scores (normalized)
    d_score = dense_scores[query_idx][doc_idx]
    s_score = sparse_scores[query_idx][doc_idx]
    
    # Normalize to [0, 1]
    d_norm = normalize_scores(dense_scores[query_idx])[doc_idx]
    s_norm = normalize_scores(sparse_scores[query_idx])[doc_idx]
    
    # Base fusion
    base_score = 0.6 * d_norm + 0.4 * s_norm
    
    # Late Interaction bonus
    li_bonus = 0.0
    if modality_flags.get('has_table', 0) or modality_flags.get('has_text', 0):
        li_bonus = compute_cellmaxsim_score(
            query_texts[query_idx],
            corpus_texts[doc_idx],
            query_embs[query_idx],
            corpus_embs[doc_idx],
            alpha=late_alpha
        )
    
    return base_score + late_alpha * li_bonus


# =============================================================================
# Main Enhanced Retrieval
# =============================================================================

def build_enhanced_preds(
    rows: List[dict],
    query_texts: List[str],
    doc_ids: List[str],
    corpus_texts: List[str],
    dense_scores: np.ndarray,
    sparse_scores: np.ndarray,
    query_embs: np.ndarray,
    corpus_embs: np.ndarray,
    router: RouterInference,
    kg_retriever: KGRetriever,
    topk: int = 20,
    late_alpha: float = 0.15,
    kg_alpha: float = 0.1,
    use_router: bool = True,
    use_kg: bool = True,
) -> Tuple[List[List[str]], List[dict]]:
    """
    Build predictions with enhanced fusion including Router and KG.
    """
    preds = []
    diagnostics = []
    
    # Get routing decisions if using router
    if use_router:
        print("[Router] Predicting modalities for queries...")
        route_preds = router.predict(query_texts)
    else:
        route_preds = np.ones((len(rows), 3), dtype=np.int64)  # All modalities
    
    for qi, row in enumerate(rows):
        # Routing flags
        has_text = route_preds[qi][0]
        has_table = route_preds[qi][1]
        has_kg = route_preds[qi][2]
        
        modality_flags = {
            'has_text': has_text,
            'has_table': has_table,
            'has_kg': has_kg
        }
        
        # Get candidate pool from dense + sparse
        d_idx = topk_indices(dense_scores[qi], 200)
        s_idx = topk_indices(sparse_scores[qi], 200)
        all_idx = list(set(d_idx.tolist() + s_idx.tolist()))
        
        # KG enhancement
        kg_bonus = {}
        if use_kg and has_kg:
            entities = kg_retriever.extract_entities_from_query(query_texts[qi])
            for e in entities:
                paths = kg_retriever.get_2hop_neighbors(e)
                if paths:
                    # Boost documents that mention related entities
                    for j in all_idx:
                        doc_text_lower = corpus_texts[j].lower()
                        for path in paths[:5]:
                            neighbor = path.get('neighbor1', '')
                            if neighbor and neighbor.lower() in doc_text_lower:
                                kg_bonus[j] = kg_bonus.get(j, 0.0) + kg_alpha
        
        # Score all candidates
        scored = []
        for j in all_idx:
            score = compute_enhanced_fusion_score(
                qi, j, dense_scores, sparse_scores,
                query_texts, corpus_texts,
                query_embs, corpus_embs,
                modality_flags,
                late_alpha=late_alpha,
                kg_alpha=kg_alpha
            )
            # Add KG bonus
            score += kg_bonus.get(j, 0.0)
            scored.append((j, score))
        
        # Sort and get top-k
        scored.sort(key=lambda x: x[1], reverse=True)
        sel = [doc_ids[j] for j, _ in scored[:topk]]
        preds.append(sel)
        
        # Diagnostics
        dense_topk = [doc_ids[j] for j in d_idx[:topk]]
        overlap = len(set(sel) & set(dense_topk))
        diagnostics.append({
            "query_idx": qi,
            "routing": {"text": int(has_text), "table": int(has_table), "kg": int(has_kg)},
            "candidate_pool_size": len(all_idx),
            "dense_overlap_at_k": overlap,
            "kg_entities_found": len(entities) if use_kg and has_kg else 0,
        })
    
    return preds, diagnostics


def method_metrics(rows: List[dict], preds: List[List[str]]) -> Tuple[dict, dict]:
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
        "any_hit@5": any_hit[5],
        "any_hit@10": any_hit[10],
        "any_hit@20": any_hit[20],
        "recall@5": recall[5],
        "recall@10": recall[10],
        "recall@20": recall[20],
        "precision@5": precision[5],
        "precision@10": precision[10],
        "precision@20": precision[20],
    }
    return summary, detail


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate TESSERA V2 Enhanced")
    parser.add_argument(
        "--experimental-ok",
        action="store_true",
        help="Acknowledge this script is experimental and not paper-mainline.",
    )
    parser.add_argument("--model-dir", type=str, required=True, help="E5 model directory")
    parser.add_argument("--router-model", type=Path, required=True, help="Trained Router model path")
    parser.add_argument("--split-file", type=Path, required=True, help="Query split file")
    parser.add_argument("--corpus-file", type=Path, required=True, help="Corpus file")
    parser.add_argument("--out-file", type=Path, required=True)
    parser.add_argument("--detail-file", type=Path, required=True)
    parser.add_argument("--max-queries", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--topk", type=int, default=20)
    parser.add_argument("--sparse-max-features", type=int, default=200000)
    parser.add_argument("--late-alpha", type=float, default=0.15, help="Late interaction weight")
    parser.add_argument("--kg-alpha", type=float, default=0.1, help="KG bonus weight")
    parser.add_argument("--router-threshold", type=float, default=0.5)
    parser.add_argument("--cache-dir", type=Path, default=Path("artifacts/retrieval"))
    parser.add_argument("--no-router", action="store_true", help="Disable router")
    parser.add_argument("--no-kg", action="store_true", help="Disable KG")
    parser.add_argument("--neo4j-uri", type=str, default="bolt://127.0.0.1:7687")
    parser.add_argument("--neo4j-user", type=str, default="neo4j")
    parser.add_argument("--neo4j-password", type=str, default="password")
    args = parser.parse_args()

    if not args.experimental_ok:
        raise SystemExit(
            "Blocked: eval_tessera_v2_enhanced.py is experimental and not paper-mainline. "
            "Re-run with --experimental-ok only for exploratory checks."
        )

    print("=" * 60)
    print("TESSERA-RAG V2 Enhanced Evaluation")
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
    
    # Load E5 model
    tokenizer, model, device, resolved = load_e5(args.model_dir)
    print(f"[E5] Model: {resolved}, Device: {device}")
    model_key = hashlib.sha1(str(resolved).encode("utf-8")).hexdigest()[:12]
    q_cache = args.cache_dir / f"e5_query_{model_key}_{len(q_texts)}_{q_key}.npy"
    c_cache = args.cache_dir / f"e5_corpus_{model_key}_{len(c_texts)}_{c_key}.npy"
    
    # Encode or load queries
    if q_cache.exists() and np.load(q_cache, mmap_mode="r").shape[0] == len(q_texts):
        print(f"[Cache] Loading query embeddings from {q_cache}")
        qv = np.load(q_cache)
    else:
        print("[Encode] Encoding queries...")
        qv = encode_texts(q_texts, tokenizer, model, device, batch_size=args.batch_size)
        np.save(q_cache, qv)
    
    # Encode or load corpus
    if c_cache.exists() and np.load(c_cache, mmap_mode="r").shape[0] == len(c_texts):
        print(f"[Cache] Loading corpus embeddings from {c_cache}")
        cv = np.load(c_cache)
    else:
        print("[Encode] Encoding corpus...")
        cv = encode_texts(c_texts, tokenizer, model, device, batch_size=args.batch_size)
        np.save(c_cache, cv)
    
    # Compute dense scores
    print("[Compute] Computing dense scores...")
    dense_scores = qv @ cv.T
    
    # Compute sparse scores
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
    
    # Load KG
    kg_retriever = None
    if not args.no_kg:
        print(f"[KG] Connecting to Neo4j at {args.neo4j_uri}")
        kg_retriever = KGRetriever(args.neo4j_uri, args.neo4j_user, args.neo4j_password)
    
    # Build baseline predictions
    print("[Eval] Building baseline predictions...")
    pred_dense = pred_from_score_matrix(dense_scores, doc_ids, args.topk)
    pred_sparse = pred_from_score_matrix(sparse_scores, doc_ids, args.topk)
    
    # Build enhanced predictions
    print("[Eval] Building enhanced TESSERA predictions...")
    pred_main, main_diag = build_enhanced_preds(
        rows, q_texts, doc_ids, c_texts,
        dense_scores, sparse_scores, qv, cv,
        router, kg_retriever,
        topk=args.topk,
        late_alpha=args.late_alpha,
        kg_alpha=args.kg_alpha,
        use_router=not args.no_router,
        use_kg=not args.no_kg,
    )
    
    # Compute metrics
    methods = {
        "baseline_dense": pred_dense,
        "baseline_sparse_tfidf": pred_sparse,
        "tessera_v2_enhanced": pred_main,
    }
    
    metrics = {}
    for name, pred in methods.items():
        m, _ = method_metrics(rows, pred)
        metrics[name] = m
        print(f"[Result] {name}: R@5={m['recall@5']:.4f}, R@10={m['recall@10']:.4f}, R@20={m['recall@20']:.4f}")
    
    # Save results
    out = {
        "queries": len(rows),
        "corpus": len(corpus),
        "config": {
            "late_alpha": args.late_alpha,
            "kg_alpha": args.kg_alpha,
            "use_router": not args.no_router,
            "use_kg": not args.no_kg,
        },
        "methods": metrics,
        "main_method": "tessera_v2_enhanced",
        "diagnostics": {
            "avg_dense_overlap": float(np.mean([d["dense_overlap_at_k"] for d in main_diag])),
            "routing_distribution": {
                "text_only": sum(1 for d in main_diag if d["routing"]["text"] and not d["routing"]["table"] and not d["routing"]["kg"]),
                "table_only": sum(1 for d in main_diag if d["routing"]["table"] and not d["routing"]["text"] and not d["routing"]["kg"]),
                "kg_only": sum(1 for d in main_diag if d["routing"]["kg"] and not d["routing"]["text"] and not d["routing"]["table"]),
                "multi_modal": sum(1 for d in main_diag if sum(d["routing"].values()) > 1),
            }
        }
    }
    
    args.out_file.parent.mkdir(parents=True, exist_ok=True)
    args.out_file.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    
    args.detail_file.parent.mkdir(parents=True, exist_ok=True)
    args.detail_file.write_text(json.dumps(main_diag, ensure_ascii=False, indent=2), encoding="utf-8")
    
    print(f"[OK] Results saved to {args.out_file}")
    print(f"[OK] Diagnostics saved to {args.detail_file}")
    
    # Cleanup
    if kg_retriever:
        kg_retriever.close()
    
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
