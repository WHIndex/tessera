#!/usr/bin/env python3
from __future__ import annotations

import argparse
from collections import defaultdict
import json
import math
from pathlib import Path
import sys
from typing import Any, Sequence

import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from tessera_exp.e2e.evidence_set_reranker import (  # noqa: E402
    ESR_FEATURE_NAMES,
    EvidenceSetRerankerBundle,
    EvidenceSetRerankerConfig,
    build_evidence_features,
    save_evidence_set_reranker_bundle,
    source_bucket,
)


def is_test_like_path(path: Path) -> bool:
    raw = str(path).lower()
    return "test" in path.name.lower() or "/test" in raw or "\\test" in raw


def iter_jsonl(path: Path, max_examples: int = 0) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            rows.append(json.loads(line))
            if int(max_examples) > 0 and len(rows) >= int(max_examples):
                break
    return rows


def ranking_for_row(row: dict[str, Any], method: str) -> list[str]:
    rankings = row.get("rankings", {}) or {}
    return [str(x) for x in (rankings.get(method) or [])]


def qrels_for_row(row: dict[str, Any]) -> dict[str, float]:
    out: dict[str, float] = {}
    for doc_id, raw in (row.get("qrels", {}) or {}).items():
        try:
            grade = float(raw)
        except Exception:
            continue
        if grade > 0.0:
            out[str(doc_id)] = grade
    return out


def collect_needed_doc_ids(rows: Sequence[dict[str, Any]], *, method: str, pool_k: int) -> set[str]:
    needed: set[str] = set()
    for row in rows:
        needed.update(ranking_for_row(row, method)[: int(pool_k)])
        needed.update(qrels_for_row(row))
    return needed


def load_corpus_texts(paths: Sequence[Path], needed_ids: set[str] | None = None) -> dict[str, str]:
    corpus: dict[str, str] = {}
    needed = set(needed_ids or [])
    for path in paths:
        if not path.exists():
            raise FileNotFoundError(f"corpus-json not found: {path}")
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, dict):
            iterator = data.items()
            for doc_id, payload in iterator:
                doc_id = str(doc_id)
                if needed and doc_id not in needed:
                    continue
                if isinstance(payload, dict):
                    text = str(payload.get("text", "") or payload.get("contents", "") or "")
                else:
                    text = str(payload or "")
                if text and doc_id not in corpus:
                    corpus[doc_id] = text
            continue
        if not isinstance(data, list):
            continue
        for item in data:
            if not isinstance(item, dict):
                continue
            doc_id = str(item.get("id", "") or item.get("doc_id", "") or item.get("chunk_id", ""))
            if not doc_id:
                continue
            if needed and doc_id not in needed:
                continue
            text = str(item.get("text", "") or item.get("contents", "") or "")
            if text and doc_id not in corpus:
                corpus[doc_id] = text
    return corpus


def make_dataset(
    rows: Sequence[dict[str, Any]],
    *,
    method: str,
    corpus_texts: dict[str, str],
    pool_k: int,
    positive_weight: float,
    hard_negative_weight: float,
    grade_power: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    xs: list[np.ndarray] = []
    ys: list[float] = []
    ws: list[float] = []
    stats: dict[str, Any] = {
        "queries": 0,
        "examples": 0,
        "positive_examples": 0,
        "positive_by_source": defaultdict(int),
        "queries_with_positive_in_pool": 0,
        "queries_without_positive_in_pool": 0,
        "missing_doc_text": 0,
    }
    for row in rows:
        ranked = ranking_for_row(row, method)
        if not ranked:
            continue
        qrels = qrels_for_row(row)
        if not qrels:
            continue
        stats["queries"] += 1
        query_text = str(row.get("query", ""))
        query_id = str(row.get("query_id", ""))
        trace = row.get("trace", {}) or {}
        max_grade = max(float(v) for v in qrels.values()) if qrels else 1.0
        row_weights: list[float] = []
        row_start = len(xs)
        has_positive = False
        for pos, doc_id in enumerate(ranked[: int(pool_k)]):
            doc_id = str(doc_id)
            doc_text = corpus_texts.get(doc_id, "")
            if not doc_text:
                stats["missing_doc_text"] += 1
            feat = build_evidence_features(
                query_text=query_text,
                query_id=query_id,
                doc_id=doc_id,
                doc_text=doc_text,
                ranked_doc_ids=ranked,
                rank_position=pos,
                trace=trace,
            )
            grade = float(qrels.get(doc_id, 0.0))
            if grade > 0.0:
                has_positive = True
                label = (grade / max(1e-6, max_grade)) ** float(grade_power)
                weight = float(positive_weight) * (1.0 + 0.35 * label)
                stats["positive_examples"] += 1
                stats["positive_by_source"][source_bucket(doc_id)] += 1
            else:
                label = 0.0
                weight = 1.0 + float(hard_negative_weight) / float(pos + 1)
            xs.append(feat)
            ys.append(float(label))
            row_weights.append(float(weight))
        if has_positive:
            stats["queries_with_positive_in_pool"] += 1
        else:
            stats["queries_without_positive_in_pool"] += 1
        if row_weights:
            scale = len(row_weights) / max(1e-6, float(sum(row_weights)))
            for i in range(row_start, len(xs)):
                ws.append(float(row_weights[i - row_start] * scale))

    stats["examples"] = int(len(xs))
    stats["positive_by_source"] = dict(stats["positive_by_source"])
    if not xs:
        return (
            np.zeros((0, len(ESR_FEATURE_NAMES)), dtype=np.float32),
            np.zeros((0,), dtype=np.float32),
            np.zeros((0,), dtype=np.float32),
            stats,
        )
    x = np.nan_to_num(np.vstack(xs).astype(np.float32), nan=0.0, posinf=1e6, neginf=-1e6)
    y = np.asarray(ys, dtype=np.float32)
    w = np.asarray(ws, dtype=np.float32)
    return x, y, w, stats


def dcg(grades: Sequence[float]) -> float:
    return float(sum((2.0 ** float(grade) - 1.0) / math.log2(rank + 2.0) for rank, grade in enumerate(grades)))


def method_metrics(rows: Sequence[dict[str, Any]], preds: Sequence[Sequence[str]], ks: Sequence[int]) -> tuple[dict[str, float], dict[str, list[float]]]:
    details: dict[str, list[float]] = {}
    rel_counts: list[int] = []
    for k in ks:
        details[f"ndcg@{k}"] = []
        details[f"map@{k}"] = []
        details[f"hits@{k}"] = []
        details[f"any_hit@{k}"] = []
    for row, pred in zip(rows, preds):
        qrels = qrels_for_row(row)
        rel = {doc_id for doc_id, grade in qrels.items() if grade > 0.0}
        rel_counts.append(len(rel))
        for k in ks:
            top = list(pred)[: int(k)]
            grades = [float(qrels.get(doc_id, 0.0)) for doc_id in top]
            ideal = sorted([float(v) for v in qrels.values() if float(v) > 0.0], reverse=True)[: int(k)]
            idcg = dcg(ideal)
            ndcg = dcg(grades) / idcg if idcg > 0.0 else 0.0
            hits = 0
            ap_sum = 0.0
            for rank, doc_id in enumerate(top, start=1):
                if doc_id in rel:
                    hits += 1
                    ap_sum += hits / rank
            details[f"ndcg@{k}"].append(float(ndcg))
            details[f"map@{k}"].append(float(ap_sum / len(rel) if rel else 0.0))
            details[f"hits@{k}"].append(float(hits))
            details[f"any_hit@{k}"].append(float(hits > 0))
    summary = {"avg_positive_qrels": float(np.mean(rel_counts)) if rel_counts else 0.0}
    for key, vals in details.items():
        summary[key] = float(np.mean(vals)) if vals else 0.0
    return summary, details


def oracle_reorder(ranking: Sequence[str], qrels: dict[str, float], pool_k: int) -> list[str]:
    pool = list(ranking[: int(pool_k)])
    tail = list(ranking[int(pool_k) :])
    order = {doc_id: i for i, doc_id in enumerate(pool)}
    return sorted(pool, key=lambda doc_id: (float(qrels.get(doc_id, 0.0)), -order[doc_id]), reverse=True) + tail


def evaluate_bundle(
    rows: Sequence[dict[str, Any]],
    *,
    bundle: EvidenceSetRerankerBundle,
    method: str,
    corpus_texts: dict[str, str],
    ks: Sequence[int],
) -> tuple[dict[str, float], dict[str, Any]]:
    preds: list[list[str]] = []
    base_preds: list[list[str]] = []
    oracle_preds: list[list[str]] = []
    changed = 0
    top1_switched = 0
    for row in rows:
        base = ranking_for_row(row, method)
        base_preds.append(base)
        result = bundle.rerank(
            query_text=str(row.get("query", "")),
            query_id=str(row.get("query_id", "")),
            ranked_doc_ids=base,
            trace=row.get("trace", {}) or {},
            corpus_texts=corpus_texts,
        )
        preds.append(result.ranked_doc_ids)
        changed += int(result.changed_count > 0)
        top1_switched += int(result.switched_top1)
        oracle_preds.append(oracle_reorder(base, qrels_for_row(row), int(bundle.config.pool_k)))
    summary, detail = method_metrics(rows, preds, ks)
    base_summary, _ = method_metrics(rows, base_preds, ks)
    oracle_summary, _ = method_metrics(rows, oracle_preds, ks)
    meta = {
        "queries": int(len(rows)),
        "changed_queries": int(changed),
        "top1_switched_queries": int(top1_switched),
        "base": base_summary,
        "oracle_pool": oracle_summary,
        "detail": detail,
    }
    return summary, meta


def objective(metrics: dict[str, float]) -> float:
    return (
        0.34 * float(metrics.get("ndcg@1", 0.0))
        + 0.31 * float(metrics.get("ndcg@5", 0.0))
        + 0.22 * float(metrics.get("map@5", 0.0))
        + 0.13 * float(metrics.get("hits@5", 0.0)) / 5.0
    )


def make_model(args: argparse.Namespace):
    if args.model_type == "hist_gbdt":
        return HistGradientBoostingRegressor(
            max_iter=int(args.max_iter),
            learning_rate=float(args.learning_rate),
            max_leaf_nodes=int(args.max_leaf_nodes),
            l2_regularization=float(args.l2_regularization),
            random_state=int(args.seed),
        )
    if args.model_type == "random_forest":
        return RandomForestRegressor(
            n_estimators=int(args.n_estimators),
            min_samples_leaf=int(args.min_samples_leaf),
            max_features=float(args.max_features),
            n_jobs=int(args.n_jobs),
            random_state=int(args.seed),
        )
    if args.model_type == "ridge":
        return Pipeline([("scale", StandardScaler()), ("ridge", Ridge(alpha=float(args.ridge_alpha)))])
    raise ValueError(f"unknown model_type: {args.model_type}")


def parse_float_list(raw: str) -> list[float]:
    return [float(x.strip()) for x in str(raw).split(",") if x.strip()]


def parse_int_list(raw: str) -> list[int]:
    return [int(x.strip()) for x in str(raw).split(",") if x.strip()]


def main() -> int:
    parser = argparse.ArgumentParser(description="Train the supervised TESSERA evidence utility model over saved rankings.")
    parser.add_argument("--train-rankings-jsonl", type=Path, required=True)
    parser.add_argument("--dev-rankings-jsonl", type=Path, required=True)
    parser.add_argument("--corpus-json", type=Path, action="append", required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--method", type=str, default="tessera")
    parser.add_argument("--pool-k", type=int, default=10)
    parser.add_argument("--topk", type=int, default=5)
    parser.add_argument("--metrics-k", type=str, default="1,5")
    parser.add_argument("--model-type", choices=["hist_gbdt", "random_forest", "ridge"], default="hist_gbdt")
    parser.add_argument("--max-train", type=int, default=0)
    parser.add_argument("--max-dev", type=int, default=0)
    parser.add_argument("--positive-weight", type=float, default=9.0)
    parser.add_argument("--hard-negative-weight", type=float, default=2.0)
    parser.add_argument("--grade-power", type=float, default=0.80)
    parser.add_argument("--max-iter", type=int, default=220)
    parser.add_argument("--learning-rate", type=float, default=0.045)
    parser.add_argument("--max-leaf-nodes", type=int, default=31)
    parser.add_argument("--l2-regularization", type=float, default=0.05)
    parser.add_argument("--n-estimators", type=int, default=500)
    parser.add_argument("--min-samples-leaf", type=int, default=5)
    parser.add_argument("--max-features", type=float, default=0.75)
    parser.add_argument("--n-jobs", type=int, default=4)
    parser.add_argument("--ridge-alpha", type=float, default=3.0)
    parser.add_argument("--blend-grid", type=str, default="0,0.05,0.10,0.16,0.24,0.36")
    parser.add_argument("--margin-grid", type=str, default="0,0.01,0.02,0.04,0.08,0.14,999")
    parser.add_argument("--preserve-grid", type=str, default="0,1")
    parser.add_argument("--coverage-weight", type=float, default=0.18)
    parser.add_argument("--source-balance-weight", type=float, default=0.10)
    parser.add_argument("--anchor-weight", type=float, default=0.06)
    parser.add_argument("--redundancy-weight", type=float, default=0.14)
    parser.add_argument("--length-cost-weight", type=float, default=0.02)
    parser.add_argument("--min-gain", type=float, default=-1e9)
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--allow-test-split-training", action="store_true")
    args = parser.parse_args()

    if not args.allow_test_split_training and (
        is_test_like_path(args.train_rankings_jsonl) or is_test_like_path(args.dev_rankings_jsonl)
    ):
        raise ValueError("Refusing to train on a path that looks like test data. Pass --allow-test-split-training only if intentional.")

    ks = sorted({int(x.strip()) for x in str(args.metrics_k).split(",") if x.strip()})
    train_rows = iter_jsonl(args.train_rankings_jsonl, int(args.max_train))
    dev_rows = iter_jsonl(args.dev_rankings_jsonl, int(args.max_dev))
    needed = collect_needed_doc_ids(train_rows, method=args.method, pool_k=int(args.pool_k))
    needed.update(collect_needed_doc_ids(dev_rows, method=args.method, pool_k=int(args.pool_k)))
    corpus_texts = load_corpus_texts(args.corpus_json, needed)

    x_train, y_train, w_train, train_stats = make_dataset(
        train_rows,
        method=args.method,
        corpus_texts=corpus_texts,
        pool_k=int(args.pool_k),
        positive_weight=float(args.positive_weight),
        hard_negative_weight=float(args.hard_negative_weight),
        grade_power=float(args.grade_power),
    )
    x_dev, y_dev, w_dev, dev_stats = make_dataset(
        dev_rows,
        method=args.method,
        corpus_texts=corpus_texts,
        pool_k=int(args.pool_k),
        positive_weight=float(args.positive_weight),
        hard_negative_weight=float(args.hard_negative_weight),
        grade_power=float(args.grade_power),
    )
    if x_train.shape[0] <= 0:
        raise ValueError("No training examples were created.")

    model = make_model(args)
    try:
        model.fit(x_train, y_train, sample_weight=w_train)
    except TypeError:
        model.fit(x_train, y_train)

    base_config = EvidenceSetRerankerConfig(
        pool_k=int(args.pool_k),
        topk=int(args.topk),
        coverage_weight=float(args.coverage_weight),
        source_balance_weight=float(args.source_balance_weight),
        anchor_weight=float(args.anchor_weight),
        redundancy_weight=float(args.redundancy_weight),
        length_cost_weight=float(args.length_cost_weight),
        min_gain=float(args.min_gain),
    )
    raw_bundle = EvidenceSetRerankerBundle(
        model=model,
        feature_names=list(ESR_FEATURE_NAMES),
        config=base_config,
        metadata={
            "method_name": "TESSERA",
            "method_formulation": (
                "learn s(q,e)=f_theta(phi(q,e,C)) from graded train/dev qrels, then greedily "
                "assemble a unified text/table/KG evidence set with utility, coverage, source "
                "complementarity, anchor preservation, redundancy, and length-cost terms."
            ),
            "dataset_id_features": False,
            "train_rankings_jsonl": str(args.train_rankings_jsonl),
            "dev_rankings_jsonl": str(args.dev_rankings_jsonl),
            "corpus_json": [str(x) for x in args.corpus_json],
            "method": str(args.method),
            "pool_k": int(args.pool_k),
            "topk": int(args.topk),
            "model_type": str(args.model_type),
            "positive_weight": float(args.positive_weight),
            "hard_negative_weight": float(args.hard_negative_weight),
            "grade_power": float(args.grade_power),
            "objective_weights": {
                "coverage_weight": float(args.coverage_weight),
                "source_balance_weight": float(args.source_balance_weight),
                "anchor_weight": float(args.anchor_weight),
                "redundancy_weight": float(args.redundancy_weight),
                "length_cost_weight": float(args.length_cost_weight),
                "min_gain": float(args.min_gain),
            },
        },
    )

    grid_results: list[dict[str, Any]] = []
    best_bundle = raw_bundle
    best_dev_summary: dict[str, float] | None = None
    best_dev_meta: dict[str, Any] | None = None
    best_score = -1e9
    for preserve in parse_int_list(args.preserve_grid):
        for blend in parse_float_list(args.blend_grid):
            for margin in parse_float_list(args.margin_grid):
                bundle = raw_bundle.with_config(
                    preserve_top=int(preserve),
                    blend_original_weight=float(blend),
                    top1_switch_margin=float(margin),
                )
                dev_summary, dev_meta = evaluate_bundle(
                    dev_rows,
                    bundle=bundle,
                    method=str(args.method),
                    corpus_texts=corpus_texts,
                    ks=ks,
                )
                score = objective(dev_summary)
                grid_results.append(
                    {
                        "preserve_top": int(preserve),
                        "blend_original_weight": float(blend),
                        "top1_switch_margin": float(margin),
                        "objective": float(score),
                        "metrics": dev_summary,
                        "changed_queries": int(dev_meta.get("changed_queries", 0)),
                        "top1_switched_queries": int(dev_meta.get("top1_switched_queries", 0)),
                    }
                )
                if score > best_score:
                    best_score = score
                    best_bundle = bundle
                    best_dev_summary = dev_summary
                    best_dev_meta = dev_meta

    train_summary, train_meta = evaluate_bundle(
        train_rows,
        bundle=best_bundle,
        method=str(args.method),
        corpus_texts=corpus_texts,
        ks=ks,
    )
    best_bundle.metadata.update(
        {
            "train_stats": train_stats,
            "dev_stats": dev_stats,
            "dev_grid": sorted(grid_results, key=lambda x: x["objective"], reverse=True)[:20],
            "best_dev_metrics": best_dev_summary,
            "best_dev_meta": {
                k: v
                for k, v in (best_dev_meta or {}).items()
                if k not in {"detail"}
            },
            "best_train_metrics": train_summary,
            "best_train_meta": {k: v for k, v in train_meta.items() if k not in {"detail"}},
            "train_target_summary": {
                "y_mean": float(np.mean(y_train)) if y_train.size else 0.0,
                "y_positive_rate": float(np.mean(y_train > 0.0)) if y_train.size else 0.0,
                "weight_mean": float(np.mean(w_train)) if w_train.size else 0.0,
                "dev_y_mean": float(np.mean(y_dev)) if y_dev.size else 0.0,
                "dev_y_positive_rate": float(np.mean(y_dev > 0.0)) if y_dev.size else 0.0,
            },
        }
    )

    args.out_dir.mkdir(parents=True, exist_ok=True)
    model_path = args.out_dir / "evidence_set_reranker.pkl"
    save_evidence_set_reranker_bundle(best_bundle, model_path)
    report = {
        "model_path": str(model_path),
        "config": best_bundle.config.__dict__,
        "train_stats": train_stats,
        "dev_stats": dev_stats,
        "best_train_metrics": train_summary,
        "best_dev_metrics": best_dev_summary,
        "best_dev_meta": {k: v for k, v in (best_dev_meta or {}).items() if k not in {"detail"}},
        "dev_grid_top10": sorted(grid_results, key=lambda x: x["objective"], reverse=True)[:10],
        "feature_names": list(ESR_FEATURE_NAMES),
        "corpus_loaded": int(len(corpus_texts)),
        "needed_doc_ids": int(len(needed)),
    }
    (args.out_dir / "evidence_set_reranker_train_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"[OK] ESR model -> {model_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
