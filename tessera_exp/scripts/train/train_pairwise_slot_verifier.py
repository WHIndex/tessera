#!/usr/bin/env python3
from __future__ import annotations

import argparse
from collections import Counter
import json
import math
from pathlib import Path
import re
import sys

import numpy as np
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, average_precision_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from tessera_exp.e2e.pairwise_slot_verifier import PESV_FEATURE_NAMES, PairwiseSlotVerifierBundle, save_pairwise_slot_verifier_bundle  # noqa: E402


TOKEN_RE = re.compile(r"[a-z0-9]+")
STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "by", "did", "do", "does",
    "for", "from", "how", "in", "is", "it", "many", "much", "of", "on",
    "or", "the", "that", "this", "to", "was", "were", "what", "when",
    "where", "which", "who", "whom", "whose", "will", "with",
}


def fam(x: str | None) -> str:
    raw = str(x or "")
    if raw.startswith(("m.", "g.")):
        return "m"
    return raw.split("_", 1)[0].lower()


def stem(x: str | None) -> str:
    raw = str(x or "")
    if "_" not in raw:
        return raw
    head, tail = raw.rsplit("_", 1)
    return head if tail.isdigit() else raw


def bucket(doc_id: str) -> str:
    raw = str(doc_id or "")
    if raw.startswith(("m.", "/m/", "g.")):
        return "kg"
    p = raw.split("_", 1)[0].lower() if "_" in raw else raw.lower()
    if p in {"ott", "tat"}:
        return "table"
    if p in {"cwq", "webqsp", "kg", "wikidata", "wd"}:
        return "kg"
    return "text"


def target_bucket(query_id: str) -> str:
    qf = fam(query_id)
    if qf in {"cwq", "webqsp"}:
        return "kg"
    if qf in {"ott", "tat"}:
        return "table"
    return "text"


def content_tokens(text: str | None) -> set[str]:
    return {tok for tok in TOKEN_RE.findall(str(text or "").lower()) if len(tok) > 1 and tok not in STOPWORDS}


def token_jaccard(a: set[str], b: set[str]) -> float:
    if not a or not b:
        return 0.0
    return float(len(a & b) / max(1, len(a | b)))


def grade(qrels: dict, doc_id: str) -> float:
    try:
        return float(qrels.get(str(doc_id), 0.0))
    except Exception:
        return 0.0


def gain_at(grade_value: float, rank0: int) -> float:
    return (2.0 ** float(grade_value) - 1.0) / math.log2(float(rank0) + 2.0)


def parts(
    query_id: str,
    doc_id: str,
    rank0: int,
    ranking: list[str],
    candidate_start0: int,
    q_tokens: set[str] | None = None,
    doc_token_map: dict[str, set[str]] | None = None,
) -> dict[str, float]:
    qf = fam(query_id)
    b = bucket(doc_id)
    anchors = {stem(x) for x in ranking[: min(3, len(ranking))]}
    anchors.add(str(query_id or ""))
    rank_rr = 1.0 / float(rank0 + 1)
    tail = 0.0
    if rank0 >= candidate_start0:
        tail = 1.0 - (rank0 - candidate_start0) / max(1, len(ranking) - candidate_start0 - 1)
    toks = (doc_token_map or {}).get(str(doc_id), set())
    lexical = float(len((q_tokens or set()) & toks) / max(1, len(q_tokens or set()))) if q_tokens else 0.0
    family_hit = 1.0 if ((qf in {"cwq", "webqsp"} and fam(doc_id) == "m") or fam(doc_id) == qf) else 0.0
    sibling = 1.0 if stem(doc_id) in anchors else 0.0
    return {
        "static": rank_rr,
        "reference": rank_rr,
        "dense": rank_rr,
        "tail": max(0.0, tail),
        "sibling": sibling,
        "source": 1.0 if b == target_bucket(query_id) else 0.0,
        "lexical": lexical,
        "family": family_hit,
        "redundancy": sibling,
        "rank_rr": rank_rr,
        "text": 1.0 if b == "text" else 0.0,
        "table": 1.0 if b == "table" else 0.0,
        "kg": 1.0 if b == "kg" else 0.0,
    }


def support(p: dict[str, float]) -> float:
    return float(sum(int(p[k] > 0.0) for k in ("reference", "dense", "tail", "sibling", "source", "family")))


def content_feature_values(
    *,
    doc_id: str,
    q_tokens: set[str],
    doc_token_map: dict[str, set[str]],
    anchor_doc_ids: list[str],
) -> dict[str, float]:
    toks = doc_token_map.get(str(doc_id), set())
    anchor_tokens: set[str] = set()
    for anchor in anchor_doc_ids:
        anchor_tokens.update(doc_token_map.get(str(anchor), set()))
    numeric_tokens = {tok for tok in toks if any(ch.isdigit() for ch in tok)}
    q_numeric_tokens = {tok for tok in q_tokens if any(ch.isdigit() for ch in tok)}
    query_overlap = len(toks & q_tokens) if q_tokens else 0
    anchor_overlap = len(toks & anchor_tokens) if anchor_tokens else 0
    query_anchor_terms = q_tokens & anchor_tokens if q_tokens and anchor_tokens else set()
    novel_query_terms = (toks & q_tokens) - query_anchor_terms if q_tokens else set()
    return {
        "query_jaccard": token_jaccard(toks, q_tokens),
        "query_coverage": float(query_overlap / max(1, len(q_tokens))) if q_tokens else 0.0,
        "query_overlap_count": float(query_overlap),
        "numeric_overlap": float(len(numeric_tokens & q_numeric_tokens) / max(1, len(q_numeric_tokens))) if q_numeric_tokens else 0.0,
        "anchor_jaccard": token_jaccard(toks, anchor_tokens),
        "anchor_overlap_count": float(anchor_overlap),
        "anchor_novelty": float(len(novel_query_terms) / max(1, len(q_tokens))) if q_tokens else 0.0,
        "len_log": float(math.log1p(len(toks))),
    }


def feature(
    query_id: str,
    query_text: str,
    old_doc: str,
    cand_doc: str,
    old_rank0: int,
    cand_rank0: int,
    ranking: list[str],
    candidate_start0: int,
    doc_token_map: dict[str, set[str]],
) -> list[float]:
    q_tokens = content_tokens(query_text)
    old = parts(query_id, old_doc, old_rank0, ranking, candidate_start0, q_tokens, doc_token_map)
    cand = parts(query_id, cand_doc, cand_rank0, ranking, candidate_start0, q_tokens, doc_token_map)
    qf = fam(query_id)
    flags = {f"qfam_{x}": 1.0 if qf == x else 0.0 for x in ("cwq", "nq", "ott", "tat", "triviaqa", "webqsp")}
    anchor_doc_ids = [d for d in ranking[: min(3, len(ranking))] if d not in {old_doc, cand_doc}]
    cand_content = content_feature_values(doc_id=cand_doc, q_tokens=q_tokens, doc_token_map=doc_token_map, anchor_doc_ids=anchor_doc_ids)
    old_content = content_feature_values(doc_id=old_doc, q_tokens=q_tokens, doc_token_map=doc_token_map, anchor_doc_ids=anchor_doc_ids)
    vals = {
        "candidate_score": cand["static"], "old_score": old["static"], "score_delta": cand["static"] - old["static"],
        "candidate_reference": cand["reference"], "old_reference": old["reference"], "reference_delta": cand["reference"] - old["reference"],
        "candidate_dense": cand["dense"], "old_dense": old["dense"], "dense_delta": cand["dense"] - old["dense"],
        "candidate_tail": cand["tail"], "old_tail": old["tail"], "tail_delta": cand["tail"] - old["tail"],
        "candidate_sibling": cand["sibling"], "old_sibling": old["sibling"], "sibling_delta": cand["sibling"] - old["sibling"],
        "candidate_source": cand["source"], "old_source": old["source"], "source_delta": cand["source"] - old["source"],
        "candidate_lexical": cand["lexical"], "old_lexical": old["lexical"], "lexical_delta": cand["lexical"] - old["lexical"],
        "candidate_family": cand["family"], "old_family": old["family"], "family_delta": cand["family"] - old["family"],
        "candidate_redundancy": cand["redundancy"], "old_redundancy": old["redundancy"], "redundancy_delta": cand["redundancy"] - old["redundancy"],
        "candidate_rank_rr": cand["rank_rr"], "old_rank_rr": old["rank_rr"], "rank_rr_delta": cand["rank_rr"] - old["rank_rr"],
        "old_slot_norm": float(old_rank0) / 4.0,
        "candidate_support_count": support(cand), "old_support_count": support(old), "support_delta": support(cand) - support(old),
        "candidate_bucket_text": cand["text"], "candidate_bucket_table": cand["table"], "candidate_bucket_kg": cand["kg"],
        "old_bucket_text": old["text"], "old_bucket_table": old["table"], "old_bucket_kg": old["kg"],
        "candidate_query_jaccard": cand_content["query_jaccard"], "old_query_jaccard": old_content["query_jaccard"], "query_jaccard_delta": cand_content["query_jaccard"] - old_content["query_jaccard"],
        "candidate_query_coverage": cand_content["query_coverage"], "old_query_coverage": old_content["query_coverage"], "query_coverage_delta": cand_content["query_coverage"] - old_content["query_coverage"],
        "candidate_query_overlap_count": cand_content["query_overlap_count"], "old_query_overlap_count": old_content["query_overlap_count"], "query_overlap_count_delta": cand_content["query_overlap_count"] - old_content["query_overlap_count"],
        "candidate_numeric_overlap": cand_content["numeric_overlap"], "old_numeric_overlap": old_content["numeric_overlap"], "numeric_overlap_delta": cand_content["numeric_overlap"] - old_content["numeric_overlap"],
        "candidate_anchor_jaccard": cand_content["anchor_jaccard"], "old_anchor_jaccard": old_content["anchor_jaccard"], "anchor_jaccard_delta": cand_content["anchor_jaccard"] - old_content["anchor_jaccard"],
        "candidate_anchor_overlap_count": cand_content["anchor_overlap_count"], "old_anchor_overlap_count": old_content["anchor_overlap_count"], "anchor_overlap_count_delta": cand_content["anchor_overlap_count"] - old_content["anchor_overlap_count"],
        "candidate_anchor_novelty": cand_content["anchor_novelty"], "old_anchor_novelty": old_content["anchor_novelty"], "anchor_novelty_delta": cand_content["anchor_novelty"] - old_content["anchor_novelty"],
        "candidate_len_log": cand_content["len_log"], "old_len_log": old_content["len_log"], "len_log_delta": cand_content["len_log"] - old_content["len_log"],
        **flags,
    }
    return [float(vals.get(name, 0.0)) for name in PESV_FEATURE_NAMES]


def collect_doc_ids(path: Path, cand_end: int) -> set[str]:
    needed: set[str] = set()
    cand_end0 = max(1, int(cand_end))
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            ranking = list((row.get("rankings") or {}).get("tessera_rag") or [])
            needed.update(str(x) for x in ranking[: max(cand_end0, 10)])
    return needed


def iter_json_array(path: Path):
    decoder = json.JSONDecoder()
    buffer = ""
    pos = 0
    eof = False
    with path.open("r", encoding="utf-8") as f:
        while True:
            if not eof and len(buffer) - pos < 1_048_576:
                chunk = f.read(1_048_576)
                if chunk:
                    buffer += chunk
                else:
                    eof = True
            while True:
                while pos < len(buffer) and buffer[pos] in " \t\r\n,[":
                    pos += 1
                if pos < len(buffer) and buffer[pos] == "]":
                    return
                if pos >= len(buffer):
                    break
                try:
                    obj, end = decoder.raw_decode(buffer, pos)
                except json.JSONDecodeError:
                    if eof:
                        raise
                    break
                yield obj
                pos = end
                if pos > 4_194_304:
                    buffer = buffer[pos:]
                    pos = 0
            if eof:
                break


def load_doc_token_map(corpus_file: Path | None, doc_ids: set[str]) -> dict[str, set[str]]:
    if corpus_file is None:
        return {}
    wanted = set(str(x) for x in doc_ids)
    out: dict[str, set[str]] = {}
    for obj in iter_json_array(corpus_file):
        if not isinstance(obj, dict):
            continue
        doc_id = str(obj.get("id", ""))
        if doc_id not in wanted:
            continue
        out[doc_id] = content_tokens(obj.get("text", ""))
        if len(out) >= len(wanted):
            break
    return out


def read_examples(path: Path, old_slot: int, cand_start: int, cand_end: int, min_gain: float, doc_token_map: dict[str, set[str]]) -> tuple[np.ndarray, np.ndarray, list[dict]]:
    xs, ys, meta = [], [], []
    old_rank0 = max(0, int(old_slot) - 1)
    cand_start0 = max(0, int(cand_start) - 1)
    cand_end0 = max(cand_start0 + 1, int(cand_end))
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            qid = str(row.get("query_id", ""))
            query_text = str(row.get("query", ""))
            qrels = row.get("qrels") or {}
            ranking = list((row.get("rankings") or {}).get("tessera_rag") or [])
            if len(ranking) <= old_rank0:
                continue
            old_doc = str(ranking[old_rank0])
            old_gain = gain_at(grade(qrels, old_doc), old_rank0)
            for cand_rank0 in range(cand_start0, min(cand_end0, len(ranking))):
                cand_doc = str(ranking[cand_rank0])
                if cand_doc == old_doc:
                    continue
                cand_gain = gain_at(grade(qrels, cand_doc), old_rank0)
                y = int((cand_gain - old_gain) > float(min_gain))
                xs.append(feature(qid, query_text, old_doc, cand_doc, old_rank0, cand_rank0, ranking, cand_start0, doc_token_map))
                ys.append(y)
                meta.append({"query_id": qid, "family": fam(qid), "old_doc": old_doc, "candidate_doc": cand_doc, "label": y})
    return np.asarray(xs, dtype=np.float32), np.asarray(ys, dtype=np.int64), meta


def split_by_query(
    x: np.ndarray,
    y: np.ndarray,
    meta: list[dict],
    *,
    val_ratio: float,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, list[dict], np.ndarray, np.ndarray, list[dict]]:
    if len(meta) == 0:
        return x, y, meta, x[:0], y[:0], []
    query_ids = sorted({str(row.get("query_id", "")) for row in meta})
    rng = np.random.default_rng(int(seed))
    shuffled = list(query_ids)
    rng.shuffle(shuffled)
    val_count = max(1, int(round(len(shuffled) * max(0.01, min(0.80, float(val_ratio))))))
    val_queries = set(shuffled[:val_count])
    train_idx = [i for i, row in enumerate(meta) if str(row.get("query_id", "")) not in val_queries]
    val_idx = [i for i, row in enumerate(meta) if str(row.get("query_id", "")) in val_queries]
    if not train_idx or not val_idx:
        split = int(len(x) * (1.0 - max(0.01, min(0.80, float(val_ratio)))))
        train_idx = list(range(split))
        val_idx = list(range(split, len(x)))
    return (
        x[train_idx],
        y[train_idx],
        [meta[i] for i in train_idx],
        x[val_idx],
        y[val_idx],
        [meta[i] for i in val_idx],
    )


def fit(x: np.ndarray, y: np.ndarray):
    if x.size == 0 or len(set(int(v) for v in y.tolist())) < 2:
        model = DummyClassifier(strategy="most_frequent")
        model.fit(x if x.size else np.zeros((1, len(PESV_FEATURE_NAMES)), dtype=np.float32), y if y.size else np.asarray([0]))
        return model
    model = Pipeline([("scaler", StandardScaler()), ("clf", LogisticRegression(max_iter=1000, class_weight="balanced"))])
    model.fit(x, y)
    return model


def prob(model, x: np.ndarray) -> np.ndarray:
    if x.size == 0:
        return np.zeros((0,), dtype=np.float32)
    p = np.asarray(model.predict_proba(x), dtype=np.float32)
    if p.ndim == 2 and p.shape[1] >= 2:
        classes = np.asarray(getattr(model, "classes_", []))
        if classes.size == p.shape[1] and 1 in classes.tolist():
            return p[:, int(np.where(classes == 1)[0][0])]
        return p[:, -1]
    return p.reshape(-1)


def evaluate(model, x: np.ndarray, y: np.ndarray, threshold: float) -> dict:
    if x.size == 0:
        return {"examples": 0}
    pr = prob(model, x)
    pred = (pr >= threshold).astype(np.int64)
    out = {
        "examples": int(len(y)),
        "positive_rate": float(np.mean(y)),
        "predicted_positive_rate": float(np.mean(pred)),
        "accuracy": float(accuracy_score(y, pred)),
        "precision": float(precision_score(y, pred, zero_division=0)),
        "recall": float(recall_score(y, pred, zero_division=0)),
        "f1": float(f1_score(y, pred, zero_division=0)),
    }
    if len(set(int(v) for v in y.tolist())) >= 2:
        out["roc_auc"] = float(roc_auc_score(y, pr))
        out["avg_precision"] = float(average_precision_score(y, pr))
    else:
        out["roc_auc"] = 0.0
        out["avg_precision"] = 0.0
    return out


def evaluate_by_family(model, x: np.ndarray, y: np.ndarray, meta: list[dict], threshold: float) -> dict[str, dict]:
    if x.size == 0:
        return {}
    scores = prob(model, x)
    pred = (scores >= threshold).astype(np.int64)
    families = sorted({str(row.get("family") or fam(row.get("query_id"))) for row in meta})
    out: dict[str, dict] = {}
    for family in families:
        idx = [i for i, row in enumerate(meta) if str(row.get("family") or fam(row.get("query_id"))) == family]
        if not idx:
            continue
        yy = y[idx]
        pp = pred[idx]
        out[family] = {
            "examples": int(len(idx)),
            "positive_rate": float(np.mean(yy)) if len(yy) else 0.0,
            "predicted_positive_count": int(np.sum(pp)),
            "predicted_positive_rate": float(np.mean(pp)) if len(pp) else 0.0,
            "precision": float(precision_score(yy, pp, zero_division=0)),
            "recall": float(recall_score(yy, pp, zero_division=0)),
            "f1": float(f1_score(yy, pp, zero_division=0)),
            "true_positive": int(np.sum((yy == 1) & (pp == 1))),
            "false_positive": int(np.sum((yy == 0) & (pp == 1))),
        }
    return out


def choose_enabled_families(
    family_metrics: dict[str, dict],
    *,
    min_precision: float,
    min_predictions: int,
    override: str | None,
) -> list[str]:
    if override:
        raw = {x.strip().lower() for x in str(override).split(",") if x.strip()}
        if "all" in raw or "*" in raw:
            return sorted(family_metrics)
        return sorted(raw)
    enabled: list[str] = []
    for family, metrics in family_metrics.items():
        if int(metrics.get("predicted_positive_count", 0)) < int(min_predictions):
            continue
        if float(metrics.get("precision", 0.0)) < float(min_precision):
            continue
        if int(metrics.get("true_positive", 0)) <= int(metrics.get("false_positive", 0)):
            continue
        enabled.append(family)
    return sorted(enabled)


def tune(model, x: np.ndarray, y: np.ndarray, min_precision: float) -> tuple[float, dict]:
    best_t, best_m, best_s = 0.68, evaluate(model, x, y, 0.68), -1.0
    fallback_t, fallback_m, fallback_s = best_t, best_m, -1.0
    for t in np.linspace(0.50, 0.95, 46):
        m = evaluate(model, x, y, float(t))
        fallback_score = m.get("precision", 0.0) + 0.05 * m.get("f1", 0.0)
        if fallback_score > fallback_s:
            fallback_t, fallback_m, fallback_s = float(t), m, fallback_score
        if m.get("precision", 0.0) < min_precision:
            continue
        s = m.get("f1", 0.0) + 0.1 * m.get("precision", 0.0)
        if s > best_s:
            best_t, best_m, best_s = float(t), m, s
    if best_s < 0.0:
        return fallback_t, fallback_m
    return best_t, best_m


def main() -> int:
    ap = argparse.ArgumentParser(description="Train Pairwise Evidence Slot Verifier from rankings_debug.jsonl")
    ap.add_argument("--train-debug-jsonl", type=Path, required=True)
    ap.add_argument("--val-debug-jsonl", type=Path, default=None)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--run-id", default="tessera_pesv_v1")
    ap.add_argument("--old-slot", type=int, default=5)
    ap.add_argument("--candidate-start", type=int, default=6)
    ap.add_argument("--candidate-end", type=int, default=10)
    ap.add_argument("--min-gain", type=float, default=0.0)
    ap.add_argument("--min-precision", type=float, default=0.70)
    ap.add_argument("--corpus-file", type=Path, default=None)
    ap.add_argument("--min-family-precision", type=float, default=0.50)
    ap.add_argument("--min-family-predictions", type=int, default=2)
    ap.add_argument("--enable-families", default="")
    ap.add_argument("--val-ratio", type=float, default=0.20)
    ap.add_argument("--seed", type=int, default=13)
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    needed_doc_ids = collect_doc_ids(args.train_debug_jsonl, args.candidate_end)
    if args.val_debug_jsonl is not None:
        needed_doc_ids.update(collect_doc_ids(args.val_debug_jsonl, args.candidate_end))
    doc_token_map = load_doc_token_map(args.corpus_file, needed_doc_ids)
    print(
        f"[stage] content tokens loaded: {len(doc_token_map)}/{len(needed_doc_ids)} docs "
        f"from {args.corpus_file if args.corpus_file is not None else 'no corpus'}",
        file=sys.stderr,
        flush=True,
    )

    x_train, y_train, train_meta = read_examples(
        args.train_debug_jsonl,
        args.old_slot,
        args.candidate_start,
        args.candidate_end,
        args.min_gain,
        doc_token_map,
    )
    if args.val_debug_jsonl is not None:
        x_val, y_val, val_meta = read_examples(
            args.val_debug_jsonl,
            args.old_slot,
            args.candidate_start,
            args.candidate_end,
            args.min_gain,
            doc_token_map,
        )
    else:
        x_train, y_train, train_meta, x_val, y_val, val_meta = split_by_query(
            x_train,
            y_train,
            train_meta,
            val_ratio=float(args.val_ratio),
            seed=int(args.seed),
        )
    model = fit(x_train, y_train)
    threshold, val_metrics = tune(model, x_val, y_val, args.min_precision)
    train_metrics = evaluate(model, x_train, y_train, threshold)
    val_family_metrics = evaluate_by_family(model, x_val, y_val, val_meta, threshold)
    train_family_metrics = evaluate_by_family(model, x_train, y_train, train_meta, threshold)
    enabled_families = choose_enabled_families(
        val_family_metrics,
        min_precision=float(args.min_family_precision),
        min_predictions=int(args.min_family_predictions),
        override=str(args.enable_families or ""),
    )
    bundle = PairwiseSlotVerifierBundle(
        model=model,
        feature_names=list(PESV_FEATURE_NAMES),
        metadata={
            "run_id": args.run_id,
            "feature_version": 2,
            "recommended_threshold": threshold,
            "enabled_families": enabled_families,
            "family_gate_policy": {
                "min_family_precision": float(args.min_family_precision),
                "min_family_predictions": int(args.min_family_predictions),
                "override": str(args.enable_families or ""),
            },
            "corpus_file": str(args.corpus_file) if args.corpus_file is not None else None,
            "needed_doc_ids": int(len(needed_doc_ids)),
            "loaded_doc_tokens": int(len(doc_token_map)),
            "old_slot": int(args.old_slot),
            "candidate_start": int(args.candidate_start),
            "candidate_end": int(args.candidate_end),
            "train_examples": int(len(y_train)),
            "val_examples": int(len(y_val)),
            "val_ratio": float(args.val_ratio),
            "seed": int(args.seed),
            "train_label_distribution": dict(Counter(int(v) for v in y_train.tolist())),
            "val_label_distribution": dict(Counter(int(v) for v in y_val.tolist())),
            "train_meta_preview": train_meta[:3],
            "val_meta_preview": val_meta[:3],
        },
    )
    bundle_path = args.out_dir / "pairwise_slot_verifier.pkl"
    save_pairwise_slot_verifier_bundle(bundle, bundle_path)
    metrics = {
        "run_id": args.run_id,
        "bundle": str(bundle_path),
        "recommended_threshold": threshold,
        "enabled_families": enabled_families,
        "train": train_metrics,
        "val": val_metrics,
        "train_by_family": train_family_metrics,
        "val_by_family": val_family_metrics,
        "train_label_distribution": dict(Counter(int(v) for v in y_train.tolist())),
        "val_label_distribution": dict(Counter(int(v) for v in y_val.tolist())),
        "needed_doc_ids": int(len(needed_doc_ids)),
        "loaded_doc_tokens": int(len(doc_token_map)),
    }
    metrics_path = args.out_dir / f"{args.run_id}_metrics.json"
    metrics_path.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(metrics, ensure_ascii=False, indent=2))
    print(f"[OK] bundle -> {bundle_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
