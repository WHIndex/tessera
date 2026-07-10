#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
import string
import time
from collections import Counter
from pathlib import Path

import numpy as np


def normalize_answer(s: str) -> str:
    s = s.lower()
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    s = "".join(ch for ch in s if ch not in set(string.punctuation))
    s = " ".join(s.split())
    return s


def f1_score(pred: str, gold: str) -> float:
    pred_toks = normalize_answer(pred).split()
    gold_toks = normalize_answer(gold).split()
    if not pred_toks and not gold_toks:
        return 1.0
    if not pred_toks or not gold_toks:
        return 0.0
    common = Counter(pred_toks) & Counter(gold_toks)
    num_same = sum(common.values())
    if num_same == 0:
        return 0.0
    precision = num_same / len(pred_toks)
    recall = num_same / len(gold_toks)
    return 2 * precision * recall / (precision + recall)


def exact_match(pred: str, gold: str) -> float:
    return 1.0 if normalize_answer(pred) == normalize_answer(gold) else 0.0


def load_gold(path: Path) -> dict[str, str]:
    rows = json.loads(path.read_text(encoding="utf-8"))
    out: dict[str, str] = {}
    for row in rows:
        qid = row.get("id")
        if qid is None:
            continue
        out[str(qid)] = str(row.get("answer", ""))
    return out


def load_preds(path: Path) -> dict[str, str]:
    out: dict[str, str] = {}
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            qid = row.get("id")
            if qid is None:
                continue
            pred = row.get("prediction", row.get("pred", row.get("answer", "")))
            out[str(qid)] = str(pred)
    return out


def _format_duration(seconds: float) -> str:
    seconds = max(0, int(seconds))
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    if h > 0:
        return f"{h:02d}:{m:02d}:{s:02d}"
    return f"{m:02d}:{s:02d}"


def bootstrap_delta(
    a_vals: np.ndarray,
    b_vals: np.ndarray,
    n_bootstrap: int,
    seed: int,
    metric_name: str,
    progress_every: int,
    progress_min_seconds: float,
) -> tuple[float, float, float, float]:
    rng = np.random.RandomState(seed)
    n = len(a_vals)
    deltas = np.zeros(n_bootstrap, dtype=np.float64)
    started_at = time.monotonic()
    last_logged_at = started_at
    for i in range(n_bootstrap):
        idx = rng.randint(0, n, size=n)
        deltas[i] = float(np.mean(b_vals[idx] - a_vals[idx]))

        done = i + 1
        now = time.monotonic()
        should_log = done == n_bootstrap
        if progress_every > 0 and done % progress_every == 0:
            should_log = True
        if progress_min_seconds > 0 and (now - last_logged_at) >= progress_min_seconds:
            should_log = True
        if should_log:
            elapsed = max(now - started_at, 1e-9)
            avg = elapsed / done
            eta = avg * (n_bootstrap - done)
            qps = done / elapsed
            pct = 100.0 * done / max(1, n_bootstrap)
            print(
                "[bootstrap-progress] "
                f"metric={metric_name} "
                f"{done}/{n_bootstrap} ({pct:.1f}%) "
                f"elapsed={_format_duration(elapsed)} "
                f"avg={avg:.6f}s/iter "
                f"eta={_format_duration(eta)} "
                f"iter_per_sec={qps:.2f}",
                flush=True,
            )
            last_logged_at = now

    delta = float(np.mean(b_vals - a_vals))
    ci_low = float(np.percentile(deltas, 2.5))
    ci_high = float(np.percentile(deltas, 97.5))
    p_value = float(2.0 * min(np.mean(deltas <= 0), np.mean(deltas >= 0)))
    p_value = min(p_value, 1.0)
    return delta, ci_low, ci_high, p_value


def main() -> int:
    parser = argparse.ArgumentParser(description="Paired bootstrap significance for E2E QA predictions")
    parser.add_argument("--gold-file", type=Path, required=True)
    parser.add_argument("--a-pred-file", type=Path, required=True, help="Baseline prediction jsonl")
    parser.add_argument("--b-pred-file", type=Path, required=True, help="Compared method prediction jsonl")
    parser.add_argument("--a-name", type=str, default="Dense-Concat")
    parser.add_argument("--b-name", type=str, default="TESSERA-RAG (Ours)")
    parser.add_argument("--n-bootstrap", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=20260327)
    parser.add_argument(
        "--progress-every",
        type=int,
        default=500,
        help="Print progress every N bootstrap iterations (0 to disable)",
    )
    parser.add_argument(
        "--progress-min-seconds",
        type=float,
        default=30.0,
        help="Print progress heartbeat every S seconds (0 to disable)",
    )
    parser.add_argument("--out-json", type=Path, required=True)
    parser.add_argument("--out-md", type=Path, required=True)
    args = parser.parse_args()

    gold = load_gold(args.gold_file)
    a_pred = load_preds(args.a_pred_file)
    b_pred = load_preds(args.b_pred_file)

    ids = sorted(set(gold.keys()) & set(a_pred.keys()) & set(b_pred.keys()))
    if not ids:
        raise SystemExit("No overlapping ids among gold/a_pred/b_pred")

    a_em, b_em, a_f1, b_f1 = [], [], [], []
    for qid in ids:
        g = gold[qid]
        pa = a_pred[qid]
        pb = b_pred[qid]
        a_em.append(exact_match(pa, g))
        b_em.append(exact_match(pb, g))
        a_f1.append(f1_score(pa, g))
        b_f1.append(f1_score(pb, g))

    a_em = np.asarray(a_em, dtype=np.float64)
    b_em = np.asarray(b_em, dtype=np.float64)
    a_f1 = np.asarray(a_f1, dtype=np.float64)
    b_f1 = np.asarray(b_f1, dtype=np.float64)

    f1_delta, f1_lo, f1_hi, f1_p = bootstrap_delta(
        a_f1,
        b_f1,
        args.n_bootstrap,
        args.seed,
        metric_name="f1",
        progress_every=args.progress_every,
        progress_min_seconds=args.progress_min_seconds,
    )
    em_delta, em_lo, em_hi, em_p = bootstrap_delta(
        a_em,
        b_em,
        args.n_bootstrap,
        args.seed + 1,
        metric_name="exact_match",
        progress_every=args.progress_every,
        progress_min_seconds=args.progress_min_seconds,
    )

    out = {
        "meta": {
            "evaluated": len(ids),
            "gold_total": len(gold),
            "a_pred_total": len(a_pred),
            "b_pred_total": len(b_pred),
            "coverage": float(len(ids) / max(1, len(gold))),
            "a_name": args.a_name,
            "b_name": args.b_name,
            "n_bootstrap": args.n_bootstrap,
            "seed": args.seed,
            "progress_every": args.progress_every,
            "progress_min_seconds": args.progress_min_seconds,
            "hypothesis": "paired bootstrap on per-query EM/F1, delta = b - a",
        },
        "rows": [
            {
                "metric": "f1",
                "a_mean": float(np.mean(a_f1)),
                "b_mean": float(np.mean(b_f1)),
                "delta": f1_delta,
                "delta_ci95_low": f1_lo,
                "delta_ci95_high": f1_hi,
                "p_value": f1_p,
                "wins": int(np.sum(b_f1 > a_f1)),
                "ties": int(np.sum(b_f1 == a_f1)),
                "losses": int(np.sum(b_f1 < a_f1)),
            },
            {
                "metric": "exact_match",
                "a_mean": float(np.mean(a_em)),
                "b_mean": float(np.mean(b_em)),
                "delta": em_delta,
                "delta_ci95_low": em_lo,
                "delta_ci95_high": em_hi,
                "p_value": em_p,
                "wins": int(np.sum(b_em > a_em)),
                "ties": int(np.sum(b_em == a_em)),
                "losses": int(np.sum(b_em < a_em)),
            },
        ],
    }

    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")

    md = [
        "# E2E Paired Bootstrap Significance",
        "",
        f"- evaluated: {out['meta']['evaluated']}",
        f"- coverage: {out['meta']['coverage']:.4f}",
        f"- compare: {args.b_name} - {args.a_name}",
        f"- n_bootstrap: {args.n_bootstrap}",
        "",
        "| Metric | A Mean | B Mean | Delta (B-A) | CI95 Low | CI95 High | p-value | Wins | Ties | Losses |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in out["rows"]:
        md.append(
            f"| {row['metric']} | {row['a_mean']:.4f} | {row['b_mean']:.4f} | {row['delta']:.4f} | "
            f"{row['delta_ci95_low']:.4f} | {row['delta_ci95_high']:.4f} | {row['p_value']:.4f} | "
            f"{row['wins']} | {row['ties']} | {row['losses']} |"
        )

    args.out_md.parent.mkdir(parents=True, exist_ok=True)
    args.out_md.write_text("\n".join(md) + "\n", encoding="utf-8")

    print(f"[OK] json -> {args.out_json}")
    print(f"[OK] markdown -> {args.out_md}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
