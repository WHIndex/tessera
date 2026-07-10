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
    hours, remainder = divmod(seconds, 3600)
    minutes, secs = divmod(remainder, 60)
    if hours > 0:
        return f"{hours:02d}:{minutes:02d}:{secs:02d}"
    return f"{minutes:02d}:{secs:02d}"


def _log_progress(
    *,
    label: str,
    done: int,
    total: int,
    started_at: float,
    last_logged_at: float,
) -> float:
    now = time.monotonic()
    elapsed = max(now - started_at, 1e-9)
    avg = elapsed / max(1, done)
    eta = avg * max(0, total - done)
    qps = done / elapsed
    pct = 100.0 * done / max(1, total)
    print(
        "[bootstrap-progress] "
        f"{label} {done}/{total} ({pct:.1f}%) "
        f"elapsed={_format_duration(elapsed)} "
        f"avg={avg:.6f}s/iter "
        f"eta={_format_duration(eta)} "
        f"iter_per_sec={qps:.2f}",
        flush=True,
    )
    return now


def paired_bootstrap_delta(
    a_vals: np.ndarray,
    b_vals: np.ndarray,
    n_bootstrap: int,
    seed: int,
    *,
    label: str,
    progress_every: int,
    progress_min_seconds: float,
) -> tuple[float, float, float, float]:
    rng = np.random.default_rng(seed)
    n = len(a_vals)
    if n_bootstrap <= 0:
        obs = float(np.mean(b_vals - a_vals))
        return obs, obs, obs, 1.0

    block_size = max(1, progress_every) if progress_every > 0 else n_bootstrap
    deltas = np.empty(n_bootstrap, dtype=np.float64)
    started_at = time.monotonic()
    last_logged_at = started_at
    for start in range(0, n_bootstrap, block_size):
        end = min(n_bootstrap, start + block_size)
        idx = rng.integers(0, n, size=(end - start, n))
        deltas[start:end] = (b_vals[idx] - a_vals[idx]).mean(axis=1)
        done = end
        now = time.monotonic()
        if done == n_bootstrap:
            last_logged_at = _log_progress(
                label=label,
                done=done,
                total=n_bootstrap,
                started_at=started_at,
                last_logged_at=last_logged_at,
            )
        elif progress_every > 0 and done % progress_every == 0:
            last_logged_at = _log_progress(
                label=label,
                done=done,
                total=n_bootstrap,
                started_at=started_at,
                last_logged_at=last_logged_at,
            )
        elif progress_min_seconds > 0 and (now - last_logged_at) >= progress_min_seconds:
            last_logged_at = _log_progress(
                label=label,
                done=done,
                total=n_bootstrap,
                started_at=started_at,
                last_logged_at=last_logged_at,
            )
    obs = float(np.mean(b_vals - a_vals))
    lo = float(np.percentile(deltas, 2.5))
    hi = float(np.percentile(deltas, 97.5))
    p = float(2.0 * min(float(np.mean(deltas <= 0.0)), float(np.mean(deltas >= 0.0))))
    p = min(max(p, 0.0), 1.0)
    return obs, lo, hi, p


def paired_cohen_d(a_vals: np.ndarray, b_vals: np.ndarray) -> float:
    diff = b_vals - a_vals
    if diff.size == 0:
        return 0.0
    mean_diff = float(np.mean(diff))
    std_diff = float(np.std(diff, ddof=1)) if diff.size > 1 else 0.0
    if std_diff <= 1e-12:
        return 0.0
    return mean_diff / std_diff


def build_metric_arrays(
    ids: list[str],
    gold: dict[str, str],
    pred_a: dict[str, str],
    pred_b: dict[str, str],
    metric: str,
) -> tuple[np.ndarray, np.ndarray]:
    a_vals = []
    b_vals = []
    for qid in ids:
        g = gold[qid]
        a = pred_a[qid]
        b = pred_b[qid]
        if metric == "f1":
            a_vals.append(f1_score(a, g))
            b_vals.append(f1_score(b, g))
        elif metric == "exact_match":
            a_vals.append(exact_match(a, g))
            b_vals.append(exact_match(b, g))
        else:
            raise ValueError(f"Unsupported metric: {metric}")
    return np.asarray(a_vals, dtype=np.float64), np.asarray(b_vals, dtype=np.float64)


def main() -> int:
    parser = argparse.ArgumentParser(description="SchemeB small-sample effect-size gate")
    parser.add_argument("--gold-file", type=Path, required=True)
    parser.add_argument("--baseline-pred-file", type=Path, required=True)
    parser.add_argument("--candidate-pred-file", type=Path, required=True)
    parser.add_argument("--baseline-name", type=str, default="baseline")
    parser.add_argument("--candidate-name", type=str, default="schemeb_heavy")
    parser.add_argument("--n-bootstrap", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=20260330)
    parser.add_argument("--min-delta-f1", type=float, default=0.020)
    parser.add_argument("--min-delta-em", type=float, default=0.010)
    parser.add_argument("--min-cohen-d-f1", type=float, default=0.35)
    parser.add_argument("--min-cohen-d-em", type=float, default=0.20)
    parser.add_argument("--max-p-value", type=float, default=0.05)
    parser.add_argument("--progress-every", type=int, default=500)
    parser.add_argument("--progress-min-seconds", type=float, default=30.0)
    parser.add_argument("--out-json", type=Path, required=True)
    parser.add_argument("--out-md", type=Path, default=None)
    args = parser.parse_args()

    gold = load_gold(args.gold_file)
    pred_a = load_preds(args.baseline_pred_file)
    pred_b = load_preds(args.candidate_pred_file)
    ids = sorted(set(gold.keys()) & set(pred_a.keys()) & set(pred_b.keys()))
    if not ids:
        raise SystemExit("No overlapping ids among gold/baseline/candidate predictions")

    a_f1, b_f1 = build_metric_arrays(ids, gold, pred_a, pred_b, metric="f1")
    a_em, b_em = build_metric_arrays(ids, gold, pred_a, pred_b, metric="exact_match")

    f1_delta, f1_lo, f1_hi, f1_p = paired_bootstrap_delta(
        a_f1,
        b_f1,
        args.n_bootstrap,
        args.seed,
        label="metric=f1",
        progress_every=args.progress_every,
        progress_min_seconds=args.progress_min_seconds,
    )
    em_delta, em_lo, em_hi, em_p = paired_bootstrap_delta(
        a_em,
        b_em,
        args.n_bootstrap,
        args.seed + 1,
        label="metric=exact_match",
        progress_every=args.progress_every,
        progress_min_seconds=args.progress_min_seconds,
    )
    f1_d = paired_cohen_d(a_f1, b_f1)
    em_d = paired_cohen_d(a_em, b_em)

    checks = {
        "f1_delta": f1_delta >= float(args.min_delta_f1),
        "em_delta": em_delta >= float(args.min_delta_em),
        "f1_cohen_d": f1_d >= float(args.min_cohen_d_f1),
        "em_cohen_d": em_d >= float(args.min_cohen_d_em),
        "f1_p_value": f1_p <= float(args.max_p_value),
        "em_p_value": em_p <= float(args.max_p_value),
    }
    gate_passed = all(checks.values())

    result = {
        "meta": {
            "evaluated": len(ids),
            "gold_total": len(gold),
            "baseline_total": len(pred_a),
            "candidate_total": len(pred_b),
            "coverage": float(len(ids) / max(1, len(gold))),
            "baseline_name": args.baseline_name,
            "candidate_name": args.candidate_name,
            "n_bootstrap": int(args.n_bootstrap),
            "seed": int(args.seed),
            "progress_every": int(args.progress_every),
            "progress_min_seconds": float(args.progress_min_seconds),
        },
        "thresholds": {
            "min_delta_f1": float(args.min_delta_f1),
            "min_delta_em": float(args.min_delta_em),
            "min_cohen_d_f1": float(args.min_cohen_d_f1),
            "min_cohen_d_em": float(args.min_cohen_d_em),
            "max_p_value": float(args.max_p_value),
        },
        "metrics": {
            "f1": {
                "baseline_mean": float(np.mean(a_f1)),
                "candidate_mean": float(np.mean(b_f1)),
                "delta": float(f1_delta),
                "delta_ci95_low": float(f1_lo),
                "delta_ci95_high": float(f1_hi),
                "p_value": float(f1_p),
                "cohen_d": float(f1_d),
                "wins": int(np.sum(b_f1 > a_f1)),
                "ties": int(np.sum(b_f1 == a_f1)),
                "losses": int(np.sum(b_f1 < a_f1)),
            },
            "exact_match": {
                "baseline_mean": float(np.mean(a_em)),
                "candidate_mean": float(np.mean(b_em)),
                "delta": float(em_delta),
                "delta_ci95_low": float(em_lo),
                "delta_ci95_high": float(em_hi),
                "p_value": float(em_p),
                "cohen_d": float(em_d),
                "wins": int(np.sum(b_em > a_em)),
                "ties": int(np.sum(b_em == a_em)),
                "losses": int(np.sum(b_em < a_em)),
            },
        },
        "checks": checks,
        "final_gate": {
            "passed": gate_passed,
            "reason": "all thresholds passed" if gate_passed else "at least one threshold failed",
        },
    }

    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")

    if args.out_md is not None:
        md = [
            "# SchemeB Effect-Size Gate",
            "",
            f"- evaluated: {result['meta']['evaluated']}",
            f"- coverage: {result['meta']['coverage']:.4f}",
            f"- baseline: {result['meta']['baseline_name']}",
            f"- candidate: {result['meta']['candidate_name']}",
            f"- gate_passed: {result['final_gate']['passed']}",
            "",
            "| Metric | Baseline | Candidate | Delta | CI95 Low | CI95 High | p-value | Cohen d | Wins | Ties | Losses |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
        for key in ("f1", "exact_match"):
            row = result["metrics"][key]
            md.append(
                f"| {key} | {row['baseline_mean']:.4f} | {row['candidate_mean']:.4f} | {row['delta']:.4f} | "
                f"{row['delta_ci95_low']:.4f} | {row['delta_ci95_high']:.4f} | {row['p_value']:.4f} | "
                f"{row['cohen_d']:.4f} | {row['wins']} | {row['ties']} | {row['losses']} |"
            )
        md.extend([
            "",
            "## Gate Checks",
            "",
        ])
        for k, v in checks.items():
            md.append(f"- {k}: {v}")

        args.out_md.parent.mkdir(parents=True, exist_ok=True)
        args.out_md.write_text("\n".join(md) + "\n", encoding="utf-8")

    print(json.dumps(result["final_gate"], ensure_ascii=False))
    print(f"[OK] json -> {args.out_json}")
    if args.out_md is not None:
        print(f"[OK] markdown -> {args.out_md}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())