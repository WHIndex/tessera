#!/usr/bin/env python3
"""
Paired bootstrap resampling for comparing two methods on per-query metrics.
Usage:
    python paired_bootstrap.py <preds_a.jsonl> <preds_b.jsonl> --name-a MethodA --name-b MethodB --out-dir ./bootstrap_out
"""
from __future__ import annotations

import argparse
import json
import random
import statistics
import sys
import time
from pathlib import Path


def load_preds(path: str) -> list[dict]:
    preds = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            preds.append(json.loads(line))
    return preds


def f1_score(pred: str, gold: str) -> float:
    """Lightweight F1 based on token overlap (same as mmRAG metric)."""
    p_toks = set(str(pred).lower().split())
    g_toks = set(str(gold).lower().split())
    if not p_toks or not g_toks:
        return 0.0
    inter = len(p_toks & g_toks)
    p = inter / len(p_toks)
    r = inter / len(g_toks)
    if p + r == 0:
        return 0.0
    return 2 * p * r / (p + r)


def exact_match(pred: str, gold: str) -> float:
    return 1.0 if str(pred).strip().lower() == str(gold).strip().lower() else 0.0


def paired_bootstrap(
    a_vals: list[float],
    b_vals: list[float],
    n_bootstrap: int = 10000,
    seed: int = 42,
    progress_every: int = 2000,
) -> dict[str, float]:
    rng = random.Random(seed)
    n = len(a_vals)
    assert n == len(b_vals)
    deltas = [b - a for a, b in zip(a_vals, b_vals)]
    observed = statistics.mean(deltas)

    boot_deltas: list[float] = []
    t0 = time.perf_counter()
    for i in range(n_bootstrap):
        idxs = [rng.randrange(n) for _ in range(n)]
        boot_deltas.append(statistics.mean(deltas[j] for j in idxs))
        if progress_every > 0 and (i + 1) % progress_every == 0:
            elapsed = time.perf_counter() - t0
            eta = elapsed / (i + 1) * (n_bootstrap - i - 1)
            print(
                f"[bootstrap-progress] {i+1}/{n_bootstrap} ({100*(i+1)/n_bootstrap:.1f}%) "
                f"elapsed={elapsed:.1f}s eta={eta:.1f}s",
                file=sys.stderr,
            )

    boot_deltas.sort()
    ci_low = boot_deltas[int(0.025 * n_bootstrap)]
    ci_high = boot_deltas[int(0.975 * n_bootstrap)]

    # Two-sided p-value: proportion of bootstrapped deltas with opposite sign or zero
    if observed >= 0:
        p_value = sum(1 for d in boot_deltas if d <= 0) / n_bootstrap
    else:
        p_value = sum(1 for d in boot_deltas if d >= 0) / n_bootstrap

    return {
        "observed_delta": observed,
        "ci95_low": ci_low,
        "ci95_high": ci_high,
        "p_value": p_value,
        "n": n,
        "n_bootstrap": n_bootstrap,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("preds_a", help="JSONL file with predictions for method A")
    parser.add_argument("preds_b", help="JSONL file with predictions for method B")
    parser.add_argument("--name-a", default="A")
    parser.add_argument("--name-b", default="B")
    parser.add_argument("--out-dir", default=".")
    parser.add_argument("--n-bootstrap", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--progress-every", type=int, default=2000)
    args = parser.parse_args()

    a_preds = load_preds(args.preds_a)
    b_preds = load_preds(args.preds_b)
    n = min(len(a_preds), len(b_preds))
    a_preds = a_preds[:n]
    b_preds = b_preds[:n]

    a_f1 = [f1_score(p.get("prediction", ""), p.get("gold", "")) for p in a_preds]
    b_f1 = [f1_score(p.get("prediction", ""), p.get("gold", "")) for p in b_preds]
    a_em = [exact_match(p.get("prediction", ""), p.get("gold", "")) for p in a_preds]
    b_em = [exact_match(p.get("prediction", ""), p.get("gold", "")) for p in b_preds]

    print(f"Comparing {args.name_b} vs {args.name_a} on {n} queries")
    print(f"  {args.name_a}: F1={statistics.mean(a_f1):.4f} EM={statistics.mean(a_em):.4f}")
    print(f"  {args.name_b}: F1={statistics.mean(b_f1):.4f} EM={statistics.mean(b_em):.4f}")

    print(f"\nBootstrap F1 (n={args.n_bootstrap})...")
    f1_res = paired_bootstrap(a_f1, b_f1, args.n_bootstrap, args.seed, args.progress_every)
    print(f"  delta={f1_res['observed_delta']:+.4f}  CI95=[{f1_res['ci95_low']:+.4f}, {f1_res['ci95_high']:+.4f}]  p={f1_res['p_value']:.4f}")

    print(f"\nBootstrap EM (n={args.n_bootstrap})...")
    em_res = paired_bootstrap(a_em, b_em, args.n_bootstrap, args.seed, args.progress_every)
    print(f"  delta={em_res['observed_delta']:+.4f}  CI95=[{em_res['ci95_low']:+.4f}, {em_res['ci95_high']:+.4f}]  p={em_res['p_value']:.4f}")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / "paired_bootstrap.json"
    with open(out_file, "w") as f:
        json.dump(
            {
                "meta": {
                    "name_a": args.name_a,
                    "name_b": args.name_b,
                    "n_queries": n,
                    "n_bootstrap": args.n_bootstrap,
                    "seed": args.seed,
                },
                "f1": f1_res,
                "exact_match": em_res,
            },
            f,
            indent=2,
        )
    print(f"\nSaved -> {out_file}")


if __name__ == "__main__":
    main()
