#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def bootstrap_mean_ci(values: np.ndarray, n_bootstrap: int, seed: int):
    rng = np.random.default_rng(seed)
    n = len(values)
    if n == 0:
        return 0.0, 0.0, 0.0
    idx = rng.integers(0, n, size=(n_bootstrap, n))
    means = values[idx].mean(axis=1)
    return float(values.mean()), float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def bootstrap_delta_ci(a: np.ndarray, b: np.ndarray, n_bootstrap: int, seed: int, paired: bool):
    rng = np.random.default_rng(seed)
    if len(a) == 0 or len(b) == 0:
        return 0.0, 0.0, 0.0, 1.0

    if paired:
        n = min(len(a), len(b))
        a = a[:n]
        b = b[:n]
        idx = rng.integers(0, n, size=(n_bootstrap, n))
        deltas = (a[idx] - b[idx]).mean(axis=1)
        obs = float((a - b).mean())
    else:
        na, nb = len(a), len(b)
        ia = rng.integers(0, na, size=(n_bootstrap, na))
        ib = rng.integers(0, nb, size=(n_bootstrap, nb))
        deltas = a[ia].mean(axis=1) - b[ib].mean(axis=1)
        obs = float(a.mean() - b.mean())

    p = float(2 * min((deltas <= 0).mean(), (deltas >= 0).mean()))
    p = min(max(p, 0.0), 1.0)
    return obs, float(np.percentile(deltas, 2.5)), float(np.percentile(deltas, 97.5)), p


def maybe_array(d: dict, key: str):
    x = d.get(key)
    if x is None:
        return None
    return np.asarray(x, dtype=np.float32)


def maybe_metric_array(d: dict, keys: list[str]):
    for key in keys:
        arr = maybe_array(d, key)
        if arr is not None:
            return arr
    return None


def add_mean_row(rows: list[dict], block: str, method: str, metric: str, arr: np.ndarray, n_bootstrap: int, seed: int):
    mean, ci_l, ci_u = bootstrap_mean_ci(arr, n_bootstrap=n_bootstrap, seed=seed)
    rows.append(
        {
            "block": block,
            "type": "mean_ci",
            "method": method,
            "metric": metric,
            "value": mean,
            "ci95_low": ci_l,
            "ci95_high": ci_u,
            "delta": None,
            "delta_ci95_low": None,
            "delta_ci95_high": None,
            "p_value": None,
        }
    )


def add_delta_row(
    rows: list[dict],
    block: str,
    method_a: str,
    method_b: str,
    metric: str,
    arr_a: np.ndarray,
    arr_b: np.ndarray,
    n_bootstrap: int,
    seed: int,
    paired: bool,
):
    obs, ci_l, ci_u, p = bootstrap_delta_ci(arr_a, arr_b, n_bootstrap=n_bootstrap, seed=seed, paired=paired)
    rows.append(
        {
            "block": block,
            "type": "delta_ci",
            "method": f"{method_a} - {method_b}",
            "metric": metric,
            "value": None,
            "ci95_low": None,
            "ci95_high": None,
            "delta": obs,
            "delta_ci95_low": ci_l,
            "delta_ci95_high": ci_u,
            "p_value": p,
        }
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Build bootstrap CI/significance report from detailed evaluation outputs")
    parser.add_argument(
        "--dense-detail",
        type=Path,
        default=Path("artifacts/results/retrieval_dev_subset_v1_detail.json"),
    )
    parser.add_argument(
        "--milvus-detail",
        type=Path,
        default=Path("artifacts/results/retrieval_milvus_dev_subset_v1_detail.json"),
    )
    parser.add_argument(
        "--neo4j-raw-detail",
        type=Path,
        default=Path("artifacts/results/neo4j_2hop_smoke_v1_detail.json"),
    )
    parser.add_argument(
        "--neo4j-covered-detail",
        type=Path,
        default=Path("artifacts/results/neo4j_2hop_smoke_covered_v1_detail.json"),
    )
    parser.add_argument(
        "--main-detail",
        type=Path,
        default=Path("artifacts/results/tessera_main_vs_baselines_dev200_v1_detail.json"),
    )
    parser.add_argument(
        "--out-json",
        type=Path,
        default=Path("artifacts/results/significance_report_v1.json"),
    )
    parser.add_argument(
        "--out-md",
        type=Path,
        default=Path("artifacts/results/significance_report_v1.md"),
    )
    parser.add_argument("--n-bootstrap", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=20260326)
    args = parser.parse_args()

    rows: list[dict] = []

    if args.dense_detail.exists() and args.milvus_detail.exists():
        dense = load_json(args.dense_detail)
        milvus = load_json(args.milvus_detail)

        metric_key_cands = [
            (["recall@5", "hit@5", "any_hit@5"], "recall@5"),
            (["recall@10", "hit@10", "any_hit@10"], "recall@10"),
            (["recall@20", "hit@20", "any_hit@20"], "recall@20"),
        ]
        for metric_keys, metric_name in metric_key_cands:
            da = maybe_metric_array(dense, metric_keys)
            ma = maybe_metric_array(milvus, metric_keys)
            if da is None or ma is None:
                continue
            add_mean_row(rows, "retrieval", "dense_subset", metric_name, da, args.n_bootstrap, args.seed)
            add_mean_row(rows, "retrieval", "milvus_subset", metric_name, ma, args.n_bootstrap, args.seed)
            add_delta_row(
                rows,
                "retrieval",
                "dense_subset",
                "milvus_subset",
                metric_name,
                da,
                ma,
                args.n_bootstrap,
                args.seed,
                paired=True,
            )

    if args.neo4j_raw_detail.exists() and args.neo4j_covered_detail.exists():
        raw = load_json(args.neo4j_raw_detail)
        cov = load_json(args.neo4j_covered_detail)

        rh = maybe_array(raw, "hit")
        ch = maybe_array(cov, "hit")
        if rh is not None and ch is not None:
            add_mean_row(rows, "graph", "neo4j_2hop_raw", "hit_rate", rh, args.n_bootstrap, args.seed)
            add_mean_row(rows, "graph", "neo4j_2hop_covered", "hit_rate", ch, args.n_bootstrap, args.seed)
            add_delta_row(
                rows,
                "graph",
                "neo4j_2hop_covered",
                "neo4j_2hop_raw",
                "hit_rate",
                ch,
                rh,
                args.n_bootstrap,
                args.seed,
                paired=False,
            )

        rl = maybe_array(raw, "latency_ms")
        cl = maybe_array(cov, "latency_ms")
        if rl is not None and cl is not None:
            add_mean_row(rows, "graph", "neo4j_2hop_raw", "latency_ms_avg", rl, args.n_bootstrap, args.seed)
            add_mean_row(rows, "graph", "neo4j_2hop_covered", "latency_ms_avg", cl, args.n_bootstrap, args.seed)
            add_delta_row(
                rows,
                "graph",
                "neo4j_2hop_covered",
                "neo4j_2hop_raw",
                "latency_ms_avg",
                cl,
                rl,
                args.n_bootstrap,
                args.seed,
                paired=False,
            )

    if args.main_detail.exists():
        main = load_json(args.main_detail)
        methods = main.get("methods", {}) if isinstance(main, dict) else {}

        method_order = [
            "baseline_dense",
            "baseline_sparse_tfidf",
            "main_tessera",
            "ablation_no_late_interaction",
            "ablation_no_uncertainty_gating",
            "ablation_no_redundancy_detection",
        ]

        method_arrays: dict[str, dict[str, np.ndarray]] = {}
        for method in method_order:
            m = methods.get(method)
            if not isinstance(m, dict):
                continue
            metric_map: dict[str, np.ndarray] = {}
            metric_key_cands = [
                (["recall@5", "hit@5", "any_hit@5"], "recall@5"),
                (["recall@10", "hit@10", "any_hit@10"], "recall@10"),
                (["recall@20", "hit@20", "any_hit@20"], "recall@20"),
            ]
            for metric_keys, metric_name in metric_key_cands:
                arr = maybe_metric_array(m, metric_keys)
                if arr is None:
                    continue
                metric_map[metric_name] = arr
                add_mean_row(rows, "main_retrieval", method, metric_name, arr, args.n_bootstrap, args.seed)
            if metric_map:
                method_arrays[method] = metric_map

        comparisons = [
            ("main_tessera", "baseline_dense"),
            ("main_tessera", "baseline_sparse_tfidf"),
            ("main_tessera", "ablation_no_late_interaction"),
            ("main_tessera", "ablation_no_uncertainty_gating"),
            ("main_tessera", "ablation_no_redundancy_detection"),
        ]
        for method_a, method_b in comparisons:
            if method_a not in method_arrays or method_b not in method_arrays:
                continue
            for metric_name in ["recall@5", "recall@10", "recall@20"]:
                arr_a = method_arrays[method_a].get(metric_name)
                arr_b = method_arrays[method_b].get(metric_name)
                if arr_a is None or arr_b is None:
                    continue
                add_delta_row(
                    rows,
                    "main_retrieval",
                    method_a,
                    method_b,
                    metric_name,
                    arr_a,
                    arr_b,
                    args.n_bootstrap,
                    args.seed,
                    paired=True,
                )

    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps({"n_bootstrap": args.n_bootstrap, "seed": args.seed, "rows": rows}, ensure_ascii=False, indent=2), encoding="utf-8")

    md = [
        "# Significance Report (Bootstrap)",
        "",
        f"- n_bootstrap: {args.n_bootstrap}",
        f"- seed: {args.seed}",
        "",
        "| Block | Type | Method | Metric | Value | CI95 | Delta | Delta CI95 | p-value |",
        "|---|---|---|---|---:|---:|---:|---:|---:|",
    ]

    def f4(x):
        return "" if x is None else f"{x:.4f}"

    for r in rows:
        ci = "" if r["ci95_low"] is None else f"[{r['ci95_low']:.4f}, {r['ci95_high']:.4f}]"
        dci = "" if r["delta_ci95_low"] is None else f"[{r['delta_ci95_low']:.4f}, {r['delta_ci95_high']:.4f}]"
        md.append(
            f"| {r['block']} | {r['type']} | {r['method']} | {r['metric']} | {f4(r['value'])} | {ci} | {f4(r['delta'])} | {dci} | {f4(r['p_value'])} |"
        )

    args.out_md.write_text("\n".join(md) + "\n", encoding="utf-8")
    print(f"[OK] json -> {args.out_json}")
    print(f"[OK] md -> {args.out_md}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
