#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def load_json(path: Path):
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def first_existing_json(paths: list[Path]):
    for p in paths:
        if p.exists():
            return p, load_json(p)
    return None, None


def find_one(pattern: str) -> Path | None:
    cands = sorted(Path("runs").glob(pattern))
    return cands[-1] if cands else None


def fmt(v):
    if isinstance(v, float):
        return f"{v:.4f}"
    return str(v)


def normalize_table1c_method_name(name: str) -> str:
    mapping = {
        "CARP (IJCAI 22)": "CARP-Adapter (IJCAI 22)",
        "TableRAG (Google)": "TableRAG-Adapter (Google)",
        "QUASAR": "QUASAR-Adapter",
    }
    return mapping.get(name, name)


def main() -> int:
    parser = argparse.ArgumentParser(description="Build paper-ready summary table from current artifacts")
    parser.add_argument("--out-md", type=Path, default=Path("artifacts/results/paper_ready_results_v1.md"))
    parser.add_argument("--out-csv", type=Path, default=Path("artifacts/results/paper_ready_results_v1.csv"))
    args = parser.parse_args()

    router_fast_fp = find_one("**/router_fast_full_t045_metrics.json")
    router_deberta_fp = find_one("**/router_deberta_full_metrics.json")

    router_fast = load_json(router_fast_fp) if router_fast_fp else None
    router_deberta = load_json(router_deberta_fp) if router_deberta_fp else None

    dense = load_json(Path("artifacts/results/retrieval_dev_subset_v1.json"))
    milvus = load_json(Path("artifacts/results/retrieval_milvus_dev_subset_v1.json"))
    tessera_main = load_json(Path("artifacts/results/tessera_main_vs_baselines_dev200_v1.json"))
    v5_adaptive_dev200_fp, v5_adaptive_dev200 = first_existing_json(
        [
            Path("artifacts/results/tmp_dev200_p12_adapt.json"),
            Path("artifacts/results/reassess_v5_dev200_adaptive_nosourcecap_20260327.json"),
            Path("artifacts/results/reassess_v4_dev200_guard10_20260327.json"),
        ]
    )
    v5_adaptive_dev766_fp, v5_adaptive_dev766 = first_existing_json(
        [
            Path("artifacts/results/reassess_v5_dev766_adaptive_p12_nocap_20260328.json"),
            Path("artifacts/results/reassess_v5_dev766_adaptive_p12_20260327.json"),
            Path("artifacts/results/reassess_v5_dev766_adaptive_nosourcecap_20260327.json"),
            Path("artifacts/results/reassess_v4_dev766_guard10_20260327.json"),
        ]
    )
    v5_adaptive_sanity_fp, v5_adaptive_sanity = first_existing_json(
        [
            Path("artifacts/results/reassess_sanity_v5_dev766_adaptive_p12_nocap_20260328_tol0.json"),
            Path("artifacts/results/reassess_sanity_v5_dev766_adaptive_p12_tol0.json"),
            Path("artifacts/results/reassess_sanity_v4_dev766_guard10_tol0.json"),
        ]
    )
    v5_adaptive_significance_fp, v5_adaptive_significance = first_existing_json(
        [
            Path("artifacts/results/reassess_significance_v5_dev766_adaptive_p12_nocap_20260328.json"),
            Path("artifacts/results/reassess_significance_v5_dev766_adaptive_p12_20260327.json"),
            Path("artifacts/results/reassess_significance_v4_dev766_guard10_20260327.json"),
        ]
    )
    table1c_e2e_fp, table1c_e2e = first_existing_json(
        [
            Path("artifacts/results/table1c_e2e_mmrag_main_aligned/table1c_e2e_metrics.json"),
            Path("artifacts/results/table1c_e2e_mmrag_official_main_aligned/table1c_e2e_metrics.json"),
            Path("artifacts/results/table1c_e2e_20260328_llm_qwen25_campe_full1286_all_v2_ablationfix/table1c_e2e_metrics.json"),
            Path("artifacts/results/table1c_e2e_20260328_llm_qwen25_campe_full1286_all_v1/table1c_e2e_metrics.json"),
            Path("artifacts/results/table1c_e2e_20260327_llm_qwen25_campe_full1286_v2_dense4/table1c_e2e_metrics.json"),
            Path("artifacts/results/table1c_e2e_20260327_llm_qwen25_adaptivectx_v1/table1c_e2e_metrics.json"),
            Path("artifacts/results/table1c_e2e_20260327_llm_qwen25_routerfix_p0/table1c_e2e_metrics.json"),
            Path("artifacts/results/table1c_e2e_20260327_llm_qwen25_p0/table1c_e2e_metrics.json"),
            Path("artifacts/results/table1c_e2e_20260327/table1c_e2e_metrics.json"),
        ]
    )
    table1c_e2e_ablation_fp, table1c_e2e_ablation = (None, None)
    table1c_sig_fp, table1c_sig = first_existing_json(
        [
            Path("artifacts/results/table1c_e2e_mmrag_main_aligned/e2e_paired_bootstrap_dense_vs_tessera.json"),
            Path("artifacts/results/table1c_e2e_mmrag_official_main_aligned/e2e_paired_bootstrap_dense_vs_tessera.json"),
            Path("artifacts/results/table1c_e2e_20260328_llm_qwen25_campe_full1286_all_v2_ablationfix/e2e_paired_bootstrap_dense_vs_tessera.json"),
            Path("artifacts/results/table1c_e2e_20260328_llm_qwen25_campe_full1286_all_v1/e2e_paired_bootstrap_dense_vs_tessera.json"),
            Path("artifacts/results/table1c_e2e_20260327_llm_qwen25_campe_full1286_v2_dense4/e2e_paired_bootstrap_dense_vs_tessera.json"),
            Path("artifacts/results/table1c_e2e_20260327_llm_qwen25_adaptivectx_v1/e2e_paired_bootstrap_dense_vs_tessera.json"),
            Path("artifacts/results/table1c_e2e_20260327_llm_qwen25_routerfix_p0/e2e_paired_bootstrap_dense_vs_tessera.json"),
            Path("artifacts/results/table1c_e2e_20260327_llm_qwen25_p0/e2e_paired_bootstrap_dense_vs_tessera.json"),
            Path("artifacts/results/table1c_e2e_20260327/e2e_paired_bootstrap_dense_vs_tessera.json"),
        ]
    )
    neo4j_raw = load_json(Path("artifacts/results/neo4j_2hop_smoke_v1.json"))
    neo4j_cov = load_json(Path("artifacts/results/neo4j_2hop_smoke_covered_v1.json"))
    neo4j_load = load_json(Path("artifacts/results/neo4j_load_subset_v1.json"))
    significance = load_json(Path("artifacts/results/significance_report_v1.json"))

    rows = []

    if router_fast:
        rows.extend(
            [
                {"block": "router", "method": "fast_t045", "metric": "test_micro_f1", "value": router_fast.get("test_micro_f1")},
                {"block": "router", "method": "fast_t045", "metric": "test_subset_acc", "value": router_fast.get("test_subset_acc")},
            ]
        )

    if router_deberta:
        rows.extend(
            [
                {"block": "router", "method": "deberta_full", "metric": "test_micro_f1", "value": router_deberta.get("test_micro_f1")},
                {"block": "router", "method": "deberta_full", "metric": "test_subset_acc", "value": router_deberta.get("test_subset_acc")},
            ]
        )

    if dense:
        rows.extend(
            [
                {"block": "retrieval", "method": "dense_subset", "metric": "recall@5", "value": dense.get("recall@5")},
                {"block": "retrieval", "method": "dense_subset", "metric": "recall@10", "value": dense.get("recall@10")},
                {"block": "retrieval", "method": "dense_subset", "metric": "recall@20", "value": dense.get("recall@20")},
            ]
        )

    if milvus:
        rows.extend(
            [
                {"block": "retrieval", "method": "milvus_subset", "metric": "recall@5", "value": milvus.get("recall@5")},
                {"block": "retrieval", "method": "milvus_subset", "metric": "recall@10", "value": milvus.get("recall@10")},
                {"block": "retrieval", "method": "milvus_subset", "metric": "recall@20", "value": milvus.get("recall@20")},
            ]
        )

    if tessera_main and isinstance(tessera_main.get("methods"), dict):
        for method, vals in tessera_main["methods"].items():
            rows.extend(
                [
                    {"block": "main_retrieval", "method": method, "metric": "recall@5", "value": vals.get("recall@5")},
                    {"block": "main_retrieval", "method": method, "metric": "recall@10", "value": vals.get("recall@10")},
                    {"block": "main_retrieval", "method": method, "metric": "recall@20", "value": vals.get("recall@20")},
                ]
            )

    if v5_adaptive_dev200 and isinstance(v5_adaptive_dev200.get("methods"), dict):
        for method in ["baseline_dense", "main_tessera"]:
            vals = v5_adaptive_dev200["methods"].get(method, {})
            rows.extend(
                [
                    {"block": "v5_adaptive_dev200", "method": method, "metric": "recall@5", "value": vals.get("recall@5")},
                    {"block": "v5_adaptive_dev200", "method": method, "metric": "recall@10", "value": vals.get("recall@10")},
                    {"block": "v5_adaptive_dev200", "method": method, "metric": "recall@20", "value": vals.get("recall@20")},
                ]
            )

    if v5_adaptive_dev766 and isinstance(v5_adaptive_dev766.get("methods"), dict):
        for method in ["baseline_dense", "main_tessera"]:
            vals = v5_adaptive_dev766["methods"].get(method, {})
            rows.extend(
                [
                    {"block": "v5_adaptive_dev766", "method": method, "metric": "recall@5", "value": vals.get("recall@5")},
                    {"block": "v5_adaptive_dev766", "method": method, "metric": "recall@10", "value": vals.get("recall@10")},
                    {"block": "v5_adaptive_dev766", "method": method, "metric": "recall@20", "value": vals.get("recall@20")},
                ]
            )

    if v5_adaptive_sanity:
        rows.extend(
            [
                {"block": "v5_adaptive_dev766", "method": "sanity", "metric": "failed_checks", "value": v5_adaptive_sanity.get("failed")},
                {"block": "v5_adaptive_dev766", "method": "sanity", "metric": "warn_checks", "value": v5_adaptive_sanity.get("warn")},
            ]
        )

    if table1c_e2e and isinstance(table1c_e2e.get("table1c_rows"), list):
        for row in table1c_e2e["table1c_rows"]:
            method = normalize_table1c_method_name(row.get("method", "unknown"))
            rows.extend(
                [
                    {"block": "table1c_e2e", "method": method, "metric": "f1", "value": row.get("f1")},
                    {"block": "table1c_e2e", "method": method, "metric": "exact_match", "value": row.get("exact_match")},
                    {"block": "table1c_e2e", "method": method, "metric": "mmrag_official_avg", "value": row.get("mmrag_official_avg")},
                    {"block": "table1c_e2e", "method": method, "metric": "recall@10", "value": row.get("recall@10")},
                    {"block": "table1c_e2e", "method": method, "metric": "routing_acc", "value": row.get("routing_acc")},
                    {"block": "table1c_e2e", "method": method, "metric": "p95_latency_ms", "value": row.get("p95_latency_ms")},
                ]
            )

    if table1c_e2e and isinstance(table1c_e2e.get("meta"), dict):
        meta = table1c_e2e.get("meta", {})
        rows.extend(
            [
                {
                    "block": "table1c_protocol",
                    "method": "run_meta",
                    "metric": "official_mmrag_mode",
                    "value": int(bool(meta.get("official_mmrag_mode", False))),
                },
                {
                    "block": "table1c_protocol",
                    "method": "run_meta",
                    "metric": "qa_context_k",
                    "value": meta.get("qa_context_k"),
                },
                {
                    "block": "table1c_protocol",
                    "method": "run_meta",
                    "metric": "likely_transductive_test_corpus",
                    "value": int(bool(meta.get("likely_transductive_test_corpus", False))),
                },
            ]
        )

    if (
        table1c_e2e_ablation
        and isinstance(table1c_e2e_ablation.get("table1c_rows"), list)
        and table1c_e2e_ablation_fp != table1c_e2e_fp
    ):
        for row in table1c_e2e_ablation["table1c_rows"]:
            key = str(row.get("key", ""))
            if key not in {"ablation_no_redundancy_e2e", "ablation_no_pathmaxsim_e2e"}:
                continue
            method = normalize_table1c_method_name(row.get("method", "unknown"))
            rows.extend(
                [
                    {"block": "table1c_e2e", "method": method, "metric": "f1", "value": row.get("f1")},
                    {"block": "table1c_e2e", "method": method, "metric": "exact_match", "value": row.get("exact_match")},
                    {"block": "table1c_e2e", "method": method, "metric": "mmrag_official_avg", "value": row.get("mmrag_official_avg")},
                    {"block": "table1c_e2e", "method": method, "metric": "recall@10", "value": row.get("recall@10")},
                    {"block": "table1c_e2e", "method": method, "metric": "routing_acc", "value": row.get("routing_acc")},
                    {"block": "table1c_e2e", "method": method, "metric": "p95_latency_ms", "value": row.get("p95_latency_ms")},
                ]
            )

    if table1c_sig and isinstance(table1c_sig.get("rows"), list):
        sig_meta = table1c_sig.get("meta", {}) if isinstance(table1c_sig, dict) else {}
        compare_name = f"{sig_meta.get('b_name', 'model_b')} - {sig_meta.get('a_name', 'model_a')}"
        for row in table1c_sig["rows"]:
            metric = str(row.get("metric", "unknown"))
            rows.extend(
                [
                    {"block": "table1c_e2e_sig", "method": compare_name, "metric": f"{metric}_delta", "value": row.get("delta")},
                    {"block": "table1c_e2e_sig", "method": compare_name, "metric": f"{metric}_delta_ci95_low", "value": row.get("delta_ci95_low")},
                    {"block": "table1c_e2e_sig", "method": compare_name, "metric": f"{metric}_delta_ci95_high", "value": row.get("delta_ci95_high")},
                    {"block": "table1c_e2e_sig", "method": compare_name, "metric": f"{metric}_p_value", "value": row.get("p_value")},
                    {"block": "table1c_e2e_sig", "method": compare_name, "metric": f"{metric}_wins", "value": row.get("wins")},
                    {"block": "table1c_e2e_sig", "method": compare_name, "metric": f"{metric}_ties", "value": row.get("ties")},
                    {"block": "table1c_e2e_sig", "method": compare_name, "metric": f"{metric}_losses", "value": row.get("losses")},
                ]
            )

    if neo4j_raw:
        rows.extend(
            [
                {"block": "graph", "method": "neo4j_2hop_raw", "metric": "hit_rate", "value": neo4j_raw.get("hit_rate")},
                {"block": "graph", "method": "neo4j_2hop_raw", "metric": "latency_ms_avg", "value": neo4j_raw.get("latency_ms_avg")},
            ]
        )

    if neo4j_cov:
        rows.extend(
            [
                {"block": "graph", "method": "neo4j_2hop_covered", "metric": "hit_rate", "value": neo4j_cov.get("hit_rate")},
                {"block": "graph", "method": "neo4j_2hop_covered", "metric": "latency_ms_avg", "value": neo4j_cov.get("latency_ms_avg")},
            ]
        )

    if neo4j_load:
        rows.extend(
            [
                {"block": "graph_load", "method": "neo4j_load_subset", "metric": "triples_loaded", "value": neo4j_load.get("triples_loaded")},
                {"block": "graph_load", "method": "neo4j_load_subset", "metric": "nodes", "value": neo4j_load.get("nodes")},
                {"block": "graph_load", "method": "neo4j_load_subset", "metric": "rels", "value": neo4j_load.get("rels")},
            ]
        )

    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    with args.out_csv.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["block", "method", "metric", "value"])
        w.writeheader()
        for r in rows:
            w.writerow(r)

    md = [
        "# Paper-ready Results (Current Snapshot)",
        "",
        "| Block | Method | Metric | Value |",
        "|---|---|---:|---:|",
    ]
    for r in rows:
        md.append(f"| {r['block']} | {r['method']} | {r['metric']} | {fmt(r['value'])} |")

    if significance and isinstance(significance.get("rows"), list):
        sig_rows = significance["rows"]

        def pick_delta(method: str, metric: str):
            for r in sig_rows:
                if r.get("type") == "delta_ci" and r.get("method") == method and r.get("metric") == metric:
                    return r
            return None

        retrieval_sig = pick_delta("dense_subset - milvus_subset", "recall@10")
        graph_sig = pick_delta("neo4j_2hop_covered - neo4j_2hop_raw", "hit_rate")
        main_dense_sig = pick_delta("main_tessera - baseline_dense", "recall@20")
        main_sparse_sig = pick_delta("main_tessera - baseline_sparse_tfidf", "recall@10")
        main_ablation_sig = pick_delta("main_tessera - ablation_no_redundancy_detection", "recall@10")

        md.append("")
        md.append("## Statistical Evidence (Bootstrap)")
        if retrieval_sig:
            md.append(
                "- retrieval recall@10 delta (dense - milvus): "
                f"{retrieval_sig.get('delta', 0.0):.4f}, "
                f"CI95=[{retrieval_sig.get('delta_ci95_low', 0.0):.4f}, {retrieval_sig.get('delta_ci95_high', 0.0):.4f}], "
                f"p={retrieval_sig.get('p_value', 1.0):.4f}"
            )
        if graph_sig:
            md.append(
                "- graph hit_rate delta (covered - raw): "
                f"{graph_sig.get('delta', 0.0):.4f}, "
                f"CI95=[{graph_sig.get('delta_ci95_low', 0.0):.4f}, {graph_sig.get('delta_ci95_high', 0.0):.4f}], "
                f"p={graph_sig.get('p_value', 1.0):.4f}"
            )
        if main_dense_sig:
            md.append(
                "- main retrieval recall@20 delta (main - dense): "
                f"{main_dense_sig.get('delta', 0.0):.4f}, "
                f"CI95=[{main_dense_sig.get('delta_ci95_low', 0.0):.4f}, {main_dense_sig.get('delta_ci95_high', 0.0):.4f}], "
                f"p={main_dense_sig.get('p_value', 1.0):.4f}"
            )
        if main_sparse_sig:
            md.append(
                "- main retrieval recall@10 delta (main - sparse): "
                f"{main_sparse_sig.get('delta', 0.0):.4f}, "
                f"CI95=[{main_sparse_sig.get('delta_ci95_low', 0.0):.4f}, {main_sparse_sig.get('delta_ci95_high', 0.0):.4f}], "
                f"p={main_sparse_sig.get('p_value', 1.0):.4f}"
            )
        if main_ablation_sig:
            md.append(
                "- ablation impact recall@10 delta (main - no_redundancy_detection): "
                f"{main_ablation_sig.get('delta', 0.0):.4f}, "
                f"CI95=[{main_ablation_sig.get('delta_ci95_low', 0.0):.4f}, {main_ablation_sig.get('delta_ci95_high', 0.0):.4f}], "
                f"p={main_ablation_sig.get('p_value', 1.0):.4f}"
            )

    if v5_adaptive_significance and isinstance(v5_adaptive_significance.get("rows"), list):
        sig_rows = v5_adaptive_significance["rows"]
        v5_delta = None
        for r in sig_rows:
            if (
                r.get("metric") == "recall@20"
                and r.get("delta") is not None
            ):
                v5_delta = r
                break
        if v5_delta:
            md.append("")
            md.append("## V5 Adaptive Statistical Evidence")
            md.append(
                "- dev766 recall@20 delta (main - dense): "
                f"{v5_delta.get('delta', 0.0):.4f}, "
                f"CI95=[{v5_delta.get('delta_ci95_low', 0.0):.4f}, {v5_delta.get('delta_ci95_high', 0.0):.4f}], "
                f"p={v5_delta.get('p_value', 1.0):.4f}"
            )

    if table1c_e2e and isinstance(table1c_e2e.get("table1c_rows"), list):
        md.append("")
        md.append("## Table 1c Snapshot (E2E QA)")
        md.append("- 注：CARP/TableRAG/QUASAR 采用统一 adapter 协议近似复现（非官方原生 pipeline）。")
        for row in table1c_e2e["table1c_rows"]:
            method = normalize_table1c_method_name(row.get("method", "unknown"))
            routing_val = row.get("routing_acc")
            routing_str = f"{routing_val:.4f}" if isinstance(routing_val, (int, float)) else "-"
            md.append(
                f"- {method}: F1={row.get('f1', 0.0):.4f}, EM={row.get('exact_match', 0.0):.4f}, "
                f"OfficialAvg={row.get('mmrag_official_avg', 0.0):.4f}, R@10={row.get('recall@10', 0.0):.4f}, Routing={routing_str}, "
                f"P95={row.get('p95_latency_ms', 0.0):.2f} ms"
            )

        meta = table1c_e2e.get("meta", {}) if isinstance(table1c_e2e, dict) else {}
        md.append(
            "- protocol: "
            f"official_mmrag_mode={bool(meta.get('official_mmrag_mode', False))}, "
            f"qa_context_k={meta.get('qa_context_k', '-')}, "
            f"likely_transductive_test_corpus={bool(meta.get('likely_transductive_test_corpus', False))}"
        )

    if (
        table1c_e2e_ablation
        and isinstance(table1c_e2e_ablation.get("table1c_rows"), list)
        and table1c_e2e_ablation_fp != table1c_e2e_fp
    ):
        md.append("- 补充（同语料独立补跑）:")
        for row in table1c_e2e_ablation["table1c_rows"]:
            key = str(row.get("key", ""))
            if key not in {"ablation_no_redundancy_e2e", "ablation_no_pathmaxsim_e2e"}:
                continue
            method = normalize_table1c_method_name(row.get("method", "unknown"))
            routing_val = row.get("routing_acc")
            routing_str = f"{routing_val:.4f}" if isinstance(routing_val, (int, float)) else "-"
            md.append(
                f"- {method}: F1={row.get('f1', 0.0):.4f}, EM={row.get('exact_match', 0.0):.4f}, "
                f"OfficialAvg={row.get('mmrag_official_avg', 0.0):.4f}, R@10={row.get('recall@10', 0.0):.4f}, Routing={routing_str}, "
                f"P95={row.get('p95_latency_ms', 0.0):.2f} ms"
            )

    if table1c_sig and isinstance(table1c_sig.get("rows"), list):
        sig_meta = table1c_sig.get("meta", {}) if isinstance(table1c_sig, dict) else {}
        md.append("")
        md.append("## Table 1c Paired Bootstrap (TESSERA vs Dense)")
        md.append(
            f"- evaluated={sig_meta.get('evaluated', '-')}, coverage={sig_meta.get('coverage', 0.0):.4f}, "
            f"n_bootstrap={sig_meta.get('n_bootstrap', '-')}"
        )
        for row in table1c_sig["rows"]:
            metric = str(row.get("metric", "unknown"))
            md.append(
                f"- {metric}: delta={row.get('delta', 0.0):.4f}, "
                f"CI95=[{row.get('delta_ci95_low', 0.0):.4f}, {row.get('delta_ci95_high', 0.0):.4f}], "
                f"p={row.get('p_value', 1.0):.4f}, "
                f"wins/ties/losses={row.get('wins', 0)}/{row.get('ties', 0)}/{row.get('losses', 0)}"
            )

    if router_fast_fp or router_deberta_fp:
        md.append("")
        md.append("## Metric Sources")
        if router_fast_fp:
            md.append(f"- router_fast: {router_fast_fp}")
        if router_deberta_fp:
            md.append(f"- router_deberta: {router_deberta_fp}")
        md.append("- dense: artifacts/results/retrieval_dev_subset_v1.json")
        md.append("- milvus: artifacts/results/retrieval_milvus_dev_subset_v1.json")
        md.append("- main_retrieval: artifacts/results/tessera_main_vs_baselines_dev200_v1.json")
        md.append("- neo4j_raw: artifacts/results/neo4j_2hop_smoke_v1.json")
        md.append("- neo4j_covered: artifacts/results/neo4j_2hop_smoke_covered_v1.json (if exists)")
        md.append("- significance: artifacts/results/significance_report_v1.json (if exists)")
        if v5_adaptive_dev200_fp:
            md.append(f"- v5_adaptive_dev200: {v5_adaptive_dev200_fp}")
        if v5_adaptive_dev766_fp:
            md.append(f"- v5_adaptive_dev766: {v5_adaptive_dev766_fp}")
        if v5_adaptive_sanity_fp:
            md.append(f"- v5_adaptive_sanity: {v5_adaptive_sanity_fp}")
        if v5_adaptive_significance_fp:
            md.append(f"- v5_adaptive_significance: {v5_adaptive_significance_fp}")
        if table1c_e2e_fp:
            md.append(f"- table1c_e2e: {table1c_e2e_fp}")
        else:
            md.append("- table1c_e2e: artifacts/results/table1c_e2e_20260328_llm_qwen25_campe_full1286_all_v1/table1c_e2e_metrics.json (missing)")
        if table1c_e2e_ablation_fp:
            md.append(f"- table1c_e2e_ablation_de: {table1c_e2e_ablation_fp}")
        if table1c_sig_fp:
            md.append(f"- table1c_e2e_sig: {table1c_sig_fp}")
        else:
            md.append("- table1c_e2e_sig: artifacts/results/table1c_e2e_20260328_llm_qwen25_campe_full1286_all_v1/e2e_paired_bootstrap_dense_vs_tessera.json (missing)")

    args.out_md.write_text("\n".join(md) + "\n", encoding="utf-8")

    print(f"[OK] markdown -> {args.out_md}")
    print(f"[OK] csv -> {args.out_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
