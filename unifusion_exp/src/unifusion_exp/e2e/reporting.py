from __future__ import annotations


def build_table1c_markdown(results: dict) -> str:
    meta = results.get("meta", {})
    rows = results.get("table1c_rows", [])

    md = [
        "# Table 1c E2E QA Metrics",
        "",
        f"- queries: {meta.get('queries', 0)}",
        f"- corpus: {meta.get('corpus', 0)}",
        f"- reader: {meta.get('reader')}",
        f"- retrieve_topk: {meta.get('retrieve_topk')}",
        f"- qa_context_k: {meta.get('qa_context_k')}",
        f"- official_mmrag_mode: {bool(meta.get('official_mmrag_mode', False))}",
        f"- router_source: {meta.get('router_source')}",
        f"- router_subset_acc_run: {float(meta.get('router_subset_acc_run', 0.0)):.4f}",
        f"- router_micro_f1_run: {float(meta.get('router_micro_f1_run', 0.0)):.4f}",
        f"- router_avg_entropy: {float(meta.get('router_avg_entropy', 0.0)):.4f}",
        f"- router_uncertain_ratio: {float(meta.get('router_uncertain_ratio', 0.0)):.4f}",
        "",
        "| 方法 | F1-score | EM | mmRAG Official Avg | Routing Acc | Recall@10 | P95 Latency (ms) |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]

    for r in rows:
        routing_acc = r.get("routing_acc")
        ra = "-" if routing_acc is None else f"{float(routing_acc):.4f}"
        mm_val = r.get("mmrag_official_avg")
        mm = "-" if mm_val is None else f"{float(mm_val):.4f}"
        md.append(
            f"| {r['method']} | {float(r['f1']):.4f} | {float(r['exact_match']):.4f} | {mm} | {ra} | {float(r['recall@10']):.4f} | {float(r['p95_latency_ms']):.2f} |"
        )

    return "\n".join(md) + "\n"
