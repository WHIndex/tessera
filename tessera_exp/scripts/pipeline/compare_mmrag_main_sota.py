#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path


OFFICIAL_QWEN_OVERALL = {
    "No retrieval": 0.1429,
    "BM25": 0.2165,
    "Contriever": 0.1852,
    "DPR": 0.1258,
    "BGE": 0.2223,
    "GTE": 0.2482,
    "Fine-tuned BGE": 0.3102,
    "Fine-tuned GTE": 0.3099,
    "Oracle": 0.3720,
}


def load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def method_is_oracle(key: str, method: str) -> bool:
    s = f"{key}|{method}".lower()
    return "oracle" in s


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Compare current E2E run against mmRAG main-setting Qwen reference numbers"
    )
    parser.add_argument(
        "--table1c-metrics",
        type=Path,
        default=Path("artifacts/results/table1c_e2e_mmrag_main_aligned/table1c_e2e_metrics.json"),
    )
    parser.add_argument(
        "--out-json",
        type=Path,
        default=Path("artifacts/results/mmrag_main_sota_compare.json"),
    )
    parser.add_argument(
        "--out-md",
        type=Path,
        default=Path("artifacts/results/mmrag_main_sota_compare.md"),
    )
    args = parser.parse_args()

    data = load_json(args.table1c_metrics)
    meta = data.get("meta", {}) if isinstance(data, dict) else {}
    rows = data.get("table1c_rows", []) if isinstance(data, dict) else []

    ours = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        key = str(row.get("key", ""))
        method = str(row.get("method", ""))
        score = float(row.get("mmrag_official_avg", 0.0))
        ours.append({"key": key, "method": method, "mmrag_official_avg": score, "is_oracle": method_is_oracle(key, method)})

    ours_non_oracle = [x for x in ours if not x["is_oracle"]]
    ours_best_non_oracle = max(ours_non_oracle, key=lambda x: x["mmrag_official_avg"], default=None)

    official_non_oracle = {k: v for k, v in OFFICIAL_QWEN_OVERALL.items() if k != "Oracle"}
    official_best_name = max(official_non_oracle, key=official_non_oracle.get)
    official_best_non_oracle = float(official_non_oracle[official_best_name])

    protocol_checks = {
        "official_mmrag_mode": bool(meta.get("official_mmrag_mode", False)),
        "qa_context_k_is_3": int(meta.get("qa_context_k", -1)) == 3,
        "not_likely_transductive_test_corpus": not bool(meta.get("likely_transductive_test_corpus", False)),
    }
    protocol_passed = all(protocol_checks.values())

    ours_best_score = float(ours_best_non_oracle["mmrag_official_avg"]) if ours_best_non_oracle else 0.0
    delta_vs_official_best = float(ours_best_score - official_best_non_oracle)

    result = {
        "input": {
            "table1c_metrics": str(args.table1c_metrics),
        },
        "protocol_checks": protocol_checks,
        "protocol_passed": protocol_passed,
        "official_reference": {
            "source": "mmRAG paper Table 9 (Qwen, Over All Chunks)",
            "scores": OFFICIAL_QWEN_OVERALL,
            "best_non_oracle_name": official_best_name,
            "best_non_oracle_score": official_best_non_oracle,
            "oracle_score": OFFICIAL_QWEN_OVERALL["Oracle"],
        },
        "ours": {
            "methods": ours,
            "best_non_oracle": ours_best_non_oracle,
        },
        "score_comparison": {
            "delta_vs_official_best_non_oracle": delta_vs_official_best,
            "ours_is_non_oracle_sota_by_score_only": bool(ours_best_non_oracle and delta_vs_official_best > 0.0),
        },
        "claim_gate": {
            "can_claim_direct_main_setting_comparison": protocol_passed,
            "can_claim_non_oracle_sota_by_score_only": bool(protocol_passed and ours_best_non_oracle and delta_vs_official_best > 0.0),
        },
    }

    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")

    md = [
        "# mmRAG Main-Setting Comparison",
        "",
        f"- input: {args.table1c_metrics}",
        f"- protocol_passed: {protocol_passed}",
        f"- official best non-oracle ({official_best_name}): {official_best_non_oracle:.4f}",
    ]
    if ours_best_non_oracle:
        md.append(
            "- ours best non-oracle: "
            f"{ours_best_non_oracle['method']} ({ours_best_non_oracle['key']}), "
            f"mmrag_official_avg={ours_best_non_oracle['mmrag_official_avg']:.4f}"
        )
    else:
        md.append("- ours best non-oracle: n/a")
    md.append(f"- delta vs official best non-oracle: {delta_vs_official_best:+.4f}")
    md.append("")
    md.append("## Protocol Checks")
    for k, v in protocol_checks.items():
        md.append(f"- {k}: {v}")
    md.append("")
    md.append("## Ours")
    md.append("| Method | Key | mmrag_official_avg | Oracle |")
    md.append("|---|---|---:|---:|")
    for x in sorted(ours, key=lambda y: y["mmrag_official_avg"], reverse=True):
        md.append(f"| {x['method']} | {x['key']} | {x['mmrag_official_avg']:.4f} | {x['is_oracle']} |")

    args.out_md.write_text("\n".join(md) + "\n", encoding="utf-8")

    print(f"[OK] json -> {args.out_json}")
    print(f"[OK] markdown -> {args.out_md}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
