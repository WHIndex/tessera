#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path


@dataclass
class SweepJob:
    mode: str
    out_dir: Path
    gpu: str


def parse_modes(raw: str) -> list[str]:
    vals = [x.strip().lower() for x in str(raw).split(",") if x.strip()]
    dedup = []
    seen = set()
    for v in vals:
        if v in seen:
            continue
        seen.add(v)
        dedup.append(v)
    return dedup


def parse_gpus(raw: str) -> list[str]:
    vals = [x.strip() for x in str(raw).split(",") if x.strip()]
    return vals or ["0"]


def load_metrics(path: Path) -> dict:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def render_summary_md(rows: list[dict], out_json: Path) -> str:
    md = [
        "# SchemeB Calibration Sweep Summary",
        "",
        f"- summary_json: {out_json}",
        "",
        "| mode | tessera_f1 | tessera_em | tessera_r10 | dense_f1 | tapas_enabled | tapas_query_ready_mean | heavy_table_w_mean | heavy_kg_w_mean | heavy_token_late_w_mean |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for r in rows:
        md.append(
            f"| {r['mode']} | {r['tessera_f1']:.4f} | {r['tessera_em']:.4f} | {r['tessera_r10']:.4f} | {r['dense_f1']:.4f} | "
            f"{r['tapas_enabled']} | {r['tapas_query_ready_mean']:.4f} | {r['heavy_table_w_mean']:.4f} | {r['heavy_kg_w_mean']:.4f} | {r['heavy_token_late_w_mean']:.4f} |"
        )
    return "\n".join(md) + "\n"


def run_job(job: SweepJob, args: argparse.Namespace, eval_script: Path) -> tuple[str, int, str]:
    job.out_dir.mkdir(parents=True, exist_ok=True)

    dense_target = job.out_dir / "qa_predictions_dense_concat_test1286.jsonl"
    if args.baseline_dense_pred_file and Path(args.baseline_dense_pred_file).exists():
        if not dense_target.exists() or not args.skip_copy_if_exists:
            shutil.copy2(Path(args.baseline_dense_pred_file), dense_target)

    cmd = [
        sys.executable,
        str(eval_script),
        "--model-dir",
        str(args.model_dir),
        "--split-file",
        str(args.split_file),
        "--corpus-file",
        str(args.corpus_file),
        "--out-dir",
        str(job.out_dir),
        "--max-queries",
        str(args.max_queries),
        "--retrieve-topk",
        str(args.retrieve_topk),
        "--qa-context-k",
        str(args.qa_context_k),
        "--reader",
        str(args.reader),
        "--preserve-dense-top",
        str(args.preserve_dense_top),
        "--tessera-late-alpha",
        str(args.tessera_late_alpha),
        "--method-preset",
        "main_only",
        "--methods",
        "dense_concat,tessera_rag",
        "--reuse-methods",
        "dense_concat",
        "--context-active-threshold",
        str(args.context_active_threshold),
        "--context-anchor-dense-k",
        str(args.context_anchor_dense_k),
        "--context-anchor-uni-k",
        str(args.context_anchor_uni_k),
        "--context-dense-pool-k",
        str(args.context_dense_pool_k),
        "--context-candidate-expand-k",
        str(args.context_candidate_expand_k),
        "--context-redundancy-lambda",
        str(args.context_redundancy_lambda),
        "--pathmaxsim-weight",
        str(args.pathmaxsim_weight),
        "--pathmaxsim-kg-threshold",
        str(args.pathmaxsim_kg_threshold),
        "--table-cellmaxsim-weight",
        str(args.table_cellmaxsim_weight),
        "--table-cellmaxsim-top-cells",
        str(args.table_cellmaxsim_top_cells),
        "--context-consistency-weight",
        str(args.context_consistency_weight),
        "--context-qa-objective-weight",
        str(args.context_qa_objective_weight),
        "--context-qa-modality-bias",
        str(args.context_qa_modality_bias),
        "--router-model",
        str(args.router_model),
    ]
    if args.planner_model:
        cmd.extend(["--planner-model", str(args.planner_model)])
    if args.verifier_model:
        cmd.extend(["--verifier-model", str(args.verifier_model)])
    cmd.extend(
        [
        "--innovation-scheme2",
        "--scheme2-cross-modal-weight",
        str(args.scheme2_cross_modal_weight),
        "--scheme2-token-maxsim-weight",
        str(args.scheme2_token_maxsim_weight),
        "--schemeb-heavy-mode",
        "--heavy-table-backend",
        str(args.heavy_table_backend),
        "--heavy-table-encoder-weight",
        str(args.heavy_table_encoder_weight),
        "--heavy-kg-path-weight",
        str(args.heavy_kg_path_weight),
        "--heavy-token-late-weight",
        str(args.heavy_token_late_weight),
        "--heavy-token-cross-modal-weight",
        str(args.heavy_token_cross_modal_weight),
        "--heavy-branch-candidate-expand-k",
        str(args.heavy_branch_candidate_expand_k),
        "--heavy-branch-candidate-table-weight",
        str(args.heavy_branch_candidate_table_weight),
        "--heavy-branch-candidate-kg-weight",
        str(args.heavy_branch_candidate_kg_weight),
        "--heavy-branch-candidate-max-total",
        str(args.heavy_branch_candidate_max_total),
        "--heavy-kg-hard-negative-mode",
        str(args.heavy_kg_hard_negative_mode),
        "--heavy-kg-hard-negative-topdocs",
        str(args.heavy_kg_hard_negative_topdocs),
        "--heavy-kg-hard-negative-max-paths",
        str(args.heavy_kg_hard_negative_max_paths),
        "--query-modality-prior-mix",
        str(args.query_modality_prior_mix),
        "--query-modality-prior-entropy-scale",
        str(args.query_modality_prior_entropy_scale),
        "--query-modality-prior-disagreement-scale",
        str(args.query_modality_prior_disagreement_scale),
        "--query-modality-prior-min",
        str(args.query_modality_prior_min),
        "--query-modality-prior-max",
        str(args.query_modality_prior_max),
        "--qa-objective-retrieval-weight",
        str(args.qa_objective_retrieval_weight),
        "--heavy-score-calibration",
        str(job.mode),
        "--heavy-score-calibration-nonzero-only",
    ])

    if bool(args.query_modality_prior_adaptive):
        cmd.append("--query-modality-prior-adaptive")

    if bool(args.qa_objective_targeted_only):
        cmd.append("--qa-objective-targeted-only")

    if str(args.heavy_table_backend).lower().strip() == "tapas":
        if not args.heavy_table_tapas_model:
            raise ValueError("heavy_table_tapas_model is required when heavy_table_backend=tapas")
        cmd.extend(
            [
                "--heavy-table-tapas-model",
                str(args.heavy_table_tapas_model),
                "--heavy-table-tapas-required",
            ]
        )

    if bool(args.heavy_remove_hard_caps):
        cmd.append("--heavy-remove-hard-caps")

    if args.reader == "ollama":
        cmd.extend(["--ollama-host", str(args.ollama_host), "--ollama-model", str(args.ollama_model)])
    elif args.reader == "openai":
        cmd.extend(
            [
                "--openai-model",
                str(args.openai_model),
                "--openai-api-key-env",
                str(args.openai_api_key_env),
                "--openai-base-url",
                str(args.openai_base_url),
                "--openai-timeout",
                str(args.openai_timeout),
                "--openai-temperature",
                str(args.openai_temperature),
                "--openai-max-tokens",
                str(args.openai_max_tokens),
            ]
        )

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(job.gpu)

    proc = subprocess.run(
        cmd,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
    )

    log_file = job.out_dir / "sweep_run.log"
    log_file.write_text(proc.stdout, encoding="utf-8")
    return job.mode, int(proc.returncode), str(log_file)


def main() -> int:
    parser = argparse.ArgumentParser(description="Run resource-efficient SchemeB calibration sweep for main method only")
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--split-file", type=Path, required=True)
    parser.add_argument("--corpus-file", type=Path, required=True)
    parser.add_argument("--router-model", type=Path, required=True)
    parser.add_argument("--planner-model", type=Path, default=None)
    parser.add_argument("--verifier-model", type=Path, default=None)
    parser.add_argument("--out-root", type=Path, required=True)

    parser.add_argument("--modes", type=str, default="none,minmax,robust,rank")
    parser.add_argument("--gpus", type=str, default="0,1")
    parser.add_argument("--parallel-workers", type=int, default=2)
    parser.add_argument("--max-queries", type=int, default=120)
    parser.add_argument("--retrieve-topk", type=int, default=20)
    parser.add_argument("--qa-context-k", type=int, default=6)
    parser.add_argument("--reader", type=str, choices=["ollama", "openai", "extractive"], default="ollama")
    parser.add_argument("--ollama-host", type=str, default="http://127.0.0.1:11434")
    parser.add_argument("--ollama-model", type=str, default="qwen2.5-7B-Instruct:latest")
    parser.add_argument("--openai-model", type=str, default="gpt-4o-mini")
    parser.add_argument("--openai-api-key-env", type=str, default="OPENAI_API_KEY")
    parser.add_argument("--openai-base-url", type=str, default="")
    parser.add_argument("--openai-timeout", type=int, default=120)
    parser.add_argument("--openai-temperature", type=float, default=0.0)
    parser.add_argument("--openai-max-tokens", type=int, default=64)

    parser.add_argument("--baseline-dense-pred-file", type=Path, default=None)
    parser.add_argument("--skip-copy-if-exists", action="store_true")

    parser.add_argument("--preserve-dense-top", type=int, default=0)
    parser.add_argument("--tessera-late-alpha", type=float, default=0.08)
    parser.add_argument("--context-active-threshold", type=float, default=0.40)
    parser.add_argument("--context-anchor-dense-k", type=int, default=4)
    parser.add_argument("--context-anchor-uni-k", type=int, default=1)
    parser.add_argument("--context-dense-pool-k", type=int, default=20)
    parser.add_argument("--context-candidate-expand-k", type=int, default=0)
    parser.add_argument("--context-redundancy-lambda", type=float, default=0.08)
    parser.add_argument("--pathmaxsim-weight", type=float, default=0.14)
    parser.add_argument("--pathmaxsim-kg-threshold", type=float, default=0.0)
    parser.add_argument("--table-cellmaxsim-weight", type=float, default=0.02)
    parser.add_argument("--table-cellmaxsim-top-cells", type=int, default=160)
    parser.add_argument("--context-consistency-weight", type=float, default=0.0)
    parser.add_argument("--context-qa-objective-weight", type=float, default=0.0)
    parser.add_argument("--context-qa-modality-bias", type=float, default=0.0)

    parser.add_argument("--scheme2-cross-modal-weight", type=float, default=0.12)
    parser.add_argument("--scheme2-token-maxsim-weight", type=float, default=0.02)

    parser.add_argument("--heavy-table-backend", type=str, choices=["hash", "tapas"], default="tapas")
    parser.add_argument("--heavy-table-tapas-model", type=Path, default=Path(""))
    parser.add_argument("--heavy-table-encoder-weight", type=float, default=0.10)
    parser.add_argument("--heavy-kg-path-weight", type=float, default=0.06)
    parser.add_argument("--heavy-token-late-weight", type=float, default=0.08)
    parser.add_argument("--heavy-token-cross-modal-weight", type=float, default=0.02)
    parser.add_argument("--heavy-branch-candidate-expand-k", type=int, default=64)
    parser.add_argument("--heavy-branch-candidate-table-weight", type=float, default=0.55)
    parser.add_argument("--heavy-branch-candidate-kg-weight", type=float, default=0.55)
    parser.add_argument("--heavy-branch-candidate-max-total", type=int, default=1600)
    parser.add_argument("--heavy-kg-hard-negative-mode", type=str, default="cross_doc_hard")
    parser.add_argument("--heavy-kg-hard-negative-topdocs", type=int, default=3)
    parser.add_argument("--heavy-kg-hard-negative-max-paths", type=int, default=24)
    parser.add_argument("--query-modality-prior-mix", type=float, default=0.35)
    parser.add_argument("--query-modality-prior-adaptive", action="store_true")
    parser.add_argument("--query-modality-prior-entropy-scale", type=float, default=0.30)
    parser.add_argument("--query-modality-prior-disagreement-scale", type=float, default=0.25)
    parser.add_argument("--query-modality-prior-min", type=float, default=0.0)
    parser.add_argument("--query-modality-prior-max", type=float, default=0.85)
    parser.add_argument("--qa-objective-retrieval-weight", type=float, default=0.04)
    parser.add_argument("--qa-objective-targeted-only", dest="qa_objective_targeted_only", action="store_true")
    parser.add_argument("--qa-objective-global", dest="qa_objective_targeted_only", action="store_false")
    parser.add_argument("--heavy-remove-hard-caps", dest="heavy_remove_hard_caps", action="store_true")
    parser.add_argument("--heavy-keep-hard-caps", dest="heavy_remove_hard_caps", action="store_false")
    parser.set_defaults(heavy_remove_hard_caps=True)
    parser.set_defaults(qa_objective_targeted_only=False)

    args = parser.parse_args()

    root = Path(__file__).resolve().parents[2]
    eval_script = root / "scripts" / "eval" / "run_e2e_table1c.py"
    if not eval_script.exists():
        raise FileNotFoundError(f"Missing eval script: {eval_script}")

    modes = parse_modes(args.modes)
    if not modes:
        raise ValueError("No valid modes provided")
    gpus = parse_gpus(args.gpus)
    workers = max(1, min(int(args.parallel_workers), len(modes), len(gpus)))

    args.out_root.mkdir(parents=True, exist_ok=True)

    jobs: list[SweepJob] = []
    for idx, mode in enumerate(modes):
        out_dir = args.out_root / f"calib_{mode}"
        jobs.append(SweepJob(mode=mode, out_dir=out_dir, gpu=gpus[idx % len(gpus)]))

    print(f"[config] modes={modes} gpus={gpus} workers={workers} max_queries={args.max_queries}")

    results = []
    if workers == 1:
        for job in jobs:
            mode, code, log_file = run_job(job, args, eval_script)
            print(f"[run] mode={mode} code={code} log={log_file}")
            results.append((mode, code, log_file))
    else:
        with ThreadPoolExecutor(max_workers=workers) as ex:
            futs = [ex.submit(run_job, job, args, eval_script) for job in jobs]
            for fut in as_completed(futs):
                mode, code, log_file = fut.result()
                print(f"[run] mode={mode} code={code} log={log_file}")
                results.append((mode, code, log_file))

    failed = [x for x in results if x[1] != 0]
    if failed:
        fail_json = args.out_root / "sweep_failures.json"
        fail_json.write_text(
            json.dumps(
                [
                    {"mode": mode, "code": code, "log": log_file}
                    for mode, code, log_file in failed
                ],
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        print(f"[fail] Some runs failed. details={fail_json}")

    rows = []
    for mode in modes:
        metrics_path = args.out_root / f"calib_{mode}" / "table1c_e2e_metrics.json"
        data = load_metrics(metrics_path)
        methods = data.get("methods", {})
        meta = data.get("meta", {})
        uni = methods.get("tessera_rag", {})
        dense = methods.get("dense_concat", {})
        dbg = meta.get("schemeb_debug_summary", {})
        rows.append(
            {
                "mode": mode,
                "metrics_path": str(metrics_path),
                "tessera_f1": float(uni.get("f1", 0.0)),
                "tessera_em": float(uni.get("exact_match", 0.0)),
                "tessera_r10": float(uni.get("recall@10", 0.0)),
                "dense_f1": float(dense.get("f1", 0.0)),
                "tapas_enabled": bool(meta.get("heavy_table_tapas_enabled", False)),
                "tapas_query_ready_mean": float(dbg.get("tapas_query_ready", {}).get("mean", 0.0)),
                "heavy_table_w_mean": float(dbg.get("heavy_table_weight_eff", {}).get("mean", 0.0)),
                "heavy_kg_w_mean": float(dbg.get("heavy_kg_path_weight_eff", {}).get("mean", 0.0)),
                "heavy_token_late_w_mean": float(dbg.get("heavy_token_late_weight_eff", {}).get("mean", 0.0)),
            }
        )

    rows.sort(key=lambda x: x["tessera_f1"], reverse=True)

    summary_json = args.out_root / "sweep_summary.json"
    summary_md = args.out_root / "sweep_summary.md"
    summary_json.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
    summary_md.write_text(render_summary_md(rows, summary_json), encoding="utf-8")

    if rows:
        print(
            "[best] "
            f"mode={rows[0]['mode']} "
            f"tessera_f1={rows[0]['tessera_f1']:.4f} "
            f"tessera_em={rows[0]['tessera_em']:.4f}"
        )
    print(f"[OK] summary_json={summary_json}")
    print(f"[OK] summary_md={summary_md}")
    return 0 if not failed else 2


if __name__ == "__main__":
    raise SystemExit(main())
