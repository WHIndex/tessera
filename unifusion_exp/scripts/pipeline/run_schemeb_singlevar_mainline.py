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
    value: str
    out_dir: Path
    gpu: str


def parse_csv(raw: str) -> list[str]:
    vals = [x.strip() for x in str(raw).split(",") if x.strip()]
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


def render_summary_md(rows: list[dict], out_json: Path, variable: str) -> str:
    md = [
        "# SchemeB Single-Variable Mainline Sweep",
        "",
        f"- variable: {variable}",
        f"- summary_json: {out_json}",
        "",
        "| value | unifusion_f1 | unifusion_em | unifusion_r10 | dense_f1 | text_only_f1 | table_only_f1 | kg_only_f1 | multi_modal_f1 | tapas_enabled |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for r in rows:
        md.append(
            f"| {r['value']} | {r['unifusion_f1']:.4f} | {r['unifusion_em']:.4f} | {r['unifusion_r10']:.4f} | {r['dense_f1']:.4f} | "
            f"{r['slice_text_f1']:.4f} | {r['slice_table_f1']:.4f} | {r['slice_kg_f1']:.4f} | {r['slice_multi_f1']:.4f} | {r['tapas_enabled']} |"
        )
    return "\n".join(md) + "\n"


def build_singlevar_override(variable: str, value: str) -> list[str]:
    v = variable.strip().lower()
    raw = value.strip()

    if v == "score_calibration":
        return ["--heavy-score-calibration", raw]
    if v == "candidate_expand_k":
        return ["--heavy-branch-candidate-expand-k", raw]
    if v == "candidate_max_total":
        return ["--heavy-branch-candidate-max-total", raw]
    if v == "qa_objective_weight":
        return ["--qa-objective-retrieval-weight", raw]
    if v == "context_qa_objective_weight":
        return ["--context-qa-objective-weight", raw]
    if v == "context_candidate_expand_k":
        return ["--context-candidate-expand-k", raw]
    if v == "context_dense_pool_k":
        return ["--context-dense-pool-k", raw]
    if v == "context_qa_modality_bias":
        return ["--context-qa-modality-bias", raw]
    if v == "context_light_rerank_weight":
        return ["--context-light-rerank-weight", raw]
    if v == "context_strong_rerank_endpoint":
        return ["--context-strong-rerank-endpoint", raw]
    if v == "context_strong_rerank_topn":
        return ["--context-strong-rerank-topn", raw]
    if v == "context_strong_rerank_timeout":
        return ["--context-strong-rerank-timeout", raw]
    if v == "query_modality_prior_mix":
        return ["--query-modality-prior-mix", raw]
    if v == "query_modality_prior_adaptive":
        if raw.lower() in {"1", "true", "yes", "on"}:
            return ["--query-modality-prior-adaptive"]
        if raw.lower() in {"0", "false", "no", "off"}:
            return []
        raise ValueError(f"Invalid boolean value for query_modality_prior_adaptive: {raw}")
    if v == "query_modality_prior_entropy_scale":
        return ["--query-modality-prior-entropy-scale", raw]
    if v == "query_modality_prior_disagreement_scale":
        return ["--query-modality-prior-disagreement-scale", raw]
    if v == "query_modality_prior_min":
        return ["--query-modality-prior-min", raw]
    if v == "query_modality_prior_max":
        return ["--query-modality-prior-max", raw]
    if v == "kg_hard_negative_mode":
        return ["--heavy-kg-hard-negative-mode", raw]
    if v == "kg_hard_negative_topdocs":
        return ["--heavy-kg-hard-negative-topdocs", raw]
    if v == "kg_hard_negative_max_paths":
        return ["--heavy-kg-hard-negative-max-paths", raw]
    if v == "remove_hard_caps":
        if raw.lower() in {"1", "true", "yes", "on"}:
            return ["--heavy-remove-hard-caps"]
        if raw.lower() in {"0", "false", "no", "off"}:
            return []
        raise ValueError(f"Invalid boolean value for remove_hard_caps: {raw}")

    raise ValueError(
        f"Unsupported variable '{variable}'. Supported: "
        "score_calibration,candidate_expand_k,candidate_max_total,qa_objective_weight,"
        "context_qa_objective_weight,context_qa_modality_bias,context_light_rerank_weight,context_strong_rerank_endpoint,"
        "context_strong_rerank_topn,context_strong_rerank_timeout,context_candidate_expand_k,context_dense_pool_k,"
        "query_modality_prior_mix,query_modality_prior_adaptive,"
        "query_modality_prior_entropy_scale,query_modality_prior_disagreement_scale,"
        "query_modality_prior_min,query_modality_prior_max,"
        "kg_hard_negative_mode,kg_hard_negative_topdocs,kg_hard_negative_max_paths,remove_hard_caps"
    )


def run_job(job: SweepJob, args: argparse.Namespace, eval_script: Path) -> tuple[str, int, str]:
    job.out_dir.mkdir(parents=True, exist_ok=True)

    dense_target = job.out_dir / "qa_predictions_dense_concat_test1286.jsonl"
    if args.baseline_dense_pred_file and Path(args.baseline_dense_pred_file).exists():
        if not dense_target.exists() or not args.skip_copy_if_exists:
            shutil.copy2(Path(args.baseline_dense_pred_file), dense_target)

    cmd = [
        sys.executable,
        "-u",
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
        "--progress-every",
        str(args.progress_every),
        "--progress-min-seconds",
        str(args.progress_min_seconds),
        "--retrieve-topk",
        str(args.retrieve_topk),
        "--qa-context-k",
        str(args.qa_context_k),
        "--reader",
        str(args.reader),
        "--preserve-dense-top",
        str(args.preserve_dense_top),
        "--unifusion-late-alpha",
        str(args.unifusion_late_alpha),
        "--method-preset",
        "main_only",
        "--methods",
        "dense_concat,unifusion_rag",
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
            str(args.heavy_score_calibration),
            "--heavy-score-calibration-nonzero-only",
        ]
    )

    if bool(args.query_modality_prior_adaptive):
        cmd.append("--query-modality-prior-adaptive")

    if bool(args.qa_objective_targeted_only):
        cmd.append("--qa-objective-targeted-only")

    if bool(args.heavy_remove_hard_caps):
        cmd.append("--heavy-remove-hard-caps")

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

    swept = str(args.variable).strip().lower()

    def drop_flag_with_value(flag: str) -> None:
        while flag in cmd:
            i = cmd.index(flag)
            if i + 1 < len(cmd):
                del cmd[i : i + 2]
            else:
                del cmd[i]

    def drop_flag_only(flag: str) -> None:
        while flag in cmd:
            cmd.remove(flag)

    if swept == "query_modality_prior_mix":
        drop_flag_with_value("--query-modality-prior-mix")
    elif swept == "context_qa_objective_weight":
        drop_flag_with_value("--context-qa-objective-weight")
    elif swept == "context_qa_modality_bias":
        drop_flag_with_value("--context-qa-modality-bias")
    elif swept == "context_light_rerank_weight":
        drop_flag_with_value("--context-light-rerank-weight")
    elif swept == "context_strong_rerank_endpoint":
        drop_flag_with_value("--context-strong-rerank-endpoint")
    elif swept == "context_strong_rerank_topn":
        drop_flag_with_value("--context-strong-rerank-topn")
    elif swept == "context_strong_rerank_timeout":
        drop_flag_with_value("--context-strong-rerank-timeout")
    elif swept == "context_candidate_expand_k":
        drop_flag_with_value("--context-candidate-expand-k")
    elif swept == "context_dense_pool_k":
        drop_flag_with_value("--context-dense-pool-k")
    elif swept == "query_modality_prior_adaptive":
        drop_flag_only("--query-modality-prior-adaptive")
    elif swept == "query_modality_prior_entropy_scale":
        drop_flag_with_value("--query-modality-prior-entropy-scale")
    elif swept == "query_modality_prior_disagreement_scale":
        drop_flag_with_value("--query-modality-prior-disagreement-scale")
    elif swept == "query_modality_prior_min":
        drop_flag_with_value("--query-modality-prior-min")
    elif swept == "query_modality_prior_max":
        drop_flag_with_value("--query-modality-prior-max")
    elif swept == "qa_objective_weight":
        drop_flag_with_value("--qa-objective-retrieval-weight")
    elif swept == "kg_hard_negative_mode":
        drop_flag_with_value("--heavy-kg-hard-negative-mode")
    elif swept == "kg_hard_negative_topdocs":
        drop_flag_with_value("--heavy-kg-hard-negative-topdocs")
    elif swept == "kg_hard_negative_max_paths":
        drop_flag_with_value("--heavy-kg-hard-negative-max-paths")
    elif swept == "candidate_expand_k":
        drop_flag_with_value("--heavy-branch-candidate-expand-k")
    elif swept == "candidate_max_total":
        drop_flag_with_value("--heavy-branch-candidate-max-total")
    elif swept == "score_calibration":
        drop_flag_with_value("--heavy-score-calibration")
    elif swept == "remove_hard_caps":
        drop_flag_only("--heavy-remove-hard-caps")

    cmd.extend(build_singlevar_override(args.variable, job.value))

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
    env["PYTHONUNBUFFERED"] = "1"

    log_file = job.out_dir / "singlevar_run.log"
    log_file.parent.mkdir(parents=True, exist_ok=True)
    with log_file.open("w", encoding="utf-8") as log_handle:
        proc = subprocess.Popen(
            cmd,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            bufsize=1,
        )
        assert proc.stdout is not None
        prefix = f"[job value={job.value} gpu={job.gpu}] "
        for line in proc.stdout:
            log_handle.write(line)
            log_handle.flush()
            print(prefix + line.rstrip("\n"), flush=True)
        proc.wait()

    return job.value, int(proc.returncode), str(log_file)


def main() -> int:
    parser = argparse.ArgumentParser(description="Run single-variable mainline sweeps for SchemeB (dense + unifusion only)")
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--split-file", type=Path, required=True)
    parser.add_argument("--corpus-file", type=Path, required=True)
    parser.add_argument("--router-model", type=Path, required=True)
    parser.add_argument("--planner-model", type=Path, default=None)
    parser.add_argument("--verifier-model", type=Path, default=None)
    parser.add_argument("--out-root", type=Path, required=True)

    parser.add_argument("--variable", type=str, required=True)
    parser.add_argument("--values", type=str, required=True)

    parser.add_argument("--gpus", type=str, default="0,1")
    parser.add_argument("--parallel-workers", type=int, default=2)
    parser.add_argument("--max-queries", type=int, default=200)
    parser.add_argument("--progress-every", type=int, default=5)
    parser.add_argument("--progress-min-seconds", type=float, default=10.0)
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
    parser.add_argument("--unifusion-late-alpha", type=float, default=0.08)
    parser.add_argument("--context-active-threshold", type=float, default=0.40)
    parser.add_argument("--context-anchor-dense-k", type=int, default=4)
    parser.add_argument("--context-anchor-uni-k", type=int, default=1)
    parser.add_argument("--context-dense-pool-k", type=int, default=20)
    parser.add_argument("--context-conflict-penalty-weight", type=float, default=0.0)
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
    parser.add_argument("--heavy-score-calibration", type=str, default="rank")
    parser.add_argument("--heavy-remove-hard-caps", dest="heavy_remove_hard_caps", action="store_true")
    parser.add_argument("--heavy-keep-hard-caps", dest="heavy_remove_hard_caps", action="store_false")
    parser.set_defaults(heavy_remove_hard_caps=True)
    parser.set_defaults(qa_objective_targeted_only=False)

    args = parser.parse_args()

    values = parse_csv(args.values)
    if not values:
        raise ValueError("No valid values provided")

    root = Path(__file__).resolve().parents[2]
    eval_script = root / "scripts" / "eval" / "run_e2e_table1c.py"
    if not eval_script.exists():
        raise FileNotFoundError(f"Missing eval script: {eval_script}")

    gpus = parse_gpus(args.gpus)
    workers = max(1, min(int(args.parallel_workers), len(values), len(gpus)))

    args.out_root.mkdir(parents=True, exist_ok=True)

    jobs: list[SweepJob] = []
    for idx, value in enumerate(values):
        safe = value.replace("/", "_").replace(" ", "_")
        out_dir = args.out_root / f"{args.variable}_{safe}"
        jobs.append(SweepJob(value=value, out_dir=out_dir, gpu=gpus[idx % len(gpus)]))

    print(f"[config] variable={args.variable} values={values} gpus={gpus} workers={workers} max_queries={args.max_queries}")

    results = []
    if workers == 1:
        for job in jobs:
            value, code, log_file = run_job(job, args, eval_script)
            print(f"[run] value={value} code={code} log={log_file}")
            results.append((value, code, log_file))
    else:
        with ThreadPoolExecutor(max_workers=workers) as ex:
            futs = [ex.submit(run_job, job, args, eval_script) for job in jobs]
            for fut in as_completed(futs):
                value, code, log_file = fut.result()
                print(f"[run] value={value} code={code} log={log_file}")
                results.append((value, code, log_file))

    failed = [x for x in results if x[1] != 0]
    if failed:
        fail_json = args.out_root / "singlevar_failures.json"
        fail_json.write_text(
            json.dumps(
                [{"value": value, "code": code, "log": log_file} for value, code, log_file in failed],
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        print(f"[fail] Some runs failed. details={fail_json}")

    rows = []
    for value in values:
        safe = value.replace("/", "_").replace(" ", "_")
        metrics_path = args.out_root / f"{args.variable}_{safe}" / "table1c_e2e_metrics.json"
        data = load_metrics(metrics_path)
        methods = data.get("methods", {})
        meta = data.get("meta", {})
        uni = methods.get("unifusion_rag", {})
        dense = methods.get("dense_concat", {})
        uni_slice = uni.get("slice_metrics", {})

        rows.append(
            {
                "value": value,
                "metrics_path": str(metrics_path),
                "unifusion_f1": float(uni.get("f1", 0.0)),
                "unifusion_em": float(uni.get("exact_match", 0.0)),
                "unifusion_r10": float(uni.get("recall@10", 0.0)),
                "dense_f1": float(dense.get("f1", 0.0)),
                "slice_text_f1": float(uni_slice.get("text_only", {}).get("f1", 0.0)),
                "slice_table_f1": float(uni_slice.get("table_only", {}).get("f1", 0.0)),
                "slice_kg_f1": float(uni_slice.get("kg_only", {}).get("f1", 0.0)),
                "slice_multi_f1": float(uni_slice.get("multi_modal", {}).get("f1", 0.0)),
                "tapas_enabled": bool(meta.get("heavy_table_tapas_enabled", False)),
            }
        )

    rows.sort(key=lambda x: x["unifusion_f1"], reverse=True)

    summary_json = args.out_root / "singlevar_summary.json"
    summary_md = args.out_root / "singlevar_summary.md"
    summary_json.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
    summary_md.write_text(render_summary_md(rows, summary_json, args.variable), encoding="utf-8")

    if rows:
        print(
            "[best] "
            f"value={rows[0]['value']} "
            f"unifusion_f1={rows[0]['unifusion_f1']:.4f} "
            f"unifusion_em={rows[0]['unifusion_em']:.4f}"
        )
    print(f"[OK] summary_json={summary_json}")
    print(f"[OK] summary_md={summary_md}")
    return 0 if not failed else 2


if __name__ == "__main__":
    raise SystemExit(main())
