#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from pathlib import Path

import ijson
import requests

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from tessera_exp.e2e.baselines import source_bucket  # noqa: E402
from tessera_exp.e2e.evaluation import evaluate_predictions, write_predictions_jsonl  # noqa: E402
from tessera_exp.e2e.metrics import exact_match, f1_score, mmrag_official_generation_score  # noqa: E402


def load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def load_context_docs(path: Path) -> dict[str, list[str]]:
    out: dict[str, list[str]] = {}
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            qid = str(row.get("id", row.get("query_id", "")))
            docs = row.get("context_doc_ids", row.get("docs", row.get("doc_ids", [])))
            out[qid] = [str(doc_id) for doc_id in docs]
    return out


def load_existing_predictions(path: Path) -> list[str]:
    if not path.exists():
        return []
    preds: list[str] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            preds.append(str(row.get("prediction", row.get("pred", row.get("answer", "")))))
    return preds


def write_context_docs_jsonl(path: Path, rows: list[dict], context_docs: list[list[str]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row, docs in zip(rows, context_docs):
            handle.write(json.dumps({"id": row.get("id"), "context_doc_ids": docs}, ensure_ascii=False) + "\n")


def load_corpus_texts(corpus_file: Path, needed_ids: set[str]) -> dict[str, str]:
    texts: dict[str, str] = {}
    if not needed_ids:
        return texts
    with corpus_file.open("rb") as handle:
        for row in ijson.items(handle, "item"):
            doc_id = str(row.get("id", ""))
            if doc_id in needed_ids:
                texts[doc_id] = str(row.get("text", ""))
                if len(texts) == len(needed_ids):
                    break
    return texts


def ollama_reader(
    host: str,
    model: str,
    query: str,
    contexts: list[str],
    timeout_s: int,
    temperature: float,
    num_predict: int,
    max_retries: int,
    retry_backoff_s: float,
    fail_soft: bool,
) -> str:
    context_block = "\n\n".join(f"[doc{i + 1}] {text[:1200]}" for i, text in enumerate(contexts[:6]))
    prompt = (
        "Answer questions using only the provided evidence. "
        "Return a short answer phrase only, no explanation.\n\n"
        f"Question: {query}\n\nEvidence:\n{context_block}\n\nShort Answer:"
    )
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": float(temperature),
            "num_predict": int(num_predict),
        },
    }
    url = str(host).rstrip("/") + "/api/generate"
    last_err: Exception | None = None
    retries = max(1, int(max_retries))
    session = requests.Session()
    session.trust_env = False
    for attempt in range(retries):
        try:
            resp = session.post(url, json=payload, timeout=int(timeout_s))
            resp.raise_for_status()
            data = resp.json()
            out = str(data.get("response", "")).strip()
            out = re.sub(r"<think>.*?</think>", "", out, flags=re.DOTALL).strip()
            out = out.replace("\n", " ").strip()
            return out[:200]
        except Exception as exc:
            last_err = exc
            print(f"[ollama] error on attempt {attempt + 1}/{retries}: {exc}", flush=True)
            time.sleep(max(0.0, float(retry_backoff_s)) * float(attempt + 1))
    if fail_soft:
        print(f"[ollama] fail-soft: returning empty answer after {retries} failed attempts: {last_err}", flush=True)
        return ""
    raise RuntimeError(f"Ollama reader failed after {retries} attempts: {last_err}")


def openai_reader(
    model: str,
    query: str,
    contexts: list[str],
    timeout_s: int,
    temperature: float,
    max_tokens: int,
    base_url: str,
    api_key_env: str,
    max_retries: int,
    retry_backoff_s: float,
    fail_soft: bool,
) -> str:
    api_key = os.environ.get(api_key_env, "").strip()
    if not api_key:
        raise RuntimeError(f"[openai] missing API key env var: {api_key_env}")
    try:
        api_key.encode("ascii")
    except UnicodeEncodeError as exc:
        raise RuntimeError(
            f"[openai] {api_key_env} contains non-ASCII characters. "
            "Re-export the real API key; do not use the placeholder text."
        ) from exc
    try:
        from openai import OpenAI
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("OpenAI Python SDK is required for --reader openai. Install with: pip install openai") from exc

    context_block = "\n\n".join(f"[doc{i + 1}] {text[:1200]}" for i, text in enumerate(contexts[:6]))
    messages = [
        {
            "role": "system",
            "content": "Answer questions using only the provided evidence. Return a short answer phrase only, no explanation.",
        },
        {
            "role": "user",
            "content": f"Question: {query}\n\nEvidence:\n{context_block}\n\nShort Answer:",
        },
    ]

    client_kwargs = {"api_key": api_key, "timeout": int(timeout_s)}
    if str(base_url).strip():
        client_kwargs["base_url"] = str(base_url).strip()
    client = OpenAI(**client_kwargs)

    last_err: Exception | None = None
    retries = max(1, int(max_retries))
    for attempt in range(retries):
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=float(temperature),
                max_tokens=int(max_tokens),
            )
            out = ""
            if resp.choices:
                out = (resp.choices[0].message.content or "").strip()
            out = re.sub(r"<think>.*?</think>", "", out, flags=re.DOTALL).strip()
            out = out.replace("\n", " ").strip()
            return out[:200]
        except Exception as exc:
            last_err = exc
            print(f"[openai] error on attempt {attempt + 1}/{retries}: {exc}", flush=True)
            time.sleep(max(0.0, float(retry_backoff_s)) * float(attempt + 1))
    if fail_soft:
        print(f"[openai] fail-soft: returning empty answer after {retries} failed attempts: {last_err}", flush=True)
        return ""
    raise RuntimeError(f"OpenAI reader failed after {retries} attempts: {last_err}")


def build_markdown(summary: dict) -> str:
    metric = summary["metrics"]
    lines = [
        f"# Reader Metrics From Context Docs: {summary['meta']['method_label']}",
        "",
        f"- context_docs: {summary['meta']['context_docs']}",
        f"- selected_queries: {summary['meta']['selected_queries']}",
        f"- qa_context_k: {summary['meta']['qa_context_k']}",
        f"- reader: {summary['meta']['reader']}",
        f"- ollama_model: {summary['meta'].get('ollama_model')}",
        f"- openai_model: {summary['meta'].get('openai_model')}",
        f"- missing_context_doc_ids: {summary['meta']['missing_context_doc_ids']}",
        "",
        "| Metric | Value |",
        "|---|---:|",
        f"| F1 | {metric['f1']:.4f} |",
        f"| Exact Match | {metric['exact_match']:.4f} |",
        f"| mmRAG Official Avg | {metric['mmrag_official_avg']:.4f} |",
        f"| Recall@10 | {metric['recall@10']:.4f} |",
        "",
        "## Slice Metrics",
        "",
        "| Slice | Count | F1 | EM | Recall@10 | Official |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for name, vals in metric.get("slice_metrics", {}).items():
        lines.append(
            f"| {name} | {int(vals['count'])} | {vals['f1']:.4f} | {vals['exact_match']:.4f} | "
            f"{vals['recall@10']:.4f} | {vals['mmrag_official_avg']:.4f} |"
        )
    lines.append("")
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description="Run an Ollama/OpenAI-compatible reader from saved context_docs_*.jsonl.")
    parser.add_argument("--split-file", type=Path, required=True)
    parser.add_argument("--corpus-file", type=Path, required=True)
    parser.add_argument("--context-docs", type=Path, required=True)
    parser.add_argument("--method-label", type=str, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--max-queries", type=int, default=0)
    parser.add_argument("--qa-context-k", type=int, default=6)
    parser.add_argument("--reader", choices=["ollama", "openai"], default="ollama")
    parser.add_argument("--ollama-host", type=str, default="http://127.0.0.1:11434")
    parser.add_argument("--ollama-model", type=str, default="llama3.3:latest")
    parser.add_argument("--ollama-timeout", type=int, default=600)
    parser.add_argument("--ollama-temperature", type=float, default=0.0)
    parser.add_argument("--ollama-num-predict", type=int, default=64)
    parser.add_argument("--ollama-max-retries", type=int, default=3)
    parser.add_argument("--ollama-retry-backoff", type=float, default=5.0)
    parser.add_argument("--ollama-fail-soft", action="store_true")
    parser.add_argument("--openai-model", type=str, default="llama-3.3-70b")
    parser.add_argument("--openai-api-key-env", type=str, default="OPENAI_API_KEY")
    parser.add_argument("--openai-base-url", type=str, default="")
    parser.add_argument("--openai-timeout", type=int, default=180)
    parser.add_argument("--openai-temperature", type=float, default=0.0)
    parser.add_argument("--openai-max-tokens", type=int, default=64)
    parser.add_argument("--openai-max-retries", type=int, default=5)
    parser.add_argument("--openai-retry-backoff", type=float, default=3.0)
    parser.add_argument("--openai-fail-soft", action="store_true")
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()

    rows_all = load_json(args.split_file)
    context_map = load_context_docs(args.context_docs)
    rows = [row for row in rows_all if str(row.get("id")) in context_map]
    if args.max_queries and args.max_queries > 0:
        rows = rows[: args.max_queries]
    if not rows:
        raise SystemExit("No rows overlap split-file and context-docs")

    context_docs = [context_map[str(row.get("id"))][: args.qa_context_k] for row in rows]
    top10_lists = [context_map[str(row.get("id"))][:10] for row in rows]
    needed_ids = {doc_id for docs in context_docs for doc_id in docs}
    print(
        f"[load] selected_rows={len(rows)} needed_context_doc_ids={len(needed_ids)} "
        f"context_docs={args.context_docs}",
        flush=True,
    )
    corpus_texts = load_corpus_texts(args.corpus_file, needed_ids)
    missing_ids = sorted(needed_ids - set(corpus_texts))

    args.out_dir.mkdir(parents=True, exist_ok=True)
    pred_path = args.out_dir / f"qa_predictions_{args.method_label}_test1286.jsonl"
    ctx_path = args.out_dir / f"context_docs_{args.method_label}_test1286.jsonl"
    existing_preds = load_existing_predictions(pred_path) if args.resume else []
    if len(existing_preds) > len(rows):
        existing_preds = existing_preds[: len(rows)]

    preds = list(existing_preds)
    start = len(preds)
    print(
        f"[config] rows={len(rows)} resume_from={start} context_k={args.qa_context_k} "
        f"missing_context_doc_ids={len(missing_ids)} reader={args.reader} "
        f"model={args.openai_model if args.reader == 'openai' else args.ollama_model}",
        flush=True,
    )
    if missing_ids:
        print(f"[warn] first missing context doc ids: {missing_ids[:10]}", flush=True)

    with pred_path.open("a" if args.resume else "w", encoding="utf-8") as pred_handle:
        if not args.resume:
            preds = []
        for idx in range(start, len(rows)):
            row = rows[idx]
            qid = str(row.get("id"))
            query = str(row.get("query", "")).strip()
            contexts = [corpus_texts[doc_id] for doc_id in context_docs[idx] if doc_id in corpus_texts]
            started = time.monotonic()
            if args.reader == "openai":
                pred = openai_reader(
                    model=args.openai_model,
                    query=query,
                    contexts=contexts,
                    timeout_s=args.openai_timeout,
                    temperature=args.openai_temperature,
                    max_tokens=args.openai_max_tokens,
                    base_url=args.openai_base_url,
                    api_key_env=args.openai_api_key_env,
                    max_retries=args.openai_max_retries,
                    retry_backoff_s=args.openai_retry_backoff,
                    fail_soft=args.openai_fail_soft,
                )
            else:
                pred = ollama_reader(
                    host=args.ollama_host,
                    model=args.ollama_model,
                    query=query,
                    contexts=contexts,
                    timeout_s=args.ollama_timeout,
                    temperature=args.ollama_temperature,
                    num_predict=args.ollama_num_predict,
                    max_retries=args.ollama_max_retries,
                    retry_backoff_s=args.ollama_retry_backoff,
                    fail_soft=args.ollama_fail_soft,
                )
            preds.append(pred)
            pred_handle.write(json.dumps({"id": qid, "prediction": pred}, ensure_ascii=False) + "\n")
            pred_handle.flush()
            if (idx + 1) % 25 == 0 or idx == len(rows) - 1:
                elapsed = time.monotonic() - started
                print(f"[progress] {idx + 1}/{len(rows)} last_qid={qid} last_s={elapsed:.2f}", flush=True)

    write_context_docs_jsonl(ctx_path, rows, context_docs)
    write_predictions_jsonl(pred_path, rows, preds)
    metrics = evaluate_predictions(
        rows=rows,
        preds=preds,
        top10_lists=top10_lists,
        exact_match_fn=exact_match,
        f1_score_fn=f1_score,
        mmrag_official_fn=mmrag_official_generation_score,
        source_bucket_fn=source_bucket,
    )
    summary = {
        "meta": {
            "split_file": str(args.split_file),
            "corpus_file": str(args.corpus_file),
            "context_docs": str(args.context_docs),
            "method_label": args.method_label,
            "selected_queries": len(rows),
            "qa_context_k": int(args.qa_context_k),
            "reader": args.reader,
            "ollama_host": args.ollama_host,
            "ollama_model": args.ollama_model if args.reader == "ollama" else None,
            "openai_model": args.openai_model if args.reader == "openai" else None,
            "openai_base_url": args.openai_base_url if args.reader == "openai" else None,
            "missing_context_doc_ids": len(missing_ids),
            "prediction_file": str(pred_path),
            "context_docs_file": str(ctx_path),
        },
        "metrics": metrics,
    }
    json_path = args.out_dir / "table1c_e2e_metrics_from_context_docs.json"
    md_path = args.out_dir / "table1c_e2e_metrics_from_context_docs.md"
    json_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    md_path.write_text(build_markdown(summary), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    print(f"[OK] predictions -> {pred_path}", flush=True)
    print(f"[OK] metrics json -> {json_path}", flush=True)
    print(f"[OK] metrics md -> {md_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
