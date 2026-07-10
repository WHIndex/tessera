#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import random
import re
import string
import sys
import time
from collections import Counter
from pathlib import Path

import ijson
import numpy as np
import pandas as pd
from deepeval.metrics import AnswerRelevancyMetric
from deepeval.models.llms.ollama_model import OllamaModel
from deepeval.test_case import LLMTestCase
from langchain_core.embeddings import Embeddings
from langchain_community.chat_models import ChatOllama
from ragas import evaluate
from ragas.dataset_schema import EvaluationDataset, SingleTurnSample
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.llms import LangchainLLMWrapper
from ragas.metrics import answer_relevancy, context_precision, context_recall, faithfulness

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from tessera_exp.utils.e5_embed import encode_texts, load_e5  # noqa: E402


def normalize_answer(text: str) -> str:
    text = text.lower()
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    text = "".join(ch for ch in text if ch not in set(string.punctuation))
    text = " ".join(text.split())
    return text


def exact_match(pred: str, gold: str) -> float:
    return 1.0 if normalize_answer(pred) == normalize_answer(gold) else 0.0


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


def load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def load_answers(path: Path) -> dict[str, str]:
    answers: dict[str, str] = {}
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            qid = row.get("id")
            if qid is None:
                continue
            pred = row.get("prediction", row.get("pred", row.get("answer", "")))
            answers[str(qid)] = str(pred)
    return answers


def load_retrieval_map(detail_file: Path, method: str) -> tuple[list[str], dict[str, list[str]]]:
    detail = load_json(detail_file)
    query_ids = [str(x) for x in detail.get("query_ids", [])]
    predictions = detail.get("predictions")
    if not isinstance(predictions, dict):
        raise ValueError(f"{detail_file} does not contain top-level predictions; rerun retrieval with --save-predictions")
    if method not in predictions:
        available = ", ".join(sorted(predictions.keys()))
        raise ValueError(f"retrieval method '{method}' not found in {detail_file}; available: {available}")

    method_preds = predictions[method]
    if len(query_ids) != len(method_preds):
        raise ValueError(
            f"query_ids/predictions length mismatch in {detail_file}: {len(query_ids)} vs {len(method_preds)}"
        )

    out: dict[str, list[str]] = {}
    for qid, docs in zip(query_ids, method_preds):
        out[qid] = [str(doc_id) for doc_id in docs]
    return query_ids, out


def collect_positive_chunk_ids(row: dict) -> set[str]:
    out: set[str] = set()
    for chunk_id, label in row.get("relevant_chunks", {}).items():
        try:
            if float(label) > 0:
                out.add(str(chunk_id))
        except Exception:
            continue
    return out


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


def select_rows(rows: list[dict], max_samples: int, seed: int) -> list[dict]:
    if max_samples <= 0 or max_samples >= len(rows):
        return rows
    rng = random.Random(seed)
    indices = sorted(rng.sample(range(len(rows)), max_samples))
    return [rows[i] for i in indices]


class CachedE5Embeddings(Embeddings):
    def __init__(self, model_dir: str, batch_size: int = 32):
        self.tokenizer, self.model, self.device, self.resolved_model_dir = load_e5(model_dir)
        self.batch_size = batch_size
        self._doc_cache: dict[str, list[float]] = {}
        self._query_cache: dict[str, list[float]] = {}

    def _embed_many(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        vecs = encode_texts(texts, self.tokenizer, self.model, self.device, batch_size=self.batch_size)
        return [vec.tolist() for vec in vecs]

    def _embed_with_cache(self, texts: list[str], cache: dict[str, list[float]]) -> list[list[float]]:
        out: list[list[float] | None] = [None] * len(texts)
        missing_texts: list[str] = []
        missing_positions: list[int] = []

        for idx, text in enumerate(texts):
            if text in cache:
                out[idx] = cache[text]
            else:
                missing_texts.append(text)
                missing_positions.append(idx)

        if missing_texts:
            vecs = self._embed_many(missing_texts)
            for text, vec in zip(missing_texts, vecs):
                cache[text] = vec
            for idx, text in zip(missing_positions, missing_texts):
                out[idx] = cache[text]

        return [x if x is not None else [] for x in out]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return self._embed_with_cache(texts, self._doc_cache)

    def embed_query(self, text: str) -> list[float]:
        if text not in self._query_cache:
            self._query_cache[text] = self._embed_many([text])[0]
        return self._query_cache[text]


def result_to_dataframe(result) -> pd.DataFrame:
    if hasattr(result, "to_pandas"):
        try:
            frame = result.to_pandas()
            if isinstance(frame, pd.DataFrame):
                return frame
        except Exception:
            pass

    for attr in ("scores", "results", "data"):
        value = getattr(result, attr, None)
        if value is None:
            continue
        if isinstance(value, pd.DataFrame):
            return value
        try:
            return pd.DataFrame(value)
        except Exception:
            continue

    try:
        return pd.DataFrame(list(result))
    except Exception as exc:
        raise TypeError(f"cannot convert ragas result to dataframe: {type(result)!r}") from exc


def find_column(frame: pd.DataFrame, candidates: list[str]) -> str | None:
    lower_map = {str(col).lower(): str(col) for col in frame.columns}
    for candidate in candidates:
        if candidate in frame.columns:
            return candidate
        lowered = candidate.lower()
        if lowered in lower_map:
            return lower_map[lowered]
    return None


def mean_or_none(frame: pd.DataFrame, candidates: list[str]) -> float | None:
    column = find_column(frame, candidates)
    if column is None:
        return None
    series = pd.to_numeric(frame[column], errors="coerce")
    if series.empty:
        return None
    value = float(series.mean())
    if not np.isfinite(value):
        return None
    return value


def deepeval_answer_relevancy_mean(
    samples: list[dict[str, str]],
    ollama_host: str,
    ollama_model: str,
    temperature: float,
) -> tuple[float | None, int, str | None]:
    if not samples:
        return None, 0, "no_samples"

    try:
        llm = OllamaModel(model=ollama_model, base_url=ollama_host, temperature=temperature)
    except Exception as exc:
        return None, 0, repr(exc)

    scores: list[float] = []
    failure_count = 0
    first_error: str | None = None

    for sample in samples:
        try:
            metric = AnswerRelevancyMetric(
                threshold=0.5,
                model=llm,
                include_reason=False,
                async_mode=False,
                verbose_mode=False,
            )
            test_case = LLMTestCase(
                input=sample["question"],
                actual_output=sample["response"],
            )
            score = metric.measure(test_case, _show_indicator=False)
            if score is None:
                failure_count += 1
                continue
            score_value = float(score)
            if np.isfinite(score_value):
                scores.append(score_value)
            else:
                failure_count += 1
        except Exception as exc:
            failure_count += 1
            if first_error is None:
                first_error = repr(exc)

    if not scores:
        return None, failure_count, first_error
    return float(np.mean(scores)), failure_count, first_error


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate mmRAG process metrics with RAGAS on retrieved contexts")
    parser.add_argument("--split-file", type=Path, required=True, help="mmrag_test.json / mmrag_dev.json")
    parser.add_argument("--corpus-file", type=Path, required=True, help="Retrieval corpus json used to resolve doc texts")
    parser.add_argument("--retrieval-detail", type=Path, required=True, help="detail json from eval_tessera_retrieval_main.py with --save-predictions")
    parser.add_argument("--retrieval-method", type=str, default="main_tessera", help="Key inside retrieval-detail predictions")
    parser.add_argument("--prediction-file", type=Path, required=True, help="qa_predictions_*.jsonl for answer text")
    parser.add_argument("--out-json", type=Path, required=True)
    parser.add_argument("--out-md", type=Path, required=True)
    parser.add_argument("--max-samples", type=int, default=0, help="Use at most N queries (0 means all)")
    parser.add_argument("--seed", type=int, default=20260421)
    parser.add_argument("--max-contexts", type=int, default=6, help="Number of retrieved contexts to keep per sample")
    parser.add_argument("--e5-model-dir", type=str, default=os.getenv("E5_MODEL_DIR", "/home/yongqi.yin/reaserch_paper/downloaded_resource/bge-large-en-v1.5"))
    parser.add_argument("--e5-batch-size", type=int, default=32)
    parser.add_argument("--ollama-host", type=str, default=os.getenv("OLLAMA_HOST", "http://127.0.0.1:11434"))
    parser.add_argument("--ollama-model", type=str, default=os.getenv("OLLAMA_MODEL", "qwen2.5-7B-Instruct:latest"))
    parser.add_argument("--deepeval-ollama-model", type=str, default=os.getenv("DEEPEVAL_OLLAMA_MODEL", "qwen3:32b"))
    parser.add_argument("--ollama-timeout", type=int, default=300)
    parser.add_argument("--ollama-num-predict", type=int, default=512)
    parser.add_argument("--ollama-temperature", type=float, default=0.0)
    args = parser.parse_args()

    rows = load_json(args.split_file)
    rows = select_rows(rows, args.max_samples, args.seed)
    answers = load_answers(args.prediction_file)
    query_ids, retrieval_map = load_retrieval_map(args.retrieval_detail, args.retrieval_method)
    query_id_set = set(query_ids)

    selected_rows = [row for row in rows if str(row.get("id")) in query_id_set and str(row.get("id")) in answers]
    if not selected_rows:
        raise SystemExit("No overlapping rows between split/predictions/retrieval detail")

    needed_doc_ids: set[str] = set()
    row_docs: dict[str, list[str]] = {}
    gold_context_ids: dict[str, set[str]] = {}
    deepeval_rows: list[dict[str, str]] = []

    for row in selected_rows:
        qid = str(row.get("id"))
        gold_ids = collect_positive_chunk_ids(row)
        gold_context_ids[qid] = gold_ids
        retrieved_ids = retrieval_map.get(qid, [])[: args.max_contexts]
        row_docs[qid] = retrieved_ids
        needed_doc_ids.update(gold_ids)
        needed_doc_ids.update(retrieved_ids)

    corpus_texts = load_corpus_texts(args.corpus_file, needed_doc_ids)
    missing_docs = sorted(needed_doc_ids - set(corpus_texts.keys()))
    if missing_docs:
        raise SystemExit(f"Missing {len(missing_docs)} corpus docs, first few: {missing_docs[:10]}")

    samples: list[SingleTurnSample] = []
    custom_rows: list[dict[str, object]] = []
    for row in selected_rows:
        qid = str(row.get("id"))
        question = str(row.get("query", "")).strip()
        reference = str(row.get("answer", "")).strip()
        response = str(answers.get(qid, "")).strip()
        retrieved_ids = row_docs.get(qid, [])
        reference_ids = sorted(gold_context_ids.get(qid, set()))
        retrieved_contexts = [corpus_texts[doc_id] for doc_id in retrieved_ids if doc_id in corpus_texts]
        reference_contexts = [corpus_texts[doc_id] for doc_id in reference_ids if doc_id in corpus_texts]

        if not question or not reference or not response:
            continue
        if not retrieved_contexts or not reference_contexts:
            continue

        deepeval_rows.append(
            {
                "id": qid,
                "question": question,
                "response": response,
            }
        )

        samples.append(
            SingleTurnSample(
                user_input=question,
                response=response,
                reference=reference,
                retrieved_contexts=retrieved_contexts,
                reference_contexts=reference_contexts,
            )
        )

        retrieved_id_set = set(retrieved_ids)
        reference_id_set = set(reference_ids)
        inter = len(retrieved_id_set & reference_id_set)
        custom_rows.append(
            {
                "id": qid,
                "answer_em": exact_match(response, reference),
                "answer_f1": f1_score(response, reference),
                "retrieved_context_count": len(retrieved_contexts),
                "reference_context_count": len(reference_contexts),
                "context_id_precision": inter / max(1, len(retrieved_id_set)),
                "context_id_recall": inter / max(1, len(reference_id_set)),
                "context_id_any_hit": 1.0 if inter > 0 else 0.0,
            }
        )

    if not samples:
        raise SystemExit("No valid samples after context filtering")

    dataset = EvaluationDataset(samples=samples)
    llm = LangchainLLMWrapper(
        ChatOllama(
            base_url=args.ollama_host,
            model=args.ollama_model,
            temperature=args.ollama_temperature,
            num_predict=args.ollama_num_predict,
            timeout=args.ollama_timeout,
        )
    )
    embeddings = LangchainEmbeddingsWrapper(CachedE5Embeddings(args.e5_model_dir, batch_size=args.e5_batch_size))

    metric_list = [answer_relevancy, faithfulness, context_precision, context_recall]
    started = time.monotonic()
    ragas_error = None
    ragas_frame = pd.DataFrame()
    try:
        result = evaluate(dataset, metrics=metric_list, llm=llm, embeddings=embeddings)
        ragas_frame = result_to_dataframe(result)
    except Exception as exc:
        ragas_error = repr(exc)

    elapsed = time.monotonic() - started

    metric_name_candidates = {
        "answer_relevancy": ["answer_relevancy", "AnswerRelevancy"],
        "faithfulness": ["faithfulness", "Faithfulness"],
        "context_precision": ["context_precision", "ContextPrecision"],
        "context_recall": ["context_recall", "ContextRecall"],
    }
    ragas_summary: dict[str, float | None] = {}
    if not ragas_frame.empty:
        for metric_name, candidates in metric_name_candidates.items():
            ragas_summary[metric_name] = mean_or_none(ragas_frame, candidates)
    else:
        for metric_name in metric_name_candidates:
            ragas_summary[metric_name] = None

    answer_relevancy_source = "ragas"
    answer_relevancy_fallback_error = None
    answer_relevancy_fallback_failures = 0
    if ragas_summary.get("answer_relevancy") is None:
        fallback_mean, fallback_failures, fallback_error = deepeval_answer_relevancy_mean(
            deepeval_rows,
            args.ollama_host,
            args.deepeval_ollama_model,
            args.ollama_temperature,
        )
        if fallback_mean is not None:
            ragas_summary["answer_relevancy"] = fallback_mean
            answer_relevancy_source = "deepeval_fallback"
            answer_relevancy_fallback_failures = fallback_failures
            answer_relevancy_fallback_error = fallback_error
        else:
            answer_relevancy_source = "missing"
            answer_relevancy_fallback_failures = fallback_failures
            answer_relevancy_fallback_error = fallback_error

    custom_df = pd.DataFrame(custom_rows)
    custom_summary = {
        "answer_em": float(custom_df["answer_em"].mean()) if not custom_df.empty else 0.0,
        "answer_f1": float(custom_df["answer_f1"].mean()) if not custom_df.empty else 0.0,
        "retrieved_context_count": float(custom_df["retrieved_context_count"].mean()) if not custom_df.empty else 0.0,
        "reference_context_count": float(custom_df["reference_context_count"].mean()) if not custom_df.empty else 0.0,
        "context_id_precision": float(custom_df["context_id_precision"].mean()) if not custom_df.empty else 0.0,
        "context_id_recall": float(custom_df["context_id_recall"].mean()) if not custom_df.empty else 0.0,
        "context_id_any_hit": float(custom_df["context_id_any_hit"].mean()) if not custom_df.empty else 0.0,
    }

    summary = {
        "meta": {
            "split_file": str(args.split_file),
            "corpus_file": str(args.corpus_file),
            "retrieval_detail": str(args.retrieval_detail),
            "retrieval_method": args.retrieval_method,
            "prediction_file": str(args.prediction_file),
            "selected_queries": len(samples),
            "max_samples": args.max_samples,
            "max_contexts": args.max_contexts,
            "seed": args.seed,
            "e5_model_dir": str(args.e5_model_dir),
            "ollama_host": args.ollama_host,
            "ollama_model": args.ollama_model,
            "deepeval_ollama_model": args.deepeval_ollama_model,
            "answer_relevancy_source": answer_relevancy_source,
            "answer_relevancy_fallback_failures": answer_relevancy_fallback_failures,
            "answer_relevancy_fallback_error": answer_relevancy_fallback_error,
            "elapsed_seconds": elapsed,
            "ragas_error": ragas_error,
        },
        "ragas": ragas_summary,
        "custom": custom_summary,
    }

    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    md_lines = [
        "# mmRAG Process Metrics",
        "",
        f"- selected_queries: {summary['meta']['selected_queries']}",
        f"- max_samples: {summary['meta']['max_samples']}",
        f"- max_contexts: {summary['meta']['max_contexts']}",
        f"- retrieval_method: {summary['meta']['retrieval_method']}",
        f"- prediction_file: {summary['meta']['prediction_file']}",
        f"- deepeval_ollama_model: {summary['meta']['deepeval_ollama_model']}",
        f"- answer_relevancy_source: {summary['meta']['answer_relevancy_source']}",
        f"- ragas_error: {summary['meta']['ragas_error']}",
        f"- elapsed_seconds: {summary['meta']['elapsed_seconds']:.2f}",
        "",
        "## Process Metrics",
        "",
        "| Metric | Mean |",
        "|---|---:|",
    ]
    for metric_name in ("answer_relevancy", "faithfulness", "context_precision", "context_recall"):
        value = ragas_summary.get(metric_name)
        md_lines.append(f"| {metric_name} | {value:.4f} |" if value is not None else f"| {metric_name} | n/a |")

    md_lines += [
        "",
        f"- answer_relevancy_source: {summary['meta']['answer_relevancy_source']}",
        f"- answer_relevancy_fallback_model: {summary['meta']['deepeval_ollama_model'] if summary['meta']['answer_relevancy_source'] == 'deepeval_fallback' else 'n/a'}",
        f"- answer_relevancy_fallback_failures: {summary['meta']['answer_relevancy_fallback_failures']}",
        f"- answer_relevancy_fallback_error: {summary['meta']['answer_relevancy_fallback_error']}",
    ]

    md_lines += [
        "",
        "## Custom Sanity",
        "",
        "| Metric | Mean |",
        "|---|---:|",
        f"| answer_em | {custom_summary['answer_em']:.4f} |",
        f"| answer_f1 | {custom_summary['answer_f1']:.4f} |",
        f"| retrieved_context_count | {custom_summary['retrieved_context_count']:.2f} |",
        f"| reference_context_count | {custom_summary['reference_context_count']:.2f} |",
        f"| context_id_precision | {custom_summary['context_id_precision']:.4f} |",
        f"| context_id_recall | {custom_summary['context_id_recall']:.4f} |",
        f"| context_id_any_hit | {custom_summary['context_id_any_hit']:.4f} |",
        "",
    ]
    args.out_md.parent.mkdir(parents=True, exist_ok=True)
    args.out_md.write_text("\n".join(md_lines), encoding="utf-8")

    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"[OK] saved -> {args.out_json}")
    print(f"[OK] saved -> {args.out_md}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())