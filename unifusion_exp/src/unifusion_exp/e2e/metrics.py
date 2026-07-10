from __future__ import annotations

import re
import string
from collections import Counter

import numpy as np

MMRAG_F1_DATASETS = {"nq", "tat", "cwq"}


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


def mmrag_parse_answer_list(value) -> list[str]:
    if isinstance(value, list):
        return [str(x) for x in value]
    try:
        parsed = eval(str(value))
        if isinstance(parsed, list):
            return [str(x) for x in parsed]
    except Exception:
        pass
    return [str(value)]


def mmrag_set_f1(gold_list: list[str], pred_list: list[str]) -> float:
    gold_set = set(gold_list)
    pred_set = set(pred_list)
    if not gold_set and not pred_set:
        return 1.0
    tp = len(gold_set & pred_set)
    fp = len(pred_set - gold_set)
    fn = len(gold_set - pred_set)
    denom = 2 * tp + fp + fn
    if denom <= 0:
        return 0.0
    return 2 * tp / denom


def mmrag_official_generation_score(query_id: str, gold_answer, pred_answer: str) -> float:
    ds = str(query_id).split("_", 1)[0].lower()
    if ds in MMRAG_F1_DATASETS:
        return mmrag_set_f1(mmrag_parse_answer_list(gold_answer), mmrag_parse_answer_list(pred_answer))
    return 1.0 if str(gold_answer) == str(pred_answer) else 0.0


def percentile95_ms(values_s: list[float]) -> float:
    if not values_s:
        return 0.0
    arr = np.asarray(values_s, dtype=np.float64) * 1000.0
    return float(np.percentile(arr, 95))
