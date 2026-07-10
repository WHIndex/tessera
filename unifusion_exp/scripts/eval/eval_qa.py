#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
import string
from collections import Counter
from pathlib import Path


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


def load_gold(path: Path):
    data = json.loads(path.read_text(encoding="utf-8"))
    out = {}
    for row in data:
        qid = row.get("id")
        ans = row.get("answer", "")
        if qid:
            out[str(qid)] = str(ans)
    return out


def load_preds(path: Path):
    out = {}
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            qid = row.get("id")
            pred = row.get("prediction", row.get("pred", row.get("answer", "")))
            if qid is not None:
                out[str(qid)] = str(pred)
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate QA predictions with EM/F1")
    parser.add_argument("--gold-file", type=Path, required=True, help="JSON list file with fields id/answer")
    parser.add_argument("--pred-file", type=Path, required=True, help="JSONL file with fields id/prediction")
    parser.add_argument("--out-file", type=Path, required=True)
    parser.add_argument("--detail-file", type=Path, default=None)
    args = parser.parse_args()

    gold = load_gold(args.gold_file)
    pred = load_preds(args.pred_file)

    ids = sorted(set(gold.keys()) & set(pred.keys()))
    if not ids:
        raise SystemExit("No overlapping ids between gold and predictions")

    em_list = []
    f1_list = []
    details = []
    for qid in ids:
        g = gold[qid]
        p = pred[qid]
        em = exact_match(p, g)
        f1 = f1_score(p, g)
        em_list.append(em)
        f1_list.append(f1)
        if args.detail_file is not None:
            details.append({"id": qid, "gold": g, "pred": p, "em": em, "f1": f1})

    out = {
        "evaluated": len(ids),
        "gold_total": len(gold),
        "pred_total": len(pred),
        "coverage": len(ids) / len(gold) if gold else 0.0,
        "exact_match": sum(em_list) / len(em_list),
        "f1": sum(f1_list) / len(f1_list),
    }

    args.out_file.parent.mkdir(parents=True, exist_ok=True)
    args.out_file.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")

    if args.detail_file is not None:
        args.detail_file.parent.mkdir(parents=True, exist_ok=True)
        args.detail_file.write_text(json.dumps(details, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"[OK] detail -> {args.detail_file}")

    print(json.dumps(out, ensure_ascii=False, indent=2))
    print(f"[OK] saved -> {args.out_file}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
