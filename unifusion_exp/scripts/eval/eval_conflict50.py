#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path


def normalize_answer(text: str) -> str:
    text = str(text).lower()
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    text = "".join(ch for ch in text if ch not in set("!\"#$%&'()*+,-./:;<=>?@[\\]^_`{|}~"))
    text = " ".join(text.split())
    return text


def exact_match(pred: str, gold: str) -> float:
    return 1.0 if normalize_answer(pred) == normalize_answer(gold) else 0.0


def f1_score(pred: str, gold: str) -> float:
    p = normalize_answer(pred).split()
    g = normalize_answer(gold).split()
    if not p and not g:
        return 1.0
    if not p or not g:
        return 0.0
    p_count: dict[str, int] = {}
    g_count: dict[str, int] = {}
    for t in p:
        p_count[t] = p_count.get(t, 0) + 1
    for t in g:
        g_count[t] = g_count.get(t, 0) + 1
    same = 0
    for t, c in p_count.items():
        same += min(c, g_count.get(t, 0))
    if same == 0:
        return 0.0
    prec = same / len(p)
    rec = same / len(g)
    return 2 * prec * rec / (prec + rec)


def parse_bool(v) -> int | None:
    if v is None:
        return None
    if isinstance(v, bool):
        return int(v)
    s = str(v).strip().lower()
    if s in {"1", "true", "yes", "y"}:
        return 1
    if s in {"0", "false", "no", "n"}:
        return 0
    return None


def load_jsonl_or_csv(path: Path) -> list[dict]:
    if not path.exists():
        raise FileNotFoundError(path)
    if path.suffix.lower() == ".csv":
        with path.open("r", encoding="utf-8", newline="") as f:
            return [dict(x) for x in csv.DictReader(f)]

    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def safe_float(x, default: float = 0.0) -> float:
    try:
        return float(x)
    except Exception:
        return default


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate Conflict-50 annotations and system decisions")
    parser.add_argument("--annotation-file", type=Path, required=True)
    parser.add_argument("--system-file", type=Path, required=True)
    parser.add_argument("--baseline-file", type=Path, default=None)
    parser.add_argument("--out-json", type=Path, required=True)
    parser.add_argument("--out-md", type=Path, required=True)
    args = parser.parse_args()

    ann_rows = load_jsonl_or_csv(args.annotation_file)
    sys_rows = load_jsonl_or_csv(args.system_file)
    base_rows = load_jsonl_or_csv(args.baseline_file) if args.baseline_file else []

    ann_by_id = {str(x.get("id", "")): x for x in ann_rows if str(x.get("id", ""))}
    sys_by_id = {str(x.get("id", "")): x for x in sys_rows if str(x.get("id", ""))}
    base_by_id = {str(x.get("id", "")): x for x in base_rows if str(x.get("id", ""))}

    overlap_ids = [qid for qid in ann_by_id.keys() if qid in sys_by_id]

    conflict_total = 0
    conflict_ok = 0
    consistency_total = 0
    consistency_ok = 0

    sys_em_vals = []
    sys_f1_vals = []
    base_em_vals = []
    base_f1_vals = []

    for qid in overlap_ids:
        ann = ann_by_id[qid]
        sys_item = sys_by_id[qid]

        gold_conflict = parse_bool(ann.get("is_conflict"))
        pred_conflict = parse_bool(sys_item.get("conflict_flag", sys_item.get("is_conflict")))
        if gold_conflict is not None and pred_conflict is not None:
            conflict_total += 1
            if int(gold_conflict) == int(pred_conflict):
                conflict_ok += 1

        preferred = str(ann.get("preferred_evidence_side", "")).strip().lower()
        selected = str(sys_item.get("selected_side", sys_item.get("evidence_side", ""))).strip().lower()
        if gold_conflict == 1 and preferred in {"table", "kg", "both"} and selected:
            consistency_total += 1
            if preferred == "both" or selected == preferred:
                consistency_ok += 1

        gold_answer = str(ann.get("gold_answer", ann.get("answer", ""))).strip()
        sys_pred = str(sys_item.get("prediction", sys_item.get("pred", ""))).strip()
        if gold_answer and sys_pred:
            sys_em_vals.append(exact_match(sys_pred, gold_answer))
            sys_f1_vals.append(f1_score(sys_pred, gold_answer))

        if qid in base_by_id and gold_answer:
            base_pred = str(base_by_id[qid].get("prediction", base_by_id[qid].get("pred", ""))).strip()
            if base_pred:
                base_em_vals.append(exact_match(base_pred, gold_answer))
                base_f1_vals.append(f1_score(base_pred, gold_answer))

    result = {
        "annotation_file": str(args.annotation_file),
        "system_file": str(args.system_file),
        "baseline_file": str(args.baseline_file) if args.baseline_file else None,
        "overlap_queries": len(overlap_ids),
        "conflict_acc": (conflict_ok / conflict_total) if conflict_total else None,
        "conflict_acc_n": conflict_total,
        "consistency_score": (consistency_ok / consistency_total) if consistency_total else None,
        "consistency_score_n": consistency_total,
        "system_subset_em": (sum(sys_em_vals) / len(sys_em_vals)) if sys_em_vals else None,
        "system_subset_f1": (sum(sys_f1_vals) / len(sys_f1_vals)) if sys_f1_vals else None,
        "system_subset_n": len(sys_f1_vals),
        "baseline_subset_em": (sum(base_em_vals) / len(base_em_vals)) if base_em_vals else None,
        "baseline_subset_f1": (sum(base_f1_vals) / len(base_f1_vals)) if base_f1_vals else None,
        "baseline_subset_n": len(base_f1_vals),
    }

    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")

    lines = [
        "# Conflict-50 Evaluation",
        "",
        f"- overlap_queries: {result['overlap_queries']}",
        f"- conflict_acc: {safe_float(result['conflict_acc']):.4f} (n={result['conflict_acc_n']})"
        if result["conflict_acc"] is not None
        else f"- conflict_acc: N/A (n={result['conflict_acc_n']})",
        f"- consistency_score: {safe_float(result['consistency_score']):.4f} (n={result['consistency_score_n']})"
        if result["consistency_score"] is not None
        else f"- consistency_score: N/A (n={result['consistency_score_n']})",
        f"- system_subset_f1/em: {safe_float(result['system_subset_f1']):.4f} / {safe_float(result['system_subset_em']):.4f} (n={result['system_subset_n']})"
        if result["system_subset_f1"] is not None
        else f"- system_subset_f1/em: N/A (n={result['system_subset_n']})",
    ]
    if result["baseline_subset_f1"] is not None:
        lines.append(
            f"- baseline_subset_f1/em: {safe_float(result['baseline_subset_f1']):.4f} / {safe_float(result['baseline_subset_em']):.4f} (n={result['baseline_subset_n']})"
        )
        lines.append(
            f"- delta_f1(system-baseline): {safe_float(result['system_subset_f1']) - safe_float(result['baseline_subset_f1']):+.4f}"
        )
        lines.append(
            f"- delta_em(system-baseline): {safe_float(result['system_subset_em']) - safe_float(result['baseline_subset_em']):+.4f}"
        )

    args.out_md.parent.mkdir(parents=True, exist_ok=True)
    args.out_md.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(f"[OK] json -> {args.out_json}")
    print(f"[OK] markdown -> {args.out_md}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
