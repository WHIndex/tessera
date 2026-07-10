#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
import random
import sys
from dataclasses import dataclass

import numpy as np
import torch
from sklearn.metrics import f1_score
from torch.utils.data import Dataset
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
)

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def read_json(path: Path):
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: Path, data):
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


@dataclass
class RouterSample:
    text: str
    labels: list[int]


class RouterDataset(Dataset):
    def __init__(self, samples: list[RouterSample], tokenizer, max_length: int = 256):
        self.samples = samples
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]
        encoded = self.tokenizer(
            s.text,
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt",
        )
        item = {k: v.squeeze(0) for k, v in encoded.items()}
        item["labels"] = torch.tensor(s.labels, dtype=torch.float32)
        return item


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_samples(path: Path, max_rows: int | None = None) -> list[RouterSample]:
    rows = read_json(path)
    if max_rows is not None:
        rows = rows[:max_rows]
    out = []
    for r in rows:
        out.append(RouterSample(text=r.get("query", ""), labels=r.get("labels_multihot", [0, 0, 0])))
    return out


def logits_to_pred(logits: np.ndarray, threshold: float) -> np.ndarray:
    probs = 1.0 / (1.0 + np.exp(-logits))
    pred = (probs >= threshold).astype(np.int64)
    empty_mask = np.sum(pred, axis=1) == 0
    if np.any(empty_mask):
        top_idx = np.argmax(probs[empty_mask], axis=1)
        pred[empty_mask] = 0
        pred[empty_mask, top_idx] = 1
    return pred


def subset_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(np.all(y_true == y_pred, axis=1)))


def resolve_model_dir(path: str) -> Path:
    p = Path(path)
    if (p / "config.json").exists():
        return p

    snapshots = p / "snapshots"
    if snapshots.exists() and snapshots.is_dir():
        cands = sorted([x for x in snapshots.iterdir() if x.is_dir()])
        if not cands:
            raise FileNotFoundError(f"No snapshot directories under {snapshots}")
        for cand in reversed(cands):
            if (cand / "config.json").exists():
                return cand
        return cands[-1]

    raise FileNotFoundError(f"Cannot resolve model directory from: {path}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Train DeBERTa router (multi-label)")
    parser.add_argument("--model-dir", type=str, required=True)
    parser.add_argument("--train-file", type=Path, required=True)
    parser.add_argument("--val-file", type=Path, required=True)
    parser.add_argument("--test-file", type=Path, default=None)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--run-id", type=str, default="router_deberta")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--max-train", type=int, default=None)
    parser.add_argument("--max-val", type=int, default=None)
    parser.add_argument("--max-test", type=int, default=None)
    args = parser.parse_args()

    ensure_dir(args.out_dir)
    set_seed(args.seed)

    model_dir = resolve_model_dir(args.model_dir)
    print(f"[model] resolved model dir: {model_dir}")

    tokenizer = AutoTokenizer.from_pretrained(str(model_dir))
    model = AutoModelForSequenceClassification.from_pretrained(
        str(model_dir),
        num_labels=3,
        problem_type="multi_label_classification",
    )

    train_samples = load_samples(args.train_file, max_rows=args.max_train)
    val_samples = load_samples(args.val_file, max_rows=args.max_val)

    train_ds = RouterDataset(train_samples, tokenizer, max_length=args.max_length)
    val_ds = RouterDataset(val_samples, tokenizer, max_length=args.max_length)

    training_args = TrainingArguments(
        output_dir=str(args.out_dir / "hf_ckpt"),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        learning_rate=args.lr,
        eval_strategy="epoch",
        save_strategy="no",
        logging_steps=20,
        report_to=[],
        fp16=torch.cuda.is_available(),
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        tokenizer=tokenizer,
    )

    trainer.train()

    val_pred = trainer.predict(val_ds)
    y_val_true = np.array([s.labels for s in val_samples], dtype=np.int64)
    y_val_pred = logits_to_pred(val_pred.predictions, threshold=args.threshold)

    metrics = {
        "run_id": args.run_id,
        "model": str(model_dir),
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "threshold": args.threshold,
        "train_rows": len(train_samples),
        "val_rows": len(val_samples),
        "val_micro_f1": float(f1_score(y_val_true, y_val_pred, average="micro", zero_division=0)),
        "val_subset_acc": subset_accuracy(y_val_true, y_val_pred),
    }

    if args.test_file is not None and args.test_file.exists():
        test_samples = load_samples(args.test_file, max_rows=args.max_test)
        test_ds = RouterDataset(test_samples, tokenizer, max_length=args.max_length)
        test_pred = trainer.predict(test_ds)
        y_test_true = np.array([s.labels for s in test_samples], dtype=np.int64)
        y_test_pred = logits_to_pred(test_pred.predictions, threshold=args.threshold)
        metrics.update(
            {
                "test_rows": len(test_samples),
                "test_micro_f1": float(f1_score(y_test_true, y_test_pred, average="micro", zero_division=0)),
                "test_subset_acc": subset_accuracy(y_test_true, y_test_pred),
            }
        )

    metrics_path = args.out_dir / f"{args.run_id}_metrics.json"
    write_json(metrics_path, metrics)

    model_out = args.out_dir / f"{args.run_id}_model"
    ensure_dir(model_out)
    model.save_pretrained(model_out)
    tokenizer.save_pretrained(model_out)

    print(json.dumps(metrics, ensure_ascii=False, indent=2))
    print(f"[OK] metrics: {metrics_path}")
    print(f"[OK] model: {model_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
