"""
Fine-tune DeBERTa-v3-small on HaluEval QA for hallucination detection.

Binary sequence classifier: (context, response) → hallucinated/not.
Uses a plain PyTorch loop (no Trainer) so device is explicit — avoids MPS OOM.
Holds out the first 500 samples to match the existing benchmark test set.
Saves to ./finetuned_nli/ for use as a ChainCheck method.

Usage:
    python scripts/finetune_deberta.py
    python scripts/finetune_deberta.py --epochs 5 --train-samples 8000
"""

from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np
import torch
import torch.nn.functional as F
from datasets import load_dataset
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer

MODEL_NAME = "microsoft/deberta-v3-small"
OUTPUT_DIR  = "./finetuned_nli"
N_TEST      = 500
N_TRAIN_MAX = 5000
MAX_LENGTH  = 256
DEVICE      = torch.device("cpu")


class PairDataset(Dataset):
    def __init__(self, samples: list[dict], tokenizer, max_length: int) -> None:
        self.encodings = tokenizer(
            [s["text_a"] for s in samples],
            [s["text_b"] for s in samples],
            truncation=True,
            max_length=max_length,
            padding="max_length",
            return_tensors="pt",
        )
        self.labels = torch.tensor([s["label"] for s in samples], dtype=torch.long)

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> dict:
        return {k: v[idx] for k, v in self.encodings.items()} | {"labels": self.labels[idx]}


def build_samples(n_test: int = N_TEST, n_train_max: int = N_TRAIN_MAX) -> tuple[list[dict], list[dict]]:
    print("Loading HaluEval QA…")
    ds = load_dataset("pminervini/HaluEval", "qa", split="data")
    samples: list[dict] = []
    for row in ds:
        ctx  = str(row.get("knowledge", "") or "")
        right = str(row.get("right_answer", "") or "")
        hall  = str(row.get("hallucinated_answer", "") or "")
        if right:
            samples.append({"text_a": ctx, "text_b": right, "label": 0})
        if hall:
            samples.append({"text_a": ctx, "text_b": hall,  "label": 1})
    test_samples  = samples[:n_test]
    train_samples = samples[n_test: n_test + n_train_max]
    print(f"Train: {len(train_samples)}  Test: {len(test_samples)}")
    return train_samples, test_samples


def evaluate(model, loader) -> dict:
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for batch in loader:
            labels = batch.pop("labels").to(DEVICE)
            inputs = {k: v.to(DEVICE) for k, v in batch.items()}
            logits = model(**inputs).logits
            preds  = logits.argmax(dim=-1).cpu().tolist()
            all_preds.extend(preds)
            all_labels.extend(labels.cpu().tolist())
    return {
        "f1":        f1_score(all_labels, all_preds, zero_division=0),
        "precision": precision_score(all_labels, all_preds, zero_division=0),
        "recall":    recall_score(all_labels, all_preds, zero_division=0),
        "accuracy":  accuracy_score(all_labels, all_preds),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs",        type=int,   default=3)
    parser.add_argument("--batch-size",    type=int,   default=8)
    parser.add_argument("--lr",            type=float, default=2e-5)
    parser.add_argument("--train-samples", type=int,   default=N_TRAIN_MAX)
    parser.add_argument("--output-dir",    type=str,   default=OUTPUT_DIR)
    args = parser.parse_args()

    train_samples, test_samples = build_samples(n_train_max=args.train_samples)

    print(f"Loading tokenizer and model ({MODEL_NAME}) on CPU…")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model     = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=2)
    model.to(DEVICE)

    train_ds = PairDataset(train_samples, tokenizer, MAX_LENGTH)
    test_ds  = PairDataset(test_samples,  tokenizer, MAX_LENGTH)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    test_loader  = DataLoader(test_ds,  batch_size=args.batch_size)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    total_steps = len(train_loader) * args.epochs
    warmup_steps = int(0.06 * total_steps)

    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        return max(0.0, (total_steps - step) / max(1, total_steps - warmup_steps))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    print(f"\nFine-tuning {MODEL_NAME} for {args.epochs} epochs on {len(train_ds)} samples (CPU)…\n")
    best_f1 = 0.0
    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss, t0 = 0.0, time.time()
        for step, batch in enumerate(train_loader, 1):
            labels = batch.pop("labels").to(DEVICE)
            inputs = {k: v.to(DEVICE) for k, v in batch.items()}
            loss   = model(**inputs, labels=labels).loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            total_loss += loss.item()
            if step % 50 == 0:
                elapsed = time.time() - t0
                print(f"  epoch {epoch} step {step}/{len(train_loader)}  "
                      f"loss={total_loss/step:.4f}  {elapsed:.0f}s elapsed")

        metrics = evaluate(model, test_loader)
        print(f"\nEpoch {epoch} — F1={metrics['f1']:.4f}  "
              f"P={metrics['precision']:.4f}  R={metrics['recall']:.4f}  "
              f"acc={metrics['accuracy']:.4f}\n")

        if metrics["f1"] > best_f1:
            best_f1 = metrics["f1"]
            model.save_pretrained(args.output_dir)
            tokenizer.save_pretrained(args.output_dir)
            with open(os.path.join(args.output_dir, "best_metrics.json"), "w") as f:
                json.dump(metrics, f, indent=2)
            print(f"  ✓ saved best model (F1={best_f1:.4f})\n")

    print(f"\n── Training complete. Best F1: {best_f1:.4f} ──")
    print(f"Model saved to {args.output_dir}/")


if __name__ == "__main__":
    main()
