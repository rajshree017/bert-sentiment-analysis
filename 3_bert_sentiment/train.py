"""
BERT Fine-tuning for Sentiment Analysis
-----------------------------------------
Dataset: IMDB Movie Reviews (HuggingFace datasets)
"""

import torch
from torch.utils.data import DataLoader
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    get_linear_schedule_with_warmup,
)
from datasets import load_dataset
from torch.optim import AdamW
from sklearn.metrics import classification_report
import numpy as np

# ── Config ──────────────────────────────────────────────────
MODEL_NAME   = "distilbert-base-uncased"   # lighter than bert-base
NUM_LABELS   = 2
EPOCHS       = 3
BATCH_SIZE   = 16
MAX_LEN      = 256
LR           = 2e-5
DEVICE       = "cuda" if torch.cuda.is_available() else "cpu"
SAVE_PATH    = "saved_model"

print(f"[INFO] Using device: {DEVICE}")

# ── Load & tokenize dataset ───────────────────────────────────
def get_dataloaders():
    dataset   = load_dataset("imdb")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    def tokenize(batch):
        return tokenizer(
            batch["text"],
            padding="max_length",
            truncation=True,
            max_length=MAX_LEN,
        )

    tokenized = dataset.map(tokenize, batched=True)
    tokenized.set_format("torch", columns=["input_ids", "attention_mask", "label"])

    train_loader = DataLoader(tokenized["train"], batch_size=BATCH_SIZE, shuffle=True)
    test_loader  = DataLoader(tokenized["test"],  batch_size=BATCH_SIZE)
    return train_loader, test_loader, tokenizer

# ── Train ─────────────────────────────────────────────────────
def train(train_loader):
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME, num_labels=NUM_LABELS
    ).to(DEVICE)

    optimizer = AdamW(model.parameters(), lr=LR, weight_decay=0.01)
    total_steps = len(train_loader) * EPOCHS
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=total_steps // 10, num_training_steps=total_steps
    )

    model.train()
    for epoch in range(1, EPOCHS + 1):
        total_loss = 0
        for step, batch in enumerate(train_loader, 1):
            ids  = batch["input_ids"].to(DEVICE)
            mask = batch["attention_mask"].to(DEVICE)
            lbls = batch["label"].to(DEVICE)

            outputs = model(input_ids=ids, attention_mask=mask, labels=lbls)
            loss    = outputs.loss

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            total_loss += loss.item()

            if step % 200 == 0:
                print(f"Epoch {epoch} | Step {step}/{len(train_loader)} | Loss {total_loss/step:.4f}")

        print(f"\n✅ Epoch {epoch} done — avg loss: {total_loss/len(train_loader):.4f}\n")

    model.save_pretrained(SAVE_PATH)
    print(f"[✓] Model saved to {SAVE_PATH}/")
    return model

# ── Evaluate ──────────────────────────────────────────────────
def evaluate(model, test_loader):
    model.eval()
    all_preds, all_labels = [], []

    with torch.no_grad():
        for batch in test_loader:
            ids  = batch["input_ids"].to(DEVICE)
            mask = batch["attention_mask"].to(DEVICE)
            lbls = batch["label"]

            logits = model(input_ids=ids, attention_mask=mask).logits
            preds  = torch.argmax(logits, dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(lbls.numpy())

    print("\n" + classification_report(all_labels, all_preds, target_names=["negative","positive"]))

# ── Predict on custom text ────────────────────────────────────
def predict(text: str, tokenizer, model) -> str:
    model.eval()
    enc = tokenizer(text, return_tensors="pt", truncation=True, max_length=MAX_LEN).to(DEVICE)
    with torch.no_grad():
        logits = model(**enc).logits
    prob   = torch.softmax(logits, dim=1)[0]
    label  = torch.argmax(prob).item()
    return f"{'POSITIVE' if label else 'NEGATIVE'} ({prob[label]*100:.1f}%)"

if __name__ == "__main__":
    train_loader, test_loader, tokenizer = get_dataloaders()
    model = train(train_loader)
    evaluate(model, test_loader)

    # Quick demo
    samples = [
        "This movie was absolutely fantastic, loved every minute!",
        "Terrible acting, boring plot, complete waste of time.",
    ]
    for s in samples:
        print(f"\n>> {s}\n   → {predict(s, tokenizer, model)}")
