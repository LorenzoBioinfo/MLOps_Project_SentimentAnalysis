from transformers import AutoTokenizer, AutoModelForSequenceClassification
from datasets import load_from_disk
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
import torch
import numpy as np
import json
import os

MODEL_PATH = "models/sentiment_model"
TWEET_PATH = "data/processed/tweet_eval_tokenized"
YT_PATH = "data/processed/youtube_comments"
REPORTS_DIR = "reports"


def evaluate_model(model, tokenizer, dataset, dataset_name, sample_size=300):
    print(f"Valutazione su {dataset_name}")
    subset = dataset["test"].select(range(min(sample_size, len(dataset["test"]))))

    texts = subset["text"]
    labels = subset["label"]

    inputs = tokenizer(texts, truncation=True, padding=True, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs)
        preds = torch.argmax(outputs.logits, dim=1).numpy()

    acc = accuracy_score(labels, preds)
    f1 = f1_score(labels, preds, average="weighted")
    cm = confusion_matrix(labels, preds).tolist()

    print(f"{dataset_name} — Accuracy: {acc:.3f}, F1: {f1:.3f}")
    return {"dataset": dataset_name, "accuracy": acc, "f1": f1, "confusion_matrix": cm}

def main():
    print("Caricamento del modello")
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    model.eval()

    tweet_ds = load_from_disk(TWEET_PATH)
    youtube_ds = load_from_disk(YT_PATH)

    tweet_metrics = evaluate_model(model, tokenizer, tweet_ds, "TweetEval")
    youtube_metrics = evaluate_model(model, tokenizer, youtube_ds, "YouTube Comments")

    os.makedirs(REPORTS_DIR, exist_ok=True)
    metrics_path = os.path.join(REPORTS_DIR, "metrics.json")

    results = {"TweetEval": tweet_metrics, "YouTube": youtube_metrics}
    with open(metrics_path, "w") as f:
        json.dump(results, f, indent=4)

    print(f"Risultati salvati in: {metrics_path}")

if __name__ == "__main__":
    main()