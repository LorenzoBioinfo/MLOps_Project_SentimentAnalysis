from transformers import AutoModelForSequenceClassification
from datasets import load_from_disk
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
import torch
import json
import os
from train_model import train_model

ACCURACY_THRESHOLD = 0.75
MODEL_PATH = "models/sentiment_model"
TWEET_PATH = "data/processed/tweet_eval_tokenized"
YT_PATH = "data/processed/youtube_tokenized"
REPORTS_DIR = "reports"


def evaluate_model(model, dataset, dataset_name, sample_size=300):
    print(f"Valutazione su {dataset_name}")

    # Prendo il sottoinsieme dei dati
    if "test" in dataset:
        subset = dataset["test"].select(range(min(sample_size, len(dataset["test"]))))
    else:
        subset = dataset["train"].train_test_split(test_size=0.1)["test"]

    input_ids = torch.tensor(subset["input_ids"])
    attention_mask = torch.tensor(subset["attention_mask"])
    labels = torch.tensor(subset["label"])

    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        preds = torch.argmax(outputs.logits, dim=1)

    acc = accuracy_score(labels.numpy(), preds.numpy())
    f1 = f1_score(labels.numpy(), preds.numpy(), average="weighted")
    cm = confusion_matrix(labels.numpy(), preds.numpy()).tolist()

    print(f"{dataset_name} — Accuracy: {acc:.3f}, F1: {f1:.3f}")
    return {"dataset": dataset_name, "accuracy": acc, "f1": f1, "confusion_matrix": cm}


def retrain_on_youtube_sample():
    youtube_data = load_from_disk(YT_PATH)["train"]
    youtube_sample = youtube_data.shuffle(seed=42).select(range(500))
    train_model(additional_data=youtube_sample, output_dir=MODEL_PATH)


def main():
    print("Caricamento del modello")

    if os.path.exists(MODEL_PATH):
        model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)
    else:
        print("Modello locale non trovato. Uso modello pre-addestrato di default.")
        model = AutoModelForSequenceClassification.from_pretrained(
            "cardiffnlp/twitter-roberta-base-sentiment-latest"
        )

    model.eval()

    tweet_ds = load_from_disk(TWEET_PATH)
    youtube_ds = load_from_disk(YT_PATH)

    tweet_metrics = evaluate_model(model, tweet_ds, "TweetEval")
    youtube_metrics = evaluate_model(model, youtube_ds, "YouTube Comments")

    print(f"Accuracy su YouTube: {youtube_metrics['accuracy']:.3f}")
    if youtube_metrics["accuracy"] < ACCURACY_THRESHOLD:
        print("Performance sotto la soglia. Avvio retraining parziale...")
        retrain_on_youtube_sample()

    os.makedirs(REPORTS_DIR, exist_ok=True)
    metrics_path = os.path.join(REPORTS_DIR, "metrics.json")

    results = {"TweetEval": tweet_metrics, "YouTube": youtube_metrics}
    with open(metrics_path, "w") as f:
        json.dump(results, f, indent=4)

    print(f"Risultati salvati in: {metrics_path}")


if __name__ == "__main__":
    main()
