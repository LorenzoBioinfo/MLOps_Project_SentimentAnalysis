from transformers import AutoModelForSequenceClassification
from datasets import load_from_disk
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    confusion_matrix
)
from prometheus_client import start_http_server, Gauge
from datetime import datetime

import torch
import json
import os
import time
import matplotlib.pyplot as plt
import seaborn as sns


# ================= CONFIG =================
ACCURACY_THRESHOLD = 0.75
F1_THRESHOLD = 0.70
MAX_FAILED_RUNS = 3

BASE_DIR = os.getenv("BASE_DIR", "/app")

MODEL_PATH = f"{BASE_DIR}/models/sentiment_model"
TWEET_PATH = f"{BASE_DIR}/data/processed/tweet_eval_tokenized"
YT_PATH = f"{BASE_DIR}/data/processed/youtube_tokenized"


REPORTS_BASE = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "reports", "monitoring")
)

PROMETHEUS_PORT = 8000
EVALUATION_INTERVAL = 3600  # 1 ora
RUNNING_CI = os.getenv("RUNNING_CI") == "1"
HF_CACHE_DIR = os.getenv("HF_CACHE_DIR", "/app/huggingface_cache")


os.makedirs(REPORTS_BASE, exist_ok=True)


# ================= PROMETHEUS =================
accuracy_gauge = Gauge("model_accuracy", "Accuracy", ["dataset"])
f1_gauge = Gauge("model_f1", "F1-score", ["dataset"])
precision_gauge = Gauge("model_precision", "Precision", ["dataset"])
recall_gauge = Gauge("model_recall", "Recall", ["dataset"])


def expose_metrics():
    try:
        start_http_server(PROMETHEUS_PORT)
        print(f"[Prometheus] running on port {PROMETHEUS_PORT}")
    except OSError:
        pass


def update_prometheus(tweet, yt):
    for name, m in [("TweetEval", tweet), ("YouTube", yt)]:
        accuracy_gauge.labels(dataset=name).set(m["accuracy"])
        f1_gauge.labels(dataset=name).set(m["f1"])
        precision_gauge.labels(dataset=name).set(m["precision"])
        recall_gauge.labels(dataset=name).set(m["recall"])


# ================= UTILITIES =================
def create_run_dir():
    run_id = datetime.now().strftime("run_%Y-%m-%d_%H-%M-%S")
    run_dir = os.path.join(REPORTS_BASE, run_id)
    os.makedirs(run_dir, exist_ok=True)
    return run_dir


def save_json(obj, path):
    with open(path, "w") as f:
        json.dump(obj, f, indent=4)


def save_confusion_matrix(cm, title, path):
    plt.figure(figsize=(5, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.title(title)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.tight_layout()
    plt.savefig(path)
    plt.close()


# ================= MODEL =================
def load_model():
    if os.path.exists(os.path.join(MODEL_PATH, "config.json")):
        model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)
        print("[Model] Loaded fine-tuned model")
    else:
        model = AutoModelForSequenceClassification.from_pretrained(
            "cardiffnlp/twitter-roberta-base-sentiment-latest",
            cache_dir=HF_CACHE_DIR
        )
        print("[Model] Loaded base HF model")

    model.eval()
    return model


def evaluate_model(model, dataset, sample_size=300):
    subset = dataset.get("test") or dataset["train"].train_test_split(test_size=0.1)["test"]
    subset = subset.select(range(min(sample_size, len(subset))))

    inputs = {
        "input_ids": torch.tensor(subset["input_ids"]),
        "attention_mask": torch.tensor(subset["attention_mask"]),
    }
    labels = torch.tensor(subset["label"])

    with torch.no_grad():
        preds = torch.argmax(model(**inputs).logits, dim=1)

    return {
        "accuracy": accuracy_score(labels, preds),
        "f1": f1_score(labels, preds, average="weighted"),
        "precision": precision_score(labels, preds, average="weighted"),
        "recall": recall_score(labels, preds, average="weighted"),
        "confusion_matrix": confusion_matrix(labels, preds).tolist(),
        "num_samples": len(labels)
    }


# ================= PLOTS =================
def plot_accuracy_trend(history, output_path):
    tweet_acc = [h["TweetEval"]["accuracy"] for h in history]
    yt_acc = [h["YouTube"]["accuracy"] for h in history]

    x = range(1, len(tweet_acc) + 1)

    plt.figure(figsize=(10, 5))
    plt.plot(x, tweet_acc, marker="o", label="TweetEval Accuracy")
    plt.plot(x, yt_acc, marker="o", label="YouTube Accuracy")
    plt.xlabel("Evaluation run")
    plt.ylabel("Accuracy")
    plt.title("Accuracy trend over time")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


# ================= CORE =================
def run_single_evaluation():
    print("[Monitoring] Starting evaluation")

    run_dir = create_run_dir()

    model = load_model()
    tweet_ds = load_from_disk(TWEET_PATH)
    yt_ds = load_from_disk(YT_PATH)

    tweet_metrics = evaluate_model(model, tweet_ds)
    yt_metrics = evaluate_model(model, yt_ds)

    save_json(tweet_metrics, os.path.join(run_dir, "tweet_metrics.json"))
    save_json(yt_metrics, os.path.join(run_dir, "youtube_metrics.json"))

    save_confusion_matrix(
        tweet_metrics["confusion_matrix"],
        "TweetEval Confusion Matrix",
        os.path.join(run_dir, "tweet_confusion_matrix.jpg")
    )

    save_confusion_matrix(
        yt_metrics["confusion_matrix"],
        "YouTube Confusion Matrix",
        os.path.join(run_dir, "youtube_confusion_matrix.jpg")
    )

    history_file = os.path.join(REPORTS_BASE, "history.json")
    history = []

    if os.path.exists(history_file):
        with open(history_file) as f:
            history = json.load(f)

    history.append({
        "run": os.path.basename(run_dir),
        "TweetEval": tweet_metrics,
        "YouTube": yt_metrics
    })

    save_json(history, history_file)

    plot_accuracy_trend(
        history,
        os.path.join(run_dir, "accuracy_trend.jpg")
    )

    update_prometheus(tweet_metrics, yt_metrics)

    print(f"[Monitoring] Evaluation completed → {run_dir}")
    return yt_metrics


def monitoring_loop():
    expose_metrics()
    failed_runs = 0

    while True:
        yt_metrics = run_single_evaluation()

        if yt_metrics["accuracy"] < ACCURACY_THRESHOLD or yt_metrics["f1"] < F1_THRESHOLD:
            failed_runs += 1
            print("[ALERT] YouTube performance below threshold")
        else:
            failed_runs = 0

        if failed_runs >= MAX_FAILED_RUNS and not RUNNING_CI:
            print("[Monitoring] Retraining trigger would happen here")
            failed_runs = 0

        if RUNNING_CI:
            break

        time.sleep(EVALUATION_INTERVAL)


if __name__ == "__main__":
    monitoring_loop()
