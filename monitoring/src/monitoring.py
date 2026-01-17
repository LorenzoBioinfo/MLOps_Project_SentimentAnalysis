from transformers import AutoModelForSequenceClassification
from datasets import load_from_disk, ClassLabel
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix
import torch
import json
import os
import sys
import time
import matplotlib.pyplot as plt
from prometheus_client import start_http_server, Gauge

# ================= CONFIG =================
ACCURACY_THRESHOLD = 0.75
F1_THRESHOLD = 0.70
MAX_FAILED_RUNS = 3

BASE_DIR = os.getenv("BASE_DIR", "/app")
MODEL_PATH = f"{BASE_DIR}/models/sentiment_model"
TWEET_PATH = f"{BASE_DIR}/data/processed/tweet_eval_tokenized"
YT_PATH = f"{BASE_DIR}/data/processed/youtube_tokenized"
REPORTS_DIR = f"{BASE_DIR}/reports"

METRICS_FILE = f"{REPORTS_DIR}/metrics.json"
PLOT_FILE = f"{REPORTS_DIR}/metrics_plot.jpeg"

PROMETHEUS_PORT = 8000
EVALUATION_INTERVAL = 3600  # 1 ora
RUNNING_CI = os.getenv("RUNNING_CI") == "1"

os.makedirs(REPORTS_DIR, exist_ok=True)

# ================= PROMETHEUS =================
accuracy_gauge = Gauge("model_accuracy", "Accuracy", ["dataset"])
f1_gauge = Gauge("model_f1", "F1-score", ["dataset"])
precision_gauge = Gauge("model_precision", "Precision", ["dataset"])
recall_gauge = Gauge("model_recall", "Recall", ["dataset"])


def expose_metrics():
    try:
        start_http_server(PROMETHEUS_PORT)
        print(f"[Prometheus] running on {PROMETHEUS_PORT}")
    except OSError:
        pass


def update_prometheus(tweet, yt):
    for name, m in [("TweetEval", tweet), ("YouTube", yt)]:
        accuracy_gauge.labels(dataset=name).set(m["accuracy"])
        f1_gauge.labels(dataset=name).set(m["f1"])
        precision_gauge.labels(dataset=name).set(m["precision"])
        recall_gauge.labels(dataset=name).set(m["recall"])


# ================= MODEL =================
def load_model():
    if os.path.exists(os.path.join(MODEL_PATH, "config.json")):
        model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)
    else:
        model = AutoModelForSequenceClassification.from_pretrained(
            "cardiffnlp/twitter-roberta-base-sentiment-latest",
            cache_dir=f"{BASE_DIR}/huggingface_cache"
        )
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
    }


# ================= PLOT =================
def generate_plot(history):
    tweet_acc = [h["TweetEval"]["accuracy"] for h in history]
    yt_acc = [h["YouTube"]["accuracy"] for h in history]

    x = range(1, len(tweet_acc) + 1)

    plt.figure(figsize=(10, 5))
    plt.plot(x, tweet_acc, marker="o", label="TweetEval Accuracy")
    plt.plot(x, yt_acc, marker="o", label="YouTube Accuracy")
    plt.xlabel("Valutazioni nel tempo")
    plt.ylabel("Accuracy")
    plt.title("Andamento Accuracy")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(PLOT_FILE)
    plt.close()

    print(f"[Monitoring] Grafico aggiornato → {PLOT_FILE}")


# ================= CORE =================
def run_single_evaluation():
    model = load_model()
    tweet_ds = load_from_disk(TWEET_PATH)
    yt_ds = load_from_disk(YT_PATH)

    tweet_metrics = evaluate_model(model, tweet_ds)
    yt_metrics = evaluate_model(model, yt_ds)

    history = []
    if os.path.exists(METRICS_FILE):
        with open(METRICS_FILE) as f:
            history = json.load(f)

    history.append({"TweetEval": tweet_metrics, "YouTube": yt_metrics})

    with open(METRICS_FILE, "w") as f:
        json.dump(history, f, indent=4)

    update_prometheus(tweet_metrics, yt_metrics)
    generate_plot(history)

    print("[Monitoring] Valutazione completata")


def monitoring_loop():
    expose_metrics()
    failed_runs = 0

    while True:
        run_single_evaluation()

        yt = json.load(open(METRICS_FILE))[-1]["YouTube"]

        if yt["accuracy"] < ACCURACY_THRESHOLD or yt["f1"] < F1_THRESHOLD:
            failed_runs += 1
            print("[ALERT] Performance YouTube sotto soglia")
        else:
            failed_runs = 0

        if failed_runs >= MAX_FAILED_RUNS and not RUNNING_CI:
            print("[Monitoring] Retraining parziale YouTube")
            failed_runs = 0

        if RUNNING_CI:
            break

        time.sleep(EVALUATION_INTERVAL)


if __name__ == "__main__":
    monitoring_loop()
