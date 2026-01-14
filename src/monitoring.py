from transformers import AutoModelForSequenceClassification
from datasets import load_from_disk, ClassLabel
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, precision_score, recall_score
import torch
import json
import os
import threading
from prometheus_client import start_http_server, Gauge
from src.train_model import train_model
import time

# CONFIG
ACCURACY_THRESHOLD = 0.75
F1_THRESHOLD = 0.70
MAX_FAILED_RUNS = 3  # Numero di alert consecutivi prima del retraining
BASE_DIR = os.getenv("BASE_DIR", ".")
MODEL_PATH = f"{BASE_DIR}/models/sentiment_model"
TWEET_PATH = f"{BASE_DIR}/data/processed/tweet_eval_tokenized"
YT_PATH = f"{BASE_DIR}/data/processed/youtube_tokenized"
REPORTS_DIR = f"{BASE_DIR}/reports"
PROMETHEUS_PORT = 8000
EVALUATION_INTERVAL = 3600  # secondi

# Flag per test/CI
RUNNING_CI = os.getenv("RUNNING_CI") == "1"

# PROMETHEUS METRICS
accuracy_gauge = Gauge("model_accuracy", "Accuracy del modello", ["dataset"])
f1_gauge = Gauge("model_f1", "F1-score del modello", ["dataset"])
precision_gauge = Gauge("model_precision", "Precision del modello", ["dataset"])
recall_gauge = Gauge("model_recall", "Recall del modello", ["dataset"])


def expose_metrics(port=PROMETHEUS_PORT):
    """Avvia un server HTTP Prometheus in background"""
    start_http_server(port)
    print(f"[Prometheus] Metrics server running at http://localhost:{port}/metrics")


def update_metrics(tweet_metrics, youtube_metrics):
    """Aggiorna i valori dei Gauge Prometheus"""
    for dataset_name, metrics in [("TweetEval", tweet_metrics), ("YouTube", youtube_metrics)]:
        accuracy_gauge.labels(dataset=dataset_name).set(metrics["accuracy"])
        f1_gauge.labels(dataset=dataset_name).set(metrics["f1"])
        precision_gauge.labels(dataset=dataset_name).set(metrics["precision"])
        recall_gauge.labels(dataset=dataset_name).set(metrics["recall"])


def evaluate_model(model, dataset, dataset_name, sample_size=300):
    """Valuta il modello su un dataset e calcola metriche principali"""
    print(f"[Monitoring] Valutazione su {dataset_name}")
    subset = dataset.get("test")
    if subset is None:
        subset = dataset["train"].train_test_split(test_size=0.1)["test"]
    subset = subset.select(range(min(sample_size, len(subset))))

    input_ids = torch.tensor(subset["input_ids"])
    attention_mask = torch.tensor(subset["attention_mask"])
    labels = torch.tensor(subset["label"])

    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        preds = torch.argmax(outputs.logits, dim=1)

    acc = accuracy_score(labels.numpy(), preds.numpy())
    f1 = f1_score(labels.numpy(), preds.numpy(), average="weighted")
    cm = confusion_matrix(labels.numpy(), preds.numpy()).tolist()
    precision = precision_score(labels.numpy(), preds.numpy(), average="weighted")
    recall = recall_score(labels.numpy(), preds.numpy(), average="weighted")

    print(f"[Monitoring] {dataset_name} — Accuracy: {acc:.3f}, F1: {f1:.3f}")
    return {"dataset": dataset_name, "accuracy": acc, "f1": f1,
            "precision": precision, "recall": recall, "confusion_matrix": cm}


def retrain_on_youtube_sample():
    """Esegue il retraining parziale sul dataset YouTube"""
    print("[Monitoring] Preparazione retraining dati YouTube...")
    youtube_data = load_from_disk(YT_PATH)["train"]
    youtube_sample = youtube_data.shuffle(seed=42).select(range(500))
    youtube_sample = youtube_sample.remove_columns(
        [col for col in youtube_sample.column_names if col not in ["text", "label"]]
    )
    label_class = ClassLabel(names=["negative", "neutral", "positive"])
    youtube_sample = youtube_sample.cast_column("label", label_class)
    train_model(additional_data=youtube_sample, output_dir=MODEL_PATH)


def send_alert(message):
    """Alert semplice: stampa su console"""
    print(f"[ALERT] {message}")


def main():
    threading.Thread(target=lambda: expose_metrics(PROMETHEUS_PORT), daemon=True).start()

    print("[Monitoring] Caricamento del modello")
    config_path = os.path.join(MODEL_PATH, "config.json")
    if os.path.exists(config_path):
        print("[Monitoring] Carico modello locale addestrato")
        model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)
    else:
        print("[Monitoring] Modello locale non valido o assente. Uso modello pre-addestrato.")
        model = AutoModelForSequenceClassification.from_pretrained(
            "cardiffnlp/twitter-roberta-base-sentiment-latest"
        )

    model.eval()
    failed_runs = 0

    while True:
        tweet_ds = load_from_disk(TWEET_PATH)
        youtube_ds = load_from_disk(YT_PATH)

        tweet_metrics = evaluate_model(model, tweet_ds, "TweetEval")
        youtube_metrics = evaluate_model(model, youtube_ds, "YouTube")

        # Salvataggio storico JSON
        os.makedirs(REPORTS_DIR, exist_ok=True)
        metrics_path = os.path.join(REPORTS_DIR, "metrics.json")

        try:
            with open(metrics_path, "r") as f:
                all_results = json.load(f)
        except FileNotFoundError:
            print("[Monitoring] Nessun file metrics.json trovato, ne creo uno nuovo.")
            all_results = []

        results = {"TweetEval": tweet_metrics, "YouTube": youtube_metrics}
        all_results.append(results)
        with open(metrics_path, "w") as f:
            json.dump(all_results, f, indent=4)

        # Aggiorna metriche Prometheus
        update_metrics(tweet_metrics=tweet_metrics, youtube_metrics=youtube_metrics)

        # Alerting + contatore per retraining
        alert_needed = False
        for dataset_name, metrics in [("YouTube", youtube_metrics)]:
            if metrics["accuracy"] < ACCURACY_THRESHOLD or metrics["f1"] < F1_THRESHOLD:
                alert_needed = True
                send_alert(
                    f"Performance {dataset_name} sotto soglia: "
                    f"Acc {metrics['accuracy']:.3f}, "
                    f"F1 {metrics['f1']:.3f}"
                )
        if alert_needed:
            failed_runs += 1
        else:
            failed_runs = 0

        if failed_runs >= MAX_FAILED_RUNS:
            if not RUNNING_CI:
                print("[Monitoring] Avvio retraining parziale dopo ripetuti fallimenti...")
                retrain_on_youtube_sample()
            failed_runs = 0  # reset contatore dopo retraining

        if RUNNING_CI:
            print("[Monitoring] RUNNING_CI attivo — esco dal loop per test GitHub Actions.")
            break

        time.sleep(EVALUATION_INTERVAL)


if __name__ == "__main__":
    main()
