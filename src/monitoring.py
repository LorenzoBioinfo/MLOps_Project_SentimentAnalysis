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
BASE_DIR = os.getenv("BASE_DIR", ".")
MODEL_PATH = f"{BASE_DIR}/models/sentiment_model"
TWEET_PATH = f"{BASE_DIR}/data/processed/tweet_eval_tokenized"
YT_PATH = f"{BASE_DIR}/data/processed/youtube_tokenized"
REPORTS_DIR = f"{BASE_DIR}/reports"
PROMETHEUS_PORT = 8000

# Flag per test/CI
RUNNING_CI = os.getenv("RUNNING_CI") == "1"

#  PROMETHEUS METRICS 
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
    accuracy_gauge.labels(dataset="TweetEval").set(tweet_metrics["accuracy"])
    f1_gauge.labels(dataset="TweetEval").set(tweet_metrics["f1"])
    precision_gauge.labels(dataset="TweetEval").set(tweet_metrics["precision"])
    recall_gauge.labels(dataset="TweetEval").set(tweet_metrics["recall"])

    accuracy_gauge.labels(dataset="YouTube").set(youtube_metrics["accuracy"])
    f1_gauge.labels(dataset="YouTube").set(youtube_metrics["f1"])
    precision_gauge.labels(dataset="YouTube").set(youtube_metrics["precision"])
    recall_gauge.labels(dataset="YouTube").set(youtube_metrics["recall"])

# MODEL EVALUATION
def evaluate_model(model, dataset, dataset_name, sample_size=300):
    print(f"[Monitoring] Valutazione su {dataset_name}")
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
    precision = precision_score(labels.numpy(), preds.numpy(), average="weighted")
    recall = recall_score(labels.numpy(), preds.numpy(), average="weighted")

    print(f"[Monitoring] {dataset_name} — Accuracy: {acc:.3f}, F1: {f1:.3f}")
    return {"dataset": dataset_name, "accuracy": acc, "f1": f1,
            "precision": precision, "recall": recall, "confusion_matrix": cm}

# RETRAINING WITH YOUTUBE DATA
def retrain_on_youtube_sample():
    print("[Monitoring] Preparazione retraining dati YouTube...")
    youtube_data = load_from_disk(YT_PATH)["train"]
    youtube_sample = youtube_data.shuffle(seed=42).select(range(500))
    youtube_sample = youtube_sample.remove_columns(
        [col for col in youtube_sample.column_names if col not in ["text", "label"]]
    )
    label_class = ClassLabel(names=["negative", "neutral", "positive"])
    youtube_sample = youtube_sample.cast_column("label", label_class)
    train_model(additional_data=youtube_sample, output_dir=MODEL_PATH)

# ALERTING 
def send_alert(message):
    """Alert semplice: stampa su console"""
    print(f"[ALERT] {message}")

#_____MAIN______
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

    tweet_ds = load_from_disk(TWEET_PATH)
    youtube_ds = load_from_disk(YT_PATH)

    tweet_metrics = evaluate_model(model, tweet_ds, "TweetEval")
    youtube_metrics = evaluate_model(model, youtube_ds, "YouTube Comments")

   
   

    # Alerting e retraining
    if youtube_metrics["accuracy"] < ACCURACY_THRESHOLD:
        send_alert(f"Performance YouTube sotto soglia: {youtube_metrics['accuracy']:.3f}")
        if not RUNNING_CI:
            print("[Monitoring] Avvio retraining parziale...")
            retrain_on_youtube_sample()

    # Salvataggio storico JSON
    os.makedirs(REPORTS_DIR, exist_ok=True)
    metrics_path = os.path.join(REPORTS_DIR, "metrics.json")

    results = {"TweetEval": tweet_metrics, "YouTube": youtube_metrics}
    all_results = []
    try:
        with open(metrics_path, "r") as f:
            all_results = json.load(f)
    except FileNotFoundError:
        print(f"[Monitoring] Nessun file metrics.json trovato, ne creo uno nuovo.")

    all_results.append(results)

    # Aggiorna metriche Prometheus
    last_run = all_results[-1]
    update_metrics(
    tweet_metrics=last_run["TweetEval"],
    youtube_metrics=last_run["YouTube"]
    )

    with open(metrics_path, "w") as f:
        json.dump(all_results, f, indent=4)

    print(f"[Monitoring] Risultati salvati in: {metrics_path}")
    print("[Monitoring] Monitoring attivo. In attesa di scrape Prometheus...")
    if not RUNNING_CI:
        print("[Monitoring] Monitoring attivo. In attesa di scrape Prometheus...")
        while True:
            time.sleep(60)
  

if __name__ == "__main__":
    main()
