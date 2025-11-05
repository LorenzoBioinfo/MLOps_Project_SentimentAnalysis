
from transformers import (
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments
)
from datasets import load_from_disk,concatenate_datasets
import evaluate
import numpy as np
import os
from huggingface_hub import HfApi


hf_token = os.environ["HF_TOKEN"]

#

MODEL_NAME = "cardiffnlp/twitter-roberta-base-sentiment-latest"
DATA_PATH = "data/processed/tweet_eval_tokenized"
OUTPUT_DIR = "models/sentiment_model"
HF_REPO = "Lordemarco/SentimentAnalysis" 

def compute_metrics(eval_pred):
    """Calcola metriche standard: accuracy e F1."""
    metric_acc = evaluate.load("accuracy")
    metric_f1 = evaluate.load("f1")
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    acc = metric_acc.compute(predictions=predictions, references=labels)
    f1 = metric_f1.compute(predictions=predictions, references=labels, average="weighted")
    return {"accuracy": acc["accuracy"], "f1": f1["f1"]}


def train_model(additional_data=None,sample_train_size=1000, sample_eval_size=300,output_dir=OUTPUT_DIR):
    print("Caricamento dataset Tweet eval preprocessato")
    dataset = load_from_disk(DATA_PATH)
    if additional_data is not None:
        print("Aggiungo dati YouTube al training set...")
        dataset["train"] = concatenate_datasets([dataset["train"], additional_data])

    # 
    print(f"Riduzione dataset: {sample_train_size} per il train, {sample_eval_size} per la validazione.")
    train_data = dataset["train"].select(range(min(sample_train_size, len(dataset["train"]))))
    eval_data = dataset["validation"].select(range(min(sample_eval_size, len(dataset["validation"]))))

   
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)

    # Parametri training
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        num_train_epochs=1,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=32,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        logging_dir="./logs",
        logging_steps=10,
        load_best_model_at_end=True,
        report_to="none", 
    )

    print("Avvio training")
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_data,
        eval_dataset=eval_data,
        compute_metrics=compute_metrics,
    )

    trainer.train()

    os.makedirs(output_dir, exist_ok=True)
    trainer.save_model(output_dir)
    print(f"Modello salvato in: {OUTPUT_DIR}")
    

    if os.getenv("HF_TOKEN"):
        print("Pushing model to Hugging Face Hub...")
        trainer.push_to_hub("Lordemarco/SentimentAnalysis")
       
if __name__ == "__main__":
    train_model()
