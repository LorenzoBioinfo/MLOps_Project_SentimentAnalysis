from datasets import load_dataset,DatasetDict
from transformers import AutoTokenizer
import argparse
import re
import os
import time

MODEL_NAME = "cardiffnlp/twitter-roberta-base-sentiment-latest"
PROCESSED_DIR = "data/processed/"
HF_CACHE_DIR = os.getenv("HF_CACHE_DIR", "/app/huggingface_cache")

os.makedirs(PROCESSED_DIR, exist_ok=True)

BASE_DIR = os.getenv("BASE_DIR", "/app")
#     FUNZIONI DI SUPPORTO     

def clean_text(text):
    """Pulisce il testo da URL, menzioni, hashtag, simboli HTML"""
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"@\w+", "", text)
    text = re.sub(r"#\w+", "", text)
    text = re.sub(r"&[a-z]+;", "", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()

def map_label(label):
    """Mappa le etichette di sentiment a numeri"""
    mapping = {"negative": 0, "neutral": 1, "positive": 2}
    if isinstance(label, str):
        return mapping.get(label.lower(), 1)
    return label

# Tokenizer globale
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME,cache_dir=HF_CACHE_DIR)

def tokenize_function(examples):
    return tokenizer(
        examples["text"],
        truncation=True,
        padding="max_length",
        max_length=128,
    )


# ----------------------------- #
#   PREPARAZIONE DEI DATASET    #
# ----------------------------- #

def safe_load_dataset(name, config=None, max_retries=3, wait_time=10):
    """
    Tenta di scaricare un dataset con retry progressivi.
    Se fallisce dopo tutti n tentativi, fallisce.
    """
    for attempt in range(max_retries):
        try:
            print(f"Tentativo {attempt+1}/{max_retries} di scaricare il dataset '{name}'...")
            if config:
                dataset = load_dataset(name, config)
            else:
                dataset = load_dataset(name)
            print(f"Dataset '{name}' caricato correttamente.")
            return dataset
        except Exception as e:
            print(f"Errore al tentativo {attempt+1}: {e}")
            if attempt < max_retries - 1:
                print(f"Attendo {wait_time}s prima di riprovare...")
                time.sleep(wait_time)
            else:
                print(
                    f"Impossibile scaricare il dataset '{name}' dopo {max_retries} tentativi.\n"
                    "Verifica la connessione Internet o il nome del dataset.\n"
                )
                raise


def prepare_tweet_eval(tokenizer, output_path):
    print("Scarico e preparo il dataset Tweet Eval...")

    ds = safe_load_dataset("tweet_eval", "sentiment")
    if isinstance(ds, dict) or "train" in ds:
        reduced_splits = {}
        for split in ds.keys():
            reduced_splits[split] = ds[split].select(range(min(1000, len(ds[split]))))
            reduced_splits[split] = reduced_splits[split].map(lambda x: {"text": clean_text(x["text"])})
            reduced_splits[split] = reduced_splits[split].map(tokenize_function, batched=True)
        ds = DatasetDict(reduced_splits)
    else:
        ds = ds.select(range(min(1000, len(ds))))
        ds = ds.map(lambda x: {"text": clean_text(x["text"])})
        ds = ds.map(tokenize_function, batched=True)

    ds.save_to_disk(output_path)
    print(f"Dataset Tweet Eval salvato in {output_path}")


def prepare_youtube(tokenizer, output_path):
    print("📥 Scarico e preparo il dataset YouTube Comments...")

    ds = safe_load_dataset("AmaanP314/youtube-comment-sentiment")
  
    if isinstance(ds, dict) or "train" in ds:
        reduced_splits = {}
        for split in ds.keys():
            reduced_splits[split] = ds[split].select(range(min(1000, len(ds[split]))))
            reduced_splits[split] = reduced_splits[split].map(
                lambda x: {
                    "text": clean_text(x["CommentText"]),
                    "label": map_label(x["Sentiment"]),
                }
            )
            reduced_splits[split] = reduced_splits[split].map(tokenize_function, batched=True)
        ds = DatasetDict(reduced_splits)
    else:
 
        ds = ds.select(range(min(1000, len(ds))))
        ds = ds.map(
            lambda x: {
                "text": clean_text(x["CommentText"]),
                "label": map_label(x["Sentiment"]),
            }
        )
    ds.save_to_disk(output_path)
    print(f"Dataset YouTube salvato in {output_path}")





if __name__ == "__main__":
 

    parser = argparse.ArgumentParser(description="Prepara dataset per sentiment analysis.")
    parser.add_argument("dataset", choices=["tweet_eval", "youtube"], help="Nome del dataset da preparare.")
    parser.add_argument(
        "--output_dir",
        type=str,
        default="data/processed", 
        help="Directory dove salvare i dataset preprocessati"
    )
    args = parser.parse_args()

 
    output_base = args.output_dir

    if args.dataset == "tweet_eval":
        prepare_tweet_eval(tokenizer, os.path.join(output_base, "tweet_eval_tokenized"))
    elif args.dataset == "youtube":
        prepare_youtube(tokenizer, os.path.join(output_base, "youtube_tokenized"))
