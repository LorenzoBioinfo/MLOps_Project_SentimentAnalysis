from datasets import load_dataset, Dataset
from transformers import AutoTokenizer
import argparse
import re
import os
import time


MODEL_NAME = "cardiffnlp/twitter-roberta-base-sentiment-latest"
PROCESSED_DIR = "data/processed/"

os.makedirs(PROCESSED_DIR, exist_ok=True)


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
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

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

def safe_load_dataset(name, config=None, max_retries=3, fallback_data=None):
    """
    Gestisce i retry del download e crea un dataset di fallback se fallisce.
    """
    for attempt in range(max_retries):
        try:
            if config:
                return load_dataset(name, config)
            return load_dataset(name)
        except Exception as e:
            print(f"Tentativo {attempt+1}/{max_retries} fallito per {name}: {e}")
            if attempt < max_retries - 1:
                time.sleep(10)
            else:
                print(f"Errore persistente nel download {name}. Uso dataset di fallback.")
                if fallback_data:
                    return Dataset.from_dict(fallback_data).train_test_split(test_size=0.4)
                raise e


def prepare_tweet_eval(tokenizer, output_path):
    print("Scarico e preparo il dataset Tweet Eval...")
    fallback_data = {
        "text": ["I love this!", "This is bad", "Just okay", "Great!", "Terrible experience"],
        "label": [2, 0, 1, 2, 0],
    }
    ds = safe_load_dataset("tweet_eval", "sentiment", fallback_data=fallback_data)
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
    fallback_data = {
        "CommentText": ["Amazing video!", "I hated this", "Not bad", "Loved it", "Awful content"],
        "Sentiment": ["positive", "negative", "neutral", "positive", "negative"],
    }
    ds = safe_load_dataset("AmaanP314/youtube-comment-sentiment", fallback_data=fallback_data)
    ds = ds["train"].select(range(1000))
    ds = ds.map(lambda x: {"text": clean_text(x["CommentText"])})
    ds = ds.map(lambda x: {"label": map_label(x["Sentiment"])})
    ds = ds.map(tokenize_function, batched=True)
    ds.save_to_disk(output_path)
    print(f"Dataset YouTube salvato in {output_path}")



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepara dataset per sentiment analysis.")
    parser.add_argument("dataset", choices=["tweet_eval", "youtube"], help="Nome del dataset da preparare.")
    args = parser.parse_args()

    if args.dataset == "tweet_eval":
        prepare_tweet_eval(tokenizer, os.path.join(PROCESSED_DIR, "tweet_eval_tokenized"))
    elif args.dataset == "youtube":
        prepare_youtube(tokenizer, os.path.join(PROCESSED_DIR, "youtube_tokenized"))
