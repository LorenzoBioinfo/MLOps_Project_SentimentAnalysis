from datasets import load_dataset
from transformers import AutoTokenizer
import argparse
import re
import os

MODEL_NAME = "cardiffnlp/twitter-roberta-base-sentiment-latest"
PROCESSED_DIR = "data/processed/"

if not os.path.exists(PROCESSED_DIR):
    os.makedirs(PROCESSED_DIR, exist_ok=True)


### Funzioni di supporto
def clean_text(text):
    """Pulisce il testo da URL, menzioni, hashtag, simboli HTM"""
    text = re.sub(r"http\S+", "", text)  
    text = re.sub(r"@\w+", "", text)     
    text = re.sub(r"#\w+", "", text)     
    text = re.sub(r"&[a-z]+;", "", text) 
    text = re.sub(r"\s+", " ", text)    
    return text.strip()

def map_label(label):
    """
    Mappa le etichette di sentiment a numeri.
    - 0: negativo
    - 1: neutro
    - 2: positivo
    """
    mapping = {"negative": 0, "neutral": 1, "positive": 2}
    if isinstance(label, str):
        return mapping.get(label.lower(), 1)
    return label


tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

def tokenize_function(examples):
    return tokenizer(
        examples["text"],
        truncation=True,
        padding="max_length",
        max_length=128,
    )

#tweet_tokenized = tweet_eval.map(tokenize_function, batched=True)
#youtube_tokenized = youtube.map(tokenize_function, batched=True)

#tweet_tokenized.save_to_disk(os.path.join(PROCESSED_DIR, "tweet_eval_tokenized"))
#youtube_tokenized.save_to_disk(os.path.join(PROCESSED_DIR, "youtube_tokenized"))

def prepare_tweet_eval(tokenizer, output_path):
    print("Scarico e preparo il dataset Tweet Eval...")
    ds = load_dataset("tweet_eval", "sentiment")
    ds=ds.map(lambda x: {"text": clean_text(x["text"])})
    ds=ds.map(tokenize_function, batched=True)
    ds.save_to_disk(output_path)
    print(f"Dataset Tweet Eval salvato in {output_path}")

def prepare_youtube(tokenizer, output_path):
    print("Scarico e preparo il dataset YouTube Comments...")
    ds = load_dataset("AmaanP314/youtube-comment-sentiment")
    ds = ds.map(lambda x: {"text": clean_text(x["CommentText"])})
    ds = ds.map(lambda x: {"label": map_label(x["Sentiment"])})
    ds.save_to_disk(output_path)
    print(f"Dataset YouTube salvato in {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepara dataset per sentiment analysis.")
    parser.add_argument("dataset", choices=["tweet_eval", "youtube"], help="Nome del dataset da preparare.")
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained("cardiffnlp/twitter-roberta-base-sentiment-latest")

    if args.dataset == "tweet_eval":
        prepare_tweet_eval(tokenizer, "data/processed/tweet_eval_tokenized")
    elif args.dataset == "youtube":
        prepare_youtube(tokenizer, "data/processed/youtube_tokenized")