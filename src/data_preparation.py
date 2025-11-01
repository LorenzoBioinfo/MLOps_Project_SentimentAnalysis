from datasets import load_dataset
from transformers import AutoTokenizer
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


# Download tweet_eval
tweet_eval = load_dataset("tweet_eval", "sentiment")
# Download youtub comment dataset
youtube = load_dataset("AmaanP314/youtube-comment-sentiment")


tweet_eval = tweet_eval.map(lambda x: {"text": clean_text(x["text"])})
youtube = youtube.map(lambda x: {"text": clean_text(x["CommentText"])})


youtube = youtube.map(lambda x: {"label": map_label(x["Sentiment"])})


tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

def tokenize_function(examples):
    return tokenizer(
        examples["text"],
        truncation=True,
        padding="max_length",
        max_length=128,
    )

tweet_tokenized = tweet_eval.map(tokenize_function, batched=True)
youtube_tokenized = youtube.map(tokenize_function, batched=True)

tweet_tokenized.save_to_disk(os.path.join(PROCESSED_DIR, "tweet_eval_tokenized"))
youtube_tokenized.save_to_disk(os.path.join(PROCESSED_DIR, "youtube_tokenized"))
