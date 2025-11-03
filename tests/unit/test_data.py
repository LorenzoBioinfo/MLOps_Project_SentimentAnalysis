# tests/test_data.py
import os
import subprocess
from datasets import load_from_disk

TWEET_PROCESSED_PATH = "data/processed/tweet_eval_tokenized"
YT_PROCESSED_PATH = "data/processed/youtube_tokenized"

def run_data_preparation(dataset_name):
    """Esegue lo script di data preparation per il dataset richiesto."""
    print(f"⚙️  Avvio data_preparation.py per il dataset: {dataset_name}")
    subprocess.run(
        ["python", "src/data_preparation.py", "--dataset", dataset_name],
        check=True
    )

def test_tweet_eval_dataset_exists_or_create():
    """Controlla o crea il dataset Tweet Eval preprocessato."""
    if not os.path.exists(TWEET_PROCESSED_PATH):
        run_data_preparation("tweet_eval")
    assert os.path.exists(TWEET_PROCESSED_PATH), "Tweet Eval non disponibile dopo la preparazione"

def test_youtube_dataset_exists_or_create():
    """Controlla o crea il dataset YouTube preprocessato."""
    if not os.path.exists(YT_PROCESSED_PATH):
        run_data_preparation("youtube")
    assert os.path.exists(YT_PROCESSED_PATH), "YouTube dataset non disponibile dopo la preparazione"

def test_tweet_eval_structure():
    """Verifica che il dataset Tweet Eval abbia la struttura corretta."""
    ds = load_from_disk(TWEET_PROCESSED_PATH)
    assert "text" in ds["test"].features, "Campo 'text' mancante in Tweet Eval"
    assert "label" in ds["test"].features, "Campo 'label' mancante in Tweet Eval"

def test_youtube_structure():
    """Verifica che il dataset YouTube abbia la struttura corretta."""
    ds = load_from_disk(YT_PROCESSED_PATH)
    assert "CommentText" in ds["train"].features or "CommentText" in ds["train"].features, \
        "Campo testuale mancante in YouTube dataset"
    assert "Sentiment" in ds["train"].features, "Campo 'label' mancante in YouTube dataset"
