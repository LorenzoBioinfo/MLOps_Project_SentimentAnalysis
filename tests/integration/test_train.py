import os
import shutil
import pytest
from src.train_model import train_model

MODEL_DIR = "models/sentiment_model"

@pytest.fixture(autouse=True)
def cleanup():
    if os.path.exists(MODEL_DIR):
        shutil.rmtree(MODEL_DIR)
    yield
    if os.path.exists(MODEL_DIR):
        shutil.rmtree(MODEL_DIR)

def test_train_model_runs():
    """Testa che il training parta e salvi un modello."""
    train_model(sample_train_size=10, sample_eval_size=5) 
    assert os.path.exists(MODEL_DIR), "La directory del modello non è stata creata"
    assert os.path.exists(os.path.join(MODEL_DIR, "config.json")), "File config.json mancante"
