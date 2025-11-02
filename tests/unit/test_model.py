from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

MODEL_NAME = "cardiffnlp/twitter-roberta-base-sentiment-latest"
LABELS = ["negative", "neutral", "positive"]

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)

def test_model_loads():
    assert model is not None
    assert tokenizer is not None

def test_model_prediction_shape():
    text = "I love this product!"
    inputs = tokenizer(text, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs)
    assert outputs.logits.shape[-1] == len(LABELS)

def test_sentiment_confidence():
    text = "I hate this"
    inputs = tokenizer(text, return_tensors="pt")
    with torch.no_grad():
        probs = torch.nn.functional.softmax(model(**inputs).logits, dim=-1)
    assert torch.isclose(probs.sum(), torch.tensor(1.0), atol=1e-3)