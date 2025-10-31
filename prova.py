from transformers import AutoTokenizer, AutoModelForSequenceClassification
from scipy.special import softmax
import torch

MODEL = "cardiffnlp/twitter-roberta-base-sentiment-latest"
tokenizer = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForSequenceClassification.from_pretrained(MODEL)


text = "Terrible"


inputs = tokenizer(text, return_tensors="pt")

with torch.no_grad():
    logits = model(**inputs).logits
    probs = softmax(logits.numpy()[0])


labels = ["negative", "neutral", "positive"]
for label, prob in zip(labels, probs):
    print(f"{label}: {prob:.4f}")
