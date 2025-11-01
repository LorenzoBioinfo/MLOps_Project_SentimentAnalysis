import os
from fastapi import FastAPI, Request, Form
from pydantic import BaseModel
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from datasets import load_dataset, load_from_disk
import torch
import random

# Caricamento del modello e dei dati se già scaricati
MODEL= "cardiffnlp/twitter-roberta-base-sentiment-latest"
TWEET_PROCESSED_PATH = "data/processed/tweet_eval_tokenized"
YT_PROCESSED_PATH = "data/processed/youtube_tokenized"

tokenizer = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForSequenceClassification.from_pretrained(MODEL)


labels = ["negative", "neutral", "positive"]


if not os.path.exists(TWEET_PROCESSED_PATH):
    tweet_eval = load_dataset("tweet_eval", "sentiment")
    raise FileNotFoundError(
        f"Dati non trovati in {TWEET_PROCESSED_PATH}. "
        "Esegui src/data_preparation.py per crearlo."
    ) 

tweet_eval = load_from_disk(TWEET_PROCESSED_PATH)


if not os.path.exists(YT_PROCESSED_PATH):
    youtube_ds = load_dataset("AmaanP314/youtube-comment-sentiment")
    raise FileNotFoundError(
        f"Dati non trovati in {YT_PROCESSED_PATH}. "
        "Esegui src/data_preparation.py per crearlo."
    ) 

youtube_ds = load_from_disk(YT_PROCESSED_PATH)

app = FastAPI(
    title="Sentiment Analysis API",
    description="Testa il modello RoBERTa di CardiffNLP su frasi personalizzate o su esempi random dal dataset TweetEval."
)
templates = Jinja2Templates(directory="app_templates/")

class TextInput(BaseModel):
    text: str


def predict_sentiment(text: str):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    with torch.no_grad():
        outputs = model(**inputs)
        probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
        pred = torch.argmax(probs, dim=1).item()
        confidence = probs[0][pred].item()
    return {"label": labels[pred], "confidence": round(confidence, 3)}


@app.get("/",response_class=HTMLResponse)
async def home( request: Request):
    #return "Ciao Mondo!"
    #return {"message": "Benvenuto nell'App di MachineInnovators Inc. per la sentiment analysis. Usa /predict o /random_tweet."}
    return templates.TemplateResponse("index.html", {"request": request})
    
@app.get("/random_tweet", response_class=HTMLResponse)
def random_tweet(request: Request):
    sample = random.choice(tweet_eval["test"])
    text = sample["text"] if "text" in sample else tokenizer.decode(sample["input_ids"], skip_special_tokens=True)
    result = predict_sentiment(text)

    
   
    true_label=labels[sample["label"]]

    return templates.TemplateResponse(
        "random_tweet.html",
        {
            "request": request,
            "text": text,
            "true_label": true_label,
            "result": result
        }
    )





@app.get("/predict", response_class=HTMLResponse)
def predict_page(request: Request):
    return templates.TemplateResponse("predict.html", {"request": request, "result": None})

@app.post("/predict", response_class=HTMLResponse)
def predict_text(request: Request, text: str = Form(...)):
    result = predict_sentiment(text)
    return templates.TemplateResponse(
        "predict.html",
        {"request": request, "text": text, "result": result}
    )


@app.get("/random_youtube_comment", response_class=HTMLResponse)
def random_youtube_comment(request: Request):
    sample = random.choice(youtube_ds["train"])  

    text = sample["text"] if "text" in sample else sample["text"]
    true_label = sample["label"] if "label" in sample else "N/A"

    if isinstance(true_label, int):
        
        label_map = {0: "negative", 1: "neutral", 2: "positive"}
        true_label = label_map.get(true_label, "N/A")

    result = predict_sentiment(text)

    return templates.TemplateResponse(
        "random_youtube.html",
        {
            "request": request,
            "text": text,
            "true_label": true_label,
            "result": result
        }
    )


if __name__=="__main__":
    import uvicorn
    uvicorn.run(app,host="0.0.0.0",port=8000)