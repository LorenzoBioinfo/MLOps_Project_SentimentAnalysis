import os
from fastapi import FastAPI, Request, Form,HTTPException,status
from pydantic import BaseModel
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from datasets import  load_from_disk
import torch
import random
import subprocess
import json

# Caricamento del modello e dei dati se già scaricati
MODEL= "cardiffnlp/twitter-roberta-base-sentiment-latest"
TWEET_PROCESSED_PATH = "data/processed/tweet_eval_tokenized"
YT_PROCESSED_PATH = "data/processed/youtube_tokenized"

tokenizer = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForSequenceClassification.from_pretrained(MODEL)


labels = ["negative", "neutral", "positive"]

def load_or_prepare_dataset(path, dataset_name):
    """
    Carica un dataset da disco se esiste, altrimenti lo genera eseguendo lo script di preparazione.
    Usa la variabile d'ambiente SKIP_DATA_PREP per saltare la preparazione se impostata.
    """
    if not os.path.exists(path):
        print(f"Dataset '{dataset_name}' non trovato in {path}.")
        if not os.environ.get("SKIP_DATA_PREP"):
            subprocess.run(["python", "src/data_preparation.py", dataset_name], check=True)
        else:
            print(f"Preparazione saltata per '{dataset_name}' (SKIP_DATA_PREP attivo).")

    print(f"Carico il dataset '{dataset_name}' da {path}")
    return load_from_disk(path)


tweet_eval = load_or_prepare_dataset(TWEET_PROCESSED_PATH, "tweet_eval")
youtube_ds = load_or_prepare_dataset(YT_PROCESSED_PATH, "youtube")


app = FastAPI(
    title="Sentiment Analysis API"
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
    return templates.TemplateResponse("index.html", {"request": request})
    
@app.get("/random_tweet", response_class=HTMLResponse)
def random_tweet(request: Request):
    sample = tweet_eval["test"][random.randrange(len(tweet_eval["test"]))]
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


ADMIN_API_KEY = os.getenv("ADMIN_API_KEY")
if not ADMIN_API_KEY:
    raise RuntimeError(
        "Variabile d'ambiente ADMIN_API_KEY non impostata!\n"
        "Impostala prima di avviare l'app, ad esempio:\n"
        "  export ADMIN_API_KEY='la_tua_chiave_super_segreta'\n"
        "Oppure inseriscila nel file .env"
    )

def verify_api_key(request: Request):
    """Verifica la presenza e la validità della chiave API nell'header."""
    client_key = request.headers.get("x-api-key")
    if client_key != ADMIN_API_KEY:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Accesso non autorizzato.")


@app.get("/admin", response_class=HTMLResponse)
async def admin_dashboard(request: Request):
    verify_api_key(request)
    metrics_path = "reports/metrics.json"
    metrics = None
    if os.path.exists(metrics_path):
        with open(metrics_path, "r") as f:
            metrics = json.load(f)
    return templates.TemplateResponse("admin.html", {"request": request, "metrics": metrics})


@app.post("/admin/train")
async def retrain_model(request: Request):
    verify_api_key(request)
    subprocess.run(["python", "src/train_model.py"], check=True)
    return {"status": "Training completato"}


@app.post("/admin/monitor")
async def run_monitoring(request: Request):
    verify_api_key(request)
    subprocess.run(["python", "src/monitoring.py"], check=True)
    return {"status": "Monitoring completato"}


@app.get("/admin/metrics", response_class=HTMLResponse)
async def view_metrics(request: Request):
    verify_api_key(request)
    metrics_path = "reports/metrics.json"
    metrics = None
    if os.path.exists(metrics_path):
        with open(metrics_path, "r") as f:
            metrics = json.load(f)
    return templates.TemplateResponse("metrics.html", {"request": request, "metrics": metrics})



if __name__=="__main__":
    import uvicorn
    uvicorn.run(app,host="0.0.0.0",port=8000)