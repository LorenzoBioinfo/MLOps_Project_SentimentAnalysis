import os
import random
import subprocess
import json
from pathlib import Path

from fastapi import FastAPI, Request, Form, HTTPException, Depends, Cookie
from fastapi.responses import HTMLResponse, RedirectResponse, FileResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from datasets import load_from_disk
import torch
from dotenv import load_dotenv

# --------------------------------------------------
# ENV
# --------------------------------------------------
load_dotenv()

MODEL_NAME = "cardiffnlp/twitter-roberta-base-sentiment-latest"
BASE_DIR = os.getenv("BASE_DIR", "/app")

TWEET_PROCESSED_PATH = os.path.join(BASE_DIR, "data/processed/tweet_eval_tokenized")
YT_PROCESSED_PATH = os.path.join(BASE_DIR, "data/processed/youtube_tokenized")

# Ora i report del monitoring vanno in /app/reports/monitoring
REPORTS_DIR = os.path.join(BASE_DIR, "reports")
MONITORING_DIR = os.path.join(REPORTS_DIR, "monitoring")

ADMIN_API_KEY = os.getenv("ADMIN_API_KEY")
if not ADMIN_API_KEY:
    raise RuntimeError("ADMIN_API_KEY non impostata")

# --------------------------------------------------
# MODELLO
# --------------------------------------------------
print("[App] Caricamento modello...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, cache_dir=f"{BASE_DIR}/huggingface_cache")
model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, cache_dir=f"{BASE_DIR}/huggingface_cache")
model.eval()

LABELS = ["negative", "neutral", "positive"]

# --------------------------------------------------
# FASTAPI + TEMPLATE
# --------------------------------------------------
app = FastAPI(title="Sentiment Analysis API")
templates = Jinja2Templates(directory=f"{BASE_DIR}/app_templates")

# Static route per i reports (grafici)
app.mount("/reports", StaticFiles(directory=REPORTS_DIR), name="reports")

# --------------------------------------------------
# DATASET
# --------------------------------------------------
def load_or_prepare_dataset(path, name):
    if not os.path.exists(path):
        subprocess.run(
            ["python", "src/data_preparation.py", name, f"{BASE_DIR}/data/processed"],
            check=True
        )
    return load_from_disk(path)

print("[App] Caricamento dataset...")
tweet_eval = load_or_prepare_dataset(TWEET_PROCESSED_PATH, "tweet_eval")
youtube_ds = load_or_prepare_dataset(YT_PROCESSED_PATH, "youtube")
print("[App] Dataset pronti")

# --------------------------------------------------
# PREDIZIONE
# --------------------------------------------------
def predict_sentiment(text: str):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    with torch.no_grad():
        outputs = model(**inputs)
        probs = torch.softmax(outputs.logits, dim=-1)
        pred = torch.argmax(probs, dim=1).item()
    return {
        "label": LABELS[pred],
        "confidence": round(probs[0][pred].item(), 3)
    }

# --------------------------------------------------
# ROTTE PUBBLICHE
# --------------------------------------------------
@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.get("/predict", response_class=HTMLResponse)
def predict_page(request: Request):
    return templates.TemplateResponse("predict.html", {"request": request})

@app.post("/predict", response_class=HTMLResponse)
def predict_text(request: Request, text: str = Form(...)):
    result = predict_sentiment(text)
    return templates.TemplateResponse(
        "predict.html",
        {"request": request, "text": text, "result": result}
    )

@app.get("/random_tweet", response_class=HTMLResponse)
def random_tweet(request: Request):
    sample = random.choice(tweet_eval["test"])
    text = sample["text"]
    result = predict_sentiment(text)
    true_label = LABELS[sample["label"]]

    return templates.TemplateResponse(
        "random_tweet.html",
        {
            "request": request,
            "text": text,
            "true_label": true_label,
            "result": result
        }
    )

@app.get("/random_youtube_comment", response_class=HTMLResponse)
def random_youtube(request: Request):
    sample = random.choice(youtube_ds["train"])
    text = sample["text"]
    true_label = LABELS[sample["label"]]
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

# --------------------------------------------------
# ADMIN AUTH
# --------------------------------------------------
def verify_admin(admin_session: str = Cookie(None)):
    if admin_session != ADMIN_API_KEY:
        raise HTTPException(status_code=401)
    return True

@app.get("/admin_login")
async def admin_login():
    response = RedirectResponse(url="/admin", status_code=303)
    response.set_cookie("admin_session", ADMIN_API_KEY, httponly=True)
    return response

# --------------------------------------------------
# ADMIN DASHBOARD
# --------------------------------------------------
@app.get("/admin", response_class=HTMLResponse)
async def admin_dashboard(request: Request, _: bool = Depends(verify_admin)):
    metrics = None
    metrics_path = os.path.join(REPORTS_DIR, "metrics.json")

    if os.path.exists(metrics_path):
        with open(metrics_path) as f:
            metrics = json.load(f)

    return templates.TemplateResponse(
        "admin.html",
        {"request": request, "metrics": metrics}
    )

# --------------------------------------------------
# TRAINING
# --------------------------------------------------
@app.post("/admin/train", response_class=HTMLResponse)
async def run_training(request: Request, _: bool = Depends(verify_admin)):
    subprocess.Popen(["python", "src/train_model.py"])
    return templates.TemplateResponse(
        "train_started.html",
        {
            "request": request,
            "message": "✅ Training avviato correttamente"
        }
    )

# --------------------------------------------------
# METRICHE (nuovo)
# --------------------------------------------------
def monitoring_run_list():
    """Ritorna lista di run presenti in /reports/monitoring"""
    base = Path(MONITORING_DIR)
    if not base.exists():
        return []
    return sorted([p.name for p in base.iterdir() if p.is_dir()])


@app.get("/admin/metrics", response_class=HTMLResponse)
async def view_metrics(request: Request, _: bool = Depends(verify_admin)):
    runs = monitoring_run_list()
    return templates.TemplateResponse("metrics.html", {"request": request, "runs": runs})


@app.get("/admin/metrics/run/{run_id}", response_class=HTMLResponse)
async def view_run(request: Request, run_id: str, _: bool = Depends(verify_admin)):
    run_dir = Path(MONITORING_DIR) / run_id

    if not run_dir.exists():
        raise HTTPException(status_code=404, detail="Run non trovata")

    tweet_metrics = json.loads((run_dir / "tweet_metrics.json").read_text())
    yt_metrics = json.loads((run_dir / "youtube_metrics.json").read_text())

    return templates.TemplateResponse(
        "run_detail.html",
        {
            "request": request,
            "run_id": run_id,
            "tweet": tweet_metrics,
            "youtube": yt_metrics
        }
    )


@app.get("/admin/metrics/run/{run_id}/artifact/{filename}")
async def get_artifact(run_id: str, filename: str, _: bool = Depends(verify_admin)):
    run_dir = Path(MONITORING_DIR) / run_id
    file_path = run_dir / filename

    if not file_path.exists():
        raise HTTPException(status_code=404, detail="Artifact non trovata")

    return FileResponse(file_path)

# --------------------------------------------------
# ADMIN UNAUTHORIZED
# --------------------------------------------------
@app.exception_handler(HTTPException)
async def admin_exception_handler(request: Request, exc: HTTPException):
    if request.url.path.startswith("/admin"):
        return templates.TemplateResponse("admin_unknown.html", {"request": request})
    return HTMLResponse(str(exc.detail), status_code=exc.status_code)

# --------------------------------------------------
# AVVIO
# --------------------------------------------------
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
