from fastapi.testclient import TestClient
from src.app import app
import os 

os.environ["SKIP_DATA_PREP"] = "true"

client = TestClient(app)

def test_home_page():
    response = client.get("/")
    assert response.status_code == 200
    assert "Benvenuto" in response.text

def test_predict_endpoint_get():
    response = client.get("/predict")
    assert response.status_code == 200
    assert "Testa il Modello" in response.text

def test_predict_endpoint_post():
    response = client.post("/predict", data={"text": "I love this!"})
    assert response.status_code == 200
    assert any(label in response.text for label in ["positive", "neutral", "negative"])

def test_random_tweet_page():
    response = client.get("/random_tweet")
    assert response.status_code == 200
    assert any(lbl in response.text for lbl in ["positive", "neutral", "negative", "Positivo", "Neutro", "Negativo"])
   
   
def test_random_youtube_page():
    response = client.get("/random_youtube_comment")
    assert response.status_code == 200
    assert any(lbl in response.text for lbl in ["positive", "neutral", "negative", "Positivo", "Neutro", "Negativo"])


