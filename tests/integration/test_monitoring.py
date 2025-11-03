import os
import json
import pytest
from src.monitoring import monitor_model

METRICS_PATH = "reports/metrics.json"

@pytest.fixture(autouse=True)
def cleanup_metrics():
    """Pulisce file metrics prima del test."""
    if os.path.exists(METRICS_PATH):
        os.remove(METRICS_PATH)
    yield
    if os.path.exists(METRICS_PATH):
        os.remove(METRICS_PATH)

def test_monitoring_creates_metrics():
    """Verifica che il monitoring crei il file metrics.json."""
    monitor_model()
    assert os.path.exists(METRICS_PATH), "metrics.json non è stato generato"

    with open(METRICS_PATH, "r") as f:
        metrics = json.load(f)
    assert "accuracy" in metrics and "f1" in metrics, "Metriche principali mancanti"
