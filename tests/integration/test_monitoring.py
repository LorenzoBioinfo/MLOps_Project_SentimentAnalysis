import os
import json
import pytest
from src.monitoring import main, REPORTS_DIR

METRICS_PATH = os.path.join(REPORTS_DIR, "metrics.json")


@pytest.fixture(autouse=True)
def cleanup_metrics():
    if os.path.exists(METRICS_PATH):
        os.remove(METRICS_PATH)
    yield
    if os.path.exists(METRICS_PATH):
        os.remove(METRICS_PATH)


def test_monitoring_creates_metrics():
    """Verifica che il monitoring crei il file metrics.json con i dati previsti."""
    os.environ["RUNNING_CI"] = "1"  
    main()

    assert os.path.exists(METRICS_PATH), "metrics.json non è stato generato"

    with open(METRICS_PATH, "r") as f:
        all_metrics = json.load(f)

    metrics = all_metrics[-1] 

    assert "TweetEval" in metrics, "Mancano metriche TweetEval"
    assert "YouTube Comments" in metrics, "Mancano metriche YouTube"

    for dataset_name, data in metrics.items():
        assert "accuracy" in data, f"Manca accuracy per {dataset_name}"
        assert "f1" in data, f"Manca F1 per {dataset_name}"
        assert "precision" in data, f"Manca precision per {dataset_name}"
        assert "recall" in data, f"Manca recall per {dataset_name}"
