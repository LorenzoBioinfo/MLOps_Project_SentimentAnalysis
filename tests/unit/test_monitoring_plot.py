import os
from monitoring.src.monitoring import plot_accuracy_trend

def test_plot_accuracy_trend(tmp_path):
    history = [
        {"TweetEval": {"accuracy": 0.9}, "YouTube": {"accuracy": 0.8}},
        {"TweetEval": {"accuracy": 0.92}, "YouTube": {"accuracy": 0.85}},
    ]
    out = tmp_path / "trend.jpg"
    plot_accuracy_trend(history, str(out))
    assert os.path.exists(out)
