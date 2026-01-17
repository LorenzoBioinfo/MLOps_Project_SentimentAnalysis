import os
import json
import glob
import pytest
from unittest import mock

import monitoring


@pytest.fixture(autouse=True)
def env_setup(tmp_path, monkeypatch):
  
    monkeypatch.setenv("BASE_DIR", str(tmp_path))
    monkeypatch.setenv("RUNNING_CI", "1")

    
    monkeypatch.setattr(monitoring, "REPORTS_BASE", str(tmp_path / "reports" / "monitoring"))
    os.makedirs(monitoring.REPORTS_BASE, exist_ok=True)

    yield


def test_run_single_evaluation_creates_reports(monkeypatch):
   
    fake_model = mock.MagicMock()
    fake_tweet_ds = mock.MagicMock()
    fake_yt_ds = mock.MagicMock()

    
    fake_metrics_tweet = {
        "accuracy": 0.8,
        "f1": 0.75,
        "precision": 0.7,
        "recall": 0.6,
        "confusion_matrix": [[1, 0], [0, 1]],
        "num_samples": 10
    }

    fake_metrics_yt = {
        "accuracy": 0.78,
        "f1": 0.72,
        "precision": 0.69,
        "recall": 0.65,
        "confusion_matrix": [[1, 1], [0, 2]],
        "num_samples": 10
    }

    
    monkeypatch.setattr(monitoring, "load_model", lambda: fake_model)
    monkeypatch.setattr(monitoring, "load_from_disk", side_effect=[fake_tweet_ds, fake_yt_ds])
    monkeypatch.setattr(monitoring, "evaluate_model", side_effect=[fake_metrics_tweet, fake_metrics_yt])

    monitoring.run_single_evaluation()

   
    run_dirs = glob.glob(os.path.join(monitoring.REPORTS_BASE, "run_*"))
    assert len(run_dirs) == 1

    run_dir = run_dirs[0]

    assert os.path.exists(os.path.join(run_dir, "tweet_metrics.json"))
    assert os.path.exists(os.path.join(run_dir, "youtube_metrics.json"))
    assert os.path.exists(os.path.join(run_dir, "tweet_confusion_matrix.jpg"))
    assert os.path.exists(os.path.join(run_dir, "youtube_confusion_matrix.jpg"))
    assert os.path.exists(os.path.join(run_dir, "accuracy_trend.jpg"))


    history_file = os.path.join(monitoring.REPORTS_BASE, "history.json")
    assert os.path.exists(history_file)

    history = json.load(open(history_file))
    assert len(history) == 1
    assert history[0]["TweetEval"]["accuracy"] == 0.8
    assert history[0]["YouTube"]["accuracy"] == 0.78


def test_monitoring_loop_exits_on_ci(monkeypatch):
    
    monkeypatch.setenv("RUNNING_CI", "1")

    
    monkeypatch.setattr(monitoring, "run_single_evaluation", lambda: {"accuracy": 0.8, "f1": 0.75})

    monitoring.monitoring_loop()
