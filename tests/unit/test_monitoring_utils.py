import os
import json
from monitoring.src.monitoring import create_run_dir, save_json


def test_create_run_dir(tmp_path, monkeypatch):
    base_dir = tmp_path / "reports"
    monkeypatch.setenv("BASE_DIR", str(tmp_path))

    run_dir = create_run_dir(base_dir=str(base_dir))
    assert os.path.exists(run_dir)


def test_save_json(tmp_path):
    data = {"a": 1}
    path = tmp_path / "test.json"
    save_json(data, str(path))

    with open(path) as f:
        loaded = json.load(f)

    assert loaded == data
