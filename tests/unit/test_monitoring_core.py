import os
import json
from monitoring import run_single_evaluation


def test_run_single_evaluation(monkeypatch, tmp_path):

    class FakeModel:
        def eval(self): pass
        def __call__(self, **inputs):
            import torch
            return type("Out", (), {"logits": torch.tensor([[0.1, 0.9, 0.0]])})

    def fake_model_loader():
        return FakeModel()

    # Fake dataset
    class FakeDataset:
        def __init__(self):
            self.data = {"input_ids": [[1,2,3]], "attention_mask": [[1,1,1]], "label": [1]}
        def get(self, name):
            return None
        def __getitem__(self, name):
            return self
        def train_test_split(self, test_size):
            return {"test": self}
        def select(self, _range):
            return self
        def __len__(self):
            return 1

    def fake_loader():
        return FakeDataset()

    yt_metrics = run_single_evaluation(
        model_loader=fake_model_loader,
        tweet_loader=fake_loader,
        yt_loader=fake_loader,
        report_base=str(tmp_path / "reports")
    )

    assert yt_metrics["accuracy"] == 1.0
