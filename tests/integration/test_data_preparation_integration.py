import data_prep.src.data_preparation as dp
from unittest import mock


def _fake_dataset():
    """Crea un dataset fake compatibile con HuggingFace DatasetDict"""
    fake_split = mock.MagicMock()
    fake_split.select.return_value = fake_split
    fake_split.map.return_value = fake_split

    fake_ds = mock.MagicMock()
    fake_ds.keys.return_value = ["train", "test"]
    fake_ds.__getitem__.return_value = fake_split

    return fake_ds, fake_split


def test_prepare_tweet_eval_integration(monkeypatch, tmp_path):
    fake_ds, fake_split = _fake_dataset()

    # mock load dataset
    monkeypatch.setattr(dp, "safe_load_dataset", lambda *args, **kwargs: fake_ds)

    # mock tokenizer
    monkeypatch.setattr(dp, "tokenize_function", lambda x: x)

    output_path = tmp_path / "tweet_eval_tokenized"

    dp.prepare_tweet_eval(dp.tokenizer, str(output_path))

    # ASSERT
    fake_ds.save_to_disk.assert_called_once_with(str(output_path))
    assert fake_split.select.called
    assert fake_split.map.called


def test_prepare_youtube_integration(monkeypatch, tmp_path):
    fake_ds, fake_split = _fake_dataset()

    monkeypatch.setattr(dp, "safe_load_dataset", lambda *args, **kwargs: fake_ds)
    monkeypatch.setattr(dp, "tokenize_function", lambda x: x)

    output_path = tmp_path / "youtube_tokenized"

    dp.prepare_youtube(dp.tokenizer, str(output_path))

    # ASSERT
    fake_ds.save_to_disk.assert_called_once_with(str(output_path))
    assert fake_split.select.called
    assert fake_split.map.called
