from unittest import mock
import data_preparation as dp


def test_prepare_tweet_eval(monkeypatch, tmp_path):
    fake_ds = mock.MagicMock()
    fake_split = mock.MagicMock()
    fake_split.select.return_value = fake_split
    fake_split.map.return_value = fake_split

    fake_ds.keys.return_value = ["train", "test"]
    fake_ds.__getitem__.return_value = fake_split

    monkeypatch.setattr(dp, "safe_load_dataset", lambda name, config=None: fake_ds)

    fake_tokenizer = mock.MagicMock()
    fake_tokenizer.return_value = {"input_ids": [0], "attention_mask": [1]}

    output_path = tmp_path / "tweet_eval_tokenized"
    dp.prepare_tweet_eval(fake_tokenizer, str(output_path))

    fake_split.save_to_disk.assert_called_once()


def test_prepare_youtube(monkeypatch, tmp_path):
    fake_ds = mock.MagicMock()
    fake_split = mock.MagicMock()
    fake_split.select.return_value = fake_split
    fake_split.map.return_value = fake_split

    fake_ds.keys.return_value = ["train"]
    fake_ds.__getitem__.return_value = fake_split

    monkeypatch.setattr(dp, "safe_load_dataset", lambda name, config=None: fake_ds)

    fake_tokenizer = mock.MagicMock()
    fake_tokenizer.return_value = {"input_ids": [0], "attention_mask": [1]}

    output_path = tmp_path / "youtube_tokenized"
    dp.prepare_youtube(fake_tokenizer, str(output_path))

    fake_split.save_to_disk.assert_called_once()
