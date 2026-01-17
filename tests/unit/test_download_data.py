import sys
import importlib
from unittest import mock


def test_download_data_calls_load_and_save(tmp_path, monkeypatch):
    
    monkeypatch.setattr(sys, "argv", ["download_data.py", str(tmp_path)])

    
    fake_dataset_1 = mock.MagicMock()
    fake_dataset_2 = mock.MagicMock()

    
    with mock.patch("download_data.load_dataset", side_effect=[fake_dataset_1, fake_dataset_2]) as mock_load:
        
        import download_data
        importlib.reload(download_data)

        
        mock_load.assert_any_call("tweet_eval", "sentiment")
        mock_load.assert_any_call("AmaanP314/youtube-comment-sentiment")

      
        fake_dataset_1.save_to_disk.assert_called_once_with(
            f"{tmp_path}/tweet_eval_sentiment"
        )
        fake_dataset_2.save_to_disk.assert_called_once_with(
            f"{tmp_path}/youtube-comment-sentiment"
        )
