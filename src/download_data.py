from datasets import load_dataset

dataset = load_dataset("tweet_eval", "sentiment")
dataset.save_to_disk("/workspaces/MLOps_Project_SentimentAnalysis/data/raw/tweet_eval_sentiment")


dataset2 = load_dataset("AmaanP314/youtube-comment-sentiment")
dataset2.save_to_disk("/workspaces/MLOps_Project_SentimentAnalysis/data/raw/youtube-comment-sentiment")