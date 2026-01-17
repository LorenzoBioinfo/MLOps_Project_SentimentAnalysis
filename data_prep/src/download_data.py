from datasets import load_dataset
import sys
output_path = sys.argv[1] 

dataset = load_dataset("tweet_eval", "sentiment")
dataset.save_to_disk(f"{output_path}/tweet_eval_sentiment")


dataset2 = load_dataset("AmaanP314/youtube-comment-sentiment")
dataset2.save_to_disk(f"{output_path}/youtube-comment-sentiment")