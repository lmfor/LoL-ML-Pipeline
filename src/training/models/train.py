import os
import pandas as pd
import tensorflow as tf 

from data_ingest.loaders.load_data import DataLoader


root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
sources_path = os.path.join(root, "data_ingest", "sources")

match_files = [
    os.path.join(sources_path, "2025_match_data.csv"),
    os.path.join(sources_path, "2026_match_data.csv")
]


class Trainer:
    def __init__(self, raw_data : pd.DataFrame):
        self.raw_data = raw_data


        self.train_data = None
        self.eval_data = None
        self.test_data = None

    def _split_training_data(self):
        n = int(self.raw_data.shape[0])
        
        # Split into train, eval, and test
        train = self.raw_data.iloc[: int(0.8 * n)]
        eval_ = self.raw_data.iloc[int(0.8 * n): int(0.9 * n)]
        test = self.raw_data.iloc[int(0.9 * n):]

        self.train_data = train
        self.eval_data = eval_
        self.test_data = test

    def train_epoch(self):
        pass

    def train_step(self):
        pass
