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

def prepare_data(self):
        self.vocab = sorted(list(set(
            [c for c in self.raw_data['champion'] if isinstance(c, str)]
        )))

        game_features = []
        game_labels = []

        # 2. Group into (N, 10) array of champion name strings
        for game_id, group in self.raw_data.groupby('gameid'):
            blue = group[group['side'].str.lower() == 'blue']
            red = group[group['side'].str.lower() == 'red']
            
            if len(blue) >= 5 and len(red) >= 5:
                # Store as raw strings
                b_champs = blue['champion'].iloc[:5].values
                r_champs = red['champion'].iloc[:5].values
                
                game_features.append(np.concatenate([b_champs, r_champs]))
                game_labels.append(int(blue['result'].iloc[0]))

        self.X = np.array(game_features) 
        self.y = np.array(game_labels)
        return self.vocab

    def build_model(self):
        pass
    
    def train(self):
        pass