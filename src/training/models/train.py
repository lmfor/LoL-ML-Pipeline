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
        self.raw_data = raw_data.dropna(subset=['gameid', 'side', 'result'])
        
        self.X = None
        self.y = None
        
        self.vocab_champs = []
        self.vocab_players = []
        self.vocab_teams = []
        self.vocab_leagues = []
        
        # 4. Model placeholder
        self.model = None

        # 5. Tracking metrics
        self.history = None

def prepare_data(self):
    self.vocab_champs = sorted(list(set(self.raw_data['champion'].dropna().astype(str))))
    self.vocab_players = sorted(list(set(self.raw_data['playername'].dropna().astype(str))))
    self.vocab_teams = sorted(list(set(self.raw_data['teamname'].dropna().astype(str))))
    self.vocab_leagues = sorted(list(set(self.raw_data['league'].dropna().astype(str))))

    game_features = []
    game_labels = []

    for game_id, group in self.raw_data.groupby('gameid'):
        blue = group[group['side'].str.lower() == 'blue']
        red = group[group['side'].str.lower() == 'red']
        
        if len(blue) >= 5 and len(red) >= 5:
            # Champions (10)
            champs = np.concatenate([blue['champion'].iloc[:5].values, red['champion'].iloc[:5].values])
            # Players (10)
            players = np.concatenate([blue['playername'].iloc[:5].values, red['playername'].iloc[:5].values])
            # Teams (2) 
            teams = np.array([blue['teamname'].iloc[0], red['teamname'].iloc[0]])
            # Bans (10)
            bans = np.concatenate([
                blue[['ban1', 'ban2', 'ban3', 'ban4', 'ban5']].iloc[0].values,
                red[['ban1', 'ban2', 'ban3', 'ban4', 'ban5']].iloc[0].values
            ])
            # League (1)
            league = np.array([group['league'].iloc[0]])

            # Concatenate everything 
            full_row = np.concatenate([champs, players, teams, bans, league])
            
            game_features.append(full_row)
            game_labels.append(int(blue['result'].iloc[0]))

    """
    Format: X[game_number_by_int][features]

    Feature Locations in X:

    blue_team_champ_picks = X[game][0:5]
    red_team_champ_picks = X[game][5:10]
    blue_team_players = X[game][10:15]
    red_team_players = X[game][15:20]
    blue_team_name = X[game][20]
    red_team_name = X[game][21]
    blue_team_bans = X[game][22:27]
    red_team_bans = X[game][27:32]
    league = X[game][32]
    """

    self.X = np.array(game_features)
    self.y = np.array(game_labels)
    return self.X, self.y

    def build_model(self):
        pass
    
    def train(self):
        pass