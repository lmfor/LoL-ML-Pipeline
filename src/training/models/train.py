import os
import pandas as pd
import numpy as np
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

        self.X = np.array(game_features).astype(str)
        self.y = np.array(game_labels)
        return self.X, self.y

    def build_model(self):
        # 33 strings
        # Format: [Champs(10), Players(10), Teams(2), Bans(10), League(1)]
        inputs = tf.keras.layers.Input(shape=(33,), dtype=tf.string)

        # slice input vector based on schema above in prepare_data
        champs_input = inputs[:, 0:10]    # 5 Blue + 5 Red picks
        players_input = inputs[:, 10:20]  # 5 Blue + 5 Red players
        teams_input = inputs[:, 20:22]    # Blue + Red team names
        bans_input = inputs[:, 22:32]     # 5 Blue + 5 Red bans
        league_input = inputs[:, 32:33]   # The league string

        # CHAMPIONS (Picks & Bans share the same vocabulary)
        lookup_champs = tf.keras.layers.StringLookup(vocabulary=self.vocab_champs)(champs_input)
        lookup_bans = tf.keras.layers.StringLookup(vocabulary=self.vocab_champs)(bans_input)
        
        champ_embed_layer = tf.keras.layers.Embedding(len(self.vocab_champs) + 1, 16)
        embed_picks = champ_embed_layer(lookup_champs)
        embed_bans = champ_embed_layer(lookup_bans)

        # PLAYERS
        lookup_players = tf.keras.layers.StringLookup(vocabulary=self.vocab_players)(players_input)
        embed_players = tf.keras.layers.Embedding(len(self.vocab_players) + 1, 12)(lookup_players)

        # TEAMS
        lookup_teams = tf.keras.layers.StringLookup(vocabulary=self.vocab_teams)(teams_input)
        embed_teams = tf.keras.layers.Embedding(len(self.vocab_teams) + 1, 8)(lookup_teams)

        # LEAGUE
        lookup_league = tf.keras.layers.StringLookup(vocabulary=self.vocab_leagues)(league_input)
        embed_league = tf.keras.layers.Embedding(len(self.vocab_leagues) + 1, 4)(lookup_league)

        # dense layers
        merged = tf.keras.layers.Concatenate()([
            tf.keras.layers.Flatten()(embed_picks),
            tf.keras.layers.Flatten()(embed_bans),
            tf.keras.layers.Flatten()(embed_players),
            tf.keras.layers.Flatten()(embed_teams),
            tf.keras.layers.Flatten()(embed_league)
        ])

        x = tf.keras.layers.Dense(128, activation='relu')(merged)
        x = tf.keras.layers.Dropout(0.3)(x)
        x = tf.keras.layers.Dense(64, activation='relu')(x)
        output = tf.keras.layers.Dense(1, activation='sigmoid')(x)

        # conglom.
        self.model = tf.keras.Model(inputs=inputs, outputs=output)
        self.model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        return self.model
    
    def fit(self, epochs=20, batch_size=32, validation_split=0.2):
            if self.model is None or self.X is None:
                raise ValueError("Model or Data not ready.")
            split_idx = int(len(self.X) * (1 - validation_split))
            
            # split data
            train_X, val_X = self.X[:split_idx], self.X[split_idx:]
            train_y, val_y = self.y[:split_idx], self.y[split_idx:]

            # explicit tell tf they are STRINGS
            train_ds = tf.data.Dataset.from_tensor_slices((
                tf.cast(train_X, tf.string), 
                train_y
            )).batch(batch_size)

            val_ds = tf.data.Dataset.from_tensor_slices((
                tf.cast(val_X, tf.string), 
                val_y
            )).batch(batch_size)

            print(f"Starting training for {epochs} epochs...")

            self.history = self.model.fit(
                train_ds,
                validation_data=val_ds,
                epochs=epochs,
                verbose=1
            )

            return self.history

if __name__ == "__main__":
    # cvs
    loader = DataLoader()
    raw_df = loader.load_data(paths=match_files)

    # trainer init and run
    trainer = Trainer(raw_df)
    trainer.prepare_data()  
    trainer.build_model()   
    
    history = trainer.fit()

    # final accuracy
    final_acc = history.history['val_accuracy'][-1]
    print(f"Final Validation Accuracy: {final_acc:.2%}")