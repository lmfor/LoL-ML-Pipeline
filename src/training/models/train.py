import os
import pandas as pd
import numpy as np
import tensorflow as tf 
from tensorflow.keras import regularizers

# Cloud Dev
import boto3

from data_ingest.loaders.load_data import DataLoader


root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
sources_path = os.path.join(root, "data_ingest", "sources")

match_files = [
    os.path.join(sources_path, "2023_match_data.csv"),
    os.path.join(sources_path, "2024_match_data.csv"),
    os.path.join(sources_path, "2025_match_data.csv"),
    os.path.join(sources_path, "2026_match_data.csv")
]


class Trainer:
    def __init__(self, raw_data : pd.DataFrame):
        self.raw_data = raw_data.dropna(subset=['gameid', 'side', 'result'])
        
        self.X = None
        self.y = None
        
        self.vocab_champs = []
        # self.vocab_players = [] # RE-ADD LATER IF NEEDED
        self.vocab_teams = []
        self.vocab_leagues = []
        
        # 4. Model placeholder
        self.model = None

        # 5. Tracking metrics
        self.history = None
    
    def download_data_from_s3(self, bucket_name, files):
        s3 = boto3.client('s3')
        for file_name in files:
            local_path = f"/app/data/{file_name}"
            if not os.path.exists(local_path):
                s3.download_file(bucket_name, f"raw_data/{file_name}", local_path)
                print(f"Downloaded {file_name} from S3 bucket {bucket_name} to {local_path}")


    def prepare_data(self):
        self.vocab_champs = sorted(list(set(self.raw_data['champion'].dropna().astype(str))))
        # self.vocab_players = sorted(list(set(self.raw_data['playername'].dropna().astype(str)))) # RE-ADD LATER
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
                # players = np.concatenate([blue['playername'].iloc[:5].values, red['playername'].iloc[:5].values]) # REMOVED TO PREVENT OVERFITTING
                # Teams (2) 
                teams = np.array([blue['teamname'].iloc[0], red['teamname'].iloc[0]])
                # Bans (10)
                bans = np.concatenate([
                    blue[['ban1', 'ban2', 'ban3', 'ban4', 'ban5']].iloc[0].values,
                    red[['ban1', 'ban2', 'ban3', 'ban4', 'ban5']].iloc[0].values
                ])
                # League (1)
                league = np.array([group['league'].iloc[0]])

                # Concatenate everything - NOW 23 STRINGS
                full_row = np.concatenate([champs, teams, bans, league])
                
                game_features.append(full_row)
                game_labels.append(int(blue['result'].iloc[0]))

        """
        Format: X[game_number_by_int][features]

        Feature Locations in X:

        blue_team_champ_picks = X[game][0:5]
        red_team_champ_picks = X[game][5:10]
        # blue_team_players = X[game][10:15] # REMOVED
        # red_team_players = X[game][15:20]  # REMOVED
        blue_team_name = X[game][10]         # SHIFTED
        red_team_name = X[game][11]          # SHIFTED
        blue_team_bans = X[game][12:17]      # SHIFTED
        red_team_bans = X[game][17:22]       # SHIFTED
        league = X[game][22]                 # SHIFTED
        """

        # random order of indices
        indices = np.random.permutation(len(game_features))

        # Reorder X and y using indices
        self.X = np.array(game_features)[indices].astype(str)
        self.y = np.array(game_labels)[indices]

        return self.X, self.y

    def build_model(self):
        # 23 strings
        # Format: [Champs(10), Teams(2), Bans(10), League(1)]
        inputs = tf.keras.layers.Input(shape=(23,), dtype=tf.string)

        # slice input vector based on schema above in prepare_data
        champs_input = inputs[:, 0:10]    # 5 Blue + 5 Red picks
        # players_input = inputs[:, 10:20] # REMOVED
        teams_input = inputs[:, 10:12]    # Blue + Red team names
        bans_input = inputs[:, 12:22]     # 5 Blue + 5 Red bans
        league_input = inputs[:, 22:23]   # The league string

        # CHAMPIONS (Picks & Bans share the same vocabulary)
        lookup_champs = tf.keras.layers.StringLookup(vocabulary=self.vocab_champs)(champs_input)
        lookup_bans = tf.keras.layers.StringLookup(vocabulary=self.vocab_champs)(bans_input)
        
        champ_embed_layer = tf.keras.layers.Embedding(len(self.vocab_champs) + 1, 16)
        embed_picks = champ_embed_layer(lookup_champs)
        embed_bans = champ_embed_layer(lookup_bans)

        # PLAYERS - REMOVED TO PREVENT OVERFITTING
        # lookup_players = tf.keras.layers.StringLookup(vocabulary=self.vocab_players)(players_input)
        # embed_players = tf.keras.layers.Embedding(len(self.vocab_players) + 1, 12)(lookup_players)

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
            # tf.keras.layers.Flatten()(embed_players), # REMOVED
            tf.keras.layers.Flatten()(embed_teams),
            tf.keras.layers.Flatten()(embed_league)
        ])

        x = tf.keras.layers.Dense(128, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.01))(merged)
        x = tf.keras.layers.Dropout(0.5)(x)
        x = tf.keras.layers.Dense(64, activation='relu')(x)
        x = tf.keras.layers.Dropout(0.3)(x)
        output = tf.keras.layers.Dense(1, activation='sigmoid')(x)

        # conglom.
        self.model = tf.keras.Model(inputs=inputs, outputs=output)
        optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001) # REDUCED LEARNING RATE
        self.model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])
        return self.model
    
    def fit(self, epochs=20, batch_size=32, validation_split=0.2):
            if self.model is None or self.X is None:
                raise ValueError("Model or Data not ready.")
            split_idx = int(len(self.X) * (1 - validation_split))
            
            # split data
            train_X, val_X = self.X[:split_idx], self.X[split_idx:]
            train_y, val_y = self.y[:split_idx], self.y[split_idx:]

            early_stop = tf.keras.callbacks.EarlyStopping(
                monitor='val_loss', 
                patience=5, 
                restore_best_weights=True
            )

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
                verbose=1,
                callbacks=[early_stop]
            )

            return self.history
    
    def upload_model_to_s3(self, model_path):
        bucket_name = os.getenv('S3_BUCKET_NAME')
        s3 = boto3.client('s3')
        s3.upload_file(model_path, bucket_name, f"models/lol_model_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.h5")
        print(f"Uploaded model to S3 bucket {bucket_name}.")

if __name__ == "__main__":
    # cvs
    loader = DataLoader()
    raw_df = loader.load_data(paths=match_files)

    # trainer init and run
    trainer = Trainer(raw_df)
    trainer.prepare_data()  
    trainer.build_model()   
    
    history = trainer.fit(epochs=100, batch_size=16, validation_split=0.2) # MORE EPOCHS, SMALLER BATCH

    # final accuracy
    final_acc = history.history['val_accuracy'][-1]
    print(f"Final Validation Accuracy: {final_acc:.2%}")