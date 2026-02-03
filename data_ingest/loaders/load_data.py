import pandas as pd
import numpy as np

FINAL_COLS = [
    "gameid",
    "url",
    "league",
    "year",
    "split",
    "playoffs",
    "date",
    "game",
    "patch",
    "participantid",
    "side",
    "position",
    "playername",
    "playerid",
    "teamname",
    "teamid",
    "firstPick",
    "champion",
    "ban1",
    "ban2",
    "ban3",
    "ban4",
    "ban5",
    "pick1",
    "pick2",
    "pick3",
    "pick4",
    "pick5",
    "result",
]

def load_data(file_path : str, delimiter = ','):
    df = pd.read_csv(file_path, delimiter=delimiter)
    
    missing = [c for c in FINAL_COLS if c not in df.columns]
    if missing:
        print(f"Warning: Missing columns: {missing}")
    
    df = df[FINAL_COLS]
    return df

def merge_dataframes(dfs: list[pd.DataFrame]) -> pd.DataFrame:
    merged_df = pd.concat(dfs, axis=0, ignore_index=True)
    return merged_df

