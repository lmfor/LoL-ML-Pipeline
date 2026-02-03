import pandas as pd
import numpy as np
from dataclasses import dataclass

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

@dataclass(frozen=True)
class DataLoader:
    def _load_data_helper(self, file_path : str, delimiter = ','):
        df = pd.read_csv(file_path, delimiter=delimiter)
        
        missing = [c for c in FINAL_COLS if c not in df.columns]
        if missing:
            print(f"Warning: Missing columns: {missing}")
        
        df = df[FINAL_COLS]
        return df

    def load_data(self, paths: list[str]) -> pd.DataFrame:
        dfs : list[pd.DataFrame] = [self._load_data_helper(path) for path in paths]
        merged_df = pd.concat(dfs, axis=0, ignore_index=True)
        return merged_df
