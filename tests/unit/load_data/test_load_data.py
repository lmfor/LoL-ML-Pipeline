import pandas as pd
import pytest

from lol_ml_pipeline.data_ingest.loaders.load_data import (
    FINAL_COLS,
    load_data,
    merge_dataframes,
)


def make_df(cols, n=2):
    data = {c: list(range(n)) for c in cols}
    return pd.DataFrame(data)


def test_load_data_returns_only_final_cols_in_order(monkeypatch):
    fake_df = make_df(FINAL_COLS, n=3)

    called = {}

    def fake_read_csv(file_path, delimiter=","):
        called["file_path"] = file_path
        called["delimiter"] = delimiter
        return fake_df

    monkeypatch.setattr(pd, "read_csv", fake_read_csv)

    out = load_data("dummy.csv", delimiter=";")

    assert called["file_path"] == "dummy.csv"
    assert called["delimiter"] == ";"
    assert list(out.columns) == FINAL_COLS
    assert len(out) == 3


def test_load_data_raises_keyerror_if_missing_required_col(monkeypatch):
    # Drop one required column to simulate schema mismatch
    cols = [c for c in FINAL_COLS if c != "result"]
    fake_df = make_df(cols, n=2)

    def fake_read_csv(file_path, delimiter=","):
        return fake_df

    monkeypatch.setattr(pd, "read_csv", fake_read_csv)

    # Your current implementation does df = df[FINAL_COLS] first,
    # which raises KeyError if any are missing (so the warning never prints).
    with pytest.raises(KeyError):
        load_data("dummy.csv")


def test_merge_dataframes_concatenates_and_resets_index():
    df1 = pd.DataFrame({"gameid": [1, 2], "result": [0, 1]})
    df2 = pd.DataFrame({"gameid": [3], "result": [1]})

    out = merge_dataframes([df1, df2])

    assert len(out) == 3
    assert list(out["gameid"]) == [1, 2, 3]
    assert list(out["result"]) == [0, 1, 1]
    assert list(out.index) == [0, 1, 2]


def test_merge_dataframes_empty_list_raises():
    # pd.concat([]) raises ValueError, so this is expected behavior.
    with pytest.raises(ValueError):
        merge_dataframes([])
