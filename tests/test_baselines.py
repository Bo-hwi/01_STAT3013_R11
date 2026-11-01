"""Tests for the M1 baselines (src/forecasting/baselines.py).

The first three tests use a tiny hand-made table, so the expected answer can be
checked by eye. The last tests use the real prediction file when it exists.

Run from the repository root::

    python -m pytest tests/test_baselines.py -v
"""
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.forecasting.baselines import (
    OUTPUT_COLUMNS, PREDICTIONS_PATH, choose_window, moving_average, seasonal_naive,
)


def tiny_table() -> pd.DataFrame:
    # 3 rows of one series; numbers chosen so each rule gives an obvious answer
    return pd.DataFrame({
        "sales":             [10.0, 20.0, 30.0],
        "sales_lag7":        [8.0, np.nan, np.nan],   # row 2 and 3 have no t-7 value
        "sales_lag14":       [7.0, 15.0, np.nan],     # row 3 has no t-14 value either
        "sales_roll7_mean":  [9.0, 18.0, 27.0],
        "sales_roll28_mean": [5.0, 5.0, 5.0],
        "score_mask":        [True, True, True],
    })


def test_seasonal_naive_uses_last_week_then_falls_back():
    forecast, n_fallback = seasonal_naive(tiny_table())
    # row 1: t-7 = 8; row 2: no t-7 -> t-14 = 15; row 3: no t-7, no t-14 -> 7-day mean = 27
    assert forecast.tolist() == [8.0, 15.0, 27.0]
    assert n_fallback == 2


def test_moving_average_reads_the_window_column():
    rows = tiny_table()
    assert moving_average(rows, 7).tolist() == [9.0, 18.0, 27.0]
    assert moving_average(rows, 28).tolist() == [5.0, 5.0, 5.0]


def test_window_is_chosen_by_lowest_validation_mae():
    # k=7 errors: |10-9|, |20-18|, |30-27| -> MAE 2.0 ; k=28 errors: 5, 15, 25 -> MAE 15.0
    best_k, tried = choose_window(tiny_table())
    assert best_k == 7
    assert {r["k"]: r["validation_MAE"] for r in tried} == {7: 2.0, 28: 15.0}


def test_window_choice_ignores_unscored_rows():
    rows = tiny_table()
    rows["score_mask"] = [False, False, True]  # only the last row counts
    _, tried = choose_window(rows)
    assert {r["k"]: r["validation_MAE"] for r in tried} == {7: 3.0, 28: 25.0}


# ---------------------------------------------------------------- real output
needs_predictions = pytest.mark.skipif(
    not Path(PREDICTIONS_PATH).exists(), reason="Run `python -m src.forecasting.baselines` first"
)


@needs_predictions
def test_prediction_file_follows_shared_format():
    pred = pd.read_csv(PREDICTIONS_PATH, parse_dates=["date"])
    assert list(pred.columns) == OUTPUT_COLUMNS
    assert set(pred["model"]) == {"SeasonalNaive", "MovingAverage"}
    assert set(pred["config"]) == {"M0"}
    assert not pred.duplicated(["date", "store_nbr", "family", "fold", "model"]).any()
    assert pred["y_pred"].notna().all()


@needs_predictions
def test_seasonal_naive_equals_sales_seven_days_earlier():
    # Independent check against the curated table (not the feature table)
    pred = pd.read_csv(PREDICTIONS_PATH, parse_dates=["date"])
    sn = pred[pred["model"] == "SeasonalNaive"].sample(300, random_state=0)
    curated = pd.read_parquet("data/processed/curated_sales_weather.parquet",
                              columns=["date", "store_nbr", "family", "sales"])
    curated["family"] = curated["family"].astype(str)
    week_before = sn.assign(date=sn["date"] - np.timedelta64(7, "D"))[["date", "store_nbr", "family"]]
    truth = week_before.merge(curated, on=["date", "store_nbr", "family"], how="left")
    assert np.allclose(sn["y_pred"].to_numpy(), truth["sales"].to_numpy())


@needs_predictions
def test_test_rows_are_inside_the_frozen_test_blocks():
    pred = pd.read_csv(PREDICTIONS_PATH, parse_dates=["date"])
    blocks = {"F1": ("2016-09-01", "2016-09-28"), "F2": ("2016-11-15", "2016-12-12"),
              "F3": ("2017-03-01", "2017-03-28"), "F4": ("2017-07-19", "2017-08-15")}
    for fold, (start, end) in blocks.items():
        dates = pred.loc[pred["fold"] == fold, "date"]
        assert dates.min() >= pd.Timestamp(start) and dates.max() <= pd.Timestamp(end)
