"""Tests for the M1 SARIMAX model (src/forecasting/sarimax.py).

1. Series building: aggregation to city x family is correct and leak-free.
2. Forecasting: one-step-ahead predictions never use the target day itself.
3. Output: same format, same scored days and same y_true as the baselines.

Run from the repository root::

    python -m pytest tests/test_sarimax.py -v
"""
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.forecasting import sarimax as S

FEATURES = Path("data/processed/model_features.parquet")
needs_features = pytest.mark.skipif(not FEATURES.exists(), reason="Run the pipeline first")
needs_output = pytest.mark.skipif(not S.PREDICTIONS_PATH.exists(), reason="Run `python -m src.forecasting.sarimax` first")


@pytest.fixture(scope="module")
def series() -> pd.DataFrame:
    return S.build_city_series(pd.read_parquet(FEATURES))


# ---------------------------------------------------------------- 1. series
@needs_features
def test_one_complete_daily_series_per_city_family(series):
    assert series.groupby(["city", "family"], observed=True).ngroups == 22 * len(S.FAMILIES)
    days = pd.date_range(S.SERIES_START, S.SERIES_END).size
    assert series.groupby(["city", "family"], observed=True).size().eq(days).all()


@needs_features
def test_city_total_equals_sum_of_open_stores(series):
    rows = pd.read_parquet(FEATURES, columns=["date", "city", "family", "sales", "store_active", "store_zero_day"])
    probe = rows[(rows["city"] == "Guayaquil") & (rows["family"] == "BEVERAGES") & (rows["date"] == "2016-06-15")]
    expected = probe.loc[(probe["store_active"] == 1) & (probe["store_zero_day"] == 0), "sales"].sum()
    got = series.query("city == 'Guayaquil' and family == 'BEVERAGES' and date == '2016-06-15'")["sales"].iloc[0]
    assert np.isclose(got, expected)


@needs_features
def test_closed_days_are_unknown_not_zero(series):
    # 25 December has no rows -> y must be NaN (the Kalman filter skips it), never log1p(0)
    assert series.loc[series["date"] == "2016-12-25", "y"].isna().all()


@needs_features
def test_exogenous_columns_have_no_gaps(series):
    for config, columns in S.EXOG.items():
        assert series[columns].notna().all().all(), config


@needs_features
def test_weather_is_previous_day(series):
    # Column built by M1 as weather of t-1; SARIMAX must not use weather of day t
    assert "precipitation_sum" not in series.columns
    assert all(c.endswith("_lag1") for c in S.EXOG["M3"] if "precipitation" in c or "temperature" in c)


# ---------------------------------------------------------------- 2. forecast
@needs_features
def test_forecast_for_day_t_ignores_sales_of_day_t(series):
    # Change the true sales of one test day: its own forecast must not move,
    # only forecasts of later days may change (they see it as history).
    s = series.query("city == 'Cuenca' and family == 'GROCERY I'").reset_index(drop=True)
    fold = {"fold_id": "F4", "train": ["2013-01-01", "2017-06-20"],
            "validation": ["2017-06-21", "2017-07-18"], "test": ["2017-07-19", "2017-08-15"]}
    order, seasonal = S.ORDER_CANDIDATES[0]
    base, _ = S.forecast_fold(s, fold, order, seasonal, "M0")  # (predictions, status)

    target_day = pd.Timestamp("2017-08-01")
    changed = s.copy()
    changed.loc[changed["date"] == target_day, "y"] += 1.0  # a big shock on that day
    shocked, _ = S.forecast_fold(changed, fold, order, seasonal, "M0")

    before = base.set_index("date")["y_pred"]
    after = shocked.set_index("date")["y_pred"]
    assert np.isclose(before[target_day], after[target_day])          # same day: unchanged
    assert not np.isclose(before[target_day + pd.Timedelta(days=1)],
                          after[target_day + pd.Timedelta(days=1)])    # next day: uses it


# ---------------------------------------------------------------- 3. output
@needs_output
def test_prediction_file_format():
    pred = pd.read_csv(S.PREDICTIONS_PATH, parse_dates=["date"])
    assert list(pred.columns) == S.OUTPUT_COLUMNS
    assert set(pred["config"]) == {"M0", "M3"}
    assert set(pred["family"]) == set(S.FAMILIES)
    assert pred["y_pred"].notna().all() and (pred["y_pred"] >= 0).all()
    assert not pred.duplicated(["date", "city", "family", "fold", "config"]).any()


@needs_output
def test_same_days_and_same_truth_as_city_level_baselines():
    pred = pd.read_csv(S.PREDICTIONS_PATH, parse_dates=["date"])
    base = S.baselines_at_city_level(pred)
    key = ["date", "city", "family", "fold"]
    sar = pred[pred["config"] == "M0"].set_index(key).sort_index()
    sn = base[base["model"] == "SeasonalNaive"].set_index(key).sort_index()
    assert sar.index.equals(sn.index)                       # identical scored days
    assert np.allclose(sar["y_true"], sn["y_true"])          # identical ground truth
