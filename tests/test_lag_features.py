"""Tests for the feature table (data/processed/model_features.parquet).

Three groups of checks:
1. Leakage: every feature uses only the past (calendar-date logic, day t excluded).
2. Contract with M3: values equal M3's own reference functions in
   src/forecasting/rolling_validation.py, so M3 can switch to ``--source features``.
3. Scoring mask: per-fold exclusion counts equal the numbers agreed with M3.

Run after the pipeline from the repository root::

    python -m pytest tests/test_lag_features.py -v
"""
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.data.clean_data import load_weather
from src.forecasting.feature_spec import FEATURES
from src.forecasting.rolling_validation import (
    add_scoring_mask, calendar_rolling_mean, calendar_sales_lag, iter_folds, read_yaml,
)
from src.features.lag_features import FORBIDDEN

REPO_ROOT = Path(__file__).resolve().parents[1]
FEATURES_PATH = REPO_ROOT / "data" / "processed" / "model_features.parquet"
KEY = ["date", "store_nbr", "family"]

pytestmark = pytest.mark.skipif(not FEATURES_PATH.exists(), reason="Run `python scripts/run_pipeline.py` first")


@pytest.fixture(scope="module")
def df() -> pd.DataFrame:
    return pd.read_parquet(FEATURES_PATH)


@pytest.fixture(scope="module")
def sample(df) -> pd.DataFrame:
    # 40 store-family series, enough to cover late openings and long closures
    series = df[["store_nbr", "family"]].drop_duplicates()
    picked = pd.concat([
        series[series["store_nbr"].isin([18, 25, 52])].groupby("store_nbr").head(5),
        series.sample(25, random_state=42),
    ]).drop_duplicates()
    return df.merge(picked, on=["store_nbr", "family"]).sort_values(KEY, ignore_index=True)


# ---------------------------------------------------------------- contract
def test_columns_match_m3_contract(df):
    expected = [
        *FEATURES["sales_lags"].values(), *FEATURES["sales_rolling"].values(),
        *FEATURES["calendar"], *FEATURES["categorical"], *FEATURES["weather"],
        *FEATURES["conditional"], FEATURES["score_mask"],
    ]
    missing = set(expected) - set(df.columns)
    assert not missing, f"Missing contract columns: {sorted(missing)}"


def test_forbidden_columns_absent(df):
    assert not set(FORBIDDEN) & set(df.columns)


def test_key_unique_and_rowcount(df):
    assert len(df) == 3_000_888
    assert not df.duplicated(KEY).any()


# ---------------------------------------------------------------- leakage
def test_sales_lags_equal_m3_calendar_lookup(df):
    for k, col in FEATURES["sales_lags"].items():
        assert np.allclose(df[col].to_numpy(float), calendar_sales_lag(df, k), equal_nan=True), col


def test_lag_across_christmas_is_nan_not_zero(df):
    # 2016-12-26: lag1 points at 2016-12-25, which has no rows -> NaN
    assert df.loc[df["date"] == "2016-12-26", "sales_lag1"].isna().all()
    # 2017-01-01: lag7 points at 2016-12-25 -> NaN
    assert df.loc[df["date"] == "2017-01-01", "sales_lag7"].isna().all()


def test_rolling_equals_m3_reference(sample):
    for k, col in FEATURES["sales_rolling"].items():
        assert np.allclose(sample[col].to_numpy(float), calendar_rolling_mean(sample, k).to_numpy(), equal_nan=True), col


def test_rolling_excludes_day_t(sample):
    # Brute force on one series around Christmas 2016: roll7 at t = mean of sales on t-7..t-1
    first_family = sample.loc[sample["store_nbr"] == 18, "family"].iloc[0]
    series = sample[(sample["store_nbr"] == 18) & (sample["family"] == first_family)].set_index("date")
    for t in pd.date_range("2016-12-20", "2017-01-05"):
        if t not in series.index:
            continue  # 25 December has no row
        window = series.loc[(series.index >= t - np.timedelta64(7, "D")) & (series.index < t), "sales"]
        assert np.isclose(series.loc[t, "sales_roll7_mean"], window.mean()), t


def test_weather_lag1_is_previous_day(df):
    wx = load_weather().set_index(["city", "date"])["precipitation_sum"]
    probe = df.drop_duplicates(["city", "date"]).sample(500, random_state=1)
    prev = pd.MultiIndex.from_arrays([probe["city"].astype(str), probe["date"] - np.timedelta64(1, "D")])
    assert np.allclose(probe["precipitation_sum_lag1"].to_numpy(float), wx.reindex(prev).to_numpy(), equal_nan=True)
    # 26 December gets the weather of 25 December even though sales has no rows that day
    assert df.loc[df["date"] == "2016-12-26", "precipitation_sum_lag1"].notna().all()


def test_only_first_day_lacks_weather_lag(df):
    missing = df.loc[df["precipitation_sum_lag1"].isna(), "date"].unique()
    assert list(missing) == [pd.Timestamp("2013-01-01")]


# ---------------------------------------------------------------- score mask
def test_score_mask_agrees_with_m3_rules(df):
    rules = read_yaml(REPO_ROOT / "configs" / "ablation_M0_M5.yaml")["scoring_exclusions"]
    # add_scoring_mask raises ValueError if M1's score_mask disagrees on any active row
    add_scoring_mask(df[KEY + ["sales", "city", "store_active", "store_zero_day", "score_mask"]], rules)


def test_m3_fold_iterator_accepts_feature_table(df):
    # Integration test: M3's iter_folds must run on this table without column clashes
    protocol = read_yaml(REPO_ROOT / "configs" / "research_protocol.yaml")
    ablation = read_yaml(REPO_ROOT / "configs" / "ablation_M0_M5.yaml")
    families = ["BEVERAGES", "DAIRY"]  # small subset keeps the test fast
    folds = list(iter_folds(df, protocol, ablation, families))
    assert [f.fold_id for f in folds] == ["F1", "F2", "F3", "F4"]
    assert all(f.test["score_mask"].any() for f in folds)


@pytest.mark.parametrize("fold,split,start,end,closure,post_only", [
    ("F1", "validation", "2016-08-04", "2016-08-31", 891, 0),
    ("F1", "test", "2016-09-01", "2016-09-28", 1848, 0),
    ("F2", "validation", "2016-10-18", "2016-11-14", 1254, 627),
    ("F2", "test", "2016-11-15", "2016-12-12", 693, 627),
    ("F3", "test", "2017-03-01", "2017-03-28", 0, 0),
    ("F4", "test", "2017-07-19", "2017-08-15", 0, 0),
])
def test_exclusion_counts_per_fold(df, fold, split, start, end, closure, post_only):
    block = df[df["date"].between(start, end) & (df["store_active"] == 1)]
    assert int((block["store_zero_day"] == 1).sum()) == closure
    # Open store-days that are still masked = the 28-day post-reopening exclusion
    assert int((~block["score_mask"] & (block["store_zero_day"] == 0)).sum()) == post_only
    assert int((~block["score_mask"]).sum()) == closure + post_only
