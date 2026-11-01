"""Integrity tests for the curated table (data contract, Proposal Appendix A).

Run after the pipeline from the repository root::

    python -m pytest tests/test_data_integrity.py -v

Tests are skipped when the curated parquet has not been built yet.
"""
from pathlib import Path

import pandas as pd
import pytest

from src.data.load_favorita import load_train

REPO_ROOT = Path(__file__).resolve().parents[1]
CURATED = REPO_ROOT / "data" / "processed" / "curated_sales_weather.parquet"
KEY = ["date", "store_nbr", "family"]

pytestmark = pytest.mark.skipif(not CURATED.exists(), reason="Run `python -m src.data.merge_data` first")


@pytest.fixture(scope="module")
def df() -> pd.DataFrame:
    return pd.read_parquet(CURATED)


def test_row_count_matches_train(df):
    # Joins must not add or drop rows
    assert len(df) == len(load_train())


def test_key_is_unique(df):
    assert not df.duplicated(KEY).any()


def test_expected_coverage(df):
    assert df["store_nbr"].nunique() == 54
    assert df["family"].nunique() == 33
    assert df["city"].nunique() == 22
    assert df["date"].min() == pd.Timestamp("2013-01-01")
    assert df["date"].max() == pd.Timestamp("2017-08-15")


def test_no_christmas_rows_created(df):
    # QA-01: closed days are absent in train and must stay absent
    assert not df["date"].dt.strftime("%m-%d").eq("12-25").any()


def test_every_present_day_is_complete(df):
    # 54 stores x 33 families on every date
    assert df.groupby("date").size().eq(54 * 33).all()


def test_weather_joined_for_every_row(df):
    core = ["precipitation_sum", "temperature_2m_max", "temperature_2m_mean",
            "relative_humidity_2m_mean", "sunshine_hours"]
    assert df[core].notna().all().all()


def test_weather_is_constant_within_city_day(df):
    # Many-to-one join: every store in a city sees the same weather on a given day
    nunique = df.groupby(["city", "date"], observed=True)["precipitation_sum"].nunique()
    assert nunique.max() == 1


def test_shared_grid_cell_detected(df):
    # QA-07: Salinas and Libertad share one ERA5 cell -> 21 cells for 22 cities
    assert df["weather_grid_id"].nunique() == 21
    cells = df.drop_duplicates("city").set_index("city")["weather_grid_id"]
    assert cells["Salinas"] == cells["Libertad"]


def test_sales_not_imputed(df):
    assert df["sales"].notna().all()
    assert (df["sales"] >= 0).all()


def test_store_active_flag(df):
    # QA-02: inactive rows exist only before the first sale and always have zero sales
    assert (df.loc[df["store_active"] == 0, "sales"] == 0).all()
    first_active = df[df["store_active"] == 1].groupby("store_nbr")["date"].min()
    assert first_active.loc[52] == pd.Timestamp("2017-04-20")


def test_wind_outliers_flagged_not_removed(df):
    # QA-09: implausible wind stays in the data but is flagged
    assert df["wind_qa_flag"].sum() > 0
    assert (df.loc[df["wind_qa_flag"] == 1, "wind_speed_10m_max"] > 100).all()


def test_flags_are_binary(df):
    flags = [c for c in df.columns if c.startswith("is_") or c in ("store_active", "store_zero_day", "wind_qa_flag")]
    for col in flags:
        assert set(df[col].unique()) <= {0, 1}, col
