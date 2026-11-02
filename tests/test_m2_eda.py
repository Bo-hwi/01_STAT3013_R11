import pandas as pd
import numpy as np
import pytest
from src.statistics.eda import (
    compute_family_summary,
    compute_weather_correlations,
    compute_seasonality_stats,
    screen_deep_study_candidates
)

def test_compute_family_summary():
    sample = pd.DataFrame({
        "family": ["BEVERAGES", "BEVERAGES", "GROCERY I"],
        "sales": [10.0, 0.0, 50.0],
        "store_active": [1, 1, 1],
        "store_zero_day": [0, 0, 0]
    })
    summary = compute_family_summary(sample)
    assert "total_sales" in summary.columns
    assert "zero_rate" in summary.columns
    assert "sales_share" in summary.columns
    bev = summary.loc["BEVERAGES"]
    assert np.isclose(bev["zero_rate"], 0.5)
    assert np.isclose(bev["sales_share"], 10.0 / 60.0)

def test_compute_weather_correlations():
    sample = pd.DataFrame({
        "family": ["BEVERAGES"] * 4,
        "sales": [10.0, 20.0, 30.0, 40.0],
        "temperature_2m_max": [20.0, 25.0, 30.0, 35.0],
        "precipitation_sum": [0.0, 5.0, 0.0, 10.0]
    })
    corr = compute_weather_correlations(sample, ["temperature_2m_max", "precipitation_sum"])
    assert "temperature_2m_max" in corr.columns
    assert corr.loc["BEVERAGES", "temperature_2m_max"] > 0.9

def test_compute_seasonality_stats():
    sample = pd.DataFrame({
        "day_of_week": [0, 1, 2, 3],
        "month": [1, 1, 2, 2],
        "sales": [100.0, 120.0, 80.0, 90.0]
    })
    stats = compute_seasonality_stats(sample)
    assert "dow" in stats
    assert "month" in stats
    assert len(stats["dow"]) == 4

def test_screen_deep_study_candidates():
    sample = pd.DataFrame({
        "family": ["BEVERAGES"] * 10 + ["RARE"] * 10,
        "sales": [10.0] * 10 + [0.0] * 8 + [5.0] * 2,
        "store_active": [1] * 20,
        "store_zero_day": [0] * 20,
        "date": pd.date_range("2013-01-01", periods=10, freq="D").tolist() * 2,
        "city": ["Quito"] * 20
    })
    screening = screen_deep_study_candidates(sample)
    assert "BEVERAGES" in screening.index
    assert screening.loc["BEVERAGES", "zero_rate"] == 0.0
