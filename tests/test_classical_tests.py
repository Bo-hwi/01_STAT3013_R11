"""Tests for M2's classical tests (src/statistics/classical_tests.py).

The first tests use tiny hand-made data with an obvious answer.
The last test checks the saved result table, when it exists.

Run from the repository root::

    python -m pytest tests/test_classical_tests.py -v
"""
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.statistics import classical_tests as C


def fake_city_days(rain_effect: float, n: int = 400, seed: int = 0) -> pd.DataFrame:
    """City-day rows where rainy days have log-sales shifted by `rain_effect`."""
    rng = np.random.default_rng(seed)
    rainy = rng.random(n) < 0.4
    y = 7 + rain_effect * rainy + rng.normal(0, 0.2, n)
    temp = rng.normal(25, 3, n)
    return pd.DataFrame({
        "city": "A", "family": "BEVERAGES", "y": y, "rainy": rainy,
        "temp_band": pd.qcut(pd.Series(temp).rank(method="first"), 3, labels=["cool", "normal", "hot"]),
        "high_sales": y > np.median(y),
    })


def test_welch_finds_a_real_rain_effect():
    result = C.welch_rain_test(fake_city_days(rain_effect=-0.10))
    assert result["p_value"] < 0.001
    assert result["difference_pct"] < 0          # rainy days sell less
    assert result["effect_size"] < -0.3          # clear negative Cohen's d


def test_welch_finds_nothing_when_there_is_no_effect():
    assert C.welch_rain_test(fake_city_days(rain_effect=0.0))["p_value"] > 0.05


def test_chi_square_effect_size_between_0_and_1():
    result = C.chi_square_rain_test(fake_city_days(rain_effect=-0.10))
    assert 0 <= result["effect_size"] <= 1
    assert result["p_value"] < 0.05


def test_benjamini_hochberg_matches_hand_calculation():
    # p sorted: 0.01, 0.02, 0.04 with m = 3 -> q = 0.03, 0.03, 0.04
    q = C.benjamini_hochberg(pd.Series([0.04, 0.01, 0.02]))
    assert np.allclose(q.to_numpy(), [0.04, 0.03, 0.03])


def test_city_day_table_counts_weather_once_per_city_day():
    # Two stores in the same city and day must collapse to ONE row (weather is shared)
    curated = pd.DataFrame({
        "date": pd.to_datetime(["2016-01-01"] * 2 + ["2016-01-02"] * 2 + ["2016-01-03"] * 2),
        "store_nbr": [1, 2] * 3, "family": "DAIRY", "city": "Quito",
        "sales": [100.0, 300.0, 50.0, 150.0, 80.0, 120.0],
        "store_active": 1, "store_zero_day": 0,
        "precipitation_sum": [5.0, 5.0, 0.0, 0.0, 2.0, 2.0],
        "temperature_2m_max": [20.0, 20.0, 22.0, 22.0, 24.0, 24.0],
    })
    table = C.city_day_table(curated)
    assert len(table) == 3
    first = table.iloc[0]
    assert np.isclose(first["y"], np.mean(np.log1p([100.0, 300.0])))
    assert first["rainy"] and first["n_stores"] == 2


RESULTS = Path("results/tables/rq1_classical_tests.csv")


@pytest.mark.skipif(not RESULTS.exists(), reason="Run `python scripts/run_m2_classical_tests.py` first")
def test_saved_table_has_21_tests_with_fdr():
    out = pd.read_csv(RESULTS)
    assert len(out) == 21                                   # 7 families x 3 tests
    assert set(out["family"]) == set(C.FAMILIES)
    assert out["q_fdr"].between(0, 1).all()
    assert (out["q_fdr"] >= out["p_value"] - 1e-12).all()   # q is never below p
