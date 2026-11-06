from pathlib import Path
import pandas as pd
import numpy as np
import pytest

from src.statistics.panel_models import (
    estimate_pooled_ols,
    estimate_fixed_effects,
    estimate_random_effects,
    hausman_test,
    apply_benjamini_hochberg,
    run_family_heterogeneity_models,
    run_earthquake_sensitivity_check,
)


@pytest.fixture
def synthetic_panel():
    np.random.seed(42)
    n = 200
    df = pd.DataFrame({
        "store_nbr": np.random.choice([1, 2, 3], size=n),
        "family": np.random.choice(["BEVERAGES", "CLEANING"], size=n),
        "city": np.random.choice(["Quito", "Guayaquil"], size=n),
        "weather_grid_id": np.random.choice(["grid_1", "grid_2"], size=n),
        "date": pd.date_range("2016-01-01", periods=n, freq="D"),
        "sales": np.random.exponential(100, size=n),
        "precipitation_sum": np.random.uniform(0, 20, size=n),
        "temperature_2m_max": np.random.uniform(15, 30, size=n),
        "relative_humidity_2m_mean": np.random.uniform(60, 95, size=n),
        "sunshine_hours": np.random.uniform(0, 12, size=n),
        "onpromotion": np.random.poisson(5, size=n),
        "is_day_off": np.random.choice([0, 1], size=n),
        "day_of_week": np.random.choice(range(7), size=n),
        "month": np.random.choice(range(1, 13), size=n),
        "year": np.random.choice([2015, 2016], size=n),
    })
    df["entity_id"] = df["store_nbr"].astype(str) + "_" + df["family"]
    return df


def test_estimate_pooled_ols(synthetic_panel):
    res = estimate_pooled_ols(
        synthetic_panel,
        weather_vars=["temperature_2m_max"],
        controls=["onpromotion", "is_day_off"],
        cluster_col="weather_grid_id"
    )
    assert "temperature_2m_max" in res["params"]
    assert "temperature_2m_max" in res["std_errors"]
    assert res["r_squared"] >= 0.0
    assert res["nobs"] == len(synthetic_panel)


def test_estimate_fixed_effects(synthetic_panel):
    res = estimate_fixed_effects(
        synthetic_panel,
        weather_vars=["temperature_2m_max"],
        controls=["onpromotion", "is_day_off"],
        entity_col="entity_id",
        cluster_col="weather_grid_id"
    )
    assert "temperature_2m_max" in res["params"]
    assert res["model_type"] == "Fixed Effects"
    assert res["cov_type"] == "clustered"
    assert "ci_lower" in res and "ci_upper" in res


def test_estimate_random_effects_and_hausman(synthetic_panel):
    fe_res = estimate_fixed_effects(
        synthetic_panel,
        weather_vars=["temperature_2m_max"],
        controls=["onpromotion"],
        entity_col="entity_id",
        cluster_col="weather_grid_id"
    )
    re_res = estimate_random_effects(
        synthetic_panel,
        weather_vars=["temperature_2m_max"],
        controls=["onpromotion"],
        entity_col="entity_id",
        cluster_col="weather_grid_id"
    )
    assert re_res["model_type"] == "Random Effects"
    assert re_res["cov_type"] == "clustered"
    assert "temperature_2m_max" in re_res["params"]
    assert "temperature_2m_max" in re_res["std_errors"]
    h_test = hausman_test(fe_res, re_res)
    assert "stat" in h_test and "p_value" in h_test


def test_benjamini_hochberg():
    p_vals = pd.Series([0.001, 0.01, 0.04, 0.10, 0.80])
    reject, q_vals = apply_benjamini_hochberg(p_vals, alpha=0.05)
    assert q_vals[0] < 0.01
    assert q_vals[-1] >= 0.80
    assert reject[0] == True


def test_heterogeneity_and_earthquake_sensitivity(synthetic_panel):
    het_df = run_family_heterogeneity_models(
        synthetic_panel,
        families=["BEVERAGES", "CLEANING"],
        weather_vars=["temperature_2m_max"],
        controls=["onpromotion"],
        entity_col="entity_id",
        cluster_col="weather_grid_id"
    )
    assert len(het_df) == 2
    assert "q_fdr" in het_df.columns
    assert "coef" in het_df.columns

    sens = run_earthquake_sensitivity_check(
        synthetic_panel,
        weather_vars=["temperature_2m_max"],
        controls=["onpromotion"],
        entity_col="entity_id",
        cluster_col="weather_grid_id"
    )
    assert "baseline_fe" in sens and "no_earthquake_fe" in sens


def test_rq1_coefficients_table_schema():
    path = Path("results/tables/rq1_coefficients.csv")
    assert path.exists(), "rq1_coefficients.csv must be produced"
    df = pd.read_csv(path)
    required = {
        "family", "predictor", "model_spec", "coef", "se_robust",
        "ci_lower_95", "ci_upper_95", "p_value", "q_fdr", "significant_fdr", "n_obs"
    }
    assert required.issubset(df.columns)
    assert len(df) >= 32  # 4 pooled + 28 family-specific (7 families * 4 vars)
    assert "temperature_2m_mean" not in df["predictor"].values, "Collinear temperature_2m_mean must be dropped"


def test_weather_signatures_for_m4_schema_and_boundaries():
    path = Path("results/tables/weather_signatures_for_m4.csv")
    assert path.exists(), "weather_signatures_for_m4.csv must exist"
    df = pd.read_csv(path)
    
    # Must not contain prescriptive pilot labels based on p-values
    for col in df.columns:
        if df[col].dtype == object:
            assert not df[col].str.contains("Prioritize pilot", case=False).any(), "No prescriptive pilot labels in M4 handoff"
            assert not df[col].str.contains("Do not deploy", case=False).any(), "No prescriptive pilot labels in M4 handoff"
            
    assert "econometric_signature" in df.columns
    assert "m4_handoff_disclaimer" in df.columns
