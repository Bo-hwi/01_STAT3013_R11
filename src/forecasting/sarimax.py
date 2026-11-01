"""M1 time-series model: SARIMAX per city x family, with and without weather.

What is SARIMAX?
    Seasonal ARIMA with eXogenous regressors. For one daily series y it models
      - AR / MA terms: today depends on the last few days and recent shocks,
      - a seasonal part with period 7: today depends on the same weekday last week,
      - exogenous regressors X: known-in-advance calendar effects (and weather).
    Unlike M3's global ML models (one model for all series), SARIMAX estimates
    separate parameters for every series.

Scope (Roles v4.2 fallback: 3-4 families, city x family level)
    4 deep-study families: BEVERAGES, BREAD/BAKERY, FROZEN FOODS (likely weather
    sensitive) and GROCERY I (dry staples, a control) x 22 cities = 88 series.
    Series value = total sales of the city's OPEN stores that day.

Two exogenous configurations (same names as the ablation file)
    M0 (no weather) : n_open_stores, is_day_off, is_payday
    M3 (core weather): M0 + precipitation_sum_lag1 + temperature_2m_max_lag1
    Weather is the value of day t-1 (forecast made at the end of day t-1).
    Assumption: which stores are open on day t is known the day before
    (planned openings / closures). The shared score_mask already relies on the
    same assumption, and no test block contains a store opening or closing day.

Protocol
    1. Order (p,d,q)(P,D,Q)7 is chosen ONCE per series by AIC on the training
       block of fold F1 (data before 2016-08-04), using config M0. No validation
       or test data is used for this choice.
    2. For every fold: estimate the parameters on train + validation, then
       predict each test day one step ahead with parameters FIXED (the model sees
       the observed sales of day t-1 when predicting day t, like every other model).
    3. The model works on log(1 + sales) to stabilise the variance; forecasts are
       converted back with exp(x) - 1 and errors are measured in sales units.

Fair comparison with the baselines
    SARIMAX predicts a city total, so the store-level baselines are summed to the
    same city x family x day level. A city-day is scored only if every open store
    of that city is scored (score_mask) that day, so all models use the same rows.

Run from the repository root (after the pipeline and the baselines)::

    python -m src.forecasting.sarimax
"""
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.tsa.statespace.sarimax import SARIMAX

from .rolling_validation import read_yaml

FEATURES_PATH = Path("data/processed/model_features.parquet")
BASELINES_PATH = Path("results/predictions/baselines.csv")
PREDICTIONS_PATH = Path("results/predictions/sarimax.csv")
TABLES_DIR = Path("results/tables")

FAMILIES = ["BEVERAGES", "BREAD/BAKERY", "FROZEN FOODS", "GROCERY I"]
ORDER_CANDIDATES = [  # small grid, fixed in advance: (p,d,q), (P,D,Q,s)
    ((1, 0, 0), (0, 1, 1, 7)),
    ((1, 0, 1), (1, 1, 1, 7)),
    ((2, 0, 1), (1, 1, 1, 7)),
]
EXOG = {
    "M0": ["n_open_stores", "is_day_off", "is_payday"],
    "M3": ["n_open_stores", "is_day_off", "is_payday", "precipitation_sum_lag1", "temperature_2m_max_lag1"],
}
SERIES_START = "2013-01-02"  # first day that has a weather value for t-1
SERIES_END = "2017-08-15"
ORDER_SELECTION_END = "2016-08-03"  # last training day of fold F1
OUTPUT_COLUMNS = ["date", "city", "family", "fold", "model", "config", "y_true", "y_pred"]


# ---------------------------------------------------------------------------
# 1. Build one daily series per city x family
# ---------------------------------------------------------------------------
def build_city_series(features: pd.DataFrame) -> pd.DataFrame:
    """Aggregate store rows to city x family x day, on a complete daily calendar."""
    rows = features[features["family"].isin(FAMILIES)].copy()
    rows["is_open"] = (rows["store_active"] == 1) & (rows["store_zero_day"] == 0)
    rows["open_sales"] = rows["sales"].where(rows["is_open"], 0.0)
    # A city-day may be scored only if no open store in it is excluded by score_mask
    rows["open_but_masked"] = rows["is_open"] & ~rows["score_mask"]

    daily = rows.groupby(["city", "family", "date"], observed=True).agg(
        sales=("open_sales", "sum"),
        n_open_stores=("is_open", "sum"),
        any_open_masked=("open_but_masked", "any"),
        is_day_off=("is_day_off", "max"),  # a holiday in any store of the city
        is_payday=("is_payday", "max"),
        precipitation_sum_lag1=("precipitation_sum_lag1", "first"),  # same for the whole city
        temperature_2m_max_lag1=("temperature_2m_max_lag1", "first"),
    ).reset_index()

    calendar = pd.date_range(SERIES_START, SERIES_END, freq="D")
    series = []
    for (city, family), g in daily.groupby(["city", "family"], observed=True):
        g = g.set_index("date").reindex(calendar)  # 25 December becomes an empty day
        g["city"], g["family"] = city, family
        g[["n_open_stores", "is_day_off", "is_payday"]] = g[["n_open_stores", "is_day_off", "is_payday"]].fillna(0)
        # Weather of 24 December is still known on 25 December: carry it forward
        g[["precipitation_sum_lag1", "temperature_2m_max_lag1"]] = g[
            ["precipitation_sum_lag1", "temperature_2m_max_lag1"]].ffill()
        # No open store (closed chain, 25 December) -> unknown, not zero demand
        g["y"] = np.log1p(g["sales"].where(g["n_open_stores"] > 0))
        g["scored"] = (g["n_open_stores"] > 0) & ~g["any_open_masked"].fillna(True).astype(bool)
        series.append(g.rename_axis("date").reset_index())
    return pd.concat(series, ignore_index=True)


# ---------------------------------------------------------------------------
# 2. Fit, choose the order, forecast
# ---------------------------------------------------------------------------
def fit(y: pd.Series, X: pd.DataFrame, order, seasonal_order):
    """Estimate SARIMAX parameters; missing y values are skipped by the Kalman filter."""
    # Plain 0..n-1 positions: dates are already consecutive days, no gaps
    # Stationarity / invertibility are enforced so the estimated parameters cannot
    # describe an exploding series (this happened for cities whose only store
    # opened late or closed for months).
    model = SARIMAX(y.reset_index(drop=True), exog=X.reset_index(drop=True),
                    order=order, seasonal_order=seasonal_order)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # convergence is recorded in the log instead
        return model.fit(disp=False, maxiter=200)


def choose_order(s: pd.DataFrame) -> tuple:
    """Lowest AIC on the F1 training block, config M0 (no validation/test data)."""
    train = s[s["date"] <= ORDER_SELECTION_END]
    best = None
    for order, seasonal in ORDER_CANDIDATES:
        result = fit(train["y"], train[EXOG["M0"]], order, seasonal)
        if best is None or result.aic < best[2]:
            best = (order, seasonal, result.aic, result)
    order, seasonal, aic, result = best
    residuals = result.resid[14:].dropna()  # skip the start-up period of the filter
    ljung_p = float(acorr_ljungbox(residuals, lags=[14])["lb_pvalue"].iloc[0])
    return order, seasonal, aic, ljung_p


def one_step_forecasts(result, history: pd.DataFrame, upto_test: pd.DataFrame, exog: list[str]) -> np.ndarray:
    """Re-run the filter over data up to the test end with the SAME parameters.

    The prediction for day t only uses observations up to day t-1.
    """
    filtered = result.apply(upto_test["y"].reset_index(drop=True),
                            exog=upto_test[exog].reset_index(drop=True), refit=False)
    log_pred = filtered.predict(start=len(history), end=len(upto_test) - 1)
    return np.expm1(np.asarray(log_pred))


def forecast_fold(s: pd.DataFrame, fold: dict, order, seasonal, config: str) -> tuple[pd.DataFrame, dict]:
    """Fit on train + validation, then one-step-ahead forecasts over the test block."""
    exog = EXOG[config]
    test_start, test_end = (pd.Timestamp(d) for d in fold["test"])
    history = s[s["date"] <= pd.Timestamp(fold["validation"][1])]
    upto_test = s[s["date"] <= test_end]

    result = fit(history["y"], history[exog], order, seasonal)
    pred = one_step_forecasts(result, history, upto_test, exog)

    # Safety net: a forecast that is infinite or above 3x the largest sales ever
    # seen in history is not plausible -> refit with the simplest order instead.
    ceiling = 3 * history["sales"].max()
    fallback = not np.all(np.isfinite(pred)) or np.nanmax(pred) > ceiling
    if fallback:
        result = fit(history["y"], history[exog], *ORDER_CANDIDATES[0])
        pred = one_step_forecasts(result, history, upto_test, exog)
    status = {"converged": bool(result.mle_retvals.get("converged", True)), "fallback": fallback}

    test = upto_test.iloc[len(history):].copy()
    test["y_pred"] = np.clip(pred, 0, ceiling)
    test = test[(test["date"] >= test_start) & test["scored"]]
    out = pd.DataFrame({
        "date": test["date"], "city": test["city"], "family": test["family"],
        "fold": fold["fold_id"], "model": "SARIMAX", "config": config,
        "y_true": test["sales"], "y_pred": test["y_pred"],
    })
    return out, status


def run_one_series(s: pd.DataFrame, folds: list[dict]) -> tuple[list, dict]:
    """Everything for one city x family series (run in parallel across series)."""
    order, seasonal, aic, ljung_p = choose_order(s)
    outputs, n_not_converged, n_fallback = [], 0, 0
    for fold in folds:
        for config in EXOG:
            out, status = forecast_fold(s, fold, order, seasonal, config)
            outputs.append(out)
            n_not_converged += int(not status["converged"])
            n_fallback += int(status["fallback"])
    log = {"city": s["city"].iloc[0], "family": s["family"].iloc[0], "order": str(order),
           "seasonal_order": str(seasonal), "aic_f1_train": round(aic, 1),
           "ljung_box_p_lag14": round(ljung_p, 4), "fits_not_converged": n_not_converged,
           "fits_fallback_order": n_fallback}
    return outputs, log


# ---------------------------------------------------------------------------
# 3. Baselines at the same city level, and metrics
# ---------------------------------------------------------------------------
def baselines_at_city_level(sarimax_rows: pd.DataFrame) -> pd.DataFrame:
    """Sum store-level baseline forecasts to city x family x day, on the same scored days."""
    base = pd.read_csv(BASELINES_PATH, parse_dates=["date"])
    base = base[base["family"].isin(FAMILIES)]
    city = base.groupby(["date", "city", "family", "fold", "model", "config"], as_index=False)[
        ["y_true", "y_pred"]].sum()
    keep = sarimax_rows[["date", "city", "family", "fold"]].drop_duplicates()
    return city.merge(keep, on=["date", "city", "family", "fold"], how="inner")[OUTPUT_COLUMNS]


def mase_scales(series: pd.DataFrame, folds: list[dict]) -> pd.DataFrame:
    """Per fold and series: mean |sales(t) - sales(t-7)| on that fold's training block."""
    rows = []
    for fold in folds:
        train_end = pd.Timestamp(fold["train"][1])
        for (city, family), s in series.groupby(["city", "family"], observed=True):
            sales = s.set_index("date")["sales"].where(s.set_index("date")["n_open_stores"] > 0)
            diff = (sales - sales.shift(7, freq="D")).loc[:train_end].abs()
            rows.append({"fold": fold["fold_id"], "city": city, "family": family, "scale": diff.mean()})
    return pd.DataFrame(rows)


def metrics(pred: pd.DataFrame, scales: pd.DataFrame, by: list[str]) -> pd.DataFrame:
    p = pred.merge(scales, on=["fold", "city", "family"], how="left")
    p["abs_error"] = (p["y_true"] - p["y_pred"]).abs()
    p["scaled_error"] = p["abs_error"] / p["scale"]
    out = p.groupby(by, as_index=False).agg(
        n_scored=("abs_error", "size"), MAE=("abs_error", "mean"),
        sum_abs_error=("abs_error", "sum"), sum_y=("y_true", "sum"), MASE=("scaled_error", "mean"))
    out["WAPE"] = out["sum_abs_error"] / out["sum_y"]
    return out.drop(columns=["sum_abs_error", "sum_y"])


def main(n_jobs: int = -1) -> None:
    protocol = read_yaml("configs/research_protocol.yaml")
    folds = protocol["rq2"]["validation"]["test_folds"]
    missing = set(FAMILIES) - set(protocol["scope"]["deep_study_families"])
    if missing:
        raise ValueError(f"SARIMAX families must be in the frozen shortlist: {missing}")

    series = build_city_series(pd.read_parquet(FEATURES_PATH))
    groups = [g for _, g in series.groupby(["city", "family"], observed=True)]
    print(f"Fitting {len(groups)} series x {len(folds)} folds x {len(EXOG)} configs ...")
    results = Parallel(n_jobs=n_jobs)(delayed(run_one_series)(g, folds) for g in groups)

    sarimax = pd.concat([o for outputs, _ in results for o in outputs], ignore_index=True)
    log = pd.DataFrame([lg for _, lg in results])
    combined = pd.concat([sarimax, baselines_at_city_level(sarimax)], ignore_index=True)
    scales = mase_scales(series, folds)

    PREDICTIONS_PATH.parent.mkdir(parents=True, exist_ok=True)
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    sarimax.to_csv(PREDICTIONS_PATH, index=False)
    log.to_csv(TABLES_DIR / "sarimax_order_log.csv", index=False)
    by_fold = metrics(combined, scales, ["fold", "model", "config"])
    by_family = metrics(combined, scales, ["fold", "family", "model", "config"])
    by_fold.to_csv(TABLES_DIR / "sarimax_vs_baselines_by_fold.csv", index=False)
    by_family.to_csv(TABLES_DIR / "sarimax_vs_baselines_by_family.csv", index=False)

    print(by_fold.round(3).to_string(index=False))
    print(f"\nOrders chosen: {log['order'].astype(str).add(' ').add(log['seasonal_order']).value_counts().to_dict()}")
    n_fits = len(groups) * len(folds) * len(EXOG)
    print(f"Fits not converged: {int(log['fits_not_converged'].sum())} of {n_fits}")
    print(f"Fits that needed the fallback order: {int(log['fits_fallback_order'].sum())} of {n_fits}")
    print(f"Series with Ljung-Box p < 0.05 (residual autocorrelation left): {int((log['ljung_box_p_lag14'] < 0.05).sum())}")
    print(f"Saved {len(sarimax):,} SARIMAX predictions to {PREDICTIONS_PATH}")


if __name__ == "__main__":
    main()
