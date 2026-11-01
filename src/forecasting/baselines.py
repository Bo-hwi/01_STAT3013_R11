"""M1 baselines: Seasonal Naive and Moving Average on the frozen folds.

Why baselines?
    They are the simplest sensible forecasts. A weather-aware ML model is only
    useful if it beats them on the same test rows.

The two models (forecast for day t, made at the end of day t-1)
    Seasonal Naive : y_hat(t) = sales(t - 7)
                     "same weekday last week". Retail sales repeat every 7 days.
    Moving Average : y_hat(t) = mean of sales on days t-k .. t-1, with k = 7 or 28.
                     k is chosen on the validation block of each fold (lowest MAE),
                     never on the test block.

Neither model learns weights. Both simply read history columns that M1 already
built in model_features.parquet (sales_lag7, sales_roll7_mean, sales_roll28_mean),
so there is no leakage risk: those columns only look at days before t.

Missing history (rare): if sales(t-7) does not exist (e.g. t-7 = 25 December, or
a store opened less than 7 days ago), Seasonal Naive falls back to sales(t-14),
then to the 7-day mean. Every fallback is counted and reported.

Shared with M3 so that every model is scored identically:
    iter_folds   -> same train / validation / test rows and the same score_mask
    mase_scales,
    metric_report -> same MAE / WAPE / MASE formulas

Run from the repository root (after `python scripts/run_pipeline.py`)::

    python -m src.forecasting.baselines
"""
from pathlib import Path

import numpy as np
import pandas as pd

from .evaluation import mase_scales, metric_report
from .rolling_validation import iter_folds, read_yaml

FEATURES_PATH = Path("data/processed/model_features.parquet")
PREDICTIONS_PATH = Path("results/predictions/baselines.csv")
TABLES_DIR = Path("results/tables")

MA_WINDOWS = {7: "sales_roll7_mean", 28: "sales_roll28_mean"}  # candidate k -> column
OUTPUT_COLUMNS = ["date", "store_nbr", "family", "city", "fold", "model", "config", "y_true", "y_pred"]


# ---------------------------------------------------------------------------
# 1. The two forecasting rules
# ---------------------------------------------------------------------------
def seasonal_naive(rows: pd.DataFrame) -> tuple[pd.Series, int]:
    """Forecast = sales 7 days earlier; fall back to 14 days, then the 7-day mean."""
    forecast = rows["sales_lag7"]
    n_fallback = int(forecast.isna().sum())
    forecast = forecast.fillna(rows["sales_lag14"]).fillna(rows["sales_roll7_mean"])
    return forecast, n_fallback


def moving_average(rows: pd.DataFrame, k: int) -> pd.Series:
    """Forecast = mean sales over the k days before t (column built by M1)."""
    return rows[MA_WINDOWS[k]]


# ---------------------------------------------------------------------------
# 2. Choosing k for the Moving Average (validation only)
# ---------------------------------------------------------------------------
def choose_window(validation: pd.DataFrame) -> tuple[int, list[dict]]:
    """Try every candidate k on the scored validation rows; keep the lowest MAE."""
    scored = validation[validation["score_mask"]]
    results = []
    for k in MA_WINDOWS:
        mae = float((scored["sales"] - moving_average(scored, k)).abs().mean())
        results.append({"k": k, "validation_MAE": mae})
    best_k = min(results, key=lambda r: r["validation_MAE"])["k"]
    return best_k, results


# ---------------------------------------------------------------------------
# 3. Run both baselines on every fold
# ---------------------------------------------------------------------------
def to_output(test: pd.DataFrame, forecast: pd.Series, fold: str, model: str) -> pd.DataFrame:
    """Shape predictions in the team's shared prediction format."""
    out = test[["date", "store_nbr", "family", "city"]].copy()
    out["fold"], out["model"], out["config"] = fold, model, "M0"  # M0 = no weather
    out["y_true"], out["y_pred"] = test["sales"], forecast
    return out[OUTPUT_COLUMNS]


def run_baselines(data: pd.DataFrame, protocol: dict, ablation: dict, families: list[str]):
    predictions, k_choices, fallbacks, scales_by_fold = [], [], [], {}

    for fold in iter_folds(data, protocol, ablation, families):
        # MASE denominator from training rows only (frozen before looking at test)
        scales_by_fold[fold.fold_id] = mase_scales(fold.train, data)
        test = fold.test[fold.test["score_mask"]]  # score only rows allowed by the shared rule

        # Seasonal Naive: nothing to choose, predict test directly
        sn_forecast, n_fallback = seasonal_naive(test)
        predictions.append(to_output(test, sn_forecast, fold.fold_id, "SeasonalNaive"))
        fallbacks.append({"fold": fold.fold_id, "rows_scored": len(test), "seasonal_naive_fallbacks": n_fallback})

        # Moving Average: pick k on validation, then apply it to test unchanged
        best_k, tried = choose_window(fold.validation)
        k_choices += [{"fold": fold.fold_id, **row, "selected": row["k"] == best_k} for row in tried]
        predictions.append(to_output(test, moving_average(test, best_k), fold.fold_id, "MovingAverage"))

    result = pd.concat(predictions, ignore_index=True)
    if result["y_pred"].isna().any() or not np.isfinite(result["y_pred"]).all():
        raise ValueError("Some baseline forecasts are missing; check the history columns")
    report = metric_report(result, scales_by_fold)
    return result, report, pd.DataFrame(k_choices), pd.DataFrame(fallbacks)


def main() -> None:
    protocol = read_yaml("configs/research_protocol.yaml")
    ablation = read_yaml("configs/ablation_M0_M5.yaml")
    families = protocol["scope"]["deep_study_families"]
    if not families:
        raise ValueError("deep_study_families is empty in configs/research_protocol.yaml")

    data = pd.read_parquet(FEATURES_PATH)
    predictions, report, k_choices, fallbacks = run_baselines(data, protocol, ablation, families)

    PREDICTIONS_PATH.parent.mkdir(parents=True, exist_ok=True)
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    predictions.to_csv(PREDICTIONS_PATH, index=False)
    report.to_csv(TABLES_DIR / "baselines_metrics.csv", index=False)
    k_choices.to_csv(TABLES_DIR / "baselines_ma_window_selection.csv", index=False)
    fallbacks.to_csv(TABLES_DIR / "baselines_fallbacks.csv", index=False)

    summary = report[report["report_level"] == "fold"][["fold", "model", "n_scored", "MAE", "WAPE", "MASE"]]
    print(summary.round(3).to_string(index=False))
    print("\nMoving Average window chosen on validation:")
    print(k_choices[k_choices["selected"]][["fold", "k", "validation_MAE"]].round(2).to_string(index=False))
    print(f"\nSeasonal Naive fallbacks: {int(fallbacks['seasonal_naive_fallbacks'].sum())}")
    print(f"Saved {len(predictions):,} predictions to {PREDICTIONS_PATH}")


if __name__ == "__main__":
    main()
