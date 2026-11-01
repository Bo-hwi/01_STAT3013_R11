"""Metrics with a frozen, training-only seasonal scale."""

import numpy as np
import pandas as pd

from .rolling_validation import calendar_sales_lag

SERIES = ["store_nbr", "family"]


def mase_scales(train: pd.DataFrame, history: pd.DataFrame) -> pd.DataFrame:
    # Targets come only from eligible training rows; lag sources retain closures.
    lagged = history[["date", *SERIES]].copy()
    lagged["lag7"] = calendar_sales_lag(history, 7)
    pairs = train[["date", *SERIES, "sales"]].merge(
        lagged, on=["date", *SERIES], validate="one_to_one", how="left"
    )
    pairs["mase_scale"] = (pairs["sales"] - pairs["lag7"]).abs()
    return pairs.groupby(SERIES, as_index=False)["mase_scale"].mean()


def metrics(predictions: pd.DataFrame, scales: pd.DataFrame) -> dict:
    values = predictions.merge(scales, on=SERIES, how="left", validate="many_to_one")
    error = (values["y_true"] - values["y_pred"]).abs()
    denominator = values["y_true"].abs().sum()
    valid = values["mase_scale"].gt(0) & np.isfinite(values["mase_scale"])
    return {
        "n_scored": len(values),
        "MAE": float(error.mean()),
        # WAPE is a ratio; multiply by 100 only when displaying a percentage.
        "WAPE": float(error.sum() / denominator) if denominator > 0 else np.nan,
        "MASE": float((error[valid] / values.loc[valid, "mase_scale"]).mean()),
        "mase_valid_rows": int(valid.sum()),
        "mase_coverage": float(valid.mean()),
    }


def metric_report(predictions: pd.DataFrame, scales_by_fold: dict) -> pd.DataFrame:
    rows = []
    # Include fold summaries and the required family-city breakdown.
    for grouping in [
        ["fold", "model", "config"],
        ["fold", "model", "config", "family", "city"],
        ["fold", "model", "config", "family", "city", "store_nbr"],
    ]:
        for key, group in predictions.groupby(grouping, observed=True):
            identifiers = dict(zip(grouping, key))
            level = "store" if "store_nbr" in grouping else "family_city" if "family" in grouping else "fold"
            rows.append({"report_level": level, **identifiers, **metrics(group, scales_by_fold[identifiers["fold"]])})
    return pd.DataFrame(rows)
