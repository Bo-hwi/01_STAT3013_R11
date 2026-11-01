"""Build the time-safe feature table ``model_features.parquet`` from the curated table.

Every feature for target date t uses only information available at the
forecast origin (end of day t-1):

- sales lags and rolling means look strictly into the past, by CALENDAR date;
- weather is the city value of day t-1 (never day t);
- calendar / holiday flags and ``onpromotion`` describe day t and are treated
  as known in advance (``onpromotion`` remains "conditional" until the team
  confirms the advance-schedule assumption).

Column names follow the contract agreed with M3 (``src/forecasting/feature_spec.py``).
Same-day weather, wind and transactions are deliberately NOT written, so they
cannot leak into a model by accident.

Run from the repository root::

    python -m src.features.lag_features
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from src.data.clean_data import WEATHER_DIR, load_weather

REPO_ROOT = Path(__file__).resolve().parents[2]
CURATED_PATH = REPO_ROOT / "data" / "processed" / "curated_sales_weather.parquet"
OUT_PATH = REPO_ROOT / "data" / "processed" / "model_features.parquet"
META_PATH = REPO_ROOT / "data" / "processed" / "model_features.meta.json"

KEY = ["date", "store_nbr", "family"]
SERIES = ["store_nbr", "family"]

SALES_LAGS = {1: "sales_lag1", 7: "sales_lag7", 14: "sales_lag14", 28: "sales_lag28"}
SALES_ROLLING = {7: "sales_roll7_mean", 28: "sales_roll28_mean"}
WEATHER_LAG1 = {
    "precipitation_sum": "precipitation_sum_lag1",
    "temperature_2m_max": "temperature_2m_max_lag1",
    "relative_humidity_2m_mean": "relative_humidity_2m_mean_lag1",
    "sunshine_hours": "sunshine_hours_lag1",
}
# Shared scoring rule from configs/ablation_M0_M5.yaml -> scoring_exclusions
MIN_CLOSURE_DAYS = 7
POST_REOPENING_DAYS = 28

# Columns copied unchanged from the curated table (all known for day t in advance)
PASSTHROUGH = [
    "date", "store_nbr", "family", "sales", "onpromotion",
    "city", "state", "store_type", "cluster", "weather_grid_id",
    "store_active", "store_zero_day",
    "year", "month", "day_of_week", "is_payday",
    "is_day_off", "is_national_holiday", "is_regional_holiday", "is_local_holiday",
    "is_event", "is_work_day", "is_earthquake_period",
]
# Must never appear in the feature table (forbidden at forecast origin, or failed QA)
FORBIDDEN = [
    "transactions", "precipitation_sum", "temperature_2m_max", "temperature_2m_mean",
    "relative_humidity_2m_mean", "sunshine_hours", "cloud_cover_mean",
    "wind_speed_10m_max", "wind_qa_flag",
]


# ---------------------------------------------------------------------------
# Sales history
# ---------------------------------------------------------------------------
def add_sales_lags(df: pd.DataFrame) -> pd.DataFrame:
    """sales_lagK = sales of the same store x family on calendar date (date - K days).

    A self-merge on the shifted date is used instead of ``shift(K)``: rows are
    missing on 25 December, so shifting by row position would be off by one day
    around Christmas. A missing date gives NaN, never 0.
    """
    base = df[KEY + ["sales"]]
    for k, col in SALES_LAGS.items():
        past = base.rename(columns={"sales": col}).assign(date=base["date"] + np.timedelta64(k, "D"))
        df = df.merge(past, on=KEY, how="left", validate="1:1")
    return df


def add_sales_rolling(df: pd.DataFrame) -> pd.DataFrame:
    """Mean sales over the calendar window [t - K, t), i.e. days t-K .. t-1.

    ``closed="left"`` excludes day t itself (no leakage); a time-based window
    ("7D") skips absent dates instead of counting rows.
    """
    df = df.sort_values(SERIES + ["date"], ignore_index=True)
    grouped = df.groupby(SERIES, observed=True, sort=False)
    for k, col in SALES_ROLLING.items():
        rolled = grouped.rolling(f"{k}D", on="date", closed="left", min_periods=1)["sales"].mean()
        df[col] = rolled.to_numpy()  # same order: groups follow the sorted frame
    return df


# ---------------------------------------------------------------------------
# Weather
# ---------------------------------------------------------------------------
def add_weather_lag1(df: pd.DataFrame, weather_dir: Path = WEATHER_DIR) -> pd.DataFrame:
    """City weather of day t-1, taken from the raw weather files.

    Raw files (not the curated table) are used so that 26 December still gets
    the weather of 25 December, a day with no sales rows.
    """
    wx = load_weather(weather_dir)[["city", "date", *WEATHER_LAG1]]
    wx = wx.rename(columns=WEATHER_LAG1).assign(date=wx["date"] + np.timedelta64(1, "D"))
    df = df.astype({"city": str}).merge(wx, on=["city", "date"], how="left", validate="m:1")
    return df


# ---------------------------------------------------------------------------
# Shared scoring mask
# ---------------------------------------------------------------------------
def build_score_mask(df: pd.DataFrame) -> pd.DataFrame:
    """Per (date, store_nbr): True if the row may be used for validation/test scoring.

    False when the store is not yet open, closed that day, or within the first
    28 calendar days after a closure of at least 7 consecutive days. Consecutive
    means adjacent calendar dates; a missing date (25 December) breaks a run,
    exactly as in M3's ``add_scoring_mask``.
    """
    days = (
        df[["store_nbr", "date", "store_active", "store_zero_day"]]
        .drop_duplicates()
        .sort_values(["store_nbr", "date"], ignore_index=True)
    )
    days["excluded_post_reopening"] = False

    for _, s in days.groupby("store_nbr", sort=False):
        closed = ((s["store_active"] == 1) & (s["store_zero_day"] == 1)).to_numpy()
        active = (s["store_active"] == 1).to_numpy()
        dates = s["date"].to_numpy()
        idx = s.index.to_numpy()
        run = 0
        for i in range(len(s)):
            adjacent = i > 0 and (dates[i] - dates[i - 1]) == np.timedelta64(1, "D")
            if closed[i]:
                run = run + 1 if (adjacent and closed[i - 1]) else 1
                continue
            if adjacent and closed[i - 1] and run >= MIN_CLOSURE_DAYS and active[i]:
                window_end = dates[i] + np.timedelta64(POST_REOPENING_DAYS - 1, "D")
                in_window = (dates >= dates[i]) & (dates <= window_end) & ~closed
                days.loc[idx[in_window], "excluded_post_reopening"] = True
            run = 0

    days["score_mask"] = (
        (days["store_active"] == 1) & (days["store_zero_day"] == 0) & ~days["excluded_post_reopening"]
    )
    return days[["store_nbr", "date", "excluded_post_reopening", "score_mask"]]


# ---------------------------------------------------------------------------
# Build + validate
# ---------------------------------------------------------------------------
def build_features(curated: pd.DataFrame, weather_dir: Path = WEATHER_DIR) -> pd.DataFrame:
    n_rows = len(curated)
    df = curated[PASSTHROUGH].copy()
    df = add_sales_lags(df)
    df = add_sales_rolling(df)
    df = add_weather_lag1(df, weather_dir)
    df = df.merge(build_score_mask(df), on=["store_nbr", "date"], how="left", validate="m:1")

    for col in ["city", "state", "store_type", "weather_grid_id"]:
        df[col] = df[col].astype("category")
    df = df.sort_values(KEY, ignore_index=True)
    validate_features(df, n_rows)
    return df


def validate_features(df: pd.DataFrame, n_rows: int) -> None:
    if len(df) != n_rows:
        raise AssertionError(f"Row count changed: {n_rows} -> {len(df)}")
    if df.duplicated(KEY).any():
        raise AssertionError("Duplicated keys in feature table")
    leaked = [c for c in FORBIDDEN if c in df.columns]
    if leaked:
        raise AssertionError(f"Forbidden columns present: {leaked}")
    if df["score_mask"].isna().any() or (df.loc[df["store_active"] == 0, "score_mask"]).any():
        raise AssertionError("score_mask must be False for inactive stores and never null")
    # Only the very first day (2013-01-01) may lack weather of the previous day
    missing_wx = df.loc[df["precipitation_sum_lag1"].isna(), "date"].unique()
    if len(missing_wx) > 1 or (len(missing_wx) == 1 and missing_wx[0] != df["date"].min()):
        raise AssertionError(f"Unexpected missing weather lags on {missing_wx[:5]}")


def write_metadata(df: pd.DataFrame) -> dict:
    meta = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "source": "data/processed/curated_sales_weather.parquet + data/raw/weather/*.csv",
        "n_rows": int(len(df)),
        "n_columns": int(df.shape[1]),
        "key": KEY,
        "forecast_origin": "end of day t-1",
        "features": {
            "sales_lags": SALES_LAGS,
            "sales_rolling": SALES_ROLLING,
            "weather_lag1": list(WEATHER_LAG1.values()),
            "known_in_advance": PASSTHROUGH[4:5] + PASSTHROUGH[12:],
            "conditional": ["onpromotion"],
        },
        "available_at": {
            "sales_lag*, sales_roll*": "end of day t-1 (observed sales)",
            "*_lag1 weather": "retrospective ERA5 value of day t-1 (publication lag ignored)",
            "calendar / holiday flags": "known in advance",
            "onpromotion": "assumed known in advance (pending team decision)",
        },
        "score_mask_rule": {
            "excluded_when": ["store_active == 0", "store_zero_day == 1",
                              f"first {POST_REOPENING_DAYS} days after a closure of >= {MIN_CLOSURE_DAYS} days"],
            "rows_scored": int(df["score_mask"].sum()),
            "rows_excluded_post_reopening": int(df["excluded_post_reopening"].sum()),
        },
        "forbidden_columns_absent": FORBIDDEN,
    }
    META_PATH.write_text(json.dumps(meta, indent=2, default=str), encoding="utf-8")
    return meta


def main() -> None:
    if not CURATED_PATH.exists():
        raise FileNotFoundError("Curated table missing: run `python -m src.data.merge_data` first")
    df = build_features(pd.read_parquet(CURATED_PATH))
    df.to_parquet(OUT_PATH, index=False)
    meta = write_metadata(df)
    print(f"Saved {OUT_PATH.relative_to(REPO_ROOT)}  ({meta['n_rows']:,} rows x {meta['n_columns']} cols)")
    print(f"Rows scored: {meta['score_mask_rule']['rows_scored']:,} | "
          f"excluded post-reopening: {meta['score_mask_rule']['rows_excluded_post_reopening']:,}")
    print("All feature checks passed.")


if __name__ == "__main__":
    main()
