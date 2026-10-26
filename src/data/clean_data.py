"""Load weather files and build the flag tables used by the curated dataset.

Every function returns a new table and never fills or deletes sales values.
The decisions behind each flag are documented in ``docs/data_quality_report.md``
(QA codes are referenced in the comments below).
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
WEATHER_DIR = REPO_ROOT / "data" / "raw" / "weather"

# Weather columns kept in the curated table.
# sunshine_duration (seconds) is dropped: it is the same variable as sunshine_hours (QA-10).
WEATHER_VALUE_COLS = [
    "precipitation_sum",          # mm/day     - core (ablation M1, M3)
    "temperature_2m_max",         # deg C      - core (ablation M2, M3)
    "temperature_2m_mean",        # deg C      - robustness
    "relative_humidity_2m_mean",  # %          - extended (ablation M4)
    "sunshine_hours",             # hours/day  - extended (ablation M4)
    "wind_speed_10m_max",         # km/h       - candidate M5, NOT QA-approved (QA-09)
    "cloud_cover_mean",           # %          - candidate M5
]
WEATHER_META_COLS = ["source_model", "grid_latitude", "grid_longitude"]

# Wind values above this are physically implausible for the region (QA-09).
WIND_PLAUSIBLE_MAX_KMH = 100.0

# Day-off holiday types: a transferred Holiday is NOT a day off on its original date;
# the matching "Transfer" row carries the actual day off.
DAY_OFF_TYPES = {"Holiday", "Transfer", "Additional", "Bridge"}


# ---------------------------------------------------------------------------
# Weather
# ---------------------------------------------------------------------------
def load_weather(weather_dir: Path = WEATHER_DIR) -> pd.DataFrame:
    """Stack the 22 per-city raw weather files into one (city, date) table."""
    files = sorted(p for p in weather_dir.glob("weather_*.csv") if p.name != "weather_download_log.csv")
    if not files:
        raise FileNotFoundError(f"No weather_*.csv files found in {weather_dir}")

    frames = [pd.read_csv(p, parse_dates=["date"]) for p in files]
    wx = pd.concat(frames, ignore_index=True)[["city", "date", *WEATHER_VALUE_COLS, *WEATHER_META_COLS]]

    # One row per (city, date) is required for a many-to-one join
    dup = wx.duplicated(["city", "date"]).sum()
    if dup:
        raise ValueError(f"Weather has {dup} duplicated (city, date) rows")
    return wx


def add_weather_qa_flags(wx: pd.DataFrame) -> pd.DataFrame:
    """Add QA flags and a grid-cell id without modifying any measured value."""
    out = wx.copy()
    # QA-09: keep raw wind but mark implausible values; M5 must not use unflagged wind
    out["wind_qa_flag"] = (out["wind_speed_10m_max"] > WIND_PLAUSIBLE_MAX_KMH).astype("int8")
    # QA-07: cities sharing an ERA5 cell (Salinas/Libertad) get the same id -> use for clustering
    out["weather_grid_id"] = (
        out["grid_latitude"].map("{:.2f}".format) + "_" + out["grid_longitude"].map("{:.2f}".format)
    )
    return out


# ---------------------------------------------------------------------------
# Holidays
# ---------------------------------------------------------------------------
def build_holiday_flags(holidays: pd.DataFrame, stores: pd.DataFrame) -> pd.DataFrame:
    """Expand the holiday calendar to one row per (date, store_nbr) with 0/1 flags.

    National rows apply to every store, Regional rows to stores in that state,
    Local rows to stores in that city. Several rows on the same date are merged
    with ``max`` so the result never duplicates sales rows.
    """
    h = holidays.copy()
    h["is_day_off"] = (h["type"].isin(DAY_OFF_TYPES) & ~h["transferred"]).astype("int8")
    h["is_event"] = (h["type"] == "Event").astype("int8")
    h["is_work_day"] = (h["type"] == "Work Day").astype("int8")
    h["is_earthquake_period"] = h["description"].str.contains("Terremoto", case=False).astype("int8")

    flag_cols = ["is_day_off", "is_event", "is_work_day", "is_earthquake_period"]
    st = stores[["store_nbr", "city", "state"]]

    nat = h[h["locale"] == "National"].merge(st, how="cross")
    reg = h[h["locale"] == "Regional"].merge(st, left_on="locale_name", right_on="state")
    loc = h[h["locale"] == "Local"].merge(st, left_on="locale_name", right_on="city")

    parts = []
    for scope, df in (("national", nat), ("regional", reg), ("local", loc)):
        g = df.groupby(["date", "store_nbr"], as_index=False)[flag_cols].max()
        g[f"is_{scope}_holiday"] = g["is_day_off"]
        parts.append(g)

    flags = (
        pd.concat(parts, ignore_index=True)
        .groupby(["date", "store_nbr"], as_index=False)
        .max()
    )
    scope_cols = ["is_national_holiday", "is_regional_holiday", "is_local_holiday"]
    flags[scope_cols] = flags[scope_cols].fillna(0)
    return flags[["date", "store_nbr", *flag_cols, *scope_cols]]


# ---------------------------------------------------------------------------
# Store activity
# ---------------------------------------------------------------------------
def build_store_day_flags(train: pd.DataFrame) -> pd.DataFrame:
    """Per (date, store_nbr): opening status and full-store zero-sales days.

    - ``store_active`` = 0 before the store's first day with non-zero total sales (QA-02).
    - ``store_zero_day`` = 1 when the whole store sold nothing that day, e.g. 1 January (QA-05).
      These are closures or missing records, not zero demand.
    """
    daily = train.groupby(["store_nbr", "date"], as_index=False, observed=True)["sales"].sum()
    first_sale = (
        daily[daily["sales"] > 0].groupby("store_nbr")["date"].min().rename("store_first_sale_date")
    )
    daily = daily.merge(first_sale, on="store_nbr", how="left")
    daily["store_active"] = (daily["date"] >= daily["store_first_sale_date"]).astype("int8")
    daily["store_zero_day"] = (daily["sales"] == 0).astype("int8")
    return daily[["date", "store_nbr", "store_active", "store_zero_day"]]
