"""Build the curated (date, store_nbr, family) table: sales + store info + calendar + weather.

Join design
-----------
``train`` is the driving table and every other table is joined many-to-one with
``validate="m:1"``. As a result:

- the row count never changes (checked),
- days absent from train (25 December, QA-01) are never created,
- missing sales are never filled with zero.

Run from the repository root::

    python -m src.data.merge_data
"""
from __future__ import annotations

import hashlib
import json
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from src.data.clean_data import (
    WEATHER_DIR,
    add_weather_qa_flags,
    build_holiday_flags,
    build_store_day_flags,
    load_weather,
)
from src.data.load_favorita import (
    FAVORITA_DIR,
    load_holidays,
    load_stores,
    load_train,
    load_transactions,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
OUT_PATH = REPO_ROOT / "data" / "processed" / "curated_sales_weather.parquet"
META_PATH = REPO_ROOT / "data" / "processed" / "curated_sales_weather.meta.json"

KEY = ["date", "store_nbr", "family"]
FLAG_COLS = [
    "is_day_off", "is_event", "is_work_day", "is_earthquake_period",
    "is_national_holiday", "is_regional_holiday", "is_local_holiday",
]


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def build_curated(favorita_dir: Path = FAVORITA_DIR, weather_dir: Path = WEATHER_DIR) -> pd.DataFrame:
    """Return the curated table; raises if any integrity rule is broken."""
    train = load_train(favorita_dir)
    stores = load_stores(favorita_dir)
    holidays = load_holidays(favorita_dir)
    transactions = load_transactions(favorita_dir)
    weather = add_weather_qa_flags(load_weather(weather_dir))
    n_train = len(train)

    # 1) Store attributes (54 stores -> many-to-one)
    df = train.merge(stores, on="store_nbr", how="left", validate="m:1")

    # 2) Store-day flags: opening status and whole-store zero days
    df = df.merge(build_store_day_flags(train), on=["date", "store_nbr"], how="left", validate="m:1")

    # 3) Holiday flags; dates with no holiday row get 0
    df = df.merge(build_holiday_flags(holidays, stores), on=["date", "store_nbr"], how="left", validate="m:1")
    df[FLAG_COLS] = df[FLAG_COLS].fillna(0).astype("int8")

    # 4) Transactions (same-day value: EDA/lag only, forbidden as a day-t forecasting feature)
    df = df.merge(transactions, on=["date", "store_nbr"], how="left", validate="m:1")

    # 5) Weather: city-day value shared by every store in the city
    df = df.merge(weather, on=["city", "date"], how="left", validate="m:1")

    # 6) Calendar fields known in advance for the target date
    df["year"] = df["date"].dt.year.astype("int16")
    df["month"] = df["date"].dt.month.astype("int8")
    df["day_of_week"] = df["date"].dt.dayofweek.astype("int8")  # Monday = 0
    df["is_payday"] = ((df["date"].dt.day == 15) | df["date"].dt.is_month_end).astype("int8")

    for col in ["city", "state", "store_type", "weather_grid_id", "source_model"]:
        df[col] = df[col].astype("category")

    df = df.sort_values(KEY, ignore_index=True)
    validate_curated(df, n_train)
    return df


def validate_curated(df: pd.DataFrame, n_train: int) -> None:
    """Integrity rules from the Proposal data contract. Raise on the first failure."""
    if len(df) != n_train:
        raise AssertionError(f"Row count changed after joins: {n_train} -> {len(df)}")
    if df.duplicated(KEY).any():
        raise AssertionError("Duplicated (date, store_nbr, family) keys")
    christmas = df["date"].dt.strftime("%m-%d").eq("12-25").sum()
    if christmas:
        raise AssertionError(f"{christmas} rows on 25 December were created")
    if df["city"].isna().any():
        raise AssertionError("Some stores have no city after the store join")
    if df["precipitation_sum"].isna().any():
        missing = df.loc[df["precipitation_sum"].isna(), "city"].unique().tolist()
        raise AssertionError(f"Weather missing after join for cities: {missing}")
    if df["sales"].isna().any():
        raise AssertionError("Sales contain NaN")


def write_metadata(df: pd.DataFrame, favorita_dir: Path, weather_dir: Path) -> dict:
    """Record provenance so the curated file can be traced to exact inputs."""
    sources = [favorita_dir / f for f in ("train.csv", "stores.csv", "holidays_events.csv", "transactions.csv")]
    sources += sorted(p for p in weather_dir.glob("weather_*.csv") if p.name != "weather_download_log.csv")
    meta = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "pandas": pd.__version__,
        "n_rows": int(len(df)),
        "n_columns": int(df.shape[1]),
        "key": KEY,
        "date_min": str(df["date"].min().date()),
        "date_max": str(df["date"].max().date()),
        "n_dates": int(df["date"].nunique()),
        "n_stores": int(df["store_nbr"].nunique()),
        "n_families": int(df["family"].nunique()),
        "n_cities": int(df["city"].nunique()),
        "n_weather_grid_cells": int(df["weather_grid_id"].nunique()),
        "rows_store_inactive": int((df["store_active"] == 0).sum()),
        "rows_store_zero_day": int((df["store_zero_day"] == 1).sum()),
        "rows_wind_qa_flag": int((df["wind_qa_flag"] == 1).sum()),
        "columns": {c: str(t) for c, t in df.dtypes.items()},
        "source_sha256": {p.name: _sha256(p) for p in sources},
    }
    META_PATH.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return meta


def main() -> None:
    df = build_curated()
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(OUT_PATH, index=False)
    meta = write_metadata(df, FAVORITA_DIR, WEATHER_DIR)

    print(f"Saved {OUT_PATH.relative_to(REPO_ROOT)}  ({meta['n_rows']:,} rows x {meta['n_columns']} cols)")
    print(f"Dates {meta['date_min']} -> {meta['date_max']} ({meta['n_dates']} days) | "
          f"stores {meta['n_stores']} | families {meta['n_families']} | cities {meta['n_cities']} | "
          f"grid cells {meta['n_weather_grid_cells']}")
    print(f"Rows: store inactive {meta['rows_store_inactive']:,} | whole-store zero day "
          f"{meta['rows_store_zero_day']:,} | wind QA flag {meta['rows_wind_qa_flag']:,}")
    print("All integrity checks passed.")


if __name__ == "__main__":
    main()
