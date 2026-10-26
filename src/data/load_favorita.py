"""Load raw Favorita CSV files with explicit dtypes and basic schema checks.

All loaders read from ``data/raw/favorita/`` relative to the repository root
unless another directory is passed. Nothing here modifies the data: cleaning
and flagging happen in ``clean_data.py``.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
FAVORITA_DIR = REPO_ROOT / "data" / "raw" / "favorita"


def _check_columns(df: pd.DataFrame, expected: list[str], name: str) -> None:
    """Fail loudly if a raw file does not have the columns we rely on."""
    missing = set(expected) - set(df.columns)
    if missing:
        raise ValueError(f"{name}: missing columns {sorted(missing)}")


def load_train(raw_dir: Path = FAVORITA_DIR) -> pd.DataFrame:
    """Daily sales per (date, store_nbr, family). ``id`` is dropped as it carries no information."""
    df = pd.read_csv(
        raw_dir / "train.csv",
        usecols=["date", "store_nbr", "family", "sales", "onpromotion"],
        dtype={"store_nbr": "int16", "family": "category", "sales": "float64", "onpromotion": "int32"},
        parse_dates=["date"],
    )
    _check_columns(df, ["date", "store_nbr", "family", "sales", "onpromotion"], "train.csv")
    return df


def load_stores(raw_dir: Path = FAVORITA_DIR) -> pd.DataFrame:
    """Store metadata: city, state, store type and cluster."""
    df = pd.read_csv(raw_dir / "stores.csv", dtype={"store_nbr": "int16", "cluster": "int16"})
    _check_columns(df, ["store_nbr", "city", "state", "type", "cluster"], "stores.csv")
    return df.rename(columns={"type": "store_type"})


def load_holidays(raw_dir: Path = FAVORITA_DIR) -> pd.DataFrame:
    """Holiday and event calendar (national, regional and local)."""
    df = pd.read_csv(raw_dir / "holidays_events.csv", parse_dates=["date"])
    _check_columns(df, ["date", "type", "locale", "locale_name", "description", "transferred"], "holidays_events.csv")
    df["transferred"] = df["transferred"].astype(bool)
    return df


def load_transactions(raw_dir: Path = FAVORITA_DIR) -> pd.DataFrame:
    """Number of transactions per (date, store_nbr). Forbidden as a same-day forecasting feature."""
    df = pd.read_csv(
        raw_dir / "transactions.csv",
        dtype={"store_nbr": "int16", "transactions": "int32"},
        parse_dates=["date"],
    )
    _check_columns(df, ["date", "store_nbr", "transactions"], "transactions.csv")
    return df


def load_oil(raw_dir: Path = FAVORITA_DIR) -> pd.DataFrame:
    """Daily WTI oil price (market days only). Optional control; not part of curated v1."""
    df = pd.read_csv(raw_dir / "oil.csv", parse_dates=["date"])
    _check_columns(df, ["date", "dcoilwtico"], "oil.csv")
    return df
