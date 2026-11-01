"""Quick QA checks: print a Markdown hash table and verify initial findings on raw data."""
from pathlib import Path

import pandas as pd

RAW = Path("data/raw/favorita")

# 1) Markdown table of file hashes, ready to paste into data/README.md
catalog = pd.read_csv("docs/source_catalog.csv")
print("| File | Rows | SHA-256 |\n|---|---|---|")
for _, r in catalog.iterrows():
    print(f"| {r['file']} | {r['n_rows']:,} | `{r['sha256']}` |")

# 2) Which calendar days are missing from train.csv?
train = pd.read_csv(RAW / "train.csv", usecols=["date", "store_nbr", "family", "sales"], parse_dates=["date"])
full_range = pd.date_range(train["date"].min(), train["date"].max(), freq="D")
missing_days = full_range.difference(train["date"].unique())
print(f"\n[train] days in range: {len(full_range)}, days present: {train['date'].nunique()}")
print(f"[train] missing days: {[d.date() for d in missing_days]}")

# 3) Is (date, store_nbr, family) unique, and is every day fully populated?
dup = train.duplicated(["date", "store_nbr", "family"]).sum()
rows_per_day = train.groupby("date").size()
print(f"[train] duplicate keys: {dup}; rows per day min/max: {rows_per_day.min()}/{rows_per_day.max()}")

# 4) Store opening: first date each store has non-zero total sales
store_daily = train.groupby(["store_nbr", "date"])["sales"].sum()
first_sale = store_daily[store_daily > 0].reset_index().groupby("store_nbr")["date"].min()
late = first_sale[first_sale > train["date"].min()].sort_values()
print(f"\n[train] stores whose first non-zero sale is after 2013-01-01: {len(late)}")
print(late.to_string())

# 5) Transactions coverage vs. full store-day grid
trans = pd.read_csv(RAW / "transactions.csv", parse_dates=["date"])
expected = train["store_nbr"].nunique() * train["date"].nunique()
print(f"\n[transactions] rows: {len(trans):,} vs full store-day grid: {expected:,} (gap {expected - len(trans):,})")

# 6) Oil: missing prices and weekend coverage
oil = pd.read_csv(RAW / "oil.csv", parse_dates=["date"])
print(f"[oil] rows: {len(oil)}, missing price: {oil['dcoilwtico'].isna().sum()}, "
      f"weekend rows: {(oil['date'].dt.dayofweek >= 5).sum()}")