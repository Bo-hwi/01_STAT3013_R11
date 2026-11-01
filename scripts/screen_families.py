"""Screen the 33 product families and apply the pre-registered deep-study shortlist rule.

The rule comes from Proposal section 2 (written before any model results):
low zero-sales rate, long enough active history, representative scale,
coverage of all cities and sector diversity. Only data BEFORE the first fold's
validation block (2016-08-04) is used, so the choice cannot depend on any
validation or test period.

Run from the repository root (needs the curated table)::

    python scripts/screen_families.py
"""
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
CURATED = REPO_ROOT / "data" / "processed" / "curated_sales_weather.parquet"
OUT = REPO_ROOT / "results" / "tables" / "family_screening.csv"

SCREEN_END = "2016-08-04"  # first validation day of fold F1 (exclusive)
CANDIDATES = [  # Proposal section 2, listed before seeing any results
    "BEVERAGES", "BREAD/BAKERY", "PRODUCE", "DAIRY",
    "FROZEN FOODS", "CLEANING", "PERSONAL CARE", "GROCERY I",
]
RULES = {  # thresholds of the pre-registered criteria
    "max_zero_rate": 0.05,            # share of active store-days with sales == 0
    "max_first_nonzero": "2013-01-31",  # full history for 28-day lags from the start
    "min_cities": 22,                 # sold in every city (needed for city x family results)
    "min_share_total_sales": 0.01,    # representative scale
}
SECTOR = {  # used to check diversity of the final list
    "GROCERY I": "dry staples", "BEVERAGES": "drinks", "PRODUCE": "fresh produce",
    "DAIRY": "fresh perishables", "BREAD/BAKERY": "fresh perishables",
    "FROZEN FOODS": "frozen", "CLEANING": "household non-food", "PERSONAL CARE": "personal non-food",
}


def main() -> None:
    df = pd.read_parquet(CURATED, columns=["date", "family", "city", "sales", "store_active", "store_zero_day"])
    # Screening window, open store-days only (closures are not demand, QA-02/QA-11)
    d = df[(df["date"] < SCREEN_END) & (df["store_active"] == 1) & (df["store_zero_day"] == 0)].copy()
    d["family"] = d["family"].astype(str)

    g = d.groupby("family")
    sold = d[d["sales"] > 0].groupby("family")
    table = pd.DataFrame({
        "zero_rate": g["sales"].apply(lambda s: (s == 0).mean()),
        "mean_daily_sales": g["sales"].mean(),
        "share_total_sales": g["sales"].sum() / d["sales"].sum(),
        "first_nonzero_date": sold["date"].min(),
        "cities_with_sales": sold["city"].nunique(),
    }).fillna({"cities_with_sales": 0})

    table["candidate"] = table.index.isin(CANDIDATES)
    table["pass_zero_rate"] = table["zero_rate"] <= RULES["max_zero_rate"]
    table["pass_history"] = table["first_nonzero_date"] <= pd.Timestamp(RULES["max_first_nonzero"])
    table["pass_cities"] = table["cities_with_sales"] >= RULES["min_cities"]
    table["pass_scale"] = table["share_total_sales"] >= RULES["min_share_total_sales"]
    passes = ["pass_zero_rate", "pass_history", "pass_cities", "pass_scale"]
    table["selected"] = table["candidate"] & table[passes].all(axis=1)
    table["sector"] = table.index.map(SECTOR)

    table = table.sort_values("share_total_sales", ascending=False)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    table.round(4).to_csv(OUT, index_label="family")

    shortlist = table.index[table["selected"]].tolist()
    print(table.loc[table["candidate"], ["zero_rate", "share_total_sales", "first_nonzero_date",
                                         "cities_with_sales", *passes, "selected"]].round(4).to_string())
    print(f"\nShortlist ({len(shortlist)}): {shortlist}")
    print(f"Sectors: {sorted(set(table.loc[shortlist, 'sector']))}")
    print(f"Saved {OUT.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()
