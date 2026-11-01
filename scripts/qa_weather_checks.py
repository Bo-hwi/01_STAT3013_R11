"""QA checks for weather data: city mapping, coordinates, shared grid cells, columns, units and outliers."""
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd

FAV = Path("data/raw/favorita")
WX = Path("data/raw/weather")
pd.set_option("display.width", 200, "display.max_columns", 30)

# ---------- 1) City mapping: stores.csv vs city_coordinates.csv ----------
stores = pd.read_csv(FAV / "stores.csv")
coords = pd.read_csv(WX / "city_coordinates.csv")
store_cities = stores.groupby(["city", "state"]).size().rename("n_stores").reset_index()
merged = store_cities.merge(coords, on="city", how="outer", suffixes=("_stores", "_coords"), indicator=True)
print("=== 1) CITY MAPPING ===")
print(merged[["city", "state_stores", "state_coords", "n_stores", "latitude", "longitude", "status", "results_found", "_merge"]].to_string())
print("Cities only in one file:", merged.loc[merged["_merge"] != "both", "city"].tolist())
print("State mismatch:", merged.loc[(merged["_merge"] == "both") & (merged["state_stores"] != merged["state_coords"]), "city"].tolist())

# ---------- 2) Load all 22 weather files and check identical schema ----------
frames, schemas = [], {}
for path in sorted(WX.glob("weather_*.csv")):
    if path.name == "weather_download_log.csv":
        continue
    df = pd.read_csv(path)
    schemas[path.name] = tuple(df.columns)
    df["source_file"] = path.name
    frames.append(df)
wx = pd.concat(frames, ignore_index=True)
print("\n=== 2) WEATHER SCHEMA ===")
print("Distinct column sets across files:", len(set(schemas.values())))
print("Columns:", list(next(iter(schemas.values()))))
print("\nFirst 2 rows of one file:")
print(frames[0].head(2).to_string())

# Identify the city and date columns by name
city_col = next((c for c in wx.columns if c.lower() in ("city", "city_name")), "source_file")
date_col = next(c for c in wx.columns if c.lower() in ("date", "time"))
wx[date_col] = pd.to_datetime(wx[date_col])

# ---------- 3) Requested vs returned coordinates (if the files carry lat/lon) ----------
lat_cols = [c for c in wx.columns if "latitude" in c.lower()]
lon_cols = [c for c in wx.columns if "longitude" in c.lower()]
print("\n=== 3) COORDINATES IN WEATHER FILES ===")
if lat_cols and lon_cols:
    print(wx.groupby(city_col)[lat_cols + lon_cols].agg(["min", "max"]).to_string())
else:
    print("No latitude/longitude columns in weather files -> grid cell not recorded.")

# Pairs of cities closer than one ERA5 grid step (0.25 degrees)
pairs = []
for (_, a), (_, b) in combinations(coords.iterrows(), 2):
    d = np.hypot(a["latitude"] - b["latitude"], a["longitude"] - b["longitude"])
    if d < 0.25:
        pairs.append((a["city"], b["city"], round(d, 3)))
print("City pairs closer than 0.25 deg:", pairs)

# ---------- 4) Are any two cities' weather series identical (shared grid cell)? ----------
num_cols = [c for c in wx.select_dtypes("number").columns if c not in lat_cols + lon_cols and "elev" not in c.lower()]
probe = next((c for c in num_cols if "temperature_2m_max" in c), num_cols[0])
pivot = wx.pivot_table(index=date_col, columns=city_col, values=probe)
same = [(a, b) for a, b in combinations(pivot.columns, 2) if np.allclose(pivot[a], pivot[b], equal_nan=True)]
print(f"\n=== 4) IDENTICAL SERIES on '{probe}' ===")
print(same if same else "None")

# ---------- 5) Range and plausibility checks per numeric column ----------
print("\n=== 5) DESCRIPTIVE STATS (all cities) ===")
print(wx[num_cols].describe(percentiles=[0.01, 0.5, 0.99]).T.round(2).to_string())

rules = {
    "precipitation": lambda s: s < 0,
    "rain": lambda s: s < 0,
    "humidity": lambda s: (s < 0) | (s > 100),
    "cloud": lambda s: (s < 0) | (s > 100),
    "sunshine_duration": lambda s: (s < 0) | (s > 86400),
    "temperature": lambda s: (s < -10) | (s > 45),
    "wind": lambda s: (s < 0) | (s > 100),
}
print("\n=== 5b) IMPLAUSIBLE VALUE COUNTS ===")
for col in num_cols:
    for key, rule in rules.items():
        if key in col.lower():
            print(f"{col}: {int(rule(wx[col]).sum())} rows outside plausible range")
            break

tmax = next((c for c in num_cols if "temperature_2m_max" in c), None)
tmin = next((c for c in num_cols if "temperature_2m_min" in c), None)
if tmax and tmin:
    print(f"Rows with {tmax} < {tmin}: {int((wx[tmax] < wx[tmin]).sum())}")

# ---------- 6) Wind outliers: top 10 values with city and date ----------
wind = next((c for c in num_cols if "wind" in c.lower()), None)
if wind:
    print(f"\n=== 6) TOP 10 '{wind}' ===")
    print(wx.nlargest(10, wind)[[city_col, date_col, wind]].to_string(index=False))
    per_city = wx.groupby(city_col)[wind].agg(["median", "max"]).sort_values("max", ascending=False)
    print("\nPer-city median / max:")
    print(per_city.round(1).to_string())