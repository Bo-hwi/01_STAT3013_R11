"""Build a catalog of raw source files: size, row count, columns, date range and SHA-256."""
import hashlib
from pathlib import Path

import pandas as pd

RAW_DIR = Path("data/raw")
OUT_PATH = Path("docs/source_catalog.csv")


def sha256_of(path: Path, chunk_size: int = 1 << 20) -> str:
    """Hash the file in chunks so large files (train.csv ~122 MB) do not fill memory."""
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(chunk_size), b""):
            h.update(chunk)
    return h.hexdigest()


def describe_csv(path: Path) -> dict:
    """Read the CSV once and collect basic schema facts."""
    df = pd.read_csv(path, low_memory=False)
    info = {
        "n_rows": len(df),
        "n_cols": df.shape[1],
        "columns": ";".join(df.columns),
        "n_missing_cells": int(df.isna().sum().sum()),
        "date_min": None,
        "date_max": None,
    }
    # Weather files may name the date column 'time' instead of 'date'
    date_col = next((c for c in ("date", "time") if c in df.columns), None)
    if date_col:
        dates = pd.to_datetime(df[date_col], errors="coerce")
        info["date_min"] = dates.min().date() if dates.notna().any() else None
        info["date_max"] = dates.max().date() if dates.notna().any() else None
    return info


def main() -> None:
    rows = []
    for path in sorted(RAW_DIR.rglob("*.csv")):
        print(f"Processing {path} ...")
        rows.append({
            "source_group": path.parent.name,      # favorita or weather
            "file": path.name,
            "size_bytes": path.stat().st_size,
            "sha256": sha256_of(path),
            **describe_csv(path),
        })
    catalog = pd.DataFrame(rows)
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    catalog.to_csv(OUT_PATH, index=False)
    print(f"\nSaved {len(catalog)} files to {OUT_PATH}")
    print(catalog[["file", "n_rows", "n_cols", "date_min", "date_max", "n_missing_cells"]].to_string())


if __name__ == "__main__":
    main()