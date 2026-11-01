"""Check real curated splits by default; use --synthetic for the reference fixture."""

import argparse
from pathlib import Path
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.forecasting.rolling_validation import iter_folds, read_yaml  # noqa: E402


def synthetic_data():
    dates = pd.date_range("2013-01-01", "2017-08-15")
    dates = dates[~((dates.month == 12) & (dates.day == 25))]
    families = [f"SYNTHETIC_{i}" for i in range(33)]
    data = pd.MultiIndex.from_product(
        [dates, [18, 25, 43], families], names=["date", "store_nbr", "family"]
    ).to_frame(index=False)
    data["city"], data["sales"] = "SYNTHETIC_CITY", 1.0
    data["store_active"], data["store_zero_day"] = 1, 0
    for store, start, end in [
        (18, "2016-08-15", "2016-12-02"),
        (25, "2016-08-22", "2016-10-26"),
        (43, "2016-11-14", "2016-11-17"),
    ]:
        closed = data.store_nbr.eq(store) & data.date.between(start, end)
        data.loc[closed, ["store_zero_day", "sales"]] = [1, 0.0]
    return data, families


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--synthetic", action="store_true")
    parser.add_argument("--data", type=Path, default=ROOT / "data/processed/curated_sales_weather.parquet")
    parser.add_argument("--families", nargs="+")
    args = parser.parse_args()
    if args.synthetic:
        data, families = synthetic_data()
        print("SYNTHETIC ONLY: three stores, 33 artificial families.")
    else:
        data = pd.read_parquet(args.data, columns=["date", "store_nbr", "family", "city", "sales", "store_active", "store_zero_day"])
        # Full-family QA is not a deep-study family selection decision.
        families = args.families or data.family.unique().tolist()
        print(f"REAL DATA: {len(data):,} source rows; {len(families)} families checked.")
    folds = iter_folds(
        data, read_yaml(ROOT / "configs/research_protocol.yaml"),
        read_yaml(ROOT / "configs/ablation_M0_M5.yaml"), families,
    )
    rows, count_frames = [], []
    for fold in folds:
        count_frames.append(fold.counts)
        for split in ["train", "validation", "test"]:
            frame = getattr(fold, split)
            rows.append({"fold": fold.fold_id, "split": split, "rows": len(frame), "start": frame.date.min(), "end": frame.date.max(), "training_rows_excluded": fold.training_rows_excluded if split == "train" else 0})
        if not args.synthetic and len(families) == 33 and fold.fold_id == "F4" and fold.training_rows_excluded != 19833:
            raise AssertionError(f"F4 training closure count differs from M1: {fold.training_rows_excluded}")
    print(pd.DataFrame(rows).to_string(index=False))
    counts = pd.concat(count_frames, ignore_index=True)
    print(counts.to_string(index=False))
    expected = [(891, 0), (1848, 0), (1254, 627), (693, 627), (0, 0), (0, 0), (0, 0), (0, 0)]
    actual = list(zip(counts.rows_excluded_closure, counts.rows_excluded_post_reopening_only))
    if (args.synthetic or len(families) == 33) and actual != expected:
        raise AssertionError(f"Closure counts differ from the reference scenario: {actual}")
    print("PASS: date order, 28-day blocks, non-overlap, and scoring exclusions.")


if __name__ == "__main__":
    main()
