"""Single entry point for the data pipeline (raw -> curated -> features).

Usage from the repository root::

    python scripts/run_pipeline.py            # build curated table and feature table
    python scripts/run_pipeline.py --test     # build, then run data and feature tests

Outputs (local only, ignored by Git):
    data/processed/curated_sales_weather.parquet  (+ .meta.json)
    data/processed/model_features.parquet         (+ .meta.json)
"""
import argparse
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.data import merge_data  # noqa: E402
from src.features import lag_features  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the Weather x Favorita data pipeline.")
    parser.add_argument("--test", action="store_true", help="run data and feature tests after building")
    args = parser.parse_args()

    print("== Stage 1/2: raw -> curated ==")
    merge_data.main()

    print("\n== Stage 2/2: curated -> model features ==")
    lag_features.main()

    if args.test:
        print("\n== Data and feature tests ==")
        result = subprocess.run(
            [sys.executable, "-m", "pytest", "tests/test_data_integrity.py", "tests/test_lag_features.py",
             "-q", "-p", "no:warnings"],
            cwd=REPO_ROOT,
        )
        sys.exit(result.returncode)


if __name__ == "__main__":
    main()
