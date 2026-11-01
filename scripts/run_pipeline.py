"""Single entry point for the data pipeline (raw -> curated).

Usage from the repository root::

    python scripts/run_pipeline.py            # build curated table
    python scripts/run_pipeline.py --test     # build, then run integrity tests

Later stages (features, baselines) will be added here as they are implemented.
"""
import argparse
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.data import merge_data  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the Weather x Favorita data pipeline.")
    parser.add_argument("--test", action="store_true", help="run data integrity tests after building")
    args = parser.parse_args()

    print("== Stage 1/1: raw -> curated ==")
    merge_data.main()

    if args.test:
        print("\n== Integrity tests ==")
        result = subprocess.run(
            [sys.executable, "-m", "pytest", "tests/test_data_integrity.py", "-q"], cwd=REPO_ROOT
        )
        sys.exit(result.returncode)


if __name__ == "__main__":
    main()
