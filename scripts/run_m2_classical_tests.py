"""Run M2's classical tests (Welch t-test, ANOVA, chi-square) and save results.

Run from the repository root::

    python scripts/run_m2_classical_tests.py

Outputs:
    results/tables/rq1_classical_tests.csv
    reports/figures/rq1_rain_boxplot.png
"""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.statistics.classical_tests import FAMILIES, city_day_table, run_all  # noqa: E402

CURATED = Path("data/processed/curated_sales_weather.parquet")
TABLE_OUT = Path("results/tables/rq1_classical_tests.csv")
FIGURE_OUT = Path("reports/figures/rq1_rain_boxplot.png")


def rain_boxplot(table: pd.DataFrame) -> None:
    """Boxplot of city-day log-sales on dry vs rainy days, one pair per family."""
    fig, ax = plt.subplots(figsize=(11, 4.5))
    data, labels, positions = [], [], []
    for i, family in enumerate(FAMILIES):
        rows = table[table["family"] == family]
        data += [rows.loc[~rows["rainy"], "y"], rows.loc[rows["rainy"], "y"]]
        positions += [3 * i, 3 * i + 1]
        labels.append(family)
    box = ax.boxplot(data, positions=positions, widths=0.8, showfliers=False, patch_artist=True)
    for j, patch in enumerate(box["boxes"]):
        patch.set_facecolor("#cfd8e3" if j % 2 == 0 else "#4a78b5")
    ax.set_xticks([3 * i + 0.5 for i in range(len(FAMILIES))])
    ax.set_xticklabels(labels, rotation=20)
    ax.set_ylabel("Mean log(1 + sales) per city-day")
    ax.set_title("Raw sales on dry (light) vs rainy (dark, >= 1 mm) days, before any controls")
    fig.tight_layout()
    FIGURE_OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURE_OUT, dpi=150)
    plt.close(fig)


def main() -> None:
    curated = pd.read_parquet(CURATED, columns=[
        "date", "store_nbr", "family", "city", "sales", "store_active", "store_zero_day",
        "precipitation_sum", "temperature_2m_max"])
    table = city_day_table(curated)
    print(f"City-day table: {len(table):,} rows "
          f"({table['city'].nunique()} cities x {table['family'].nunique()} families)")

    results = run_all(table)
    TABLE_OUT.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(TABLE_OUT, index=False)
    rain_boxplot(table)

    show = results[["family", "test", "difference_pct", "effect_name", "effect_size", "p_value", "q_fdr"]]
    print(show.round(4).to_string(index=False))
    print(f"\nSaved {TABLE_OUT} and {FIGURE_OUT}")


if __name__ == "__main__":
    main()
