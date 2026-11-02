"""Run full statistical EDA on 33 product families and export summary tables + figures."""

from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


import seaborn as sns
import pandas as pd
import numpy as np

from src.statistics.eda import (
    compute_family_summary,
    compute_weather_correlations,
    compute_seasonality_stats,
    screen_deep_study_candidates,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = REPO_ROOT / "data" / "processed" / "curated_sales_weather.parquet"
RESULTS_DIR = REPO_ROOT / "results" / "tables"
FIGURES_DIR = REPO_ROOT / "reports" / "figures"


def main():
    print("Loading curated dataset...")
    df = pd.read_parquet(DATA_PATH)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    print("Computing 33-family summary...")
    summary = compute_family_summary(df, open_only=True)
    summary_path = RESULTS_DIR / "eda_33_families_summary.csv"
    summary.to_csv(summary_path)
    print(f"Saved: {summary_path}")

    # Plot 1: Top families by sales share
    print("Generating Figure: eda_family_sales_share.png...")
    plt.figure(figsize=(10, 8))
    top_families = summary["sales_share"].sort_values(ascending=True) * 100
    colors = ["#1f77b4" if s >= 1.0 else "#aec7e8" for s in top_families.values]
    plt.barh(top_families.index, top_families.values, color=colors)
    plt.xlabel("Share of Total Sales (%)")
    plt.title("Corporación Favorita: Sales Share by Product Family (33 Families)")
    plt.axvline(1.0, color="red", linestyle="--", alpha=0.7, label="1% Pre-screening Threshold")
    plt.legend()
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "eda_family_sales_share.png", dpi=200)
    plt.close()

    # Plot 2: Zero-sales rate distribution
    print("Generating Figure: eda_zero_sales_distribution.png...")
    plt.figure(figsize=(10, 8))
    zero_rates = summary["zero_rate"].sort_values(ascending=True) * 100
    bar_colors = ["#2ca02c" if z <= 5.0 else "#d62728" for z in zero_rates.values]
    plt.barh(zero_rates.index, zero_rates.values, color=bar_colors)
    plt.xlabel("Zero Sales Rate on Open Store Days (%)")
    plt.title("Zero-Sales Rate across 33 Product Families")
    plt.axvline(5.0, color="black", linestyle="--", alpha=0.7, label="5% Intermittency Cap")
    plt.legend()
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "eda_zero_sales_distribution.png", dpi=200)
    plt.close()

    # Plot 3: Weather correlation heatmap
    print("Computing weather correlations...")
    weather_vars = [
        "precipitation_sum",
        "temperature_2m_max",
        "temperature_2m_mean",
        "relative_humidity_2m_mean",
        "sunshine_hours",
    ]
    corrs = compute_weather_correlations(df, weather_vars)
    corrs.to_csv(RESULTS_DIR / "eda_weather_correlations.csv")

    print("Generating Figure: eda_weather_correlation_heatmap.png...")
    plt.figure(figsize=(8, 11))
    sns.heatmap(corrs, cmap="coolwarm", center=0, annot=True, fmt=".2f", cbar_kws={"label": "Pearson Correlation (r)"})
    plt.title("Raw Correlation: Daily Sales vs Weather Measures across 33 Families")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "eda_weather_correlation_heatmap.png", dpi=200)
    plt.close()

    # Pre-registered screening check
    print("Screening candidates against frozen 7-family list...")
    screening = screen_deep_study_candidates(df)
    screening.to_csv(RESULTS_DIR / "eda_deep_study_screening.csv")
    print(screening)
    print("\nEDA Execution complete!")


if __name__ == "__main__":
    main()
