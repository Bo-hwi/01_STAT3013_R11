"""Build the readable Decision Matrix summary and its figures from M4's outputs.

Reads (unchanged):  results/tables/decision_matrix.csv, results/tables/cost_scenarios.csv
Writes:             results/tables/decision_matrix_summary.csv
                    reports/figures/decision_matrix_labels.png
                    reports/figures/cost_index_policies.png

Run from the repository root::

    python scripts/m4_evidence_summary.py
"""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.decision.evidence_summary import LABELS, summarise  # noqa: E402

MATRIX = Path("results/tables/decision_matrix.csv")
COSTS = Path("results/tables/cost_scenarios.csv")
SUMMARY = Path("results/tables/decision_matrix_summary.csv")
FIG_LABELS = Path("reports/figures/decision_matrix_labels.png")
FIG_COSTS = Path("reports/figures/cost_index_policies.png")

BLUE, LIGHT, GREY, RED = "#1B4F8F", "#C9D6E8", "#BDBDBD", "#B83A3A"
plt.rcParams.update({"font.family": "serif", "font.size": 10, "axes.spines.top": False,
                     "axes.spines.right": False, "savefig.dpi": 220, "savefig.bbox": "tight"})


def label_map(summary: pd.DataFrame) -> None:
    """Family x city grid coloured by evidence label."""
    families = sorted(summary["family"].unique())
    cities = sorted(summary["city"].unique())
    grid = (summary.assign(code=summary["evidence_label"].map({l: i for i, l in enumerate(LABELS)}))
            .pivot(index="family", columns="city", values="code").loc[families, cities])
    fig, ax = plt.subplots(figsize=(6.3, 2.9))
    ax.imshow(grid.to_numpy(), cmap=ListedColormap([BLUE, GREY, RED]), vmin=0, vmax=2, aspect="auto")
    ax.set_xticks(range(len(cities)))
    ax.set_xticklabels(cities, rotation=60, ha="right", fontsize=7.5)
    ax.set_yticks(range(len(families)))
    ax.set_yticklabels(families, fontsize=8.5)
    ax.set_xticks([x - 0.5 for x in range(1, len(cities))], minor=True)
    ax.set_yticks([y - 0.5 for y in range(1, len(families))], minor=True)
    ax.grid(which="minor", color="white", linewidth=1)
    ax.tick_params(which="minor", length=0)
    counts = summary["evidence_label"].value_counts()
    ax.legend(handles=[Patch(color=c, label=f"{l} ({counts.get(l, 0)})")
                       for c, l in zip([BLUE, GREY, RED], LABELS)],
              loc="upper center", bbox_to_anchor=(0.5, 1.22), ncol=3, frameon=False, fontsize=8.5)
    FIG_LABELS.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG_LABELS)
    plt.close(fig)


def cost_chart(costs: pd.DataFrame) -> None:
    """Pooled cost index of each policy per cost scenario (P1 = 1)."""
    pooled = costs[(costs["report_level"] == "pooled") & (costs["variant"] == "corrected")]
    m0 = pooled[pooled["config"] == "M0"].set_index("scenario")   # P4 from M3's no-weather point model
    scen = ["S1", "S3", "S5"]
    series = [("P1 seasonal naive", [1.0] * 3, GREY),
              ("P2 quantile, no weather", m0.loc[scen, "cost_index_P2"], LIGHT),
              ("P3 quantile, weather", m0.loc[scen, "cost_index_P3"], BLUE),
              ("P4 point model + offset", m0.loc[scen, "cost_index_P4"], "#E39B3B")]
    fig, ax = plt.subplots(figsize=(6.3, 3.0))
    width = 0.2
    for i, (name, vals, col) in enumerate(series):
        xs = [k + (i - 1.5) * width for k in range(3)]
        ax.bar(xs, vals, width, color=col, label=name)
        if i:
            for x, v in zip(xs, vals):
                ax.text(x, v + 0.012, f"{v:.3f}", ha="center", fontsize=7.5)
    ax.set_xticks(range(3))
    ax.set_xticklabels(["1:1 (tau 0.50)", "1:3 (tau 0.75)", "1:5 (tau 0.83)"])
    ax.set_ylabel("Simulated cost index (P1 = 1)")
    ax.set_ylim(0, 1.12)
    ax.legend(ncol=2, frameon=False, fontsize=8, loc="upper center", bbox_to_anchor=(0.5, 1.25))
    fig.savefig(FIG_COSTS)
    plt.close(fig)


def main() -> None:
    summary = summarise(pd.read_csv(MATRIX))
    summary.to_csv(SUMMARY, index=False)
    label_map(summary)
    cost_chart(pd.read_csv(COSTS))
    print(summary["evidence_label"].value_counts().to_string())
    print(summary.groupby("family")["evidence_label"].value_counts().unstack(fill_value=0).to_string())
    print(f"Saved {SUMMARY}, {FIG_LABELS}, {FIG_COSTS}")


if __name__ == "__main__":
    main()
