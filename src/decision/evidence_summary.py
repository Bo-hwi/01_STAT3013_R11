"""Readable Decision Matrix summary with evidence-based labels (Proposal section 10).

Why this file?
    results/tables/decision_matrix.csv (built by final_handoff.py) keeps every
    detail as JSON inside the cells and labels all 154 cells "Validate".
    This module reads that table, without changing it, and produces:
      * one flat row per family x city (no JSON), and
      * a label taken from the evidence, following Proposal section 10.

Evidence used per cell and cost scenario (S1 = 1:1, S3 = 1:3, S5 = 1:5)
    Weather value = cost of P3 (quantile WITH weather) minus cost of P2
    (quantile WITHOUT weather), in % of P2's cost, with M4's 95%
    fixed-week block-bootstrap interval.
        upper bound < 0  -> weather LOWERS cost  ("lower")
        lower bound > 0  -> weather RAISES cost  ("raise")
        interval has 0   -> no reliable difference ("unclear")
    Fold consistency = number of the 4 test folds in which weather lowered cost.

Label rule (written from the Proposal wording, applied mechanically)
    Prioritize pilot  : weather lowers cost in >= 2 of 3 scenarios, raises it in
                        none, and lowers it in >= 3 of 4 folds in those scenarios.
    Do not deploy     : weather raises cost in >= 1 scenario and lowers it in none.
    Further validation: everything else (unclear, mixed, or no interval available).

Caveats kept with every label: intervals are pointwise (no multiple-testing
control over 140 cells x 3 scenarios), ERA5 is retrospective, and costs are
simulated ratios, not money.
"""
import json

import pandas as pd

SCENARIOS = {"S1": "1:1", "S3": "1:3", "S5": "1:5"}
METRIC = "P3_minus_P2:relative_difference_pct"
LABELS = ["Prioritize pilot", "Further validation", "Do not deploy"]


def weather_effect(confidence_interval: str, scenario: str) -> dict:
    """Point estimate and 95% CI (percent) of P3 - P2 for one scenario, or None values."""
    ci = json.loads(confidence_interval)
    for e in ci.get("estimates", []):
        if (e.get("metric") == METRIC and e.get("scenario") == scenario
                and e.get("variant") == "corrected" and e.get("status") == "AVAILABLE"):
            return {"pct": e["point_estimate"], "low": e["lower"], "high": e["upper"]}
    return {"pct": None, "low": None, "high": None}


def direction(effect: dict) -> str:
    if effect["low"] is None:
        return "no_ci"
    if effect["high"] < 0:
        return "lower"
    if effect["low"] > 0:
        return "raise"
    return "unclear"


def favorable_folds(fold_consistency: str, scenario: str) -> int:
    """Folds (out of 4) where P3 cost < P2 cost."""
    by = json.loads(fold_consistency).get("by_scenario", {})
    return int(by.get(scenario, {}).get("P3_minus_P2", {}).get("favorable", 0))


def label(directions: dict, folds: dict) -> str:
    lowers = [s for s, d in directions.items() if d == "lower"]
    raises = [s for s, d in directions.items() if d == "raise"]
    if len(lowers) >= 2 and not raises and all(folds[s] >= 3 for s in lowers):
        return "Prioritize pilot"
    if raises and not lowers:
        return "Do not deploy"
    return "Further validation"


def summarise(matrix: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, r in matrix.iterrows():
        row = {"family": r["family"], "city": r["city"], "sample_size": r["sample_size"],
               "weather_signature_M2": r["weather_signature"]}
        directions, folds = {}, {}
        for s, ratio in SCENARIOS.items():
            eff = weather_effect(r["confidence_interval"], s)
            directions[s], folds[s] = direction(eff), favorable_folds(r["fold_consistency"], s)
            row[f"weather_cost_change_pct_{ratio}"] = eff["pct"]
            row[f"ci_low_{ratio}"] = eff["low"]
            row[f"ci_high_{ratio}"] = eff["high"]
            row[f"direction_{ratio}"] = directions[s]
            row[f"folds_weather_cheaper_{ratio}"] = folds[s]
        row["evidence_label"] = label(directions, folds)
        row["original_label_M4"] = r["recommendation"]
        row["caveats"] = ("pointwise 95% CI, no multiple-testing control; retrospective ERA5; "
                          "operational availability unverified; simulated cost ratios, not money")
        rows.append(row)
    return pd.DataFrame(rows)
