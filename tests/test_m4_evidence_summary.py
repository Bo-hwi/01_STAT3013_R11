"""Tests for the evidence-based Decision Matrix summary (src/decision/evidence_summary.py).

Run from the repository root::

    python -m pytest tests/test_m4_evidence_summary.py -v
"""
import json
from pathlib import Path

import pandas as pd
import pytest

from src.decision import evidence_summary as E


def fake_row(intervals: dict, favorable: dict) -> dict:
    """One Decision Matrix row; intervals = {scenario: (point, low, high) or None}."""
    estimates = [{"metric": E.METRIC, "scenario": s, "variant": "corrected", "status": "AVAILABLE",
                  "point_estimate": v[0], "lower": v[1], "upper": v[2]}
                 for s, v in intervals.items() if v is not None]
    folds = {"by_scenario": {s: {"P3_minus_P2": {"favorable": n}} for s, n in favorable.items()}}
    return {"family": "BEVERAGES", "city": "Quito", "sample_size": 224, "weather_signature": "-",
            "confidence_interval": json.dumps({"estimates": estimates}),
            "fold_consistency": json.dumps(folds), "recommendation": "Validate"}


def run(intervals, favorable):
    return E.summarise(pd.DataFrame([fake_row(intervals, favorable)])).iloc[0]


def test_weather_lowering_cost_in_two_scenarios_is_a_pilot():
    out = run({"S1": (-0.5, -0.9, -0.1), "S3": (-0.4, -0.8, -0.05), "S5": (0.1, -0.2, 0.4)},
              {"S1": 4, "S3": 3, "S5": 2})
    assert out["evidence_label"] == "Prioritize pilot"


def test_pilot_needs_fold_consistency():
    out = run({"S1": (-0.5, -0.9, -0.1), "S3": (-0.4, -0.8, -0.05), "S5": (0.1, -0.2, 0.4)},
              {"S1": 2, "S3": 3, "S5": 2})          # only 2 of 4 folds in S1
    assert out["evidence_label"] == "Further validation"


def test_weather_raising_cost_is_do_not_deploy():
    out = run({"S1": (0.0, -0.1, 0.1), "S3": (0.2, -0.1, 0.5), "S5": (0.8, 0.3, 1.3)},
              {"S1": 2, "S3": 1, "S5": 0})
    assert out["evidence_label"] == "Do not deploy"


def test_mixed_evidence_needs_more_validation():
    out = run({"S1": (-0.3, -0.5, -0.1), "S3": (0.0, -0.2, 0.2), "S5": (0.6, 0.2, 1.0)},
              {"S1": 4, "S3": 2, "S5": 0})
    assert out["evidence_label"] == "Further validation"


def test_missing_interval_is_never_a_pilot():
    out = run({"S1": None, "S3": None, "S5": None}, {"S1": 4, "S3": 4, "S5": 4})
    assert out["evidence_label"] == "Further validation"
    assert out["direction_1:1"] == "no_ci"


SUMMARY = Path("results/tables/decision_matrix_summary.csv")


@pytest.mark.skipif(not SUMMARY.exists(), reason="Run `python scripts/m4_evidence_summary.py` first")
def test_saved_summary_is_flat_and_complete():
    s = pd.read_csv(SUMMARY)
    assert len(s) == 154 and not s.duplicated(["family", "city"]).any()
    assert set(s["evidence_label"]) <= set(E.LABELS)
    assert not s.astype(str).apply(lambda c: c.str.startswith("{")).any().any()   # no JSON left
