"""Tests for M3's ML point forecasts (notebooks/03_forecasting_ml/train_ml.ipynb).

Checks the saved prediction file, so M4 and the report can trust it:
1. Format: same columns as baselines.csv and sarimax.csv.
2. Completeness: 3 models x 5 weather configs, every spec scores the same rows.
3. Fairness: the scored rows and y_true are identical to the M1 baselines,
   so MAE can be compared directly across all models.

Run from the repository root (after running the notebook)::

    python -m pytest tests/test_point_models.py -v
"""
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

PRED = Path("results/predictions/point_models.csv.gz")
BASELINES = Path("results/predictions/baselines.csv")
OUTPUT_COLUMNS = ["date", "store_nbr", "family", "city", "fold", "model", "config", "y_true", "y_pred"]
KEY = ["fold", "date", "store_nbr", "family"]

pytestmark = pytest.mark.skipif(not PRED.exists(), reason="Run notebooks/03_forecasting_ml/train_ml.ipynb first")


@pytest.fixture(scope="module")
def pred() -> pd.DataFrame:
    return pd.read_csv(PRED, parse_dates=["date"])


def test_shared_format(pred):
    assert list(pred.columns) == OUTPUT_COLUMNS
    assert set(pred["model"]) == {"Ridge", "ExtraTrees", "HistGradientBoosting"}
    assert set(pred["config"]) == {"M0", "M1", "M2", "M3", "M4"}
    assert set(pred["fold"]) == {"F1", "F2", "F3", "F4"}


def test_predictions_are_valid(pred):
    assert pred["y_pred"].notna().all()
    assert np.isfinite(pred["y_pred"]).all() and (pred["y_pred"] >= 0).all()
    assert not pred.duplicated(KEY + ["model", "config"]).any()


def test_every_spec_scores_the_same_rows(pred):
    # 15 specs (3 models x 5 configs) per scored row -> paired M0 vs Mk comparison is fair
    assert pred.groupby(KEY).size().eq(15).all()


def test_test_rows_inside_frozen_blocks(pred):
    blocks = {"F1": ("2016-09-01", "2016-09-28"), "F2": ("2016-11-15", "2016-12-12"),
              "F3": ("2017-03-01", "2017-03-28"), "F4": ("2017-07-19", "2017-08-15")}
    for fold, (start, end) in blocks.items():
        dates = pred.loc[pred["fold"] == fold, "date"]
        assert dates.min() >= pd.Timestamp(start) and dates.max() <= pd.Timestamp(end)


@pytest.mark.skipif(not BASELINES.exists(), reason="Run `python -m src.forecasting.baselines` first")
def test_same_rows_and_truth_as_baselines(pred):
    base = pd.read_csv(BASELINES, parse_dates=["date"])
    base = base[base["model"] == "SeasonalNaive"].set_index(KEY).sort_index()
    ml = pred[(pred["model"] == "Ridge") & (pred["config"] == "M0")].set_index(KEY).sort_index()
    assert ml.index.equals(base.index)                    # identical scored rows
    assert np.allclose(ml["y_true"], base["y_true"])      # identical ground truth
