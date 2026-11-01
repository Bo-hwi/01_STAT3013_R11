"""Synthetic checks; no real-data claims or parquet dependency."""

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import pandas as pd

from src.forecasting.evaluation import mase_scales, metrics
from src.forecasting.feature_spec import FEATURES
from src.forecasting.point_models import OUTPUT, build_pipeline, prepare_features, run_m0, tune
from src.forecasting.rolling_validation import (
    add_scoring_mask, calendar_rolling_mean, calendar_sales_lag, iter_folds, read_yaml, validate_table,
)

ROOT = Path(__file__).resolve().parents[1]
LAGS = FEATURES["sales_lags"]


def table(start="2016-08-01", end="2017-08-15", stores=(18, 25, 43), families=("A",)):
    dates = pd.date_range(start, end)
    dates = dates[~((dates.month == 12) & (dates.day == 25))]
    result = pd.MultiIndex.from_product([dates, stores, families], names=["date", "store_nbr", "family"]).to_frame(index=False)
    result["city"] = "City"
    result["store_active"], result["store_zero_day"] = 1, 0
    result["sales"] = (result["date"] - pd.Timestamp("2013-01-01")).dt.days.astype(float) + result["store_nbr"]
    # Synthetic calendar flags are fixtures, not inferred real holidays.
    for column in FEATURES["calendar"]:
        result[column] = 0
    result["day_of_week"] = result["date"].dt.dayofweek
    result["month"] = result["date"].dt.month
    return result


class ForecastingChecks(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.protocol = read_yaml(ROOT / "configs/research_protocol.yaml")
        cls.ablation = read_yaml(ROOT / "configs/ablation_M0_M5.yaml")
        cls.rules = cls.ablation["scoring_exclusions"]

    def test_duplicate_yaml_keys_fail(self):
        with TemporaryDirectory() as directory:
            path = Path(directory) / "duplicate.yaml"
            path.write_text("rq2:\n  validation:\n    scheme: first\n    scheme: second\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "Duplicate YAML key 'scheme'"):
                read_yaml(path)

    def test_calendar_lags_preserve_missing_christmas(self):
        data = validate_table(table("2016-12-23", "2016-12-27"))
        lag = calendar_sales_lag(data, 1)
        self.assertTrue(np.isnan(lag[data["date"].eq("2016-12-26")]).all())
        self.assertTrue(np.isfinite(lag[data["date"].eq("2016-12-27")]).all())

    def test_closure_counts_match_m1_scenario(self):
        families = [f"F{i}" for i in range(33)]
        data = table(families=families)
        for store, start, end in [(18, "2016-08-15", "2016-12-02"), (25, "2016-08-22", "2016-10-26"), (43, "2016-11-14", "2016-11-17")]:
            closed = data["store_nbr"].eq(store) & data["date"].between(start, end)
            data.loc[closed, ["store_zero_day", "sales"]] = [1, 0]
        folds = list(iter_folds(data, self.protocol, self.ablation, families))
        actual = pd.concat([f.counts for f in folds]).set_index(["fold_id", "split"])
        expected = {
            ("F1", "validation"): (891, 0), ("F1", "test"): (1848, 0),
            ("F2", "validation"): (1254, 627), ("F2", "test"): (693, 627),
            ("F3", "validation"): (0, 0), ("F3", "test"): (0, 0),
            ("F4", "validation"): (0, 0), ("F4", "test"): (0, 0),
        }
        for key, counts in expected.items():
            row = actual.loc[key]
            self.assertEqual((row.rows_excluded_closure, row.rows_excluded_post_reopening_only), counts)
            self.assertEqual(row.rows_before_exclusions, row.rows_scored + row.rows_excluded_total_unique)
        for fold in folds:
            self.assertTrue(fold.train["store_zero_day"].eq(0).all())
            self.assertLess(fold.train.date.max(), fold.validation.date.min())
            self.assertLess(fold.validation.date.max(), fold.test.date.min())
            self.assertTrue(fold.refit_rows()["store_zero_day"].eq(0).all())
        # Cooldown rows are excluded from scoring, but retained for refitting.
        self.assertTrue(folds[1].refit_rows()["excluded_post_reopening"].any())

    def test_reopening_window_boundary_and_short_closure(self):
        data = table("2017-01-01", "2017-03-01", stores=(1,))
        data.loc[data.date.between("2017-01-03", "2017-01-09"), "store_zero_day"] = 1
        data = add_scoring_mask(validate_table(data), self.rules).set_index("date")
        self.assertFalse(data.loc["2017-01-10", "score_mask"])
        self.assertFalse(data.loc["2017-02-06", "score_mask"])
        self.assertTrue(data.loc["2017-02-07", "score_mask"])
        short = table("2017-01-01", "2017-02-01", stores=(1,))
        short.loc[short.date.between("2017-01-03", "2017-01-08"), "store_zero_day"] = 1
        short = add_scoring_mask(validate_table(short), self.rules).set_index("date")
        self.assertTrue(short.loc["2017-01-09", "score_mask"])

    def test_shared_mask_mismatch_fails(self):
        data = table()
        data["score_mask"] = False
        with self.assertRaisesRegex(ValueError, "disagrees"):
            add_scoring_mask(validate_table(data), self.rules)

    def test_store_flags_must_agree_across_families(self):
        data = table(families=("A", "B"))
        data.loc[0, "store_zero_day"] = 1
        with self.assertRaisesRegex(ValueError, "disagree"):
            add_scoring_mask(validate_table(data), self.rules)

    def test_ready_features_match_temporary_path(self):
        data = validate_table(table())
        temporary = prepare_features(data, ready=False)
        ready = data.copy()
        for lag, name in LAGS.items():
            ready[name] = calendar_sales_lag(data, lag)
        for days, name in FEATURES["sales_rolling"].items():
            ready[name] = calendar_rolling_mean(data, days)
        ready["score_mask"] = True
        ready = prepare_features(ready, ready=True)
        pd.testing.assert_frame_equal(temporary[list(LAGS.values())], ready[list(LAGS.values())])
        broken = data.copy()
        for lag, name in LAGS.items():
            broken[name] = calendar_sales_lag(data, lag)
        broken["score_mask"] = True
        broken[LAGS[7]] = broken[LAGS[7]].fillna(0)
        with self.assertRaisesRegex(ValueError, "audit failed"):
            prepare_features(broken, ready=True)

    def test_preprocessing_learns_training_only(self):
        data = prepare_features(table(), ready=False)
        fold = next(iter_folds(data, self.protocol, self.ablation, ["A"]))
        numeric = [LAGS[1]]
        pipeline = build_pipeline("Ridge", numeric, {"alpha": 1.0}, seed=42)
        pipeline.fit(fold.train[numeric + ["store_nbr", "family", "city"]], fold.train.sales)
        imputer = pipeline.named_steps["preprocessing"].named_transformers_["numeric"].named_steps["imputer"]
        self.assertAlmostEqual(imputer.statistics_[0], fold.train[LAGS[1]].median())
        validation = fold.validation.copy()
        validation[LAGS[1]] = 1e9
        pipeline.predict(validation[numeric + ["store_nbr", "family", "city"]])
        self.assertAlmostEqual(imputer.statistics_[0], fold.train[LAGS[1]].median())

    def test_mase_scales_ignore_future_targets(self):
        data = validate_table(table())
        train = data.loc[data.date.le("2016-10-17")]
        original = mase_scales(train, data)
        data.loc[data.date.gt("2016-10-17"), "sales"] = 1e12
        pd.testing.assert_frame_equal(original, mase_scales(train, data))
        np.testing.assert_allclose(original.mase_scale, 7.0)

    def test_metrics_use_individual_series_scales(self):
        predictions = pd.DataFrame({"store_nbr": [1, 2], "family": ["A", "A"], "y_true": [10.0, 10.0], "y_pred": [6.0, 6.0]})
        scales = pd.DataFrame({"store_nbr": [1, 2], "family": ["A", "A"], "mase_scale": [2.0, 8.0]})
        result = metrics(predictions, scales)
        self.assertEqual(result["MAE"], 4.0)
        self.assertEqual(result["WAPE"], 0.4)
        self.assertEqual(result["MASE"], 1.25)
        predictions["y_true"] = 0.0
        scales["mase_scale"] = 0.0
        result = metrics(predictions, scales)
        self.assertTrue(np.isnan(result["WAPE"]))
        self.assertTrue(np.isnan(result["MASE"]))
        self.assertEqual(result["mase_coverage"], 0.0)

    def test_small_end_to_end_and_test_targets_do_not_select_parameters(self):
        data = prepare_features(table(stores=(1, 2)), ready=False)
        grids = {"Ridge": [{"alpha": 1.0}, {"alpha": 10.0}], "ExtraTrees": [{"n_estimators": 3, "min_samples_leaf": 2}]}
        predictions, report, diagnostics = run_m0(data, self.protocol, self.ablation, ["A"], grids)
        self.assertEqual(list(predictions), OUTPUT)
        self.assertEqual(len(predictions), 4 * 28 * 2 * 2)
        self.assertTrue(np.isfinite(predictions.y_pred).all())
        self.assertEqual(set(report.fold), {"F1", "F2", "F3", "F4"})
        changed = data.copy()
        changed.loc[changed.date.ge("2017-07-19"), "sales"] = 1e9
        other, _, other_diagnostics = run_m0(changed, self.protocol, self.ablation, ["A"], grids)
        pd.testing.assert_frame_equal(diagnostics["tuning"], other_diagnostics["tuning"])
        np.testing.assert_allclose(predictions.y_pred, other.y_pred)

    def test_rolling_uses_past_calendar_window_only(self):
        data = validate_table(table("2016-12-19", "2016-12-28", stores=(1,)))
        data["sales"] = data["date"].dt.day.astype(float)
        rolling = calendar_rolling_mean(data, 7)
        index = data.index[data.date.eq("2016-12-26")][0]
        # Dec 19-24 are observed; Dec 25 is absent, Dec 26 is excluded.
        self.assertEqual(rolling.loc[index], np.mean([19, 20, 21, 22, 23, 24]))
        data.loc[data.date.ge("2016-12-26"), "sales"] = 1e9
        self.assertEqual(calendar_rolling_mean(data, 7).loc[index], rolling.loc[index])


if __name__ == "__main__":
    unittest.main()
