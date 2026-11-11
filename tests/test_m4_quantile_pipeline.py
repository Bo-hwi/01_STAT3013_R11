"""Synthetic integration checks only; scores here are NOT research results."""

from copy import deepcopy
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from threadpoolctl import threadpool_limits

from src.decision import quantile_pipeline as qp
from src.decision.quantile_models import canonical_tau_key, pinball_loss
from src.forecasting.feature_spec import FEATURES, m0_numeric
from src.forecasting.point_models import prepare_features
from src.forecasting.rolling_validation import add_scoring_mask, iter_folds, read_yaml


ROOT = Path(__file__).resolve().parents[1]
GRID = [dict(max_iter=2, max_leaf_nodes=3, min_samples_leaf=2),
        dict(max_iter=2, max_leaf_nodes=5, min_samples_leaf=2)]


@pytest.fixture(autouse=True)
def small_thread_budget():
    with threadpool_limits(limits=1):
        yield


@pytest.fixture(scope="module")
def contracts():
    protocol = read_yaml(ROOT / "configs/research_protocol.yaml")
    ablation = read_yaml(ROOT / "configs/ablation_M0_M5.yaml")
    # Use the real first shared fold on a small synthetic panel. No new splitter.
    protocol["rq2"]["validation"]["test_folds"] = protocol["rq2"]["validation"]["test_folds"][:1]
    return protocol, ablation


@pytest.fixture(scope="module")
def features(contracts):
    protocol, ablation = contracts
    families = protocol["scope"]["deep_study_families"]
    data = pd.MultiIndex.from_product(
        [pd.date_range("2016-06-01", "2016-09-28"), [1, 2], families],
        names=["date", "store_nbr", "family"],
    ).to_frame(index=False)
    data["city"] = np.where(data.store_nbr.eq(1), "SYNTHETIC_A", "SYNTHETIC_B")
    data["store_active"], data["store_zero_day"] = 1, 0
    data["sales"] = 10.5 + data.date.dt.day + 2 * data.store_nbr + data.family.map(dict(zip(families, range(7))))
    for column in FEATURES["calendar"]:
        data[column] = 0
    data["day_of_week"], data["month"] = data.date.dt.dayofweek, data.date.dt.month
    closed = data.store_nbr.eq(1) & data.date.between("2016-07-28", "2016-08-06")
    data.loc[closed, ["store_zero_day", "sales"]] = [1, 0]
    data = prepare_features(data, ready=False)
    previous_day = (data.date - pd.Timedelta(days=1)).dt.day.to_numpy()
    data["precipitation_sum_lag1"] = (previous_day % 5).astype(float)
    data["temperature_2m_max_lag1"] = 20.0 + previous_day / 10
    # Missing weather must NOT remove rows from M3 scoring.
    data.loc[data.date.eq("2016-09-10"), "precipitation_sum_lag1"] = np.nan
    # Decoys must never become predictors merely because they are numeric.
    data["onpromotion"], data["transactions"], data["precipitation_sum"] = 999999, 999999, 999999
    data = add_scoring_mask(data, ablation["scoring_exclusions"])
    return data.drop(columns=["excluded_closure", "excluded_post_reopening"])


@pytest.fixture(scope="module")
def synthetic_run(features, contracts):
    with threadpool_limits(limits=1):
        return qp.run_direct_quantiles(features, *contracts, "synthetic_test_only", GRID)


def test_feature_whitelists_and_estimator_configuration(contracts):
    _, ablation = contracts
    for config in qp.CONFIGS:
        numeric, categorical = qp.feature_columns(config, ablation)
        assert numeric == m0_numeric() + ([] if config == "M0" else FEATURES["weather"][:2])
        assert categorical == FEATURES["categorical"]
        assert not set(numeric + categorical) & {"sales", "onpromotion", "transactions", "score_mask", "store_zero_day", "precipitation_sum"}
        estimator = qp.build_quantile_pipeline(numeric, 0.8333333333, GRID[0], 42).named_steps["model"]
        assert estimator.loss == "quantile"
        assert estimator.quantile == 5 / 6
        assert estimator.early_stopping is False
        assert estimator.categorical_features == [False] * len(numeric) + [True] * 3


def test_preprocessing_is_train_only_and_accepts_unseen_categories(features, contracts):
    fold = next(iter_folds(features, *contracts, contracts[0]["scope"]["deep_study_families"]))
    numeric, categorical = qp.feature_columns("M3", contracts[1])
    validation = fold.validation.copy()
    validation.loc[:, numeric] = 1e9
    validation["store_nbr"], validation["family"], validation["city"] = "NEW", "NEW", "NEW"
    fitted, _, _ = qp.select_quantile_model(fold.train, validation, numeric, 0.75, GRID, 42)
    transformer = fitted.named_steps["preprocessing"]
    expected = fold.train[numeric].median().fillna(0).to_numpy()
    np.testing.assert_allclose(transformer.named_transformers_["numeric"].statistics_, expected)
    encoder = transformer.named_transformers_["categorical"]
    assert all("NEW" not in categories for categories in encoder.categories_)
    assert np.isfinite(fitted.predict(validation[numeric + categorical])).all()
    np.testing.assert_allclose(transformer.named_transformers_["numeric"].statistics_, expected)


def test_fitting_rows_selection_order_and_fresh_refit(monkeypatch, features, contracts):
    recorded = []
    original = qp.build_quantile_pipeline
    fold = next(iter_folds(features, *contracts, contracts[0]["scope"]["deep_study_families"]))

    def recording_builder(*args):
        model = original(*args)
        fit = model.fit

        def record_fit(X, y):
            recorded.append((model, X.copy(), y.copy()))
            return fit(X, y)

        model.fit = record_fit
        return model

    monkeypatch.setattr(qp, "build_quantile_pipeline", recording_builder)
    qp.run_direct_quantiles(features, *contracts, "synthetic_fit_audit", GRID)
    n_tuning = 2 * 3 * len(GRID)
    assert len(recorded) == n_tuning + 6
    assert len(set(id(item[0]) for item in recorded)) == len(recorded)
    for _, X, y in recorded[:n_tuning]:
        pd.testing.assert_frame_equal(X, fold.train[X.columns])
        pd.testing.assert_series_equal(y, fold.train.sales)
    refit = fold.refit_rows()
    assert refit.store_zero_day.eq(0).all()
    assert refit.date.max() < fold.test.date.min()
    for _, X, y in recorded[n_tuning:]:
        pd.testing.assert_frame_equal(X, refit[X.columns])
        pd.testing.assert_series_equal(y, refit.sales)


def test_prediction_schema_masks_keys_and_equal_budget(synthetic_run, features, contracts):
    predictions, metrics, diagnostics = synthetic_run
    assert list(predictions) == qp.PREDICTION_COLUMNS
    assert not predictions.duplicated(qp.PREDICTION_KEY).any()
    assert set(predictions.family) == set(contracts[0]["scope"]["deep_study_families"])
    assert set(predictions.tau_key) == {canonical_tau_key(t) for t in qp.QUANTILE_LEVELS}
    assert set(predictions.fit_stage) == {"train_only_selected", "train_plus_validation_refit"}
    assert predictions.forecast_origin.eq(predictions.date - pd.Timedelta(days=1)).all()
    fold = next(iter_folds(features, *contracts, contracts[0]["scope"]["deep_study_families"]))
    for split in ("validation", "test"):
        frame = getattr(fold, split)
        expected = frame.loc[frame.score_mask, qp.KEY].sort_values(qp.KEY).reset_index(drop=True)
        for config in qp.CONFIGS:
            for tau in qp.QUANTILE_LEVELS:
                subset = predictions.loc[(predictions.split == split) & (predictions.config == config)
                                         & (predictions.tau_key == canonical_tau_key(tau))]
                pd.testing.assert_frame_equal(subset[qp.KEY].sort_values(qp.KEY).reset_index(drop=True), expected)
    assert predictions.loc[predictions.split.eq("test") & predictions.store_nbr.eq("1"), "date"].min() == pd.Timestamp("2016-09-04")
    tuning = diagnostics["tuning"]
    assert tuning.groupby(["fold", "config", "tau_key"]).size().eq(len(GRID)).all()
    assert tuning.groupby(["fold", "config", "tau_key"])["selected"].sum().eq(1).all()
    assert metrics.sample_size.gt(0).all()
    assert predictions.corrected_prediction.ge(0).all()
    assert predictions.corrected_crossing.eq(False).all()
    assert diagnostics["manifest"]["calibration"] == "none"


def test_test_labels_do_not_select_or_refit_models(features, contracts, synthetic_run):
    changed = features.copy()
    # Last target date has no later feature rows: recompute history to avoid
    # masking leakage with stale features, while preserving allowed daily lags.
    changed.loc[changed.date.eq("2016-09-28"), "sales"] += 1e6
    changed = changed.drop(columns=list(FEATURES["sales_lags"].values()) + list(FEATURES["sales_rolling"].values()))
    changed = prepare_features(changed, ready=False)
    other = qp.run_direct_quantiles(changed, *contracts, "synthetic_test_only", GRID)
    pd.testing.assert_frame_equal(synthetic_run[2]["tuning"], other[2]["tuning"])
    np.testing.assert_array_equal(synthetic_run[0].raw_prediction, other[0].raw_prediction)
    assert not synthetic_run[0].y_true.equals(other[0].y_true)


def test_metric_aggregation_raw_corrected_ties_and_crossing():
    rows = pd.DataFrame(dict(date=pd.to_datetime(["2016-09-01", "2016-09-02"]),
                             store_nbr=[1, 1], family=["A", "A"], city=["X", "X"], sales=[0., 4.]))
    predictions = qp.prediction_rows(rows, np.array([[8, -1, 3], [2, 4, 5.]]), "F1", "M0", "test", "synthetic_metrics")
    report = qp.summarize_predictions(predictions)
    fold = report[report.report_level.eq("fold")]
    for variant in ("raw", "corrected"):
        for tau in qp.QUANTILE_LEVELS:
            group = predictions[predictions.tau_key.eq(canonical_tau_key(tau))]
            metric = fold[fold.variant.eq(variant) & fold.tau_key.eq(canonical_tau_key(tau))].iloc[0]
            q = group[f"{variant}_prediction"].to_numpy()
            assert metric.sample_size == 2
            assert metric.pinball_loss == pytest.approx(pinball_loss(rows.sales, q, tau).mean())
            assert metric.coverage == np.mean(rows.sales.to_numpy() <= q)
            assert metric.strict_less_rate == np.mean(rows.sales.to_numpy() < q)
            assert metric.coverage_gap == metric.coverage - tau
            assert metric.coverage == metric.strict_less_rate + metric.tie_rate
            assert metric.crossing_rate == (0.5 if variant == "raw" else 0)
    with pytest.raises(ValueError, match="unique"):
        qp.summarize_predictions(pd.concat([predictions, predictions]))


@pytest.mark.parametrize("fault", ["weather", "mask", "lag", "dates", "grid", "family"])
def test_invalid_contracts_fail_before_fitting(monkeypatch, features, contracts, fault):
    protocol, ablation = deepcopy(contracts)
    frame = features.copy()
    grid = GRID
    if fault == "weather":
        ablation["experiments"]["M3"]["weather_features"] = ["precipitation_sum"]
    elif fault == "mask":
        frame["score_mask"] = False
    elif fault == "lag":
        frame["sales_lag7"] = -1
    elif fault == "dates":
        protocol["rq2"]["validation"]["test_folds"][0]["validation"][1] = "2016-09-02"
    elif fault == "family":
        protocol["scope"]["deep_study_families"] = ["BEVERAGES"]
    else:
        grid = [dict(early_stopping=True)]

    def no_fitting(*args):
        pytest.fail("Invalid contracts should fail before model fitting")

    monkeypatch.setattr(qp, "build_quantile_pipeline", no_fitting)
    with pytest.raises(ValueError):
        qp.run_direct_quantiles(frame, protocol, ablation, "synthetic_invalid", grid)


def test_tie_break_and_selection_use_validation_only(features, contracts):
    fold = next(iter_folds(features, *contracts, contracts[0]["scope"]["deep_study_families"]))
    numeric, _ = qp.feature_columns("M0", contracts[1])
    _, _, records = qp.select_quantile_model(fold.train, fold.validation, numeric, 0.5, [GRID[0], GRID[0]], 42)
    assert [r["selected"] for r in records] == [True, False]
    _, _, changed_records = qp.select_quantile_model(
        fold.train, fold.validation.assign(sales=1e6), numeric, 0.5, [GRID[0]], 42)
    assert changed_records[0]["validation_pinball"] != records[0]["validation_pinball"]


def test_output_paths_are_m4_only_and_never_overwrite(tmp_path, synthetic_run):
    # Writer contract exercised only in pytest's temporary directory.
    qp.save_run(*synthetic_run, root=tmp_path)
    prediction_path, directory = qp.output_paths(tmp_path, "synthetic_test_only")
    before = prediction_path.read_bytes()
    assert prediction_path.name.startswith("m4_quantiles_")
    manifest = json.loads((directory / "manifest.json").read_text())
    assert manifest["selection_metric"] == "raw_mean_validation_pinball"
    with pytest.raises(FileExistsError):
        qp.save_run(*synthetic_run, root=tmp_path)
    assert prediction_path.read_bytes() == before


def test_fold_metrics_are_row_weighted_not_mean_of_group_means():
    rows = pd.DataFrame(dict(date=pd.to_datetime(["2016-09-01"] * 3), store_nbr=[1, 2, 3],
                             family=["A", "A", "B"], city=["X"] * 3, sales=[0., 2., 10.]))
    predictions = qp.prediction_rows(rows, np.zeros((3, 3)), "F1", "M0", "test", "synthetic_weighting")
    report = qp.summarize_predictions(predictions)
    median_raw = report[report.tau_key.eq("tau_0.50") & report.variant.eq("raw")]
    overall = median_raw[median_raw.report_level.eq("fold")].iloc[0]
    assert overall.pinball_loss == 2.0  # (0 + 1 + 5) / 3, not (0.5 + 5) / 2
    assert overall.sample_size == 3
    assert sorted(median_raw.loc[median_raw.report_level.eq("family_city"), "sample_size"]) == [1, 2]
    predictions.loc[0, "tau_key"] = "incorrect"
    with pytest.raises(ValueError, match="tau_key"):
        qp.summarize_predictions(predictions)


@pytest.mark.parametrize("raw", [np.ones((1, 3), dtype=complex), np.ma.array([[1., 2., 3.]])])
def test_prediction_assembly_preserves_phase11_input_guards(raw):
    rows = pd.DataFrame(dict(date=pd.to_datetime(["2016-09-01"]), store_nbr=[1],
                             family=["A"], city=["X"], sales=[1.]))
    with pytest.raises(ValueError, match="complex|masked"):
        qp.prediction_rows(rows, raw, "F1", "M0", "test", "synthetic_bad_matrix")


def test_missing_real_data_has_no_synthetic_fallback(tmp_path):
    with pytest.raises(FileNotFoundError, match="no synthetic fallback"):
        qp.main(["--experiment-id", "synthetic_missing", "--features", str(tmp_path / "missing.parquet")])
