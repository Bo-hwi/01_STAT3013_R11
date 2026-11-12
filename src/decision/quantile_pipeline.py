"""M4 direct quantile forecasting. See docs/m4_phase2a.md for the fixed procedure.

CLI: python -m src.decision.quantile_pipeline --experiment-id YOUR_RUN_ID
Real runs require M1's feature Parquet; this module never fabricates input data.
"""

import argparse
from importlib.metadata import version
import json
from pathlib import Path
import re
import sys

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder

from src.forecasting.feature_spec import FEATURES, m0_numeric
from src.forecasting.point_models import prepare_features
from src.forecasting.rolling_validation import KEY, iter_folds, read_yaml
from .quantile_models import (
    QUANTILE_LEVELS, canonical_tau_key, coverage_gap, enforce_quantile_order,
    normalize_tau, pinball_loss, quantile_coverage,
)
from .quantile_contracts import (
    freeze_provenance, json_snapshot, scored_key_digest, stable_mean,
    validate_prediction_panel, verify_frozen_files, semantic_hash,
)
from .quantile_artifacts import inspect_run, output_paths


ROOT = Path(__file__).resolve().parents[2]
CONFIGS = ("M0", "M3")
PILOT_LABEL = "PILOT / NOT FINAL RESEARCH RESULTS"
MODEL = "HistGradientBoostingQuantile"
CORE_WEATHER = FEATURES["weather"][:2]
DEFAULT_GRID = (
    {"max_iter": 100, "max_leaf_nodes": 15, "min_samples_leaf": 20, "learning_rate": 0.05},
    {"max_iter": 100, "max_leaf_nodes": 31, "min_samples_leaf": 20, "learning_rate": 0.05},
)
TUNABLE = {"max_iter", "max_leaf_nodes", "min_samples_leaf", "learning_rate", "l2_regularization", "max_depth"}
PREDICTION_COLUMNS = [
    "date", "store_nbr", "family", "city", "fold", "config", "tau", "tau_key",
    "y_true", "raw_prediction", "corrected_prediction", "raw_crossing",
    "corrected_crossing", "split", "fit_stage", "model", "experiment_id",
    "forecast_origin", "horizon_days",
]
PREDICTION_KEY = [*KEY, "fold", "config", "tau_key", "split", "model", "experiment_id"]


def feature_columns(config, ablation):
    """Explicit whitelist: never infer features from numeric columns in a frame."""
    if config not in CONFIGS:
        raise ValueError("Phase 2A supports M0 and M3 only")
    weather = list(ablation["experiments"][config]["weather_features"])
    expected = [] if config == "M0" else list(CORE_WEATHER)
    if weather != expected or not ablation["experiments"][config]["enabled"]:
        raise ValueError(f"Unexpected frozen weather whitelist for {config}")
    return m0_numeric() + weather, list(FEATURES["categorical"])


def build_quantile_pipeline(numeric, tau, parameters, seed):
    """Learn imputation/encoding on fit rows only; dense native categoricals.

    Unknown categories become NaN, which HGB treats as missing. No one-hot
    expansion or scaling; native categorical flags prevent ordinal distance use.
    Early stopping is disabled: max_iter is selected only on shared validation.
    """
    if set(parameters) - TUNABLE:
        raise ValueError(f"Unsupported tuning parameters: {sorted(set(parameters) - TUNABLE)}")
    tau = normalize_tau(tau)
    categorical = list(FEATURES["categorical"])
    preprocessing = ColumnTransformer([
        ("numeric", SimpleImputer(strategy="median", keep_empty_features=True), numeric),
        ("categorical", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=np.nan,
                                       encoded_missing_value=np.nan), categorical),
    ], remainder="drop", sparse_threshold=0)
    estimator = HistGradientBoostingRegressor(
        loss="quantile", quantile=tau, early_stopping=False, random_state=seed,
        categorical_features=[False] * len(numeric) + [True] * len(categorical),
        **parameters,
    )
    return Pipeline([("preprocessing", preprocessing), ("model", estimator)])


def select_quantile_model(train, validation, numeric, tau, candidates, seed):
    """No test argument. Select raw mean pinball; ties keep first candidate.

    Returned model is train-only. Its validation predictions are selection
    diagnostics, NOT an independent calibration set or final performance claim.
    """
    tau = normalize_tau(tau)
    scored = validation.loc[validation["score_mask"]]
    if train.empty or scored.empty or not candidates:
        raise ValueError("Training, scored validation and candidate grid must be nonempty")
    columns = numeric + list(FEATURES["categorical"])
    best, best_score, records = None, np.inf, []
    for candidate_id, parameters in enumerate(candidates):
        fitted = build_quantile_pipeline(numeric, tau, parameters, seed)
        fitted.fit(train[columns], train["sales"])
        prediction = fitted.predict(scored[columns])
        score = stable_mean(pinball_loss(scored["sales"], prediction, tau))
        records.append({"candidate_id": candidate_id, "parameters": json.dumps(parameters, sort_keys=True),
                        "validation_pinball": score, "sample_size": len(scored)})
        if score < best_score:
            best_score, best = score, (candidate_id, fitted, dict(parameters))
    for record in records:
        record["selected"] = record["candidate_id"] == best[0]
    return best[1], best[2], records


def prediction_rows(frame, raw, fold, config, split, experiment_id):
    """Build long predictions from an aligned n-by-3 matrix, preserving keys."""
    # Validate before float coercion, preserving Phase 1.1 mask/complex guards.
    corrected = enforce_quantile_order(raw)
    raw = np.asarray(raw, dtype=float)
    if raw.shape != (len(frame), len(QUANTILE_LEVELS)):
        raise ValueError("Prediction matrix must align with frame rows and canonical taus")
    raw_crossing = np.any(raw[:, :-1] > raw[:, 1:], axis=1)
    corrected_crossing = np.any(corrected[:, :-1] > corrected[:, 1:], axis=1)
    outputs = []
    for column, tau in enumerate(QUANTILE_LEVELS):
        out = frame[[*KEY, "city"]].copy()
        out["fold"], out["config"], out["tau"], out["tau_key"] = fold, config, tau, canonical_tau_key(tau)
        out["y_true"] = frame["sales"]
        out["raw_prediction"], out["corrected_prediction"] = raw[:, column], corrected[:, column]
        out["raw_crossing"], out["corrected_crossing"] = raw_crossing, corrected_crossing
        out["split"], out["experiment_id"], out["model"] = split, experiment_id, MODEL
        out["fit_stage"] = "train_only_selected" if split == "validation" else "train_plus_validation_refit"
        out["forecast_origin"] = frame["date"] - pd.Timedelta(days=1)
        out["horizon_days"] = 1
        outputs.append(out[PREDICTION_COLUMNS])
    return pd.concat(outputs, ignore_index=True)


def summarize_predictions(predictions):
    """Row-weighted fold and family-city metrics, separately for raw/corrected.

    Crossing rate is the fraction of observation triplets with any crossing,
    repeated per tau for readability, not a fraction of individual columns.
    """
    validate_prediction_panel(predictions)
    base = ["experiment_id", "model", "fold", "config", "split", "tau_key"]
    reports = []
    for level, grouping in [("fold", base), ("family_city", base + ["family", "city"])]:
        for key, group in predictions.groupby(grouping, observed=True, sort=True):
            tau = normalize_tau(group["tau"].iloc[0])
            truth = group["y_true"].to_numpy()
            for variant in ("raw", "corrected"):
                prediction = group[f"{variant}_prediction"].to_numpy()
                reports.append({
                    **dict(zip(grouping, key)), "report_level": level, "variant": variant,
                    "tau": tau, "sample_size": len(group),
                    "pinball_loss": stable_mean(pinball_loss(truth, prediction, tau)),
                    "coverage": quantile_coverage(truth, prediction),
                    "strict_less_rate": float(np.mean(truth < prediction)),
                    "tie_rate": float(np.mean(truth == prediction)),
                    "coverage_gap": coverage_gap(truth, prediction, tau),
                    "crossing_rate": float(group[f"{variant}_crossing"].mean()),
                    "negative_prediction_rate": float(np.mean(prediction < 0)),
                })
    return pd.DataFrame(reports)


def _scope_ids(requested, available, name):
    """Validate a nonempty unique subset; execute in shared contract order."""
    if requested is None:
        return tuple(available)
    if isinstance(requested, (str, bytes)):
        raise ValueError(f"{name} must be a sequence of IDs, not a string")
    try:
        values = list(requested)
    except TypeError as exc:
        raise ValueError(f"{name} must be a sequence of IDs") from exc
    if not values or any(not isinstance(value, str) for value in values):
        raise ValueError(f"{name} requires nonempty string IDs")
    if len(set(values)) != len(values):
        raise ValueError(f"Duplicate {name}")
    if set(values) - set(available):
        raise ValueError(f"Unsupported {name}: {sorted(set(values) - set(available))}")
    return tuple(value for value in available if value in values)


def _scoped_folds(data, protocol, ablation, families, fold_ids):
    """Select Fold objects from the shared splitter; never split dates here."""
    remaining = set(fold_ids)
    for fold in iter_folds(data, protocol, ablation, families):
        if fold.fold_id in remaining:
            remaining.remove(fold.fold_id)
            yield fold
            if not remaining:
                break
    if remaining:
        raise ValueError(f"Shared iterator did not yield requested folds: {sorted(remaining)}")


def run_direct_quantiles(frame, protocol, ablation, experiment_id, parameter_grid=None, input_context=None,
                         fold_ids=None, configs=None):
    """Run shared folds on M1 features; return predictions, metrics, diagnostics.

    1. Audit calendar lags, then reuse shared eligibility/masks/folds.
    2. Select requested config/tau models on validation before test prediction.
    3. Save selected train-only validation diagnostics; refit fresh pipelines on
       Fold.refit_rows(), then predict scored test features with fixed parameters.
    4. Clip/sort without labels; no calibration. Test targets do not enter model
       selection/refit; observed earlier test sales may supply next-day lags.
    """
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", experiment_id):
        raise ValueError("experiment_id must be a safe nonempty filename identifier")
    protocol, ablation = json_snapshot(protocol), json_snapshot(ablation)
    families = list(protocol["scope"]["deep_study_families"])
    if len(families) != 7 or len(set(families)) != 7:
        raise ValueError("Require the frozen seven-family shortlist from protocol")
    if tuple(protocol["quantile_and_decision"]["compare_weather"]) != CONFIGS:
        raise ValueError("Expected the shared M0/M3 comparison")
    selected_configs = _scope_ids(configs, CONFIGS, "configs")
    selected_fold_ids = _scope_ids(fold_ids, [f["fold_id"] for f in
        protocol["rq2"]["validation"]["test_folds"]], "fold_ids")
    if tuple(normalize_tau(t) for t in protocol["quantile_and_decision"]["quantiles"]) != QUANTILE_LEVELS:
        raise ValueError("Expected the three canonical quantiles in increasing order")
    validation_rules = protocol["rq2"]["validation"]
    if (protocol["rq2"]["forecast_horizon_days"] != 1
            or validation_rules["refit_before_test"] != "train_plus_validation"
            or validation_rules["model_parameters_during_test"] != "fixed"):
        raise ValueError("Phase 2A requires shared one-day, fixed-parameter refit rules")
    candidates = [dict(p) for p in (DEFAULT_GRID if parameter_grid is None else parameter_grid)]
    candidates = json_snapshot(candidates)
    if not candidates:
        raise ValueError("Candidate grid must be nonempty")
    for candidate in candidates:
        if set(candidate) - TUNABLE:
            raise ValueError("Grid cannot override loss, tau, seed, categorical encoding or early stopping")
    whitelists = {config: feature_columns(config, ablation) for config in selected_configs}
    data = prepare_features(frame, ready=True)
    if data["sales"].lt(0).any():
        raise ValueError("Observed sales must be nonnegative")
    for numeric, _ in whitelists.values():
        for column in numeric:
            if column not in data or not pd.api.types.is_numeric_dtype(data[column]):
                raise ValueError(f"Missing or nonnumeric feature: {column}")
            if np.iscomplexobj(data[column]):
                raise ValueError(f"Feature must be real finite-or-NaN: {column}")
            values = data[column].to_numpy(dtype=float)
            if np.isinf(values).any():
                raise ValueError(f"Feature must be real finite-or-NaN: {column}")
    seed = int(protocol["project"]["seed"])
    # First shared-fold pass freezes scored identities before ANY model fitting.
    # Retain only small key tables, not all expanding training DataFrames.
    expected_tables, key_contract = [], []
    for fold in _scoped_folds(data, protocol, ablation, families, selected_fold_ids):
        for split in ("validation", "test"):
            rows = getattr(fold, split)
            keys = rows.loc[rows.score_mask, [*KEY, "city"]].copy()
            keys["fold"], keys["split"] = fold.fold_id, split
            expected_tables.append(keys)
            key_contract.append({"fold": fold.fold_id, "split": split, "rows": len(keys),
                                 "sha256": scored_key_digest(keys)})
    del fold
    expected_rows = pd.concat(expected_tables, ignore_index=True)
    provenance = freeze_provenance(ROOT, protocol, ablation, input_context)
    manifest = json_snapshot({
        "experiment_id": experiment_id, "model": MODEL, "seed": seed,
        "quantiles": list(QUANTILE_LEVELS), "families": families,
        "folds": [f for f in protocol["rq2"]["validation"]["test_folds"] if f["fold_id"] in selected_fold_ids],
        "selected_scope": {"fold_ids": list(selected_fold_ids), "configs": list(selected_configs)},
        "features": {c: n + k for c, (n, k) in whitelists.items()},
        "parameter_grid": candidates, "budget_per_config_tau_fold": len(candidates),
        "selection_metric": "raw_mean_validation_pinball", "tie_break": "first_candidate",
        "refit": "fresh_pipeline_on_eligible_train_plus_validation", "early_stopping": False,
        "calibration": "none", "correction": "clip_zero_then_sort_per_row",
        "validation_warning": "selection diagnostics; not independent calibration or performance estimates",
        "pilot_only": (input_context or {}).get("mode") == "pilot",
        "result_label": PILOT_LABEL if (input_context or {}).get("mode") == "pilot" else ablation["result_label"],
        "python": sys.version,
        "packages": {p: version(p) for p in ("numpy", "pandas", "scikit-learn", "scipy", "pyarrow", "PyYAML")},
        "contracts": {"protocol": protocol, "ablation": ablation},
        "scored_key_contract": key_contract, "provenance": provenance,
    })
    manifest["run_spec_sha256"] = semantic_hash(manifest)
    verify_frozen_files(provenance)
    outputs, tuning, split_log, exclusion_log = [], [], [], []
    for fold in _scoped_folds(data, protocol, ablation, families, selected_fold_ids):
        exclusion_log.append(fold.counts)
        for split in ("train", "validation", "test"):
            rows = getattr(fold, split)
            split_log.append({"fold": fold.fold_id, "split": split, "rows": len(rows),
                              "min_date": rows["date"].min(), "max_date": rows["date"].max(),
                              "eligible_fit_rows": int(rows["store_zero_day"].eq(0).sum()),
                              "scored_rows": int(rows["score_mask"].sum()),
                              "training_rows_excluded": fold.training_rows_excluded if split == "train" else 0})
        selected = {}
        for config in selected_configs:
            numeric, _ = whitelists[config]
            for tau in QUANTILE_LEVELS:
                model, parameters, records = select_quantile_model(
                    fold.train, fold.validation, numeric, tau, candidates, seed)
                selected[config, canonical_tau_key(tau)] = (model, parameters)
                tuning.extend({"fold": fold.fold_id, "config": config, "tau_key": canonical_tau_key(tau),
                               **record} for record in records)
        # Selection is now frozen for this fold. No test frame entered tuning.
        refit_rows = fold.refit_rows()
        for config in selected_configs:
            numeric, categorical = whitelists[config]
            columns = numeric + categorical
            validation = fold.validation.loc[fold.validation["score_mask"]]
            test = fold.test.loc[fold.test["score_mask"]]
            validation_raw, test_raw = [], []
            for tau in QUANTILE_LEVELS:
                selected_model, parameters = selected[config, canonical_tau_key(tau)]
                validation_raw.append(selected_model.predict(validation[columns]))
                refitted = build_quantile_pipeline(numeric, tau, parameters, seed)
                refitted.fit(refit_rows[columns], refit_rows["sales"])
                test_raw.append(refitted.predict(test[columns]))
            for split, rows, raw in [("validation", validation, validation_raw), ("test", test, test_raw)]:
                outputs.append(prediction_rows(rows, np.column_stack(raw), fold.fold_id, config, split, experiment_id))
    predictions = pd.concat(outputs, ignore_index=True)
    validate_prediction_panel(predictions, expected_rows, selected_configs)
    verify_frozen_files(provenance)
    return predictions, summarize_predictions(predictions), {
        "tuning": pd.DataFrame(tuning), "splits": pd.DataFrame(split_log),
        "exclusions": pd.concat(exclusion_log, ignore_index=True), "manifest": manifest,
    }


def save_run(predictions, metrics, diagnostics, root=ROOT):
    """Compatibility entry point for the single-directory transaction writer."""
    from .quantile_artifacts import save_run as transactional_save
    return transactional_save(predictions, metrics, diagnostics, root)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment-id", required=True)
    parser.add_argument("--features", type=Path, default=ROOT / "data/processed/model_features.parquet")
    args = parser.parse_args(argv)
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", args.experiment_id):
        raise ValueError("Unsafe experiment identifier")
    if not args.features.is_file():
        raise FileNotFoundError("M1 model_features.parquet is required; no synthetic fallback is permitted")
    if inspect_run(ROOT, args.experiment_id)["status"] != "absent":
        raise FileExistsError("Experiment exists or is incomplete; inspect it before retrying")
    protocol = read_yaml(ROOT / "configs/research_protocol.yaml")
    ablation = read_yaml(ROOT / "configs/ablation_M0_M5.yaml")
    from .quantile_preflight import production_preflight
    data, preflight = production_preflight(args.features, protocol, ablation, ROOT)
    result = run_direct_quantiles(data, protocol, ablation, args.experiment_id,
                                  input_context={"mode": "production", **preflight})
    save_run(*result)
    print(f"Saved M4 run {args.experiment_id}; {len(result[0])} prediction rows")


if __name__ == "__main__":
    main()
