"""M0 only: validation tuning, fresh refit, and fixed-parameter test prediction."""

import argparse
import json
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .evaluation import mase_scales, metric_report
from .feature_spec import FEATURES, m0_numeric
from .rolling_validation import (
    KEY, calendar_rolling_mean, calendar_sales_lag, iter_folds, read_yaml,
    temporary_m0_features, validate_table,
)

CATEGORICAL = FEATURES["categorical"]
OUTPUT = ["date", "store_nbr", "family", "city", "fold", "model", "config", "y_true", "y_pred"]


def prepare_features(data: pd.DataFrame, ready: bool) -> pd.DataFrame:
    data = validate_table(data)
    if ready:
        if "score_mask" not in data:
            raise ValueError("M1 feature table must contain score_mask")
        # Audit supplied lags against calendar lookup before any fitting.
        for lag, column in FEATURES["sales_lags"].items():
            if column not in data:
                raise ValueError(f"Missing M1 lag column: {column}")
            expected = calendar_sales_lag(data, lag)
            if not np.allclose(data[column].to_numpy(dtype=float), expected, equal_nan=True):
                raise ValueError(f"Calendar lag audit failed: {column}")
        for days, column in FEATURES["sales_rolling"].items():
            if column not in data or not np.allclose(data[column].to_numpy(dtype=float), calendar_rolling_mean(data, days), equal_nan=True):
                raise ValueError(f"Calendar rolling audit failed: {column}")
    else:
        data = temporary_m0_features(data)
    missing = set(m0_numeric()) - set(data.columns)
    if missing:
        raise ValueError(f"Missing M0 features: {sorted(missing)}")
    for column in m0_numeric():
        if not pd.api.types.is_numeric_dtype(data[column]) or np.isinf(data[column].to_numpy(dtype=float)).any():
            raise ValueError(f"M0 feature must be numeric and finite-or-NaN: {column}")
    for column in CATEGORICAL:
        data[column] = data[column].astype(str)
    return data


def build_pipeline(model: str, numeric: list[str], parameters: dict, seed: int) -> Pipeline:
    numeric_steps = [("imputer", SimpleImputer(strategy="median", add_indicator=True, keep_empty_features=True))]
    if model == "Ridge":
        numeric_steps.append(("scaler", StandardScaler()))
        estimator = Ridge(solver="lsqr", **parameters)
    elif model == "ExtraTrees":
        estimator = ExtraTreesRegressor(random_state=seed, n_jobs=-1, **parameters)
    else:
        raise ValueError(f"Unsupported model: {model}")
    preprocessing = ColumnTransformer([
        ("numeric", Pipeline(numeric_steps), numeric),
        ("categorical", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL),
    ], remainder="drop")
    return Pipeline([("preprocessing", preprocessing), ("model", estimator)])


def tune(train: pd.DataFrame, validation: pd.DataFrame, model: str, numeric: list[str], candidates: list[dict], seed: int):
    """This function deliberately accepts no test frame."""
    features = numeric + CATEGORICAL
    validation = validation.loc[validation["score_mask"]]
    if not candidates:
        raise ValueError("Empty parameter grid")
    best_parameters, best_mae = None, np.inf
    records = []
    for parameters in candidates:
        pipeline = build_pipeline(model, numeric, parameters, seed)
        pipeline.fit(train[features], train["sales"])
        prediction = pipeline.predict(validation[features])
        score = float(mean_absolute_error(validation["sales"], prediction))
        records.append({"parameters": json.dumps(parameters, sort_keys=True), "validation_MAE": score})
        if score < best_mae:
            best_parameters, best_mae = parameters.copy(), score
    return best_parameters, records


def run_m0(data: pd.DataFrame, protocol: dict, ablation: dict, families: list[str], grids: dict | None = None, check_only: bool = False):
    if ablation["experiments"]["M0"]["weather_features"] != []:
        raise ValueError("This runner implements M0 only")
    if ablation["evaluation"]["mase_seasonality_days"] != 7:
        raise ValueError("Frozen MASE seasonality must be 7")
    # These small grids are explicit runtime choices, not frozen test decisions.
    if grids is None:
        grids = {
            "Ridge": [{"alpha": alpha} for alpha in [0.1, 1.0, 10.0]],
            "ExtraTrees": [dict(n_estimators=n, min_samples_leaf=leaf) for n, leaf in product([100, 200], [5, 20])],
        }
    numeric = m0_numeric()
    predictions, tuning, counts, sanity, scales_by_fold = [], [], [], [], {}
    for fold in iter_folds(data, protocol, ablation, families):
        counts.append(fold.counts)
        for split, frame in [("train", fold.train), ("validation", fold.validation), ("test", fold.test)]:
            sanity.append({"fold": fold.fold_id, "split": split, "rows": len(frame), "min_date": frame["date"].min(), "max_date": frame["date"].max(), "training_rows_excluded": fold.training_rows_excluded if split == "train" else 0})
        if check_only:
            continue
        # No validation/test target enters this frozen denominator.
        scales_by_fold[fold.fold_id] = mase_scales(fold.train, data)
        for model in ["Ridge", "ExtraTrees"]:
            best, records = tune(fold.train, fold.validation, model, numeric, grids[model], seed=42)
            tuning.extend({"fold": fold.fold_id, "model": model, **row, "selected": row["parameters"] == json.dumps(best, sort_keys=True)} for row in records)
            # A new pipeline learns imputation, scaling, and encoding on refit rows.
            refit = fold.refit_rows()
            fitted = build_pipeline(model, numeric, best, seed=42)
            fitted.fit(refit[numeric + CATEGORICAL], refit["sales"])
            test = fold.test.loc[fold.test["score_mask"]]
            prediction = fitted.predict(test[numeric + CATEGORICAL])
            if not np.isfinite(prediction).all():
                raise ValueError("Non-finite predictions")
            output = test[["date", "store_nbr", "family", "city"]].copy()
            output["fold"], output["model"], output["config"] = fold.fold_id, model, "M0"
            output["y_true"], output["y_pred"] = test["sales"], prediction
            predictions.append(output[OUTPUT])
    diagnostics = {"exclusions": pd.concat(counts, ignore_index=True), "splits": pd.DataFrame(sanity), "tuning": pd.DataFrame(tuning)}
    if check_only:
        return pd.DataFrame(columns=OUTPUT), pd.DataFrame(), diagnostics
    result = pd.concat(predictions, ignore_index=True)
    if result.duplicated([*KEY, "fold", "model", "config"]).any():
        raise ValueError("Duplicate prediction keys")
    return result, metric_report(result, scales_by_fold), diagnostics


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", choices=["curated", "features"], default="curated")
    parser.add_argument("--families", nargs="+", required=True)
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args()
    protocol = read_yaml("configs/research_protocol.yaml")
    ablation = read_yaml("configs/ablation_M0_M5.yaml")
    source = {"curated": "data/processed/curated_sales_weather.parquet", "features": "data/processed/model_features.parquet"}[args.source]
    data = prepare_features(pd.read_parquet(source, engine="pyarrow"), ready=args.source == "features")
    predictions, report, diagnostics = run_m0(data, protocol, ablation, args.families, check_only=args.check_only)
    print(diagnostics["splits"].to_string(index=False))
    print(diagnostics["exclusions"].to_string(index=False))
    if args.check_only:
        return
    destination = Path("results/predictions/point_models.csv")
    destination.parent.mkdir(parents=True, exist_ok=True)
    predictions.to_csv(destination, index=False)
    tables = Path("results/tables")
    tables.mkdir(parents=True, exist_ok=True)
    report.to_csv(tables / "point_models_metrics.csv", index=False)
    for name, table in diagnostics.items():
        table.to_csv(tables / f"point_models_{name}.csv", index=False)
    print(f"Saved {len(predictions)} scored predictions to {destination}")


if __name__ == "__main__":
    main()
