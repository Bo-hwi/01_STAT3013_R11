"""M4 panel validation and immutable, pre-fit run provenance."""

from hashlib import sha256
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .quantile_models import QUANTILE_LEVELS, canonical_tau_key, enforce_quantile_order


TAU_KEYS = tuple(canonical_tau_key(t) for t in QUANTILE_LEVELS)
OBSERVATION = ["experiment_id", "model", "fold", "split", "date", "store_nbr", "family"]
MATCH_KEYS = ["fold", "split", "date", "store_nbr", "family", "city"]


def file_hash(path):
    digest = sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def json_snapshot(value):
    """Detach caller-owned dictionaries and reject non-JSON/NaN before fitting."""
    return json.loads(json.dumps(value, sort_keys=True, allow_nan=False))


def semantic_hash(value):
    return sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def stable_mean(values):
    """Mean of finite nonnegative losses without overflowing intermediate sums."""
    values = np.asarray(values, dtype=float)
    if values.ndim != 1 or not values.size or not np.isfinite(values).all() or (values < 0).any():
        raise ValueError("Mean requires nonempty finite nonnegative losses")
    maximum = float(values.max())
    if maximum == 0:
        return 0.0
    with np.errstate(over="ignore", under="ignore"):
        total = float(values.sum())
        # Preserve ordinary retail-scale arithmetic; scale only on overflow.
        result = total / values.size if np.isfinite(total) else maximum * float(np.mean(values / maximum))
    if not np.isfinite(result) or result == 0:
        raise ValueError("Mean is outside the nonzero finite float64 range")
    return result


def normalized_keys(rows):
    rows = rows[MATCH_KEYS].copy()
    rows["date"] = pd.to_datetime(rows["date"], errors="raise")
    for column in set(MATCH_KEYS) - {"date"}:
        rows[column] = rows[column].astype(str)
    return rows.sort_values(MATCH_KEYS).reset_index(drop=True)


def scored_key_digest(rows):
    normalized = normalized_keys(rows)
    return sha256(normalized.to_csv(index=False, date_format="%Y-%m-%d").encode()).hexdigest()


def validate_prediction_panel(panel, expected_rows=None, expected_configs=None):
    """Gate metrics/cost consumers: full triplets, aligned rows, truth and flags.

    expected_rows (fold/split/date/store/family/city) catches even observations
    missing from EVERY config. Production runner supplies this from shared folds.
    Without it, only internal panel consistency can be established.
    """
    required = OBSERVATION + ["city", "config", "tau", "tau_key", "y_true",
                             "raw_prediction", "corrected_prediction", "raw_crossing", "corrected_crossing"]
    if panel.empty or set(required) - set(panel):
        raise ValueError("Prediction panel is empty or missing required columns")
    if panel[required].isna().any().any():
        raise ValueError("Prediction panel has null required keys or values")
    dates = pd.to_datetime(panel["date"], errors="raise")
    if dates.dt.tz is not None or not dates.eq(dates.dt.normalize()).all():
        raise ValueError("Prediction dates must be timezone-naive calendar dates")
    if "horizon_days" in panel and not panel["horizon_days"].eq(1).all():
        raise ValueError("Prediction horizon must be one day")
    if "forecast_origin" in panel and not pd.to_datetime(panel["forecast_origin"]).eq(dates - pd.Timedelta(days=1)).all():
        raise ValueError("Forecast origin disagrees with one-day-ahead contract")
    unique = OBSERVATION + ["config", "tau_key"]
    if panel.duplicated(unique).any():
        raise ValueError("Prediction panel must have unique prediction keys")
    # Normalize just the distinct taus, not hundreds of thousands of scalar calls.
    mapping = {t: canonical_tau_key(t) for t in panel["tau"].unique()}
    if not panel["tau"].map(mapping).eq(panel["tau_key"]).all():
        raise ValueError("tau and tau_key disagree")
    if set(panel["tau_key"]) != set(TAU_KEYS):
        raise ValueError("Prediction panel must contain all canonical tau keys")
    configs = tuple(sorted(panel["config"].unique())) if expected_configs is None else tuple(expected_configs)
    if set(panel["config"]) != set(configs) or len(set(configs)) != len(configs):
        raise ValueError("Prediction config set does not match expected configs")
    if not panel["split"].isin(["validation", "test"]).all():
        raise ValueError("Unsupported prediction split")
    if (panel["y_true"] < 0).any():
        raise ValueError("Prediction demand proxy must be nonnegative")
    group = OBSERVATION + ["config"]
    counts = panel.groupby(group, dropna=False).size()
    if not counts.eq(3).all():
        raise ValueError("Each observation/config must have exactly three canonical taus")
    # Keys/truth must be identical across taus AND configs.
    if panel.groupby(OBSERVATION, dropna=False)[["city", "y_true"]].nunique().gt(1).any().any():
        raise ValueError("Prediction city/truth disagrees across taus or configs")
    if not panel.groupby(OBSERVATION, dropna=False)["config"].nunique().eq(len(configs)).all():
        raise ValueError("Configs do not have matched scored observations")
    ranked = panel.assign(_tau_rank=panel["tau_key"].map(dict(zip(TAU_KEYS, range(3)))))
    ranked = ranked.sort_values(group + ["_tau_rank"])
    raw = ranked["raw_prediction"].to_numpy(dtype=float).reshape(-1, 3)
    corrected = ranked["corrected_prediction"].to_numpy(dtype=float).reshape(-1, 3)
    expected_corrected = enforce_quantile_order(raw)
    if not np.isfinite(corrected).all() or not np.array_equal(corrected, expected_corrected):
        raise ValueError("Corrected predictions must equal nonnegative sorted raw predictions")
    if not np.isfinite(panel["y_true"].to_numpy(dtype=float)).all():
        raise ValueError("Prediction truth must be finite")
    for name, matrix in (("raw", raw), ("corrected", corrected)):
        flags = ranked[f"{name}_crossing"]
        if not flags.isin([True, False]).all():
            raise ValueError("Crossing flags must be boolean")
        actual = np.repeat(np.any(matrix[:, :-1] > matrix[:, 1:], axis=1), 3)
        if not np.array_equal(flags.to_numpy(dtype=bool), actual):
            raise ValueError(f"Inconsistent {name} crossing flags")
    if expected_rows is not None:
        if expected_rows[MATCH_KEYS].isna().any().any() or expected_rows.duplicated(MATCH_KEYS).any():
            raise ValueError("Expected scored rows must have unique non-null keys")
        expected = normalized_keys(expected_rows)
        for config in configs:
            actual = normalized_keys(panel.loc[(panel["config"] == config) & (panel["tau_key"] == TAU_KEYS[0])])
            if not actual.equals(expected):
                raise ValueError("Predictions do not match expected scored rows")
    return panel


def freeze_provenance(root, protocol, ablation, input_context=None):
    """Capture effective specs, source bytes and available input evidence PRE-FIT."""
    root = Path(root)
    files = [root / name for name in (
        "src/decision/quantile_pipeline.py", "src/decision/quantile_models.py",
        "src/decision/cost_simulation.py", "src/decision/quantile_contracts.py",
        "src/decision/quantile_preflight.py", "src/decision/quantile_artifacts.py",
        "scripts/m4_pilot.py",
        "src/forecasting/feature_spec.py", "src/forecasting/point_models.py",
        "src/forecasting/rolling_validation.py", "configs/research_protocol.yaml",
        "src/features/lag_features.py", "src/data/clean_data.py", "src/data/merge_data.py",
        "configs/ablation_M0_M5.yaml", "configs/cost_scenarios.yaml",
    )]
    return json_snapshot({
        "file_sha256": {str(p.resolve()): file_hash(p) for p in files},
        "effective_protocol_sha256": semantic_hash(protocol),
        "effective_ablation_sha256": semantic_hash(ablation),
        "input": input_context or {"mode": "in_memory", "upstream_provenance": "unverified"},
    })


def verify_frozen_files(provenance):
    for name, digest in provenance["file_sha256"].items():
        if file_hash(name) != digest:
            raise ValueError(f"Code/config changed after specification freeze: {name}")
    for name, digest in provenance["input"].get("file_sha256", {}).items():
        if file_hash(name) != digest:
            raise ValueError(f"Input evidence changed after preflight: {name}")
