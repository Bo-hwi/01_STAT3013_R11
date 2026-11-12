"""Production-only M1 delivery gate. Synthetic training APIs do not call this."""

import json
from pathlib import Path
import re

import numpy as np
import pandas as pd

from src.data.clean_data import WEATHER_VALUE_COLS, WEATHER_META_COLS, load_weather, build_store_day_flags
from src.features.lag_features import PASSTHROUGH, SALES_LAGS, SALES_ROLLING, WEATHER_LAG1
from src.forecasting.point_models import prepare_features
from src.forecasting.rolling_validation import KEY, iter_folds
from .quantile_contracts import file_hash


class ProductionPreflightError(ValueError):
    def __init__(self, report):
        self.report = report
        super().__init__("Production preflight failed: " + "; ".join(report["hard_failures"]))


def production_preflight(feature_path, protocol, ablation, root):
    """Validate against EXISTING catalog/metadata, never fabricated expectations.

    Hard failures stop the CLI. Warnings describe observational limitations;
    unavailable evidence is recorded separately, never promoted to verified.
    M1 metadata does not publish processed-file hashes: locally calculated hashes
    bind this run's bytes but do not authenticate an approved upstream delivery.
    """
    root, feature_path = Path(root), Path(feature_path)
    report = {"hard_failures": [], "warnings": [], "unavailable": [], "verified": [], "file_sha256": {}}

    def check(condition, message):
        if not condition:
            report["hard_failures"].append(message)

    def gate():
        if report["hard_failures"]:
            raise ProductionPreflightError(report)

    paths = {
        "features": feature_path,
        "feature_meta": feature_path.with_suffix(".meta.json"),
        "curated": feature_path.parent / "curated_sales_weather.parquet",
        "curated_meta": feature_path.parent / "curated_sales_weather.meta.json",
        "catalog": root / "docs/source_catalog.csv",
    }
    for name, path in paths.items():
        check(path.is_file(), f"Missing required M1 evidence: {name}: {path}")
    gate()
    initial = {str(p.resolve()): file_hash(p) for p in paths.values()}
    report["file_sha256"].update(initial)
    try:
        fm = json.loads(paths["feature_meta"].read_text(encoding="utf-8"))
        cm = json.loads(paths["curated_meta"].read_text(encoding="utf-8"))
        catalog = pd.read_csv(paths["catalog"])
        features = pd.read_parquet(feature_path)
        curated = pd.read_parquet(paths["curated"])
    except (ValueError, OSError) as exc:
        check(False, f"Unreadable M1 delivery: {exc}")
        gate()
    if any(file_hash(name) != digest for name, digest in initial.items()):
        check(False, "M1 input bytes changed while reading")
    check(isinstance(fm, dict) and isinstance(cm, dict), "M1 metadata must be JSON objects")
    check(set(["file", "sha256", "n_rows", "date_min", "date_max"]).issubset(catalog), "Catalog schema invalid")
    gate()
    check(not catalog["file"].duplicated().any(), "Catalog source filenames are ambiguous")
    check(catalog["sha256"].map(lambda value: isinstance(value, str) and bool(re.fullmatch(r"[0-9a-fA-F]{64}", value))).all(),
          "Catalog contains invalid SHA-256 representations")
    train_record = catalog.loc[catalog["file"].eq("train.csv")]
    check(len(train_record) == 1, "Catalog must identify exactly one train.csv")
    feature_columns = set(PASSTHROUGH + list(SALES_LAGS.values()) + list(SALES_ROLLING.values())
                          + list(WEATHER_LAG1.values()) + ["score_mask"])
    curated_columns = set(PASSTHROUGH + WEATHER_VALUE_COLS + WEATHER_META_COLS
                          + ["transactions", "wind_qa_flag"])
    check(set(features) == feature_columns, "Feature schema differs from existing M1 builder contract")
    check(set(curated) == curated_columns, "Curated schema differs from existing M1 builder contract")
    for label, table, meta in (("features", features, fm), ("curated", curated, cm)):
        check(meta.get("n_rows") == len(table), f"{label} metadata row count mismatch")
        check(meta.get("n_columns") == len(table.columns), f"{label} metadata column count mismatch")
        check(meta.get("key") == KEY, f"{label} metadata key mismatch")
        check(set(KEY).issubset(table), f"{label} lacks keys")
    gate()
    for label, table in (("features", features), ("curated", curated)):
        check(not table[KEY].isna().any().any() and not table.duplicated(KEY).any(), f"{label} null/duplicate keys")
        dates = pd.to_datetime(table.date, errors="coerce")
        check(dates.notna().all() and dates.eq(dates.dt.normalize()).all(), f"{label} invalid calendar dates")
        table["date"] = dates
        check(pd.api.types.is_integer_dtype(table.store_nbr), f"{label} store_nbr must be integer")
        check(pd.api.types.is_numeric_dtype(table.sales), f"{label} sales must be numeric")
    gate()
    record = train_record.iloc[0]
    start, end = map(pd.Timestamp, protocol["scope"]["source_period"])
    check(pd.Timestamp(record.date_min) == start and pd.Timestamp(record.date_max) == end,
          "Protocol source period and train catalog disagree")
    # Missing December 25 follows M1 validate_curated()/QA-01, not invented counts.
    calendar = pd.date_range(start, end)
    calendar = calendar[~((calendar.month == 12) & (calendar.day == 25))]
    for label, table in (("features", features), ("curated", curated)):
        check(len(table) == int(record.n_rows), f"{label} truncated row count versus train catalog")
        check(pd.DatetimeIndex(table.date.unique()).sort_values().equals(calendar), f"{label} date coverage mismatch")
        for column, expected in (("store_nbr", protocol["scope"]["stores_expected"]),
                                 ("city", protocol["scope"]["cities_expected"]),
                                 ("family", protocol["scope"]["families_eda_expected"])):
            check(table[column].notna().all() and table[column].nunique() == expected, f"{label} {column} scope mismatch")
        check(table.groupby("date").size().eq(protocol["scope"]["stores_expected"] * protocol["scope"]["families_eda_expected"]).all(),
              f"{label} missing daily store-family keys")
        check(table.groupby("store_nbr").city.nunique().eq(1).all(), f"{label} store-city mapping inconsistent")
    check(set(protocol["scope"]["deep_study_families"]).issubset(set(features.family)), "Frozen shortlist missing")
    check(cm.get("date_min") == str(start.date()) and cm.get("date_max") == str(end.date()), "Curated metadata dates mismatch")
    for name, actual in (("n_dates", len(calendar)), ("n_stores", curated.store_nbr.nunique()),
                         ("n_cities", curated.city.nunique()), ("n_families", curated.family.nunique()),
                         ("n_weather_grid_cells", curated.weather_grid_id.nunique())):
        check(cm.get(name) == actual, f"Curated metadata {name} mismatch")
    gate()
    for column in WEATHER_LAG1.values():
        check(pd.api.types.is_numeric_dtype(features[column]), f"Nonnumeric weather lag: {column}")
    check(curated[["city", "weather_grid_id", "grid_latitude", "grid_longitude"]].notna().all().all(), "Missing weather grid metadata")
    gate()
    grid_ids = curated.grid_latitude.map("{:.2f}".format) + "_" + curated.grid_longitude.map("{:.2f}".format)
    check(curated.weather_grid_id.astype(str).eq(grid_ids).all(), "weather_grid_id disagrees with M1 coordinate rule")
    check(curated.groupby(["city", "date"])[WEATHER_VALUE_COLS[:5] + ["weather_grid_id"]].nunique().le(1).all().all(),
          "Weather disagrees within city/date")
    expected_flags = build_store_day_flags(curated)
    actual_flags = curated[["date", "store_nbr", "store_active", "store_zero_day"]].drop_duplicates()
    try:
        pd.testing.assert_frame_equal(actual_flags.sort_values(["date", "store_nbr"]).reset_index(drop=True),
                                      expected_flags.sort_values(["date", "store_nbr"]).reset_index(drop=True), check_dtype=False)
    except AssertionError:
        check(False, "Store activity/closure flags disagree with M1 sales-derived rules")
    left, right = (t.sort_values(KEY).reset_index(drop=True) for t in (features, curated))
    try:
        pd.testing.assert_frame_equal(left[PASSTHROUGH], right[PASSTHROUGH], check_dtype=False, check_categorical=False)
    except AssertionError:
        check(False, "Feature passthrough values/keys disagree with curated data")
    recorded = cm.get("source_sha256", {})
    expected_sources = {"train.csv", "stores.csv", "holidays_events.csv", "transactions.csv"}
    expected_sources.update(n for n in catalog.file if n.startswith("weather_") and n != "weather_download_log.csv")
    check(sum(n.startswith("weather_") for n in expected_sources) == protocol["scope"]["cities_expected"],
          "Catalog does not contain expected per-city weather sources")
    known = catalog.set_index("file")["sha256"].to_dict()
    check(expected_sources.issubset(recorded), "Curated metadata lacks required source hashes")
    for name, digest in recorded.items():
        check(name in known and digest == known[name], f"Source hash conflicts with catalog: {name}")
    for row in catalog.itertuples():
        candidates = [root / "data/raw/favorita" / row.file, root / "data/raw/weather" / row.file]
        present = [p for p in candidates if p.is_file()]
        if present:
            digest = file_hash(present[0])
            report["file_sha256"][str(present[0].resolve())] = digest
            check(digest == row.sha256, f"Raw source bytes disagree with catalog: {row.file}")
    missing_raw = sorted(name for name in expected_sources if not any((root / d / name).is_file()
                         for d in ("data/raw/favorita", "data/raw/weather")))
    if missing_raw:
        report["unavailable"].append({"raw_source_byte_verification_missing": missing_raw})
    report["unavailable"].append("No upstream-approved processed-file digest is published by M1 metadata; local hashes are run bindings only")
    report["unavailable"].append("ERA5 available_at timestamps and external coordinate-source validation are not established")
    report["warnings"].append("Retrospective ERA5 and sales-derived closure masks; not an operational availability backtest")
    for col in WEATHER_VALUE_COLS[:5]:
        check(pd.api.types.is_numeric_dtype(curated[col]) and np.isfinite(curated[col].to_numpy(dtype=float)).all(), f"Invalid curated weather: {col}")
    check(curated.precipitation_sum.ge(0).all(), "Negative precipitation")
    check(curated.relative_humidity_2m_mean.between(0, 100).all(), "Humidity outside physical percent range")
    check(curated.sunshine_hours.between(0, 24).all(), "Sunshine outside physical hours/day range")
    if not curated.temperature_2m_max.between(-10, 45).all():
        report["warnings"].append("Temperature outside scripts/qa_weather_checks.py screening range; investigate")
    gate()
    try:
        audited = prepare_features(features, ready=True)
        fold_counts, test_counts = [], {}
        for fold in iter_folds(audited, protocol, ablation, protocol["scope"]["deep_study_families"]):
            fold_counts.append(fold.counts)
            test_counts[fold.fold_id] = int(fold.test.score_mask.sum())
        del fold
    except ValueError as exc:
        check(False, f"Shared feature/fold audit failed: {exc}")
        gate()
    check(fm.get("score_mask_rule", {}).get("rows_scored") == int(features.score_mask.sum()), "M1 total scored-count metadata mismatch")
    counts = pd.concat(fold_counts, ignore_index=True)
    report["fold_scored_counts"] = counts.to_dict(orient="records")
    reference_path = root / "results/tables/baselines_metrics.csv"
    if reference_path.is_file():
        report["file_sha256"][str(reference_path.resolve())] = file_hash(reference_path)
        reference = pd.read_csv(reference_path)
        check({"report_level", "fold", "n_scored"}.issubset(reference), "M1 baseline count-reference schema invalid")
        gate()
        for fold_id, actual_count in test_counts.items():
            values = reference.loc[(reference.report_level == "fold") & (reference.fold == fold_id), "n_scored"].unique()
            check(len(values) == 1 and int(values[0]) == actual_count, f"Test scored count differs from existing M1 baseline table: {fold_id}")
        report["warnings"].append("Baseline count reference has no input hash binding; historical comparison only")
    else:
        report["unavailable"].append("Independent M1 per-fold test scored-count reference unavailable")
    report["unavailable"].append("Independent M1 per-fold validation-count artifact unavailable; counts computed with shared rules")
    weather_names = sorted(n for n in expected_sources if n.startswith("weather_"))
    if weather_names and all((root / "data/raw/weather" / n).is_file() for n in weather_names):
        weather = load_weather(root / "data/raw/weather").set_index(["city", "date"])
        keys = pd.MultiIndex.from_arrays([features.city.astype(str), features.date - pd.Timedelta(days=1)])
        for original, lagged in WEATHER_LAG1.items():
            check(np.allclose(features[lagged].to_numpy(float), weather[original].reindex(keys).to_numpy(float),
                              equal_nan=True, rtol=1e-10, atol=1e-10), f"Weather lag provenance mismatch: {lagged}")
    else:
        report["unavailable"].append("Raw weather lag reconstruction unavailable; only curated/feature QA can be checked")
    for lagged in WEATHER_LAG1.values():
        check(pd.api.types.is_numeric_dtype(features[lagged]), f"Nonnumeric weather lag: {lagged}")
    gate()
    if curated.wind_qa_flag.eq(1).any():
        report["warnings"].append("QA-09 wind outliers retained upstream; wind is excluded from feature schema")
    for lagged in WEATHER_LAG1.values():
        values = features[lagged].to_numpy(float)
        check(not np.isinf(values).any(), f"Infinite weather feature: {lagged}")
        check(not features.loc[features.date.ne(start), lagged].isna().any(), f"Missing weather lags outside first source day: {lagged}")
    check(features.precipitation_sum_lag1.dropna().ge(0).all(), "Negative precipitation lag")
    check(features.relative_humidity_2m_mean_lag1.dropna().between(0, 100).all(), "Humidity lag outside percent range")
    check(features.sunshine_hours_lag1.dropna().between(0, 24).all(), "Sunshine lag outside hours/day range")
    # Re-verify every file that supplied evidence, before permitting any model fit.
    check(all(file_hash(name) == digest for name, digest in report["file_sha256"].items()), "Input evidence changed during preflight")
    gate()
    report["verified"] = ["builder schemas", "catalog counts/calendar/scope", "unique complete sales keys",
                          "curated passthrough", "metadata/catalog source hashes", "shared lags/masks/folds"]
    return features, report
