"""Calendar-safe features, shared scoring exclusions, and frozen time splits."""

from dataclasses import dataclass
from pathlib import Path
from collections.abc import Iterator, Mapping, Sequence

import numpy as np
import pandas as pd
import yaml

from .feature_spec import FEATURES

KEY = ["date", "store_nbr", "family"]


class UniqueKeyLoader(yaml.SafeLoader):
    """Reject duplicate mapping keys instead of silently keeping the last value."""


def _unique_mapping(loader, node, deep=False):
    mapping = {}
    for key_node, value_node in node.value:
        key = loader.construct_object(key_node, deep=deep)
        if key in mapping:
            raise ValueError(f"Duplicate YAML key {key!r} at line {key_node.start_mark.line + 1}")
        mapping[key] = loader.construct_object(value_node, deep=deep)
    return mapping


UniqueKeyLoader.add_constructor(yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG, _unique_mapping)


def read_yaml(path: str | Path) -> dict:
    with Path(path).open(encoding="utf-8") as stream:
        result = yaml.load(stream, Loader=UniqueKeyLoader)
    if not isinstance(result, dict):
        raise ValueError(f"Empty or invalid YAML: {path}")
    return result


def validate_table(frame: pd.DataFrame) -> pd.DataFrame:
    required = KEY + ["sales", "city", "store_active", "store_zero_day"]
    missing = set(required) - set(frame.columns)
    if missing:
        raise ValueError(f"Missing columns: {sorted(missing)}")
    data = frame.copy()
    data["date"] = pd.to_datetime(data["date"], errors="raise")
    if data[required].isna().any().any():
        raise ValueError("Required columns contain null values")
    if data["date"].dt.tz is not None or not data["date"].eq(data["date"].dt.normalize()).all():
        raise ValueError("Dates must be timezone-naive calendar dates")
    if data.duplicated(KEY).any():
        raise ValueError("Duplicate date-store-family keys")
    for column in ["store_active", "store_zero_day"]:
        if not data[column].isin([0, 1]).all():
            raise ValueError(f"{column} must contain only 0/1")
    if not np.isfinite(data["sales"].to_numpy(dtype=float)).all():
        raise ValueError("Sales must be finite")
    return data.sort_values(KEY).reset_index(drop=True)


def calendar_sales_lag(data: pd.DataFrame, days: int) -> np.ndarray:
    """Look up t-days by key; absent dates remain NaN, never zero."""
    lookup = data.set_index(KEY)["sales"]
    keys = pd.MultiIndex.from_arrays(
        [data["date"] - pd.Timedelta(days=days), data["store_nbr"], data["family"]],
        names=KEY,
    )
    return lookup.reindex(keys).to_numpy(dtype=float)


def temporary_sales_lags(data: pd.DataFrame, lag_columns: Mapping[int, str]) -> pd.DataFrame:
    """Use caller-confirmed names so the M1 feature table has the same contract."""
    result = data.copy()
    for days, column in lag_columns.items():
        if days <= 0 or column in data.columns:
            raise ValueError(f"Invalid lag or existing destination column: {days}, {column}")
        result[column] = calendar_sales_lag(data, days)
    return result


def calendar_rolling_mean(data: pd.DataFrame, days: int) -> pd.Series:
    """Mean over [t-days, t), skipping absent calendar dates."""
    result = pd.Series(np.nan, index=data.index, dtype=float)
    for _, group in data.groupby(["store_nbr", "family"], observed=True, sort=False):
        ordered = group.sort_values("date")
        means = ordered.set_index("date")["sales"].rolling(
            f"{days}D", closed="left", min_periods=1
        ).mean()
        result.loc[ordered.index] = means.to_numpy()
    return result


def temporary_m0_features(data: pd.DataFrame) -> pd.DataFrame:
    """Build M0 history features before filtering any source rows."""
    result = temporary_sales_lags(data, FEATURES["sales_lags"])
    for days, column in FEATURES["sales_rolling"].items():
        if column in result:
            raise ValueError(f"Existing destination column: {column}")
        result[column] = calendar_rolling_mean(data, days)
    return result


def add_scoring_mask(data: pd.DataFrame, rules: dict) -> pd.DataFrame:
    """Detect runs on the full store calendar before splitting or family selection."""
    minimum = int(rules["post_reopening"]["minimum_consecutive_closure_days"])
    cooldown = int(rules["post_reopening"]["excluded_calendar_days"])
    if minimum < 1 or cooldown < 1:
        raise ValueError("Closure and reopening windows must be positive")
    flags = ["store_active", "store_zero_day"]
    if data.groupby(["store_nbr", "date"])[flags].nunique().gt(1).any().any():
        raise ValueError("Store-day closure flags disagree across families")
    days = data[["store_nbr", "date"] + flags].drop_duplicates()
    days = days.sort_values(["store_nbr", "date"]).reset_index(drop=True)
    closed = days["store_active"].eq(1) & days["store_zero_day"].eq(1)
    same_store = days["store_nbr"].eq(days["store_nbr"].shift())
    adjacent = same_store & days["date"].diff().eq(pd.Timedelta(days=1))
    run_start = ~adjacent | closed.ne(closed.shift())
    run_id = run_start.cumsum()
    run_length = closed.groupby(run_id).cumsum()
    reopened = (
        adjacent & ~closed & days["store_active"].eq(1)
        & closed.shift(fill_value=False) & run_length.shift(fill_value=0).ge(minimum)
    )
    last_reopening = days["date"].where(reopened).groupby(days["store_nbr"]).ffill()
    elapsed = (days["date"] - last_reopening).dt.days
    days["excluded_closure"] = days["store_zero_day"].eq(1)
    days["excluded_post_reopening"] = elapsed.ge(0) & elapsed.lt(cooldown) & ~closed
    days["computed_score_mask"] = (
        days["store_active"].eq(1)
        & ~days["excluded_closure"] & ~days["excluded_post_reopening"]
    )
    result = data.merge(
        days[["store_nbr", "date", "excluded_closure", "excluded_post_reopening", "computed_score_mask"]],
        on=["store_nbr", "date"], how="left", validate="many_to_one",
    )
    if "score_mask" in result:
        if not result["score_mask"].isin([0, 1]).all():
            raise ValueError("M1 score_mask must be non-null boolean or 0/1")
        active = result["store_active"].eq(1)
        mismatch = result.loc[active, "score_mask"].astype(bool).ne(
            result.loc[active, "computed_score_mask"]
        )
        if mismatch.any():
            raise ValueError(f"M1 score_mask disagrees with shared rules: {mismatch.sum()} rows")
    result["score_mask"] = result.pop("computed_score_mask")
    return result


@dataclass
class Fold:
    fold_id: str
    train: pd.DataFrame
    validation: pd.DataFrame
    test: pd.DataFrame
    counts: pd.DataFrame
    training_rows_excluded: int

    def refit_rows(self) -> pd.DataFrame:
        # Cooldown targets remain eligible for fitting, but closure targets do not.
        rows = pd.concat([self.train, self.validation], ignore_index=True)
        return rows.loc[rows["store_zero_day"].eq(0)].copy()


def exclusion_counts(frame: pd.DataFrame, fold: str, split: str) -> dict:
    closure = frame["excluded_closure"]
    post_only = frame["excluded_post_reopening"] & ~closure
    return {
        "fold_id": fold, "split": split,
        "rows_before_exclusions": len(frame),
        "rows_excluded_closure": int(closure.sum()),
        "rows_excluded_post_reopening_only": int(post_only.sum()),
        "rows_excluded_total_unique": int((~frame["score_mask"]).sum()),
        "rows_scored": int(frame["score_mask"].sum()),
    }


def iter_folds(
    frame: pd.DataFrame, protocol: dict, ablation: dict, families: Sequence[str]
) -> Iterator[Fold]:
    if not families or len(set(families)) != len(families):
        raise ValueError("Pass a nonempty, unique family shortlist from M2")
    data = validate_table(frame)
    unknown = set(families) - set(data["family"])
    if unknown:
        raise ValueError(f"Unknown families: {sorted(unknown)}")
    if ablation["row_filter"] != "store_active == 1":
        raise ValueError("Unexpected row filter")
    rules = ablation["scoring_exclusions"]
    if rules["modify_training_rows"] != "exclude_store_zero_day" or rules["modify_lag_inputs"] is not False:
        raise ValueError("Unexpected training/lag exclusion policy")
    data = add_scoring_mask(data, rules)
    data = data.loc[data["store_active"].eq(1) & data["family"].isin(families)]
    specifications = protocol["rq2"]["validation"]["test_folds"]
    if not specifications:
        raise ValueError("No frozen test folds")
    previous_test_end = None
    previous_train_end = None
    first_train_start = None
    seen = set()
    parsed = []
    # Validate the complete schedule before yielding any fold.
    for spec in specifications:
        ranges = {part: tuple(pd.Timestamp(x) for x in spec[part]) for part in ["train", "validation", "test"]}
        (ts, te), (vs, ve), (ss, se) = (ranges[p] for p in ["train", "validation", "test"])
        if not ts <= te < vs <= ve < ss <= se:
            raise ValueError(f"Invalid temporal order: {spec}")
        if (ve - vs).days != 27 or (se - ss).days != 27 or ss - ve != pd.Timedelta(days=1):
            raise ValueError("Validation/test must be adjacent inclusive 28-day blocks")
        if vs - te != pd.Timedelta(days=1):
            raise ValueError("Training must end immediately before validation")
        if previous_test_end is not None and ss <= previous_test_end:
            raise ValueError("Test folds overlap or are out of order")
        if first_train_start is None:
            first_train_start = ts
        if ts != first_train_start or (previous_train_end is not None and te <= previous_train_end):
            raise ValueError("Training must expand from a fixed start date")
        if spec["fold_id"] in seen:
            raise ValueError("Duplicate fold ID")
        seen.add(spec["fold_id"])
        previous_test_end, previous_train_end = se, te
        parsed.append((spec["fold_id"], ranges))
    for fold_id, ranges in parsed:
        subsets = {p: data.loc[data["date"].between(*bounds)].copy() for p, bounds in ranges.items()}
        counts = pd.DataFrame([exclusion_counts(subsets[p], fold_id, p) for p in ["validation", "test"]])
        training_rows_excluded = int(subsets["train"]["store_zero_day"].eq(1).sum())
        subsets["train"] = subsets["train"].loc[subsets["train"]["store_zero_day"].eq(0)].copy()
        if subsets["train"].empty or any(not subsets[p]["score_mask"].any() for p in ["validation", "test"]):
            raise ValueError(f"Empty training or scored split in {fold_id}")
        yield Fold(fold_id, counts=counts, training_rows_excluded=training_rows_excluded, **subsets)
