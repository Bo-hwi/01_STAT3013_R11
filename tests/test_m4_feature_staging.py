"""Synthetic feature regeneration only: no forecasting/model fitting."""

import json
from pathlib import Path
import shutil
from unittest.mock import patch

import pandas as pd
import pytest
import yaml

from test_m4_quantile_pipeline import contracts, features, small_thread_budget
from test_m4_production_readiness import delivery
from src.decision import feature_staging as fs
from src.decision.quantile_contracts import file_hash


@pytest.fixture
def prepared_delivery(delivery):
    root, path, protocol, ablation = delivery
    for relative in ["src/data/clean_data.py", "src/forecasting/rolling_validation.py",
                     "src/decision/quantile_preflight.py", "src/decision/quantile_contracts.py",
                     "src/forecasting/point_models.py", "src/forecasting/feature_spec.py"]:
        target = root / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(fs.ROOT / relative, target)
    (root / "configs").mkdir()
    for name, value in [("research_protocol.yaml", protocol), ("ablation_M0_M5.yaml", ablation)]:
        (root / "configs" / name).write_text(yaml.safe_dump(value), encoding="utf-8")
    candidate = fs.m1.build_features(pd.read_parquet(path.parent / "curated_sales_weather.parquet"),
                                     weather_dir=root / "data/raw/weather")
    old = candidate.assign(excluded_post_reopening=False)
    old.to_parquet(path, index=False)
    path.with_suffix(".meta.json").write_text(json.dumps(fs.collect_m1_metadata(old)), encoding="utf-8")
    return root, path, candidate


def test_metadata_matches_actual_m1_writer(prepared_delivery):
    _, _, frame = prepared_delivery
    collected = fs.collect_m1_metadata(frame)
    # Intercept the IO method only; never redirect/patch M1 globals or write its canonical path.
    with patch.object(Path, "write_text") as write:
        fs.m1.write_metadata(frame)
    actual = json.loads(write.call_args.args[0])
    actual.pop("created_at_utc")
    collected.pop("created_at_utc")
    assert actual == collected
    assert collected["n_columns"] == 34


def test_unknown_metadata_writer_fails_closed():
    with pytest.raises(fs.MetadataContractError):
        fs.metadata_expression("def write_metadata(df):\n    return {}\n")


@pytest.mark.parametrize("location", ["outside", "canonical"])
def test_unsafe_output_parent_rejected_before_writes(tmp_path, location):
    parent = tmp_path / ("elsewhere" if location == "outside" else "data/processed")
    with pytest.raises(ValueError, match="dedicated subdirectory"):
        fs.regenerate_candidate(tmp_path, parent)
    assert not parent.exists()


def test_exclusive_writers_preserve_existing_bytes(tmp_path):
    target = tmp_path / "existing.json"
    fs._exclusive_json(target, {"old": True})
    before = target.read_bytes()
    with pytest.raises(FileExistsError):
        fs._exclusive_json(target, {"new": True})
    with pytest.raises(FileExistsError):
        fs._verified_copy(target, target)
    assert target.read_bytes() == before


@pytest.mark.parametrize("fault", ["extra", "missing", "order"])
def test_schema_requires_exact_ordered_contract(prepared_delivery, fault):
    _, _, frame = prepared_delivery
    frame = (frame.assign(excluded_post_reopening=False) if fault == "extra" else
             frame.drop(columns="sales_lag7") if fault == "missing" else frame[frame.columns[::-1]])
    with pytest.raises(ValueError, match="34-column"):
        fs.validate_candidate_schema(frame)


def test_exact_comparison_detects_values_missingness_and_dtype(prepared_delivery):
    _, _, frame = prepared_delivery
    other = frame.copy()
    other.loc[50, "sales_lag7"] = -123.
    other.loc[100, "sales_roll7_mean"] = float("nan")
    other["store_nbr"] = other.store_nbr.astype("int64")
    result = fs.compare_features(frame, other)
    assert result["exact_keys"] is False  # dtype belongs to the exact key contract
    other["store_nbr"] = frame.store_nbr
    result = fs.compare_features(frame, other)
    assert result["exact_keys"]
    assert not result["columns"]["sales_lag7"]["exact"]
    assert not result["columns"]["sales_roll7_mean"]["nan_pattern_equal"]


def test_staging_full_preflight_and_unique_nonoverwrite(prepared_delivery):
    root, path, frame = prepared_delivery
    original = {p: file_hash(p) for p in path.parent.iterdir() if p.is_file()}
    first, report = fs.regenerate_candidate(root)
    assert report["recommendation"] == "GO_FOR_ADOPTION_REVIEW"
    assert report["preflight_status"] == "PASS"
    assert report["comparison"]["all_shared_columns_exact"]
    assert report["comparison"]["existing_only_columns"] == ["excluded_post_reopening"]
    assert report["original_inputs_unchanged"] and report["code_configs_unchanged"]
    marker = json.loads((first / "CANDIDATE_COMPLETE.json").read_text())
    assert marker["report_sha256"] == file_hash(first / "reconciliation_report.json")
    second, _ = fs.regenerate_candidate(root)
    assert first != second
    assert all(file_hash(p) == digest for p, digest in original.items())
    assert file_hash(first / "curated_sales_weather.parquet") == original[path.parent / "curated_sales_weather.parquet"]


def test_failed_copy_leaves_marked_incomplete_stage(prepared_delivery, monkeypatch):
    root, path, _ = prepared_delivery
    digest = file_hash(path)

    def fail_copy(*args):
        raise OSError("synthetic disk failure")

    monkeypatch.setattr(fs, "_verified_copy", fail_copy)
    with pytest.raises(OSError, match="synthetic disk failure"):
        fs.regenerate_candidate(root)
    stage, = (path.parent / "m4_feature_candidates").iterdir()
    assert (stage / "FAILED.json").is_file()
    assert not (stage / "CANDIDATE_COMPLETE.json").exists()
    assert file_hash(path) == digest


def test_preflight_failure_prevents_adoption(prepared_delivery):
    root, path, _ = prepared_delivery
    meta_path = path.parent / "curated_sales_weather.meta.json"
    meta = json.loads(meta_path.read_text())
    meta["n_rows"] += 1
    meta_path.write_text(json.dumps(meta), encoding="utf-8")
    _, report = fs.regenerate_candidate(root)
    assert report["recommendation"] == "NO_GO"
    assert report["preflight_status"] == "HARD_FAILURE"
    assert report["preflight"]["hard_failures"]


def test_missing_input_rejected_before_staging(prepared_delivery):
    root, path, _ = prepared_delivery
    path.with_suffix(".meta.json").unlink()  # Synthetic fixture only.
    with pytest.raises(FileNotFoundError):
        fs.regenerate_candidate(root)
    assert not (path.parent / "m4_feature_candidates").exists()


def test_reordered_rows_align_exactly_and_duplicate_keys_fail(prepared_delivery):
    _, _, frame = prepared_delivery
    shuffled = frame.iloc[::-1].reset_index(drop=True)
    result = fs.compare_features(frame, shuffled)
    assert result["exact_keys"] and result["all_shared_columns_exact"]
    assert not result["original_key_order_equal"]
    with pytest.raises(ValueError, match="duplicate/null keys"):
        fs.compare_features(frame, pd.concat([frame, frame.iloc[:1]], ignore_index=True))
