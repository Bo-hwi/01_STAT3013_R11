"""Controlled M4 staging adapter for the unmodified M1 feature builder.

No training or canonical writes. CLI: python -m src.decision.feature_staging
Candidates are never automatically adopted, moved or deleted.
"""

import ast
from datetime import datetime, timezone
import gc
import inspect
from importlib.metadata import version
import json
from pathlib import Path
import platform
import shutil
import tempfile
from time import perf_counter
import warnings

import pandas as pd

from src.features import lag_features as m1
from src.forecasting.rolling_validation import read_yaml
from .quantile_contracts import file_hash
from .quantile_preflight import production_preflight, ProductionPreflightError


ROOT = Path(__file__).resolve().parents[2]


class MetadataContractError(ValueError):
    pass


def metadata_expression(source=None):
    """Extract ONLY M1's metadata dict, never its fixed-path write statement.

    Fail closed if the writer stops being exactly dict assignment / write /
    return. This reuses the actual M1 expression rather than copying a schema
    that can drift. No M1 module global is patched or reassigned.
    """
    source = inspect.getsource(m1.write_metadata) if source is None else source
    tree = ast.parse(source)
    if len(tree.body) != 1 or not isinstance(tree.body[0], ast.FunctionDef):
        raise MetadataContractError("Unsupported M1 metadata writer structure")
    body = tree.body[0].body
    valid = (
        len(body) == 3 and isinstance(body[0], ast.Assign)
        and len(body[0].targets) == 1 and isinstance(body[0].targets[0], ast.Name)
        and body[0].targets[0].id == "meta" and isinstance(body[0].value, ast.Dict)
        and isinstance(body[1], ast.Expr) and isinstance(body[1].value, ast.Call)
        and isinstance(body[1].value.func, ast.Attribute)
        and isinstance(body[1].value.func.value, ast.Name)
        and body[1].value.func.value.id == "META_PATH"
        and body[1].value.func.attr == "write_text"
        and isinstance(body[2], ast.Return) and isinstance(body[2].value, ast.Name)
        and body[2].value.id == "meta"
    )
    if not valid:
        raise MetadataContractError("M1 metadata writer changed; review adapter before generating artifacts")
    return ast.Expression(body=body[0].value)


def collect_m1_metadata(frame):
    expression = metadata_expression()
    namespace = dict(vars(m1))
    namespace["df"] = frame
    # Execute the inspected dict expression only. The write statement is not
    # compiled/executed, so META_PATH never participates in output selection.
    result = eval(compile(expression, "<M1 metadata dictionary only>", "eval"), namespace)
    return json.loads(json.dumps(result, default=str, allow_nan=False))


def expected_columns():
    return (m1.PASSTHROUGH + list(m1.SALES_LAGS.values()) + list(m1.SALES_ROLLING.values())
            + list(m1.WEATHER_LAG1.values()) + ["score_mask"])


def validate_candidate_schema(frame):
    expected = expected_columns()
    if len(expected) != 34 or len(set(expected)) != 34 or list(frame) != expected:
        raise ValueError("Candidate must match the ordered 34-column current M1 contract")
    m1.validate_features(frame, len(frame))


def compare_features(candidate, existing):
    """Compare all shared values EXACTLY, after key alignment; no repair/tolerance."""
    validate_candidate_schema(candidate)
    if not set(candidate).issubset(existing):
        raise ValueError("Existing artifact lacks candidate columns")
    for label, frame in (("candidate", candidate), ("existing", existing)):
        if frame[m1.KEY].isna().any().any() or frame.duplicated(m1.KEY).any():
            raise ValueError(f"{label} contains duplicate/null keys")
    same_order = candidate[m1.KEY].equals(existing[m1.KEY])
    left, right = candidate, existing
    if not same_order:
        left = candidate.sort_values(m1.KEY).reset_index(drop=True)
        right = existing.sort_values(m1.KEY).reset_index(drop=True)
    keys_exact = left[m1.KEY].equals(right[m1.KEY])
    result = {"candidate_rows": len(left), "existing_rows": len(right), "exact_keys": keys_exact,
              "original_key_order_equal": same_order,
              "existing_only_columns": sorted(set(existing) - set(candidate)), "columns": {}}
    if not keys_exact:
        result["all_shared_columns_exact"] = False
        return result
    for column in candidate:
        a, b = left[column].reset_index(drop=True), right[column].reset_index(drop=True)
        dtype_equal = pd.api.types.is_dtype_equal(a.dtype, b.dtype)
        missing_equal = bool(a.isna().equals(b.isna()))
        try:
            pd.testing.assert_series_equal(a, b, check_exact=True, check_dtype=True, check_categorical=True)
            exact = True
            discrepancy = None
        except AssertionError as exc:
            exact, discrepancy = False, str(exc)[:1000]
        result["columns"][column] = {
            "exact": exact, "dtype_equal": dtype_equal, "candidate_dtype": str(a.dtype),
            "existing_dtype": str(b.dtype), "nan_pattern_equal": missing_equal,
            "candidate_nulls": int(a.isna().sum()), "existing_nulls": int(b.isna().sum()),
            "discrepancy": discrepancy,
        }
    result["all_shared_columns_exact"] = all(c["exact"] for c in result["columns"].values())
    return result


def _exclusive_json(path, value):
    serialized = json.dumps(value, indent=2, allow_nan=False)
    with Path(path).open("x", encoding="utf-8") as stream:
        stream.write(serialized)


def _verified_copy(source, destination):
    with Path(source).open("rb") as src, Path(destination).open("xb") as dst:
        shutil.copyfileobj(src, dst, length=1024 * 1024)
    digest = file_hash(source)
    if file_hash(destination) != digest:
        raise ValueError("Staged curated evidence copy failed hash verification")
    return digest


def _resources():
    try:
        import psutil
        process = psutil.Process()
        memory, cpu = process.memory_info(), process.cpu_times()
        return {"rss_bytes": memory.rss, "process_lifetime_peak_working_set_bytes": getattr(memory, "peak_wset", None),
                "cpu_user_seconds": cpu.user, "cpu_system_seconds": cpu.system,
                "note": "Windows peak is process-lifetime, not an isolated builder increment"}
    except (ImportError, OSError):
        return {"measurement": "unavailable"}


def regenerate_candidate(root=ROOT, staging_parent=None):
    """Build/compare/preflight in a unique directory; leave originals untouched."""
    root = Path(root).resolve()
    processed = (root / "data/processed").resolve()
    parent = (processed / "m4_feature_candidates" if staging_parent is None else Path(staging_parent)).resolve()
    if not parent.is_relative_to(processed) or parent == processed:
        raise ValueError("Staging parent must be a dedicated subdirectory inside data/processed")
    metadata_expression()  # Check metadata compatibility before any filesystem write.
    names = ["curated_sales_weather.parquet", "curated_sales_weather.meta.json",
             "model_features.parquet", "model_features.meta.json"]
    inputs = [processed / name for name in names]
    inputs += sorted((root / "data/raw").rglob("*.csv"))
    code_paths = [Path(inspect.getfile(m1)).resolve(), Path(__file__).resolve(),
                  root / "src/data/clean_data.py", root / "src/forecasting/rolling_validation.py",
                  root / "src/decision/quantile_preflight.py", root / "src/forecasting/point_models.py",
                  root / "src/decision/quantile_contracts.py",
                  root / "src/forecasting/feature_spec.py", root / "configs/research_protocol.yaml",
                  root / "configs/ablation_M0_M5.yaml", root / "docs/source_catalog.csv"]
    reference = root / "results/tables/baselines_metrics.csv"
    if reference.is_file():
        inputs.append(reference)
    for path in inputs + code_paths:
        if not path.is_file():
            raise FileNotFoundError(f"Required input/provenance file missing: {path}")
    if not list((root / "data/raw/weather").glob("weather_*.csv")):
        raise FileNotFoundError("Raw per-city weather files are required for regeneration")
    start = perf_counter()
    original_hashes = {str(p): file_hash(p) for p in inputs}
    code_hashes = {str(p): file_hash(p) for p in code_paths}
    parent.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix="candidate-", dir=parent))
    report = {"stage": str(stage), "created_at_utc": datetime.now(timezone.utc).isoformat(),
              "original_input_sha256": original_hashes, "code_config_sha256": code_hashes,
              "metadata_contract": "dictionary expression extracted from unmodified M1 write_metadata; no writer call/global override",
              "environment": {"python": platform.python_version(), "platform": platform.platform(),
                              "packages": {name: version(name) for name in ["numpy", "pandas", "pyarrow", "PyYAML"]}},
              "timings_seconds": {}, "resources_before": _resources()}
    try:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            tick = perf_counter()
            curated = pd.read_parquet(processed / names[0])
            print("Regenerating candidate via M1 build_features()", flush=True)
            candidate = m1.build_features(curated, weather_dir=root / "data/raw/weather")
            del curated
            gc.collect()
            validate_candidate_schema(candidate)
            report["timings_seconds"]["read_and_build"] = perf_counter() - tick
            tick = perf_counter()
            meta = collect_m1_metadata(candidate)
            candidate_path = stage / "model_features.parquet"
            with candidate_path.open("xb") as stream:
                candidate.to_parquet(stream, index=False)
            _exclusive_json(stage / "model_features.meta.json", meta)
            report["schema"] = {name: str(dtype) for name, dtype in candidate.dtypes.items()}
            report["column_order"] = list(candidate)
            del candidate
            gc.collect()
            report["timings_seconds"]["write_candidate"] = perf_counter() - tick
            tick = perf_counter()
            # Copies are evidence for read-only preflight consumption. Their
            # hashes are rechecked; no hardlinks/symlinks to canonical inputs.
            report["staged_curated_sha256"] = {
                name: _verified_copy(processed / name, stage / name) for name in names[:2]
            }
            report["timings_seconds"]["copy_curated_evidence"] = perf_counter() - tick
            tick = perf_counter()
            candidate = pd.read_parquet(candidate_path)
            existing = pd.read_parquet(processed / "model_features.parquet")
            report["comparison"] = compare_features(candidate, existing)
            existing_meta = json.loads((processed / "model_features.meta.json").read_text(encoding="utf-8"))
            report["metadata_differences"] = {k: {"existing": existing_meta.get(k), "candidate": meta.get(k)}
                                               for k in sorted(set(meta) | set(existing_meta))
                                               if existing_meta.get(k) != meta.get(k)}
            del candidate, existing
            gc.collect()
            report["timings_seconds"]["exact_comparison"] = perf_counter() - tick
            tick = perf_counter()
            print("Running unchanged production_preflight() on staged candidate", flush=True)
            protocol = read_yaml(root / "configs/research_protocol.yaml")
            ablation = read_yaml(root / "configs/ablation_M0_M5.yaml")
            try:
                audited, preflight = production_preflight(candidate_path, protocol, ablation, root)
                report["preflight_status"] = "PASS"
                del audited
            except ProductionPreflightError as exc:
                preflight = exc.report
                report["preflight_status"] = "HARD_FAILURE"
            report["preflight"] = preflight
            gc.collect()
            report["timings_seconds"]["production_preflight"] = perf_counter() - tick
            report["runtime_warnings"] = sorted(set(str(w.message) for w in caught))
        report["original_inputs_unchanged"] = all(file_hash(name) == h for name, h in original_hashes.items())
        report["code_configs_unchanged"] = all(file_hash(name) == h for name, h in code_hashes.items())
        if not report["original_inputs_unchanged"] or not report["code_configs_unchanged"]:
            raise ValueError("Original input or code/config changed during candidate generation")
        report["output_sha256"] = {p.name: file_hash(p) for p in stage.iterdir() if p.is_file()}
        report["recommendation"] = "GO_FOR_ADOPTION_REVIEW" if (
            report["comparison"]["all_shared_columns_exact"] and report["preflight_status"] == "PASS"
            and set(report["comparison"]["existing_only_columns"]) <= {"excluded_post_reopening"}
        ) else "NO_GO"
        report["resources_after"] = _resources()
        report["timings_seconds"]["total_before_report_write"] = perf_counter() - start
        _exclusive_json(stage / "reconciliation_report.json", report)
        _exclusive_json(stage / "CANDIDATE_COMPLETE.json", {"report_sha256": file_hash(stage / "reconciliation_report.json"),
                                                           "adopted": False})
        return stage, report
    except Exception as exc:
        # Never remove partial staging; make failure explicit when possible.
        failure = {"exception": type(exc).__name__, "message": str(exc), "adopted": False,
                   "stage": str(stage), "elapsed_seconds": perf_counter() - start}
        try:
            _exclusive_json(stage / "FAILED.json", failure)
        except OSError:
            pass
        raise


def main():
    stage, report = regenerate_candidate()
    print(json.dumps({"stage": str(stage), "recommendation": report["recommendation"],
                      "preflight_status": report["preflight_status"], "timings_seconds": report["timings_seconds"],
                      "resources": report["resources_after"]}, indent=2))


if __name__ == "__main__":
    main()
