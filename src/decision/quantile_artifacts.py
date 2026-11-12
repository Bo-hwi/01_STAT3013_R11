"""M4 single-directory, checksummed artifact transactions (no external CSV)."""

import json
import re
from pathlib import Path
import tempfile

import pandas as pd

from .quantile_contracts import (
    TAU_KEYS, file_hash, json_snapshot, scored_key_digest,
    validate_prediction_panel, verify_frozen_files,
)


def output_paths(root, experiment_id):
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", experiment_id):
        raise ValueError("Unsafe experiment identifier")
    directory = Path(root) / "results/experiments" / f"m4_quantiles_{experiment_id}"
    return directory / f"m4_quantiles_{experiment_id}.csv", directory


def _validate_saved_panel(panel, manifest):
    if set(panel.experiment_id) != {manifest["experiment_id"]}:
        raise ValueError("Panel and manifest experiment identifiers disagree")
    configs = list(manifest["features"])
    validate_prediction_panel(panel, expected_configs=configs)
    contracts = manifest["scored_key_contract"]
    actual_splits = set(zip(panel.fold, panel.split))
    if actual_splits != {(r["fold"], r["split"]) for r in contracts}:
        raise ValueError("Panel fold/split set disagrees with frozen scored-key contract")
    for reference in contracts:
        for config in configs:
            rows = panel.loc[(panel.fold == reference["fold"]) & (panel.split == reference["split"])
                             & (panel.config == config) & (panel.tau_key == TAU_KEYS[0])]
            if len(rows) != reference["rows"] or scored_key_digest(rows) != reference["sha256"]:
                raise ValueError("Saved panel disagrees with frozen scored keys")


def inspect_run(root, experiment_id):
    """Detect absent, incomplete, corrupt or checksum-verified complete runs."""
    _, directory = output_paths(root, experiment_id)
    stages = list(directory.parent.glob(f".{directory.name}.staging-*"))
    if not directory.exists():
        return {"status": "incomplete" if stages else "absent", "staging": [str(p) for p in stages]}
    marker = directory / "COMPLETE.json"
    if not marker.is_file():
        return {"status": "incomplete", "reason": "completion marker missing"}
    try:
        record = json.loads(marker.read_text(encoding="utf-8"))
        if record["experiment_id"] != experiment_id or not record["files"]:
            raise ValueError("Invalid completion record")
        expected = {"manifest.json", "metrics.csv", "tuning.csv", "splits.csv", "exclusions.csv",
                    f"m4_quantiles_{experiment_id}.csv"}
        if set(record["files"]) != expected:
            raise ValueError("Completion file set mismatch")
        for name, digest in record["files"].items():
            if Path(name).name != name or file_hash(directory / name) != digest:
                raise ValueError("Output checksum mismatch")
        manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
        if manifest["output_sha256"] != {k: v for k, v in record["files"].items() if k != "manifest.json"}:
            raise ValueError("Manifest and completion hashes disagree")
    except (ValueError, KeyError, TypeError, OSError) as exc:
        return {"status": "corrupt", "reason": str(exc)}
    return {"status": "complete", "directory": str(directory)}


def _write_csv(path, table):
    with path.open("x", encoding="utf-8", newline="") as stream:
        table.to_csv(stream, index=False)


def save_run(predictions, metrics, diagnostics, root):
    """Prevalidate -> unique sibling staging -> verify -> rename final directory.

    Failures retain a detectable incomplete staging directory for diagnosis;
    they never create final artifacts. No automatic cleanup or overwrite.
    Directory rename is on the same filesystem. This is logical finalization,
    not a guarantee of durability against power loss or filesystem corruption.
    """
    from .quantile_pipeline import summarize_predictions

    manifest = json_snapshot(diagnostics["manifest"])
    experiment_id = manifest["experiment_id"]
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", experiment_id):
        raise ValueError("Unsafe experiment identifier")
    _validate_saved_panel(predictions, manifest)
    try:
        pd.testing.assert_frame_equal(metrics, summarize_predictions(predictions), check_dtype=False)
    except AssertionError as exc:
        raise ValueError("Metrics do not match validated predictions") from exc
    verify_frozen_files(manifest["provenance"])
    if inspect_run(root, experiment_id)["status"] != "absent":
        raise FileExistsError("Experiment exists or has incomplete staging; use a new ID or inspect it")
    destination, directory = output_paths(root, experiment_id)
    directory.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix=f".{directory.name}.staging-", dir=directory.parent))
    tables = {destination.name: predictions, "metrics.csv": metrics,
              **{f"{k}.csv": diagnostics[k] for k in ("tuning", "splits", "exclusions")}}
    for name, table in tables.items():
        _write_csv(stage / name, table)
        loaded = pd.read_csv(stage / name, float_precision="round_trip")
        if list(loaded) != list(table) or len(loaded) != len(table):
            raise ValueError(f"CSV validation failed: {name}")
    loaded_panel = pd.read_csv(stage / destination.name, parse_dates=["date", "forecast_origin"],
                               dtype={"store_nbr": str, "family": str, "city": str}, float_precision="round_trip")
    _validate_saved_panel(loaded_panel, manifest)
    hashes = {name: file_hash(stage / name) for name in tables}
    manifest["output_sha256"] = hashes
    with (stage / "manifest.json").open("x", encoding="utf-8") as stream:
        json.dump(manifest, stream, indent=2, allow_nan=False)
    hashes = {**hashes, "manifest.json": file_hash(stage / "manifest.json")}
    # Re-read bytes and recheck frozen evidence before marking complete.
    if any(file_hash(stage / name) != digest for name, digest in hashes.items()):
        raise ValueError("Staged output changed during validation")
    verify_frozen_files(manifest["provenance"])
    with (stage / "COMPLETE.json").open("x", encoding="utf-8") as stream:
        json.dump({"experiment_id": experiment_id, "files": hashes}, stream, indent=2, allow_nan=False)
    if directory.exists():
        raise FileExistsError("Final directory appeared during staging; refusing overwrite")
    # os.rename on Windows refuses an existing destination. A competing complete
    # run has a nonempty directory and cannot be replaced on POSIX either.
    stage.rename(directory)
    if inspect_run(root, experiment_id)["status"] != "complete":
        raise ValueError("Finalized run failed integrity verification")
    return directory
