# M4 Phase 2B.0: scoped pilot execution path

Status: implemented and tested with synthetic fixtures only. Real-data training
requires separate user approval. PILOT / NOT FINAL RESEARCH RESULTS.

## Shared pipeline API

`run_direct_quantiles(..., fold_ids=None, configs=None)` preserves the original
defaults: all folds in the shared protocol, both M0/M3, and the existing grid.
Explicit selections must be nonempty, unique, supported string IDs; scalar
strings, duplicates and unknown IDs fail before fitting. Execution follows
shared schedule/config order. `_scoped_folds()` filters shared `iter_folds()`
objects and stops after the last requested fold, without constructing a new
splitter or changing shared dates, eligibility, cooldown or closure rules.

The shared schedule is validated by `iter_folds()` before its first yield.
The original protocol/ablation snapshots remain in the manifest, with selected
fold definitions and `selected_scope` recorded separately. Both scored-key
collection and fitting use the same selection. Panel validation expects exactly
the selected configs and scored observations.

## Locked pilot entry point

After separate approval, run from the repository root in PowerShell:

```powershell
& '.\.venv\Scripts\python.exe' -B -m scripts.m4_pilot
```

The default experiment ID contains UTC time and a random suffix. An optional
`--experiment-id` may supply another safe unique ID; existing or incomplete runs
are refused. `--help` never trains. There are no CLI overrides for the data path,
fold, config, tau or candidate grid.

Fixed input: `data/processed/m4_feature_candidates/candidate-t77rmmvr/model_features.parquet`.
Expected SHA-256: `e4186aeb3a2da9996dcb9deec164cc70ca11c0a869b932720c9786c64a3ab17d`.
The entry point verifies the reconciliation report digest/completion marker,
four staged evidence digests, reconciliation status, absence of `FAILED.json`
and the approved feature digest. It then calls unchanged production preflight;
any hard failure aborts before pipeline invocation. Evidence is checked again
after preflight against the original digests, not a recomputed baseline.

Locked scope: F1, M0, canonical tau 0.50/0.75/5/6, frozen seven families, all
eligible actual train rows and all scored validation/test rows. No downsampling.

| F1 split | Inclusive dates |
|---|---|
| train | 2013-01-01 to 2016-08-03 |
| validation | 2016-08-04 to 2016-08-31 |
| test | 2016-09-01 to 2016-09-28 |

One independent pilot-only candidate: `max_iter=100`, `max_leaf_nodes=15`,
`min_samples_leaf=20`, `learning_rate=0.05`. Shared grids/config files are untouched.
Three train-only fits use raw scored validation pinball (one candidate per tau),
followed by three fresh train-plus-validation refits: six total model fits.
Test targets never enter selection/fitting; daily observed past sales remain
permitted lag inputs for the shared one-day-ahead evaluation.

All preprocessing, selection/refitting, clipping/sorting, metrics, panel gates
and transactional saving remain in the existing M4 pipeline. The entry point
does not implement any of those operations itself.

## Frozen evidence and outputs

Before fitting, the pipeline detaches the run specification, including scope,
candidate grid, seed, full contract snapshots, package versions, scored-key
digests and code/config/input digests. `run_spec_sha256` hashes that pre-fit
manifest before adding the digest field or finalized output hashes. The pilot
entry-point source is included in code provenance. Input evidence also binds
the staging report and its completion marker. Frozen files are rechecked before
fitting, after fitting and by the existing artifact transaction.

Pilot manifests carry `pilot_only: true` and result label
`PILOT / NOT FINAL RESEARCH RESULTS`; validation remains selection diagnostics,
not an independent calibration set. No calibration is performed.

`save_run()` writes a unique experiment bundle under
`results/experiments/m4_quantiles_<experiment_id>/`, retaining its existing
non-overwrite, staging, panel, metric and checksum checks. The entry point
requires `inspect_run()` to report `complete` after saving. Canonical M1 files
and shared prediction handoff paths are never written.

Tests in `tests/test_m4_pilot.py` use generated fixtures, including synthetic
source/metadata bundles and tiny model fits. They do not train on local real
Parquet data and their scores are not research findings. Existing full F1–F4
synthetic integration tests continue to verify the default full-run behavior.
