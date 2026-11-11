# M4 Phase 2A — direct quantile protocol

This is an implementation specification, not an experiment result or an amendment
to the frozen shared YAML. Real runs require M1's full `model_features.parquet`.
Synthetic tests are isolated from the repository's research output directories.

## Selection and refit (declared before test evaluation)

1. Read the frozen seven-family shortlist and three taus from research_protocol.
   Normalize tau through Phase 1.1, and use canonical string tau keys for joins.
2. Audit M1 sales lags/rolling with M3's `prepare_features(ready=True)`. Do this
   on source history before filtering; no sales dates are synthesized or filled.
3. Use shared `iter_folds()` unchanged: active rows only, closure targets excluded
   from fitting, shared score_mask for validation/test, cooldown retained in fit.
4. Fit separate global HGB quantile models for each fold/config/tau. M0 uses
   `m0_numeric()` plus shared identifiers; M3 adds exactly the two core weather
   lag1 columns. Promotion, same-day weather, transactions and mask/closure flags
   are never explanatory variables. Missing weather does not change scored keys.
5. Each candidate gets a fresh Pipeline: numeric median imputation (all-missing
   columns retained as zero), plus ordinal categorical encoding. HGB receives
   explicit native-categorical flags. Unknown categories become missing, not
   ordered distances. No scaling or one-hot expansion is needed.
6. Disable internal early stopping. Select by mean RAW pinball on scored shared
   validation, independently per config/tau/fold. Both configs receive the same
   candidate grid and seed; exact ties keep the first candidate. Default grid:
   max_iter=100, learning_rate=0.05, min_samples_leaf=20, max_leaf_nodes in [15,31].
7. Complete all six selections for a fold before any test prediction. Keep the
   selected train-only validation predictions as selection diagnostics. They are
   not independent calibration samples or unbiased held-out performance estimates.
8. Refit NEW pipelines on `Fold.refit_rows()` (eligible train+validation), then
   predict test with fixed parameters. Observed earlier test-day sales may feed
   the next day's audited lags, per the shared one-day rolling protocol. No test
   labels enter selection/refitting. This is not a 28-day fixed-origin forecast.
9. Preserve raw predictions. For each observation triplet, clip at zero and sort
   into increasing canonical tau order. This correction uses no labels and is
   not calibration. Evaluate both raw and corrected variants separately.

## Outputs

The long prediction schema includes date/store/family/city, fold, config, tau,
tau_key, y_true, raw_prediction, corrected_prediction, crossing flags, split,
fit_stage, model, experiment_id, forecast_origin and horizon_days.

Reports include mean pinball, P(y<=q), P(y<q), tie rate, signed coverage gap,
sample size, any-crossing-per-triplet rate and negative prediction rate. Report
fold and family-city levels separately; do not treat rows as independent time
samples for uncertainty. Calibration/paired bootstrap/cost policies are later work.

CLI (only when real inputs and experimental design are approved):

```powershell
.\.venv\Scripts\python.exe -m src.decision.quantile_pipeline --experiment-id YOUR_UNIQUE_ID
```

All finalized outputs now live together under
`results/experiments/m4_quantiles_<experiment_id>/`: prediction CSV named
`m4_quantiles_<experiment_id>.csv`, metrics/tuning/split/exclusion CSVs,
manifest.json and COMPLETE.json. There is no separate CSV published into
results/predictions: keeping one directory enables a same-filesystem finalize.
Existing paths are never overwritten. No M1/M3 artifact names are reused.
The in-memory training API still returns results without writing files.

## Phase 2A.1 production data gate

The CLI requires model_features.parquet, its .meta.json, curated_sales_weather.parquet,
its .meta.json and docs/source_catalog.csv. The synthetic training API does not
impose these delivery requirements. Counts/hashes are taken from existing evidence,
not guessed or generated as expected research values.

Hard failures stop before fitting: missing required evidence, metadata/catalog
conflicts, builder-schema violations, duplicate/null/missing sales keys, truncated
coverage, source scope mismatches, changed source bytes, invalid weather/grid
values, inconsistent sales-derived store flags, supplied lag/mask disagreement,
or mismatch with available M1 baseline scored counts. The date calendar excludes
December 25 according to M1's existing contract. Numeric weather ranges are QA
screens, not proof of measurement accuracy. Temperature outside M1's screening
range is a warning requiring investigation rather than automatic data deletion.

The preflight report separates hard_failures, warnings, unavailable and verified.
Existing M1 metadata does NOT contain an upstream-approved processed-file hash.
Locally computing such a hash binds a run to bytes; it cannot authenticate them.
If raw files are absent, source byte checks and independent weather-lag
reconstruction are marked unavailable. If raw weather is available, city/date
lag values are reconstructed. Independent per-fold validation counts are not
published by current M1 metadata; shared-rule counts are reported without
claiming independent verification. An existing baseline count table is a
historical reference without input-hash binding. ERA5 publication timestamps,
external coordinate provenance and inventory/stockouts remain unverified.

## Frozen provenance, panel validation and finalization

Before fitting, freeze detached effective config snapshots, semantic hashes,
raw code/config byte hashes, package versions, candidate grid, seed, available
input evidence and per-fold/split scored-key digests. Shared iter_folds is used
for both the key-only pre-fit pass and the training pass; no splitter is added.
Only key tables are retained from the first pass, not all expanding train frames.
Files that supplied frozen evidence are checked again before returning/saving.
In-memory runs explicitly record unverified upstream provenance.

validate_prediction_panel is the required gate before metrics and future cost
integration. It enforces non-null required keys/values, unique keys, three exact
canonical tau keys per observation/config, matched configs and truth, finite
predictions, clip/sort correction, crossing flags and one-day origin if supplied.
Pass expected_rows to detect observations missing from every config; the runner
does this using scored shared-fold identities. The writer rechecks frozen key
digests. Array-only Newsvendor primitives still require caller key alignment.

Mean pinball uses ordinary arithmetic when safe and a scaled fallback if the
sum overflows. Crossing uses direct adjacent comparisons, not subtraction.

save_run prevalidates JSON and metrics, writes exclusively into a unique sibling
staging directory, reloads/validates CSVs, computes output checksums, writes the
manifest and completion marker, and renames to the non-existing final directory.
The manifest hashes all output CSVs; COMPLETE.json hashes those files plus the
manifest, avoiding a circular self-hash. inspect_run reports absent, incomplete,
corrupt or complete after checking hashes. Failures retain staging for diagnosis,
never publish a partial final directory, and block reuse of that ID. There is no
automatic cleanup/recovery. Finalization is not a power-loss durability guarantee.

Synthetic writer/delivery tests use pytest temporary directories only. Four-fold
tests use small panels and tiny iteration counts, never research conclusions.

## Decisions to ratify before real experiments

- HGB native categorical preprocessing and the proposed two-candidate budget.
- Selection on raw pinball followed by post-hoc clipping/sorting; corrected
  validation performance does not reselect parameters.
- No calibration in Phase 2A. Later calibration needs an agreed chronological
  design compatible with train+validation refit; do not repurpose selected
  validation residuals as independent calibration evidence.
- Weather remains retrospective ERA5. Precomputed weather-lag availability and
  source QA are inherited from M1, not reverified without raw weather data.
- Shared YAML still has a TODO for primary_quantile_model; this implementation
  deliberately leaves shared configuration unchanged for team review.
