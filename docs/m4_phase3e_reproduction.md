# Phase 3E.1: M4-owned point reproduction, fitting PENDING_APPROVAL

Label: **M4_REPRODUCED_M3**. No official M3 deliverable or exact numerical
reproduction claim. This implementation does not itself approve execution.

Source is pinned to Git commit `b35f5cc8f5dd2d07b0a545b6e5a60963afdff6bc`.
`p4_reproduction.py` verifies SHA-256 of the notebook and published logs, parses
only the original GRIDS assignment and preprocess/estimator/audit_imputer/
nonnegative_predict functions. It never runs notebook top-level experiments or
its output cells. No notebook integration/merge is required; Git objects must
remain available. Missing/mismatched objects fail rather than substitute models.

## Frozen proposed reproduction procedure

- All F1-F4 x M0/M3, seed 42, seven shared families, verified 34-column candidate.
- Original 3 Ridge + 2 ExtraTrees + 2 HGB candidates. Original feature whitelist,
  native categorical/ordinal/one-hot encodings, imputer, scaling and float32 numeric
  conversion. Shared `prepare_features()` audits lags before conversion.
- Shared `iter_folds()` only; closure exclusions, scored masks and fresh eligible
  train+validation refit stay unchanged. Numeric libraries use two threads.
- Select first minimum **clipped validation MAE** within each algorithm; select
  first minimum among Ridge, ExtraTrees, HistGradientBoosting, in that order.
  No tolerance-based ties or test-driven selection. Save all 56 candidate MAEs.
- Cache winning train-only validation forecasts at selection time. Fit only the
  winning algorithm fresh on refit rows for each fold/config. Budget: **56 + 8 =
  64 fits**; no hidden fit for extracting validation forecasts.
- Save full precision raw and clipped forecasts, model, config, fold, split,
  selected parameters, fit stage, keys, targets. Expected selected validation
  rows=82,320 and test rows=82,152 (two configs); authoritative counts come from
  shared masks/frozen contracts, never from hardcoded count-only acceptance.
- Compare every candidate MAE and selected hyperparameter against published logs,
  and selected-algorithm test MAE against its published summary. Report signed
  differences and selection changes; no forcing a published match. These
  comparisons never alter selection or frozen settings.

Material accepted approach differences requiring execution sign-off: M4 candidate
has a different physical hash from published M3 input; only local canonical versus
candidate semantic equivalence was verified. Python/NumPy/SciPy/PyArrow differ;
record installed versions. Added per-fold validation-only algorithm selection is
new. Only winner refits are reproduced. Exports retain full precision rather than
M3's four-decimal shared CSV. No bitwise/published-MAE match is promised. Synthetic
tests do not establish real-data performance or equality to published M3 numbers.

## Execution gate and bundles

Only after implementation review, prepare a new frozen request (no model fits):

```powershell
& '.\.venv\Scripts\python.exe' -B -m src.decision.p4_reproduction --prepare-request docs/m4_p4_reproduction.request.json
```

After actual team approval, a new receipt `docs/m4_p4_reproduction.approved.json`
must contain status APPROVED, the complete unchanged request object, its
`semantic_hash(request)` as request_sha256, actual approved_by/approval_evidence/
approval_communicated_date. No receipt is created or approval inferred here.

Only after separate execution authorization, the exact training command is:

```powershell
& '.\.venv\Scripts\python.exe' -B -m src.decision.p4_reproduction --approval docs/m4_p4_reproduction.approved.json
```

Before fitting: validate approval/code/config/input/environment bindings; candidate
marker; production preflight; original Git objects; completed frozen M4 quantile
reference; exact scored keys AND targets for both splits. Freeze evidence in
STARTED.json. Output goes to unique `results/experiments/m4_reproduced_m3_<UTC>_<UUID>/`.
Full-precision predictions.csv, tuning.csv, expected_rows.csv, published comparison
tables, manifest and completion checksums are round-trip verified. Failed staging
is retained with FAILED.json, never reused or treated as complete. No official
tables, canonical inputs or previous bundles are overwritten.

## Separate P4 calibration: PENDING_APPROVAL

Phase 3E.2a identifier fix: comparisons use isolated copies with store IDs represented
as canonical positive int64 decimal strings. Null/boolean/fractional/out-of-range
IDs and noncanonical text (whitespace, leading zeros, decimal/scientific notation)
fail. Duplicate normalized keys, missing/extra rows or target disagreements still
fail one-to-one matching. No training inputs, other keys or values are converted.
`prefit_validation()` is the same non-fitting path called by the approved runner.
For a read-only real-data gate check, with no authorization to fit:

```powershell
& '.\.venv\Scripts\python.exe' -B -m src.decision.p4_reproduction --prefit-check
```

The prior request/receipt remains historical evidence; code changes require a new
request and explicit approval before a retry. Do not overwrite the old records.

`p4_calibration.py` proposes signed `y_true - raw_validation_prediction`, pooling
fold x selected model x config x family, linear quantiles, minimum 30 scored
validation observations, fallback fold x selected model x config (also n>=30).
Q=max(0, **clipped fresh-refit test forecast** + validation residual quantile).
Retain fractions; use one selected pool for all taus; never use test residuals.

P1 approval explicitly covers only SeasonalNaive/M0, not this rule. Reproduction
approval also does not approve P4 calibration. `calibration_request(bundle)` binds
the completed reproduction and code; `apply_approved(receipt_path)` rejects pending,
stale or unrelated approvals and returns quantities/audit only after validation.
No P4 economic evaluation writer is supplied in this phase. Synthetic arithmetic
is private and all unapproved outputs are labeled PENDING_APPROVAL.

Validation is reused for hyperparameter/algorithm selection and calibration, so
calibration is not independent. Applying train-only residuals to a refitted model
creates a predictor mismatch. No conformal coverage, causal or operational
guarantee. F1 is exploratory/post-pilot; F2-F4 confirmation status remains pending.
