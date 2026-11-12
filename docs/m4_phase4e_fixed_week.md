# Phase 4E.1 — fixed weekly paired uncertainty supplement

Status: **PROPOSED_NOT_APPROVED**. This document and the frozen request authorize
no official real-data bootstrap execution. Old approvals do not authorize this
supplement. No forecasting model, residual calibration or cost policy is refitted.

Reference: pinned `b35f5cc8f5dd2d07b0a545b6e5a60963afdff6bc`,
`notebooks/03_forecasting_ml/train_ml.ipynb`, code cell 17. The notebook bytes are
SHA-256 verified using the existing reproduction reference loader, never executed.
Shared YAML remains unchanged and authoritative. Main-estimator TODO, entry-point
layout, official-M3 substitution and joint-report acceptance are separate gaps.

## Frozen proposal

- Shared F1–F4 28-day test windows; fixed block ID = floor(calendar days since
  fold test-start / 7). This is not ISO-week grouping or moving-block sampling.
- `np.random.default_rng(42).integers(0,4,size=(1000,4,4))`, exactly the pinned M3
  RNG call. Same draw tensor for every policy, scenario, variant and eligible cell.
  Each fold samples four of its own weekly block IDs with replacement. Never
  concatenate calendar dates across gaps or folds into a block.
- Six contrasts: P3−P2, P4-M3−P4-M0, P2−P1, P3−P1, P4-M0−P1, P4-M3−P1.
- Three scenarios from frozen cost_scenarios.yaml. Verified stored costs only.
  Corrected is primary. Raw-clipped sensitivity changes **P2/P3 only**; P1/P4
  stay fixed, as in the existing source bundles. This is not raw-P4 calibration.
- Sample weekly **cost sums and observation counts together**. Mean differences
  divide by sampled counts. P1 indices and normalized differences divide by the
  sampled P1 total, not its original total or a mean of row-level ratios.
- Primary outputs: six mean differences, six P1-normalized differences, six
  relative differences (comparison-policy denominator; percent), four P1 indices.
  P1's self-index is identically one when defined and gets no inferential CI.
  Shared weather value is (P3−P2)/P1, **not** (P3−P2)/P2.
- Pooled plus per-fold intervals; family–city pooled intervals only for cells
  with exactly all 112 scored dates and four complete folds: 140 eligible cells.
  Seven Esmeraldas and seven Salinas cells retain point estimates, CI bounds NA
  and explicit incomplete-calendar flags. No imputation, resetting week IDs,
  dropping partial weeks or relabeling missing dates as complete weeks.
- Nominal 95% percentile CI, linear quantiles. Record requested/attempted/valid/
  undefined replicates, null-reference inclusion, below/above-null fractions and
  strict sign-change fraction. Null=1 for cost indices, 0 for differences.
  Sign-change is undefined if observed estimate equals its null. Zero denominator
  gives NA; finite-replicate CI reports how many replicates were excluded and is
  not presented as an unconditional coverage guarantee. Ineligible cells attempt
  zero resamples, distinct from attempted-but-undefined ratios.

## Scope and limitations

This copies M3 sampling, not its MAE estimand: negative cost differences indicate
improvement, whereas M3 MAE uplift uses the opposite sign. Per-fold/cell cost CI
and indices are explicit extensions. There are only four weekly blocks per fold;
1,000 draws do not create more temporal information. All within-day/store/family
observations travel together. Folds are fixed strata; cross-fold dependence and
dependence beyond the sampled week are not modeled. Observations are not iid.
Fixed policies exclude selection, calibration and fresh-refit uncertainty.
F1 is exposed/exploratory; F2–F4 prior exposure is unresolved. No multiple-testing
control, causal, operational, actual inventory-cost/profit/ROI or conformal claims.
Sales is a censored demand proxy; ERA5 availability remains retrospective.
P4 remains `M4_REPRODUCED_M3`. No test-based retuning or policy selection.

Existing moving-block bundle stays immutable and is never renamed as fixed-week
evidence. Existing matrix directions/recommendations and canonical exports stay
unchanged. Future approved CI can enter a versioned evidence supplement with its
own scope, denominators, calendar flags and source reference, not retroactive
preregistration or automatic recommendation changes.

## Integrity and approval

Request preparation verifies completed source bundles and historical hashes,
full matched frozen keys/targets, all scenarios, dates, fixed P1/P4 sensitivity,
finite stored costs and exact 140/14 calendar scope. It performs **no RNG draws**
or CI calculations. The request binds code/tests/docs/config/input/source hashes,
packages/Python, pinned notebook, scored-key digest and per-cell calendar support.

An independent approved receipt must contain:
`status=APPROVED`, the **unchanged exact request** in `request`, its semantic
SHA-256 in `request_sha256`, and truthful nonempty `approved_by`,
`approval_evidence`, `approval_communicated_date` (YYYY-MM-DD). Do not invent names
or signatures. The runner re-creates and compares the request, rejecting any
source, code, package, scope or hash drift before random sampling. Prior P1/P4/
forecasting approvals cannot substitute for this explicit supplement approval.

Preparation command (does not execute analysis):

```powershell
& '.\.venv\Scripts\python.exe' -B -m src.decision.fixed_week_bootstrap --prepare-request docs/m4_fixed_week_bootstrap.request.json
```

Only after explicit approval and a valid receipt:

```powershell
& '.\.venv\Scripts\python.exe' -B -m src.decision.fixed_week_bootstrap --approval docs/m4_fixed_week_bootstrap.approved.json
```

Runner writes a unique authorization record before draws, then uses exclusive
staging and unique `results/experiments/m4_fixed_week_supplement_*` finalization.
Outputs: intervals, point totals, cell calendar support, stored weekly totals,
shared draw tensor, STARTED/manifest/COMPLETE with checksums. Replicates are
reconstructible from weekly totals and the draw tensor; no massive per-cell
replicate CSV is required. Failed runs preserve diagnostics; no overwrite or
weakening of guards. Prior bundles and canonical files remain untouched.
