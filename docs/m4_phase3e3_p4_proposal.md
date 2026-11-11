# Phase 3E.3 P4 clipped-residual proposal — PENDING_APPROVAL

This is a new proposal, not authorization to calibrate/evaluate real-data P4.
The historical `p4_calibration.py` and `m4_phase3e_reproduction.md` remain unchanged
because their hashes are bound into the completed reproduction's provenance.
The active proposed implementation is `p4_policy_evaluation.py`.

## Explicit change from the earlier P4 proposal

Previously: residual = y_true minus **raw** validation prediction, with declared
fold/model/config/family pooling. Now: residual = y_true minus **clipped** validation
prediction, pooling **fold/config/family**. This aligns the residual predictor with
the nonnegative point rule used for validation MAE selection and test base. It can
change residual quantiles; no real-data P4 quantiles/costs have been calculated to
choose this change. It follows the user's explicit proposed specification.

The selected model is unique per fold/config and is preserved as audit metadata.
The reused residual primitive retains model as an invariant guard; it does not
split any declared fold/config group further. Conflicting model mappings fail.

## Proposed frozen method

- F1-F4 x M0/M3; use only the completed M4_REPRODUCED_M3 point bundle.
- r = y_true - clipped_validation_prediction, from scored **train-only** forecasts.
- Group by fold/config/family; empirical quantile uses **linear** interpolation.
- Minimum **30 observations**; sparse/unseen family fallback is fold/config with
  the same minimum. Fail if the fallback is insufficient. One pool for all taus.
- Base = **clipped fresh train+validation refit test prediction**.
- Q=max(0, base + validation residual quantile); retain fractional quantities.
- Preserve selected algorithm/parameters/validation MAE, declared train/refit
  cutoffs, validation/test windows and fit stages. Never reselect from test data.
- Shared YAML S1/S3/S5 scenarios, canonical 0.50/0.75/5/6; exactly matched 41,076
  frozen test observations per scenario and config. Fixed P1/P2/P3 costs remain
  unchanged. P2/P3 corrected and raw-clipped sensitivities are separate rows;
  P4 uses the single proposed calibration rule in both comparisons.
- Cost indices = sums(policy costs)/sums(P1 costs); zero P1 denominator => NaN.
  Report pooled/fold/family/city/cell summaries and over/understock quantities.

## Evidence and safeguards

Request preparation verifies all three source completion markers/checksums,
historical approval digests and frozen source/input/config hashes. Exact point
keys/targets and train-only/fresh-refit stages are checked, as are fixed-policy
scenario keys and truth. No new P4 residual quantile or cost is computed while
preparing the request. Historical approval validation uses its recorded hash map,
not a newly expanded directory glob; no bound hashes are ignored or rewritten.

The reproduction source is:
`results/experiments/m4_reproduced_m3_20261108T200231Z_d1e59483f28f4c01a76f02f972367216/`.
Fixed references are the completed `m4_p1_20261108T173741Z_4d68ed38ce2f4d1299d1e464d47dfa64`
and `m4_costs_p2_p3_20261108T134955Z_b112ad3e7866` bundles.

## Approval and eventual execution

Prepare an exclusively created new request, without official calibration:

```powershell
& '.\.venv\Scripts\python.exe' -B -m src.decision.p4_policy_evaluation --prepare-request docs/m4_p4_calibration.clipped.request.json
```

After actual explicit approval, create a NEW receipt containing status APPROVED,
the exact unchanged request object, request_sha256=semantic_hash(request), actual
approved_by, approval_evidence and approval_communicated_date. Neither P1 nor
reproduction approval covers this changed P4 method. No approved receipt is
generated in this preparation phase.

Only after approval and execution authorization:

```powershell
& '.\.venv\Scripts\python.exe' -B -m src.decision.p4_policy_evaluation --approval docs/m4_p4_calibration.clipped.approved.json
```

The runner rechecks approval/bindings, then writes a new unique staged
`results/experiments/m4_p4_costs_<UTC>_<UUID>/` bundle: P4 rows, calibration audit,
matched costs, comparisons, STARTED/manifest/COMPLETE checksums. It round-trips
CSVs and preserves failed staging diagnostics; previous outputs never overwrite.

## Limitations requiring explicit acknowledgement

Validation is reused for hyperparameter AND algorithm selection and residual
calibration; it is not an independent calibration sample. Refitting changes the
predictor underlying validation residuals, so coverage/cost improvement is not
guaranteed. Observations are dependent across dates/stores; n is not independent
sample size. ERA5 and closure masks are retrospective. Sales proxy censored demand.
Costs are simulated relative outcomes, not actual costs, profit or ROI. No conformal,
causal or operational guarantee. F1 is exploratory/post-pilot; F2-F4 confirmation
status remains pending prior-exposure verification. No P4 bootstrap intervals are
provided in this phase; existing P2/P3 bootstrap does not supply P4 uncertainty.
