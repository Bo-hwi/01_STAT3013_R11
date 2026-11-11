# Phase 4A descriptive Decision Matrix specification

No forecasting fits, retuning or new bootstrap. All policies are frozen source
artifacts. This aggregation rule is declared in Phase 4A before matrix aggregation;
it is NOT claimed to have been preregistered before the underlying test experiments.

Primary matrix: seven frozen families x 22 observed/frozen cities x three shared
scenarios = 462 rows. One readable summary per family/city = 154 rows. Primary
direct-quantile costs use corrected predictions. Raw-clipped P2/P3 sensitivity
is saved separately; P1/P4 keep their already approved calibration definitions.

## Direction classification (no tuned effect-size/sample threshold)

Within each family/city/scenario, classify each weather comparison separately:
P3 minus P2 and P4-M3 minus P4-M0. Require finite matched deltas in all four F1-F4
folds with positive sample sizes. All four deltas strictly negative =>
directionally_favorable; all strictly positive => directionally_unfavorable.
Zeros or opposing signs => mixed_or_inconsistent. Missing folds/nonfinite values
=> insufficient_evidence. These describe directions, not confidence/significance.

Consolidated classification: insufficient if either comparison is insufficient;
favorable/unfavorable only if BOTH comparisons have that same classification;
otherwise mixed. No M2 annotations, policy rankings or bootstrap bounds enter this
rule. No sample/effect-size threshold is tuned. All cells say no deployment claim.

## Distinct evidence domains

Costs and indices are simulated economic outcomes; indices use ratios of summed
costs on exact matched observations, NaN on zero P1 cost. Weather deltas compare
fixed paired policies, not arbitrary best-policy rankings. Forecast diagnostics
are tau-specific raw/corrected pinball loss, coverage/gap/strict-less/tie rates,
crossing and negative rates. P4 clipped point MAE is separately labeled and is not
interchangeable with quantile pinball loss. M2 RQ1 annotations are family-level
associations repeated as context, not city-specific forecasts or causal findings.
M2's original family FDR labels do not imply RQ2 cell-wise multiple-testing control.

## Uncertainty and limitations

Existing P2/P3 bootstrap intervals are copied/verified only for pooled F1-F4 and
individual folds, with original L=3/7/14 sensitivity (L=7 primary), 5,000 resamples,
seed 42. Design was analyst-declared, not previously team-approved. No pooled CI is
attached to family/city cells. P1-comparison/P4 uncertainty is unavailable. There
is no new resampling, cell CI, p-value, significance or multiple-testing correction.

F1 exploratory/post-pilot; F2-F4 prior exposure verification pending. Validation
reuse and refit/calibration mismatch persist for P4, whose point source remains
M4_REPRODUCED_M3 rather than official M3. Retrospective ERA5, sales-derived
closure/cooldown masks, QA09 excluded wind, uncertain operational availability,
limited store-52 history and censored sales proxy are flagged. No actual inventory
cost/profit/ROI, causal weather, or deployment claims are authorized by this matrix.

## Outputs and integrity

New unique `results/experiments/m4_decision_matrix_<UTC>_<UUID>/`: matrix.csv,
matrix_raw_sensitivity.csv, family_city_summary.csv/html, fold_consistency.csv,
forecast_diagnostics.csv, point_forecast_diagnostics.csv, policy_summary.csv/md,
weather_summary.csv, supported_bootstrap.csv, source_QA.json, figures PNG/SVG,
M4_RESULTS.md, STARTED/manifest/COMPLETE with checksums. Output staging on same
filesystem, exclusive file creation, round-trip tables, preserve failed staging,
never overwrite prior bundles. Source maps are verified again before completion.

Execute (aggregation only):

```powershell
& '.\.venv\Scripts\python.exe' -B -m src.decision.consolidated_matrix
```
