# M4 Phase 3D.0 — P1 infrastructure, methodology PENDING_APPROVAL

**No real-data P1 policy evaluation is authorised by this implementation.**
No production evaluation CLI, fitted ML models or output-writing entry point is
provided. `evaluate_p1_synthetic()` rejects real/unverified inputs before residual
estimation. No approved-method flag is provided. Approval and a production gate
must be added in a later authorised phase; do not relabel real data as synthetic.

## Existing contract and adapter

`configs/cost_scenarios.yaml` defines P1 as Seasonal Naive point forecast plus
the tau quantile of **validation** residuals. The M1 baseline functions remain
unchanged: `seasonal_naive()` uses lag7, then lag14, then rolling7;
`to_output()` supplies the legacy nine-column point schema. P1 is not selected
from whichever baseline has best test scores.

`src/decision/p1_residual_policy.py::build_p1_inputs()` takes full ready feature
history, existing baseline predictions and shared configs. It audits lags using
M3 `prepare_features(ready=True)` and uses shared `iter_folds()` only. It calls
the existing M1 rule on scored validation rows and reconstructs test values solely
to audit existing SeasonalNaive/M0 predictions. Existing test predictions are
retained only after exact key, y_true and prediction-rule agreement. Missing,
duplicate, extra scored keys, wrong folds/config or mismatched targets fail.
MovingAverage rows are not P1. No M1 CLI/global path is redirected or patched.

`load_real_p1_inputs()` is a **read-only point-input loader**, not policy evaluation:
it verifies the 34-column candidate bundle, runs production preflight, reads
`data/processed/baselines.csv`, checks reference final M4 scored-key digests and
freezes input/code evidence. It returns memory objects only. It is not executed
as part of Phase 3D.0 synthetic testing. Canonical 35-column features are not used.

Point key: date/store_nbr/family/city/fold/model/config. Outputs additionally carry
split, score_mask, fixed-rule fit_stage, end-of-t−1 forecast_origin and horizon=1.
Fallback diagnostics count lag7 missing, lag14 used and rolling7 used.
Baseline CSV lacks an upstream manifest/completion marker: local exact-rule and
target verification provide reproducibility, not authenticated upstream approval.

## Explicit pending methodology

`ResidualSpec(pooling=..., quantile_method=..., minimum_samples=..., fallback=...)`
requires every choice. No real-method default is selected. Status is always
`PENDING_APPROVAL`. Supported grouping columns are store_nbr/family/city;
fold/model/config are **always retained**, including in the broadest fallback.
The fallback chain must be unique strictly coarser subsets, declared in advance.
An empty tuple means fold/model/config-wide pooling, never cross-fold pooling.

Supported empirical methods: linear/lower/higher/nearest/midpoint. Minimum sample
size counts residual rows; unique dates are separately reported because rows are
temporally/cross-store dependent. No automatic tuning of pooling, method or
threshold is implemented. Synthetic choices in tests are not real-data proposals.

`fit_residual_adjuster()` only accepts scored validation predictions inside their
declared fold windows. Residual is **r=y_true−y_pred**, retaining its sign and sales
units. No abs(), no residual computed on test, no future-fold or other-model
fallback and no clipping of point forecasts before residual construction.

`ResidualAdjuster.predict()` needs no test targets. It selects one eligible pool
per observation for ALL taus, then Q_tau=max(0,y_pred+q_tau(residual)). Quantiles
come from one empirical distribution so ordering is preserved by monotone
quantiles/clipping; fractional Q is never rounded. Report selected pooling level,
fallback usage, calibration n/dates/end, quantile method, minimum n and spec digest.
If no declared pool meets minimum n, fail instead of dropping rows or fabricating
an offset. Missing cell groups (notably Salinas in F2 validation) therefore require
an approved fallback if cell pooling is adopted.

The synthetic integration helper reads tau/costs from the shared scenarios,
computes per-observation cost via the existing Newsvendor primitive and retains
over/understock quantities. All outputs say SYNTHETIC DEVELOPMENT ONLY and
PENDING_APPROVAL. No production costs, P1 indices or recommendations are written.

## Limitations and decisions still requiring ratification

- Choose pooling hierarchy, empirical method, minimum residual sample size and
  sparse/unseen-group fallback **before** economic test evaluation.
- Seasonal Naive has no learned parameter/refit; the same fixed rule applies to
  validation and test with observed past sales allowed for next-day lags.
- Validation forecasts are one-step out-of-sample relative to the fixed rule;
  this is not independence of correlated residuals or an untouched confirmation
  sample. Masks inherit retrospective sales-derived closure detection.
- No independent calibration/conformal coverage guarantee is claimed. Future
  P4 reuse of validation for tuning adds selection optimism; train+validation
  refit changes the predictor underlying residuals and can change calibration.
  This P1 module does not implement or approve P4 calibration.
- F1 test results were already inspected; F2–F4 confirmation remains pending.
  Sales are a censored demand proxy. Simulated costs are not actual inventory
  cost, profit or ROI; no policy/model selection from test outcomes.

Tests use synthetic fixtures only, with estimator fitting forbidden. They check
existing-function reuse, scoring exclusions, exact baseline alignment, residual
signs/quantiles, chronology, pooling/fallback, target blindness, fractional clipping,
cost–pinball identity and the pending-method production guard.
