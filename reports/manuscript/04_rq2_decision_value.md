# M4 — RQ2 Quantile & Decision Value

## Methods
Direct quantiles use HistGradientBoostingRegressor(loss="quantile"), tau .50/.75/5⁄6,
seed42, M0/M3 and shared F1-F4; two-candidate validation-only raw pinball selection,
train-only preprocessing and fresh train+validation refit. Corrections clip Q>=0
and sort quantiles; raw forecasts remain in Drive bundles. P1=Seasonal Naive with
validation-residual quantile; P2/P3=direct M0/M3 quantiles. P4 uses ACTUAL official
M3 rounded test forecasts, with unchanged M4 train-only clipped-validation residual
quantiles (linear, n>=30, fold/config/family; fallback fold/config). No official raw
forecast is invented. Exact 4-decimal agreement is not full-precision equality.

Simulated C=c_over*(Q-D)+ + c_under*(D-Q)+; ratios 1:1/1:3/1:5. Indices are ratios
of cost sums on identical 41,076 scored test observations, not mean row ratios.
MAE weights over/under symmetrically whereas Newsvendor cost is asymmetric, so
lower point MAE need not yield lower simulated cost. Pinball and point MAE differ.

## Results — Newsvendor Cost Comparison
| scenario | P1 | P2 | P3 | P4_M0 | P4_M3 |
| --- | --- | --- | --- | --- | --- |
| S1 | 1 | 0.733967 | 0.733745 | 0.677546 | 0.678045 |
| S3 | 1 | 0.715333 | 0.715157 | 0.710818 | 0.710307 |
| S5 | 1 | 0.704479 | 0.705461 | 0.721036 | 0.720526 |

Values above and cost_scenarios.png come from the SAME verified stored costs.
Raw-clipped sensitivity changes P2/P3 only and stays separate in cost_scenarios.csv.
All existing fold-level results and 462 scenario cells are provided beside the
154x15 canonical Decision Matrix. M2 weather signatures are family-level context,
not city-specific causal evidence. Pilot/Validate/No-go are short display aliases
for Prioritize pilot/Further validation/Do not deploy. All actual cells remain
Validate because operational availability and exposure are unresolved; do not
force examples of each label or infer deployment from pointwise intervals.

## Robustness and uncertainty
Approved fixed-week CIs use this hybrid P4 cost source, four fixed weekly blocks/fold, 1,000 paired shared draws, seed42, 95% percentile. Only 140 complete 112-date cells receive their OWN CIs; 14 incomplete Esmeraldas/Salinas cells remain unavailable.

## Limitations
Only four weekly blocks/fold; no selection/refit/calibration uncertainty or multiple-
testing control. F1 was exposed in pilots; F2-F4 prior exposure unverified. ERA5 is
retrospective with unverified operational availability. Observed sales may be
censored by unobserved stockouts, hence a demand proxy. Costs/indices are not
money, profit, ROI or causal weather value. Calibration is not independent and
train-only residuals transfer to fresh-refit forecasts from a different source.
Original official input hashes/software differ; raw official forecasts unavailable.

## Conclusions
This is a research/report handoff, not operational deployment. Keep source QA and
uncertainty flags, conservative labels, and M4 self-approval versus team sign-off
distinct. Historical bundles remain unchanged. Team must review G4 acceptance.
