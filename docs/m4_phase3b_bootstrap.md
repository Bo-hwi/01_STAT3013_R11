# Phase 3B: descriptive paired moving-block bootstrap

**ANALYST-DECLARED METHODOLOGICAL DECISIONS, NOT PRIOR TEAM APPROVAL.**
Shared research_protocol.yaml/cost_scenarios.yaml name paired block bootstrap
95% intervals but do not specify block length, repetitions or interval construction.
Earlier sign-off deferred bootstrap design; this analysis does not retroactively
claim approval. It is authorised as descriptive analysis by the Phase 3B request.

Before resampling, persist STARTED.json with code/docs/source hashes and this
fixed design: **seed 42; 5000 resamples; primary L=7 calendar days; sensitivities
L=3 and L=14; nominal 95% percentile intervals, linear quantiles**. Choices are
fixed before computing intervals; do not choose L or a preferred result afterward.
Seven days reflects the weekly retail/calendar cycle (weekday and lag7 are shared
features), not observed bootstrap performance. Three/fourteen days probe shorter/
longer dependence assumptions. A 28-day fold contains only four 7-day blocks or
two 14-day blocks per resample, so effective temporal information remains small.

## Verification and sampling

Source is the immutable completed `m4_costs_p2_p3_20261108T134955Z_b112ad3e7866`
bundle. Verify its marker/checksums, frozen evaluator/config/input hashes and the
underlying forecast provenance. Validate unique keys, matched P2/P3 observed
demand, and common observations across scenarios/variants before aggregation.
Corrected outcomes are primary. Raw-clipped sensitivity is a separate variant;
neither changes policy or forecasting parameters.

Each date contributes the **whole empirical cross-section** of scored stores,
families and cities. Daily sums and observation counts are a lossless compression
for these estimands; no store-day rows are drawn independently. For every sampled
date, apply identical multiplicities to P2/P3, all three scenarios and both variants.
Store/family/city composition remains empirical, not a sampled store population.

Within each fold, split dates into contiguous calendar segments. In EACH segment,
sample uniform starts of non-circular overlapping L-day blocks with replacement,
concatenate and truncate the last block to the original segment's day count.
Never create an individual block across folds or gaps. Concatenated block joins
may jump in time: that is the declared moving-block construction, not a claim
that the entire bootstrap path is consecutive. Segments shorter than L fail;
no days are dropped and L is never silently shortened.

For replicate r, retain all day rows via their counts n:

- mean delta = sum(sampled cost_P3 - sampled cost_P2) / sum(sampled n);
- relative delta (%) = 100*(sum(sampled cost_P3)-sum(sampled cost_P2)) /
  sum(sampled cost_P2), not an average of observation/day-level ratios.

Unequal daily scored-row counts are resampled with their day; do not average
daily means equally. Per-fold draws are independent SeedSequence children of
`[42, L]` in sorted fold order; sensitivity execution order does not change draws.
Pooled replicates combine corresponding replicate IDs across folds, stratifying
by fold/segment and preserving each fold's date count. They do not construct
blocks across the long gaps between test windows. Fold row weights can vary
with sampled daily counts. No new fold labels/regimes are sampled.

## Intervals, signs, undefined ratios

Report pooled and individual folds, by scenario/variant/L, for both estimands:
observed point, 2.5/97.5 percentiles, zero inclusion, finite/undefined replicate
counts, negative/positive/exact-zero fractions, and strict opposite-point-sign
fraction. A zero observed point has no defined opposite sign. Undefined/zero-P2
ratios remain NaN; any relative interval uses finite replicates only, reports
that conditioning and valid counts; no finite replicates means no interval.
These fractions are not p-values or posterior probabilities of weather benefit.

Outputs are a unique `results/experiments/m4_bootstrap_p2_p3_*` bundle, with
daily_costs.csv, intervals.csv, replicates.csv, date_layout.csv, exact resampling
indices (NPZ without pickle), STARTED/manifest/COMPLETE JSON. Outputs are written
exclusively to staging, round-trip checked and checksummed; completion markers
do not self-hash. Source/prior bundles are never edited. Failures remain visible.

```powershell
& '.\.venv\Scripts\python.exe' -B -m src.decision.paired_bootstrap
```

## Interpretation limits

**Descriptive bootstrap intervals for simulated costs only**. Fixed fitted
forecasts, selected hyperparameters and cost assumptions are conditioned on:
no model-refitting/training-selection uncertainty is incorporated. Dependence
within days and within sampled blocks is retained; longer-range/cross-fold
dependence and structural change are not fully represented. Four short windows,
edge weighting of non-circular blocks, limited effective blocks, closure-driven
composition and within-segment stationarity assumptions affect reliability.
There are 112 original test dates, not 41076 independent store-day draws.

Intervals are pointwise across many scenarios/folds/variants/lengths, without
multiple-comparison adjustment. Zero exclusion does not establish formal
significance beyond this chosen bootstrap design, causality, operational
performance, actual inventory cost/profit/ROI, or a policy recommendation.
F1 remains exploratory/post-pilot. F2–F4 confirmation is pending prior-exposure
verification. All sales-censoring, retrospective ERA5 and upstream QA/provenance
limits from Phase 3A continue to apply. Do not relabel this as preregistered or
an independent confirmation of conclusions used to choose later experiments.
