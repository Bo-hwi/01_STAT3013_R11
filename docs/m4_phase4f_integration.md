# Phase 4F — versioned handoff and final team acceptance

READY_FOR_TEAM_REVIEW means the technical revision is verified, not that the
team has accepted all historical exceptions or authorized live migration.
No forecasting, calibration, economic-policy or bootstrap execution is needed.

## What is integrated

The 2026-11-10 approved fixed-week supplement supplies economic intervals for
six contrasts and P1 policy indices. Four fixed calendar weeks/fold, 1,000 common
paired draws, seed 42, pointwise 95% linear percentile. P1-normalized differences
use the sampled P1 denominator; relative percentages use the named comparator.
They are distinct fields/units. Forecast MAE/pinball CI remains unavailable.

Only exact pooled, fold and family-city pooled keys receive intervals. Fourteen
incomplete cells keep point estimates and NA CI, including their original sample
sizes and missing-fold flags. No pooled/fold CI is assigned to a cell. Family-only,
city-only and fold-cell economic CI remain unavailable: no approved analysis at
those grains. Existing corrected-primary and direct-only raw sensitivity remain
separate. P1/P4 policies do not change between variants.

The 154x15 matrix preserves all old sample sizes, forecast metrics, means,
indices, fold consistency, source QA, M2 annotations, descriptive directions and
recommendations. Only CI/evidence narrative fields change. All recommendations
stay Further validation. This does not imply a preregistered decision rule or
multiple-testing adjustment. Existing canonical outputs and completed bundles
are immutable; revision canonical/ is a proposed replacement subtree for later
explicit migration, not a write to the live paths.

Legacy moving-block bounds are retained as explicitly named sensitivity fields
in the versioned cost table and in the unchanged old bundle. New economic CI
records name metric/comparator/units/scope/variant/denominator/seed/sampling,
valid/undefined counts and source bundle. Point estimates are checked against
stored cost means/indices, never re-evaluated from quantities/demand.

## Required entry point

`src/decision/decision_matrix.py` is still zero bytes and its hash is bound by
historical approved evidence. Installing a wrapper now would invalidate live
frozen-source verification. Therefore a thin wrapper is supplied ONLY as a
tested proposal in compatibility/src/decision/decision_matrix.py. It exposes
build_decision_matrix(base, intervals, support) and forwards CLI to the versioned
adapter. Team must accept the alternate path, or approve a later migration with
archival source snapshots and version-aware historical verification. No frozen
source, manifest or hash is rewritten. Current CLI:

```powershell
& '.\.venv\Scripts\python.exe' -B -m src.decision.uncertainty_handoff --help
& '.\.venv\Scripts\python.exe' -B -m src.decision.uncertainty_handoff
```

The second command verifies existing evidence and creates a unique revision;
it does not run models, calibration, costs or bootstrap. Existing output targets
are never overwritten. Bundle outputs include manifest/checksums/COMPLETE,
compliance, artifact map, versioned canonical subtree and integration evidence.

## Team acceptance checklist (all items remain unconfirmed until evidenced)

- [ ] Accept this versioned revision and approve any later canonical migration.
- [ ] Accept the frozen YAML primary-model TODO as a historical documentation gap,
  with actual pre-run HGB sign-off dated 2026-11-08. Do not backdate a YAML edit.
- [ ] Accept approved M4_REPRODUCED_M3 as the P4 input substitute for official M3;
  retain input/environment differences and no exact numerical reproduction claim.
- [ ] Accept modular quantile_pipeline/policy_evaluation layout and decide the
  exact decision_matrix.py entry-point migration; proposed wrapper is not installed.
- [ ] Acknowledge F1 exposure/exploratory status. Verify and record prior F2–F4
  test exposure before any confirmatory claim; until then keep status unverified.
- [ ] Accept pointwise fixed-policy CI limitations: four blocks/fold, fixed strata,
  no model-selection/calibration/refit uncertainty or multiplicity control.
- [ ] Accept 14 incomplete-calendar cells with CI unavailable; no imputation,
  fold-cell/forecast-error CI or operational policy authorization is supplied.
- [ ] Keep retrospective ERA5 availability, censored sales-demand proxy,
  non-independent residual calibration, no causal/money/profit/ROI claims.
- [ ] Review M4 Methods/Results and figure captions; integrate IEEE/slides with
  member-owned sections. No other member's technical section is changed here.
- [ ] Inspect Git status and ignored artifacts before a later authorized commit;
  agree Drive '4. Processed data' destination, access and hash manifest. No commit,
  push, upload or sharing is performed in Phase 4F.
- [ ] Record actual G4/G5 acceptance/submission evidence when those events occur.

Proposed team message: "M4 versioned fixed-week uncertainty handoff is ready for
review. Calculations and point estimates are unchanged; 140 cells have their own
economic CI, 14 have documented missing-calendar CI. Please confirm the checklist,
especially historical TODO, M4 reproduction substitution, F2-F4 exposure, module
layout and migration/delivery. This is not a deployment or confirmatory claim."
