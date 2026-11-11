# Phase 4C compliance/export specification

Authority: user-supplied extract at
`../m4_requirements.md` (Proposal v4.2 §§8.4/9/10/Appendix A; Roles v4.2 §§F/H/I/K).
Read and hash the original, copy evidence into a NEW audit bundle; never change it.
Frozen repository configs prevail. Original Word documents are not independently
available, but this extract is explicitly authoritative for the present task.

No retraining, retuning, quantity/cost-function reevaluation, economic runners,
bootstrap resampling, shared-config or prior-bundle changes. Only schema adapters,
read-only aggregation of STORED over/understock units, manuscript and compliance
evidence. Inspect existing targets; exclusive creation or verified identical-file
reuse, never overwrite. Source experiment bundles stay authoritative.

## Canonical schemas

- quantiles.csv.gz: TEST ONLY, exactly date/store_nbr/family/city/fold/model/config/
  tau/y_true/y_pred. y_pred=stored corrected quantile. model retains actual
  HistGradientBoostingQuantile identity, not a claim the shared TODO was resolved.
  tau labels follow YAML 0.5/0.75/0.8333333333; joins use canonical tau keys and the
  original mathematical 5/6 remains preserved in the authoritative long panel.
- decision_matrix.csv: ONE ROW per family/city (154), exactly the 15 shared-config
  columns. baseline_error/weather_error/delta_error_pct are explicit per-tau JSON
  because the extract does not designate a single primary tau. cost_scenarios
  includes all three frozen scenarios and five policy costs/indices. Keep the
  existing 462-row direction classifications inside these JSON records.
- cost_scenarios.csv: stored P1/P2/P3/P4 summaries at all existing report levels,
  plus source-derived over/understock means and explicit weather index deltas.
  Existing P2/P3 pooled/fold L7 relative-delta CIs are attached ONLY to that named
  weather estimand; never to policy/P1 indices, P4 or cells. All unsupported CIs
  remain unavailable. L3/L14 sensitivity retained in audit copies.

## Recommendations distinct from directions

All 154 cells get **Further validation**: cell-level paired uncertainty is missing,
operational weather availability is unverified and prior exposure unresolved.
Deployment status remains not authorized. No automatic Favorable->Prioritize pilot
or Unfavorable->Do not deploy. Those other allowed labels are not emitted without
new defensible evidence/rule authorization. This conservative Phase4C mapping is
not claimed preregistered before test exposure; historical timing compliance is
partial. Recommendation is research triage, not an operational action.

## Unresolved hard gaps

primary_quantile_model remains TODO in frozen YAML despite pre-run HGB sign-off;
do not retroactively modify configs or claim the BEFORE-full-run field rule passed.
decision_matrix.py is a zero-byte placeholder bound by completed-run source hashes;
do not fill it and invalidate provenance. Implement adapter in
decision_matrix_canonical.py; record exact required entry-point gap. Similar
modular training/evaluation code lives outside the two primitive modules listed
by Roles; report functional evidence and exact-file layout separately.

Existing M4 paired moving-block bootstrap (L7 primary; 5,000 resamples; L3/14
sensitivity) meets general paired 95% protocol but differs from pinned M3's
fixed four weekly blocks and 1,000 draws. Thus exact 'same design as M3' is PARTIAL.
No new bootstrap is authorized. Proposed minimum primary contrasts: P3-P2,
P4-M3-P4-M0, and each of P2/P3/P4-M0/P4-M3 versus P1. Existing pooled/fold P3-P2
results cover one contrast only. Add five contrasts x 3 scenarios x 5 scopes=75
coarse summaries and six contrasts x 462 cells=2,772 cell summaries, subject to
team approval of harmonized design, cell calendar/gap feasibility and multiplicity
interpretation. Never treat store-day rows as independent or reconstruct absent
Salinas F1. P4-versus-quantile inferential rankings need extra predeclared contrasts
if claimed. No numerical uncertainty results are invented here.

Overall Roles/G4 readiness remains NOT READY until the requirement-level blockers
are resolved/accepted. Computational artifacts remain complete and unchanged.
