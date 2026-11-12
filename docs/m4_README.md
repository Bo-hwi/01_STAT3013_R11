# M4 — Quantiles, simulated decision value, G4 handoff

Huy / `Bo-hwi`, branch `feature/m4-quantile-decision`. G4: Wednesday 18/11/2026.
M4 approvals are self-approvals; team review/sign-off is separate.

## Data flow and modules

1. `quantile_models.py`, `cost_simulation.py`: tested pinball/coverage/quantile
   corrections and asymmetric-cost/index primitives.
2. `quantile_pipeline.py`, `quantile_preflight.py`, `quantile_artifacts.py`: shared
   M1/M3 feature/fold contracts, HGB selection/refit and sealed experiment outputs.
3. `p1_execution.py`, `p1_residual_policy.py`: Seasonal Naive residual policy.
4. `p4_reproduction.py`: M4-owned M3 reproduction, NOT official M3 predictions.
5. `p4_official_hybrid.py`: unchanged M4 train-only clipped-validation offsets +
   ACTUAL official M3 rounded test forecasts. This is the P4 source for final costs.
6. `fixed_week_bootstrap.py`: shared fixed-week paired bootstrap math. Historical
   reproduced-P4 intervals cannot be assigned to official hybrid P4.
7. `decision_matrix.py` → `final_handoff.py`: executable entry point consuming
   explicit sealed bundles, matching keys and official values, and exporting
   cost tables/154-cell matrix/figures/manuscript without model/cost reruns.

All outputs in one handoff run use the same verified hybrid cost source. The
462 family–city–scenario details and fold cost table accompany the exact 154x15
canonical matrix. Pilot/Validate/No-go are display aliases for Proposal §10's
Prioritize pilot/Further validation/Do not deploy. Rules require QA/coverage,
fold consistency and appropriate uncertainty; actual data are not forced to
contain all labels. Operational availability is unverified: conservative Validate
labels are retained. No retrospective preregistration/multiplicity claim.

## Inputs to obtain from Drive (not committed)

- Official `point_models.csv.gz`, SHA-256
  `71e790d360f80a6c3c6d86142391e2ff2947d4d79ec808c2ba3e8095d1623927`.
- Completed P1, P2/P3 direct-cost, final direct-quantile and official hybrid P4
  bundles named in the command below, including manifests and COMPLETE markers.
- M2 `weather_signatures_for_m4.csv` (family annotations only).
- Historical migration archive `m4_handoff_migration_archive_20261111T155113Z`:
  exact original code/config/receipt bytes before authorized post-run edits.
- Official-P4 downstream bootstrap bundle
  `m4_p4_official_bootstrap_20261111T170146Z` (receipt
  `docs/m4_official_downstream.approved.json`, request semantic SHA-256
  `343325e3dfe94d4bc08652192b33d50eb0d78724b09b4930600ec41efdec6931`, M4 (Huy)
  self-approved 2026-11-12). Old reproduced-P4 CIs remain historical sensitivity
  evidence only.

These are Drive artifacts. Large predictions, raw datasets and experiment bundles
must not be put into Git. Frozen checksums identify exact evidence, not portable
permission to execute historical training receipts on a different environment.
The explicit archive verifies migrated historical bytes; old manifests are never
rewritten and global integrity guards are not disabled. Missing historical training
inputs are reported as unverified; sealed-output checksums and exact official/key
matching remain mandatory for handoff generation.

## Reproduce handoff (no forecasting/economic rerun)

Run from repository root after hydrating the above inputs. `--output-dir` must
NOT exist; live canonical migration is explicit and recorded separately.

```powershell
& '.\.venv\Scripts\python.exe' -B -m src.decision.decision_matrix `
  --p4-bundle results/experiments/m4_p4_official_hybrid_20261111T143107Z_25d4b1fea9fe400b8b8c935025b3e574 `
  --p1-bundle results/experiments/m4_p1_20261108T173741Z_4d68ed38ce2f4d1299d1e464d47dfa64 `
  --direct-bundle results/experiments/m4_costs_p2_p3_20261108T134955Z_b112ad3e7866 `
  --quantile-bundle results/experiments/m4_quantiles_final_f1_f4_m0_m3_20261108T123157Z_eb881cc4c56a `
  --official-predictions data/processed/point_models.csv.gz `
  --annotations results/tables/weather_signatures_for_m4.csv `
  --historical-archive results/experiments/m4_handoff_migration_archive_20261111T155113Z `
  --bootstrap-bundle results/experiments/m4_p4_official_bootstrap_20261111T170146Z `
  --output-dir results/experiments/m4_g4_handoff_reproduction
```

The CLI rejects a bootstrap bundle tied to historical reproduced P4. Omitting
`--bootstrap-bundle` reproduces the earlier points-only export, where the matrix
marks official-P4 CI pending; no pooled/reproduced-P4 bounds are borrowed for
cells. No hidden local defaults.

The bootstrap bundle itself was produced (seed 42, 1,000 shared fixed-week draws,
95% percentile; no model fits or cost recalculation) with:

```powershell
& '.\.venv\Scripts\python.exe' -B -m src.decision.final_handoff <same six input flags + --historical-archive> `
  --bootstrap-approval docs/m4_official_downstream.approved.json `
  --bootstrap-output results/experiments/m4_p4_official_bootstrap_<UTC timestamp>
```

Current canonical tables/figures/manuscript come from
`m4_g4_official_handoff_ci_20261111T170146Z`; the promotion and prior canonical
copies are recorded in `m4_g4_canonical_migration_ci_20261111T170146Z`.

### Official-P4 CI summary (pooled F1–F4, corrected, % vs comparison policy)

| Scenario | P2 vs P1 | P3 vs P1 | P4_M0 vs P1 | P4_M3 vs P1 | P3 vs P2 | P4_M3 vs P4_M0 |
| --- | --- | --- | --- | --- | --- | --- |
| S1 | −26.6 [−31.9, −19.4] | −26.6 [−31.9, −19.4] | −32.2 [−37.9, −24.9] | −32.2 [−37.8, −24.8] | −0.03 [−0.04, −0.02] | +0.07 [−0.61, +0.77] |
| S3 | −28.5 [−35.3, −19.2] | −28.5 [−35.3, −19.3] | −28.9 [−35.6, −20.4] | −29.0 [−35.6, −20.1] | −0.02 [−0.10, +0.06] | −0.07 [−1.18, +1.04] |
| S5 | −29.6 [−38.3, −18.7] | −29.5 [−38.2, −18.6] | −27.9 [−35.5, −18.6] | −27.9 [−35.4, −18.3] | +0.14 [−0.01, +0.30] | −0.07 [−1.47, +1.31] |

Every quantile policy beats Seasonal Naive (P1) on simulated cost, with intervals
excluding zero. Adding weather (M3 vs M0) makes no material difference: P3−P2 is
tiny (excludes zero only at S1), and P4_M3−P4_M0 includes zero in all scenarios.
At cell level, 140 complete cells have their own CIs and 14 incomplete
Esmeraldas/Salinas cells remain NA. All 154 labels stay Validate because
operational availability and F2–F4 exposure are unverified. Intervals rest on
only four weekly blocks per fold and carry no multiplicity control.

## Historical and current evidence

`docs/m4_github_handoff.md` records the earlier code-only delivery. Phase4C/4F
exports and P4 reproduced-test economics are historical, not current official-P4
handoff. The protocol now names HistGradientBoostingRegressor, with the exact
requested note `decided 2026-11-08 (team chat), committed later`. The old TODO
and original receipts remain recoverable from the migration archive/Git history;
request scopes, semantic hashes and original dates were not changed. Outer receipt
attribution is corrected to **M4 (Huy) self-approved**, not group sign-off.

Local-artifact tests skip ONLY when their two historical files are absent; when
present they execute normally, including malformed/corrupt inputs. Full and clean-
checkout test outcomes and PR scope are recorded in the final handoff validation
report. No model training, recalibration or test-target selection is part of this
handoff command. Labels are research triage; no actual money/profit/ROI, causal,
operational or conformal guarantee. F1 exposed; F2-F4 exposure remains unverified.
