# M4 code handoff — 2026-11-11

> Historical code-only delivery note. The current G4 handoff is documented in
> [m4_README.md](m4_README.md): live protocol/receipt/entry-point migration is now
> explicitly authorized by Huy, and official-test P4 point outputs are exported.
> Hybrid downstream CI still requires its own recorded approval. Statements below
> about a zero-byte entry point and unchanged live exports describe the earlier
> delivery, not the current handoff.

Branch: `feature/m4-quantile-decision`. This is code/methodology/evidence delivery,
not a merge to main, canonical migration or claim that all research integration
is complete. Member 2's unrelated notebook working-copy change is excluded.

## Current research status

- Direct quantiles, P1/P2/P3 costs and historical P4 reproduction/economics exist.
- Original fixed-week supplement and Phase4F revision used reproduced P4 test
  forecasts; preserve those as historical evidence, not official-source P4 CI.
- Official M3 test CSV was reconciled: selected forecasts equal M4 predictions
  rounded to four decimals, but are not exactly equal at full precision.
- Approved hybrid P4 costs now use unchanged M4 train-only validation offsets
  and official M3 test bases. No forecasting training or residual refit occurred.
- Hybrid P4 bootstrap, Decision Matrix and final manuscript/canonical migration
  are still pending a separately frozen downstream scope. Do not reuse historical
  P4 CIs under official-source provenance merely because differences are tiny.

Latest verified local bundles (not included in Git):

```text
results/experiments/m4_official_m3_reconciliation_20261111T134400Z_0f08ccb997e341a49d4c749bd9a720c2/
results/experiments/m4_p4_official_hybrid_20261111T143107Z_25d4b1fea9fe400b8b8c935025b3e574/
results/experiments/m4_p4_hybrid_execution_audit_20261111T143356Z_72e8410aed414caea493384f3f4553ec/
```

## Delivery boundaries and reproducibility

Code, synthetic tests, methodology documents and frozen approval requests/receipts
are included. Datasets, `.venv`, generated predictions, completed experiment
bundles and stale canonical report exports are excluded. Transfer approved data/
bundles separately with their checksums, using the team Drive process.

Pinned M3 notebook/reference logs are read from Git commit
`b35f5cc8f5dd2d07b0a545b6e5a60963afdff6bc`, merged by PR #11. A shallow checkout
of this feature branch may not contain that object. Fetch the existing main
history if needed (`git fetch origin main`); do not rerun the notebook to obtain
an absent prediction file. Official file must be supplied from the team copy.

Frozen receipts deliberately bind historical local paths, file bytes, package
versions and scope. They are audit evidence, not portable permission to execute
on a different machine. Do not rewrite historical receipts/hashes to make guards
pass. For a changed environment/path, prepare a new request and authorization.
Windows Application Control currently blocks fresh Parquet DLL imports locally;
do not bypass policy. Reconciliation verified immutable scored-key CSV contracts
and reports fresh Parquet-mask reconstruction as unavailable.

Some handoff-document tests read ignored historical artifacts (Phase4C compliance
and manuscript). Hydrate the verified bundles before running those tests. Their
absence on a clean clone is an artifact-availability limitation, not permission
to fabricate fixtures at production artifact paths or silently skip validation.

Non-training regression check before this code delivery:

```powershell
$env:PYTEST_DISABLE_PLUGIN_AUTOLOAD='1'
& '.\.venv\Scripts\python.exe' -B -m pytest tests/test_m4_uncertainty_handoff.py tests/test_m4_fixed_week_bootstrap.py tests/test_m4_paired_bootstrap.py tests/test_decision.py tests/test_m4_decision_primitives.py tests/test_m4_handoff.py tests/test_m4_official_hybrid.py tests/test_m4_p4_policy.py -q -p no:cacheprovider -W error
```

Result: **212 passed in 20.98s**. No real-data experiment is launched by this test
command. Current source files bound to historical hashes are preserved, including
the zero-byte `src/decision/decision_matrix.py`; its tested compatibility proposal
is in the versioned Phase4F package, not installed over frozen historical source.

All simulated costs remain relative assumptions, not actual money/profit/ROI.
F1 is exposed/exploratory; F2-F4 exposure is unverified. P4 validation calibration
is not independent. No causal, operational, conformal or multiplicity guarantee.
