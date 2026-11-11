# P1 residual-quantile policy proposal and approval gate

Status: **PROPOSED_NOT_APPROVED**. No official P1 economic results have been
produced under this proposal. The shared research/cost YAML defines validation
residual adjustment, but not its pooling, interpolation, sample threshold or
fallback. `m4_final_signoff.approved.json` explicitly defers P1/P4 calibration;
its direct-quantile approval is not approval of this proposal.

## Proposal requiring team confirmation

- Signed residual: `r_validation = y_true - y_pred_raw`. M1 SeasonalNaive's
  `y_pred` is the unmodified fixed-rule output (lag7, lag14, rolling7 fallback);
  it is not a clipped or rounded ML forecast. Residuals retain their signs.
- Pool within **fold x model x config x family**. Never borrow from another fold,
  model or config, including future folds.
- Empirical quantile: NumPy `quantile(method='linear')`.
- Minimum **30 scored validation observations**, not 30 independent days.
- Sparse/unseen family fallback: **fold x model x config** with the same n>=30.
  If neither pool qualifies, fail instead of dropping observations.
- Same selected pool for every tau; `Q=max(0, test forecast + validation residual
  quantile)`. Preserve fractional quantities. No residual clipping.
- Shared YAML scenarios map S1/S3/S5 to canonical 0.50/0.75/5/6 via existing APIs.

SeasonalNaive has no learned parameters to refit. Validation and test use the same
one-day-ahead fixed rule and observed past sales. Validation residuals are
temporally/cross-store dependent; no independent calibration, conformal coverage
or operational guarantee is claimed. P4 tuning/refit issues remain a separate
decision. F1 is exploratory/post-pilot; F2-F4 exposure verification is pending.
Costs are simulated relative outcomes, not actual inventory cost, profit or ROI.

## Implementation and execution after approval

`p1_residual_policy.py` reuses M1 functions and shared `iter_folds()`, reconstructs
scored validation forecasts, and verifies existing test forecasts exactly.
`p1_execution.py` proposes the above settings, validates external approval before
loading real inputs, then repeats candidate marker/preflight/hash/key checks.
It reads the completed P2/P3 cost bundle and checks exact test keys and y_true for
every scenario/policy/variant. Corrected and raw-clipped sensitivity stay separate.
Indices are ratios of cost sums on matched rows; zero P1 denominator returns NaN,
including P1/P1. It recomputes stored costs using the configured primitives.

Prepare a frozen request only when ready to seek approval (does not evaluate costs):

```powershell
& '.\.venv\Scripts\python.exe' -B -m src.decision.p1_execution --prepare-request docs/m4_p1.request.json
```

This exclusively creates a new file and binds method, candidate files, baseline,
P2/P3 bundle, code and configs. Existing files cannot be overwritten. Request
preparation is not approval. Any subsequent bound-file change requires reapproval.

After actual team confirmation, create a **new** JSON receipt with:

```json
{
  "status": "APPROVED",
  "request": "REPLACE with the complete unchanged request OBJECT, not a string",
  "request_sha256": "semantic_hash(request) from quantile_contracts",
  "approved_by": "actual supplied approval identity",
  "approval_evidence": "actual team/user confirmation reference",
  "approval_communicated_date": "actual YYYY-MM-DD"
}
```

Do not invent approval evidence. No receipt is supplied by this implementation.
The receipt is a local evidence record, not a cryptographic signature.

Only after approval and separate execution authorization:

```powershell
& '.\.venv\Scripts\python.exe' -B -m src.decision.p1_execution --approval docs/m4_p1.approved.json
```

Outputs go to unique `results/experiments/m4_p1_<UTC>_<UUID>/` with P1 rows,
calibration audit, paired rows, indices (pooled/fold/family/city/cell levels),
approval/provenance manifest, checksums and completion marker. Originals remain
unchanged. Failed staging remains incomplete for diagnosis; never reuse it.
The earlier generic infrastructure document's no-production-CLI statement applies
to Phase 3D.0; this proposal adds a gated path, not current authorization.
