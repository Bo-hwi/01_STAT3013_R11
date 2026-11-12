# Phase 5B proposal — official-test / M4-validation hybrid P4

**PROPOSED_NOT_APPROVED**. User asked to proceed with preparing the next step;
this is not a fabricated approval of a newly frozen economic execution request.
Original P1/P4/reproduction/bootstrap receipts do not authorize this changed source.

## Frozen scientific scope

Keep the approved residual definition y_true minus M4 clipped train-only validation
forecast. Reuse the unchanged 192 approved historical residual-quantile records,
derived using fold/config/family pooling (selected model invariant per fold/config),
linear interpolation, minimum n=30, fallback fold/config. Restore the existing
ResidualAdjuster from those offsets; no new residual quantile choice, calibration
method or forecasting model is fitted. Group counts, dates, model labels, taus and
minimum sizes must still match the immutable validation evidence.

Replace ONLY the test base with official point_models.csv.gz y_pred as delivered:
nonnegative and rounded to four decimals by pinned M3 notebook cell 23. Do not
unround, substitute M4 precision, invent official raw forecasts or fit another model.
Q=max(0, official_y_pred + frozen validation offset), retaining fractional Q.

Select the exact frozen M4 model mapping: ExtraTrees for F1, HGB for F2-F4, both
M0/M3. No algorithm selection from test labels or test MAEs. Preserve shared
cutoffs/masks and 41,076 matched observations/config/scenario. Test targets are
joined ONLY after order quantities are constructed.

Official SHA-256:
`71e790d360f80a6c3c6d86142391e2ff2947d4d79ec808c2ba3e8095d1623927`.
Reconciliation proves all 82,152 selected forecasts match rounded M4 predictions,
but zero are exactly equal at full precision. This is not an approved economic
equivalence tolerance. New simulated costs/units/indices therefore need a distinct
versioned evaluation rather than provenance relabeling of historical costs.

Use all three shared YAML cost scenarios, existing tested Newsvendor/pinball and
cost-index primitives. P1/P2/P3 rows stay fixed. Ratios of summed costs, not mean
row ratios; zero P1 denominator yields NA. P4 stays identical between primary and
sensitivity comparisons; raw-clipped sensitivity changes P2/P3 only.

## Approval boundary and outputs

This request covers ONLY revised P4 Q/cost/units, calibration-evidence copy,
matched P1-P4 comparisons and summaries at existing report levels. It excludes
bootstrap, Decision Matrix, manuscript/canonical migration and Git/Drive delivery.
After verified hybrid costs, prepare a separate fixed-week downstream request:
same approved 1,000 draws/seed42/fixed weeks, 140 eligible cells, 14 unavailable;
replace P4 inputs only and retain valid P1/P2/P3 evidence. Do not reuse old P4 CI
as official-source CI. Matrix/figures/report revisions follow in new bundles.

No previous bundle, M1-M3 module, shared config, canonical dataset or output is
modified. Unique transactional staging, round-trip checks, output checksums and
completion marker are required. Preserve failure evidence; no fallback algorithm,
source substitution or hash bypass. Preparation selects/validates inputs and
restores existing offsets in memory, but computes no real-data Q or costs.

The receipt must include APPROVED, unchanged request, semantic SHA-256, truthful
approved_by/approval_evidence/approval_communicated_date. User authorization of
this exact request is required; names/signatures/meetings must not be invented.

```powershell
# Prepare once; existing requests are never overwritten:
& '.\.venv\Scripts\python.exe' -B -m src.decision.p4_official_hybrid --prepare-request docs/m4_p4_official_hybrid.request.json
# ONLY after explicit approval and a valid receipt:
& '.\.venv\Scripts\python.exe' -B -m src.decision.p4_official_hybrid --approval docs/m4_p4_official_hybrid.approved.json
```

## Provenance and claims limitations

This is a HYBRID policy: M4_REPRODUCED_M3 validation, OFFICIAL_M3 test. Do not
describe the validation residuals as official M3 validation predictions. Original
M3 feature-file hash differs from local files; exact original feature values are
unavailable. Same pinned preprocessing/refit code is evidenced, not exact original
execution/checkpoint identity. Drive-copy hash identifies the supplied artifact;
independent upstream output checksum is unavailable. Current Parquet DLL reads
are blocked by Windows Application Control; do not bypass or modify policy.
Frozen input/code hashes plus immutable point/quantile scored-key contracts
support reconciliation; a fresh independent Parquet mask rebuild is unavailable.

Residual calibration is not independent: tuning/algorithm selection reused
validation, and train-only residuals transfer to a fresh-refit test predictor.
F1 is exploratory/post-pilot; F2-F4 test exposure remains unverified. ERA5 is
retrospective; sales is a censored demand proxy. Simulated ratios are not money,
profit, ROI, causal value, conformal guarantees or deployment authorization.
Do not retune labels, thresholds or policy choice based on revised test costs.
