# M4 Phase 3A — simulated P2/P3 Newsvendor evaluation

**SIMULATED ECONOMIC OUTCOMES: not actual inventory costs, profit or ROI.**
No training, forecast changes, calibration, P1/P4 residuals, policy selection,
bootstrap or confirmatory claims are performed by this stage.

`src/decision/policy_evaluation.py` reuses the immutable final quantile bundle,
`configs/cost_scenarios.yaml`, tested cost/quantile primitives and panel gates.
The CLI verifies COMPLETE.json, all source artifact checksums, frozen input/code
hashes, the full prediction panel and recomputed forecast metrics BEFORE costs.
Both source evidence and evaluator/config/docs hashes are frozen before economic
calculation and rechecked before output finalization. No forecast bundle is edited.

- P2 = corrected direct M0; P3 = corrected direct M3. Primary `variant=corrected`.
- Sensitivity `variant=raw_clipped_sensitivity`: raw prediction clipped at zero,
  without sorting. Original source_prediction is preserved. Raw crossing therefore
  does not make the corrected-policy calculation ambiguous.
- Test rows only; exact matched date/store_nbr/family/city/fold/tau_key, identical
  y_true across configs, three canonical taus per observation. Production uses
  source scored-key digests to detect observations missing from BOTH configs.
- Scenario ID, c_over, c_under and tau come from YAML. Unique canonical tau
  mappings and tau=c_under/(c_over+c_under) are validated. Current ratios are
  1:1, 1:3, 1:5; no empirical retailer-price assumptions are introduced.
- Fractional Q is retained. Cost is c_over*max(Q-D,0)+c_under*max(D-Q,0), with
  D=observed sales and feasible Q=max(source_prediction,0). The per-row identity
  cost=(c_over+c_under)*pinball(D,Q,tau) is checked at THIS Q, never unclipped raw Q.
- Report row-weighted mean/total relative simulated cost, mean/total overstock
  and understock quantities, mean cost delta P3-P2, ratio sum(P3)/sum(P2), and
  relative delta 100*(ratio-1). Zero P2 denominator gives NaN ratio/relative delta
  and an explicit denominator flag; absolute mean delta remains defined.
- Report separately by scenario/variant at pooled, fold, family, city, family-city,
  fold-family, fold-city and fold-family-city levels. No average across assumed
  cost scenarios is used to select a policy. Negative delta is descriptive lower
  simulated cost, not a recommendation to adopt P3.

`cost_ratio_P3_over_P2` is NOT the shared `cost_index` normalized to P1. P1-based
indices and weather_value defined through P1 remain unavailable. No P1/P4 is
fabricated from missing validation predictions/residuals.

Outputs are created exclusively in a unique staging directory, round-trip checked,
hashed, marked COMPLETE, then renamed to a non-existing `results/experiments/
m4_costs_p2_p3_<UTC>_<random>/`. The bundle has row_costs.csv, comparisons.csv,
STARTED.json, manifest.json and COMPLETE.json. Marker hashes all other outputs
without a self-hash. Incomplete/failed runs remain for diagnosis; no retry,
overwrite or automatic deletion. `inspect_evaluation()` checks output integrity.

Execution (no model fitting):

```powershell
& '.\.venv\Scripts\python.exe' -B -m src.decision.policy_evaluation --source-experiment final_f1_f4_m0_m3_20261108T123157Z_eb881cc4c56a
```

F1 remains exploratory/post-pilot; F2–F4 confirmation remains pending prior-exposure
verification. Inputs inherit retrospective ERA5, sales-derived closure masks,
source QA and availability limitations. Sales censoring/unknown stockouts mean
the simulated understock term is not an observed stockout. One-period linear
costs omit actual inventories, lead times, pack sizes and prices. Quantities are
family-level sales units; pooling across heterogeneous families is a modelling
assumption, not a physical inventory total. No cost CI/significance is claimed.
