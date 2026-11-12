# Phase 4B: Member 4 report and handoff contract

Scope: assemble/report verified artifacts, not retrain or rerun economic policies.
Roles/Proposal v4.2 are not present in the accessible local repository, fetched
origin/main tree or connected Drive search (Favorita, STAT3013). Their exact
requirements and canonical schemas remain unverified. README, shared YAML and
explicit M4 approvals provide the available scope evidence; references to the
missing documents are not substitutes for their text.

`m4_handoff.py` verifies all eight source bundles and historical hash maps, checks
point/quantile panels and matched cost observation keys, and reconciles stored
cost aggregates/indices with the final matrix. It never calls model fitting,
residual calibration, economic evaluation runners or bootstrap resampling.

Create a unique `results/experiments/m4_handoff_<UTC>_<UUID>/` package with a
Vietnamese Methods/Results section, final tables, corrected report figures,
artifact map, figure index, risk checklist, reproducibility instructions and
manifest/completion checksums. Sources remain authoritative. The report package
is readable standalone; full source revalidation still requires the repository
and authoritative local bundles/data (which are not all embedded).

Candidate canonical exports are placed ONLY inside that package:
`exports/results/predictions/quantiles.csv` (byte copy of full raw/corrected
validation/test long panel), `exports/results/tables/cost_scenarios.csv` (byte copy
of the 15-row pooled policy summary, not the three-row cost-assumptions table),
and `exports/results/tables/decision_matrix.csv` (byte copy of the 462-row matrix).
No canonical repository file is created/replaced. Schema acceptance and required
split/column conventions require team/role-document confirmation.

Break-even headroom is algebra on existing stored means/indices: C(no weather)
minus C(weather). Positive values are conditional simulated integration-cost
headroom in relative units; nonpositive means no positive headroom in that sample.
It is not actual budget, money, ROI or operational validation.

READY_FOR_HANDOFF means M4 artifacts/report package are ready for peer review and
team integration, not that missing Roles v4.2 compliance, final submission,
test-exposure attestation or deployment readiness has been certified.

```powershell
& '.\.venv\Scripts\python.exe' -B -m src.decision.m4_handoff --build
& '.\.venv\Scripts\python.exe' -B -m src.decision.m4_handoff --verify <handoff_directory>
```

Only non-training tests/checksum verification should be used for this final phase.
Keep F1 exploratory, F2-F4 exposure unresolved, ERA5 retrospective, demand censored,
P4 M4_REPRODUCED_M3 and calibration non-independent; no actual cost/profit/ROI,
causality, cell/P1/P4 CI or deployment claims. P2/P3 bootstrap supports only its
recorded pooled/fold estimands and analyst-declared L3/L7/L14 design.
