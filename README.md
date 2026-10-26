# Weather × Favorita — Retail Demand Forecasting

**Course:** STAT3013 · **Repository:** `Bo-hwi/01_STAT3013_R11` · **Timeline:** 4 weeks (26/10–22/11/2026)

## Project overview

**Evaluating the Business Value of Weather Information in Retail Demand Forecasting.** This research combines Corporación Favorita daily store-family sales with historical weather observations to investigate statistical weather–sales associations and the *incremental out-of-sample* value of weather-aware forecasting. A simulated replenishment-cost analysis translates forecasts into conditional business recommendations. Weather benefits are hypotheses to test, not guaranteed outcomes.

## Research questions

- **RQ1 — Statistical association:** How are weather conditions associated with daily sales, and how does the relationship differ across product families and locations? Methods: pooled OLS, fixed effects, random effects, robust uncertainty, sensitivity analysis. Associations are **not causal claims**.
- **RQ2 — Forecasting and decision value:** Does historical weather improve held-out demand forecasts, and under what assumptions can it reduce *simulated* replenishment costs? Methods: Seasonal Naive, trailing Moving Average, Ridge, Extra Trees, one global boosted model, weather-feature ablations, quantile regression, and Newsvendor simulation.

## Research scope and data

- **Coverage:** 54 stores, 22 cities, 33 product families for EDA; 6–8 families selected for deeper modeling according to a pre-specified, training-only rule.
- **Prediction unit:** `date × store_nbr × family`; target `sales`.
- **Forecast origin:** end of day `t−1`, predicting day `t`.
- **Sales:** [Kaggle Store Sales — Time Series Forecasting](https://www.kaggle.com/competitions/store-sales-time-series-forecasting/data).
- **Weather:** [Open-Meteo Historical Weather API](https://open-meteo.com/en/docs/historical-weather-api) / ERA5 reanalysis. City coordinates, units, grid, missingness and source timestamps must be audited.

**Critical limitation:** ERA5 reanalysis is published with a lag. Historical weather variables may not have been available at the forecast origin. Such results must be labeled **retrospective weather-aware**; do not present them as deployable operational backtests without verified availability timestamps or an appropriate timely feed. Observed sales are only a proxy for unconstrained demand; stockouts/inventory may be unknown.

## Team responsibilities

| Member | Primary ownership | Main deliverables |
|---|---|---|
| M1 — Data & Baselines | `src/data/`, `src/features/`, `src/forecasting/baselines.py` | Audited joins, reproducible features, Seasonal Naive and Moving Average |
| M2 — Statistical Research | `src/statistics/` | 33-family EDA, panel OLS/FE/RE, heterogeneity, robust uncertainty |
| M3 — Forecasting & ML | `src/forecasting/` (excluding baselines) | Ridge, Extra Trees, global boosted model, rolling folds, weather ablation |
| M4 — Quantile & Decisions | `src/decision/` | Quantile forecasting, Newsvendor cost simulation, decision matrix and report integration |

All members contribute code, quality checks, report sections, presentation, and peer review. See the approved Proposal v4 and Team Roles v4 for exact scope.

## Repository map

```text
configs/     Research protocol, ablation definitions and cost scenarios
 data/        Local-only raw, interim and processed data; README tracks provenance
notebooks/   Exploration and figures grouped by M1–M4
src/data/    Data ingestion, cleaning and weather joins
src/features/ Time-safe lag and rolling features
src/statistics/ RQ1 descriptive and panel inference
src/forecasting/ Baselines, point models, rolling evaluation
src/decision/ Quantiles, costs and weather–product–location decisions
results/     Small selected tables/figures and local prediction outputs
reports/     Proposal, manuscript, IEEE-style paper and slides
docs/        Dictionary, QA, assumptions and methodology
scripts/     Reproducible entry points
tests/       Data integrity, leakage and prediction contracts
.github/     Collaboration rules and CI
```

## Local setup

Python 3.11 is recommended. From repository root:

```bash
python -m venv .venv
# Windows PowerShell:
.venv\Scripts\Activate.ps1
# macOS/Linux: source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Download Favorita files using the Kaggle competition page and weather using the documented Open-Meteo pipeline. Place local files under `data/raw/favorita/` and `data/raw/weather/`. **Do not commit large raw datasets or secrets.** Consult `data/README.md` for required filenames, licenses, checksums and regeneration instructions once documented by M1.

**Pipeline status:** `python scripts/run_pipeline.py --test` builds `data/processed/curated_sales_weather.parquet` from local raw files and runs data integrity tests (M1, curated v1). Feature, baseline and model stages are not implemented yet.

## Experimental protocol and evaluation

- Primary metric: MAE; secondary WAPE only with positive denominator and MASE where valid.
- Non-overlapping, time-ordered rolling-origin test folds; all preprocessing and hyperparameter tuning must use training/inner-validation data only.
- Compare identical store-family-date test keys and folds across models. Report paired changes, family/city/fold heterogeneity, and uncertainty intervals.
- **M0–M5 are weather feature sets, not six separate ML algorithms.** Define them in `configs/ablation_M0_M5.yaml` and freeze before final testing.
- Quantile levels: 0.50, 0.75, ~0.8333; assumed over:under cost ratios 1:1, 1:3, 1:5. Report assumptions, coverage, simulated costs and limitations.
- The `configs/research_protocol.yaml` file contains explicit TODOs for unfinalized research decisions; resolve these before held-out evaluation.

## Collaboration

Work on short-lived `feature/<member>-<topic>` branches, submit Pull Requests to `main`, and request peer review. Avoid committing generated predictions, local environments, credentials and unlicensed source data. Track tasks and evidence using GitHub Issues/Projects.

## Reproducibility checklist

- [ ] Source links, licenses, file hashes and weather coordinates recorded
- [ ] Data keys unique and weather join many-to-one
- [ ] Lag/rolling features respect forecast origin and availability
- [ ] Deep-study family shortlist and test folds frozen before final evaluation
- [ ] Models share prediction keys, folds and metrics
- [ ] Statistical uncertainty and weather publication lag documented
- [ ] Quantile/cost policies tuned without looking at test outcomes
- [ ] Report tables and figures link to experiment IDs

