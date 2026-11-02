# M2 Statistical Research & Panel Inference Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement the complete Research Question 1 (RQ1) statistical research pipeline for Member 2 (M2) — including 33-family EDA, Panel Econometrics (Pooled OLS, Fixed Effects, Random Effects) with Clustered Standard Errors and Benjamini-Hochberg FDR correction, sensitivity analysis, coefficient tables, figures, and methodology documentation.

**Architecture:** 
1. **Branch 1 (`feature/m2-eda`):** Data ingestion from local processed directory, EDA module (`src/statistics/eda.py`), automated tests (`tests/test_m2_eda.py`), and exploratory notebook (`notebooks/02_statistical_eda.ipynb`) covering all 33 families.
2. **Branch 2 (`feature/m2-panel-inference`):** Panel econometrics module (`src/statistics/panel_models.py`) implementing Pooled OLS, Entity Fixed Effects (`store_nbr × family`), and Random Effects using `linearmodels` / `statsmodels`, Clustered SEs by City-Day, BH-FDR correction, sensitivity checks, result exporting (`results/tables/rq1_coefficients.csv`), and documentation (`docs/rq1_assumptions.md`).

**Tech Stack:** Python 3.13, `pandas`, `pyarrow`, `numpy`, `scipy`, `statsmodels`, `linearmodels`, `matplotlib`, `seaborn`, `pytest`.

**Spec:** `Favorita_Proposal_v4.2_4weeks.docx` and `Favorita_Roles_v4.2_4weeks.docx` (Section 7, Section 12, Roles D.1-D.3, Gates G0 & G2).

## Global Constraints
- Target: `log1p(sales)` at grain `(date, store_nbr, family)`.
- Explanatory weather variables: `precipitation_sum`, `temperature_2m_max`, `temperature_2m_mean`, `relative_humidity_2m_mean`, `sunshine_hours`.
- FORBIDDEN weather variable: `wind_speed_10m_max` (excluded per QA-09 due to corrupted readings up to 783 km/h).
- Standard errors MUST be clustered (by `city` or `city_date`) to account for shared ERA5 grid cell observations across stores in the same city.
- Multiple hypothesis testing MUST be controlled via Benjamini-Hochberg FDR at $\alpha = 0.05$ (reporting both raw $p$-values and FDR $q$-values).
- Retrospective weather $t$ is permitted for association (RQ1), but all interpretations must strictly refrain from making causal claims.
- Deep-study family shortlist frozen to 7 families: `GROCERY I`, `BEVERAGES`, `CLEANING`, `DAIRY`, `BREAD/BAKERY`, `PERSONAL CARE`, `FROZEN FOODS`.

## Review Focus
- Handling of zero-sales observations without dropping valid active store days (`log1p(sales)`).
- Proper clustering: ensure cluster variable has at least 22 clusters (cities) or cluster by `city_date`.
- Singularity / Collinearity checks: time-invariant store attributes absorbed by Entity Fixed Effects.
- Sensitivity robustness: verify coefficient stability when excluding the 2016 Ecuador earthquake period (2016-04-16 to 2016-05-16).
- Format and schema contract of `results/tables/rq1_coefficients.csv` matching the protocol requirements.

---

## Phase 1: Environment & EDA (`feature/m2-eda`)

### Task 1: Data Setup & Integrity Check for M2
**Branch:** `feature/m2-eda` (created from `main`)

**Files:**
- Create: `tests/test_m2_data_ready.py`
- Copy: `d:\Study\PTTk\Data\4. Processed data\curated_sales_weather.parquet` -> `d:\Study\PTTk\LAB\data\processed\curated_sales_weather.parquet`
- Copy: `d:\Study\PTTk\Data\4. Processed data\model_features.parquet` -> `d:\Study\PTTk\LAB\data\processed\model_features.parquet`

**Interfaces:**
- Consumes: Parquet files in `d:\Study\PTTk\Data\4. Processed data\`
- Produces: Verified `data/processed/curated_sales_weather.parquet` in repository workspace

- [ ] **Step 1: Create branch `feature/m2-eda` and copy processed data into `LAB/data/processed/`**
  Command:
  ```powershell
  git checkout -b feature/m2-eda
  Copy-Item -Path "d:\Study\PTTk\Data\4. Processed data\curated_sales_weather.parquet" -Destination "d:\Study\PTTk\LAB\data\processed\"
  Copy-Item -Path "d:\Study\PTTk\Data\4. Processed data\curated_sales_weather.meta.json" -Destination "d:\Study\PTTk\LAB\data\processed\"
  Copy-Item -Path "d:\Study\PTTk\Data\4. Processed data\model_features.parquet" -Destination "d:\Study\PTTk\LAB\data\processed\"
  Copy-Item -Path "d:\Study\PTTk\Data\4. Processed data\model_features.meta.json" -Destination "d:\Study\PTTk\LAB\data\processed\"
  ```

- [ ] **Step 2: Write test `tests/test_m2_data_ready.py`**
  ```python
  from pathlib import Path
  import pandas as pd

  def test_curated_sales_weather_exists_and_valid():
      path = Path("data/processed/curated_sales_weather.parquet")
      assert path.exists(), "curated_sales_weather.parquet missing"
      df = pd.read_parquet(path)
      assert df.shape[0] == 3000888
      assert df["family"].nunique() == 33
      assert df["city"].nunique() == 22
      assert df["store_nbr"].nunique() == 54
      required_cols = {"date", "store_nbr", "family", "sales", "city", "precipitation_sum", "temperature_2m_max"}
      assert required_cols.issubset(df.columns)
  ```

- [ ] **Step 3: Run test to verify data readiness**
  Command: `python -m pytest tests/test_m2_data_ready.py -v`
  Expected: PASS

- [ ] **Step 4: Commit**
  ```powershell
  git add tests/test_m2_data_ready.py
  git commit -m "test(m2): verify local curated dataset availability and schema for M2"
  ```

---

### Task 2: 33-Family EDA Computation Module
**Branch:** `feature/m2-eda`

**Files:**
- Create: `src/statistics/eda.py`
- Create: `tests/test_m2_eda.py`

**Interfaces:**
- Produces:
  - `compute_family_summary(df: pd.DataFrame) -> pd.DataFrame`
  - `compute_weather_correlations(df: pd.DataFrame, weather_cols: list[str]) -> pd.DataFrame`
  - `compute_seasonality_stats(df: pd.DataFrame) -> dict[str, pd.DataFrame]`
  - `screen_deep_study_candidates(df: pd.DataFrame) -> pd.DataFrame`

- [ ] **Step 1: Write failing tests in `tests/test_m2_eda.py`**
  ```python
  import pandas as pd
  import numpy as np
  from src.statistics.eda import (
      compute_family_summary,
      compute_weather_correlations,
      compute_seasonality_stats,
      screen_deep_study_candidates
  )

  def test_compute_family_summary():
      sample = pd.DataFrame({
          "family": ["BEVERAGES", "BEVERAGES", "GROCERY I"],
          "sales": [10.0, 0.0, 50.0],
          "store_active": [1, 1, 1],
          "store_zero_day": [0, 0, 0]
      })
      summary = compute_family_summary(sample)
      assert "total_sales" in summary.columns
      assert "zero_rate" in summary.columns
      assert "sales_share" in summary.columns
      bev = summary.loc["BEVERAGES"]
      assert np.isclose(bev["zero_rate"], 0.5)

  def test_compute_weather_correlations():
      sample = pd.DataFrame({
          "family": ["BEVERAGES"] * 4,
          "sales": [10.0, 20.0, 30.0, 40.0],
          "temperature_2m_max": [20.0, 25.0, 30.0, 35.0],
          "precipitation_sum": [0.0, 5.0, 0.0, 10.0]
      })
      corr = compute_weather_correlations(sample, ["temperature_2m_max", "precipitation_sum"])
      assert "temperature_2m_max" in corr.columns
      assert corr.loc["BEVERAGES", "temperature_2m_max"] > 0.9
  ```

- [ ] **Step 2: Run test to verify failure**
  Command: `python -m pytest tests/test_m2_eda.py -v`
  Expected: FAIL (functions not defined)

- [ ] **Step 3: Implement functions in `src/statistics/eda.py`**
  Implement `compute_family_summary`, `compute_weather_correlations`, `compute_seasonality_stats`, and `screen_deep_study_candidates` ensuring robust handling of zero sales, missing days, and store active status.

- [ ] **Step 4: Run test to verify it passes**
  Command: `python -m pytest tests/test_m2_eda.py -v`
  Expected: PASS

- [ ] **Step 5: Commit**
  ```powershell
  git add src/statistics/eda.py tests/test_m2_eda.py
  git commit -m "feat(m2): implement statistical EDA calculation module for 33 families"
  ```

---

### Task 3: EDA Execution, Visualizations & Notebook
**Branch:** `feature/m2-eda`

**Files:**
- Create: `notebooks/02_statistical_eda.ipynb`
- Create: `scripts/run_m2_eda.py`
- Produces: `reports/figures/eda_family_sales_share.png`, `reports/figures/eda_zero_sales_distribution.png`, `reports/figures/eda_weather_correlation_heatmap.png`, `results/tables/eda_33_families_summary.csv`

**Interfaces:**
- Consumes: `src.statistics.eda`, `data/processed/curated_sales_weather.parquet`
- Produces: Reproducible summary tables and PNG figures in `reports/figures/` and `results/tables/`

- [ ] **Step 1: Write `scripts/run_m2_eda.py`**
  Script loads `curated_sales_weather.parquet`, calls `src.statistics.eda`, computes:
  - 33-family sales distribution, zero rate, and sales share.
  - Heatmap of weather-sales correlations across all 33 families.
  - Day-of-week and monthly seasonality.
  - Exports summary table to `results/tables/eda_33_families_summary.csv` and figures to `reports/figures/`.

- [ ] **Step 2: Execute script and verify generated tables and figures**
  Command: `python scripts/run_m2_eda.py`
  Expected: Output `results/tables/eda_33_families_summary.csv` (33 rows) and 3 PNG figures.

- [ ] **Step 3: Create clean executed `notebooks/02_statistical_eda.ipynb` showcasing the narrative**
  Narrative includes:
  - 33-family scale and zero-rate distribution.
  - Candidate screening verifying the 7-family deep study shortlist (`GROCERY I`, `BEVERAGES`, `CLEANING`, `DAIRY`, `BREAD/BAKERY`, `PERSONAL CARE`, `FROZEN FOODS`) and explaining `PRODUCE` exclusion.
  - Weather distribution across Sierra vs Costa cities.

- [ ] **Step 4: Commit and Merge `feature/m2-eda`**
  ```powershell
  git add scripts/run_m2_eda.py notebooks/02_statistical_eda.ipynb results/tables/eda_33_families_summary.csv reports/figures/eda_*.png
  git commit -m "feat(m2): complete 33-family EDA script, figures and notebook"
  git checkout main
  git merge --no-ff feature/m2-eda -m "Merge branch 'feature/m2-eda' into main"
  ```

---

## Phase 2: Panel Econometrics & RQ1 Inference (`feature/m2-panel-inference`)

### Task 4: Panel Econometric Models Implementation
**Branch:** `feature/m2-panel-inference` (created from updated `main`)

**Files:**
- Create: `src/statistics/panel_models.py`
- Create: `tests/test_panel_models.py`

**Interfaces:**
- Produces:
  - `estimate_pooled_ols(df: pd.DataFrame, weather_vars: list[str], controls: list[str], cluster_col: str) -> RegressionResultSummary`
  - `estimate_fixed_effects(df: pd.DataFrame, weather_vars: list[str], controls: list[str], entity_col: str, cluster_col: str) -> RegressionResultSummary`
  - `estimate_random_effects(df: pd.DataFrame, weather_vars: list[str], controls: list[str], entity_col: str) -> RegressionResultSummary`
  - `hausman_test(fe_result, re_result) -> dict`

- [ ] **Step 1: Create branch `feature/m2-panel-inference`**
  Command: `git checkout -b feature/m2-panel-inference`

- [ ] **Step 2: Write failing test in `tests/test_panel_models.py`**
  ```python
  import pandas as pd
  import numpy as np
  from src.statistics.panel_models import estimate_pooled_ols, estimate_fixed_effects

  def test_panel_models_synthetic():
      np.random.seed(42)
      n = 200
      df = pd.DataFrame({
          "store_nbr": np.random.choice([1, 2, 3], size=n),
          "family": np.random.choice(["BEVERAGES", "CLEANING"], size=n),
          "city": np.random.choice(["Quito", "Guayaquil"], size=n),
          "date": pd.date_range("2016-01-01", periods=n, freq="D"),
          "sales": np.random.exponential(100, size=n),
          "precipitation_sum": np.random.uniform(0, 20, size=n),
          "temperature_2m_max": np.random.uniform(15, 30, size=n),
          "onpromotion": np.random.poisson(5, size=n),
          "is_day_off": np.random.choice([0, 1], size=n)
      })
      df["entity_id"] = df["store_nbr"].astype(str) + "_" + df["family"]
      
      ols_res = estimate_pooled_ols(df, ["temperature_2m_max"], ["onpromotion", "is_day_off"], cluster_col="city")
      assert "temperature_2m_max" in ols_res.params
      assert ols_res.se_robust is not None

      fe_res = estimate_fixed_effects(df, ["temperature_2m_max"], ["onpromotion", "is_day_off"], entity_col="entity_id", cluster_col="city")
      assert "temperature_2m_max" in fe_res.params
      assert fe_res.entity_effects == "store_nbr_family"
  ```

- [ ] **Step 3: Run test to verify failure**
  Command: `python -m pytest tests/test_panel_models.py -v`
  Expected: FAIL

- [ ] **Step 4: Implement `src/statistics/panel_models.py`**
  - Use `linearmodels.panel.PanelOLS` and `RandomEffects` (or `statsmodels.regression.linear_model.OLS` with clustered covariance).
  - Ensure entity index is set to `(entity_id, time)`.
  - Implement cluster covariance computation by `city` or `city_date`.
  - Implement Hausman test calculation comparing $\hat{\beta}_{FE}$ and $\hat{\beta}_{RE}$.

- [ ] **Step 5: Run test to verify pass**
  Command: `python -m pytest tests/test_panel_models.py -v`
  Expected: PASS

- [ ] **Step 6: Commit**
  ```powershell
  git add src/statistics/panel_models.py tests/test_panel_models.py
  git commit -m "feat(m2): implement Pooled OLS, Entity Fixed Effects and Random Effects with clustered SEs"
  ```

---

### Task 5: FDR Correction, Heterogeneity & Sensitivity Analysis
**Branch:** `feature/m2-panel-inference`

**Files:**
- Modify: `src/statistics/panel_models.py`
- Modify: `tests/test_panel_models.py`

**Interfaces:**
- Produces:
  - `apply_benjamini_hochberg(p_values: pd.Series | np.ndarray, alpha: float = 0.05) -> tuple[np.ndarray, np.ndarray]`
  - `run_family_heterogeneity_models(df: pd.DataFrame, families: list[str]) -> pd.DataFrame`
  - `run_earthquake_sensitivity_check(df: pd.DataFrame, model_spec: str) -> pd.DataFrame`

- [ ] **Step 1: Write failing test in `tests/test_panel_models.py` for FDR and sensitivity**
  ```python
  from src.statistics.panel_models import apply_benjamini_hochberg

  def test_benjamini_hochberg():
      p_vals = pd.Series([0.001, 0.01, 0.04, 0.10, 0.80])
      reject, q_vals = apply_benjamini_hochberg(p_vals, alpha=0.05)
      assert q_vals[0] < 0.01
      assert q_vals[-1] >= 0.80
      assert reject[0] == True
  ```

- [ ] **Step 2: Run test to verify failure**
  Command: `python -m pytest tests/test_panel_models.py -k test_benjamini_hochberg -v`
  Expected: FAIL

- [ ] **Step 3: Implement BH-FDR correction, family heterogeneity, and sensitivity in `src/statistics/panel_models.py`**
  - Implement Benjamini-Hochberg ranking $q_i = \min_{k \ge i} \frac{m}{k} p_{(k)}$.
  - Implement loop over the 7 shortlisted families estimating family-specific weather elasticities.
  - Implement earthquake sensitivity excluding `2016-04-16` to `2016-05-16`.

- [ ] **Step 4: Run test to verify pass**
  Command: `python -m pytest tests/test_panel_models.py -v`
  Expected: PASS

- [ ] **Step 5: Commit**
  ```powershell
  git add src/statistics/panel_models.py tests/test_panel_models.py
  git commit -m "feat(m2): add Benjamini-Hochberg FDR correction and earthquake sensitivity analysis"
  ```

---

### Task 6: Run Full RQ1 Inferences, Export Results & Plots
**Branch:** `feature/m2-panel-inference`

**Files:**
- Create: `scripts/run_m2_rq1.py`
- Produces:
  - `results/tables/rq1_coefficients.csv`
  - `results/tables/rq1_model_comparison_ols_fe_re.csv`
  - `results/tables/rq1_sensitivity_earthquake.csv`
  - `reports/figures/rq1_weather_coefficients_forest.png`
  - `reports/figures/rq1_family_heterogeneity.png`

**Interfaces:**
- Output Schema for `results/tables/rq1_coefficients.csv`:
  `[family, region, model_spec, predictor, coef, se_robust, ci_lower_95, ci_upper_95, p_value, q_fdr, n_obs, r_squared]`

- [ ] **Step 1: Write execution script `scripts/run_m2_rq1.py`**
  Loads `curated_sales_weather.parquet`, filters for the 7 frozen families, runs:
  - Pooled OLS vs Fixed Effects vs Random Effects on all 7 families pooled.
  - Family-by-family Fixed Effects (heterogeneity).
  - Calculates clustered SEs by city.
  - Computes BH-FDR $q$-values.
  - Runs sensitivity excluding earthquake period.
  - Generates Forest Plots of 95% Confidence Intervals.
  - Exports CSV tables and PNG figures.

- [ ] **Step 2: Run the full inference pipeline**
  Command: `python scripts/run_m2_rq1.py`
  Expected: Execution completes cleanly, generating all tables and figures.

- [ ] **Step 3: Write verification test for results table schema**
  Add test in `tests/test_panel_models.py` checking that `results/tables/rq1_coefficients.csv` exists and strictly adheres to the required protocol columns.
  Command: `python -m pytest tests/test_panel_models.py -v`
  Expected: PASS

- [ ] **Step 4: Commit**
  ```powershell
  git add scripts/run_m2_rq1.py results/tables/rq1_*.csv reports/figures/rq1_*.png tests/test_panel_models.py
  git commit -m "feat(m2): execute full RQ1 panel regressions, export coefficients and forest plots"
  ```

---

### Task 7: Complete RQ1 Methodology & Assumptions Documentation
**Branch:** `feature/m2-panel-inference`

**Files:**
- Modify: `docs/rq1_assumptions.md` (fill from 0 bytes)

- [ ] **Step 1: Write comprehensive documentation in `docs/rq1_assumptions.md`**
  Covering:
  1. **Research Question & Hypotheses:** H1a, H1b formulation.
  2. **Model Formulation:** Mathematical specification of Fixed Effects, Entity definition (`store_nbr × family`), controls.
  3. **Standard Errors & Clustering Justification:** Mathematical explanation of why naive IID SE is invalid (Moulton problem / shared ERA5 grid) and justification of City-level clustering.
  4. **Multiple Testing Correction:** FDR Benjamini-Hochberg methodology and interpretation of $q$-values.
  5. **Non-Causal Epistemological Disclaimer:** Detailed justification of retrospective association vs causal inference.
  6. **Sensitivity Findings:** Robustness under earthquake omission.
  7. **Handoff to M4:** Key statistical sensitivities that inform M4's decision matrix uncertainty flags.

- [ ] **Step 2: Commit and Merge `feature/m2-panel-inference` into `main`**
  ```powershell
  git add docs/rq1_assumptions.md
  git commit -m "docs(m2): complete RQ1 assumptions, clustering justification, and methodology report"
  git checkout main
  git merge --no-ff feature/m2-panel-inference -m "Merge branch 'feature/m2-panel-inference' into main (Gate G2 Complete)"
  ```

---

## Plan Self-Review Checklist
- [x] Spec coverage: Covers 33-family EDA, 7-family deep study, Pooled OLS, Fixed Effects, Random Effects, Clustered SE, BH-FDR, sensitivity, and all required deliverable files.
- [x] Granular steps: Every step has a clear command, code, or test with expected outputs.
- [x] Type & interface consistency: File paths, column names (`sales`, `precipitation_sum`, `temperature_2m_max`, etc.), and function signatures are aligned.
- [x] Git branch hygiene: Structured cleanly into two isolated feature branches (`feature/m2-eda` and `feature/m2-panel-inference`) merged cleanly into `main`.

