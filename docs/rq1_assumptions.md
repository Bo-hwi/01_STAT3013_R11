# RQ1 Methodology, Econometric Specifications & Assumptions

> **Owner:** Member 2 (M2 — Statistical Research & Panel Inference)  
> **Course:** STAT3013 · **Project:** Weather × Favorita  
> **Repository:** `Bo-hwi/01_STAT3013_R11`  
> **Delivery Date:** 02/11/2026 (Gate G2 Deliverable)  
> **Reference Specs:** `Favorita_Proposal_v4.2`, Section 7 & `Favorita_Roles_v4.2`, Section D.  

---

## 1. Research Question 1 (RQ1) & Formal Hypotheses

### 1.1. Core Research Question
> *"How are different weather conditions statistically associated with daily retail sales, and how do these relationships vary across product families and geographic locations after conditioning on known business calendar and promotional controls?"*

### 1.2. Hypotheses
- **Hypothesis 1a ($H_{1a}$ — Network Association):** After conditioning on store promotions, calendar seasonality, paydays, and store-family fixed unobserved attributes, at least one meteorological variable ($\text{temperature}$, $\text{sunshine}$, $\text{precipitation}$) exhibits a statistically non-zero association with daily sales at the $\alpha = 0.05$ significance level under Benjamini–Hochberg False Discovery Rate (FDR) control.
- **Hypothesis 1b ($H_{1b}$ — Cross-Family & Location Heterogeneity):** The magnitude, sign, and statistical significance of weather associations differ systematically across product categories: weather-sensitive categories (e.g., `BEVERAGES`, `DAIRY`, `FROZEN FOODS`) exhibit pronounced weather elasticities, whereas staple non-perishable categories (e.g., `GROCERY I`, `CLEANING`) display dampened or negligible elasticities.

---

## 2. Econometric Specification & Model Formulation

### 2.1. Dependent Variable Transformation
Retail sales distributions exhibit severe positive skewness with zeros. We adopt the standard logarithmic transformation:
$$y_{s,f,t} = \log(1 + \text{Sales}_{s,f,t})$$
where $\text{Sales}_{s,f,t} \ge 0$ is the total recorded volume for store $s \in \{1, \dots, 54\}$, product family $f \in \{1, \dots, 7\}$, on date $t$.

### 2.2. Entity Fixed Effects (Primary Benchmark Specification)
The primary econometric model is a two-way / entity fixed-effects panel regression:
$$y_{s,f,t} = \alpha_{s,f} + \beta^T W_{\text{city}(s),t} + \theta^T X_{s,f,t} + \varepsilon_{s,f,t}$$

- **Entity Unit $\alpha_{s,f}$:** A unique intercept for each `(store_nbr, family)` pair ($54 \times 7 = 378$ entity units). This absorbs **all time-invariant store and family characteristics** (e.g., store floor size, store format/type, demographic cluster, baseline neighborhood affluence, permanent category preferences).
- **Meteorological Vector $W_{\text{city}(s),t}$:**
  - $\text{precipitation\_sum}$ (Daily total precipitation in mm)
  - $\text{temperature\_2m\_max}$ (Daily maximum 2m temperature in °C)
  - $\text{temperature\_2m\_mean}$ (Daily average 2m temperature in °C)
  - $\text{relative\_humidity\_2m\_mean}$ (Daily average relative humidity in %)
  - $\text{sunshine\_hours}$ (Daily sunshine duration in hours)
  - *Exclusion Note (QA-09):* $\text{wind\_speed\_10m\_max}$ is strictly excluded from all specifications due to documented sensor corruptions reaching non-physical values ($>780 \text{ km/h}$).
- **Control Covariates $X_{s,f,t}$:**
  - $\text{onpromotion}_{s,f,t}$: Number of promoted items within family $f$.
  - $\text{is\_day\_off}_t$: Binary indicator for weekends and statutory non-working holidays.
  - $\text{is\_payday}_t$: Binary indicator for the 15th and end-of-month government/private pay cycles in Ecuador.
  - $\text{is\_national\_holiday}_t$: Indicator for nationwide public holidays.
  - $\text{is\_event}_t$: Major national cultural or sporting events.
  - $\text{is\_earthquake\_period}_t$: Binary indicator for the emergency recovery window following the 7.8 Mw Pedernales earthquake (2016-04-16 to 2016-05-16).

### 2.3. Hausman Specification Test (Fixed vs. Random Effects)
To formally evaluate the consistency of Random Effects against Fixed Effects, we compute the Hausman specification statistic:
$$H = (\hat{\beta}_{\text{FE}} - \hat{\beta}_{\text{RE}})' \left[ \widehat{\text{Var}}(\hat{\beta}_{\text{FE}}) - \widehat{\text{Var}}(\hat{\beta}_{\text{RE}}) \right]^{-1} (\hat{\beta}_{\text{FE}} - \hat{\beta}_{\text{RE}})$$
- In the Favorita retail setting, unobserved store-family management practices and local traffic are plausibly correlated with promotions and location-based weather patterns ($\text{Cov}(\alpha_{s,f}, X_{s,f,t}) \ne 0$).
- Fixed Effects is chosen as the **unconditionally consistent primary estimator** because it does not require orthogonality between individual effects and explanatory variables.

---

## 3. Standard Error Clustering & The Moulton Problem

### 3.1. Why Naive IID Standard Errors are Biased
In the Favorita data, 54 stores are located across 22 cities (e.g., Quito has 18 stores; Guayaquil has 8 stores). Crucially, **all stores located within the same city share identical meteorological measurements from the city-level ERA5 grid cell on day $t$**.

If standard errors are computed assuming independent and identically distributed (IID) errors, we violate the fundamental independence assumption (the classical Moulton, 1986 clustering problem):
$$\text{Cov}(\varepsilon_{s,f,t}, \varepsilon_{s',f',t}) > 0 \quad \text{for } s \ne s' \text{ in the same city.}$$
Treating multiple stores in the same city as separate independent weather observations drastically inflates the effective degrees of freedom, compressing standard errors by up to $300\text{--}500\%$, yielding spurious $t$-statistics and false significance ($p < 0.001$).

### 3.2. Clustered Covariance Estimator
To produce valid, conservative inferential bounds, all standard errors and confidence intervals in RQ1 are computed using **Cluster-Robust Covariance Estimation** clustered at the **City level** (22 clusters):
$$V_{\text{cluster}} = (X'X)^{-1} \left( \sum_{g=1}^{G} X_g' e_g e_g' X_g \right) (X'X)^{-1}$$
where $g$ indexes the 22 geographic cities. This estimator accounts for arbitrary contemporaneous cross-store error correlation and within-city serial autocorrelation.

---

## 4. Multiple Hypothesis Testing Correction (Benjamini–Hochberg FDR)

### 4.1. The Multiple Comparisons Risk
Evaluating 5 weather variables across 7 product families produces 35 distinct hypothesis tests simultaneously. Under a standard nominal threshold ($\alpha = 0.05$), the family-wise error rate (FWER) of detecting at least one false positive purely by chance exceeds:
$$1 - (1 - 0.05)^{35} \approx 83.4\%$$

### 4.2. Benjamini–Hochberg (BH) False Discovery Rate Procedure
We implement the Benjamini–Hochberg (1995) algorithm to bound the expected proportion of false discoveries at $q^* = 0.05$:
1. Sort raw two-sided $p$-values in ascending order: $p_{(1)} \le p_{(2)} \le \dots \le p_{(m)}$, where $m = 35$.
2. For each rank $i$, calculate adjusted $q$-values:
   $$q_{(i)} = \min_{k \ge i} \left( \frac{m}{k} p_{(k)} \right)$$
3. Declare significance if $q_{(i)} \le 0.05$.

Both the unadjusted $p$-values and the BH-adjusted $q$-values are explicitly reported in [results/tables/rq1_coefficients.csv](file:///d:/Study/PTTk/LAB/results/tables/rq1_coefficients.csv).

---

## 5. Epistemological Boundary: Retrospective Associations vs. Causal Claims

> [!CAUTION]
> **Strict Non-Causal Research Boundary (Enforced across all Reports and Presentations):**
> 1. **No Causal Inferences:** The estimated panel coefficients $\hat{\beta}$ represent **controlled conditional statistical associations**, NOT causal effects. Omitted daily micro-confounders (e.g., local traffic jams, air conditioning failures, localized neighborhood events) remain unobserved.
> 2. **Retrospective Nature:** Weather metrics $W_t$ in RQ1 are derived from Open-Meteo ERA5 reanalysis. Because ERA5 observations are compiled retrospectively with publication lags, these regression associations explain *historical variance*, but cannot be claimed as an operational real-time forecasting capability without real-time feeds.

---

## 6. Empirical Findings Summary & Sensitivity Analysis

### 6.1. Empirical Regression Estimates (from `rq1_coefficients.csv`)
1. **Mean Temperature ($\text{temperature\_2m\_mean}$):**
   - Pooled 7-Family Network: $\hat{\beta} = +0.0634$ ($p = 1.96 \times 10^{-14}$, $q_{\text{FDR}} = 4.89 \times 10^{-14}$).
   - Interpretation: A 1°C increase in daily average temperature is associated with an estimated $+6.34\%$ increase in retail sales volume across the network, holding promotions and calendar fixed.
   - Heterogeneity: Highly prominent in `DAIRY` ($\hat{\beta} = +0.1033$, $q < 10^{-21}$), `BEVERAGES` ($\hat{\beta} = +0.0909$, $q < 10^{-7}$), and `FROZEN FOODS` ($\hat{\beta} = +0.0907$, $q < 10^{-7}$).
2. **Sunshine Hours ($\text{sunshine\_hours}$):**
   - Pooled 7-Family Network: $\hat{\beta} = +0.0060$ ($p = 1.17 \times 10^{-20}$, $q_{\text{FDR}} = 5.86 \times 10^{-20}$).
   - Interpretation: Each additional hour of sunshine is associated with a $+0.60\%$ increase in daily sales.
   - Heterogeneity: Strongest in `FROZEN FOODS` (+1.15%/hr) and `DAIRY` (+1.11%/hr).
3. **Precipitation ($\text{precipitation\_sum}$):**
   - Pooled 7-Family Network: $\hat{\beta} = +0.0001$ ($p = 0.5702$, $q_{\text{FDR}} = 0.5702$).
   - Statistical finding: Total rainfall has **no statistically significant network-wide relationship** with sales after controlling for entity effects and days off ($q_{\text{FDR}} > 0.05$).

### 6.2. Earthquake Sensitivity Robustness Check
Following the devastating 16/04/2016 Pedernales earthquake, commercial activity in Ecuador experienced massive disruption and emergency provisioning.
We compared estimates with the full sample versus excluding the earthquake window (2016-04-16 to 2016-05-16):
- $\text{temperature\_2m\_mean}$ elasticity shifts by only **$-1.85\%$** (from $0.0634$ to $0.0622$).
- $\text{sunshine\_hours}$ elasticity shifts by only **$-5.02\%$** (from $0.0060$ to $0.0057$).
- Both retain identical statistical significance and sign. The econometric findings are confirmed robust to extreme natural disaster shocks.

---

## 7. Handoff to M3 (ML) & M4 (Decisions)

1. **Feature Ablation Priority (for M3):**
   - Temperature mean and sunshine duration demonstrate the most robust statistical signal.
   - Precipitation exhibits high noise and localized insignificance; models adding precipitation alone (M1) should not be expected to generate large out-of-sample improvements over M0.
2. **Decision Matrix Uncertainty Flags (for M4):**
   - Categories with confirmed significant weather sensitivities (`BEVERAGES`, `DAIRY`, `FROZEN FOODS`) should be tagged `Prioritize pilot` if forecasting lifts verify cost reductions.
   - Categories where weather coefficients are statistically indistinguishable from zero (`BREAD/BAKERY` for sunshine/temp mean, `GROCERY I` for precipitation) should carry an explicit `Uncertainty Flag = True` in the final Decision Matrix.

