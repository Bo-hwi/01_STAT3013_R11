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
- **Hypothesis 1b ($H_{1b}$ — Cross-Family & Location Heterogeneity):** The magnitude, sign, and statistical significance of weather associations differ systematically across product categories: weather-sensitive categories (e.g., `BEVERAGES`, `FROZEN FOODS`) exhibit pronounced weather elasticities, whereas staple non-perishable categories (e.g., `GROCERY I`, `CLEANING`) display dampened or negligible elasticities.

---

## 2. Econometric Specification & Model Formulation

### 2.1. Dependent Variable Transformation
Retail sales distributions exhibit severe positive skewness with zeros. We adopt the standard logarithmic transformation:
$$y_{s,f,t} = \log(1 + \text{Sales}_{s,f,t})$$
where $\text{Sales}_{s,f,t} \ge 0$ is the total recorded volume for store $s \in \{1, \dots, 54\}$, product family $f \in \{1, \dots, 7\}$, on date $t$.

### 2.2. Entity Fixed Effects with Calendar Controls (Primary Benchmark Specification)
The primary econometric model is a two-way / entity fixed-effects panel regression with calendar controls:
$$\log(1 + \text{Sales}_{s,f,t}) = \alpha_{s,f} + \beta^\top W_{g(s),t} + \theta^\top X_{s,f,t} + \gamma_{\text{calendar}}(t) + \varepsilon_{s,f,t}$$

- **Entity Unit $\alpha_{s,f}$:** A unique intercept for each `(store_nbr, family)` pair ($54 \times 7 = 378$ entity units). This absorbs **all time-invariant store and family characteristics** (e.g., store floor size, store format/type, demographic cluster, baseline neighborhood affluence, permanent category preferences).
- **Meteorological Vector $W_{g(s),t}$ (4 Clean Weather Regressors):**
  - $\text{precipitation\_sum}$ (Daily total precipitation in mm)
  - $\text{temperature\_2m\_max}$ (Daily maximum 2m temperature in °C)
  - $\text{relative\_humidity\_2m\_mean}$ (Daily average relative humidity in %)
  - $\text{sunshine\_hours}$ (Daily sunshine duration in hours)
  - *Multicollinearity Resolution:* $\text{temperature\_2m\_mean}$ is strictly excluded from all regression specifications due to an extreme empirical correlation ($r = 0.975$) with $\text{temperature\_2m\_max}$. Including both regressors induced severe variance inflation (VIF $> 20$), artificial coefficient sign-flipping, and uninterpretable elasticities. Retaining solely $\text{temperature\_2m\_max}$ aligns with `configs/ablation_M0_M5.yaml`.
  - *Sensor Exclusion (QA-09):* $\text{wind\_speed\_10m\_max}$ is strictly excluded due to documented sensor corruptions exceeding $780 \text{ km/h}$.
- **Calendar Fixed Effects $\gamma_{\text{calendar}}(t)$ (Proposal Section 7.1):**
  - Day-of-week dummies (`dow_1` to `dow_6`, baseline Monday): captures massive weekend shopping surges (+25% to +27%).
  - Month-of-year dummies (`month_2` to `month_12`, baseline January): absorbs seasonal fluctuations in weather (e.g., dry vs. rainy seasons, school vacations, December holiday shopping surge of +33.6%). Without monthly dummies, weather regressors erroneously absorb seasonal calendar demand shifts.
  - Annual secular dummies (`year_2014` to `year_2017`, baseline 2013): controls for multi-year chain expansion, macroeconomic inflation, and upward secular sales growth (+30% to +53%).
- **Business Control Covariates $X_{s,f,t}$:**
  - $\text{onpromotion}_{s,f,t}$: Number of promoted items within family $f$.
  - $\text{is\_day\_off}_t$: Binary indicator for weekends and statutory non-working holidays.
  - $\text{is\_payday}_t$: Binary indicator for the 15th and end-of-month pay cycles in Ecuador.
  - $\text{is\_national\_holiday}_t$: Nationwide public holidays.
  - $\text{is\_event}_t$: Major national cultural or sporting events.
  - $\text{is\_earthquake\_period}_t$: Binary indicator for the emergency recovery window following the 7.8 Mw Pedernales earthquake (2016-04-16 to 2016-05-16).

### 2.3. Hausman Specification Test (Fixed vs. Random Effects)
To formally evaluate the consistency of Random Effects against Fixed Effects, we compute the Hausman specification statistic:
$$H = (\hat{\beta}_{\text{FE}} - \hat{\beta}_{\text{RE}})' \left[ \widehat{\text{Var}}(\hat{\beta}_{\text{FE}}) - \widehat{\text{Var}}(\hat{\beta}_{\text{RE}}) \right]^{-1} (\hat{\beta}_{\text{FE}} - \hat{\beta}_{\text{RE}})$$
Fixed Effects serves as the robust primary estimator because it does not require orthogonality between unobserved entity traits and promotional or weather shocks ($\text{Cov}(\alpha_{s,f}, X_{s,f,t}) = 0$).

---

## 3. Standard Error Clustering & The Moulton Problem

### 3.1. Why Naive IID Standard Errors are Biased
In the Favorita data, 54 stores are located across 22 cities. If standard errors are computed assuming IID errors, we violate independence (the classical Moulton, 1986 clustering problem):
$$\text{Cov}(\varepsilon_{s,f,t}, \varepsilon_{s',f',t}) > 0 \quad \text{for } s \ne s' \text{ in the same climate grid.}$$
Treating multiple stores in the same area as separate independent weather observations drastically inflates degrees of freedom and deflates standard errors, producing spurious $p$-values.

### 3.2. Clustered Covariance Estimator on `weather_grid_id` (21 Clusters)
All standard errors in RQ1 (Pooled OLS, Fixed Effects, and Random Effects) are clustered at the **`weather_grid_id` level (21 clusters)**:
$$V_{\text{cluster}} = (X'X)^{-1} \left( \sum_{g=1}^{21} X_g' e_g e_g' X_g \right) (X'X)^{-1}$$
- **Data Dictionary & QA-07 Alignment:** While there are 22 cities, Salinas and Libertad share a single 0.25° ERA5 weather grid cell. Because weather shock treatment is assigned at the grid cell level, clustering must be defined at `weather_grid_id` (21 clusters) rather than city.
- **Random Effects Clustering:** Random Effects standard errors are explicitly clustered by `weather_grid_id`, correcting previously unclustered robust standard errors that produced hyper-inflated confidence (e.g., $p \approx 10^{-49}$). With clustering, RE standard errors ($0.0040$) closely mirror FE clustered errors ($0.0040$), providing sound econometric bounds.

---

## 4. Multiple Hypothesis Testing Correction (Benjamini–Hochberg FDR)

### 4.1. The Multiple Comparisons Risk
Evaluating 4 weather variables across 7 product families produces 28 family-specific hypothesis tests. Under a standard nominal threshold ($\alpha = 0.05$), the family-wise error rate exceeds $76\%$.

### 4.2. Benjamini–Hochberg (BH) False Discovery Rate Procedure
We implement the Benjamini–Hochberg (1995) algorithm to bound the expected proportion of false discoveries at $q^* = 0.05$:
1. Sort raw two-sided $p$-values in ascending order: $p_{(1)} \le p_{(2)} \le \dots \le p_{(m)}$, where $m = 28$.
2. For each rank $i$, calculate adjusted $q$-values:
   $$q_{(i)} = \min_{k \ge i} \left( \frac{m}{k} p_{(k)} \right)$$
3. Declare significance if $q_{(i)} \le 0.05$.

Both unadjusted $p$-values and BH-adjusted $q$-values are reported in [results/tables/rq1_coefficients.csv](file:///d:/Study/PTTk/LAB/results/tables/rq1_coefficients.csv).

---

## 5. Epistemological Boundary: Retrospective Associations vs. Causal Claims

> [!CAUTION]
> **Strict Non-Causal Research Boundary (Enforced across all Reports and Presentations):**
> 1. **No Causal Inferences:** The estimated panel coefficients $\hat{\beta}$ represent **controlled conditional statistical associations**, NOT causal effects. Omitted daily micro-confounders (e.g., local traffic jams, air conditioning failures, localized neighborhood events) remain unobserved.
> 2. **Retrospective Nature:** Weather metrics $W_t$ in RQ1 are derived from Open-Meteo ERA5 reanalysis. Because ERA5 observations are compiled retrospectively with publication lags, these regression associations explain *historical variance*, but cannot be claimed as an operational real-time forecasting capability without real-time feeds.

---

## 6. Empirical Findings Summary & Sensitivity Analysis

### 6.1. Empirical Regression Estimates (from `rq1_coefficients.csv`)
1. **Maximum Temperature ($\text{temperature\_2m\_max}$):**
   - Pooled 7-Family Network: $\hat{\beta} = +0.0119$ ($p = 0.0032$, $q_{\text{FDR}} = 0.0063$).
   - Interpretation: A 1°C increase in daily maximum temperature is associated with an estimated $+1.19\%$ increase in retail sales volume across the network, holding promotions, calendar, and entity effects fixed.
   - Family Heterogeneity: Statistically significant in `BEVERAGES` ($\hat{\beta} = +0.0376$, $+3.76\%$/°C, $q = 0.0006$) and `FROZEN FOODS` ($\hat{\beta} = +0.0369$, $+3.69\%$/°C, $q < 0.0001$).
2. **Precipitation ($\text{precipitation\_sum}$):**
   - Pooled 7-Family Network: $\hat{\beta} = -0.0005$ ($-0.05\%$/mm, $p = 0.0007$, $q_{\text{FDR}} = 0.0028$).
   - Family Heterogeneity: Statistically significant only in `BREAD/BAKERY` ($\hat{\beta} = -0.00067$, $-0.067\%$/mm, $q = 0.0009$). Other categories show no statistically significant rain sensitivity after FDR control.
3. **Relative Humidity & Sunshine:**
   - Pooled Humidity: $\hat{\beta} = +0.0009$ ($p = 0.0549$, $q_{\text{FDR}} = 0.0732$, insignificant).
   - Pooled Sunshine: $\hat{\beta} = +0.00058$ ($p = 0.5412$, $q_{\text{FDR}} = 0.5412$, insignificant). Once calendar month and day-of-week dummies are controlled for, sunshine hours no longer exhibit spurious network-wide significance.
4. **Weather-Insensitive Categories:**
   - 4 out of 7 categories (`GROCERY I`, `CLEANING`, `DAIRY`, `PERSONAL CARE`) show no statistically significant association with any weather regressor at $q < 0.05$ once calendar seasonality and entity fixed effects are conditioned upon.

### 6.2. Regional Heterogeneity (Sierra vs. Costa)
- **Sierra (Highlands):** $\text{temperature\_2m\_max}$ elasticity is $+0.57\%$/°C ($q < 10^{-9}$), sunshine elasticity is $+0.18\%$/hr ($q < 10^{-5}$), precipitation is insignificant ($p = 0.16$).
- **Costa (Coastal Lowlands):** $\text{temperature\_2m\_max}$ elasticity is **$+1.77\%$/°C** ($q = 0.041$), more than **3 times higher** than in the Sierra, and precipitation exhibits a negative association ($-0.046\%$/mm, $q = 0.041$).

### 6.3. Earthquake Sensitivity Robustness Check
Comparing full sample against the exclusion of the earthquake disruption window (2016-04-16 to 2016-05-16):
- $\text{temperature\_2m\_max}$ elasticity shifts by only **$+1.66\%$** (from $0.01187$ to $0.01207$).
- $\text{precipitation\_sum}$ elasticity shifts by only $+12.6\%$ (from $-0.00050$ to $-0.00044$).
- All statistical signs and significance levels are preserved, confirming robustness to catastrophic natural disaster shocks.

---

## 7. Handoff to M3 (ML) & M4 (Decisions)

1. **Feature Ablation Priority (for M3):**
   - Maximum temperature is the primary meteorological signal with strong category selectivity (`BEVERAGES` and `FROZEN FOODS`).
   - Precipitation exhibits minor negative effects for fresh bakery products.
   - Sunshine duration and humidity exhibit negligible independent predictive signal once calendar seasonality is modeled.
2. **Decision Matrix Statistical Input (for M4 — `weather_signatures_for_m4.csv`):**
   - **No Prescriptive Deployment Labels from RQ1:** RQ1 reports purely econometric association metrics (point estimates $\hat{\beta}$, semi-elasticity $\%$, standard errors, and FDR $q$-values).
   - **Boundary Requirement:** Prescriptive operational labels (such as "Prioritize pilot" or "Do not deploy") must not be assigned solely based on statistical $p$-values. Member 4 assigns Decision Matrix classifications by evaluating whether statistical sensitivity translates into out-of-sample forecast lift ($\Delta\text{RMSE}$, $\Delta\text{WAPE}$, DM-test from RQ2/M3) and exceeds the Newsvendor cost threshold ($C_u / (C_u + C_o)$).
