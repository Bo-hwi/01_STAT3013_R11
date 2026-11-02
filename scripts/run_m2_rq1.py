"""Execute full RQ1 panel econometric regressions, family heterogeneity,
regional climate comparison, FDR correction, sensitivity analysis,
and produce coefficient tables and figures.
"""

from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd

from src.statistics.panel_models import (
    estimate_pooled_ols,
    estimate_fixed_effects,
    estimate_random_effects,
    hausman_test,
    apply_benjamini_hochberg,
    run_family_heterogeneity_models,
    run_earthquake_sensitivity_check,
    estimate_regional_heterogeneity,
)

DATA_PATH = REPO_ROOT / "data" / "processed" / "curated_sales_weather.parquet"
RESULTS_DIR = REPO_ROOT / "results" / "tables"
FIGURES_DIR = REPO_ROOT / "reports" / "figures"

FROZEN_FAMILIES = [
    "GROCERY I",
    "BEVERAGES",
    "CLEANING",
    "DAIRY",
    "BREAD/BAKERY",
    "PERSONAL CARE",
    "FROZEN FOODS",
]

WEATHER_VARS = [
    "precipitation_sum",
    "temperature_2m_max",
    "temperature_2m_mean",
    "relative_humidity_2m_mean",
    "sunshine_hours",
]

CONTROLS = [
    "onpromotion",
    "is_day_off",
    "is_payday",
    "is_event",
    "is_national_holiday",
    "is_earthquake_period",
]


def main():
    print("Loading curated dataset for RQ1 inference...")
    df = pd.read_parquet(DATA_PATH)
    
    # Filter for the 7 frozen deep-study families on open store days
    mask = df["family"].isin(FROZEN_FAMILIES) & df["store_active"].eq(1) & df["store_zero_day"].eq(0)
    data = df.loc[mask].copy()
    data["entity_id"] = data["store_nbr"].astype(str) + "_" + data["family"].astype(str)
    data["date"] = pd.to_datetime(data["date"])
    
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    
    print(f"Data filtered: {len(data):,} observations across 7 families.")
    
    # 1. Overall Model Comparison: Pooled OLS, Fixed Effects, Random Effects
    print("1. Estimating Pooled OLS (Clustered by City)...")
    ols_res = estimate_pooled_ols(data, WEATHER_VARS, CONTROLS, cluster_col="city")
    
    print("2. Estimating Entity Fixed Effects (store_nbr x family, Clustered by City)...")
    fe_res = estimate_fixed_effects(data, WEATHER_VARS, CONTROLS, entity_col="entity_id", cluster_col="city")
    
    print("3. Estimating Random Effects...")
    re_res = estimate_random_effects(data, WEATHER_VARS, CONTROLS, entity_col="entity_id")
    
    h_test = hausman_test(fe_res, re_res)
    print(f"Hausman test: chi2 = {h_test['stat']:.2f}, p-value = {h_test['p_value']:.4e} -> Preferred: {h_test['preferred']}")
    
    # Export Model Comparison Table
    comp_rows = []
    for var in WEATHER_VARS:
        comp_rows.append({
            "predictor": var,
            "pooled_ols_coef": ols_res["params"].get(var),
            "pooled_ols_se": ols_res["std_errors"].get(var),
            "pooled_ols_pval": ols_res["p_values"].get(var),
            "fixed_effects_coef": fe_res["params"].get(var),
            "fixed_effects_se": fe_res["std_errors"].get(var),
            "fixed_effects_pval": fe_res["p_values"].get(var),
            "random_effects_coef": re_res["params"].get(var),
            "random_effects_se": re_res["std_errors"].get(var),
            "random_effects_pval": re_res["p_values"].get(var),
        })
    comp_df = pd.DataFrame(comp_rows)
    comp_df.to_csv(RESULTS_DIR / "rq1_model_comparison_ols_fe_re.csv", index=False)
    print(f"Saved: {RESULTS_DIR / 'rq1_model_comparison_ols_fe_re.csv'}")
    
    # 2. Family Heterogeneity: Run Fixed Effects per Family
    print("4. Estimating Family Heterogeneity Models (7 Families)...")
    het_df = run_family_heterogeneity_models(
        data,
        families=FROZEN_FAMILIES,
        weather_vars=WEATHER_VARS,
        controls=CONTROLS,
        entity_col="entity_id",
        cluster_col="city"
    )
    
    # Add overall Pooled FE rows to the coefficients table
    overall_rows = []
    for var in WEATHER_VARS:
        overall_rows.append({
            "family": "ALL_7_FAMILIES",
            "predictor": var,
            "model_spec": "Fixed Effects (Pooled)",
            "coef": fe_res["params"][var],
            "se_robust": fe_res["std_errors"][var],
            "ci_lower_95": fe_res["ci_lower"][var],
            "ci_upper_95": fe_res["ci_upper"][var],
            "p_value": fe_res["p_values"][var],
            "n_obs": fe_res["nobs"],
            "r_squared": fe_res["r_squared"]
        })
    overall_df = pd.DataFrame(overall_rows)
    reject_all, q_all = apply_benjamini_hochberg(overall_df["p_value"])
    overall_df["q_fdr"] = q_all
    overall_df["significant_fdr"] = reject_all
    
    rq1_coefs = pd.concat([overall_df, het_df], ignore_index=True)
    coef_path = RESULTS_DIR / "rq1_coefficients.csv"
    rq1_coefs.to_csv(coef_path, index=False)
    print(f"Saved: {coef_path}")
    
    # 3. Regional Heterogeneity: Sierra vs Costa
    print("5. Estimating Regional Climate Heterogeneity (Sierra vs Costa)...")
    reg_df = estimate_regional_heterogeneity(
        data,
        weather_vars=WEATHER_VARS,
        controls=CONTROLS,
        entity_col="entity_id",
        cluster_col="city"
    )
    reg_path = RESULTS_DIR / "rq1_regional_heterogeneity.csv"
    reg_df.to_csv(reg_path, index=False)
    print(f"Saved: {reg_path}")
    
    # Plot Regional Comparison
    print("6. Plotting Regional Comparison (Sierra vs Costa)...")
    fig, ax = plt.subplots(figsize=(9, 5))
    x_pos = np.arange(len(WEATHER_VARS))
    width = 0.35
    
    sierra_vals = reg_df[reg_df["region"] == "Sierra"].set_index("predictor").loc[WEATHER_VARS, "coef"].values
    costa_vals = reg_df[reg_df["region"] == "Costa"].set_index("predictor").loc[WEATHER_VARS, "coef"].values
    
    ax.bar(x_pos - width/2, sierra_vals, width, label="Sierra (Andes Highlands)", color="#2ca02c")
    ax.bar(x_pos + width/2, costa_vals, width, label="Costa (Coastal Lowlands)", color="#ff7f0e")
    
    ax.set_xticks(x_pos)
    ax.set_xticklabels([v.replace("_2m_", " ").replace("_", " ").title() for v in WEATHER_VARS], rotation=15, ha="right")
    ax.set_ylabel("Semi-Elasticity on log(1 + sales)")
    ax.set_title("Geographic Weather Sensitivity: Sierra vs Costa Fixed Effects")
    ax.axhline(0, color="black", linestyle="--", alpha=0.5)
    ax.legend()
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "rq1_regional_comparison.png", dpi=200)
    plt.close()
    
    # 4. Earthquake Sensitivity Check
    print("7. Running Earthquake Sensitivity Check...")
    sens_res = run_earthquake_sensitivity_check(
        data,
        weather_vars=WEATHER_VARS,
        controls=CONTROLS,
        entity_col="entity_id",
        cluster_col="city"
    )
    sens_rows = []
    for var in WEATHER_VARS:
        base_c = sens_res["baseline_fe"]["params"][var]
        base_se = sens_res["baseline_fe"]["std_errors"][var]
        no_eq_c = sens_res["no_earthquake_fe"]["params"][var]
        no_eq_se = sens_res["no_earthquake_fe"]["std_errors"][var]
        sens_rows.append({
            "predictor": var,
            "baseline_coef": base_c,
            "baseline_se": base_se,
            "no_earthquake_coef": no_eq_c,
            "no_earthquake_se": no_eq_se,
            "delta_coef": no_eq_c - base_c,
            "pct_change": ((no_eq_c - base_c) / abs(base_c)) * 100 if base_c != 0 else 0.0
        })
    sens_df = pd.DataFrame(sens_rows)
    sens_df.to_csv(RESULTS_DIR / "rq1_sensitivity_earthquake.csv", index=False)
    print(f"Saved: {RESULTS_DIR / 'rq1_sensitivity_earthquake.csv'}")
    
    # 5. Generate Figures
    print("8. Plotting Forest Plot of Weather Elasticities...")
    fig, ax = plt.subplots(figsize=(10, 7))
    y_pos = np.arange(len(FROZEN_FAMILIES))
    palette = {"temperature_2m_max": "#d62728", "precipitation_sum": "#1f77b4"}
    
    for i, var in enumerate(["temperature_2m_max", "precipitation_sum"]):
        v_df = het_df[het_df["predictor"] == var].set_index("family").loc[FROZEN_FAMILIES]
        offset = -0.15 if i == 0 else 0.15
        label = "Max Temperature (°C)" if var == "temperature_2m_max" else "Precipitation (mm)"
        ax.errorbar(
            v_df["coef"], y_pos + offset,
            xerr=[v_df["coef"] - v_df["ci_lower_95"], v_df["ci_upper_95"] - v_df["coef"]],
            fmt='o', color=palette[var], label=label, capsize=4, markersize=7
        )
        
    ax.axvline(0, color="black", linestyle="--", alpha=0.6)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(FROZEN_FAMILIES)
    ax.set_xlabel("Estimated Semi-Elasticity on log(1 + sales) with 95% Clustered CI")
    ax.set_title("RQ1 Weather Sensitivity: Fixed Effects Estimates by Product Family")
    ax.legend()
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "rq1_weather_coefficients_forest.png", dpi=200)
    plt.close()
    
    print("9. Plotting Family Heterogeneity Heatmap...")
    pivot_coef = het_df.pivot(index="family", columns="predictor", values="coef")
    pivot_sig = het_df.pivot(index="family", columns="predictor", values="significant_fdr")
    
    annot = pivot_coef.round(4).astype(str)
    for fam in pivot_coef.index:
        for pred in pivot_coef.columns:
            if pivot_sig.loc[fam, pred]:
                annot.loc[fam, pred] += "*"
                
    plt.figure(figsize=(9, 6))
    sns.heatmap(pivot_coef, cmap="coolwarm", center=0, annot=annot, fmt="", cbar_kws={"label": "Coefficient (Semi-Elasticity)"})
    plt.title("Weather Heterogeneity across 7 Families (* = FDR q < 0.05)")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "rq1_family_heterogeneity.png", dpi=200)
    plt.close()
    
    # 6. Generate Handoff Table for Member 4 (Decision Matrix Weather Signatures)
    print("10. Generating Handoff Table: weather_signatures_for_m4.csv...")
    signatures = []
    for fam in FROZEN_FAMILIES:
        fam_res = het_df[het_df["family"] == fam]
        sig_preds = fam_res[fam_res["significant_fdr"]]["predictor"].tolist()
        
        # Describe signature in human-readable and decision terms
        if "temperature_2m_mean" in sig_preds and "sunshine_hours" in sig_preds:
            desc = "High Sensitivity (Temp Mean + Sunshine)"
            uncertainty_flag = False
            rec_tag = "Prioritize pilot candidate"
        elif "temperature_2m_mean" in sig_preds or "sunshine_hours" in sig_preds:
            desc = "Moderate Sensitivity (Temp or Sunshine)"
            uncertainty_flag = False
            rec_tag = "Further validation candidate"
        else:
            desc = "Weather Insensitive (Staple/Controls dominate)"
            uncertainty_flag = True
            rec_tag = "Do not deploy (Hold baseline)"
            
        temp_mean_coef = fam_res.loc[fam_res["predictor"] == "temperature_2m_mean", "coef"].iloc[0]
        sun_coef = fam_res.loc[fam_res["predictor"] == "sunshine_hours", "coef"].iloc[0]
        rain_p = fam_res.loc[fam_res["predictor"] == "precipitation_sum", "p_value"].iloc[0]
        
        signatures.append({
            "family": fam,
            "weather_signature": desc,
            "significant_predictors": ", ".join(sig_preds) if sig_preds else "None",
            "temp_mean_elasticity_pct": round(temp_mean_coef * 100, 2),
            "sunshine_elasticity_pct_per_hr": round(sun_coef * 100, 3),
            "rain_significant_fdr": "precipitation_sum" in sig_preds,
            "uncertainty_flag": uncertainty_flag,
            "preliminary_m4_recommendation": rec_tag
        })
        
    sig_df = pd.DataFrame(signatures)
    sig_path = RESULTS_DIR / "weather_signatures_for_m4.csv"
    sig_df.to_csv(sig_path, index=False)
    print(f"Saved: {sig_path}")
    print(sig_df[["family", "weather_signature", "uncertainty_flag"]])
    
    print("\nAll Advanced Inferences & Handoff Tables Completed Successfully!")


if __name__ == "__main__":
    main()
