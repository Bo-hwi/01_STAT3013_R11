"""M2 Panel Econometric Models: Pooled OLS, Entity Fixed Effects, Random Effects,
Clustered Standard Errors, Regional Heterogeneity, and Specification Diagnostics for RQ1.
"""

from typing import Any, Dict, List, Optional
import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.api as sm
from linearmodels.panel import PanelOLS, RandomEffects


def estimate_pooled_ols(
    df: pd.DataFrame,
    weather_vars: List[str],
    controls: List[str],
    cluster_col: Optional[str] = "weather_grid_id",
    outcome_col: str = "sales",
) -> Dict[str, Any]:
    """Estimates a Pooled OLS model: log1p(sales) ~ const + weather + controls,
    with clustered standard errors.
    """
    data = df.copy()
    data["log_sales"] = np.log1p(data[outcome_col].clip(lower=0))
    
    x_cols = weather_vars + controls
    sub = data[["log_sales"] + x_cols + ([cluster_col] if cluster_col else [])].dropna()
    
    y = sub["log_sales"]
    X = sm.add_constant(sub[x_cols])
    
    model = sm.OLS(y, X)
    if cluster_col and cluster_col in sub.columns:
        res = model.fit(cov_type="cluster", cov_kwds={"groups": sub[cluster_col]})
    else:
        res = model.fit(cov_type="HC1")
        
    ci = res.conf_int(alpha=0.05)
    
    return {
        "model_type": "Pooled OLS",
        "nobs": int(res.nobs),
        "r_squared": float(res.rsquared),
        "params": res.params.to_dict(),
        "std_errors": res.bse.to_dict(),
        "p_values": res.pvalues.to_dict(),
        "ci_lower": ci[0].to_dict(),
        "ci_upper": ci[1].to_dict(),
        "cov_type": res.cov_type,
        "cov_matrix": res.cov_params(),
        "raw_result": res,
    }


def estimate_fixed_effects(
    df: pd.DataFrame,
    weather_vars: List[str],
    controls: List[str],
    entity_col: str = "entity_id",
    cluster_col: Optional[str] = "weather_grid_id",
    outcome_col: str = "sales",
) -> Dict[str, Any]:
    """Estimates an Entity Fixed Effects model: log1p(sales) ~ weather + controls + Entity_Effects,
    controlling for time-invariant unobserved heterogeneity across store_nbr x family.
    Standard errors are clustered by cluster_col.
    """
    data = df.copy()
    data["log_sales"] = np.log1p(data[outcome_col].clip(lower=0))
    
    x_cols = weather_vars + controls
    needed = ["log_sales", entity_col] + x_cols + ([cluster_col] if cluster_col else [])
    if "date" in data.columns:
        needed.append("date")
    sub = data[needed].dropna()
    
    # If date is not provided, create an integer time index per entity
    if "date" in sub.columns:
        sub = sub.set_index([entity_col, "date"])
    else:
        sub["time_idx"] = sub.groupby(entity_col).cumcount()
        sub = sub.set_index([entity_col, "time_idx"])
        
    y = sub["log_sales"]
    active_x = [c for c in x_cols if sub[c].nunique() > 1]
    X = sub[active_x]
    
    model = PanelOLS(y, X, entity_effects=True, check_rank=False)
    
    if cluster_col and cluster_col in sub.columns:
        # Cluster standard errors by weather_grid_id
        clusters = sub[cluster_col]
        res = model.fit(cov_type="clustered", clusters=clusters)
    else:
        res = model.fit(cov_type="robust")
        
    ci = res.conf_int(level=0.95)
    
    return {
        "model_type": "Fixed Effects",
        "entity_effects": "store_nbr_family",
        "nobs": int(res.nobs),
        "r_squared": float(res.rsquared_within),
        "r_squared_overall": float(res.rsquared_overall),
        "params": res.params.to_dict(),
        "std_errors": res.std_errors.to_dict(),
        "p_values": res.pvalues.to_dict(),
        "ci_lower": ci["lower"].to_dict(),
        "ci_upper": ci["upper"].to_dict(),
        "cov_type": "clustered" if (cluster_col and cluster_col in sub.columns) else "robust",
        "cov_matrix": res.cov,
        "raw_result": res,
    }


def estimate_random_effects(
    df: pd.DataFrame,
    weather_vars: List[str],
    controls: List[str],
    entity_col: str = "entity_id",
    cluster_col: Optional[str] = "weather_grid_id",
    outcome_col: str = "sales",
) -> Dict[str, Any]:
    """Estimates a Random Effects model: log1p(sales) ~ const + weather + controls,
    with clustered standard errors.
    """
    data = df.copy()
    data["log_sales"] = np.log1p(data[outcome_col].clip(lower=0))
    
    x_cols = weather_vars + controls
    needed = ["log_sales", entity_col] + x_cols + ([cluster_col] if cluster_col else [])
    if "date" in data.columns:
        needed.append("date")
    sub = data[needed].dropna()
    
    if "date" in sub.columns:
        sub = sub.set_index([entity_col, "date"])
    else:
        sub["time_idx"] = sub.groupby(entity_col).cumcount()
        sub = sub.set_index([entity_col, "time_idx"])
        
    y = sub["log_sales"]
    active_x = [c for c in x_cols if sub[c].nunique() > 1]
    X = sm.add_constant(sub[active_x])
    
    model = RandomEffects(y, X, check_rank=False)
    if cluster_col and cluster_col in sub.columns:
        clusters = sub[cluster_col]
        res = model.fit(cov_type="clustered", clusters=clusters)
    else:
        res = model.fit(cov_type="robust")
    ci = res.conf_int(level=0.95)
    
    return {
        "model_type": "Random Effects",
        "nobs": int(res.nobs),
        "r_squared": float(res.rsquared_overall),
        "params": res.params.to_dict(),
        "std_errors": res.std_errors.to_dict(),
        "p_values": res.pvalues.to_dict(),
        "ci_lower": ci["lower"].to_dict(),
        "ci_upper": ci["upper"].to_dict(),
        "cov_type": "clustered" if (cluster_col and cluster_col in sub.columns) else "robust",
        "cov_matrix": res.cov,
        "raw_result": res,
    }


def hausman_test(fe_res: Dict[str, Any], re_res: Dict[str, Any]) -> Dict[str, Any]:
    """Computes the Hausman specification test statistic comparing FE and RE:
    H = (b_FE - b_RE)' * [Var(b_FE) - Var(b_RE)]^(-1) * (b_FE - b_RE)
    H0: Random Effects is consistent and efficient.
    H1: Fixed Effects is consistent, RE is inconsistent (endogenous individual effects).
    """
    common_vars = [k for k in fe_res["params"].keys() if k in re_res["params"] and k != "const"]
    
    b_fe = np.array([fe_res["params"][k] for k in common_vars])
    b_re = np.array([re_res["params"][k] for k in common_vars])
    diff = b_fe - b_re
    
    cov_fe = fe_res["cov_matrix"].loc[common_vars, common_vars].to_numpy()
    cov_re = re_res["cov_matrix"].loc[common_vars, common_vars].to_numpy()
    cov_diff = cov_fe - cov_re
    
    try:
        inv_cov_diff = np.linalg.pinv(cov_diff)
        stat = float(diff.T @ inv_cov_diff @ diff)
        df = len(common_vars)
        p_val = float(1 - stats.chi2.cdf(stat, df))
    except Exception:
        stat = np.nan
        df = len(common_vars)
        p_val = np.nan
        
    return {
        "stat": stat,
        "df": df,
        "p_value": p_val,
        "preferred": "Fixed Effects" if (np.isnan(p_val) or p_val < 0.05) else "Random Effects",
    }


def apply_benjamini_hochberg(
    p_values: pd.Series | np.ndarray,
    alpha: float = 0.05
) -> tuple[np.ndarray, np.ndarray]:
    """Applies Benjamini-Hochberg (BH) False Discovery Rate (FDR) control.
    
    Returns
    -------
    reject : np.ndarray of bool
        True if hypothesis is rejected at FDR level alpha.
    q_values : np.ndarray of float
        Adjusted p-values (q-values).
    """
    p_arr = np.asarray(p_values, dtype=float)
    n = len(p_arr)
    if n == 0:
        return np.array([], dtype=bool), np.array([], dtype=float)
        
    order = np.argsort(p_arr)
    sorted_p = p_arr[order]
    
    # Calculate BH adjusted p-values
    q_sorted = np.zeros(n, dtype=float)
    q_sorted[-1] = sorted_p[-1]
    
    for i in range(n - 2, -1, -1):
        rank = i + 1
        q_sorted[i] = min(q_sorted[i + 1], sorted_p[i] * n / rank)
        
    q_values = np.zeros(n, dtype=float)
    q_values[order] = np.clip(q_sorted, 0.0, 1.0)
    reject = q_values <= alpha
    
    return reject, q_values


def run_family_heterogeneity_models(
    df: pd.DataFrame,
    families: List[str],
    weather_vars: List[str],
    controls: List[str],
    entity_col: str = "entity_id",
    cluster_col: Optional[str] = "weather_grid_id"
) -> pd.DataFrame:
    """Estimates Fixed Effects models for each family separately to examine heterogeneity,
    then applies Benjamini-Hochberg FDR across all estimated weather coefficients.
    """
    rows = []
    
    for fam in families:
        fam_df = df.loc[df["family"] == fam]
        if len(fam_df) < 50:
            continue
            
        fe_res = estimate_fixed_effects(
            fam_df,
            weather_vars=weather_vars,
            controls=controls,
            entity_col=entity_col,
            cluster_col=cluster_col
        )
        
        for var in weather_vars:
            if var in fe_res["params"]:
                rows.append({
                    "family": fam,
                    "predictor": var,
                    "model_spec": "Fixed Effects",
                    "coef": fe_res["params"][var],
                    "se_robust": fe_res["std_errors"][var],
                    "ci_lower_95": fe_res["ci_lower"][var],
                    "ci_upper_95": fe_res["ci_upper"][var],
                    "p_value": fe_res["p_values"][var],
                    "n_obs": fe_res["nobs"],
                    "r_squared": fe_res["r_squared"]
                })
                
    res_df = pd.DataFrame(rows)
    if not res_df.empty:
        reject, q_vals = apply_benjamini_hochberg(res_df["p_value"])
        res_df["q_fdr"] = q_vals
        res_df["significant_fdr"] = reject
    return res_df


def run_earthquake_sensitivity_check(
    df: pd.DataFrame,
    weather_vars: List[str],
    controls: List[str],
    entity_col: str = "entity_id",
    cluster_col: Optional[str] = "weather_grid_id"
) -> Dict[str, Any]:
    """Compares Fixed Effects estimates with and without the 2016 Ecuador earthquake period
    (2016-04-16 to 2016-05-16).
    """
    # Baseline with full period
    base_fe = estimate_fixed_effects(
        df, weather_vars=weather_vars, controls=controls, entity_col=entity_col, cluster_col=cluster_col
    )
    
    # Filter out earthquake period
    if "date" in df.columns:
        no_eq_df = df.loc[~df["date"].between("2016-04-16", "2016-05-16")]
    elif "is_earthquake_period" in df.columns:
        no_eq_df = df.loc[df["is_earthquake_period"].eq(0)]
    else:
        no_eq_df = df
        
    no_eq_controls = [c for c in controls if c != "is_earthquake_period"]
    no_eq_fe = estimate_fixed_effects(
        no_eq_df, weather_vars=weather_vars, controls=no_eq_controls, entity_col=entity_col, cluster_col=cluster_col
    )
    
    return {
        "baseline_fe": base_fe,
        "no_earthquake_fe": no_eq_fe
    }


SIERRA_CITIES = {
    "Quito", "Cayambe", "Cuenca", "Ambato", "Riobamba",
    "Latacunga", "Ibarra", "Loja", "Guaranda"
}
COSTA_CITIES = {
    "Guayaquil", "Daule", "Playas", "Libertad", "Salinas",
    "Manta", "El Carmen", "Machala", "Babahoyo", "Quevedo",
    "Esmeraldas", "Santo Domingo"
}


def estimate_regional_heterogeneity(
    df: pd.DataFrame,
    weather_vars: List[str],
    controls: List[str],
    entity_col: str = "entity_id",
    cluster_col: Optional[str] = "weather_grid_id"
) -> pd.DataFrame:
    """Estimates Fixed Effects models separately for Sierra (highlands) and Costa (coastal lowlands)
    to evaluate geographical climate heterogeneity, with BH-FDR adjustment.
    """
    data = df.copy()
    data["region"] = np.where(
        data["city"].isin(SIERRA_CITIES), "Sierra",
        np.where(data["city"].isin(COSTA_CITIES), "Costa", "Other")
    )
    
    rows = []
    for region in ["Sierra", "Costa"]:
        reg_df = data.loc[data["region"] == region]
        if len(reg_df) < 50:
            continue
        fe_res = estimate_fixed_effects(
            reg_df, weather_vars=weather_vars, controls=controls,
            entity_col=entity_col, cluster_col=cluster_col
        )
        for var in weather_vars:
            if var in fe_res["params"]:
                rows.append({
                    "region": region,
                    "predictor": var,
                    "model_spec": f"Fixed Effects ({region})",
                    "coef": fe_res["params"][var],
                    "se_robust": fe_res["std_errors"][var],
                    "ci_lower_95": fe_res["ci_lower"][var],
                    "ci_upper_95": fe_res["ci_upper"][var],
                    "p_value": fe_res["p_values"][var],
                    "n_obs": fe_res["nobs"],
                    "r_squared": fe_res["r_squared"]
                })
                
    res_df = pd.DataFrame(rows)
    if not res_df.empty:
        reject, q_vals = apply_benjamini_hochberg(res_df["p_value"])
        res_df["q_fdr"] = q_vals
        res_df["significant_fdr"] = reject
    return res_df
