"""M2 Panel Econometric Models: Pooled OLS, Entity Fixed Effects, Random Effects,
Clustered Standard Errors, and Specification Diagnostics for RQ1.
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
    cluster_col: Optional[str] = "city",
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
    cluster_col: Optional[str] = "city",
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
    X = sub[x_cols]
    
    model = PanelOLS(y, X, entity_effects=True)
    
    if cluster_col and cluster_col in sub.columns:
        # Cluster standard errors by city
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
    outcome_col: str = "sales",
) -> Dict[str, Any]:
    """Estimates a Random Effects model: log1p(sales) ~ const + weather + controls."""
    data = df.copy()
    data["log_sales"] = np.log1p(data[outcome_col].clip(lower=0))
    
    x_cols = weather_vars + controls
    needed = ["log_sales", entity_col] + x_cols
    if "date" in data.columns:
        needed.append("date")
    sub = data[needed].dropna()
    
    if "date" in sub.columns:
        sub = sub.set_index([entity_col, "date"])
    else:
        sub["time_idx"] = sub.groupby(entity_col).cumcount()
        sub = sub.set_index([entity_col, "time_idx"])
        
    y = sub["log_sales"]
    X = sm.add_constant(sub[x_cols])
    
    model = RandomEffects(y, X)
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
