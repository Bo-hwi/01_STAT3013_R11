"""M2 Exploratory Data Analysis module for the 33 product families in Favorita."""

from typing import Dict, List, Optional
import numpy as np
import pandas as pd


def compute_family_summary(df: pd.DataFrame, open_only: bool = True) -> pd.DataFrame:
    """Computes summary statistics per product family: total sales, sales share,
    zero-sales rate, mean, median, std, and number of observations.
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing at least 'family', 'sales'. Optionally 'store_active', 'store_zero_day'.
    open_only : bool, default True
        If True and store status columns exist, filter to open store-days only.
        
    Returns
    -------
    pd.DataFrame
        Indexed by 'family', sorted descending by total sales.
    """
    data = df
    if open_only and "store_active" in df.columns:
        mask = df["store_active"].eq(1)
        if "store_zero_day" in df.columns:
            mask = mask & df["store_zero_day"].eq(0)
        data = df.loc[mask]

    grouped = data.groupby("family")["sales"]
    
    total_sales = grouped.sum()
    grand_total = total_sales.sum()
    sales_share = total_sales / grand_total if grand_total > 0 else 0.0
    
    zero_count = grouped.apply(lambda s: (s <= 0).sum())
    total_count = grouped.count()
    zero_rate = zero_count / total_count
    
    summary = pd.DataFrame({
        "total_sales": total_sales,
        "sales_share": sales_share,
        "zero_rate": zero_rate,
        "mean_sales": grouped.mean(),
        "median_sales": grouped.median(),
        "std_sales": grouped.std(),
        "count": total_count,
    }).sort_values(by="total_sales", ascending=False)
    
    return summary


def compute_weather_correlations(
    df: pd.DataFrame, 
    weather_cols: List[str], 
    family_col: str = "family", 
    sales_col: str = "sales"
) -> pd.DataFrame:
    """Computes Pearson correlation between sales and each weather variable per family.
    
    Parameters
    ----------
    df : pd.DataFrame
        Dataset containing sales and weather columns.
    weather_cols : list of str
        List of numeric weather columns.
        
    Returns
    -------
    pd.DataFrame
        Indexed by family, with columns for each weather variable.
    """
    families = df[family_col].unique()
    results = {}
    
    for fam in families:
        fam_df = df.loc[df[family_col] == fam]
        row = {}
        for col in weather_cols:
            if col in fam_df.columns:
                sub = fam_df[[sales_col, col]].dropna()
                if len(sub) > 1 and sub[col].std() > 0 and sub[sales_col].std() > 0:
                    row[col] = float(np.corrcoef(sub[sales_col], sub[col])[0, 1])
                else:
                    row[col] = np.nan
            else:
                row[col] = np.nan
        results[fam] = row
        
    return pd.DataFrame.from_dict(results, orient="index")


def compute_seasonality_stats(
    df: pd.DataFrame, 
    sales_col: str = "sales"
) -> Dict[str, pd.DataFrame]:
    """Computes average sales by day of week and month."""
    stats = {}
    if "day_of_week" in df.columns:
        stats["dow"] = df.groupby("day_of_week")[sales_col].agg(["mean", "std", "count"])
    if "month" in df.columns:
        stats["month"] = df.groupby("month")[sales_col].agg(["mean", "std", "count"])
    return stats


def screen_deep_study_candidates(
    df: pd.DataFrame,
    zero_rate_threshold: float = 0.05,
    cutoff_date: Optional[str] = "2016-08-04",
) -> pd.DataFrame:
    """Screens candidate families according to the pre-registered protocol rule:
    - Zero rate <= 5%
    - Has sales across all 22 cities
    - First non-zero sale <= 2013-01-31
    - Evaluated before cutoff_date (fold F1 start).
    """
    data = df
    if cutoff_date and "date" in df.columns:
        data = data.loc[pd.to_datetime(data["date"]) < pd.to_datetime(cutoff_date)]
    if "store_active" in data.columns and "store_zero_day" in data.columns:
        data = data.loc[data["store_active"].eq(1) & data["store_zero_day"].eq(0)]
        
    summary = compute_family_summary(data, open_only=False)
    
    # Check cities and first sales date
    metrics = []
    for fam in summary.index:
        fam_df = data.loc[data["family"] == fam]
        n_cities = fam_df["city"].nunique() if "city" in fam_df.columns else 22
        non_zero = fam_df.loc[fam_df["sales"] > 0]
        first_sale = non_zero["date"].min() if len(non_zero) > 0 and "date" in non_zero.columns else None
        
        zero_rate = summary.loc[fam, "zero_rate"]
        share = summary.loc[fam, "sales_share"]
        
        passes_zero = zero_rate <= zero_rate_threshold
        passes_city = n_cities == 22
        passes_date = pd.to_datetime(first_sale) <= pd.to_datetime("2013-01-31") if first_sale else True
        passes_all = passes_zero and passes_city and passes_date
        
        metrics.append({
            "family": fam,
            "sales_share": share,
            "zero_rate": zero_rate,
            "n_cities": n_cities,
            "first_sale": str(first_sale)[:10] if first_sale else "N/A",
            "passes_screening": passes_all
        })
        
    return pd.DataFrame(metrics).set_index("family")

