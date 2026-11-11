"""M2 classical hypothesis tests for RQ1 (step 1, before the panel models).

Why these tests?
    The course outline asks for t-tests / chi-square. They give a first, simple
    answer: "do sales differ on rainy / hot days?". They do NOT control for
    season, weekday, promotions or store size. The panel models in
    panel_models.py (step 2) add those controls. Comparing step 1 with step 2
    shows how much of a raw difference is really weather.

Unit of observation: city x family x day
    Weather is measured once per city per day, shared by every store in that
    city. Testing on store rows would count the same weather many times and
    make p-values far too small (pseudo-replication). So each family's sales
    are first averaged over the open stores of a city on each day:
        y = mean over open stores of log(1 + sales)

Three tests per family (7 families -> 21 tests, BH-FDR across all 21):
    1. Welch t-test   : y on rainy days (precipitation >= 1 mm) vs dry days.
                        Levene's test is reported to justify Welch (unequal variances).
    2. One-way ANOVA  : y across city-relative temperature terciles
                        (cool / normal / hot compared with that city's own days),
                        so coastal vs highland climate is not mistaken for weather.
    3. Chi-square     : rainy (yes/no) x high-sales day (y above the city median
                        for that family), 2 x 2 table, effect size Cramer's V.

Effect sizes are reported because with ~36,000 city-days per family even tiny
differences become "significant".
"""
import numpy as np
import pandas as pd
from scipy import stats

FAMILIES = ["GROCERY I", "BEVERAGES", "CLEANING", "DAIRY",
            "BREAD/BAKERY", "PERSONAL CARE", "FROZEN FOODS"]
RAIN_THRESHOLD_MM = 1.0   # standard "wet day" threshold
ALPHA = 0.05


# ---------------------------------------------------------------------------
# 1. Data: one row per city x family x day
# ---------------------------------------------------------------------------
def city_day_table(curated: pd.DataFrame) -> pd.DataFrame:
    """Average log-sales over open stores per city x family x day, attach weather."""
    open_rows = curated[curated["family"].isin(FAMILIES)
                        & curated["store_active"].eq(1)
                        & curated["store_zero_day"].eq(0)].copy()
    open_rows["log_sales"] = np.log1p(open_rows["sales"].clip(lower=0))
    open_rows["family"] = open_rows["family"].astype(str)
    open_rows["city"] = open_rows["city"].astype(str)

    table = (open_rows
             .groupby(["city", "family", "date"], observed=True)
             .agg(y=("log_sales", "mean"),
                  n_stores=("store_nbr", "nunique"),
                  precipitation_sum=("precipitation_sum", "first"),   # same for all stores in a city
                  temperature_2m_max=("temperature_2m_max", "first"))
             .reset_index())

    table["rainy"] = table["precipitation_sum"] >= RAIN_THRESHOLD_MM
    # Temperature tercile inside each city (each city compared with itself)
    table["temp_band"] = (table.groupby("city")["temperature_2m_max"]
                          .transform(lambda s: pd.qcut(s.rank(method="first"), 3,
                                                       labels=["cool", "normal", "hot"])))
    # High-sales day = above that city's median for that family
    median = table.groupby(["city", "family"])["y"].transform("median")
    table["high_sales"] = table["y"] > median
    return table


# ---------------------------------------------------------------------------
# 2. The three tests (each returns one result row)
# ---------------------------------------------------------------------------
def welch_rain_test(rows: pd.DataFrame) -> dict:
    """H0: mean log-sales equal on rainy and dry days. H1: they differ."""
    rainy, dry = rows.loc[rows["rainy"], "y"], rows.loc[~rows["rainy"], "y"]
    levene_p = stats.levene(rainy, dry).pvalue
    t = stats.ttest_ind(rainy, dry, equal_var=False)
    pooled_sd = np.sqrt((rainy.var() + dry.var()) / 2)
    diff = rainy.mean() - dry.mean()
    return {"test": "Welch t-test (rainy vs dry)", "statistic": t.statistic, "p_value": t.pvalue,
            "effect_name": "Cohen's d", "effect_size": diff / pooled_sd,
            "difference_pct": 100 * (np.exp(diff) - 1),   # log difference -> % difference
            "n": len(rows), "detail": f"rainy={len(rainy)}, dry={len(dry)}, Levene p={levene_p:.3g}"}


def anova_temperature_test(rows: pd.DataFrame) -> dict:
    """H0: mean log-sales equal across cool / normal / hot days. H1: at least one differs."""
    groups = [rows.loc[rows["temp_band"] == band, "y"] for band in ["cool", "normal", "hot"]]
    f = stats.f_oneway(*groups)
    grand = rows["y"].mean()
    ss_between = sum(len(g) * (g.mean() - grand) ** 2 for g in groups)
    ss_total = ((rows["y"] - grand) ** 2).sum()
    hot_vs_cool = groups[2].mean() - groups[0].mean()
    return {"test": "One-way ANOVA (temperature terciles)", "statistic": f.statistic, "p_value": f.pvalue,
            "effect_name": "eta squared", "effect_size": ss_between / ss_total,
            "difference_pct": 100 * (np.exp(hot_vs_cool) - 1),   # hot vs cool days
            "n": len(rows), "detail": "means cool/normal/hot = " + "/".join(f"{g.mean():.3f}" for g in groups)}


def chi_square_rain_test(rows: pd.DataFrame) -> dict:
    """H0: rainy days and high-sales days are independent. H1: they are associated."""
    table = pd.crosstab(rows["rainy"], rows["high_sales"])
    chi2, p, dof, _ = stats.chi2_contingency(table)
    cramers_v = np.sqrt(chi2 / (table.to_numpy().sum() * (min(table.shape) - 1)))
    share_rainy = rows.loc[rows["rainy"], "high_sales"].mean()
    share_dry = rows.loc[~rows["rainy"], "high_sales"].mean()
    return {"test": "Chi-square (rainy x high-sales day)", "statistic": chi2, "p_value": p,
            "effect_name": "Cramer's V", "effect_size": cramers_v,
            "difference_pct": 100 * (share_rainy - share_dry),   # percentage points
            "n": len(rows), "detail": f"high-sales share rainy={share_rainy:.3f}, dry={share_dry:.3f}, dof={dof}"}


# ---------------------------------------------------------------------------
# 3. Run all tests and correct for multiple testing
# ---------------------------------------------------------------------------
def benjamini_hochberg(p_values: pd.Series) -> pd.Series:
    """BH-FDR adjusted q-values (same rule as the panel models)."""
    p = p_values.to_numpy()
    order = np.argsort(p)
    ranked = p[order] * len(p) / np.arange(1, len(p) + 1)
    q = np.minimum.accumulate(ranked[::-1])[::-1].clip(max=1)
    out = np.empty_like(q)
    out[order] = q
    return pd.Series(out, index=p_values.index)


def run_all(table: pd.DataFrame) -> pd.DataFrame:
    results = []
    for family in FAMILIES:
        rows = table[table["family"] == family]
        for test in (welch_rain_test, anova_temperature_test, chi_square_rain_test):
            results.append({"family": family, **test(rows)})
    out = pd.DataFrame(results)
    out["q_fdr"] = benjamini_hochberg(out["p_value"])
    out["significant_fdr"] = out["q_fdr"] < ALPHA
    return out
