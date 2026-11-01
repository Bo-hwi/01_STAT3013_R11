"""Single feature-name contract confirmed by M1."""

FEATURES = {
    "sales_lags": {1: "sales_lag1", 7: "sales_lag7", 14: "sales_lag14", 28: "sales_lag28"},
    "sales_rolling": {7: "sales_roll7_mean", 28: "sales_roll28_mean"},
    "categorical": ["store_nbr", "family", "city"],
    "calendar": [
        "day_of_week", "month", "is_day_off", "is_national_holiday",
        "is_regional_holiday", "is_local_holiday", "is_event", "is_work_day",
        "is_payday", "is_earthquake_period",
    ],
    # Keep promotion excluded until advance availability is verified.
    "conditional": ["onpromotion"],
    "weather": [
        "precipitation_sum_lag1", "temperature_2m_max_lag1",
        "relative_humidity_2m_mean_lag1", "sunshine_hours_lag1",
    ],
    "score_mask": "score_mask",
}


def m0_numeric() -> list[str]:
    return [*FEATURES["sales_lags"].values(), *FEATURES["sales_rolling"].values(), *FEATURES["calendar"]]
