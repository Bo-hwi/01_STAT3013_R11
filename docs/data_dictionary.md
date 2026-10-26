# Data Dictionary — `curated_sales_weather.parquet`

**Owner:** M1 · **Version:** curated v1 (27/10/2026) · **Built by:** `python scripts/run_pipeline.py`
**Location:** `data/processed/curated_sales_weather.parquet` (local only, not committed) + `curated_sales_weather.meta.json` (row counts, package versions, SHA-256 of every input file)

## Grain and size

- **One row = one `(date, store_nbr, family)`**, unique key.
- 3,000,888 rows × 35 columns; 1,684 dates (01/01/2013–15/08/2017, without the four 25 December days); 54 stores; 33 families; 22 cities; 21 weather grid cells.
- Driving table is `train.csv`; all other tables are joined many-to-one, so no row is added, dropped or imputed.

## Columns

### Keys and target

| Column | Type | Unit | Source | Description |
|---|---|---|---|---|
| `date` | datetime | day | train | Sales date *t* |
| `store_nbr` | int16 | — | train | Store id (1–54) |
| `family` | category | — | train | Product family (33 levels) |
| `sales` | float64 | units (fractional for weight-based items) | train | **Target.** Observed sales — a proxy for demand, censored by stockouts |
| `onpromotion` | int32 | number of items | train | Items of the family on promotion that day (not a discount %) |

### Store attributes

| Column | Type | Source | Description |
|---|---|---|---|
| `city`, `state` | category | stores | Location; `city` is the weather join key |
| `store_type` | category | stores | Store type A–E (renamed from `type`) |
| `cluster` | int16 | stores | Favorita grouping of similar stores |

### Data-quality flags (0/1)

| Column | Rule | QA ref | How to use |
|---|---|---|---|
| `store_active` | 0 before the store's first day with non-zero total sales | QA-02 | **Exclude rows with 0** from models and metrics: not zero demand |
| `store_zero_day` | 1 if the whole store sold nothing that day (incl. 1 January, temporary closures, inactive period) | QA-05 | Closure, not demand; report baseline errors with and without |
| `wind_qa_flag` | 1 if `wind_speed_10m_max` > 100 km/h | QA-09 | Wind is excluded from M0–M4; any M5 run must handle these rows |

### Calendar and holidays (0/1; known in advance for date *t*)

| Column | Rule |
|---|---|
| `is_day_off` | Any applicable Holiday (not transferred), Transfer, Additional or Bridge day for the store's country/state/city |
| `is_national_holiday`, `is_regional_holiday`, `is_local_holiday` | Day off by scope: national → all stores; regional → stores in that `state`; local → stores in that `city` |
| `is_event` | National event (e.g. Mother's Day, World Cup, earthquake aftermath days) |
| `is_work_day` | Official make-up working day (usually a Saturday) |
| `is_earthquake_period` | Manabí earthquake relief period, 16/04/2016–16/05/2016 (from holiday "Terremoto" events) |
| `is_payday` | 15th or last day of the month (public-sector paydays in Ecuador) |
| `year`, `month`, `day_of_week` | Calendar parts; `day_of_week` Monday = 0 |

### Transactions

| Column | Type | Description |
|---|---|---|
| `transactions` | float64 (NaN when absent) | Store transactions that day. **Forbidden as a day-*t* forecasting feature** (only lagged use). NaN on 7,448 store-days: 6,729 before opening and 719 active days without a record |

### Weather (city-day, ERA5 reanalysis via Open-Meteo)

| Column | Unit | Ablation role | QA status |
|---|---|---|---|
| `precipitation_sum` | mm/day | Core: M1 Rain, M3 | Passed (QA-10) |
| `temperature_2m_max` | °C | Core: M2 Temp, M3 | Passed |
| `temperature_2m_mean` | °C | Robustness | Passed |
| `relative_humidity_2m_mean` | % | Extended: M4 | Passed (range check) |
| `sunshine_hours` | hours/day | Extended: M4 | Passed; `sunshine_duration` (seconds) dropped as duplicate |
| `wind_speed_10m_max` | km/h | Candidate M5 | **Not approved** (QA-09) |
| `cloud_cover_mean` | % | Candidate M5 | Range passed; M5 still conditional |
| `source_model` | — | — | `ERA5` |
| `grid_latitude`, `grid_longitude` | degrees | — | ERA5 cell actually returned by the API |
| `weather_grid_id` | — | — | `"lat_lon"` of the grid cell; Salinas and Libertad share one id (QA-07). **Use this, not `city`, when clustering standard errors by weather** |

## Rules for downstream use

1. **Same-day weather (`*_t`) is allowed for RQ1 association only.** For RQ2 forecasting, build lagged weather in `src/features/` and label results *retrospective weather-aware* (ERA5 publication lag).
2. Lags must be computed by **calendar date** (e.g. `date − 7 days`), not by row offset, because 25 December is missing.
3. Filter `store_active == 1` before training or scoring any model.
4. Do not fill missing `transactions` with 0.
