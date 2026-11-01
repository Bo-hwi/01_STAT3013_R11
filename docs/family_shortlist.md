# Deep-study family shortlist (frozen 01/11/2026)

**Decision owner:** M2 (RQ1) — decided by the project leader (M1) on 01/11/2026 to unblock M3, M4 and M1 (SARIMAX), using the pre-registered rule from Proposal section 2. M2 may raise an objection with evidence before **02/11/2026 12:00**; after that the list is frozen for all experiments.
**Reproduce:** `python scripts/screen_families.py` → `results/tables/family_screening.csv`

## Final list (7 families)

| Family | Share of sales | Zero rate | Sector |
|---|---|---|---|
| GROCERY I | 33.0% | 0.00% | Dry staples |
| BEVERAGES | 19.4% | 0.00% | Drinks |
| CLEANING | 9.6% | 0.00% | Household non-food |
| DAIRY | 6.1% | 0.00% | Fresh perishables |
| BREAD/BAKERY | 4.1% | 0.00% | Fresh perishables |
| PERSONAL CARE | 2.4% | 0.00% | Personal non-food |
| FROZEN FOODS | 1.4% | 0.04% | Frozen |

Together about 76% of total sales; six sectors from staples to non-food, so the study can compare weather-sensitive (drinks, fresh, frozen) with less sensitive (dry staples, household) families.

## Rule (written in the Proposal before any model was run)

- **Candidates:** the 8 families listed in Proposal section 2: BEVERAGES, BREAD/BAKERY, PRODUCE, DAIRY, FROZEN FOODS, CLEANING, PERSONAL CARE, GROCERY I.
- **Screening data:** open store-days only (`store_active == 1`, `store_zero_day == 0`) **before 2016-08-04**, the first validation day of fold F1. No validation or test period is looked at, and no model result is used.
- **Criteria (all must pass):**

  | Criterion | Threshold | Reason |
  |---|---|---|
  | Zero-sales rate | ≤ 5% | Intermittent series distort MAE and the newsvendor cost |
  | First non-zero sale | ≤ 2013-01-31 | Full history for 28-day lags and training |
  | Cities with sales | all 22 | Needed for family × city results and the Decision Matrix |
  | Share of total sales | ≥ 1% | Representative scale |

## Excluded candidate: PRODUCE

PRODUCE fails two criteria: zero rate 28.9% and first non-zero sale on 2013-03-16. The zeros come in whole quarters (share of zero store-days: 92% in 2013Q1, about 53–55% in 2013Q2–Q4, 55% in 2014Q2 and 2015Q1), and the series is clean only from 2015Q3. This pattern looks like a recording gap rather than real zero demand. Including it would need a separate shorter training window, which the 4-fold design does not support.

**Limitation to report:** PRODUCE is a plausibly weather-sensitive family, so excluding it may understate the value of weather. If time allows, a sensitivity run can use PRODUCE with data from 2015-07-01 only, reported separately and not mixed into the main results.

## Not selected for scale reasons

Families outside the candidate list were not considered, as pre-registered. Other large families (POULTRY 3.2%, MEATS 3.1%, DELI 2.4%) would also pass the thresholds; they are not added to keep compute within the 13-day schedule.
