# M4 final quantile protocol — proposed, pending team sign-off

Đây là M4-only specification sau Phase 2C; không sửa shared YAML, không chứng
nhận approval thay nhóm và không phải final research results. Runner đề xuất:
`scripts/m4_final_run.py`. Chỉ `--prepare-signoff` được chạy trong lượt chuẩn bị.

## Frozen design đề nghị

- Full shared F1–F4; global M0/M3; tau 0.50, 0.75, **5/6**; seed **42**.
- Frozen families: GROCERY I, BEVERAGES, CLEANING, DAIRY, BREAD/BAKERY,
  PERSONAL CARE, FROZEN FOODS (`configs/research_protocol.yaml`,
  `docs/family_shortlist.md`). Không chọn lại dựa trên pilot test metrics.
- M0: 4 sales lags, 2 rolling means, 10 calendar features và categorical
  store_nbr/family/city. M3 thêm đúng precipitation_sum_lag1 và
  temperature_2m_max_lag1 (`src/forecasting/feature_spec.py`). Không promotion,
  same-day weather, transactions, wind hoặc closure flags làm predictors.
- Estimator: HGB `loss='quantile'`; explicit native-categorical feature flags.
  Numeric median imputation, giữ all-missing columns với zero fallback.
  Ordinal encoding categorical; unknown/missing categories thành missing;
  không OneHot/scaling. Fit transformers trên train khi selection, trên
  eligible train+validation khi fresh refit. `early_stopping=False`.
- Giữ nguyên **original two-candidate grid** trong `quantile_pipeline.DEFAULT_GRID`:

| Parameter | Candidate 0 | Candidate 1 |
|---|---:|---:|
| max_iter | 100 | 100 |
| max_leaf_nodes | 15 | 31 |
| min_samples_leaf | 20 | 20 |
| learning_rate | 0.05 | 0.05 |

- Selection riêng mỗi fold/config/tau bằng mean **raw pinball loss** trên scored
  validation. Equal budget; tie chọn candidate đầu tiên. Không dùng test labels
  để tuning, calibration, feature/grid/variant selection hoặc chọn policy.
- Fresh refit qua shared `Fold.refit_rows()`: train+validation, loại closure
  targets; giữ cooldown targets cho fitting, loại cooldown khỏi scoring.
- Shared `iter_folds()` là splitter duy nhất. Lag/rolling audit trên history
  trước filtering; không sửa lag inputs, không điền ngày thiếu bằng zero.
- One-day-ahead ở end of t−1; observed earlier test sales được dùng cho next-day
  lags. Đây không phải 28-day fixed-origin forecast.
- **Không calibration** trong direct quantile run. Giữ raw predictions;
  corrected = clip zero rồi sort theo canonical tau order. Không reselect bằng
  corrected validation loss; sorting không bảo đảm calibration/loss cải thiện.

## Shared dates và audit counts

Mọi training window bắt đầu 2013-01-01. Counts áp dụng cho cả hai configs.

| Fold | Train end | Validation inclusive | Test inclusive |
|---|---|---|---|
| F1 | 2016-08-03 | 2016-08-04 → 2016-08-31 | 2016-09-01 → 2016-09-28 |
| F2 | 2016-10-17 | 2016-10-18 → 2016-11-14 | 2016-11-15 → 2016-12-12 |
| F3 | 2017-01-31 | 2017-02-01 → 2017-02-28 | 2017-03-01 → 2017-03-28 |
| F4 | 2017-06-20 | 2017-06-21 → 2017-07-18 | 2017-07-19 → 2017-08-15 |

| Fold | Eligible train | Scored validation | Scored test | Eligible refit |
|---|---:|---:|---:|---:|
| F1 | 446544 | 10199 | 9996 | 456743 |
| F2 | 473522 | 9989 | 10108 | 483644 |
| F3 | 511700 | 10388 | 10388 | 522088 |
| F4 | 564074 | 10584 | 10584 | 574658 |

F2 có **10122 eligible validation fit rows**, gồm 133 cooldown rows không được
score. Không tính refit bằng train + scored validation. Budget: **48 selection
fits + 24 fresh refits = 72**. Không thêm runs theo pilot test performance.

## Input, evidence và outputs

Chỉ dùng `data/processed/m4_feature_candidates/candidate-t77rmmvr/model_features.parquet`,
SHA-256 `e4186aeb3a2da9996dcb9deec164cc70ca11c0a869b932720c9786c64a3ab17d`,
cùng metadata và verified staged curated evidence. Không thay canonical M1 files.
Candidate marker/report/hash checks và production preflight phải PASS trước fit.
Warnings/unavailable evidence không được đổi thành verified evidence.

Output: `results/experiments/m4_quantiles_final_f1_f4_m0_m3_<unique-id>/`.
Existing transactional `save_run()` ghi predictions, metrics, tuning, splits,
exclusions, manifest và COMPLETE.json; không export handoff CSV tự động.

- 493416 long prediction rows: validation 246960, test 246456; **hai prediction
  columns raw/corrected trong mỗi row**, không nhân đôi row theo variant.
- Unique key: date/store/family/fold/config/tau_key/split/model/experiment_id.
  Có city, y_true, raw/corrected predictions và crossing flags, fit_stage,
  forecast_origin, horizon_days. Canonical tau keys, không join bằng float equality.
- Metrics tách `variant=raw/corrected`: pinball, coverage, strict-less rate,
  tie rate, coverage gap, crossing rate, negative prediction rate, sample size.
  Expected: 96 fold-level + 14616 family–city rows.
- 48 tuning records/24 selected, 12 split records, 8 exclusion records.
- Matched scored keys/y_true cho M0/M3; đủ 3 taus; nonnegative ordered corrected
  quantities; validate trước metrics/save và reload sau save. Frozen files,
  code/config/input hashes, versions và scored-key digests được kiểm tra lại.
- Raw/corrected per-fold/per-tau results phải cùng được công bố. Nếu aggregate
  ba taus: equal weights, nói rõ không phải economic cost. Per-fold results
  không được thay bằng pooled metric duy nhất.

## Exposure, claims và unresolved decisions

**F1 test metrics đã được xem** trong pilot F1/M0 và paired F1/M0-M3; F1 phải
được mô tả là exploratory/post-pilot, không phải untouched confirmation set.
Đổi ID, rerun hoặc thêm candidate gốc không phục hồi tính untouched của F1.
F1 history vẫn hợp lệ trong chronological expanding training của folds sau.

**F2–F4 confirmatory status: PENDING TEAM VERIFICATION** về prior test exposure
qua tất cả thành viên. Khi pending, chỉ báo cáo diagnostics; không claim
confirmatory. Nhóm có thể ghi exploratory hoặc team-verified-uninspected kèm
evidence; điều này là attestation, không phải bằng chứng hệ thống xác thực danh tính.

Các quyết định cần ký: estimator/preprocessing/grid; no-calibration direct-refit
protocol; P1/P4 residual calibration compatible với point-model refit; paired
date-block bootstrap design; artifact handoff/schema; exposure và claims limits.
P1/P4 và bootstrap có thể được **explicitly deferred out of scope** cho quantile
prediction run, với lý do; không được claim cost comparison/CI khi chưa giải quyết.
Selected validation residuals không phải independent calibration evidence.

Retrospective ERA5 availability, sales-derived closure masks, censored sales proxy,
missing upstream-approved processed digests/coordinate evidence và historical
baseline hash binding phải được disclose. Không operational/causal/ROI claims.
Không dùng pilot để chọn policy rồi gọi evaluation tiếp theo là independent
confirmation. `primary_quantile_model` trong shared YAML còn TODO; approval này
là M4-specific addendum, không âm thầm sửa YAML.

## Review orchestration của Phase 2C

API signature tương thích, nhưng inline proposal chưa có approval receipt,
không pin approved source/config/package snapshot, dùng asserts và chưa kiểm tra
full fold dates/counts hoặc reload outputs. Plain full CLI có `--features` và
preflight nhưng không có candidate completion/expected-digest guard. Pilot CLI
bị khóa F1/M0/one candidate, không phải final CLI. Runner M4 mới giải quyết các
gaps này, không duplicate modelling/preprocessing/metrics/writer logic.

Approval snapshot gắn code/config/docs/input hashes, package versions và design;
mọi thay đổi yêu cầu prepare/sign-off lại. Request không tự hash chính nó.
Execution copy **approved bytes** sang audit receipt và ghi execution start
**trước preflight/fitting**; bind receipt/source approval vào model provenance.
Preflight/counter/integrity failure dừng, giữ FAILED.json nếu ghi được; không
delete/retry/overwrite tự động. Fitting không checkpoint/resume; nếu thất bại
trước save có thể không có model bundle. Peak OS process memory và wall time
ghi trong audit; không dùng pilot peak/runtime làm full-run guarantee.
