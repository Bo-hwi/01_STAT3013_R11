M0: học và kiểm tra theo từng bước
================================

Đã cài requirements trong `.venv` của workspace bằng Python 3.13; chưa kiểm tra Python 3.14.
Các lệnh `python` dưới đây cần chạy với environment đã kích hoạt, hoặc thay bằng `.venv/Scripts/python.exe`.

Tên cột đã được M1 xác nhận và nằm trong `src/forecasting/feature_spec.py`, tại một dict `FEATURES`.
Danh sách family bắt buộc truyền vào; không tự chọn bằng validation hoặc test.
Mã hiện dùng sales lag 1/7/14/28, rolling mean 7/28 ngày, calendar và store/family/city.
`onpromotion` chỉ được bổ sung khi nhóm xác minh lịch promotion có sẵn trước ngày dự báo; M0 hiện chưa dùng cột này.
M0 không dùng weather, transactions, target sales hay các cờ đóng cửa làm biến giải thích.

1. Đọc cấu hình và chia folds (`rolling_validation.py`)
----------------------------------------------------

Chạy từ repository root, trước khi cần parquet:

```powershell
python scripts/check_m3_splits.py --synthetic
```

Sanity check: cuối output phải có `PASS`, mỗi validation/test dài 28 ngày và train đứng trước validation.
Đây là dữ liệu giả của ba store và 33 family giả, không phải toàn bộ dữ liệu thật.
Counts closure/post-reopening phải là:

| Fold | Validation | Test |
|---|---|---|
| F1 | 891 / 0 | 1848 / 0 |
| F2 | 1254 / 627 | 693 / 627 |
| F3 | 0 / 0 | 0 / 0 |
| F4 | 0 / 0 | 0 / 0 |

`iter_folds()` chạy theo thứ tự: kiểm tra khóa dữ liệu → tính mask trên toàn lịch sử → lọc active/family → chia ngày → bỏ closure targets trong train.
Trả về `Fold.train`, `Fold.validation`, `Fold.test` và `Fold.counts`.
Validation/test vẫn giữ các hàng bị loại để kiểm toán; khi scoring dùng `frame.loc[frame.score_mask]`.
Mask được tính trước khi cắt fold nên nhận ra closure bắt đầu ở block trước.
Nếu bảng M1 đã có `score_mask`, mã đối chiếu từng hàng active và dừng khi khác quy tắc chung.

```python
from src.forecasting.rolling_validation import iter_folds, read_yaml

protocol = read_yaml("configs/research_protocol.yaml")
ablation = read_yaml("configs/ablation_M0_M5.yaml")
# families is the shortlist provided by M2; data is the full history table.
for fold in iter_folds(data, protocol, ablation, families):
    print(fold.fold_id, len(fold.train), len(fold.validation), len(fold.test))
    print(fold.counts.to_string(index=False))
    assert fold.train.date.max() < fold.validation.date.min()
    assert fold.validation.date.max() < fold.test.date.min()
    assert fold.train.store_zero_day.eq(0).all()
```

Lý do: bỏ closure khỏi target huấn luyện nhưng giữ hàng nguồn để lag không dịch sai; cooldown 28 ngày chỉ loại scoring.
Khi refit, `fold.refit_rows()` ghép train+validation rồi bỏ closure targets, vẫn giữ cooldown targets.
Moving Average/SARIMAX phải tuân thủ cùng mask và training eligibility; không nén các ngày bị loại thành chuỗi liên tiếp.

2. Feature tạm thời, có cùng tên với bảng M1
-------------------------------------------

`prepare_features(data, ready=False)` gọi builder tạm thời trong `rolling_validation.py`.
Sales lag tra khóa `(date-k, store_nbr, family)`; rolling dùng cửa sổ thời gian `[t-k, t)`, bỏ qua ngày thiếu.
Tính features trước khi bỏ closure rows và inactive rows khỏi các fold.

```python
import pandas as pd
from src.forecasting.point_models import prepare_features

SOURCE = "curated"  # Change only this line to "features" after M1 delivery.
paths = {
    "curated": "data/processed/curated_sales_weather.parquet",
    "features": "data/processed/model_features.parquet",
}
data = prepare_features(
    pd.read_parquet(paths[SOURCE], engine="pyarrow"),
    ready=SOURCE == "features",
)
assert data.loc[data.date.eq("2016-12-26"), "sales_lag1"].isna().all()
```

Sanity check: lag1 ngày 26/12 thiếu vì không có 25/12; không điền target sales bằng 0.
Đường `features` kiểm tra lại bốn lag và hai rolling mean, yêu cầu bảng M1 giữ nguyên lịch sử nguồn.
Các giá trị NaN của feature sẽ được imputer xử lý khi fit; không thay đổi target hay lịch ngày.

3. Tuning bằng validation, refit rồi test (`point_models.py`)
------------------------------------------------------------

`tune()` chỉ nhận train và validation, không nhận test.
Mỗi ứng viên dùng một Pipeline mới: numeric imputer → scaler cho Ridge; categorical dùng one-hot với unseen categories được chấp nhận.
ExtraTrees dùng imputer và one-hot, không cần scaler; `random_state=42`.
Pipeline đảm bảo imputer/scaler/encoder chỉ học train khi tuning.
Tham khảo nguyên tắc Pipeline và leakage trong [tài liệu scikit-learn](https://scikit-learn.org/stable/common_pitfalls.html).

Grid mặc định: Ridge alpha = 0.1/1/10; ExtraTrees n_estimators = 100/200 và min_samples_leaf = 5/20.
Đây là grid khởi đầu công khai, cần thống nhất trước khi xem test; không tuyên bố là tối ưu.
Chọn MAE thấp nhất trên các validation rows có mask True, rồi tạo Pipeline mới fit `fold.refit_rows()`.
Test dùng tham số cố định; các ngày sau được dùng dữ liệu quan sát của ngày trước làm lag cho dự báo một ngày.
Không clipping dự báo trong phiên bản này; mọi ứng viên dùng cùng quy tắc.

```powershell
$families = (Read-Host "Danh sach family tu M2, cach nhau bang dau phay").Split(',') |
    ForEach-Object { $_.Trim() }
python -m src.forecasting.point_models --source curated --families $families --check-only
```

Sanity check: kiểm tra khoảng ngày, số rows train sau loại closure, cột `training_rows_excluded` và counts validation/test.
Counts M1 cung cấp là cho đủ 33 family; chạy shortlist nhỏ sẽ có counts khác và phải kiểm tra đúng phạm vi.
Chỉ khi đã kiểm tra xong mới bỏ `--check-only` để chạy mô hình.

```powershell
python -m src.forecasting.point_models --source curated --families $families
```

Sau khi M1 bàn giao, chỉ đổi `--source curated` thành `--source features`; giữ nguyên families và danh sách feature.

4. Metrics (`evaluation.py`)
---------------------------

MAE lấy trung bình sai số tuyệt đối; WAPE là tổng sai số tuyệt đối chia tổng trị tuyệt đối target, mẫu số 0 trả NaN.
Scale MASE là MAE seasonal naive t−7 riêng từng store×family, chỉ lấy các target đủ điều kiện trong train gốc.
Nguồn lag vẫn giữ closure rows; scale được cố định trước khi refit và không dùng validation để tính lại.
MASE gộp là trung bình sai số đã chia scale riêng của từng hàng hợp lệ; scale thiếu/0 trả NA và báo coverage.
Report có mức fold, family×city và store×family×city; F3 mưa cực lớn nên phải đọc từng fold.

5. Output và checks
-------------------

`results/predictions/point_models.csv` chỉ chứa test rows được scoring, đúng 10 cột yêu cầu.
`results/tables/point_models_metrics.csv` chứa metrics và coverage.
`point_models_tuning.csv`, `point_models_exclusions.csv`, `point_models_splits.csv` trong cùng thư mục giúp kiểm toán.
Chưa có dữ liệu thật thì không tạo prediction CSV giả vào đường dẫn kết quả chính thức.

```powershell
python -m unittest discover -s tests -p test_m3_forecasting.py -v
```

Checks bao gồm thiếu 25/12, biên cooldown ngày 28/29, counts M1, mask sai, rolling không chạm sales_t, scale chỉ học train, Pipeline không học validation và test targets không đổi lựa chọn tham số.

Trạng thái môi trường
--------------------

Đã tải 31 raw files từ Drive vào `data/raw/`; toàn bộ SHA-256 khớp catalog M1.
Pipeline trên `origin/feature/m1-data-catalog` đã chạy trong worktree riêng, không merge PR #1: tạo 3.000.888 rows × 35 columns và integrity tests đạt `12 passed`.
Curated parquet và metadata đã được chép vào `data/processed/` của workspace chính.
`python scripts/check_m3_splits.py` mặc định kiểm tra parquet thật; chỉ dùng `--synthetic` khi cần fixture.
Kiểm tra thật trên đủ 33 family tái tạo đúng counts F1/F2, không loại scoring rows ở F3/F4 và loại 19.833 closure targets khỏi F4 train.
QA đủ 33 family không thay thế việc M2 chốt shortlist để chạy nghiên cứu sâu.
PR code được xếp sau PR cấu hình; pipeline tái tạo curated cần PR M1 được tích hợp trước.
Python hiện có là 3.13, không phải 3.14; cần chạy lại trên Python 3.14 của nhóm trước khi chốt môi trường.
