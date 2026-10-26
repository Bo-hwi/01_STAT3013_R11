# Data Quality Report

**Owner:** M1 — Data & Baselines · **Version:** v0.1 (27/10/2026)
**Scripts:** `scripts/build_source_catalog.py`, `scripts/qa_quick_checks.py`
**Inputs:** raw files listed in `data/README.md` (hashes verified)

Mỗi phát hiện được trình bày theo 4 ý: **Quan sát → Bằng chứng → Giải thích → Quyết định / Ảnh hưởng**.

## Tóm tắt

| Mã | Vấn đề | Mức độ | Ảnh hưởng tới |
|---|---|---|---|
| QA-01 | train.csv thiếu 4 ngày 25/12 | Cao | M1 (join, lag, Seasonal Naive), M3 |
| QA-02 | 8 cửa hàng mở muộn; transactions thiếu 7.448 store-ngày | Cao | M2, M3, M4 |
| QA-03 | oil.csv không có cuối tuần, 43 giá trị trống | Thấp | M2 (nếu dùng làm control) |
| QA-04 | test.csv không có nhãn | Thấp | Cả nhóm |
| QA-05 | Ngày 01/01 có dòng nhưng sales gần như bằng 0 | Trung bình | M1 (baseline), M2, M3 |

## Phát hiện ban đầu (27/10)

### QA-01. train.csv thiếu 4 ngày lịch

- **Quan sát:** Khoảng 01/01/2013–15/08/2017 có 1.688 ngày lịch, nhưng train.csv chỉ có 1.684 ngày.
- **Bằng chứng:** Các ngày thiếu là 2013-12-25, 2014-12-25, 2015-12-25 và 2016-12-25. Mỗi ngày có mặt đều đủ 1.782 dòng (54 store × 33 family), không có khóa `(date, store_nbr, family)` trùng. Tổng 1.684 × 1.782 = 3.000.888 dòng, khớp số dòng của file.
- **Giải thích:** Toàn bộ chuỗi đóng cửa vào ngày Giáng sinh nên dữ liệu gốc không ghi dòng nào.
- **Quyết định / Ảnh hưởng:**
  - Không tự thêm dòng sales = 0 cho các ngày này. Ngày đóng cửa không phải nhu cầu bằng 0.
  - Khi join weather (đủ 1.688 ngày), dùng train làm bảng chính (left join), nên các ngày 25/12 không xuất hiện trong bảng curated.
  - Seasonal Naive (t−7) và lag 7/14/28 phải tính theo **ngày lịch** (ví dụ `date − 7 ngày`), không theo "dòng thứ 7 phía trước". Nếu dịch theo dòng, mọi giá trị sau 25/12 sẽ bị lệch một ngày.
  - Với ngày t mà t−7 rơi vào 25/12: không có giá trị dự báo; quy tắc dự phòng sẽ được chốt cùng M3.

### QA-02. Cửa hàng mở muộn và transactions thiếu

- **Quan sát:** Một số cửa hàng có sales = 0 liên tục ở đầu chuỗi vì chưa hoạt động. transactions.csv có ít dòng hơn lưới store × ngày đầy đủ.
- **Bằng chứng:**
  - Script ban đầu báo 53 cửa hàng có ngày bán đầu tiên sau 01/01/2013, nhưng 45 cửa hàng trong số đó bắt đầu ngày 02/01/2013, tức là chỉ do ngày 01/01 đóng cửa (xem QA-05). Chỉ có **8 cửa hàng thật sự mở muộn**:

    | Store | Ngày bán đầu tiên |
    |---|---|
    | 36 | 2013-05-09 |
    | 53 | 2014-05-29 |
    | 20 | 2015-02-13 |
    | 29 | 2015-03-20 |
    | 21 | 2015-07-24 |
    | 42 | 2015-08-21 |
    | 22 | 2015-10-09 |
    | 52 | 2017-04-20 |

  - transactions.csv có 83.488 dòng, trong khi lưới đầy đủ là 54 store × 1.684 ngày = 90.936, thiếu 7.448 store-ngày.
- **Giải thích:** train.csv tạo đủ mọi tổ hợp store × family × ngày, kể cả trước khi cửa hàng mở, và điền sales = 0. Ngược lại transactions.csv chỉ ghi những ngày có giao dịch. Phần lớn khoảng thiếu 7.448 đến từ giai đoạn trước khi 8 cửa hàng trên mở cửa; phần còn lại có thể là các ngày đóng cửa tạm thời và chưa được kiểm tra.
- **Quyết định / Ảnh hưởng:**
  - Tạo cờ `store_active` = 0 cho mọi ngày trước ngày bán đầu tiên của cửa hàng. Các dòng này không phải nhu cầu bằng 0.
  - M2 loại các dòng `store_active = 0` khỏi hồi quy panel; M3 và M4 loại khỏi train và test để không làm sai MAE.
  - Store 52 chỉ có khoảng 4 tháng dữ liệu, gần như không đủ lịch sử cho lag 28 và rolling. M3 cần cân nhắc khi chia fold.
  - Một bài học quy trình: con số tự động in ra (53) phải được đọc lại trong ngữ cảnh trước khi kết luận.

### QA-03. oil.csv không có cuối tuần và có giá trị trống

- **Quan sát:** Giá dầu chỉ có vào ngày giao dịch của thị trường.
- **Bằng chứng:** 1.218 dòng cho khoảng 01/01/2013–31/08/2017; 0 dòng rơi vào thứ Bảy hoặc Chủ nhật; 43 giá trị `dcoilwtico` trống.
- **Giải thích:** Giá dầu WTI chỉ được niêm yết trong ngày làm việc của thị trường; các ngày lễ thị trường không có giá.
- **Quyết định / Ảnh hưởng:** Đây là biến kiểm soát tùy chọn. Nếu dùng, chỉ lấp bằng giá gần nhất **trước đó** (forward-fill), không dùng giá của ngày sau, để tránh rò rỉ thông tin tương lai. Mặc định chưa đưa vào curated v1.

### QA-04. test.csv không có nhãn

- **Quan sát:** test.csv có 28.512 dòng (1.782 store-family × 16 ngày, 16–31/08/2017) nhưng không có cột `sales`.
- **Bằng chứng:** Cột của test.csv: `id, date, store_nbr, family, onpromotion`; sample_submission.csv cũng 28.512 dòng.
- **Giải thích:** Đây là tập dùng để nộp bài trên Kaggle; nhãn thật bị giữ kín.
- **Quyết định / Ảnh hưởng:** Không dùng làm tập kiểm tra. Mọi đánh giá ngoài mẫu dùng các fold theo thời gian bên trong train.csv (M3 chốt fold trước 04/11).

### QA-05. Ngày 01/01 có dòng nhưng sales gần như bằng 0

- **Quan sát:** Khác với 25/12 (không có dòng), ngày 01/01/2013 vẫn có đủ 1.782 dòng nhưng gần như toàn bộ sales bằng 0.
- **Bằng chứng:** Trên tổng sales theo cửa hàng, chỉ store 25 có sales > 0 vào 01/01/2013; 45 cửa hàng có ngày bán đầu tiên là 02/01/2013.
- **Giải thích:** Tết dương lịch, gần như toàn chuỗi đóng cửa.
- **Quyết định / Ảnh hưởng:**
  - Đánh cờ ngày nghỉ toàn chuỗi; không coi sales = 0 hôm đó là nhu cầu thật.
  - Seasonal Naive cho ngày 08/01 sẽ lấy đúng giá trị 0 của 01/01 và cho sai số lớn. Cần báo cáo riêng hiệu ứng này khi so sánh baseline.
  - Chưa kiểm tra 01/01 của 2014–2017 (xem việc còn mở).

## Việc còn mở

- [ ] Kiểm tra ngày 01/01 của 2014–2017 có cùng hành vi như 2013 không.
- [ ] Tìm các ngày đóng cửa tạm thời ở giữa chuỗi (sales = 0 toàn cửa hàng trong vài ngày liên tiếp) để giải thích phần còn lại của khoảng thiếu transactions.
- [ ] QA tọa độ 22 thành phố: Libertad (tọa độ rơi vào La Libertad, tỉnh Santa Elena, trong khi file ghi Guayas), 5 tọa độ `EXTERNAL_COORDINATE` (Ambato, Playas, Puyo, Quevedo, Riobamba), Salinas và Libertad có thể dùng chung ô lưới ERA5.
- [ ] Kiểm tra 16 cột weather: tên, đơn vị, khoảng giá trị hợp lý, ngoại lệ tốc độ gió.
- [ ] Đánh dấu giai đoạn động đất 16/04/2016 để M2 làm phân tích độ nhạy.
