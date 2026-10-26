# Data Quality Report

**Owner:** M1 — Data & Baselines · **Version:** v0.2 (27/10/2026)
**Scripts:** `scripts/build_source_catalog.py`, `scripts/qa_quick_checks.py`, `scripts/qa_weather_checks.py`
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
| QA-06 | Mapping 22 thành phố khớp hoàn toàn giữa stores.csv và tọa độ | Đạt | Cả nhóm |
| QA-07 | Salinas và Libertad dùng chung một ô lưới ERA5 | Trung bình | M2 (suy luận), M3 |
| QA-08 | Một số thành phố ven biển nhận ô lưới không phải ô gần nhất | Thấp | Ghi chú giới hạn |
| QA-09 | `wind_speed_10m_max` có giá trị phi lý (tới 783 km/h) | **Cao** | M3 (M5), M2 |
| QA-10 | Các biến weather còn lại nằm trong khoảng hợp lý | Đạt | M2, M3, M4 |

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

## Phát hiện về dữ liệu thời tiết (27/10)

**Cấu trúc chung:** 22 file có cùng 16 cột: `city, state, date, precipitation_sum, temperature_2m_max, temperature_2m_mean, relative_humidity_2m_mean, sunshine_duration, wind_speed_10m_max, cloud_cover_mean, source_model, requested_latitude, requested_longitude, grid_latitude, grid_longitude, sunshine_hours`. Mỗi file 1.688 ngày, tổng 37.136 dòng = 22 × 1.688. `source_model = ERA5`. Mỗi file đã lưu cả tọa độ yêu cầu lẫn tọa độ ô lưới trả về, nên metadata grid mà Proposal yêu cầu đã có sẵn.

### QA-06. Mapping thành phố khớp hoàn toàn

- **Quan sát:** Cả 22 thành phố trong stores.csv đều có đúng một dòng tọa độ, tên tỉnh khớp nhau.
- **Bằng chứng:** Outer join theo `city`: 22/22 dòng `both`, không có thành phố chỉ nằm ở một file, không có tỉnh lệch. Quito có nhiều cửa hàng nhất (18), tiếp theo Guayaquil (8).
- **Giải thích:** File tọa độ được tạo từ chính danh sách thành phố của stores.csv.
- **Quyết định / Ảnh hưởng:**
  - Join weather theo `(city, date)` là many-to-one, an toàn.
  - stores.csv ghi Libertad thuộc **Guayas**, trong khi tọa độ (-2.233, -80.910) nằm ở La Libertad, nay thuộc tỉnh Santa Elena (tách khỏi Guayas năm 2007). Đây là cách đặt nhãn tỉnh của dữ liệu Favorita, không phải lỗi join. Nếu M2 gom theo vùng/tỉnh, cần ghi chú Libertad về mặt địa lý ở cạnh Salinas.
  - 5 tọa độ `EXTERNAL_COORDINATE` (Ambato, Playas, Puyo, Quevedo, Riobamba) đều trả về ô lưới gần tọa độ yêu cầu và tên tỉnh khớp; vẫn cần ghi nguồn tọa độ ngoài trong `data/README.md`.

### QA-07. Salinas và Libertad dùng chung một ô lưới ERA5

- **Quan sát:** Hai thành phố cách nhau khoảng 0,045° (khoảng 5 km) và nhận cùng ô lưới (-2.25, -80.75).
- **Bằng chứng:** Cặp duy nhất có khoảng cách < 0,25°. Thống kê gió theo thành phố trùng khớp (median 15,2, max 42,0 km/h ở cả hai). Tuy vậy chuỗi `temperature_2m_max` không trùng hoàn toàn.
- **Giải thích:** Hai thành phố nằm cùng một ô ERA5 (~0,25° ≈ 28 km). Nhiệt độ hơi khác nhau có thể do Open-Meteo hiệu chỉnh nhiệt độ theo độ cao của tọa độ yêu cầu; điều này chưa được kiểm chứng.
- **Quyết định / Ảnh hưởng:** Coi thời tiết của Salinas và Libertad là **không độc lập**. M2 không được tính hai thành phố này như hai cụm thời tiết riêng khi tính sai số chuẩn theo cụm; M3 khi báo cáo theo thành phố cần ghi chú. Số "cụm thời tiết" thực tế là 21, không phải 22.

### QA-08. Ô lưới trả về không phải lúc nào cũng là ô gần nhất

- **Quan sát:** Với một số thành phố ven biển, ô lưới trả về lệch xa hơn mức làm tròn đơn giản.
- **Bằng chứng:** Esmeraldas yêu cầu kinh độ -79,654 nhưng nhận -79,50 (ô gần nhất theo làm tròn là -79,75). Libertad yêu cầu -80,910 nhưng nhận -80,75 (gần nhất là -81,00). Playas yêu cầu vĩ độ -2,638 nhưng nhận -2,50 (gần nhất là -2,75).
- **Giải thích:** Có thể Open-Meteo chọn ô lưới trên đất liền thay vì ô trên biển cho tọa độ ven biển; cần đối chiếu tài liệu API để xác nhận.
- **Quyết định / Ảnh hưởng:** Không đổi dữ liệu. Ghi vào phần giới hạn: thời tiết thành phố là giá trị đại diện của ô lưới khoảng 28 km, không phải số đo tại cửa hàng.

### QA-09. `wind_speed_10m_max` có giá trị phi lý

- **Quan sát:** Có các giá trị tốc độ gió vượt xa mức vật lý hợp lý cho khu vực.
- **Bằng chứng:**
  - Phân vị 99% chỉ là 20,9 km/h, nhưng max là **783,1 km/h**. 11 dòng vượt 100 km/h.
  - Ngày **26/04/2014** xuất hiện đồng thời ở 5 thành phố lân cận: Babahoyo 783,1; Daule 460,6; Guaranda 402,5; Quevedo 310,7; Guayaquil 222,4 km/h.
  - Babahoyo và Daule còn có giá trị 110–154 km/h vào 14/01/2013 và 25/02/2014. Ambato (104,4) và Riobamba (101,2) cũng vượt ngưỡng.
- **Giải thích:** 783 km/h vượt xa cả bão cấp mạnh nhất. Việc nhiều thành phố lân cận cùng có cực trị vào cùng một ngày cho thấy đây là lỗi trong dữ liệu nguồn/API theo vùng không gian, không phải hiện tượng thời tiết thật.
- **Quyết định / Ảnh hưởng:**
  - Đúng như Proposal đã dự kiến: biến gió **không dùng** trong M0–M4. Cấu hình M5 (gió/mây) giữ ở trạng thái "chưa đạt QA", mặc định ngoài phạm vi chính.
  - Không âm thầm xóa hay winsorize. Nếu sau này M5 được chạy, phải đánh cờ các dòng > 100 km/h, đối chiếu lại với nguồn khác và chạy phân tích độ nhạy có/không có các dòng này.
  - M2 không đưa gió vào hồi quy chính.

### QA-10. Các biến weather còn lại nằm trong khoảng hợp lý

- **Quan sát và bằng chứng:**

  | Biến | Min | Median | Max | Nhận xét |
  |---|---|---|---|---|
  | precipitation_sum (mm/ngày) | 0,0 | 2,5 | 202,8 | Không âm. Max 202,8 cao nhưng có thể xảy ra ở vùng ven biển; cần xem thành phố và ngày |
  | temperature_2m_max (°C) | 13,0 | 26,4 | 36,8 | Hợp lý cho cả vùng núi lẫn ven biển |
  | temperature_2m_mean (°C) | 10,5 | 23,2 | 29,6 | Hợp lý |
  | relative_humidity_2m_mean (%) | 35 | — | 99 | Trong khoảng 0–100 |
  | sunshine_duration (giây) | 0 | 36.191 | 43.597 | Max ≈ 12,1 giờ, đúng độ dài ngày ở xích đạo |
  | sunshine_hours (giờ) | 0 | 10,05 | 12,11 | Đúng bằng sunshine_duration / 3600 |
  | cloud_cover_mean (%) | 3 | 87 | 100 | Trong khoảng 0–100 |

- **Quyết định / Ảnh hưởng:** Mưa và nhiệt độ (biến lõi của M1–M3 trong ablation) đạt QA. Độ ẩm và nắng (M4) đạt QA ở mức kiểm tra khoảng giá trị. **Chỉ dùng một trong hai** `sunshine_duration` hoặc `sunshine_hours` vì chúng là cùng một biến khác đơn vị; đề xuất dùng `sunshine_hours`.

## Việc còn mở

- [ ] Kiểm tra ngày 01/01 của 2014–2017 có cùng hành vi như 2013 không.
- [ ] Tìm các ngày đóng cửa tạm thời ở giữa chuỗi (sales = 0 toàn cửa hàng trong vài ngày liên tiếp) để giải thích phần còn lại của khoảng thiếu transactions.
- [ ] Xác định thành phố và ngày của lượng mưa 202,8 mm, và xem các ngày mưa cực trị có tập trung trong mùa El Niño 2016–2017 không.
- [ ] Kiểm tra `precipitation_sum` của Salinas và Libertad có trùng nhau hoàn toàn không (xác nhận QA-07).
- [ ] Ghi nguồn tọa độ ngoài cho 5 thành phố `EXTERNAL_COORDINATE` vào `data/README.md`.
- [ ] Đánh dấu giai đoạn động đất 16/04/2016 để M2 làm phân tích độ nhạy.
