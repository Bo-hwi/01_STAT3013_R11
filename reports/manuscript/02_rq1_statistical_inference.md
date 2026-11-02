# RQ1: Statistical Association & Panel Inference (Bản thảo Báo cáo Nghiên cứu)

**Tác giả:** Member 2 (M2 — Statistical Research & Panel Inference)  
**Môn học:** STAT3013 · **Đề tài:** Weather × Favorita — Retail Demand Forecasting  
**Tập tin số liệu đi kèm:** `results/tables/rq1_coefficients.csv`, `results/tables/rq1_model_comparison_ols_fe_re.csv`, `results/tables/weather_signatures_for_m4.csv`, `reports/figures/rq1_weather_coefficients_forest.png`, `reports/figures/rq1_regional_comparison.png`  

---

## 1. Đặt vấn đề & Câu hỏi Nghiên cứu 1 (RQ1)

Trong chuỗi cung ứng bán lẻ, câu hỏi liệu thời tiết có tác động thực sự đến hành vi tiêu dùng hàng ngày hay không thường bị ngộ nhận qua các tương quan đơn biến bề nổi. Mục tiêu của **RQ1** là tách biệt mối liên hệ thống kê thực chất giữa các biến số khí tượng (nhiệt độ, lượng mưa, độ ẩm, số giờ nắng) với lượng bán hàng ngày sau khi đã kiểm soát chặt chẽ các đặc tính cố định của từng cửa hàng và các yếu tố lịch kinh doanh (khuyến mãi, chu kỳ ngày trong tuần, mùa vụ theo tháng, xu hướng năm, chu kỳ nhận lương).

### Hai giả thuyết khoa học:
- **Giả thuyết $H_{1a}$ (Mối liên hệ cấp mạng lưới):** Sau khi kiểm soát các đặc tính cố định cửa hàng - ngành hàng, biến kiểm soát lịch $\gamma_{\text{calendar}}(t)$, khuyến mãi và lịch kinh doanh, tồn tại ít nhất một yếu tố thời tiết có mối liên hệ thống kê khác 0 với lượng bán ở mức ý nghĩa $\alpha = 0.05$ dưới sự kiểm soát False Discovery Rate (FDR).
- **Giả thuyết $H_{1b}$ (Dị biệt hóa theo ngành hàng & địa phương):** Mức độ nhạy cảm thời tiết có sự khác biệt rõ rệt giữa các nhóm hàng: các ngành hàng nhạy cảm nhiệt độ (`BEVERAGES`, `FROZEN FOODS`) có hệ số co giãn cao, trong khi các ngành hàng thiết yếu không dễ hỏng (`GROCERY I`, `CLEANING`, `PERSONAL CARE`) hầu như không nhạy cảm với thời tiết.

---

## 2. Phương pháp Kinh tế lượng (Econometric Methodology)

### 2.1. Biến phụ thuộc & Dạng hàm
Do lượng bán lẻ hàng ngày có phân phối lệch phải mạnh và có các ngày bán bằng 0, mô hình sử dụng phép biến đổi logarit chuẩn:
$$y_{s,f,t} = \log(1 + \text{Sales}_{s,f,t})$$
với $s \in \{1, \dots, 54\}$ đại diện cho cửa hàng, $f \in \{1, \dots, 7\}$ là 7 nhóm ngành hàng trọng tâm, và $t$ là ngày quan sát.

### 2.2. Đặc tả Mô hình Fixed Effects có Biến Kiểm soát Lịch (Trọng tâm Nghiên cứu)
Đặc tả chủ đạo theo Proposal Mục 7.1 là mô hình hồi quy dữ liệu bảng hiệu ứng cố định thực thể kết hợp hiệu ứng lịch:
$$\log(1 + \text{Sales}_{s,f,t}) = \alpha_{s,f} + \beta^\top W_{g(s),t} + \theta^\top X_{s,f,t} + \gamma_{\text{calendar}}(t) + \varepsilon_{s,f,t}$$

- **$\alpha_{s,f}$ (Entity Intercepts):** Hấp thụ toàn bộ $54 \times 7 = 378$ hiệu ứng cố định không quan sát được của từng cặp `(store_nbr, family)` (diện tích cửa hàng, mô hình siêu thị, mật độ dân cư lân cận, thị hiếu tiêu dùng dài hạn).
- **$W_{g(s),t}$ (Vector thời tiết 4 biến):** 
  - Lượng mưa ($\text{precipitation\_sum}$, mm)
  - Nhiệt độ tối đa ($\text{temperature\_2m\_max}$, °C)
  - Độ ẩm trung bình ($\text{relative\_humidity\_2m\_mean}$, %)
  - Số giờ nắng ($\text{sunshine\_hours}$, giờ)
  - *Xử lý đa cộng tuyến nghiêm ngặt:* Biến nhiệt độ trung bình ($\text{temperature\_2m\_mean}$) bị loại bỏ hoàn toàn do có hệ số tương quan tuyến tính $r = 0.975$ với $\text{temperature\_2m\_max}$. Việc đưa cả hai biến nhiệt độ vào cùng một phương trình dẫn đến đa cộng tuyến nghiêm trọng (VIF $> 20$), làm đảo dấu hệ số một cách phi thực tế và vô hiệu hóa khả năng diễn giải kinh tế lượng. Chỉ giữ lại $\text{temperature\_2m\_max}$ hoàn toàn nhất quán với `configs/ablation_M0_M5.yaml`.
  - *Biến tốc độ gió:* Bị loại bỏ theo biên bản QA-09 do lỗi cảm biến ERA5 đo vượt $780 \text{ km/h}$.
- **$\gamma_{\text{calendar}}(t)$ (Thành phần kiểm soát lịch - Proposal 7.1):**
  - Biến giả ngày trong tuần (`dow_1` đến `dow_6`): hấp thụ hiệu ứng bùng nổ tiêu dùng cuối tuần (+25% Thứ 6, +27% Thứ 7).
  - Biến giả tháng trong năm (`month_2` đến `month_12`): hấp thụ trọn vẹn yếu tố mùa vụ khí hậu (mùa khô, mùa mưa, tháng 12 mua sắm Giáng sinh tăng vọt +33.6%). Nếu không kiểm soát tháng, hệ số thời tiết (đặc biệt là số giờ nắng) sẽ bị gánh sai lệch mùa vụ.
  - Biến giả năm (`year_2014` đến `year_2017`): kiểm soát xu hướng mở rộng quy mô kinh doanh và tăng trưởng doanh số theo năm (+30% đến +53%).
- **$X_{s,f,t}$ (Vector biến kiểm soát kinh doanh):** Số lượng mặt hàng khuyến mãi ($\text{onpromotion}$), Ngày nghỉ cuối tuần/lễ ($\text{is\_day\_off}$), Ngày nhận lương 15 và cuối tháng ($\text{is\_payday}$), Sự kiện quốc gia ($\text{is\_event}$), Ngày lễ quốc gia ($\text{is\_national\_holiday}$), Giai đoạn khắc phục sau động đất 2016 ($\text{is\_earthquake\_period}$).

### 2.3. Khắc phục Vấn đề Cụm lặp đo (Clustering theo `weather_grid_id`)
Mặc dù Favorita có 54 cửa hàng tại 22 thành phố, nhưng trên lưới tọa độ ERA5 $0.25^\circ$, hai thành phố duyên hải Salinas và Libertad dùng chung một ô lưới khí tượng duy nhất (QA-07). Do cú sốc thời tiết được gán ở cấp độ ô lưới chứ không phải cấp thành phố hành chính, việc gom cụm sai số chuẩn (Moulton, 1986) phải được thực hiện theo **`weather_grid_id` (21 cụm)**:
$$V_{\text{cluster}} = (X'X)^{-1} \left( \sum_{g=1}^{21} X_g' e_g e_g' X_g \right) (X'X)^{-1}$$
Đặc biệt, mô hình **Random Effects** cũng được cấu hình sai số chuẩn phân cụm theo `weather_grid_id` (`cov_type="clustered"`), loại bỏ hoàn toàn hiện tượng $p$-value nhỏ giả tạo ($10^{-49}$) do giả định IID trước đây.

### 2.4. Kiểm soát False Discovery Rate (Benjamini–Hochberg)
Ước lượng 4 biến thời tiết cho 7 ngành hàng tạo ra 28 giả thuyết đồng thời. Chúng tôi áp dụng quy trình Benjamini–Hochberg (BH) để kiểm soát FDR ở mức $q^* = 0.05$:
$$q_{(i)} = \min_{k \ge i} \left( \frac{m}{k} p_{(k)} \right)$$

---

## 3. Kết quả Thực nghiệm & Phân tích Dị biệt (Empirical Findings)

Dựa trên dữ liệu 585.242 quan sát ngày bán hàng mở cửa của 7 nhóm hàng:

### 3.1. Kết quả Toàn mạng lưới (Pooled 7 Families)

Bảng tổng hợp từ mô hình Fixed Effects toàn chuỗi:

| Biến khí tượng | Hệ số $\hat{\beta}$ | Sai số chuẩn Cụm (SE) | Khoảng tin cậy 95% (CI) | $p$-value | $q$-value (FDR) | Kết luận thống kê |
|---|---|---|---|---|---|---|
| **Nhiệt độ Max ($\text{temp\_max}$)** | **+0.0119** | 0.0040 | $[+0.0040, +0.0197]$ | 0.0032 | 0.0063 | **Có ý nghĩa dương ($**$)** |
| **Lượng mưa ($\text{precipitation}$)** | **-0.0005** | 0.0001 | $[-0.0008, -0.0002]$ | 0.0007 | 0.0028 | **Có ý nghĩa âm nhẹ ($***$)** |
| **Độ ẩm TB ($\text{humidity}$)** | +0.0009 | 0.0005 | $[-0.0000, +0.0018]$ | 0.0549 | 0.0732 | Không có ý nghĩa ($q > 0.05$) |
| **Số giờ nắng ($\text{sunshine}$)** | +0.0006 | 0.0010 | $[-0.0013, +0.0025]$ | 0.5412 | 0.5412 | Không có ý nghĩa ($q > 0.05$) |

**Nhận định cốt lõi:**
1. **Nhiệt độ tối đa có hệ số co giãn dương thực chất:** Khi nhiệt độ cao nhất trong ngày tăng thêm 1°C, lượng bán toàn mạng lưới tăng trung bình **+1.19%** ($p = 0.0032$). Đây là con số thực tế, giải thích được, thay thế hoàn toàn cho các con số méo mó trước đó do đa cộng tuyến.
2. **Lượng mưa có mối liên hệ âm nhẹ:** Lượng mưa tăng thêm 1 mm liên hệ với mức giảm doanh số rất nhỏ **-0.05%** ($p = 0.0007$).
3. **Số giờ nắng không còn ý nghĩa giả mạo:** Trước khi có biến kiểm soát tháng, số giờ nắng xuất hiện p-value siêu nhỏ ($10^{-20}$) vì số giờ nắng tháng 9 (9,8 giờ) cao hơn tháng 3 (7,4 giờ), trùng với chu kỳ mùa vụ doanh số. Sau khi kiểm soát $\gamma_{\text{calendar}}(t)$, số giờ nắng có $p = 0.541$ — minh chứng rõ rệt cho sự cần thiết của kiểm soát lịch.

### 3.2. Dị biệt hóa theo Nhóm ngành hàng (Family Heterogeneity)
Ước lượng Fixed Effects độc lập cho từng nhóm ngành hàng làm nổi bật sự tập trung của độ nhạy thời tiết ($H_{1b}$ được xác nhận):

1. **Nhóm Nhạy cảm Nhiệt độ Cao:**
   - **`BEVERAGES` (Đồ uống):** Nhạy cảm mạnh nhất với nhiệt độ tối đa: $\hat{\beta} = +0.0376$ (tăng **+3.76%/°C**, $q = 0.0006$). Lượng mưa, độ ẩm và nắng không có ý nghĩa sau FDR.
   - **`FROZEN FOODS` (Thực phẩm đông lạnh):** Nhạy cảm nhiệt độ tối đa cao thứ hai: $\hat{\beta} = +0.0369$ (tăng **+3.69%/°C**, $q < 0.0001$).
2. **Nhóm Nhạy cảm Lượng mưa:**
   - **`BREAD/BAKERY` (Bánh mì tươi):** Lượng mưa có hệ số âm có ý nghĩa thống kê: $\hat{\beta} = -0.00067$ (giảm **-0.067%/mm mưa**, $q = 0.0009$), phản ánh tâm lý ngại ra đường mua bánh tươi trong ngày mưa bão.
3. **Nhóm Hầu như Trơ với Thời tiết:**
   - **`GROCERY I` (Hàng khô), `CLEANING` (Hóa phẩm), `DAIRY` (Sữa), `PERSONAL CARE` (Chăm sóc cá nhân):** Toàn bộ các hệ số thời tiết đều có $q_{\text{FDR}} \ge 0.15$. Sau khi kiểm soát lịch kinh doanh và khuyến mãi, thời tiết hàng ngày không giải thích thêm biến thiên doanh số của các ngành hàng thiết yếu này.

### 3.3. Dị biệt hóa Địa lý: Vùng núi Sierra vs Duyên hải Costa
- **Sierra (Vùng núi Andes):** Hệ số nhiệt độ tối đa là $+0.57\%$/°C ($q < 10^{-9}$), số giờ nắng $+0.18\%$/giờ ($q < 10^{-5}$), lượng mưa không có ý nghĩa ($p = 0.16$).
- **Costa (Vùng đất thấp duyên hải):** Hệ số nhiệt độ tối đa lên tới **$+1.77\%$/°C** ($q = 0.041$), cao gấp **hơn 3 lần** so với vùng Sierra; lượng mưa có hệ số âm rõ rệt ($-0.046\%$/mm, $q = 0.041$).

### 3.4. Kiểm tra Độ nhạy với Thiên tai (Earthquake Robustness)
Loại bỏ 1 tháng khủng hoảng động đất 2016 (Pedernales) cho thấy:
- Hệ số nhiệt độ tối đa chỉ dịch chuyển **$+1.66\%$** (từ $0.01187$ lên $0.01207$).
- Hệ số lượng mưa chỉ dịch chuyển $+12.6\%$ (từ $-0.00050$ về $-0.00044$).
- Mọi dấu và mức ý nghĩa thống kê được bảo toàn, khẳng định tính vững chắc của kết quả kinh tế lượng.

---

## 4. Bàn giao Số liệu cho M3 & M4

1. **Đối với M3 (Mô hình Dự báo Máy học):**
   - Biến thời tiết có tín hiệu giải thích rõ ràng nhất là `temperature_2m_max`, đặc biệt cho 2 ngành hàng `BEVERAGES` và `FROZEN FOODS`.
   - Các biến độ ẩm và số giờ nắng không có ý nghĩa bổ sung sau khi đã có lịch thời gian; việc thêm chúng vào mô hình chuỗi thời gian có nguy cơ gây quá khớp (overfitting).
2. **Đối với M4 (Decision Matrix & Đánh giá Kinh tế Chuỗi cung ứng):**
   - Tệp bàn giao `results/tables/weather_signatures_for_m4.csv` cung cấp **chữ ký thống kê mô tả thuần túy** (hệ số co giãn, sai số chuẩn, $q$-value FDR).
   - **Ranh giới học thuật:** RQ1 **không tự ý gán nhãn can thiệp vận hành** như "Prioritize pilot" hay "Do not deploy" chỉ dựa vào $p$-value. M4 sẽ kết hợp độ nhạy thống kê từ RQ1 với mức cải thiện sai số dự báo ngoài mẫu từ RQ2 ($\Delta\text{RMSE}$, $\Delta\text{WAPE}$) và ma trận chi phí Newsvendor ($C_u / (C_u + C_o)$) để ra quyết định triển khai chính xác.
