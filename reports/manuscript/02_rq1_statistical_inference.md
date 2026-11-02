# RQ1: Statistical Association & Panel Inference (Bản thảo Báo cáo Nghiên cứu)

**Tác giả:** Member 2 (M2 — Statistical Research & Panel Inference)  
**Môn học:** STAT3013 · **Đề tài:** Weather × Favorita — Retail Demand Forecasting  
**Tập tin số liệu đi kèm:** `results/tables/rq1_coefficients.csv`, `reports/figures/rq1_weather_coefficients_forest.png`  

---

## 1. Đặt vấn đề & Câu hỏi Nghiên cứu 1 (RQ1)

Trong chuỗi cung ứng bán lẻ, câu hỏi liệu thời tiết có tác động thực sự đến hành vi tiêu dùng hàng ngày hay không thường bị ngộ nhận qua các tương quan đơn biến bề nổi. Mục tiêu của **RQ1** là tách biệt mối liên hệ thống kê thực chất giữa các biến số khí tượng (nhiệt độ, lượng mưa, độ ẩm, số giờ nắng) với lượng bán hàng ngày sau khi đã kiểm soát chặt chẽ các đặc tính cố định của từng cửa hàng và các yếu tố lịch kinh doanh (khuyến mãi, ngày nghỉ, chu kỳ nhận lương).

### Hai giả thuyết khoa học:
- **Giả thuyết $H_{1a}$ (Mối liên hệ cấp mạng lưới):** Sau khi kiểm soát các đặc tính cố định cửa hàng - ngành hàng, khuyến mãi và lịch kinh doanh, tồn tại ít nhất một yếu tố thời tiết có mối liên hệ thống kê khác 0 với lượng bán ở mức ý nghĩa $\alpha = 0.05$ dưới sự kiểm soát False Discovery Rate (FDR).
- **Giả thuyết $H_{1b}$ (Dị biệt hóa theo ngành hàng & địa phương):** Mức độ nhạy cảm thời tiết có sự khác biệt rõ rệt giữa các nhóm hàng: các ngành hàng nhạy cảm thời tiết (Đồ uống, Thực phẩm đông lạnh, Sữa tươi) có hệ số co giãn cao, trong khi các ngành hàng thiết yếu không dễ hỏng (Hàng khô, Hóa phẩm tẩy rửa) có hệ số co giãn nhỏ hoặc không có ý nghĩa thống kê.

---

## 2. Phương pháp Kinh tế lượng (Econometric Methodology)

### 2.1. Biến phụ thuộc & Dạng hàm
Do lượng bán lẻ hàng ngày có phân phối lệch phải mạnh và có các ngày bán bằng 0, mô hình sử dụng phép biến đổi logarit chuẩn:
$$y_{s,f,t} = \log(1 + \text{Sales}_{s,f,t})$$
với $s \in \{1, \dots, 54\}$ đại diện cho cửa hàng, $f \in \{1, \dots, 7\}$ là 7 nhóm ngành hàng trọng tâm, và $t$ là ngày quan sát.

### 2.2. Đặc tả Mô hình Fixed Effects (Trọng tâm Nghiên cứu)
Đặc tả chủ đạo là mô hình hồi quy dữ liệu bảng hiệu ứng cố định thực thể (Entity Fixed Effects):
$$\log(1 + \text{Sales}_{s,f,t}) = \alpha_{s,f} + \beta^T W_{\text{city}(s),t} + \theta^T X_{s,f,t} + \varepsilon_{s,f,t}$$

- **$\alpha_{s,f}$ (Entity Intercepts):** Hấp thụ toàn bộ $54 \times 7 = 378$ hiệu ứng cố định không quan sát được của từng cặp `(store_nbr, family)` (diện tích cửa hàng, mô hình siêu thị, mật độ dân cư lân cận, thị hiếu tiêu dùng dài hạn).
- **$W_{\text{city}(s),t}$ (Vector thời tiết):** Lượng mưa ($\text{precipitation\_sum}$), Nhiệt độ tối đa ($\text{temperature\_2m\_max}$), Nhiệt độ trung bình ($\text{temperature\_2m\_mean}$), Độ ẩm trung bình ($\text{relative\_humidity\_2m\_mean}$), Số giờ nắng ($\text{sunshine\_hours}$). Biến tốc độ gió bị loại bỏ theo biên bản QA-09 do lỗi cảm biến.
- **$X_{s,f,t}$ (Vector biến kiểm soát):** Số lượng mặt hàng khuyến mãi ($\text{onpromotion}$), Ngày nghỉ cuối tuần/lễ ($\text{is\_day\_off}$), Ngày nhận lương 15 và cuối tháng ($\text{is\_payday}$), Sự kiện quốc gia ($\text{is\_event}$), Ngày lễ quốc gia ($\text{is\_national\_holiday}$), Giai đoạn khắc phục sau động đất 2016 ($\text{is\_earthquake\_period}$).

### 2.3. Khắc phục vấn đề Cụm lặp đo (The Moulton Clustering Problem)
Do 54 cửa hàng phân bổ trong 22 thành phố (ví dụ Quito có 18 cửa hàng, Guayaquil có 8 cửa hàng) đều nhận chung một giá trị đo thời tiết từ ô lưới ERA5 của thành phố trong ngày $t$, phần dư giữa các cửa hàng trong cùng thành phố có tương quan không gian mạnh ($\text{Cov}(\varepsilon_{s,f,t}, \varepsilon_{s',f',t}) > 0$). Sử dụng sai số chuẩn IID thông thường sẽ phóng đại $t$-statistic lên gấp 3-5 lần.
**Giải pháp:** Toàn bộ sai số chuẩn được ước lượng theo phương pháp **Cluster-Robust Covariance Matrix** phân cụm theo 22 thành phố.

### 2.4. Kiểm soát False Discovery Rate (Benjamini–Hochberg)
Khi ước lượng đồng thời 35 hệ số thời tiết (5 biến thời tiết $\times$ 7 ngành hàng), xác suất xuất hiện ít nhất một kết quả dương tính giả là $1 - (1-0.05)^{35} \approx 83.4\%$. Chúng tôi áp dụng quy trình Benjamini–Hochberg (BH) để kiểm soát tỷ lệ phát hiện sai (FDR) ở ngưỡng $q^* = 0.05$:
$$q_{(i)} = \min_{k \ge i} \left( \frac{m}{k} p_{(k)} \right)$$

---

## 3. Kết quả Thực nghiệm & Phân tích Dị biệt (Empirical Findings)

Dựa trên dữ liệu 585.242 quan sát ngày bán hàng mở cửa của 7 nhóm hàng:

### 3.1. Kết quả Toàn mạng lưới (Pooled 7 Families)
Bảng tổng hợp từ mô hình Fixed Effects gộp:

| Biến khí tượng | Hệ số $\hat{\beta}$ | Sai số chuẩn Cụm (SE) | Khoảng tin cậy 95% (CI) | $p$-value | $q$-value (FDR) | Kết luận thống kê |
|---|---|---|---|---|---|---|
| **Nhiệt độ TB ($\text{temp\_mean}$)** | **+0.0634** | 0.0083 | $[+0.0471, +0.0796]$ | $1.96 \times 10^{-14}$ | $4.89 \times 10^{-14}$ | **Có ý nghĩa rất cao ($***$)** |
| **Số giờ nắng ($\text{sunshine}$)** | **+0.0060** | 0.0006 | $[+0.0047, +0.0073]$ | $1.17 \times 10^{-20}$ | $5.86 \times 10^{-20}$ | **Có ý nghĩa rất cao ($***$)** |
| **Nhiệt độ Max ($\text{temp\_max}$)** | **-0.0101** | 0.0036 | $[-0.0171, -0.0030]$ | 0.0049 | 0.0082 | **Có ý nghĩa âm ($**$)** |
| **Độ ẩm TB ($\text{humidity}$)** | +0.0013 | 0.0010 | $[-0.0007, +0.0032]$ | 0.2011 | 0.2514 | Không có ý nghĩa |
| **Lượng mưa ($\text{precipitation}$)** | +0.0001 | 0.0002 | $[-0.0003, +0.0005]$ | 0.5702 | 0.5702 | **Không có ý nghĩa ($q > 0.05$)** |

**Nhận định cốt lõi:**
1. Nhiệt độ trung bình ngày và số giờ nắng có tương quan dương mạnh nhất với lượng bán lẻ. Khi nhiệt độ trung bình tăng thêm 1°C, lượng bán toàn mạng lưới tăng trung bình **+6.34%** ($p < 10^{-13}$).
2. Lượng mưa ($\text{precipitation\_sum}$) không thể hiện mối liên hệ có ý nghĩa thống kê trên quy mô toàn mạng lưới ($p = 0.57$). Điều này bác bỏ trực giác thông thường cho rằng "cứ mưa là doanh số giảm trên toàn bộ chuỗi".

### 3.2. Dị biệt hóa theo Nhóm ngành hàng (Heterogeneity Analysis)
Ước lượng Fixed Effects độc lập cho từng nhóm ngành hàng làm nổi bật sự phân hóa mạnh mẽ ($H_{1b}$ được chấp nhận):

1. **Nhóm Nhạy cảm Thời tiết Cao:**
   - **`DAIRY` (Sữa/Bơ):** Nhạy cảm cực mạnh với nhiệt độ trung bình ($\hat{\beta} = +0.1033$, tăng +10.3%/°C) và số giờ nắng ($\hat{\beta} = +0.0111$, tăng +1.11%/giờ nắng). Đặc biệt, độ ẩm tăng cũng gắn liền với lượng mua sữa tăng ($\hat{\beta} = +0.0053$, $q = 1.8 \times 10^{-7}$).
   - **`BEVERAGES` (Đồ uống):** Hệ số co giãn theo nhiệt độ trung bình đạt **+9.09%/°C** ($q = 1.2 \times 10^{-8}$) và số giờ nắng **+0.58%/giờ** ($q < 10^{-3}$).
   - **`FROZEN FOODS` (Thực phẩm đông lạnh):** Hệ số co giãn theo số giờ nắng lớn nhất toàn bộ nghiên cứu (**+1.15%/giờ nắng**, $q = 1.06 \times 10^{-9}$) và nhiệt độ trung bình **+9.07%/°C**.
2. **Nhóm Ít nhạy cảm / Hàng thiết yếu:**
   - **`GROCERY I` (Hàng khô thiết yếu):** Hệ số nhiệt độ trung bình chỉ tăng nhẹ +2.45%/°C; lượng mưa không có ý nghĩa ($p = 0.18$).
   - **`BREAD/BAKERY` (Bánh mì tươi):** Không bị ảnh hưởng bởi nhiệt độ trung bình ($p = 0.15$) hay lượng mưa ($p = 0.36$), chủ yếu biến động theo chu kỳ ngày nghỉ và khuyến mãi.

### 3.3. Kiểm tra Độ nhạy với Thiên tai (Earthquake Robustness)
Loại bỏ 1 tháng khủng hoảng sau trận động đất 16/04/2016 tại Pedernales cho thấy:
- Hệ số nhiệt độ trung bình chỉ dịch chuyển **$-1.85\%$** (từ 0.0634 về 0.0622).
- Hệ số số giờ nắng chỉ dịch chuyển **$-5.02\%$** (từ 0.0060 về 0.0057).
- Toàn bộ dấu và mức ý nghĩa thống kê được bảo toàn nguyên vẹn, chứng minh mô hình có độ bền vững cao trước các cú sốc bất thường.

---

## 4. Ý nghĩa Thực tiễn & Khuyến nghị Bàn giao

1. **Đối với M3 (Mô hình Máy học & Dự báo Điểm):**
   - Không nên kỳ vọng cấu hình M1 (chỉ bổ sung lượng mưa) mang lại đột phá lớn vì lượng mưa có tín hiệu thống kê yếu.
   - Cấu hình M3 (bổ sung nhiệt độ) và M4 (bổ sung số giờ nắng) mang tín hiệu giải thích mạnh nhất.
2. **Đối với M4 (Mô hình Quyết định Chi phí Newsvendor):**
   - Ưu tiên gắn cờ `Prioritize pilot` cho 3 ngành hàng: `BEVERAGES`, `DAIRY`, `FROZEN FOODS`.
   - Các ngành hàng như `GROCERY I` và `BREAD/BAKERY` cần gắn cờ `Uncertainty Flag = True` do độ nhạy thời tiết thấp, chi phí tích hợp API thời tiết có thể không bù đắp được giá trị giảm chi phí dự trữ.
