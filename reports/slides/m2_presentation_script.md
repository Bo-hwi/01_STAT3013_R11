# Kịch bản & Nội dung Thuyết trình Member 2 (M2 — RQ1)

**Thời lượng thuyết trình gợi ý:** 4 – 5 phút  
**Chủ đề:** RQ1 — Nghiên cứu Liên hệ Thống kê & Dị biệt hóa Thời tiết – Doanh số (Panel Econometrics)  

---

### Câu hỏi 1: Bài toán của tôi trong dự án là gì?
> *"Vai trò của tôi là Member 2, chịu trách nhiệm chính về Câu hỏi Nghiên cứu 1 (RQ1). Tôi cần xác định xem trong mạng lưới 54 siêu thị Favorita, các yếu tố thời tiết quan sát (nhiệt độ tối đa, lượng mưa, độ ẩm, số giờ nắng) có thực sự liên hệ thống kê với lượng bán hàng ngày hay không sau khi đã loại trừ các yếu tố gây nhiễu từ cửa hàng, lịch kinh doanh (ngày trong tuần, mùa vụ theo tháng, xu hướng năm) và khuyến mãi; đồng thời làm rõ sự khác biệt giữa các nhóm ngành hàng và khu vực địa lý."*

---

### Câu hỏi 2: Giả định kinh tế lượng và công thức toán học là gì?
> *"Tôi sử dụng mô hình hồi quy dữ liệu bảng **Entity Fixed Effects kết hợp Biến kiểm soát Lịch** trên biến phụ thuộc đã lấy logarit:
> 
> $$\log(1 + \text{Sales}_{s,f,t}) = \alpha_{s,f} + \beta^\top W_{g(s),t} + \theta^\top X_{s,f,t} + \gamma_{\text{calendar}}(t) + \varepsilon_{s,f,t}$$
> 
> **Ba trụ cột phương pháp luận:**
> 1. $\alpha_{s,f}$ là hiệu ứng cố định theo từng cặp `(store_nbr, family)` (378 thực thể), hấp thụ toàn bộ đặc tính cố định của cửa hàng và ngành hàng.
> 2. **Kiểm soát lịch $\gamma_{\text{calendar}}(t)$ chặt chẽ:** Bổ sung các biến giả ngày trong tuần (thứ 2 đến Chủ nhật), tháng trong năm (tháng 1 đến tháng 12) và năm (2013-2017). Điều này ngăn ngừa việc hệ số thời tiết bị gánh sai lệch từ hiệu ứng mùa vụ thời tiết và tăng trưởng doanh số tự nhiên.
> 3. **Gom cụm sai số chuẩn theo ô lưới thời tiết (`weather_grid_id`, 21 cụm):** Hai thành phố duyên hải Salinas và Libertad dùng chung một ô lưới khí tượng ERA5 (QA-07). Do đó, cả Fixed Effects lẫn Random Effects đều bắt buộc phải gom cụm theo 21 ô lưới để khắc phục bẫy Moulton và ngăn ngừa $p$-value nhỏ giả tạo."*

---

### Câu hỏi 3: Dữ liệu đầu vào và xử lý đa cộng tuyến như thế nào?
> *"
> - **Biến mục tiêu:** $\log(1 + \text{Sales})$ đo lường ở cấp cửa hàng $\times$ ngành hàng $\times$ ngày (hơn 585.000 dòng dữ liệu sạch mở cửa).
> - **Loại bỏ đa cộng tuyến nhiệt độ:** Dữ liệu EDA cho thấy nhiệt độ trung bình và nhiệt độ tối đa có tương quan tới $0.975$. Nếu giữ cả hai, mô hình bị đa cộng tuyến nặng (VIF $> 20$), làm đảo dấu hệ số phi lý. Tôi chỉ giữ lại duy nhất **nhiệt độ tối đa (`temperature_2m_max`)**, hoàn toàn đồng nhất với cấu hình ablation của nhóm.
> - **4 biến thời tiết chính:** Lượng mưa, nhiệt độ tối đa, độ ẩm và số giờ nắng (đã loại bỏ tốc độ gió do lỗi cảm biến QA-09).
> - **Biến kiểm soát:** Khuyến mãi (`onpromotion`), ngày nghỉ, chu kỳ nhận lương 15 và cuối tháng, sự kiện quốc gia, giai đoạn động đất, cùng hệ thống biến giả lịch."*

---

### Câu hỏi 4: Cách ước lượng và kiểm định mô hình như thế nào?
> *"
> 1. Ước lượng hồi quy bảng bằng thư viện `linearmodels.panel.PanelOLS` và `RandomEffects` với sai số chuẩn cụm theo 21 ô lưới thời tiết.
> 2. **Hiệu chỉnh đa giả thuyết (Multiple Testing):** Khi ước lượng đồng thời 28 hệ số thời tiết (4 biến $\times$ 7 ngành hàng), tôi áp dụng thuật toán **Benjamini–Hochberg False Discovery Rate (BH-FDR)** ở ngưỡng $q^* = 0.05$ để loại trừ phát hiện dương tính giả.
> 3. **Kiểm tra độ nhạy (Sensitivity):** Chạy lại mô hình khi loại trừ 1 tháng khủng hoảng động đất 2016. Hệ số nhiệt độ tối đa chỉ lệch $+1.66\%$, bảo toàn tuyệt đối dấu và mức ý nghĩa."*

---

### Câu hỏi 5: Kết quả thực nghiệm và Insight cốt lõi là gì?
> *"Kết quả thực nghiệm sau khi kiểm soát lịch mang lại 4 phát hiện mang tính then chốt:
> 1. **Nhiệt độ tối đa có tác động dương thực chất:** Nhiệt độ tối đa tăng 1°C liên hệ với mức tăng doanh số toàn chuỗi **+1.19%** ($p = 0.0032$, $q = 0.0063$).
> 2. **Lượng mưa có liên hệ âm nhẹ:** Lượng mưa tăng 1 mm làm giảm nhẹ **-0.05%** doanh số toàn mạng lưới ($p = 0.0007$).
> 3. **Số giờ nắng hết ý nghĩa giả mạo:** Trước khi có kiểm soát tháng, số giờ nắng có $p < 10^{-19}$ giả tạo do trùng chu kỳ mùa vụ. Khi có biến giả tháng, số giờ nắng có $p = 0.54$ (hoàn toàn không có ý nghĩa độc lập).
> 4. **Tập trung hóa độ nhạy ngành hàng & địa lý:**
>    - `BEVERAGES` (+3.76%/°C, $q = 0.0006$) và `FROZEN FOODS` (+3.69%/°C, $q < 0.0001$) là hai ngành hàng nhạy cảm nhiệt độ rõ rệt nhất.
>    - `BREAD/BAKERY` nhạy cảm âm với mưa (-0.067%/mm, $q = 0.0009$).
>    - Các ngành hàng như `GROCERY I`, `CLEANING`, `DAIRY`, `PERSONAL CARE` hoàn toàn trơ với biến động thời tiết hàng ngày ($q \ge 0.15$).
>    - Khu vực duyên hải Costa có độ nhạy cảm nhiệt độ cao gấp **hơn 3 lần** khu vực miền núi Sierra (+1.77%/°C so với +0.57%/°C)."*

---

### Câu hỏi 6: Hạn chế, Ranh giới học thuật và Bàn giao cho M3/M4 là gì?
> *"Có 2 nguyên tắc ranh giới học thuật nghiêm ngặt:
> 1. **Không ngộ nhận quan hệ nhân quả (No Causal Claims):** Đây là các mối liên hệ thống kê có kiểm soát lịch (controlled conditional associations), không phải tác động can thiệp nhân quả.
> 2. **Ranh giới bàn giao mô hình quyết định cho M4:** Tệp `weather_signatures_for_m4.csv` của M2 cung cấp chữ ký thống kê và hệ số co giãn, **tuyệt đối không tự ý gán nhãn khuyến nghị triển khai (như 'Prioritize pilot' hay 'Do not deploy') chỉ dựa vào $p$-value**. Quyết định triển khai thuộc về Member 4 khi tích hợp độ nhạy từ M2 với độ nâng dự báo ngoài mẫu từ M3 (RQ2) và ma trận tổn thất chi phí Newsvendor."*
