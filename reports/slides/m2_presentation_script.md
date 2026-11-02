# Kịch bản & Nội dung Thuyết trình Member 2 (M2 — RQ1)

**Thời lượng thuyết trình gợi ý:** 4 – 5 phút  
**Chủ đề:** RQ1 — Nghiên cứu Liên hệ Thống kê & Dị biệt hóa Thời tiết – Doanh số (Panel Econometrics)  

---

### Câu hỏi 1: Bài toán của tôi trong dự án là gì?
> *"Vai trò của tôi là Member 2, chịu trách nhiệm chính về Câu hỏi Nghiên cứu 1 (RQ1). Tôi cần xác định xem trong mạng lưới 54 siêu thị Favorita, các yếu tố thời tiết quan sát (nhiệt độ, lượng mưa, độ ẩm, số giờ nắng) có thực sự liên hệ thống kê với lượng bán hàng ngày hay không sau khi đã loại trừ các yếu tố gây nhiễu từ cửa hàng, lịch kinh doanh và khuyến mãi; đồng thời làm rõ sự khác biệt giữa các nhóm ngành hàng."*

---

### Câu hỏi 2: Giả định kinh tế lượng và công thức toán học là gì?
> *"Tôi sử dụng mô hình hồi quy dữ liệu bảng **Entity Fixed Effects (Hiệu ứng Cố định)** trên biến phụ thuộc đã lấy logarit:
> 
> $$\log(1 + \text{Sales}_{s,f,t}) = \alpha_{s,f} + \beta^T W_{\text{city}(s),t} + \theta^T X_{s,f,t} + \varepsilon_{s,f,t}$$
> 
> **Hai giả định cốt lõi:**
> 1. $\alpha_{s,f}$ là hiệu ứng cố định theo từng cặp `(store_nbr, family)` (378 thực thể). Nó hấp thụ toàn bộ các đặc tính bất biến theo thời gian của từng siêu thị và ngành hàng. Kiểm định Hausman cũng chứng minh mô hình Fixed Effects vượt trội hơn Random Effects.
> 2. **Cụm sai số chuẩn (Clustered Standard Errors):** Nhiều cửa hàng trong cùng một thành phố nhận chung một ô lưới thời tiết ERA5. Nếu dùng sai số chuẩn IID thông thường sẽ mắc bẫy Moulton (phóng đại độ tin cậy giả mạo). Do đó tôi bắt buộc phải gom cụm sai số chuẩn theo 22 thành phố."*

---

### Câu hỏi 3: Dữ liệu đầu vào và biến mục tiêu là gì?
> *"
> - **Biến mục tiêu:** $\log(1 + \text{Sales})$ đo lường ở cấp cửa hàng $\times$ ngành hàng $\times$ ngày (hơn 585.000 dòng dữ liệu sạch mở cửa).
> - **Biến giải thích thời tiết:** Lượng mưa, nhiệt độ tối đa, nhiệt độ trung bình, độ ẩm và số giờ nắng (đã loại bỏ tốc độ gió do lỗi cảm biến QA-09).
> - **Biến kiểm soát:** Khuyến mãi (`onpromotion`), ngày nghỉ cuối tuần/lễ, chu kỳ nhận lương ngày 15 và cuối tháng, sự kiện quốc gia và giai đoạn động đất."*

---

### Câu hỏi 4: Cách ước lượng và kiểm định mô hình như thế nào?
> *"
> 1. Ước lượng hồi quy bảng bằng thư viện `linearmodels.panel.PanelOLS` với cụm sai số theo thành phố.
> 2. **Hiệu chỉnh đa giả thuyết (Multiple Testing):** Khi ước lượng 35 hệ số thời tiết (5 biến thời tiết $\times$ 7 ngành hàng), tôi áp dụng thuật toán **Benjamini–Hochberg False Discovery Rate (BH-FDR)** ở ngưỡng $q^* = 0.05$ để loại trừ các phát hiện dương tính giả.
> 3. **Kiểm tra độ nhạy (Sensitivity):** Chạy lại mô hình khi loại trừ 1 tháng khủng hoảng động đất 2016 để đảm bảo kết quả không bị méo mó bởi biến cố thiên tai."*

---

### Câu hỏi 5: Kết quả thực nghiệm và Insight cốt lõi là gì?
> *"Bảng kết quả và biểu đồ Forest Plot cho thấy 3 insight rất bất ngờ:
> 1. **Nhiệt độ TB và Số giờ nắng là động lực chính:** Nhiệt độ trung bình tăng 1°C gắn liền với doanh số toàn chuỗi tăng trung bình **+6.34%** ($p < 10^{-13}$); mỗi giờ nắng tăng thêm làm tăng **+0.6%** doanh số ($p < 10^{-19}$).
> 2. **Lượng mưa không có ý nghĩa toàn chuỗi:** Trái với suy nghĩ trực giác, lượng mưa toàn chuỗi có $p = 0.57$ ($q > 0.05$), tức không có bằng chứng mưa làm giảm doanh số trên quy mô toàn mạng lưới.
> 3. **Dị biệt hóa ngành hàng:** Sự co giãn tập trung rất cao ở 3 ngành hàng: `DAIRY` (+10.3%/°C), `BEVERAGES` (+9.1%/°C) và `FROZEN FOODS` (+1.15%/giờ nắng). Ngược lại, các ngành hàng khô thiết yếu như `GROCERY I` và `BREAD/BAKERY` hầu như trơ với biến động thời tiết."*

---

### Câu hỏi 6: Hạn chế và Ranh giới phương pháp là gì?
> *"Có 2 ranh giới học thuật bắt buộc phải nhấn mạnh:
> 1. **Không phải quan hệ nhân quả (No Causal Claims):** Đây là các mối liên hệ thống kê có kiểm soát (controlled statistical associations). Chúng ta không đo lường được các yếu tố vi mô như điều hòa nhiệt độ hỏng hay tắc đường cục bộ.
> 2. **Tính chất hồi cứu (Retrospective):** Dữ liệu ERA5 có độ trễ công bố, giải thích biến động lịch sử nhưng chưa thể xem là dự báo thời gian thực nếu chưa có nguồn cấp feed thời tiết kịp thời. Kết quả này được chuyển giao cho M3 và M4 để làm căn cứ chọn biến và gán cờ rủi ro."*
