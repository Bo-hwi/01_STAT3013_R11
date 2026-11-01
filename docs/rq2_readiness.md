# Nghiệm thu Nhiệm vụ 1 — Member 3

**Protocol: M3-RQ2-v1.1. Trạng thái tổng: chưa hoàn tất Nhiệm vụ 1 ở cấp nhóm.**
Thiết kế đã được M3 lựa chọn theo yêu cầu người dùng; chưa có bằng chứng nhóm phê duyệt, dữ liệu đạt QA hoặc pipeline chạy được. Không suy ra thành viên chưa làm ở nơi khác chỉ vì repo không có artifact.

## 1. Cách đọc trạng thái

- **DONE:** artifact/kiểm tra được nêu thực sự đã tồn tại; không đồng nghĩa đã chạy thí nghiệm.
- **BLOCKED:** thiếu thông tin/bằng chứng bên ngoài cần để kiểm chứng điều kiện.
- **PENDING APPROVAL:** đã có quyết định M3, chờ xác nhận dùng chung từ chủ sở hữu/nhóm.
- Khi có phản hồi, ghi nguồn/bằng chứng và người xác nhận trước đổi trạng thái. Không tự đổi execution_ready chỉ vì tài liệu đã được lưu.

## 2. Checklist Nhiệm vụ 1

| ID | Điều kiện | Trạng thái | Bằng chứng hiện có / còn thiếu | Phụ trách | Điều kiện để hoàn tất |
|---|---|---|---|---|---|
| T01 | Bài toán target/grain/horizon/origin | DONE | Proposal/Role nhất quán; ghi tại rq2_protocol.md và YAML | M3 | Hoàn thành phần định nghĩa; chưa chứng minh availability thực tế |
| T02 | Protocol kỹ thuật và hợp đồng đánh giá được lưu | DONE | rq2_protocol.md, research_protocol.yaml, ablation_M0_M5.yaml | M3 | Có artifact thiết kế; không coi là phê duyệt nhóm |
| T03 | Kiểm tra cấu hình và tính nhất quán | DONE | Đã parse YAML với kiểm tra duplicate keys; đúng 3 folds × 28 ngày, 84 ngày test; 126 fits lõi + 36 M4; schema/links hợp lệ; hash xác nhận chỉ bốn file cho phép thay đổi; phần ngoài RQ2 giữ nguyên | M3 | Hoàn thành kiểm tra artifact thiết kế, không phải test pipeline hoặc kiểm chứng dữ liệu |
| T04 | Độ phủ đầu vào cho lịch folds | BLOCKED | data/ chỉ có placeholders; training_start null | M1 + M3 | M1 xác nhận schema/độ phủ dự kiến, lịch sử chung; chưa cần clean release đầy đủ |
| T05 | Availability và mapping features | BLOCKED | Dictionary/QA chưa có nội dung; promotion mặc định tắt | M1 + M3 | Có mô tả tên nguồn, đơn vị, timezone, sales/promotion/weather assumptions |
| T06 | Quy tắc shortlist train-only | BLOCKED | deep_study_families rỗng; chưa có xác nhận tiêu chí | M2 | Xác nhận tiêu chí và cutoff trước validation F1; shortlist cuối thuộc Nhiệm vụ 3 |
| T07 | Khai báo pilot/exploratory exposure | BLOCKED | Không có pilot log trong repo | Cả nhóm | Ghi giai đoạn đã xem và việc sử dụng kết quả; không giả định chưa tiếp xúc |
| T08 | Máy chạy và ngân sách tài nguyên | BLOCKED | RAM/CPU chưa xác nhận; runtime/RAM đo được để null | M3 + nhóm | Xác định nơi chạy và giới hạn; benchmark train thực hiện sau trong nhiệm vụ triển khai |
| T09 | Folds/tags và lịch fit dùng chung | PENDING APPROVAL | Ba folds có ngày cụ thể, chưa có phản hồi M1/M4 | M1/M3/M4 | Ghi nhận đồng ý lịch sau xác nhận độ phủ, không cần labels/predictions đã sinh |
| T10 | MAE/WAPE/MASE, eligibility, clipping | PENDING APPROVAL | M3 đã chọn quy tắc, chưa xác nhận baseline/quantile tương thích | M1/M3/M4 | Xác nhận mẫu chung, missing-day policy và chính sách hậu xử lý |
| T11 | Mô hình, ablation, gate và compute protocol | PENDING APPROVAL | Có lựa chọn duy nhất trong YAML, M4 gate chưa được đánh giá | M3 + nhóm | Duyệt thiết kế/gate; kết quả QA/compute thực tế được kiểm tra ở nhiệm vụ phù hợp |
| T12 | Schema/metadata bàn giao | PENDING APPROVAL | Có config alias weather_set và schema chín cột | M1/M3/M4 | Xác nhận tên/mapping/interface và M1 chốt Moving Average trước test |
| T13 | Lịch kế hoạch toàn nhóm | PENDING APPROVAL | README/project ghi 12 tuần; Word v4.1 ghi 4 tuần | Nhóm | Xác nhận kế hoạch hiện hành; chỉ cập nhật file dùng chung khi được thống nhất |
| T14 | Bàn giao và nghiệm thu protocol chung | PENDING APPROVAL | Không có bằng chứng đã bàn giao/duyệt trong repo | Nhóm | Có nguồn ghi nhận quyết định và bên nhận; đóng các thiếu hụt đầu vào thiết kế T04–T08 |

## 3. Thông tin cần xác nhận — không tự gửi thay người dùng

| Người | Câu hỏi ngắn | Đầu ra cần nhận |
|---|---|---|
| M1 | Dữ liệu dự kiến bao phủ 26/04–15/08/2017 và có ít nhất một năm trước đó không? Schema/khóa/city mapping là gì? | Schema và tóm tắt độ phủ/ngày đầu hợp lệ; phân biệt dự kiến với kiểm chứng |
| M1 | Missing day, đóng cửa và sales bằng 0 được phân biệt thế nào? Baseline dùng chung mẫu đủ 28 ngày lịch sử được không? | Activity/missing-day policy và phản hồi eligibility |
| M1 | Sales có sẵn cuối ngày không? Promotion biết trước theo giả định nào? Weather dùng timezone/đơn vị/tên nguồn nào? | Availability contract dự kiến, mapping các cột và metadata |
| M1 | Moving Average chốt cửa sổ nào và xuất baseline config BASELINE được không? | Xác nhận interface baseline; không yêu cầu chạy baseline ở Nhiệm vụ 1 |
| M2 | Tiêu chí shortlist chỉ dùng dữ liệu trước 26/04/2017 được không? Khi nào có shortlist cuối? | Quy tắc train-only và thời điểm dự kiến; không cần kết quả panel |
| M2 | Những ngày nào đã xem trong EDA/pilot và dùng vào quyết định chọn mẫu? | Ghi nhận exploratory exposure, phối hợp cả nhóm |
| M4 | Cùng folds, fit một lần/fold và cập nhật lag hằng ngày có dùng được cho quantile/cost không? | Xác nhận thiết kế thời gian dùng chung |
| M4 | Schema config, predictions đã cắt âm, raw artifact và metadata có đủ không? | Chấp thuận interface hoặc khác biệt cụ thể để giải quyết trước triển khai |
| Nhóm | Đã có quyết định/pilot nào ngoài repo? Máy chạy ở đâu, giới hạn RAM/thời gian gì? Lịch 4 hay 12 tuần hiện hành? | Nguồn ghi nhận, người xác nhận và phạm vi quyết định; không tự coi lời đề xuất là duyệt |

## 4. Điều kiện bắt đầu Nhiệm vụ 2

**Chưa bắt đầu Nhiệm vụ 2 trong đợt bàn giao này.** Người dùng xác nhận riêng bước thực hiện kế tiếp.

| ID | Điều kiện | Trạng thái hiện tại | Điều kiện gỡ vướng |
|---|---|---|---|
| N2-01 | Thiết kế kiểm tra/ranh giới M1–M3 | DONE | Protocol mô tả keys, lag/rolling, preprocessing, missing và availability |
| N2-02 | Phối hợp đầu vào features với M1 | PENDING APPROVAL | M1 phản hồi hợp đồng features; M3 không sửa ETL/release thay M1 |
| N2-03 | Dữ liệu thật để kiểm tra | BLOCKED | M1 cung cấp curated/feature release hoặc mẫu đại diện kèm version/hash |
| N2-04 | Metadata đi kèm | BLOCKED | Dictionary/schema, QA log, activity/missing policy, city mapping, đơn vị/timezone và định nghĩa từng feature |
| N2-05 | Giả định tại origin | BLOCKED | Sales/promotion availability được ghi rõ; weather hồi cứu được gắn nhãn, không đòi chứng minh ERA5 operational để làm hồi cứu |

Có thể chuẩn bị thiết kế kiểm tra sau khi phối hợp M1; kiểm tra thật chỉ bắt đầu khi N2-03–N2-05 có đầu vào. Không buộc chờ OLS/FE/RE, baseline predictions, shortlist cuối hoặc kết quả uncertainty để bắt đầu Nhiệm vụ 2.

## 5. Điều kiện thực nghiệm sau này — chưa thuộc nghiệm thu Nhiệm vụ 1

Coverage/leakage trên dữ liệu thật, gate M4 thực đo, smoke test RAM/runtime, test pipeline, predictions, CI và tái lập từ code cuối **chưa được kiểm chứng**. Không dùng sự thiếu vắng các kết quả này để yêu cầu viết model trước khi xong thiết kế.

Interpreter đã quan sát khi audit không phải môi trường dự án đạt chuẩn: pandas ngoài range requirements, thiếu pyarrow/pytest; chưa có environment lock được kiểm chứng. Việc tạo môi trường và benchmark thuộc triển khai sau, không tự cài gói trong Nhiệm vụ 1.

## 6. Quy tắc đóng Nhiệm vụ 1

Kết quả kiểm tra bàn giao: YAML hợp lệ và không trùng keys; khớp protocol version, ma trận ablation, lưới tham số, early stopping, clipping, schema và bootstrap/Holm; Markdown UTF-8, local links hợp lệ. Kiểm tra diff nhận diện CRLF của file Windows bằng `git -c core.whitespace=cr-at-eol diff --check`; không đổi cấu hình Git. So sánh hash toàn bộ file ngoài thư mục loại trừ xác nhận không sửa artifact người dùng hoặc tạo dữ liệu/kết quả giả. Không huấn luyện, không chạy pipeline và không dùng các file tests rỗng làm bằng chứng đã kiểm thử model.

Sửa phép cộng trong kế hoạch: M4 thêm 36 fits (12 Ridge + 15 Extra Trees + 9 HGB), không phải 33. Không đổi ma trận hay số ứng viên để làm khớp con số sai.

Đóng phần deliverables M3 khi T01–T03 DONE. Chỉ tuyên bố hoàn tất Nhiệm vụ 1 cấp nhóm khi các xác nhận đầu vào thiết kế và giao diện ở T04–T14 được ghi nhận; các số liệu thực nghiệm được để lại đúng nhiệm vụ sau. Việc nhận shortlist cuối thuộc Nhiệm vụ 3, không phải điều kiện phải làm trước để đóng thiết kế.

Nếu điều kiện dữ liệu làm lịch/phạm vi không khả thi, version protocol và nêu lý do trước test. Không tự chuyển quyền sở hữu, sửa code thành viên khác, commit/push/merge hoặc chuyển sang nhiệm vụ tiếp theo.
