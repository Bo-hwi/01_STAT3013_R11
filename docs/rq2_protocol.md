# Technical Protocol — Member 3: Forecasting & Machine Learning

**Version: M3-RQ2-v1.1. Trạng thái: thiết kế kỹ thuật M3 đã chọn, chờ nhóm xác nhận dùng chung.**
Không có dữ liệu hoặc thí nghiệm chứng minh các điều kiện đầu vào đã đạt. `execution_ready: false` được giữ cho đến khi các điều kiện được kiểm chứng; không coi việc có YAML là có pipeline chạy được.

## 1. Căn cứ, phạm vi và nguồn cấu hình

- Căn cứ: `Project Proposal.docx` v4.1, `Role_Member3.docx` (Nhiệm vụ 1), `RULES.md` và kết quả rà soát repository.
- M3 phụ trách point forecasting, rolling validation, weather ablation và đánh giá RQ2. M1 giữ ETL/features/baseline; M2 giữ RQ1/shortlist; M4 giữ quantile/cost/Decision Matrix.
- [Research protocol](../configs/research_protocol.yaml), phần `rq2`, là nguồn tham số chạy; [ablation](../configs/ablation_M0_M5.yaml) là nguồn công thức features/ma trận. Tài liệu này giải thích cách dùng, không duy trì bản sao lưới tham số.
- [Readiness](rq2_readiness.md) ghi DONE, BLOCKED, PENDING APPROVAL và thông tin nhóm cần cung cấp. Không tự thay lịch toàn nhóm, suy đoán dữ liệu hoặc chuyển quyền sở hữu.

## 2. Bảng quyết định và lý do

| Hạng mục | Quyết định M3 | Lý do, tính khả thi và giới hạn |
|---|---|---|
| Bài toán | Sales ngày t, khóa date × store_nbr × family; origin cuối t−1; horizon một ngày | Giữ đơn vị Proposal, dễ đối chiếu baseline và bàn giao M4 |
| Kịch bản weather | Retrospective weather-aware | ERA5 có độ trễ; lag không chứng minh dữ liệu đã sẵn tại origin |
| Target | Sales thang gốc, không log-transform | MAE dễ trình bày; sales chỉ là proxy demand, không phải nhu cầu thực |
| Phạm vi | 6–8 family do M2 cung cấp; cả ba thuật toán pooled trên cùng shortlist | Tránh hàng trăm mô hình riêng; không tự chọn family theo test |
| Validation | Ba test folds liên tiếp, mỗi fold 28 ngày; validation 28 ngày, train mở rộng | Bao phủ nhiều chu kỳ tuần với compute vừa sức; chỉ đại diện 84 ngày test |
| Fit lại | Tuning bằng validation, fit mới train + validation, một lần trước mỗi test fold | Tận dụng lịch sử mà không học từ test hiện tại |
| Trong test | Cập nhật features hằng ngày, giữ model/preprocessing cố định | Đây là dự báo một ngày lặp lại, không phải dự báo 28 ngày từ một origin |
| Post-processing | max(0, dự báo raw), không làm tròn/chặn trên | Sales không âm và có thể phân số; áp dụng cả validation/test |
| Model | Ridge, Extra Trees, đúng một global boosted là HistGradientBoosting | Dùng scikit-learn sẵn có; không thêm LightGBM/framework |
| Tuning | Ordered explicit candidates: 3 Ridge, 4 Extra Trees, 2 HGB | Lưới nhỏ; cùng lưới cho ablation của cùng thuật toán; không tìm thêm vì test xấu |
| Early stopping | HGB tắt early stopping và warm start | Tránh validation ngẫu nhiên nội bộ hoặc tái sử dụng estimator không kiểm soát |
| Đánh giá | MAE chính, WAPE/MASE khi hợp lệ; cùng khóa test | Đo đúng RQ2 và tránh chọn mẫu riêng cho model thành công |
| Uncertainty | Paired circular date-block bootstrap; Holm cho ba so sánh chính | Giữ phụ thuộc chung giữa city/store cùng ngày; giới hạn suy luận được nêu rõ |
| Tái lập/compute | Seed 42, chạy thí nghiệm tuần tự, tối đa bốn luồng | Giảm quá tải; chưa có số RAM hoặc runtime được đo |
| Bàn giao | `config` thống nhất cho M0–M5, schema cố định và metadata | Tránh phải suy đoán weather_set/fold khi M4 tích hợp |

Đây là phương án đã chọn có điều kiện kiểm chứng, không phải kết quả thực nghiệm chứng minh là tối ưu. Lịch, shortlist, availability và compute không được tự đánh dấu đã xác nhận.

## 3. Lịch folds và quy trình sử dụng thời gian

Lịch đầy đủ nằm ở `rq2.validation.test_folds`, khoảng ngày bao gồm cả hai đầu. F1 test 24/05–20/06/2017; F2 21/06–18/07/2017; F3 19/07–15/08/2017. Validation là 28 ngày ngay trước mỗi test; train kết thúc trước validation một ngày.

Ngày bắt đầu train để `null` đến khi M1 xác nhận lịch sử hợp lệ. Cần ít nhất một năm lịch sử chung trước validation F1 để thiết kế khả thi; không dùng yêu cầu này để tự loại cửa hàng. Không chứng minh có dữ liệu sạch chỉ từ khoảng nguồn trong Proposal.

1. Chốt protocol, quy tắc shortlist train-only, manifest và lịch sử pilot trước test. M2 chỉ dùng dữ liệu trước validation F1 để chọn family.
2. Trong từng fold, fit imputer/scaler/encoder và từng ứng viên trên train. Dự báo validation tuần tự theo origin; không fit model trên validation trong giai đoạn chọn tham số.
3. Tính MAE validation sau clipping. Chọn ứng viên đầu tiên có score không vượt minimum score quá tolerance trong YAML; không dùng test để phá hòa.
4. Tạo pipeline mới với tham số đã chọn, fit toàn bộ train + validation. Không giữ trạng thái pipeline tuning.
5. Dự báo từng ngày test, dùng dữ liệu lịch sử đã sẵn có. Sales ngày trước chỉ dùng làm lag khi đã quan sát tại origin; ghi rõ giả định sales có sẵn cuối ngày.
6. Không fit lại hoặc điều chỉnh preprocessing trong test fold. Lưu đủ khóa, raw predictions, dự báo đã xử lý và metadata.

Test F1 trở thành validation F2 là dữ liệu quá khứ hợp lệ trong thiết kế rolling đã cố định. Không xem điểm test F1 rồi sửa thủ công thuật toán/features/lưới trước F2. Các folds không độc lập; giữ nội dung này khi trình bày uncertainty. Giai đoạn từng xem trong pilot phải ghi exploratory, không tuyên bố hoàn toàn chưa tiếp xúc.

## 4. Features, eligibility và preprocessing

### Công thức và availability

`ablation_M0_M5.yaml` khóa tên nguồn/đơn vị: mưa `precipitation_sum`, nhiệt độ `temperature_2m_max`, humidity `relative_humidity_2m_mean`, sunshine chuyển giây sang giờ, candidate wind `wind_speed_10m_max`. Không thay bằng biến gần tên mà không ghi thay đổi.

`lag_d(t) = x(t-d)`. Rolling W ngày dùng đúng `[t-W, t-1]`, yêu cầu đủ W quan sát theo ngày lịch; không lấy W dòng gần nhất qua ngày thiếu. Sales theo store–family, weather theo city. Các cột lag/rolling được đặt tên theo template YAML. Dữ liệu weather thiếu cửa sổ thì feature là missing, không xóa riêng dòng test weather.

Calendar lấy ngày địa phương của nguồn Ecuador; M1 phải xác nhận timezone weather và cách chuyển sang ngày city. Không dùng timezone Asia/Saigon của máy người thực hiện. Weekday/month là categorical; holiday là ba flags national/regional/local, áp dụng đúng vị trí cửa hàng và ngày nghỉ có hiệu lực sau transfer. Mapping cột từ pipeline M1 phải được xác nhận; không tự coi holiday quốc gia áp dụng cho mọi loại sự kiện.

M0 gồm business/calendar/sales features. Rain, temperature và extended thêm theo YAML; không dùng same-day actual weather. `onpromotion_t` mặc định tắt; chỉ bật đồng nhất mọi config khi có giả định lịch biết trước được ghi nhận trước test. Không anomaly/oil/transactions/interactions thủ công trong phiên bản này.

Không backfill từ tương lai; không điền missing sales bằng 0. ERA5 là hồi cứu: kết quả không được gọi operational hoặc dự báo thời tiết quá khứ khi chưa có bằng chứng timestamp.

### Mẫu đánh giá chung

- Lập manifest theo fold/store/family/date trước chạy model, độc lập với lỗi dự báo của từng thuật toán. Giữ số dòng dự kiến và số dòng thực có target để báo coverage.
- Mẫu chấm điểm yêu cầu target quan sát hữu hạn không âm, thuộc phạm vi hoạt động do M1 xác nhận và đủ 28 ngày sales quan sát hữu hạn không âm trước origin. Giá trị 0 thực sự quan sát vẫn hợp lệ.
- Quy tắc này xác định estimand là chuỗi có lịch sử đầy đủ. Báo số lượng/tỷ lệ loại theo lý do và family/city/fold; không suy rộng sang cold-start hoặc chuỗi thiếu ngày.
- Thiếu target không được chấm; không lấp target. Không thay manifest vì weather thiếu. Một family hoàn toàn không có mẫu test hợp lệ chặn đánh giá chính thức; không tự loại family đó.
- Thiếu/dư/trùng khóa, y_true lệch giữa models, NaN/infinity của dự báo là lỗi run, không âm thầm lấy giao các dòng thành công hoặc thay bằng 0.

### Pipeline theo thuật toán

Numeric dùng median imputation với missing indicators học trên train; feature order cố định sau fit. Cột hoàn toàn thiếu ở train chặn cấu hình bị ảnh hưởng, không tự fill toàn cột bằng 0. Missing mới ở validation/test dùng imputer đã fit, không học thêm median/indicator.

Ridge chuẩn hóa numeric; one-hot categorical dạng sparse với unknown ignore. Extra Trees cùng imputation và one-hot sparse nhưng không scale. HGB không scale; category được ánh xạ mã không âm train-only và categorical mask tường minh, unknown là missing. Cardinality từng cột không vượt max_bins. Không xem số định danh là đại lượng có thứ tự. Giá trị tham số cụ thể chỉ duy trì ở `rq2.tuning`.

## 5. Ablation, tuning và compute

Ridge/Extra Trees chạy M0–M3; HGB chạy M0/M3. Nếu gate M4 đạt, cả ba bắt buộc chạy M4. M5 được định nghĩa với wind nhưng chưa lên lịch; muốn chạy phải ghi scope change và QA trước test. Quy tắc này phù hợp phần giới hạn compute và gate trong Proposal/Role, không coi M4 là tùy chọn theo điểm số.

Gate M4 xem trước test scores: M1 xác nhận nguồn, đơn vị, join và phạm vi vật lý; coverage từng biến humidity/sunshine trên city–day dự kiến của train + validation F1 đạt ngưỡng trong YAML; có bằng chứng compute đủ. Coverage test được QA đầu vào riêng, không dùng test uplift để bật/tắt gate. Gate chưa được đánh giá thì không tự bật M4. Nếu gate thất bại, ghi lý do và bỏ M4 nhất quán cho cả ba model.

Lõi có 126 fits: Ridge 48, Extra Trees 60, HGB 18 (gồm tuning và refit). M4 thêm 36 fits: Ridge 12, Extra Trees 15, HGB 9. Kế hoạch trước ghi 33 là lỗi cộng, được sửa mà không đổi ma trận hoặc lưới tham số. Chưa tính baseline, bootstrap, smoke test hoặc lỗi chạy lại. Các model dùng training loss theo YAML, không tuyên bố squared-error estimator tối ưu trực tiếp MAE; model selection dùng validation MAE sau clipping.

Không chạy song song nhiều thí nghiệm. Giới hạn cả Extra Trees workers, OpenMP và BLAS bằng `min(4, CPU khả dụng)`, không chỉ đặt n_jobs. Không lưu mọi model ứng viên; giữ tuning scores và metadata. Chưa có hardware/runtime thực tế: để null trong cấu hình. Smoke test và đo compute chỉ trên train trong nhiệm vụ triển khai sau, không phải điều kiện phải huấn luyện ngay để hoàn tất Nhiệm vụ 1. Nếu không khả thi, block run và version protocol trước test; không tự cắt family/folds hoặc lấy mẫu train ngẫu nhiên.

## 6. Metrics và so sánh

- MAE = tổng sai số tuyệt đối / số quan sát hợp lệ. Gộp fold theo từng dòng, không lấy trung bình đều scores fold khi số dòng khác nhau. Báo store–family, family, city, family–city, fold và overall.
- Overall nhiều family là mixed-scale score bổ trợ, không phải cộng thành một lượng hàng đồng nhất; không dùng MAE tuyệt đối để xếp hạng độ dễ dự báo giữa family khác scale.
- WAPE (%) = 100 × tổng sai số tuyệt đối / tổng sales; mẫu số 0 thì NA.
- MASE của một store–family/fold = MAE test / trung bình |sales(d) − sales(d−7)| trên train + validation trước test; chỉ dùng cặp ngày thực sự quan sát cách nhau bảy ngày. Mẫu số 0 hoặc không có cặp thì NA, không epsilon hóa.
- Tổng hợp MASE lấy trung bình đều các store–family/fold hợp lệ trong nhóm báo cáo và kèm valid_count/total_count. Không loại các dòng đó khỏi MAE.
- Delta MAE = MAE_M0 − MAE_weather, uplift (%) = 100 × delta / MAE_M0, cùng model và khóa. Mẫu số M0 bằng 0 thì phần trăm NA. Số dương là cải thiện; báo số folds delta > 0, cả cải thiện bằng 0 hoặc âm.
- Baselines M1 là so sánh thuật toán riêng; không gọi chênh lệch baseline–ML là weather uplift. M1 vẫn chốt cửa sổ Moving Average trước test, không coi danh sách ứng viên scaffold là đã duyệt.

## 7. Uncertainty và nhiều so sánh

Giữ paired error differences trên cùng quan sát. Dùng paired circular block bootstrap theo ngày, thông số tại `rq2.uncertainty`:

1. Với mỗi replicate và mỗi fold, lấy độc lập vị trí bắt đầu block đều trên 28 ngày lịch; đi liên tiếp và vòng về đầu chính fold đó. Ghép rồi cắt đúng 28 vị trí. Không ghép block qua ranh giới fold.
2. Mỗi ngày được lấy k lần thì mọi row/cặp dự báo của ngày đó xuất hiện k lần. Giữ các city/store/family cùng ngày đi chung; dùng cùng lịch resampling cho mọi model/config/phân khúc của replicate.
3. Tính lại MAE bằng tổng sai số / số row được resample, sau đó delta theo cùng chiều M0 − weather. Không lấy trung bình đều ngày nếu số row khác nhau. Replicate không có row trong phân khúc thì CI/p-value phân khúc báo NA có lý do, không âm thầm loại replicate.
4. Dùng percentile CI và seed trong YAML. Chạy cả block chính và sensitivity; nếu chiều bằng chứng thay đổi, ghi uncertainty flag và báo cả hai, không chọn block cho kết quả đẹp.
5. Ba phép thử chính là M3 vs M0 trên test gộp cho từng thuật toán. Với delta quan sát D và bootstrap D_b, đặt C_b = D_b − D; p hai phía = (1 + số |C_b| >= |D|)/(B+1). Holm cho đúng ba p-values: sắp tăng, so lần lượt alpha/3, alpha/2, alpha; dừng bác bỏ ở lần đầu không đạt. Không báo significance nếu một phép thử chính chưa hợp lệ.
6. Family/city và các ablation khác là exploratory, CI pointwise. Không dùng Holm ba phép thử overall để tuyên bố hàng loạt phân khúc có ý nghĩa. Không xem CI 95% là xác suất model đúng.

Bootstrap chỉ mô tả uncertainty có điều kiện trên tập city, kỳ test và models đã fit; không resample lại toàn bộ quá trình tuning/refit và không bảo đảm suy rộng sang city/mùa khác. Chỉ có 84 ngày test, các folds có lịch sử chồng nhau và giả định phụ thuộc trong block là giới hạn phải trình bày. P-value là xấp xỉ bootstrap, không có bằng chứng thực nghiệm calibration tại thời điểm viết.

## 8. Hợp đồng bàn giao và tái lập

`rq2.handoff` khóa schema: `date, store_nbr, family, city, fold, model, config, y_true, y_pred`. Date ISO theo ngày nguồn; store_nbr integer, family/city/model/config/fold là identifiers, y_true/y_pred numeric hữu hạn thang sales gốc. Khóa duy nhất là date/store/family/fold/model/config; city phải khớp mapping M1.

`config` là tên mới thống nhất cho alias `weather_set` trong Proposal; chỉ ánh xạ khi ý nghĩa là feature set M0–M5. Không tự đồng nhất với `split_id`/`feature_set` của bảng features. Baseline model có tên riêng và config BASELINE. Interface này cần M1/M4 xác nhận trước tích hợp.

Xuất point predictions và bảng so sánh theo đường dẫn YAML. `y_pred` đã clipping; raw predictions lưu trong `results/experiments/<experiment_id>/raw_predictions.csv`. Không để CSV chính thức thiếu khóa, nonfinite hoặc sai y_true. Metric không định nghĩa xuất NA kèm lý do trong bảng/metadata.

Metadata gồm experiment ID, protocol version, data hash/version, code revision/dirty state, package versions, seed, features/availability, folds, tham số và validation scores, manifest/coverage, hậu xử lý, lệnh chạy và compute đo được. Models của các lần tuning khác phải có experiment ID riêng để không ghi đè.

Python 3.11 là môi trường mục tiêu theo README; cài theo requirements và chốt versions thực tế sau smoke test. Không bịa lockfile từ môi trường chưa kiểm chứng. Tôn trọng `.gitignore`: không commit datasets, model hoặc predictions bị bỏ qua. M4 chỉ dùng y_true test để đánh giá, không tune point-to-order policy/threshold bằng test.

## 9. Quy trình triển khai sau thiết kế và điều kiện thay đổi

1. Nhận xác nhận đầu vào thiết kế từ M1/M2/M4 và khai báo pilot; ghi trạng thái trong readiness. Không cần baseline đã chạy hoặc shortlist cuối để ghi xong thiết kế Nhiệm vụ 1.
2. Trong Nhiệm vụ 2, phối hợp M1 kiểm tra feature contract/QA/leakage trên dữ liệu thật. Trong Nhiệm vụ 3, tiếp nhận shortlist cuối từ M2; không tự làm thay.
3. Trước chạy thí nghiệm chính thức: freeze shortlist, folds, manifest, gate, môi trường và các lựa chọn phương pháp; kiểm chứng compute trên train ở nhiệm vụ triển khai.
4. Triển khai pipeline/tests, tuning/refit/predict theo lịch khóa; không đổi luật sau test. Lưu metadata, chạy QA outputs, uncertainty rồi bàn giao M4.

Thiếu dữ liệu/QA không tự đổi origin hoặc bỏ cảnh báo retrospective. Sai độ phủ ngày, không còn mẫu của một family hoặc vượt compute phải có protocol version mới, lý do và thống nhất nhóm trước final evaluation. Các mốc lịch trong Proposal là dự kiến, không tự trở thành ràng buộc tuyệt đối.

Thay đổi so với v1.0 trong chat: đặc tả rõ circular resampling/p-value, nguồn/đơn vị/tên features và schema, tách quyết định thiết kế với bằng chứng dữ liệu; giữ họ mô hình và lịch folds, kiểm tra lại ngân sách thành 126 fits lõi và 36 fits M4. Đây là bàn giao thiết kế, không phải báo cáo đã chạy mô hình.

## 10. Nguồn đối chiếu phương pháp

- Tài liệu nội bộ: [Proposal](<Project Proposal.docx>), [Role Member 3](Role_Member3.docx), [RULES](../RULES.md).
- [Forecasting: Principles and Practice — time-series cross-validation và scaled errors](https://otexts.com/fpp3/toolbox.html): nguyên tắc dùng lịch sử và mẫu số MASE.
- [HistGradientBoostingRegressor](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.HistGradientBoostingRegressor.html): early stopping, categorical mask và giới hạn categories.
- [ExtraTreesRegressor](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.ExtraTreesRegressor.html), [OneHotEncoder](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.OneHotEncoder.html): giao diện estimator/encoding; kiểm tra lại với phiên bản môi trường đã khóa khi triển khai.
