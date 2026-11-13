# Sổ Nhật Ký Xử Lý Sự Cố & Bài Học Kỹ Thuật (Troubleshooting Log)

Nhật ký này ghi lại tất cả các vấn đề kỹ thuật, lỗi phát sinh, nguyên nhân gốc rễ (Root Cause), giải pháp xử lý (Resolution) và bài học kinh nghiệm trong suốt quá trình xây dựng dự án Data Science Financial Accounts Payable.

---

## Cấu Trúc Bản Ghi Sự Cố Chuẩn
Mỗi sự cố được ghi lại theo định dạng:
- **Mã sự cố (ID)**: `ERR-YYYYMMDD-XX`
- **Thời gian & Giai đoạn**: Ngày giờ và Task/Module gặp sự cố
- **Triệu chứng (Symptom)**: Mô tả hiện tượng hoặc thông báo lỗi (Error message / Stack trace)
- **Nguyên nhân cốt lõi (Root Cause)**: Lý do kỹ thuật sâu xa dẫn đến lỗi
- **Giải pháp xử lý (Resolution)**: Cách thức giải quyết cụ thể (kèm commit / diff nếu có)
- **Bài học kinh nghiệm (Lesson Learned)**: Nguyên tắc cần nhớ để ngăn ngừa tái diễn

---

## Danh Mục Sự Cố & Ghi Nhận Thực Chiến

### ERR-20261113-01: Bảo Tồn Di Sản v1 Và Cách Ly Môi Trường Phát Triển
- **Thời gian & Giai đoạn**: 2026-11-13 | Task 1: Setup & Scaffolding
- **Triệu chứng**: Kho mã nguồn cũ chứa cả OCR (PaddleOCR), RAG LLM (LangChain/ChromaDB) và Streamlit v1 trộn lẫn với các notebook phân tích, gây phình to ngữ cảnh và nhiễu kiến trúc cho dự án Data Science mới.
- **Nguyên nhân cốt lõi**: Quá trình nghiên cứu thử nghiệm (R&D) trước đây phát triển nhiều nhánh công nghệ khác nhau trên cùng một nhánh `main`.
- **Giải pháp xử lý**:
  1. Tạo Git Tag có chú thích `v1-legacy` tại commit `8ea7b7a` để lưu giữ toàn bộ mã nguồn cũ vĩnh viễn.
  2. Tạo Git worktree độc lập `feat/ds-rebuild` tại `.worktrees/feat-ds-rebuild` để thực hiện tái cấu trúc sạch sẽ mà không ảnh hưởng nhánh chính.
  3. Xóa các tệp tin di sản khỏi index của nhánh mới và thiết lập bộ khung thư mục chuẩn Data Science.
- **Bài học kinh nghiệm**: Luôn bảo tồn lịch sử qua Git tag trước khi tái cấu trúc lớn; sử dụng Git Worktree để cách ly không gian làm việc.

### ERR-20261113-02: Phân Trang Dữ Liệu Lớn Trên ArcGIS REST FeatureServer
- **Thời gian & Giai đoạn**: 2026-11-13 | Thiết kế Ingestion (Task 2)
- **Triệu chứng**: Endpoint ArcGIS REST API giới hạn số dòng tối đa trên một request (`maxRecordCount` thường là 1,000 đến 2,000 dòng). Nếu chỉ gửi một request đơn giản, sẽ chỉ nhận được phần nhỏ của 1.3 triệu dòng.
- **Nguyên nhân cốt lõi**: Cơ chế bảo vệ tài nguyên của máy chủ dịch vụ bản đồ / dữ liệu không gian công cộng ArcGIS.
- **Giải pháp xử lý**:
  - Triển khai vòng lặp phân trang tự động sử dụng `resultOffset` tăng dần theo `page_size = 2000` và `resultRecordCount = 2000`.
  - Kiểm tra trường `exceededTransferLimit: true` trong phản hồi JSON từ ArcGIS để xác định còn dữ liệu phía sau hay đã đạt đến trang cuối cùng.
  - Thiết lập timeout và kiểm tra mã trạng thái HTTP (`raise_for_status`) với cơ chế retry an toàn.
- **Bài học kinh nghiệm**: Khi làm việc với API công cộng có giới hạn truyền tải, luôn khai thác triệt để các trường metadata của giao thức (như `exceededTransferLimit`) thay vì dựa vào phỏng đoán số lượng bản ghi.

### ERR-20261113-03: Nguy Cơ Data Leakage Qua Timestamp Thanh Toán
- **Thời gian & Giai đoạn**: 2026-11-13 | Thiết kế Temporal Features (Task 5)
- **Triệu chứng**: Mô hình đạt độ chính xác giả tạo gần 100% nếu trong quá trình tạo đặc trưng vô tình sử dụng thông tin thanh toán xảy ra sau ngày nhận hoá đơn.
- **Nguyên nhân cốt lõi**: Trong cơ sở dữ liệu AP, một dòng thanh toán chứa đồng thời `Invoice_Date`, `Payment_Date`, `Payment_Number`, `Check_Amount`. Tại thời điểm `Invoice_Date = t`, các thông tin `Payment_*` chưa hề tồn tại trong thực tế.
- **Giải pháp xử lý**:
  - Nghiêm cấm đưa bất kỳ trường `Payment_*` nào của hoá đơn hiện tại vào không gian đặc trưng.
  - Khi tổng hợp lịch sử uy tín của nhà cung cấp, chỉ được phép sử dụng các hoá đơn đã hoàn tất thanh toán trước ngày `Invoice_Date` của hoá đơn hiện tại (`prior.Payment_Date < current.Invoice_Date`).
  - Viết unit test riêng biệt để kiểm tra chống rò rỉ (`test_features_ignore_future_payments_after_invoice_date`).
- **Bài học kinh nghiệm**: Point-in-time correctness là yếu tố sống còn trong mọi bài toán Machine Learning trên chuỗi thời gian và tài chính.

### ERR-20261113-04: Xử Lý Biên Timestamp Vượt Ngưỡng Nanosecond (Pandas datetime64[ns] Bounds) & Thực Chiến Ingestion 1.3M Dòng
- **Thời gian & Giai đoạn**: 2026-11-13 | Task 2: Reproducible Raw Ingestion & Pandera Validation
- **Triệu chứng**: Khi chuyển đổi cột `Invoice_Date` và `Payment_Date` từ epoch milliseconds dạng số nguyên sang `datetime64[ns]`, xuất hiện lỗi `OutOfBoundsDatetime` nếu gặp giá trị vượt ngoài khoảng giới hạn nanosecond của Pandas (`1677-09-21` đến `2262-04-11`) hoặc các giá trị dữ liệu bất thường (ví dụ năm 2083 hoặc epoch cực lớn/âm).
- **Nguyên nhân cốt lõi**: Pandas 2.2+ hỗ trợ nhiều độ phân giải thời gian (`s`, `ms`, `us`, `ns`), nhưng khi ép kiểu về `datetime64[ns]` để đồng nhất chuẩn Parquet/Pandera, các mốc thời gian ngoài biên nanosecond sẽ ném ngoại lệ nếu không có cơ chế chặn ngưỡng.
- **Giải pháp xử lý**:
  - Triển khai hàm `_coerce_to_naive_datetime64_ns(series)`:
    1. Chuyển đổi an toàn từ epoch ms sang datetime UTC (`pd.to_datetime(series, unit="ms", utc=True, errors="coerce")`).
    2. Gỡ bỏ timezone (`.dt.tz_localize(None)`).
    3. So sánh với `pd.Timestamp.min` và `pd.Timestamp.max` để mask các giá trị vượt biên thành `NaT` trước khi `.astype("datetime64[ns]")`.
  - Thiết lập schema Pandera `RAW_AP_SCHEMA` và `ARROW_RAW_SCHEMA` để kiểm định kiểu dữ liệu và đảm bảo trường `FID` không bao giờ rỗng (`nullable=False`).
- **Kết quả nghiệm thu thực tế (Live Ingestion Evidence)**:
  - **Tệp đầu ra**: `data/raw/nashville_raw.parquet`
  - **Dung lượng**: 32.90 MB (Snappy compressed, 654 row groups)
  - **Tổng số bản ghi**: Chính xác 1,307,617 dòng (khớp 100% với metadata ArcGIS FeatureServer `count=1307617`).
  - **Thời gian thực thi**: ~14 phút 53 giây với cơ chế stream chunk Parquet, ghi nguyên tử (atomic write) và tự động dọn sạch thư mục tạm `.tmp_chunks_nashville_raw`.
- **Bài học kinh nghiệm**: Luôn xử lý biên nanosecond cho timestamp trước khi ép kiểu trong Pandas, và kết hợp Parquet chunking để vừa tiết kiệm RAM vừa hỗ trợ resume an toàn khi download dữ liệu lớn.

### ERR-20261113-05: Sai Số Dấu Phẩy Động Nhị Phân & Tham Số Hóa Script Đa Câu Lệnh Trong DuckDB
- **Thời gian & Giai đoạn**: 2026-11-13 | Task 3: DuckDB Grain Resolution & Data Quality Audit
- **Triệu chứng**:
  1. Khi cộng dồn các dòng chi phí phân bổ âm và dương bằng `SUM(Distribution_Amount)` trong DuckDB (kiểu DOUBLE), xuất hiện các sai số nhị phân siêu nhỏ quanh số 0 (ví dụ `1e-14`), khiến một số bản ghi bù trừ công nợ lẽ ra bằng đúng 0 đồng lại bị nhận diện nhầm thành số dương và lọt vào tập hóa đơn thanh toán sạch (`invoices_clean.parquet`).
  2. Khi chạy kịch bản SQL chứa nhiều câu lệnh ngăn cách bởi dấu chấm phẩy qua `con.execute(sql_script, params)`, DuckDB ném lỗi: `Not implemented Error: Prepared parameters are only supported for the last statement`. Nếu tách chuỗi thủ công mà truyền toàn bộ dict tham số cho mọi câu lệnh thì DuckDB ném tiếp lỗi: `Invalid Input Error: Parameter argument/count mismatch, identifiers of the excess parameters...` vì các tham số thừa không xuất hiện trong câu lệnh đó.
- **Nguyên nhân cốt lõi**:
  - Chuẩn IEEE 754 của số thực dấu phẩy động không bảo toàn tuyệt đối các phép trừ khử tiền tệ nếu không có hàm làm tròn số học (Rounding).
  - Trình biên dịch DuckDB prepared statements kiểm tra cực kỳ chặt chẽ: chỉ chuẩn bị tham số cho 1 câu lệnh duy nhất và cấm tuyệt đối tham số thừa ngoài cú pháp của câu lệnh đó.
- **Giải pháp xử lý**:
  - Áp dụng `ROUND(SUM(Distribution_Amount), 2)` trong câu lệnh SQL gom nhóm để triệt tiêu hoàn toàn nhiễu `1e-14`, phân tách chính xác tuyệt đối 11,927 bản ghi credit memos (`invoice_amount <= 0`) và 811,914 hóa đơn dương (`invoice_amount > 0`).
  - Trong Python `build_dataset.py`, tách kịch bản SQL theo dấu chấm phẩy và lọc động danh sách tham số tương ứng với từng câu lệnh: `stmt_params = {k: v for k, v in params.items() if f"${k}" in stmt}` trước khi thực thi.
- **Kết quả nghiệm thu thực tế (Live Execution Evidence)**:
  - **Tập thô**: 1,307,617 dòng raw.
  - **Dòng vi phạm thời gian bị loại (`dropped_bad_dates`)**: 57,152 dòng (81 dòng gõ nhầm 1991, 190 dòng gõ nhầm 2083, 56,881 dòng `Invoice_Date > Payment_Date`).
  - **Dòng thiếu khóa (`dropped_missing_keys`)**: 0 dòng.
  - **Tập hợp nhất (`grouped_rows`)**: 823,841 hóa đơn (tỷ lệ nén 1.52 dòng thô/hóa đơn).
    - Hóa đơn dương sạch (`clean_invoice_rows`): 811,914 hóa đơn (23.80 MB Snappy Parquet).
    - Hóa đơn bù trừ/giảm trừ (`credit_memo_rows`): 11,927 hóa đơn (0.37 MB Snappy Parquet).
  - **Thời gian xử lý toàn trình**: ~2.8 giây.
- **Bài học kinh nghiệm**: Luôn dùng hàm `ROUND(..., 2)` đối với các trường tiền tệ tài chính khi tổng hợp bằng DuckDB/SQL, và xây dựng cơ chế bóc tách tham số độc lập cho từng prepared statement.

### ERR-20261113-06: Triệt Tiêu Thiên Lệch Cắt Phải (Right-Censoring Bias) Bằng Vùng Đệm 61 Ngày & Đỉnh Nhọn Xác Suất Net 30
- **Thời gian & Giai đoạn**: 2026-11-13 | Task 4: Statistical EDA & Hypothesis Testing (`notebooks/01_eda_statistical_audit.ipynb`)
- **Triệu chứng**:
  1. Trong những tháng cuối năm 2025 (đặc biệt tháng 11 và tháng 12/2025), tỷ lệ trễ hạn quan sát sụp đổ giả tạo một cách phi tự nhiên: tháng 11 giảm xuống 9.12%, tháng 12 giảm về 0.00%, đồng thời `median_days_to_pay` rơi xuống 16 ngày (thay vì 30 ngày như thông thường). Nếu đưa 2 tháng này vào tập đánh giá Test, mô hình sẽ bị "ảo tưởng" rằng hiệu suất thanh toán của thành phố đạt mức hoàn hảo.
  2. Phân phối `days_to_pay` có trung vị (median = 30) và phân vị 75 (p75 = 30) trùng khít nhau, với IQR chỉ bằng 3.0 ngày, trong khi giá trị lớn nhất đạt tới 3,666 ngày (độ lệch skewness = 13.30, kurtosis = 338.99).
- **Nguyên nhân cốt lõi**:
  - Dữ liệu hóa đơn AP được trích xuất với mốc chốt thanh toán tối đa `max(Payment_Date) = 2025-12-31`. Các hóa đơn phát hành trong tháng 11 và 12/2025 nếu mất $>30$ hoặc $>60$ ngày để thanh toán thì tại thời điểm 31/12/2025 chúng **chưa được thanh toán** (vẫn đang mở), nên không thể xuất hiện trong sổ cái lưu vết thanh toán. Do đó, chỉ những hóa đơn được thanh toán siêu tốc (trong vòng vài ngày đến 16-30 ngày) mới kịp xuất hiện, gây ra hiện tượng Cắt Phải (Right-Censoring Bias / Truncation Distortion).
  - Khái niệm Net 30 là quy định pháp lý thanh toán công (Prompt Payment Act) của chính quyền Nashville, khiến các phòng ban kế toán gom và giải ngân hàng loạt đúng vào ngày thứ 30.
- **Giải pháp xử lý**:
  - Khảo sát thực chứng phân vị: 90% hóa đơn (`p90`) được thanh toán trong vòng 61 ngày (`days_to_pay <= 61`, tỷ lệ hóa đơn mất $>60$ ngày chỉ là 10.08%).
  - Thiết lập mốc kết thúc tập kiểm thử `test_end = 2025-10-31`, tạo ra vùng đệm thời gian đúng bằng 61 ngày (từ 01/11/2025 đến 31/12/2025). Mọi hóa đơn phát hành trước hoặc vào ngày 31/10/2025 đều có đủ ít nhất 61 ngày để được thanh toán và quan sát đầy đủ, triệt tiêu 90% rủi ro cắt phải.
  - Sử dụng các thước đo thống kê phi tham số (Non-parametric: Median, IQR, Percentiles) và kiểm định thứ hạng Mann-Whitney U thay vì Mean, Std hay Student's t-test khi so sánh các nhóm dữ liệu.
- **Bài học kinh nghiệm**: Trong bài toán dự báo tài chính AP hoặc Survival Analysis có mốc chốt dữ liệu (data snapshot cutoff), luôn phân tích phân vị thời gian hoàn tất (`p90 completion time`) để tạo vùng đệm an toàn trước mốc chốt, ngăn ngừa triệt để Right-Censoring Bias.

### ERR-20261113-07: Thiên Lệch Cắt Trái (Left-Truncation Bias) Trước 2022-07-01 & Tối Ưu Point-in-Time Rolling Trên 811k Dòng
- **Thời gian & Giai đoạn**: 2026-11-13 | Task 5: Point-in-Time Feature Engineering & Time Splits (`src/features/build_features.py`)
- **Triệu chứng**:
  1. Khi kiểm tra phân phối `is_late` theo tháng trên `data/interim/invoices_clean.parquet`, 21,329 hóa đơn phát hành trước `2022-07-01` có tỷ lệ trễ hạn lên tới **52.40%**, đặc biệt từ tháng `2022-01` đến `2022-05` tỷ lệ trễ hạn đạt **100.00%** (trung vị `days_to_pay` từ 69 đến 252 ngày), lệch hoàn toàn so với mức nền ~24–28%.
  2. Tính toán các đặc trưng lịch sử nghiêm ngặt theo điều kiện `prior.payment_date < curr.invoice_date` (bao gồm cửa sổ trượt 90d/365d và trung vị mở rộng `prior_vendor_median_amount`) bằng self-join trực tiếp trên 811,914 dòng gặp nút thắt hiệu năng nghiêm trọng do các siêu nhà cung cấp (56,015 hóa đơn) và siêu phòng ban (MNPS 366,343 hóa đơn).
- **Nguyên nhân cốt lõi**:
  - Sổ cái thanh toán của Nashville chỉ bắt đầu ghi nhận các khoản giải ngân từ `min(Payment_Date) = 2022-07-01`. Mọi hóa đơn phát hành đầu năm 2022 nếu trả đúng hạn ($\le 30$ ngày) đã được chi trả trước `2022-07-01` và biến mất khỏi tập dữ liệu — tạo ra **Left-Truncation Bias** (chỉ các hóa đơn trễ kéo dài mới xuất hiện).
  - Self-join $O(N^2)$ sinh ra hàng chục tỷ cặp trung gian khi một nhóm có hàng trăm nghìn bản ghi.
- **Giải pháp xử lý**:
  - Giữ toàn bộ 811,914 dòng khi tính lịch sử thanh toán (vì các khoản trả của hóa đơn cũ đều xảy ra từ `2022-07-01` trở đi), sau đó trong `split_out_of_time` áp dụng chặn biên kép: `train_start = "2022-07-01"` (loại 21,329 dòng cắt trái) và `test_end = "2025-10-31"` (loại 22,308 dòng cắt phải).
  - Triển khai kiến trúc lai $O(N \log N)$: nén bảng tổng tích lũy theo ngày kết hợp **DuckDB `ASOF LEFT JOIN`** cho các cửa sổ đếm/trung bình/độ lệch chuẩn 7d/30d/90d/365d, và **Two-Heap Two-Pointer** (`_compute_vendor_prior_medians`) cho trung vị mở rộng `amount_vs_vendor_median_ratio`.
- **Kết quả nghiệm thu thực tế (Live Feature Execution Evidence)**:
  - **Thời gian tạo 20 đặc trưng cho 811,914 dòng**: **4.04 giây** (toàn trình đọc/ghi Parquet ~5.4 giây).
  - **Tập Train (`2022-07-01`..`2024-06-30`)**: 456,414 dòng (24.55 MB), `late_rate = 27.53%`, `cold_start = 4.47%`.
  - **Tập Val (`2024-07-01`..`2024-12-31`)**: 115,974 dòng (6.40 MB), `late_rate = 19.09%`, `cold_start = 2.17%`.
  - **Tập Test (`2025-01-01`..`2025-10-31`)**: 195,889 dòng (10.73 MB), `late_rate = 20.76%`, `cold_start = 1.97%`.
  - **Kiểm định toàn vẹn**: 0 giá trị `NaN`/`Inf`/null trên toàn bộ 20 cột đặc trưng; đối chiếu 50 dòng ngẫu nhiên từ tập `val` khớp **50/50 (100%)** với phép tính brute-force Pandas.
- **Bài học kinh nghiệm**: Luôn kiểm tra cả hai đầu phân phối thời gian (Left-Truncation tại đầu dữ liệu và Right-Censoring tại cuối dữ liệu) khi biến mốc chia tập (`invoice_date`) khác với biến mốc thu thập sổ cái (`payment_date`).





### ERR-20261113-08: RuntimeWarning Trong Khởi Chạy Runpy Và Thiết Lập Chuẩn Đánh Giá Kép (Dual-Track Evaluation)
- **Thời gian & Giai đoạn**: 2026-11-13 | Task 6: Dual-Track Evaluation Suite & Local Baselines (`src/models/evaluate.py`, `src/models/train_baseline.py`)
- **Triệu chứng**:
  1. Khi chạy module huấn luyện baseline qua `python -m src.models.train_baseline`, Python phát ra cảnh báo: `RuntimeWarning: 'src.models.train_baseline' found in sys.modules after import of package 'src.models', but prior to execution of 'src.models.train_baseline'; this may result in unpredictable behaviour`.
  2. Việc đánh giá hồi quy thời gian thanh toán `days_to_pay` ở cấp độ hóa đơn đơn lẻ không phản ánh trực tiếp bức tranh tổng thể về rủi ro ngân quỹ (Cashflow Risk), trong khi đánh giá phân loại trễ hạn nhị phân `is_late` nếu chỉ dùng một ngưỡng xác suất cố định (như 0.5) hoặc một giả định chi phí đơn lẻ sẽ dẫn đến kết luận thiên lệch.
- **Nguyên nhân cốt lõi**:
  - Tệp `src/models/__init__.py` thực hiện import sớm `train_and_eval_baselines` từ `src.models.train_baseline`. Khi Python chạy cờ `-m src.models.train_baseline`, trình nạp module `runpy` khởi tạo package cha `src.models` trước, khiến `train_baseline` bị nạp vào `sys.modules` trước khi `runpy` thực thi tệp đó dưới tên `__main__`.
  - Nghiệp vụ quản lý công nợ (AP) quan tâm đồng thời đến hai khía cạnh: (1) Khả năng ưu tiên rà soát các hóa đơn rủi ro cao nhất tương ứng với năng lực nhân sự hữu hạn (Lift@K, Precision@K, Value Captured@K, phân tích độ nhạy chi phí), và (2) Khả năng dự báo chính xác tổng lượng tiền mặt chi trả theo các khung kỳ hạn ngân quỹ 7 ngày, 14 ngày và 30 ngày (Cash Outflow Horizon Forecasting).
- **Giải pháp xử lý**:
  - Gỡ bỏ import sớm của `train_baseline` trong `src/models/__init__.py`, chỉ giữ các hàm đánh giá dùng chung (`evaluate_classification`, `evaluate_regression_and_cashflow`, `inverse_transform_days`), triệt tiêu hoàn toàn cảnh báo `RuntimeWarning`.
  - Xây dựng bộ công cụ đánh giá chuẩn `evaluate.py`:
    - Track 1 (Classification): Tính toán `PR-AUC`, `ROC-AUC`, `Brier Score`, `Lift@K`, `Precision@K`, `Late Value Captured@K` trên các lát cắt dung lượng rà soát `5%`, `10%`, `20%`, kèm bảng quét độ nhạy tác động nghiệp vụ (`business_impact_sensitivity`) trên dải tỷ lệ chi phí phạt trễ hạn so với chi phí kiểm tra `(1:1, 5:1, 10:1)`. Xử lý an toàn trường hợp lát cắt đơn lớp (Single-class edge case).
    - Track 2 (Regression & Cashflow): Áp dụng biến đổi ngược `inverse_transform_days` với ngưỡng chặn không âm (`np.clip(np.expm1(preds), 0.0, None)`), đo lường `MAE`, `RMSE`, `MedianAE` và tính toán sai số dòng tiền thực chi vs dự báo chi theo từng kỳ hạn `7d`, `14d`, `30d` (`actual_outflow`, `predicted_outflow`, `abs_error`, `mape_pct`).
  - Huấn luyện mô hình cơ sở tuyến tính (`train_baseline.py`):
    - Đóng gói pipeline scikit-learn kết hợp `ColumnTransformer` (`StandardScaler` cho 18 biến số thực và `OneHotEncoder(handle_unknown="ignore", min_frequency=50)` cho 2 biến phân loại `operating_unit`, `primary_department`).
    - Huấn luyện `LogisticRegression(max_iter=1000)` trên nhãn `is_late` và `Ridge(alpha=10.0)` trên mục tiêu log `log1p(days_to_pay)` trên 456,414 dòng Train, đánh giá độc lập trên Val (115,974 dòng) và Test (195,889 dòng).
- **Kết quả nghiệm thu thực tế (Live Baseline Metrics Evidence)**:
  - **Tập Validation (`val.parquet`)**:
    - Phân loại: PR-AUC = `0.3182`, ROC-AUC = `0.6557`, Brier Score = `0.1501` (Tỷ lệ nền: `19.09%`).
    - Năng lực rà soát top 20%: Precision = `32.73%`, Lift = `1.71x`, bắt giữ được `50.67%` tổng giá trị hóa đơn trễ hạn ($133.9M / $264.3M).
    - Hồi quy: MAE = `12.76 ngày`, RMSE = `29.57 ngày`, MedianAE = `6.14 ngày`.
    - Dự báo dòng tiền kỳ hạn 30 ngày: Thực chi $1.34 tỷ, Dự báo $1.21 tỷ, Sai số MAPE = `9.64%`.
  - **Tập Test Hold-out (`test.parquet`)**:
    - Phân loại: PR-AUC = `0.2984`, ROC-AUC = `0.5853`, Brier Score = `0.1701` (Tỷ lệ nền: `20.76%`).
    - Năng lực rà soát top 20%: Precision = `32.06%`, Lift = `1.54x`, bắt giữ được `40.38%` tổng giá trị hóa đơn trễ hạn ($181.9M / $450.4M).
    - Hồi quy: MAE = `10.97 ngày`, RMSE = `20.14 ngày`, MedianAE = `6.64 ngày`.
    - Dự báo dòng tiền kỳ hạn 30 ngày: Thực chi $3.00 tỷ, Dự báo $2.81 tỷ, Sai số MAPE = `6.65%`.
- **Bài học kinh nghiệm**:
  - Không import tệp chứa `if __name__ == '__main__':` trực tiếp trong `__init__.py` của package để tránh cảnh báo xung đột namespace của `runpy`.
  - Mô hình tuyến tính cơ sở (Logistic & Ridge) đã mang lại giá trị thực tế rõ rệt: rà soát top 20% hóa đơn nguy cơ cao giúp giữ lại hơn 40% đến 50% số tiền chậm trả, và dự báo dòng tiền 30 ngày đạt sai số MAPE dưới 10%, tạo mốc chuẩn (benchmark) vững chắc để so sánh với các mô hình GBDT (LightGBM/XGBoost) ở các task tiếp theo.
