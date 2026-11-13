# Thiết kế Dự án: Hệ thống Phân tích Công nợ & Dự đoán Trễ hạn Thanh toán Hóa đơn (End-to-End Data Science Rebuild)

**Ngày lập:** 2026-11-13  
**Dự án:** `D:\Antigravity\projects\project_data_analysis`  
**Trạng thái:** Bản nháp chờ người dùng duyệt (Draft for Approval)

---

## 1. Mục tiêu, Bối cảnh & Nguyên tắc Thiết kế

### Ý định & Lý do (Why)
Người dùng muốn "đập đi xây lại" toàn bộ repository `project_data_analysis` theo hướng **Data Science thuần túy từ gốc** để:
1. Hệ thống hóa và học lại bài bản toàn bộ vòng đời một dự án Data Science thực chiến (từ kiểm định dữ liệu thô, SQL, thống kê, feature engineering chống rò rỉ dữ liệu đến huấn luyện, giải thích mô hình và đánh giá giá trị nghiệp vụ).
2. Sở hữu một dự án chủ lực (flagship portfolio project) mạch lạc, dùng **dữ liệu thật công khai có nguồn gốc uy tín**, phục vụ vòng phỏng vấn Fresher/Junior Data Science sau Tết.

### Mục tiêu (Goals)
- **Dữ liệu thật, thô, nguồn chính thống:** Thay thế hoàn toàn dữ liệu giả lập 50 dòng trong `ap_analysis.py` bằng bộ dữ liệu giao dịch chi trả nhà cung cấp công khai của Chính quyền Metro Nashville (*Metro Vendor Payments*).
- **Chuẩn quy trình Data Science thực tế:**
  1. Thu thập tái lập được (*Reproducible Ingestion*) & Kiểm định dữ liệu thô (*Data Integrity / Pandera Contract*).
  2. Chuẩn hóa cấp độ phân tích (*Grain Resolution*) bằng **SQL (DuckDB)** và khám phá dữ liệu kèm **kiểm định giả thuyết thống kê (Statistical EDA)**.
  3. Thiết kế đặc trưng lịch sử theo dòng thời gian (*Time-aware Feature Engineering*), tuyệt đối tránh *Data Leakage*.
  4. Chia tập theo thời gian (*Out-of-Time Split*), huấn luyện và đối chiếu **Baseline vs Tree-based Models (LightGBM / XGBoost / CatBoost)** trên Kaggle sau khi có tín hiệu từ người dùng.
  5. Phân tích sai số (*Error Analysis*), giải thích mô hình bằng **SHAP**, và lượng hóa tác động dòng tiền/nghiệp vụ.
- **Tài liệu học tập & Nhật ký vướng mắc song song:** Với mỗi giai đoạn, tạo tài liệu học tiếng Việt đi kèm giải thích bản chất toán/nghiệp vụ và ghi lại các lỗi dữ liệu/kỹ thuật thực tế đã gặp và cách xử lý.

### Phi mục tiêu & Điều cần tránh (Non-Goals & Anti-Goals)
- **Không làm tiếp RAG / LLM / OCR trong phạm vi dự án này:** Tạm gác `financial-rag/`, `paddleocr_qwen_extraction.py` và hai notebook Scam Detection để tránh loãng định vị CV.
- **Không dùng dữ liệu tổng hợp giả lập hay file CSV trôi nổi không rõ nguồn gốc trên Kaggle.**
- **Không chạy train trên Kaggle khi chưa được phép:** Tuyệt đối đợi người dùng báo dự án Kaggle hiện tại chạy xong rồi mới import dataset và submit job train lên Kaggle.
- **Anti-goals (Thất bại dù code chạy được):**
  - Chia train/test ngẫu nhiên (`train_test_split` random) gây rò rỉ dữ liệu tương lai -> điểm cao giả tạo nhưng sụp đổ khi phỏng vấn.
  - Dùng `Distribution_Amount` ở cấp dòng phân bổ mà không gộp về cấp hóa đơn, làm đếm trùng hóa đơn có nhiều dòng hạch toán.
  - Chỉ báo cáo `Accuracy` trên dữ liệu mất cân bằng mà thiếu `PR-AUC`, `ROC-AUC`, `Calibration` và phân tích sai số theo ngưỡng nghiệp vụ.

---

## 2. Phương án Đã chọn & Đánh đổi (Approaches)

Trong giai đoạn định hướng, 3 phương án nguồn dữ liệu công khai uy tín đã được xem xét:
1. **Phương án 1 (Đã chọn): Giao dịch cấp hóa đơn từ Open Data Chính phủ Mỹ — Metro Nashville Vendor Payments.**  
   - *Được:* Dữ liệu thô cấp từng chứng từ (>1,3 triệu dòng phân bổ, 07/2022–12/2025), có cả `Invoice_Date` và `Payment_Date`, nhiều lỗi kế toán thực tế (ngày lỗi 1991/2083, dòng điều chỉnh âm, 1 séc trả nhiều hóa đơn, 1 hóa đơn tách nhiều dòng phân bổ) -> rèn tối đa kỹ năng làm sạch & tư duy Data Science; khớp trực tiếp với bài toán AP trước đây.  
   - *Đánh đổi:* Phản ánh chi trả khu vực công (không có cột `Payment_Terms` ghi sẵn trên từng dòng, nên cần quy ước ngưỡng trễ hạn chuẩn theo luật Prompt Payment là >30 ngày và kiểm tra độ nhạy ở ngưỡng >45 ngày).
2. **Phương án 2: UK Payment Practices Reporting (`check-payment-practices.service.gov.uk`).**  
   - *Được:* Dữ liệu doanh nghiệp tư nhân B2B thật, có sẵn tỷ lệ trả trễ.  
   - *Đánh đổi:* Dữ liệu tổng hợp 6 tháng/công ty, không ở cấp từng hóa đơn lẻ -> mất phần thực hành dữ liệu giao dịch thô.
3. **Phương án 3: Kết hợp cả hai.**  
   - *Đánh đổi:* Quá dàn trải, dễ sa vào ghép dữ liệu khác bối cảnh thay vì đào sâu mô hình và thống kê.

---

## 3. Nguồn Dữ liệu & Mô hình Hóa Bài toán (Các quyết định kỹ thuật đề xuất)

### 3.1 Nguồn dữ liệu thực tế đã kiểm chứng
- **Cổng dữ liệu:** Nashville Open Data Portal (ArcGIS FeatureServer công khai, không cần API key).
- **Endpoint:** `https://services2.arcgis.com/HdTo6HJqh92wn4D8/arcgis/rest/services/Metro_Vendor_Payments_view/FeatureServer/0`
- **Quy mô quan sát thực tế (ngày 2026-11-13):** `1.307.617` dòng; phạm vi `Payment_Date` từ `2022-07-01` đến `2025-12-31`.
- **Schema gốc:** `FID`, `Operating_Unit`, `Department_Number`, `Department_Name`, `Supplier_Number`, `Invoice_Number`, `Invoice_Date`, `Payment_Date`, `Payment_Number`, `Distribution_Amount`, `Check_Amount`.

### 3.2 Xử lý Grain & Làm sạch dữ liệu (Data Contract)
- **Vấn đề Grain (Độ hạt dữ liệu):** Quan sát mẫu thực tế cho thấy 1 `Invoice_Number` của cùng `Supplier_Number` có thể xuất hiện trên nhiều dòng phân bổ kế toán (`Distribution_Amount`) cùng `Payment_Number` và `Check_Amount`.
  - *Quyết định thiết kế (Đề xuất):* Dùng DuckDB SQL gộp tầng thô (`raw_distributions`) về tầng phân tích chính **Invoice-Payment Grain** theo khóa `(Operating_Unit, Supplier_Number, Invoice_Number, Payment_Number, Invoice_Date, Payment_Date)`, trong đó `invoice_amount = SUM(Distribution_Amount)`, `line_item_count = COUNT(*)`, `dept_count = COUNT(DISTINCT Department_Number)`, và giữ `primary_department = MIN(Department_Name)` (cùng `MIN(Department_Number)`) làm đơn vị đại diện (những hóa đơn phân bổ qua nhiều phòng ban sẽ được đánh dấu qua `dept_count > 1`).
- **Kiểm định & Phân loại bất thường (Audit Rules):**
  - Tách riêng (và ghi thống kê lý do loại trong báo cáo EDA) các bản ghi có: `Invoice_Date` nằm ngoài khoảng hợp lệ (quan sát thực tế có giá trị năm `1991` và `2083`), `Invoice_Date > Payment_Date`, thiếu `Supplier_Number`/`Invoice_Number`, hoặc `invoice_amount <= 0` (credit memo / hoàn trả — giữ lại thành feature cờ báo *nhà cung cấp có lịch sử điều chỉnh âm* trước khi lọc tập huấn luyện hóa đơn phải trả dương).

### 3.3 Định nghĩa Biến Mục tiêu & Chiến lược Chia Dữ liệu (Song song 2 Bài toán End-to-End)
- **Biến cơ sở:** `days_to_pay = (Payment_Date - Invoice_Date).days`.
- **Bài toán 1 — Phân loại Nguy cơ Trễ hạn (Binary Classification):**
  - Nhãn: `is_late = 1` nếu `days_to_pay > 30` (vượt chuẩn Prompt Payment 30 ngày), ngược lại `0` *(kiểm tra thêm phân phối thực tế và độ nhạy tại ngưỡng 45 ngày trong EDA)*.
  - Đánh giá: `PR-AUC`, `ROC-AUC`, `Brier Score` & `Calibration Curve`, `Lift@K` / `Precision@K` theo năng lực rà soát thủ công của kế toán, và phân tích nhạy cảm giá trị nghiệp vụ (*Business Impact sensitivity curve* theo dải tỷ lệ chi phí/năng lực kiểm tra, không hardcode một giả định cố định).
- **Bài toán 2 — Hồi quy Số ngày Thanh toán & Dự báo Dòng tiền (Regression & Cashflow Timing):**
  - Nhãn: `days_to_pay` (huấn luyện trên mục tiêu biến đổi `log1p(days_to_pay)` hoặc loss chịu ngoại lai như Huber/Quantile rồi quy đổi ngược về số ngày thực để so sánh công bằng).
  - Đánh giá: `MAE`, `RMSE`, `MedianAE` ở cấp từng hóa đơn, kèm sai số tổng hợp **Dự báo dòng tiền chi ra (Cash Outflow Forecast 7/14/30 ngày tới)** bằng cách cộng dồn `invoice_amount` theo `predicted_payment_date = Invoice_Date + round(pred_days)` và đối chiếu trực tiếp với dòng tiền thực chi trên tập Test.
- **Chia tập Out-of-Time (Chống Data Leakage & Right-Censoring):**
  - Sắp xếp theo `Invoice_Date`.
  - **Train:** `2022-07-01` đến `2024-06-30` (~2 năm tài khóa).
  - **Validation:** `2024-07-01` đến `2024-12-31` (6 tháng, chọn hyperparameters, calibrate xác suất và tối ưu ngưỡng).
  - **Test (Hold-out cuối cùng):** `2025-01-01` đến `2025-10-31` (cắt sớm hơn `max(Payment_Date) = 2025-12-31` đúng **61 ngày** để tránh **Right-Censoring Bias** — tránh làm méo cả nhãn `is_late` lẫn phân phối đuôi dài của `days_to_pay` cho các hóa đơn phát hành cuối năm nhưng chưa kịp thanh toán; mốc này sẽ được kiểm chứng lại bằng phân phối đuôi dài trong EDA).

### 3.4 Đặc trưng Chống Rò rỉ (Point-in-Time Features — chỉ dùng thông tin biết tại `Invoice_Date`)
> **Tuyệt đối KHÔNG dùng** `Payment_Date`, `Payment_Number`, `Check_Amount` làm biến đầu vào mô hình vì chỉ có sau khi đã thanh toán.

Các nhóm đặc trưng được tính **tuyệt đối từ quá khứ** (`prior_payment_date < current.invoice_date`, hoặc mở rộng rolling window có độ trễ an toàn):
1. **Đặc trưng hóa đơn hiện tại:** `invoice_amount`, `log_amount`, `line_item_count`, tỷ lệ `invoice_amount / vendor_expanding_median_amount`, số hóa đơn cùng ngày của cùng nhà cung cấp/phòng ban.
2. **Lịch sử nhà cung cấp (Vendor Historical Behavior):** số hóa đơn đã thanh toán trước `Invoice_Date`, tỷ lệ trễ hạn lịch sử (có Laplace smoothing), trung bình/trung vị/độ lệch chuẩn `days_to_pay` trong 90 ngày và 365 ngày trước đó, tần suất từng có credit memo (`amount < 0`).
3. **Tải công việc & Mức độ nghẽn của Phòng ban (Department Workload):** tổng số hóa đơn và tổng giá trị phòng ban tiếp nhận trong 7/30 ngày trước `Invoice_Date`, tỷ lệ trễ hạn lịch sử của phòng ban đó.
4. **Mùa vụ & Lịch tài khóa (Calendar / Fiscal):** tháng, quý, ngày trong tuần, khoảng cách tới cuối năm tài khóa Metro Nashville (30/06 hàng năm — thường có dồn thanh toán cuối năm).

---

## 4. Kiến trúc Kho lưu trữ & Hệ thống Học tập

### 4.1 Dọn dẹp Repository (Quyết định đề xuất)
- Đánh git tag `v1-legacy` tại commit `HEAD` hiện tại (`8ea7b7a`) để bảo toàn toàn bộ code cũ và file demo (`financial-rag/`, `paddleocr_qwen_extraction.py`, `ap_analysis.py`, `colab_learning.ipynb`, `kaggle_production.ipynb`, demo GIF/WebP) trong lịch sử Git.
- Xóa các file cũ khỏi nhánh `main` để `main` sạch 100% cho dự án Data Science mới.

### 4.2 Cấu trúc Dự án Mới
```text
project_data_analysis/
├── README.md                          # Tổng quan bài toán, kiến trúc, kết quả & hướng dẫn chạy
├── pyproject.toml / requirements.txt  # Quản lý phụ thuộc chuẩn (duckdb, pandas, pandera, lightgbm, xgboost, shap, pytest...)
├── .env.example                       # Biến cấu hình (KAGGLE_USERNAME, DATA_LIMIT...)
├── data/                              # Gitignored (trừ .gitkeep & sample nhỏ nếu cần)
│   ├── raw/                           # Dữ liệu thô tải từ Nashville Open Data
│   ├── interim/                       # Dữ liệu đã chuẩn hóa grain
│   └── processed/                     # Bảng feature sẵn sàng cho local & push lên Kaggle Dataset
├── docs/
│   ├── superpowers/specs/             # Bản thiết kế này
│   └── learning/                      # TÀI LIỆU HỌC TẬP & NHẬT KÝ VƯỚNG MẮC
│       ├── 00_roadmap_and_concepts.md # Lộ trình kiến thức DS cho dự án
│       ├── 01_data_audit_and_sql.md   # Bài học: Grain, Data Quality, DuckDB SQL
│       ├── 02_eda_and_statistics.md   # Bài học: Phân phối lệch, Kiểm định giả thuyết, Right-censoring
│       ├── 03_features_and_leakage.md # Bài học: Point-in-time features, Out-of-time split
│       ├── 04_modeling_and_eval.md    # Bài học: Baseline, GBDT (Classification & Regression), PR-AUC, Calibration, MAE/Cashflow, SHAP
│       └── troubleshooting_log.md     # Nhật ký vướng mắc thực tế & cách giải quyết
├── sql/
│   └── build_invoice_grain.sql        # Query DuckDB chuẩn hóa từ dòng phân bổ -> hóa đơn
├── src/
│   ├── config.py                      # Đường dẫn, mốc ngày split, ngưỡng trễ hạn
│   ├── data/
│   │   ├── fetch_nashville.py         # Tải dữ liệu thô phân trang + retry từ ArcGIS API
│   │   ├── validate_raw.py            # Pandera schema & báo cáo chất lượng dữ liệu
│   │   └── build_dataset.py           # Chạy DuckDB SQL tạo invoice-level parquet
│   ├── features/
│   │   └── build_features.py          # Point-in-time feature engineering dùng chung cho cả 2 bài toán (testable, no leakage)
│   ├── models/
│   │   ├── train_baseline.py          # Logistic Regression (Classification) & Ridge (Regression) tại local
│   │   ├── train_gbdt.py              # Huấn luyện LightGBM + XGBoost cho cả 2 bài toán (is_late + days_to_pay) & Calibration
│   │   └── evaluate.py                # Classification (PR-AUC, ROC-AUC, Brier, Lift@K), Regression (MAE, RMSE, Cashflow Error) & SHAP
│   └── app/
│       └── dashboard.py               # Streamlit dashboard: EDA, Risk Score + SHAP (Classification) & Dự báo dòng tiền (Regression)
├── notebooks/
│   ├── 01_eda_statistical_audit.ipynb # EDA trực quan + kiểm định giả thuyết (chạy local)
│   └── 02_kaggle_training.ipynb       # Notebook đẩy lên Kaggle để train & tune song song cả 2 bài toán
└── tests/
    ├── test_grain_and_cleaning.py     # Test gộp dòng, lọc ngày lỗi, xử lý credit memo
    ├── test_no_leakage.py             # Test tự động chứng minh feature tại t không đọc Payment_Date >= t
    └── test_metrics.py                # Test metric phân loại, metric hồi quy & tổng hợp cashflow
```

### 4.3 Quy trình Phối hợp Local vs Kaggle
1. **Giai đoạn 1 (Local — làm ngay sau khi chốt kế hoạch):** Xây pipeline tải dữ liệu thô, viết SQL gộp grain, viết unit tests (`pytest`), thực hiện Data Audit + EDA thống kê, xây & kiểm thử bộ feature chống leakage, và chạy **Baseline models (Logistic + Ridge)** tại local + xuất bộ tài liệu học `01`–`03`.
2. **Giai đoạn 2 (Kaggle — CHỈ thực hiện sau khi người dùng xác nhận dự án Kaggle kia đã xong):**
   - Xuất tập feature sạch (`data/processed/`) lên **Kaggle Dataset** bằng công cụ Kaggle.
   - Đẩy `notebooks/02_kaggle_training.ipynb` lên **Kaggle Kernels** để train & tune song song cả LightGBM/XGBoost Classifier (`is_late`) và LightGBM/XGBoost Regressor (`days_to_pay`) + SHAP.
   - Kéo kết quả (metrics cả 2 bài toán, model artifacts, biểu đồ SHAP/calibration/cashflow forecast) về repo, cập nhật dashboard, `README.md` và bài học `04_modeling_and_eval.md`.

---

## 5. Những điểm do Người thiết kế tự đề xuất mặc định (để Người dùng rà soát)

Các điểm sau chưa nêu cụ thể trong trao đổi trước nên được đề xuất mặc định trong tài liệu này (bạn có thể yêu cầu đổi khi duyệt):
1. **Giữ code cũ bằng Git tag `v1-legacy` tại commit `8ea7b7a`** rồi dọn nhánh `main` sạch sẽ, thay vì giữ thư mục `archive/` cồng kềnh trong repo.
2. **Quy ước mốc trễ hạn chính là `> 30 ngày`** (chuẩn Prompt Payment phổ biến tại Mỹ) cho nhãn phân loại nhị phân `is_late`, kèm kiểm tra độ nhạy ở mốc 45 ngày trong EDA.
3. **Ngăn Right-Censoring** bằng cách cắt điểm cuối `Invoice_Date` của tập Test tại `2025-10-31` (sớm hơn `max(Payment_Date) = 2025-12-31` đúng 61 ngày) để bảo toàn cả nhãn `is_late` lẫn phần đuôi của `days_to_pay`, và sẽ kiểm chứng lại qua phân phối thực tế ở bước EDA.
4. **Xử lý đa phòng ban khi gộp grain** bằng cách đếm `dept_count = COUNT(DISTINCT Department_Number)` và lấy `MIN(Department_Name)` làm phòng ban chính đại diện cho hóa đơn.
5. **Phân tích tác động nghiệp vụ (Business Impact)** dùng dải nhạy cảm tham số (ví dụ rà soát top `K%` hóa đơn rủi ro cao hoặc dải tỷ lệ chi phí sai số), không gán cứng một con số giả định duy nhất.
6. **Dùng DuckDB + Parquet** ở local để xử lý mượt 1,3 triệu dòng ngay trên máy cá nhân mà vẫn luyện trực tiếp kỹ năng SQL thực chiến.
