# Lộ Trình Vòng Đời Data Science & Hệ Thống Khái Niệm Cốt Lõi
## Dự Án Financial Accounts Payable: Dự Báo Rủi Ro Trễ Hạn & Dòng Tiền Thanh Toán

Tài liệu này cung cấp cái nhìn tổng quan toàn diện về vòng đời dự án Data Science thực chiến trên dữ liệu thực tế **Metro Nashville Vendor Payments** (~1.3 triệu bản ghi thanh toán công). Dự án được thiết kế chuẩn mực từ thu thập dữ liệu, xử lý grain, chống rò rỉ thông tin (leakage), huấn luyện mô hình học máy, diễn giải SHAP, đến đóng gói dashboard tương tác.

---

### 1. Bối Cảnh Nghiệp Vụ & Đặt Bài Toán (Business Understanding & Problem Framing)

#### 1.1. Nghiệp Vụ Accounts Payable (AP - Công Nợ Phải Trả)
Trong quản trị tài chính doanh nghiệp và cơ quan công quyền, quy trình AP quản lý toàn bộ vòng đời từ khi nhận hoá đơn (Invoice) từ nhà cung cấp (Vendor/Supplier) đến khi phê duyệt, phát hành séc/chuyển khoản (Payment).
- **Rủi ro trễ hạn thanh toán (Late Payment)**: Gây tổn hại uy tín tín dụng, phát sinh chi phí phạt hợp đồng, làm gián đoạn chuỗi cung ứng hàng hoá/dịch vụ.
- **Kế hoạch dòng tiền (Cashflow Forecasting)**: Bộ phận ngân quỹ (Treasury) cần dự báo chính xác thời điểm thực tế tiền sẽ rời khỏi tài khoản để tối ưu hoá vốn lưu động và lãi suất vay ngắn hạn.

#### 1.2. Bài Toán Kép Trong Học Máy (Dual-Track Machine Learning)
Hệ thống giải quyết đồng thời hai mục tiêu bổ trợ nhau:
1. **Phân loại nhị phân (Classification - `is_late`)**:
   - Mục tiêu: Dự báo một hoá đơn mới tiếp nhận có nguy cơ thanh toán trễ (`days_to_pay > 30`) hay không.
   - Ứng dụng: Gắn cờ cảnh báo sớm để đội ngũ AP ưu tiên xử lý phê duyệt trước hạn chót.
2. **Hồi quy thời gian (Regression - `days_to_pay`)**:
   - Mục tiêu: Dự báo số ngày thực tế từ lúc nhận hoá đơn đến khi thanh toán hoàn tất.
   - Biến đổi mục tiêu: Áp dụng `log1p(days_to_pay)` nhằm xử lý hiện tượng phân phối lệch phải (right-skewed distribution) và đảm bảo tính hội tụ.
   - Ứng dụng: Xếp lịch dòng tiền chi tiết theo các khung thời gian (0-15 ngày, 16-30 ngày, 31-60 ngày, 60+ ngày).

---

### 2. Vòng Đời 7 Giai Đoạn Của Dự Án Data Science (DS Lifecycle)

```
[Giai đoạn 1: Ingestion & Contracts] 
   -> [Giai đoạn 2: Grain Resolution & Cleaning]
      -> [Giai đoạn 3: Exploratory Data Analysis]
         -> [Giai đoạn 4: Strict Temporal Feature Engineering]
            -> [Giai đoạn 5: Modeling & Probability Calibration]
               -> [Giai đoạn 6: Model Explainability & Cashflow Simulation]
                  -> [Giai đoạn 7: Production App & Monitoring]
```

#### Giai Đoạn 1: Thu Thập Tự Động & Hợp Đồng Dữ Liệu (Ingestion & Schema Contracts)
- **Nguồn dữ liệu**: ArcGIS REST FeatureServer chính thức của Metro Nashville Open Data.
- **Kỹ thuật Ingestion**:
  - Phân trang tuần tự với `resultOffset` và `resultRecordCount` (2,000 dòng/batch).
  - Nhận diện cờ `exceededTransferLimit` để tự động kéo toàn bộ dữ liệu sạch sẽ, không ngắt quãng.
  - Lưu trữ định dạng Parquet nén Snappy để tối ưu dung lượng và tốc độ I/O.
- **Hợp Đồng Dữ Liệu (Data Contracts với Pandera)**:
  - Khai báo schema tường minh (`DataFrameSchema`), kiểm tra kiểu dữ liệu, các trường khoá không rỗng (`FID`), kiểm tra hợp lệ của timestamp epoch-milliseconds.

#### Giai Đoạn 2: Chuẩn Hoá Hạt Dữ Liệu Bằng DuckDB (Grain Resolution & SQL Cleansing)
- **Vấn đề phân mảnh dòng (Multi-line item problem)**:
  - Dữ liệu gốc ở cấp độ `Distribution` (một hoá đơn có thể bị chia thành nhiều dòng ngân sách cho nhiều phòng ban).
  - Cần tổng hợp về đúng cấp độ `Invoice-Payment grain`: `SUM(Distribution_Amount)`, đếm số dòng (`line_item_count`), đếm số phòng ban liên quan (`dept_count`).
- **Phân luồng dữ liệu lỗi (Corrupted Records Routing)**:
  - Hoá đơn có ngày tháng phi lý (năm 1991, 2083), ngày hoá đơn sau ngày thanh toán (`Invoice_Date > Payment_Date`), hoặc số tiền âm/bằng 0 (Credit Memo).
  - Thay vì âm thầm loại bỏ, tách riêng ra bảng audit (`interim/corrupted_audit.parquet`) và tạo cờ lịch sử `vendor_has_credit_history` để bổ trợ cho mô hình.

#### Giai Đoạn 3: Khám Phá Thống Kê & Phân Tích Thực Trạng (EDA)
- Khảo sát phân phối thời gian xử lý: tỷ lệ thanh toán trong hạn vs trễ hạn (> 30 ngày).
- Phát hiện các quy luật chu kỳ: hiệu ứng cuối quý, cuối năm tài chính (Fiscal Year end tháng 6 đối với chính quyền công), ảnh hưởng của quy mô tiền tệ đối với thời gian kiểm duyệt.
- Kiểm định giả thuyết thống kê về sự khác biệt giữa các phòng ban (Police, Public Works, Schools) và các nhóm nhà cung cấp.

#### Giai Đoạn 4: Kỹ Thuật Tạo Đặc Trưng & Chống Rò Rỉ Tuyệt Đối (Anti-Leakage Temporal Feature Engineering)
- **Nguyên tắc vàng chống Data Leakage**:
  - Tại thời điểm nhận hoá đơn `Invoice_Date = t`, tuyệt đối **KHÔNG ĐƯỢC** sử dụng bất kỳ thông tin nào xuất hiện tại hoặc sau `t`.
  - Nghiêm cấm nhìn vào `Payment_Date`, `Payment_Number`, hoặc `Check_Amount` của chính hoá đơn đó hoặc các hoá đơn tương lai.
  - Khi tính lịch sử nhà cung cấp trước thời điểm `t`, chỉ được phép lấy các hoá đơn có ngày thanh toán thực tế hoàn tất **trước `t`** (`prior.Payment_Date < current.Invoice_Date`).
- **Chia tập dữ liệu theo dòng thời gian (Out-Of-Time Temporal Split)**:
  - `Train`: 2022-07-01 đến 2024-06-30 (2 năm, bao trọn 2 chu kỳ năm tài chính).
  - `Validation`: 2024-07-01 đến 2024-12-31 (6 tháng, dùng để early stopping và tinh chỉnh siêu tham số).
  - `Test`: 2025-01-01 đến 2025-10-31 (10 tháng, dùng đánh giá độc lập cuối cùng).
  - **Right-Censoring Buffer**: Khoảng đệm 61 ngày từ 2025-10-31 đến ngày kết thúc dữ liệu (2025-12-31) để đảm bảo các hoá đơn cuối kỳ đã có đủ thời gian thanh toán, tránh nhầm lẫn hoá đơn chưa tới hạn thành hoá đơn trễ hạn.

#### Giai Đoạn 5: Xây Dựng Mô Hình Đa Tầng & Hiệu Chuẩn Xác Suất (Modeling & Calibration)
- **Baseline Models**: Logistic Regression có cân bằng trọng số (`class_weight='balanced'`), Ridge Regression cho hồi quy số ngày.
- **Tree-based Boosting**: LightGBM và XGBoost với các đặc trưng tương tác phi tuyến, target encoding ngoại suy an toàn (out-of-fold).
- **Hiệu Chuẩn Xác Suất (Probability Calibration)**:
  - Sử dụng Isotonic Regression hoặc Platt Scaling trên tập Validation.
  - Đảm bảo xác suất dự báo rủi ro 70% phản ánh đúng 70% tần suất trễ hạn trong thực tế, giúp bộ phận tài chính tin tưởng vào ngưỡng cut-off rủi ro.

#### Giai Đoạn 6: Diễn Giải Mô Hình & Mô Phỏng Dòng Tiền (Explainability & Cashflow Simulation)
- **SHAP (SHapley Additive exPlanations)**:
  - TreeExplainer tính toán đóng góp biên (marginal contribution) của từng đặc trưng (số tiền hoá đơn, độ phức tạp phòng ban, độ trễ lịch sử của vendor).
  - Biểu đồ Waterfall cho từng hoá đơn cụ thể để giải thích vì sao bị xếp vào nhóm rủi ro cao.
- **Mô phỏng lịch trình dòng tiền (Cashflow Schedule)**:
  - Quy đổi kết quả hồi quy `days_to_pay` thành ngày thanh toán dự kiến (`predicted_payment_date`).
  - Tổng hợp ngân sách chi trả dự kiến theo các tuần và tháng tương lai.

#### Giai Đoạn 7: Triển Khai Ứng Dụng Tương Tác (Streamlit UI)
- Bảng điều khiển tài chính phục vụ người dùng cuối (AP Manager / CFO).
- Mô phỏng nhập hoá đơn mới và nhận kết quả đánh giá rủi ro tức thì kèm gợi ý can thiệp.

---

### 3. Quy Chuẩn Kỹ Thuật Dự Án (Engineering Standards)
1. **Kiểm thử hướng hành vi (TDD - Test-Driven Development)**: Luôn viết test kiểm chứng thất bại (RED) trước khi lập trình giải pháp và đạt trạng thái vượt qua (GREEN).
2. **Lean Code Hygiene**: Không dùng lệnh `print()` debug rải rác; sử dụng Python `logging` có cấu trúc; type hints 100% cho các hàm và module.
3. **Quản lý cấu hình bất biến (Immutable Configuration)**: Tất cả đường dẫn, siêu tham số và ngưỡng thời gian được quản lý tập trung trong `src/config.py` bằng `frozen dataclass`.
