# Báo Cáo Kiểm Toán Chất Lượng Dữ Liệu & Hợp Nhất Cấp Độ Hạt SQL (DuckDB Grain Resolution & Quality Audit)

Tài liệu này ghi nhận toàn bộ quá trình phân tích cấp độ hạt (Grain Profiling), kiểm toán chất lượng dữ liệu (Data Quality Audit Funnel) và thiết kế truy vấn DuckDB SQL chuyển đổi 1,307,617 dòng giao dịch thô thành tập dữ liệu hóa đơn chuẩn hóa phục vụ bài toán dự báo rủi ro chậm trả AP (Accounts Payable Late Payment Risk).

---

## 1. Tóm Tắt Cốt Lõi (Executive Summary - Minto Pyramid)

- **Kết luận then chốt**: Bộ dữ liệu thô Nashville Vendor Payments ghi nhận ở **cấp độ dòng phân bổ chi phí** (Line Item Distribution Grain), **không phải cấp độ hóa đơn** (Invoice Grain). Việc phân tích trực tiếp trên dữ liệu thô sẽ phóng đại số lượng hóa đơn thực tế lên **1.52 lần** và gây sai lệch nghiêm trọng đối với các chỉ số tài chính.
- **Phễu kiểm toán thực chứng (Audited Evidence)**:
  - Tổng số bản ghi thô nạp vào: **1,307,617** dòng.
  - Số bản ghi thiếu khóa chính định danh (`dropped_missing_keys`): **0** dòng (100% bản ghi có đủ khóa định danh).
  - Số bản ghi vi phạm logic thời gian (`dropped_bad_dates`): **57,152** dòng (chiếm **4.37%**), bao gồm 81 lỗi gõ nhầm năm 1991, 190 lỗi gõ nhầm năm 2083 và 56,881 dòng có ngày hóa đơn đi sau ngày thanh toán (`Invoice_Date > Payment_Date`).
  - Số dòng thô hợp lệ đưa vào tổng hợp: **1,250,465** dòng.
  - Số hóa đơn hợp nhất (`grouped_rows`): **823,841** hóa đơn.
    - Hóa đơn thanh toán dương chuẩn (`clean_invoice_rows`): **811,914** hóa đơn (lưu tại `data/interim/invoices_clean.parquet`).
    - Bản ghi giảm trừ công nợ / Credit Memos (`credit_memo_rows`): **11,927** hóa đơn (lưu tại `data/interim/invoices_credit.parquet`).
- **Chỉ số kinh doanh trọng yếu (Business KPIs Grounding)**:
  - Thời gian xử lý thanh toán trung bình (`avg_days_to_pay`): **37.37 ngày**; trung vị (`median_days_to_pay`): **30.0 ngày**.
  - Tỷ lệ chậm trả thanh toán (`late_rate_pct` trên ngưỡng Net 30): **24.76%** (tương đương **201,026** hóa đơn bị trễ hạn).

---

## 2. Xác Định Cấp Độ Hạt Dữ Liệu (Grain Profiling)

### 2.1. Bản Chất Cấp Độ Hạt Của Dữ Liệu Thô (Raw Feed)
Trong hệ thống ERP tài chính của chính quyền đô thị Nashville (Metro Nashville), mỗi lần một nhà cung cấp gửi hóa đơn cho nhiều hạng mục thuộc các ngân sách/phòng ban khác nhau, hệ thống sinh ra nhiều dòng `Distribution_Amount`. 
- Cột `FID`: Định danh tuần tự của dòng phân bổ trong dịch vụ ArcGIS.
- Khóa định danh hóa đơn kinh doanh: Một hóa đơn trong thực tế được xác định duy nhất bởi bộ 6 trường:
  $$\text{Invoice Grain} = (\text{Operating\_Unit}, \text{Supplier\_Number}, \text{Invoice\_Number}, \text{Payment\_Number}, \text{Invoice\_Date}, \text{Payment\_Date})$$

### 2.2. Sự Cần Thiết Của Việc Hợp Nhất (Consolidation Rationale)
Nếu không gom nhóm về cấp độ hạt hóa đơn:
1. **Phóng đại mẫu (Sample Inflation)**: 1 hóa đơn có 10 dòng phân bổ sẽ bị mô hình ML xem như 10 quan sát độc lập, làm sai lệch trọng số huấn luyện.
2. **Data Leakage & Target Distortion**: Số tiền `Check_Amount` (tổng chi phiếu) bị lặp lại ở từng dòng phân bổ, trong khi `Distribution_Amount` chỉ là một phần chi phí con.
3. **Méo mó phân phối trễ hạn**: Hóa đơn lớn có nhiều dòng phân bổ sẽ chi phối tỷ lệ trễ hạn (`is_late`), dẫn đến ngụy biện thống kê (Simpson's Paradox).

---

## 3. Phễu Kiểm Toán Chất Lượng Dữ Liệu (Data Quality Audit Funnel)

Bảng đối chiếu số liệu thực tế được ghi nhận chính xác tại `data/interim/invoices_audit_summary.json`:

| Giai đoạn trong phễu | Số lượng bản ghi | Tỷ lệ so với Raw | Ý nghĩa kiểm toán |
| :--- | :---: | :---: | :--- |
| **1. Tổng bản ghi thô (Raw Rows)** | **1,307,617** | 100.00% | Toàn bộ dữ liệu trích xuất từ ArcGIS REST FeatureServer. |
| **2. Bản ghi thiếu khóa (`dropped_missing_keys`)** | **0** | 0.00% | Không có bản ghi nào bị khuyết `Operating_Unit`, `Supplier_Number`, `Invoice_Number`, `Payment_Number` hay ngày tháng. |
| **3. Bản ghi sai lệch ngày (`dropped_bad_dates`)** | **57,152** | 4.37% | Loại bỏ để bảo vệ tính toàn vẹn của mô hình thời gian. |
| *— Lỗi gõ nhầm quá khứ (< 2015-01-01)* | *81* | *0.006%* | Các năm 1991, 1998, 2001 do nhân viên nhập liệu gõ sai. |
| *— Lỗi gõ nhầm tương lai (> 2025-12-31)* | *190* | *0.015%* | Các mốc thời gian năm 2083 hoặc ngoài dải cho phép. |
| *— Ngày lập sau ngày trả (`Invoice_Date > Payment_Date`)* | *56,881* | *4.350%* | Nghịch lý kế toán (thanh toán tạm ứng hoặc nhập hồi tố). |
| **4. Bản ghi thô hợp lệ đưa vào tổng hợp** | **1,250,465** | 95.63% | Dữ liệu sạch đủ điều kiện tham gia gom nhóm. |
| **5. Tổng hóa đơn hợp nhất (`grouped_rows`)** | **823,841** | 63.00% | Kết quả sau khi GROUP BY bộ 6 khóa định danh. |
| *— Hóa đơn dương chuẩn (`clean_invoice_rows`)* | **811,914** | 62.09% | `invoice_amount > 0`: Dùng cho Feature Engineering & Huấn luyện. |
| *— Giảm trừ công nợ (`credit_memo_rows`)* | **11,927** | 0.91% | `invoice_amount <= 0`: Lưu riêng để tính đặc trưng lịch sử uy tín. |

---

## 4. Phân Tích Cấu Trúc Hóa Đơn Hợp Nhất (Consolidation Analysis)

### 4.1. Hóa Đơn Đa Dòng (Multi-line Consolidation)
- **Hóa đơn đơn dòng (Single-line)**: **710,607** hóa đơn (chiếm **87.52%** tập clean).
- **Hóa đơn đa dòng (Multi-line)**: **101,307** hóa đơn (chiếm **12.48%** tập clean).
- **Tỷ lệ nén trung bình**: 1,250,465 dòng thô / 823,841 hóa đơn = **1.5178 dòng/hóa đơn**.
- **Kỷ lục dòng phân bổ**: Hóa đơn có nhiều dòng phân bổ nhất chứa tới **574 dòng chi tiết** trên cùng một phiếu thanh toán.

### 4.2. Hóa Đơn Đa Phòng Ban (Cross-Department Invoices)
- **Hóa đơn đơn phòng ban (`dept_count = 1`)**: **810,029** hóa đơn (**99.77%**).
- **Hóa đơn đa phòng ban (`dept_count > 1`)**: **1,885** hóa đơn (**0.23%**).
- **Kỷ lục phòng ban cùng chi trả**: Có hóa đơn được đồng chi trả bởi **35 phòng ban** khác nhau của thành phố Nashville.
- **Quy tắc đại diện**: Trong truy vấn DuckDB, phòng ban đại diện (`primary_department` và `primary_dept_number`) được chọn theo thứ tự chuẩn hóa `MIN(Department_Name)` và `MIN(Department_Number)`.

---

## 5. Phân Phối Độ Trễ Thanh Toán & Rủi Ro Chậm Trả (Days to Pay & Late Risk)

Dựa trên dữ liệu 811,914 hóa đơn sạch tại `invoices_clean.parquet`:

- **Khung thời gian hoạt động**:
  - `Invoice_Date` nhỏ nhất: **2015-01-16**; lớn nhất: **2025-12-30**.
  - `Payment_Date` nhỏ nhất: **2022-07-01**; lớn nhất: **2025-12-31**.
- **Chỉ số thống kê về thời gian thanh toán (`days_to_pay`)**:
  - Giá trị nhỏ nhất (`MIN`): **0 ngày** (thanh toán ngay trong ngày nhận hóa đơn).
  - Trung vị (`MEDIAN` - 50%): **30.0 ngày** (trùng khớp với điều khoản thương mại Net 30 chuẩn).
  - Trung bình (`MEAN`): **37.37 ngày** (lệch phải do đuôi thanh toán trễ kéo dài).
  - Phân vị 90% (`P90`): **61.0 ngày** (10% hóa đơn mất hơn 2 tháng để được thanh toán).
  - Phân vị 95% (`P95`): **92.0 ngày** (5% hóa đơn mất hơn 3 tháng).
  - Phân vị 99% (`P99`): **226.0 ngày** (1% hóa đơn mất hơn 7.5 tháng).
  - Giá trị lớn nhất (`MAX`): **3,666 ngày** (~10 năm trễ hạn thanh toán).
- **Phân loại nhãn trễ hạn (`is_late`)**:
  - Hóa đơn đúng hạn (`days_to_pay <= 30`): **610,888** hóa đơn (**75.24%**).
  - Hóa đơn chậm trả (`days_to_pay > 30`): **201,026** hóa đơn (**24.76%**).
  - Tỷ lệ mất cân bằng mẫu: Khoảng 1 : 3.04 (tỷ lệ rất phù hợp cho các thuật toán cây như LightGBM/XGBoost mà không bị mất cân bằng quá cực đoan).

---

## 6. Phân Bổ Theo Phòng Ban & Nhà Cung Cấp Trọng Yếu

### 6.1. Top 5 Phòng Ban Có Khối Lượng Hóa Đơn Lớn Nhất
1. **MNPS (Metropolitan Nashville Public Schools - Mã 80)**: 366,343 hóa đơn (chiếm **45.12%** tổng số hóa đơn), tổng chi trả đạt **2.76 tỷ USD**.
2. **General Services (Mã 10)**: 70,316 hóa đơn (8.66%), tổng chi trả **354.3 triệu USD**.
3. **Water and Sewer (Mã 65)**: 57,167 hóa đơn (7.04%), tổng chi trả **1.74 tỷ USD**.
4. **Parks (Mã 40)**: 36,927 hóa đơn (4.55%), tổng chi trả **235.7 triệu USD**.
5. **Administrative (Mã 1)**: 34,825 hóa đơn (4.29%), tổng chi trả **3.38 tỷ USD**.

### 6.2. Top 5 Nhà Cung Cấp Có Khối Lượng Hóa Đơn Lớn Nhất
1. **Supplier 3422**: 56,015 hóa đơn, tổng thanh toán **49.5 triệu USD**.
2. **Supplier 1002609**: 38,162 hóa đơn, tổng thanh toán **10.0 triệu USD**.
3. **Supplier 336**: 30,879 hóa đơn, tổng thanh toán **18.1 triệu USD**.
4. **Supplier 7841**: 23,623 hóa đơn, tổng thanh toán **270.8 triệu USD**.
5. **Supplier 1013891**: 19,311 hóa đơn, tổng thanh toán **9.6 triệu USD**.

---

## 7. Kiến Trúc Kỹ Thuật DuckDB SQL (`sql/build_invoice_grain.sql`)

1. **Hiệu năng vượt trội**: Xử lý 1,307,617 dòng thô, phân loại kiểm toán, gom nhóm 6 khóa, tính toán 6 biểu thức tổng hợp và trích xuất 2 tệp Parquet chuẩn nén Snappy chỉ trong **~2.8 giây**.
2. **Cơ chế chống nhiễu số học (Floating-point Hygiene)**: Sử dụng hàm `ROUND(SUM(Distribution_Amount), 2)` để triệt tiêu các sai số biểu diễn nhị phân (như `0.00000000000001` làm trôi hóa đơn 0 đồng sang danh mục dương).
3. **Bảo toàn thông tin Credit Memos**: Tách riêng 11,927 bản ghi giảm trừ công nợ sang `data/interim/invoices_credit.parquet`, cho phép Task 5 xây dựng đặc trưng `vendor_has_prior_credit` mà không gây rò rỉ dữ liệu (No Data Leakage).
