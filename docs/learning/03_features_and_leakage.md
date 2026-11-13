# Thiết Kế Đặc Trưng Chống Rò Rỉ Dữ Liệu (Point-in-Time Feature Engineering & Out-of-Time Splits)

Tài liệu này ghi nhận toàn bộ thiết kế kiến trúc, bằng chứng định lượng và các quyết định kỹ thuật tại `src/features/build_features.py` cùng bộ kiểm thử tự động `tests/test_no_leakage.py` cho bài toán dự báo trễ hạn và dòng tiền AP trên **811,914** hóa đơn sạch của Metro Nashville.

---

## 1. Tóm Tắt Cốt Lõi (Executive Summary - Minto Pyramid)

- **Kết luận then chốt 1: Mọi đặc trưng tại thời điểm phát hành hóa đơn ($t = \text{invoice\_date}$) chỉ được phép đọc thông tin đã xảy ra trước $t$ ($\text{prior.payment\_date} < \text{curr.invoice\_date}$).**
  - Các cột hậu kiểm (`payment_date`, `payment_number`, `check_amount`, `days_to_pay`, `is_late`) tuyệt đối bị loại khỏi `FEATURE_COLS`.
  - Danh mục `FEATURE_COLS` gồm đúng **20 đặc trưng** (18 biến số `NUMERIC_FEATURES` + 2 biến phân loại `CATEGORICAL_FEATURES`), được cố định hợp đồng (schema contract) để các module huấn luyện Task 6 và Task 7 import trực tiếp.
- **Kết luận then chốt 2: Kiến trúc lai DuckDB `ASOF LEFT JOIN` + Two-Heap Running Median xử lý toàn bộ 811,914 dòng trong 4.04 giây với độ chính xác tuyệt đối.**
  - Thay vì self-join $O(N^2)$ (sẽ bùng nổ lên hơn $1.5 \times 10^9$ cặp cho riêng nhà cung cấp lớn nhất có 56,015 hóa đơn), toàn bộ chỉ số đếm/tổng/phương sai trượt 90 ngày và 365 ngày được nén theo ngày và tra cứu qua `ASOF LEFT JOIN` ($O(M \log M)$).
  - Trung vị lịch sử nhà cung cấp (`prior_vendor_median_amount`) được tính chính xác 100% bằng cấu trúc hai heap (max-heap + min-heap) quét hai con trỏ theo dòng thời gian.
  - Kiểm chứng ngẫu nhiên 50 dòng từ tập `val.parquet` đối chiếu với phép lọc brute-force Pandas cho kết quả khớp **50/50 (100%)**.
- **Kết luận then chốt 3: Chia tập Out-of-Time loại bỏ đồng thời Thiên lệch Cắt Trái (Left-Truncation) và Thiên lệch Cắt Phải (Right-Censoring).**
  - **Train (`2022-07-01` .. `2024-06-30`)**: **456,414 dòng** (24.55 MB), tỷ lệ trễ hạn **27.53%**, trung vị **30.0 ngày**, trung bình **38.24 ngày**, tỷ lệ nhà cung cấp mới (cold-start) **4.47%**.
  - **Validation (`2024-07-01` .. `2024-12-31`)**: **115,974 dòng** (6.40 MB), tỷ lệ trễ hạn **19.09%**, trung vị **30.0 ngày**, trung bình **33.05 ngày**, cold-start **2.17%**.
  - **Test (`2025-01-01` .. `2025-10-31`)**: **195,889 dòng** (10.73 MB), tỷ lệ trễ hạn **20.76%**, trung vị **30.0 ngày**, trung bình **30.62 ngày**, cold-start **1.97%**.
  - **Loại bỏ 2 vùng biên méo phân phối**: 21,329 dòng trước `2022-07-01` (bị cắt trái) và 22,308 dòng sau `2025-10-31` (bị cắt phải). Tổng cộng: $456,414 + 115,974 + 195,889 + 21,329 + 22,308 = 811,914$ dòng (bảo toàn 100%).

---

## 2. Bản Đồ Chống Rò Rỉ Dữ Liệu (Anti-Leakage Matrix)

Tại thời điểm kế toán nhận hóa đơn mới vào ngày $t = \text{invoice\_date}$:

| Nhóm thông tin | Trạng thái tại ngày $t$ | Quy tắc xử lý trong `build_features.py` |
| :--- | :--- | :--- |
| `invoice_amount`, `line_item_count`, `dept_count`, `operating_unit`, `primary_department`, `invoice_date` | **Đã biết** | Dùng trực tiếp: `log_invoice_amount`, lịch tài khóa, `same_day_vendor_inv_count`. |
| Hóa đơn trước của cùng phòng ban đã phát hành (`prior.invoice_date < t`) | **Đã biết** | Đếm khối lượng hồ sơ tiếp nhận: `dept_inv_count_7d`, `dept_inv_count_30d` trên $[t-7, t)$ và $[t-30, t)$. |
| Hóa đơn trước đã thanh toán xong (`prior.payment_date < t`) | **Đã biết** | Tổng hợp hành vi thanh toán quá khứ: `vendor_prior_paid_count`, `vendor_prior_late_rate_smoothed`, `avg_days_90d/365d`, `std_days_365d`, `amount_vs_vendor_median_ratio`, `dept_prior_late_rate`. |
| Credit memos có hiệu lực trước $t$ (`COALESCE(payment_date, invoice_date) < t`) | **Đã biết** | Cờ nhị phân `vendor_has_prior_credit` (dựa trên `MIN(effective_date) < t`). |
| Hóa đơn trước nhưng **chưa thanh toán tại $t$** (`prior.payment_date >= t`) | **CHƯA BIẾT (Tương lai)** | **Cấm đọc** nhãn `is_late`, `days_to_pay` hoặc `invoice_amount` đã thanh toán của nó. Test `test_features_ignore_future_payments_after_invoice_date` và `test_same_day_payment_not_counted_as_prior` kiểm chứng điều này. |
| `payment_date`, `payment_number`, `check_amount`, `days_to_pay`, `is_late` của chính dòng hiện tại | **CHƯA BIẾT (Tương lai)** | **Loại hoàn toàn** khỏi `FEATURE_COLS`, chỉ giữ làm metadata/target trong Parquet. |

---

## 3. Đặc Tả 20 Đặc Trưng Đầu Vào (`FEATURE_COLS`)

Toàn bộ **18 đặc trưng số** (`NUMERIC_FEATURES`) được đảm bảo hữu hạn 100% (`np.isfinite == True`, không `NaN`, không `Inf`) trên cả 3 tập `train`/`val`/`test`:

### 3.1. Đặc trưng quy mô hóa đơn hiện tại (4 biến)
1. `log_invoice_amount`: $\ln(1 + \max(\text{invoice\_amount}, 0))$ — nén phân phối power-law (từ $\$0.01$ tới $\$73.18\text{M}$).
2. `line_item_count`: Số dòng phân bổ hạch toán trên hóa đơn (trên tập Train: trung vị 1, p95 = 4, max = 574).
3. `dept_count`: Số phòng ban đồng chi trả (đánh dấu hóa đơn liên phòng ban, max = 35).
4. `amount_vs_vendor_median_ratio`: Tỷ lệ `invoice_amount / prior_vendor_median_amount` (với `prior.payment_date < t`). Cold-start mặc định `1.0`. Phát hiện hóa đơn có giá trị bất thường so với thói quen chính nhà cung cấp đó.

### 3.2. Lịch sử thanh toán nhà cung cấp (6 biến)
5. `vendor_prior_paid_count`: Tổng số hóa đơn đã hoàn tất thanh toán trước $t$ (trên Train: trung vị 513, p95 = 17,484, max = 37,782).
6. `vendor_prior_late_rate_smoothed`: Làm mịn Bayes (Laplace/Beta prior) với tham số cố định $\alpha = 2.5, \beta = 7.5$:
   $$\hat{p}_{\text{late}} = \frac{\text{prior\_late\_count} + 2.5}{\text{prior\_paid\_count} + 10.0}$$
   - **Tại sao dùng prior cố định ($\mu_0 = 0.25$, $n_0 = 10$) thay vì global mean?** Nếu lấy trung bình toàn bộ tập dữ liệu, thông tin phân phối tương lai của Val/Test sẽ rò rỉ vào Train. Prior cố định phản ánh chuẩn Net-30 (~25% trễ hạn) và giữ điểm khởi đầu an toàn cho cold-start (`0.25`) mà không rò rỉ liên tập.
7. `vendor_prior_avg_days_90d`: Trung bình `days_to_pay` các hóa đơn thanh toán trong $[t - 90\text{d}, t)$. Cold-start hoặc không có giao dịch 90d gán mốc chuẩn `30.0` ngày (tránh gán `0.0` khiến mô hình tuyến tính hiểu nhầm là "nhà cung cấp trả tức thì 0 ngày").
8. `vendor_prior_avg_days_365d`: Trung bình `days_to_pay` trong $[t - 365\text{d}, t)$, mặc định `30.0` khi chưa có lịch sử.
9. `vendor_prior_std_days_365d`: Độ lệch chuẩn mẫu ($ddof=1$) của `days_to_pay` trong $[t - 365\text{d}, t)$, mặc định `0.0` nếu số lượng $\le 1$.
10. `vendor_has_prior_credit`: Cờ nhị phân `1` nếu nhà cung cấp từng có ít nhất 1 credit memo hoàn thành trước $t$ (`60.95%` trên Train, `69.47%` trên Test).

### 3.3. Tải công việc & Lịch sử phòng ban + Cụm cùng ngày (4 biến)
11. `dept_inv_count_7d`: Số hóa đơn phòng ban tiếp nhận trong $[t - 7\text{d}, t)$ (trung vị Train: 392, p95: 2,757). Đo mức độ tắc nghẽn ngắn hạn của bộ phận kế toán phòng ban.
12. `dept_inv_count_30d`: Số hóa đơn phòng ban tiếp nhận trong $[t - 30\text{d}, t)$ (trung vị Train: 1,656, p95: 11,263).
13. `dept_prior_late_rate`: Tỷ lệ trễ hạn lịch sử làm mịn của `primary_department` trên các hóa đơn có `prior.payment_date < t` (công thức $\frac{\text{late} + 2.5}{\text{count} + 10.0}$).
14. `same_day_vendor_inv_count`: Số lượng hóa đơn cùng `supplier_number` gửi trong đúng ngày $t$ (biết ngay tại thời điểm $t$, trung vị 8, max 399 — nắm bắt hiệu ứng nộp chứng từ theo lô).

### 3.4. Lịch & Mùa vụ năm tài khóa (4 biến) + Phân loại (2 biến)
15. `inv_month`: Tháng dương lịch ($1..12$).
16. `inv_quarter`: Quý ($1..4$).
17. `inv_dayofweek`: Thứ trong tuần theo chuẩn Python/ISO ($0 = \text{Thứ Hai} \dots 6 = \text{Chủ Nhật}$; dùng `ISODOW(inv_day) - 1` trong DuckDB để đồng nhất với `pandas.Series.dt.dayofweek`).
18. `days_to_fiscal_year_end`: Số ngày từ `invoice_date` đến ngày khóa sổ năm tài chính Metro Nashville gần nhất (**30/06** hàng năm, từ `0` đến `365`). Nắm bắt trực tiếp đỉnh nghẽn tháng 5 và xả thanh toán tháng 6.
19-20. **Biến phân loại (`CATEGORICAL_FEATURES`)**: `operating_unit`, `primary_department`.

---

## 4. Kiến Trúc Tối Ưu Hiệu Năng: DuckDB `ASOF LEFT JOIN` + Two-Heap Median

### 4.1. Bài toán hiệu năng trên 811,914 hóa đơn
Trong tập dữ liệu có hiệu ứng Super-Pareto: nhà cung cấp lớn nhất (`Supplier 3422`) có **56,015 hóa đơn**; phòng ban lớn nhất (`MNPS`) có **366,343 hóa đơn**. Nếu dùng SQL self-join `ON a.supplier = b.supplier AND b.payment_date < a.invoice_date`:
- Riêng `MNPS` sẽ tạo $\frac{366,343^2}{2} \approx 6.7 \times 10^{10}$ (67 tỷ) cặp trung gian — làm tràn RAM hoặc mất hàng chục phút!

### 4.2. Giải pháp $O(N \log N)$ chạy trong 4.04 giây
1. **Nén theo hạt ngày (`Daily Aggregation`)**: Gom `base_invoices` theo `(supplier_number, pay_day)` và `(primary_department, pay_day)` để tính `day_cnt`, `day_late`, `day_days`, `day_days_sq` (với `CAST(... AS DOUBLE)`).
2. **Tổng tiền tố (`Prefix Sums`)**: Tính tổng tích lũy theo `pay_day` bằng Window Function (`SUM(...) OVER (PARTITION BY ... ORDER BY pay_day)`).
3. **Tra cứu trượt thời gian bằng `ASOF LEFT JOIN`**:
   - Với tập khóa truy vấn duy nhất `(supplier_number, inv_day)` (250,234 cặp thay vì 811,914 dòng):
     - `ASOF LEFT JOIN vendor_pay_cum c1 ON ... AND (q.inv_day - 1) >= c1.pay_day` $\to$ toàn bộ lịch sử `< inv_day`.
     - `ASOF LEFT JOIN vendor_pay_cum c90 ON ... AND (q.inv_day - 91) >= c90.pay_day` $\to$ lịch sử `< inv_day - 90d`.
     - Hiệu `c1 - c90` cho giá trị **chính xác tuyệt đối** trong cửa sổ $[t - 90\text{d}, t)$!
   - Tương tự cho cửa sổ 365 ngày (`c1 - c365`) dùng công thức phương sai ổn định trên tổng bình phương, và cho cửa sổ tải phòng ban 7 ngày / 30 ngày (`di1 - di7`, `di1 - di30`).
4. **Two-Heap Running Median (`_compute_vendor_prior_medians`)**:
   - Duyệt danh sách hóa đơn đã sắp xếp theo `(supplier_number, pay_day)` và danh sách ngày truy vấn `(supplier_number, inv_day)` bằng 2 con trỏ.
   - Mỗi hóa đơn chỉ được đưa vào cặp Max-Heap (`lo`) / Min-Heap (`hi`) **đúng 1 lần** ($O(\log K)$), đọc trung vị tức thì trong $O(1)$ (~0.75 giây cho toàn bộ 811,914 dòng!).

---

## 5. Tại Sao Phải Cắt Cả Hai Đầu Của Tập Thời Gian? (Left-Truncation vs Right-Censoring)

Khi chạy kiểm toán theo tháng trên `invoices_clean.parquet`, ta phát hiện một quy luật đối xứng quan trọng:

1. **Đầu trái — Thiên lệch Cắt Trái (Left-Truncation Bias trước `2022-07-01`)**:
   - Hệ thống Metro Nashville bắt đầu xuất bản sổ cái thanh toán từ **`min(Payment_Date) = 2022-07-01`**.
   - Bất kỳ hóa đơn nào phát hành trong tháng `2022-01` đến `2022-05` nhưng được thanh toán đúng hạn ($\le 30$ ngày) đều đã giải ngân xong trước `2022-07-01` nên **không tồn tại trong dữ liệu**.
   - Kết quả đo thực tế: **100.00%** hóa đơn tháng `2022-01`..`2022-05` trong tập clean đều có `is_late = 1` (trung vị 69–252 ngày), tổng cộng **21,329 dòng** trước `2022-07-01` có `late_rate = 52.40%`!
   - **Xử lý**: Giữ 21,329 dòng này trong bước tính đặc trưng lịch sử (vì `payment_date >= 2022-07-01` của chúng là sự kiện thanh toán thật xảy ra trong giai đoạn Train), nhưng **loại khỏi tập `train.parquet`** (`train_start = "2022-07-01"`) để mô hình không học tỷ lệ trễ giả tạo 100%.
2. **Đầu phải — Thiên lệch Cắt Phải (Right-Censoring Bias sau `2025-10-31`)**:
   - Ngày chốt dữ liệu thanh toán là `2025-12-31`, nên 22,308 hóa đơn phát hành tháng 11–12/2025 chỉ có `late_rate = 6.24%` (tháng 12 là `0.00%`).
   - **Xử lý**: Đặt `test_end = "2025-10-31"`, bảo toàn vùng đệm 61 ngày ($\ge p_{90}$).

---

## 6. Ghi Nhớ Phỏng Vấn (Interview Takeaways)

1. **"Làm sao bạn chứng minh feature pipeline không bị data leakage?"**
   - Phân tách rạch ròi thời điểm biết thông tin: tại $t = \text{invoice\_date}$, chỉ được join các hóa đơn có `prior.payment_date < t` (dấu `<` nghiêm ngặt, không `<=` để loại trừ thanh toán cùng ngày).
   - Loại bỏ toàn bộ các trường phát sinh sau thanh toán (`payment_date`, `payment_number`, `check_amount`, `days_to_pay`, `is_late`) qua hằng số hợp đồng `FEATURE_COLS`.
   - Có unit test tự động (`tests/test_no_leakage.py`) chứng minh: nếu hóa đơn A lập tháng 1 nhưng tháng 3 mới trả, thì hóa đơn B lập tháng 2 phải thấy `vendor_prior_paid_count == 0`.
2. **"Tại sao lại làm mịn tỷ lệ trễ hạn bằng hằng số thay vì lấy trung bình toàn tập?"**
   - Lấy trung bình toàn tập (kể cả Val/Test) là target distribution leakage. Dùng Bayesian Beta prior cố định ($\alpha=2.5, \beta=7.5$) dựa trên chuẩn Net-30 vừa chống rò rỉ liên tập, vừa điều tiết các nhà cung cấp chỉ mới có 1–2 hóa đơn.
3. **"Nếu phỏng vấn hỏi cách tối ưu rolling feature theo ngày không đồng đều trên 1 triệu dòng?"**
   - Không chạy `groupby().rolling()` trên chuỗi thô có trùng ngày hay self-join $O(N^2)$; thay vào đó nén về bảng tích lũy theo ngày rồi dùng **DuckDB `ASOF LEFT JOIN`** cho các chỉ số cộng tính (count/sum/sum-of-squares) và **Two-Heap Two-Pointer** cho các thống kê thứ tự (median), giảm thời gian chạy xuống **~4 giây**.
