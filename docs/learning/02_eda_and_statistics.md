# Báo Cáo Phân Tích Khám Phá Dữ Liệu (EDA) & Kiểm Định Giả Thuyết Thống Kê
## Metro Nashville Accounts Payable Late Payment Prediction

Tài liệu này tổng hợp toàn bộ kết quả phân tích khám phá dữ liệu thống kê (Statistical Exploratory Data Analysis), kiểm toán phân phối, bằng chứng thực nghiệm về thiên lệch cắt phải (Right-Censoring Bias) và 4 bài toán kiểm định giả thuyết thống kê chính thức (Formal Hypothesis Testing) thực thi trên tập dữ liệu 811,914 hóa đơn thanh toán sạch của chính quyền đô thị Nashville (Metro Nashville). Toàn bộ số liệu và đồ thị trong tài liệu này được tạo ra và kiểm chứng trực tiếp từ notebook thực thi: `notebooks/01_eda_statistical_audit.ipynb`.

---

## 1. Tóm Tắt Cốt Lõi (Executive Summary - Minto Pyramid)

- **Kết luận then chốt 1: Phân phối thời gian xử lý thanh toán mang tính chất lệch phải cực đoan (Heavy Right-Skewed) và tập trung đột biến tại Net 30.**
  - Chỉ số trung vị (`Median`) và phân vị 75 (`p75`) cùng hội tụ chính xác tại **30.0 ngày** (IQR chỉ là 3.0 ngày: $p25 = 27.0$, $p75 = 30.0$).
  - Trong khi đó, số bình quân (`Mean = 37.37 ngày`) bị kéo lệch bởi đuôi cực dài vươn tới 3,666 ngày (độ lệch skewness = 13.30, kurtosis = 338.99). Thống kê tham số (Mean, Std) hoàn toàn sai lệch; bắt buộc phải sử dụng thống kê phi tham số (Median, IQR, Percentiles) và kiểm định Mann-Whitney U.
- **Kết luận then chốt 2: Thiên lệch cắt phải (Right-Censoring Bias) được triệt tiêu bằng vùng đệm 61 ngày.**
  - Trong 2 tháng cuối của tập dữ liệu (tháng 11 và 12/2025), tỷ lệ chậm trả quan sát sụp đổ giả tạo về 9.12% và 0.00% do ngày cắt thanh toán tối đa là `2025-12-31`.
  - Phân vị 90 (`p90`) của toàn bộ danh mục là **61.0 ngày** (chỉ 10.08% hóa đơn mất $>60$ ngày để thanh toán). Việc thiết lập mốc kết thúc tập kiểm thử `test_end = 2025-10-31` bảo đảm đúng **61 ngày đệm** (tháng 11 + 12), cho phép $\ge 90\%$ hóa đơn được giải ngân tự nhiên mà không bị méo mó do cắt cụt.
- **Kết luận then chốt 3: Hiệu ứng mùa vụ tài khóa (Fiscal Year-End Seasonality) và đơn vị vận hành (Operating Unit) là tín hiệu dự báo mạnh mẽ.**
  - Tháng 5 ghi nhận tỷ lệ trễ hạn kỷ lục **37.27%** do các phòng ban dồn ứ hồ sơ phê duyệt trước thời điểm chốt sổ ngân sách.
  - Tháng 6 ghi nhận tỷ lệ trễ hạn hạ nhiệt xuống **22.18%** ($\chi^2 = 233.83, p = 8.70 \times 10^{-53}$) nhờ nỗ lực giải ngân tất toán khẩn cấp của kho bạc thành phố.
  - Hệ thống trường công lập `METRO_MNPS` có tỷ lệ trễ hạn **30.92%**, cao hơn đáng kể so với chính quyền tổng hợp `METRO_GG` (**19.66%**, $\chi^2 = 13,700.04, p < 10^{-100}$, Cramér's $V = 0.1299$).
- **Kết luận then chốt 4: Mức độ tập trung nhà cung cấp mang tính chất Siêu Pareto (Super-Pareto Concentration).**
  - Top 20% nhà cung cấp (3,938 nhà cung cấp) nắm giữ **99.73%** tổng ngân sách chi tiêu ($12.91 tỷ / $12.94 tỷ) và phát sinh **96.78%** tổng số hóa đơn bị trễ hạn.

---

## 2. Bảng Đối Chiếu Chỉ Số Thống Kê Thực Chứng (Grounding KPI Matrix)

Bảng đối chiếu toàn diện các đại lượng thống kê được trích xuất trực tiếp từ kết quả chạy notebook:

| Hạng mục phân tích | Chỉ số / Phân vị thực tế | Giá trị đo lường | Ý nghĩa kinh doanh & Kỹ thuật mô hình |
| :--- | :--- | :---: | :--- |
| **Phễu Kiểm Toán** | Tổng bản ghi thô nạp vào | 1,307,617 dòng | Toàn bộ dữ liệu trích xuất từ ArcGIS REST FeatureServer |
| | Dòng vi phạm thời gian (`bad_dates`) | 57,152 dòng (4.37%) | Gõ nhầm năm (1991, 2083) hoặc ngày hóa đơn đi sau ngày thanh toán |
| | Hóa đơn sạch thanh toán (`clean_invoices`) | 811,914 dòng (98.55%) | Tập dữ liệu dương (`invoice_amount > 0`) đưa vào huấn luyện |
| | Hóa đơn bù trừ công nợ (`credit_memos`) | 11,927 dòng (1.45%) | Tách riêng bảo vệ mô hình khỏi giá trị âm |
| **Phân Phối `days_to_pay`** | Min / p25 / Median (p50) / p75 | 0d / 27d / 30d / 30d | Điểm tập trung cao độ tại Net 30, IQR hẹp (3.0 ngày) |
| | Mean / Standard Deviation | 37.37d / 52.21d | Mean lớn hơn Median 7.37 ngày do đuôi phải kéo dài |
| | p90 / p95 / p99 / Max | 61d / 92d / 226d / 3,666d | 90% hóa đơn giải ngân trong 61 ngày; max trễ 10 năm |
| | Skewness / Kurtosis | 13.30 / 338.99 | Lệch phải cực lớn, độ nhọn leptokurtic cực cao |
| **Phân Phối `invoice_amount`** | Min / p25 / Median (p50) / p75 | $0.01 / $127.00 / $385.71 / $1,700.00 | Đa số hóa đơn có giá trị nhỏ; 50% dưới $385.71 |
| | Mean / Standard Deviation | $15,943.75 / $292,783.34 | Bị chi phối bởi các dự án xây dựng hạ tầng hàng chục triệu USD |
| | p90 / p95 / p99 / Max | $8,059.96 / $23,747.07 / $265,189.87 / $73,182,359.43 | Phân phối quyền lực (Power-law), bắt buộc dùng Log transform |
| | Skewness / Kurtosis | 98.56 / 14,976.79 | Mức độ bất đối xứng cực độ |
| **Ngưỡng Trễ Hạn** | Net 15 (`days_to_pay > 15`) | 72.63% (589,715 hóa đơn) | Ngưỡng quá khắt khe, không phù hợp thực tế quy trình hành chính |
| | **Net 30 (`days_to_pay > 30`)** | **24.76% (201,026 hóa đơn)** | **Chuẩn mực thanh toán công; tỷ lệ mất cân bằng mẫu 3:1 lý tưởng** |
| | Net 45 (`days_to_pay > 45`) | 14.41% (116,958 hóa đơn) | Ngưỡng trễ hạn nợ nghiêm trọng (Delinquent) |
| | Net 60 (`days_to_pay > 60`) | 10.08% (81,870 hóa đơn) | Khớp chính xác với ngưỡng phân vị 90 (61 ngày) |
| **Kiểm Định Giả Thuyết** | Đa phòng ban (`dept_count > 1`) | Mann-Whitney U = 664,699,538 | $p = 4.29 \times 10^{-23}, z = -9.72, r = 0.0108$ (Bác bỏ $H_0$) |
| | Hóa đơn giá trị cao ($> \$1,700$) | Mann-Whitney U = 57,883,994,058.5 | $p < 10^{-100}, z = -42.57, r = 0.0472$ (Bác bỏ $H_0$) |
| | Tháng 6 (Tất toán tài khóa) vs Khác | $\chi^2 = 233.83, df = 1$ | $p = 8.70 \times 10^{-53}$, Cramér's $V = 0.0170$ (Bác bỏ $H_0$) |
| | `METRO_GG` vs `METRO_MNPS` | $\chi^2 = 13,700.04, df = 1$ | $p < 10^{-100}$, Cramér's $V = 0.1299$ (Tác động vừa - Bác bỏ $H_0$) |
| **Phân Tích Pareto** | Tổng ngân sách chi tiêu | $12,944,954,204.07 (~$12.94 tỷ) | 19,688 nhà cung cấp độc lập |
| | Tỷ lệ chi tiêu Top 10 nhà cung cấp | 40.17% ($5.20 tỷ) | Phụ thuộc cao vào nhóm nhà thầu giáo dục & xây lắp lớn |
| | Tỷ lệ chi tiêu Top 20% nhà cung cấp | 99.73% ($12.91 tỷ) | Hiện tượng siêu tập trung (Super-Pareto) |
| | Tỷ lệ trễ hạn Top 20% nhà cung cấp | 96.78% (194,553 / 201,026) | Rủi ro hoạt động tập trung phần lớn ở nhóm nhà cung cấp định kỳ |

---

## 3. Phân Tích Chi Tiết 6 Trụ Cột Thống Kê & EDA

### 3.1. Phễu Kiểm Toán Chất Lượng Dữ Liệu (Data Audit Funnel)

![01_data_audit_funnel.png](file:///D:/Antigravity/projects/project_data_analysis/.worktrees/feat-ds-rebuild/artifacts/eda/01_data_audit_funnel.png)

Quá trình hợp nhất từ 1,307,617 dòng thô sang 811,914 hóa đơn sạch đi qua 5 giai đoạn nghiêm ngặt:
1. **100% bản ghi thô có đầy đủ khóa định danh**: Không có bất kỳ dòng nào bị khuyết 6 khóa kinh doanh (`Operating_Unit`, `Supplier_Number`, `Invoice_Number`, `Payment_Number`, `Invoice_Date`, `Payment_Date`).
2. **Loại bỏ 57,152 dòng vi phạm tính toàn vẹn thời gian (4.37%)**:
   - 81 dòng gõ sai năm về quá khứ (1991, 1998, 2001).
   - 190 dòng gõ sai năm vượt tương lai (2083).
   - 56,881 dòng ghi nhận ngày hóa đơn xảy ra sau ngày thanh toán (`Invoice_Date > Payment_Date`), một bất thường logic do quy trình tạo chứng từ bù sau khi đã xuất séc thanh toán.
3. **Tổng hợp cấp độ hạt hóa đơn (Invoice Grain Resolution)**:
   - 1,250,465 dòng thô hợp lệ được DuckDB gom nhóm thành **823,841 hóa đơn duy nhất**, tỷ lệ nén đạt $1.52$ dòng phân bổ trên mỗi hóa đơn.
   - 11,927 hóa đơn (chiếm 1.45%) có tổng tiền $\le 0$ được bóc tách riêng thành tệp Credit Memos (`data/interim/invoices_credit.parquet`).
   - 811,914 hóa đơn có tổng tiền dương được lưu trữ tại `data/interim/invoices_clean.parquet` với đầy đủ 14 cột định danh, thuộc tính thời gian và biến mục tiêu `is_late`.

---

### 3.2. Phân Phối Lệch Phải & Giới Hạn Của Thống Kê Tham Số

![02_distributions_skew.png](file:///D:/Antigravity/projects/project_data_analysis/.worktrees/feat-ds-rebuild/artifacts/eda/02_distributions_skew.png)

#### A. Hiện tượng "Đỉnh Nhọn Net 30" (Spike at 30 Days)
- Trong các giáo trình thống kê cơ bản, các biến ngẫu nhiên thường được giả định xấp xỉ phân phối chuẩn (Gaussian: đối xứng, skewness $\approx 0$, kurtosis $\approx 3$).
- Tuy nhiên, trong thanh toán công vụ:
  - Luật thanh toán công (Prompt Payment Act) quy định thời hạn tiêu chuẩn là 30 ngày.
  - Các hệ thống ERP (như Oracle/SAP của Metro Nashville) thiết lập quy tắc tự động lên lịch thanh toán vào ngày thứ 30 kể từ ngày nhận hóa đơn.
  - Do đó, xác suất xuất hiện thời gian giải ngân tại mốc 27 đến 30 ngày đạt mật độ khổng lồ: $p25 = 27$ ngày, $p50 = 30$ ngày và $p75 = 30$ ngày!
  - Khoảng trải giữa (IQR) chỉ vỏn vẹn **3.0 ngày**, biểu thị sự tập trung cực kỳ chặt chẽ của các quy trình thanh toán đúng hạn.

#### B. Nguy cơ khi sử dụng Thống kê Tham số (Parametric Fallacy)
- Nếu một chuyên viên phân tích báo cáo: *"Thời gian thanh toán trung bình của thành phố là 37.37 ngày với độ lệch chuẩn 52.21 ngày"*, họ sẽ dẫn dắt ban lãnh đạo đến những nhận định hoàn toàn sai lệch:
  1. Dải $\mu \pm \sigma$ sẽ là $[-14.84, 89.58]$ ngày: Một con số vô nghĩa vì số ngày thanh toán không thể âm.
  2. Số trung bình $37.37$ ngày tạo cảm giác rằng phần lớn hóa đơn đang bị thanh toán chậm (vì $> 30$ ngày). Nhưng trên thực tế, **hơn 75% hóa đơn được thanh toán trong vòng 30 ngày**! Số trung bình bị kéo lệch lên trên bởi 1% số hóa đơn có thời gian giải ngân kéo dài từ 226 đến 3,666 ngày (tranh chấp hợp đồng, kiện tụng hoặc vướng mắc giải phóng mặt bằng).
- **Nguyên tắc kỹ thuật**: Đối với dữ liệu AP, bắt buộc phải dùng **Median (30 ngày)** và **IQR (3 ngày)** làm thước đo vị trí và độ phân tán trung tâm, kết hợp các mốc phân vị cao **p90 (61 ngày), p95 (92 ngày), p99 (226 ngày)** để giám sát đuôi rủi ro.

#### C. Quy luật lũy thừa của Giá trị hóa đơn (`invoice_amount`)
- Giá trị hóa đơn có trung vị chỉ là **$385.71** và 75% hóa đơn có giá trị dưới **$1,700.00**.
- Tuy nhiên, giá trị trung bình lên tới **$15,943.75** và giá trị lớn nhất đạt **$73,182,359.43** (hóa đơn xây dựng công trình công cộng).
- Độ lệch skewness đạt **98.56** và kurtosis đạt **14,976.79**. Bất kỳ mô hình học máy nào (đặc biệt là hồi quy tuyến tính, Logistic Regression hay Neural Networks) nếu nạp trực tiếp giá trị tiền thô sẽ bị bão hòa trọng số bởi các siêu ngoại lai.
- **Giải pháp**: Biến đổi logarit $\log_{10}(\text{invoice\_amount})$ giúp nén thang đo từ $[10^{-2}, 10^8]$ về $[-2, 7.86]$, đưa phân phối về dạng chuông gần chuẩn, hỗ trợ thuật toán học máy phân tách đặc trưng ổn định.

---

### 3.3. Bằng Chứng Thực Nghiệm Về Thiên Lệch Cắt Phải (Right-Censoring Verification)

![03_right_censoring_verification.png](file:///D:/Antigravity/projects/project_data_analysis/.worktrees/feat-ds-rebuild/artifacts/eda/03_right_censoring_verification.png)

#### A. Bản chất toán học của Right-Censoring trong dữ liệu AP
Giả sử ta thu thập dữ liệu thanh toán đến mốc thời gian $T_{\text{cutoff}} = \text{2025-12-31}$.
Một hóa đơn phát hành tại thời điểm $t_i$. Để hóa đơn này được ghi nhận trạng thái thanh toán với thời gian xử lý $D_i = t_{\text{pay}} - t_i$, điều kiện cần là:
$$t_{\text{pay}} = t_i + D_i \le T_{\text{cutoff}} \iff D_i \le T_{\text{cutoff}} - t_i$$

Nếu một hóa đơn phát hành vào ngày **15/12/2025** ($T_{\text{cutoff}} - t_i = 16$ ngày):
- Nếu hóa đơn đó được thanh toán trong vòng 10 ngày $\to$ Xuất hiện trong tập dữ liệu với $D_i = 10 \le 30$ (Đúng hạn).
- Nếu hóa đơn đó cần 45 ngày để thanh toán $\to$ Đến ngày 31/12/2025 hóa đơn vẫn chưa được trả tiền, do đó **không có bản ghi thanh toán nào được tạo ra trong hệ thống** tại ngày chốt sổ!
- **Hệ quả chết người**: Những hóa đơn thanh toán trễ phát sinh vào cuối kỳ thu thập bị loại bỏ khỏi mẫu một cách có hệ thống, chỉ những hóa đơn thanh toán siêu nhanh mới lọt vào tập dữ liệu.

#### B. Minh chứng số liệu sụp đổ trong 5 tháng cuối năm 2025
Quan sát bảng theo dõi chuỗi thời gian năm 2025 từ notebook:

| Tháng phát hành hóa đơn | Số lượng hóa đơn (`n`) | Median `days_to_pay` | Mean `days_to_pay` | p90 `days_to_pay` | Tỷ lệ trễ hạn (`late_rate_30`) |
| :---: | :---: | :---: | :---: | :---: | :---: |
| **2025-08** | 20,526 | 30.0 ngày | 29.24 ngày | 36.0 ngày | 13.13% |
| **2025-09** | 20,582 | 30.0 ngày | 28.92 ngày | 42.0 ngày | 17.16% |
| **2025-10** (Mốc cắt Test) | 20,299 | 29.0 ngày | 27.39 ngày | 38.0 ngày | 16.39% |
| **2025-11** (Bị cắt cụt) | 15,250 | 29.0 ngày | 25.60 ngày | 30.0 ngày | **9.12% (Sụp đổ)** |
| **2025-12** (Bị cắt cụt) | 7,058 | **16.0 ngày** | **15.62 ngày** | **25.0 ngày** | **0.00% (Ảo tưởng tuyệt đối)** |

Tháng 12/2025 có số ngày thanh toán trung vị chỉ là **16.0 ngày**, không có bất kỳ hóa đơn nào bị trễ hạn (0.00%), và số lượng hóa đơn sụt giảm nghiêm trọng từ ~20,000 xuống 7,058 hóa đơn. Đây là minh chứng giáo khoa hoàn hảo về **Right-Censoring Bias**.

#### C. Chứng minh tính chuẩn xác của Vùng Đệm 61 Ngày (`test_end = 2025-10-31`)
- Phân vị 90 của toàn bộ 811,914 hóa đơn lịch sử là $p90 = 61.0$ ngày. Chỉ có **10.08%** hóa đơn mất $>60$ ngày để giải ngân.
- Khoảng cách từ mốc kết thúc tập kiểm thử `test_end = 2025-10-31` đến mốc đóng dữ liệu `max_valid_payment_date = 2025-12-31` chính xác là:
  $$\Delta t = 30 \text{ ngày (tháng 11)} + 31 \text{ ngày (tháng 12)} = 61 \text{ ngày}$$
- Vùng đệm 61 ngày bảo đảm rằng mọi hóa đơn phát hành trong tập Test (kết thúc vào 31/10/2025) đều có đủ ít nhất 61 ngày thời gian quan sát thực tế. Do đó, $\ge 90\%$ các trường hợp trễ hạn tự nhiên đều được ghi nhận đầy đủ, loại bỏ hiện tượng rò rỉ cắt cụt và bảo vệ tính công tâm tuyệt đối cho tập đánh giá mô hình.

---

### 3.4. Độ Nhạy Ngưỡng Trễ Hạn & Tính Mùa Vụ Năm Tài Khóa

![04_threshold_and_seasonality.png](file:///D:/Antigravity/projects/project_data_analysis/.worktrees/feat-ds-rebuild/artifacts/eda/04_threshold_and_seasonality.png)

#### A. Phân tích độ nhạy ngưỡng thanh toán
Trong các hợp đồng thương mại và mua sắm công vụ, điều khoản công nợ có thể linh hoạt theo thỏa thuận:
- **Ngưỡng Net 15 (`> 15 ngày`)**: Chiếm **72.63%** (589,715 hóa đơn). Trong khu vực công, quy trình đối soát chứng từ 3 bên (Three-way Matching: Hóa đơn, Đơn đặt hàng PO, Phiếu nhập kho) hiếm khi hoàn tất trước 15 ngày. Ngưỡng này quá gắt gao và gây nhiễu cảnh báo giả.
- **Ngưỡng Net 30 (`> 30 ngày`)**: Chiếm **24.76%** (201,026 hóa đơn). Đây là chuẩn mực Prompt Payment Act. Tỷ lệ mất cân bằng giữa lớp trễ (24.76%) và lớp đúng hạn (75.24%) xấp xỉ **1:3**, một tỷ lệ vàng cho các thuật toán phân loại nhị phân mà không đòi hỏi kỹ thuật oversampling cực đoan (như SMOTE).
- **Ngưỡng Net 45 (`> 45 ngày`)**: Chiếm **14.41%** (116,958 hóa đơn). Đại diện cho các trường hợp vướng mắc giải trình hoặc trục trặc hành chính vừa.
- **Ngưỡng Net 60 (`> 60 ngày`)**: Chiếm **10.08%** (81,870 hóa đơn). Đại diện cho các khoản nợ chậm trễ nghiêm trọng.

#### B. Chu kỳ mùa vụ năm tài khóa (Metro Nashville Fiscal Year Close - June 30)
Năm tài chính của chính quyền đô thị Nashville kéo dài từ ngày 01/07 năm trước đến ngày **30/06 năm sau**:
- **Tháng 4 và Tháng 5 (Giai đoạn nước rút trước hạn chót)**:
  - Tỷ lệ trễ hạn tăng vọt từ mức bình quân 24% lên **30.45%** (tháng 4) và đạt đỉnh **37.27%** (tháng 5).
  - Nguyên nhân: Các đơn vị dồn dập gửi hóa đơn quyết toán các gói thầu trong năm tài khóa để kịp hấp thụ hết dự toán ngân sách đã phê duyệt, gây quá tải bộ phận kiểm soát thanh toán.
- **Tháng 6 (Tháng chốt sổ tài khóa - Fiscal Close Acceleration)**:
  - Tỷ lệ trễ hạn giảm đột ngột từ 37.27% xuống **22.18%**.
  - Nguyên nhân: Kho bạc và phòng tài chính tập trung toàn lực, phê duyệt khẩn cấp ngày đêm để giải ngân và đóng sổ sách kế toán trước ngày 30/06, hạn chế chuyển nợ sang năm tài khóa sau.
- **Tháng 7 và Tháng 8 (Đầu năm tài khóa mới)**:
  - Tỷ lệ trễ hạn duy trì ở mức thấp kỷ lục trong năm: **19.48%** (tháng 7) và **18.44%** (tháng 8). Khối lượng ngân sách mới được cấp dồi dào, các thủ tục giải ngân đầu năm diễn ra thông suốt.

---

### 3.5. Kiểm Định Giả Thuyết Thống Kê Thực Nghiệm (Formal Hypothesis Testing)

![05_hypothesis_tests.png](file:///D:/Antigravity/projects/project_data_analysis/.worktrees/feat-ds-rebuild/artifacts/eda/05_hypothesis_tests.png)

Để xác nhận xem các mối quan hệ quan sát được trong EDA có phải là ngẫu nhiên hay phản ánh quy luật cấu trúc thực tế, chúng tôi tiến hành 4 kiểm định thống kê chính thức:

#### Kiểm định 1: Ảnh hưởng của việc phân bổ Đa phòng ban (`dept_count > 1`) đến `days_to_pay`
- **Lý do chọn kiểm định Mann-Whitney U**: Do biến `days_to_pay` lệch phải nặng nề, kiểm định Student's t-test (vốn giả định phân phối chuẩn) sẽ cho kết quả sai lệch nghiêm trọng. Mann-Whitney U là kiểm định phi tham số so sánh thứ hạng (rank sums) giữa hai tổng thể độc lập.
- **Dữ liệu**:
  - Nhóm đa phòng ban ($n_1 = 1,885$ hóa đơn): Median = 29.0 ngày, Mean = 33.59 ngày.
  - Nhóm đơn phòng ban ($n_2 = 810,029$ hóa đơn): Median = 30.0 ngày, Mean = 37.38 ngày.
- **Kết quả kiểm định**:
  - Thống kê kiểm định $U = 664,699,538.00$.
  - Giá trị $z = -9.7158$, $p\text{-value} = 4.29 \times 10^{-23} < 0.001$.
  - Kích thước hiệu ứng (Effect Size): $r = \frac{|z|}{\sqrt{n_1 + n_2}} = \frac{9.7158}{\sqrt{811,914}} = 0.0108$.
- **Kết luận**: Bác bỏ giả thuyết vô hiệu $H_0$. Hóa đơn liên quan đến nhiều phòng ban được xử lý nhanh hơn một chút so với hóa đơn đơn phòng ban một cách có ý nghĩa thống kê, mặc dù kích thước hiệu ứng thực tế là nhỏ ($r \approx 0.01$). Điều này phản ánh thực tế rằng các hóa đơn liên phòng ban thường là các hợp đồng lớn được theo dõi chặt chẽ bởi lãnh đạo thành phố.

#### Kiểm định 2: Hóa đơn giá trị cao ($> \$1,700$) vs Hóa đơn bình thường
- **Dữ liệu**: Ngưỡng phân vị 75 ($p75 = \$1,700.00$).
  - Nhóm giá trị cao ($n_1 = 202,860$ hóa đơn): Mean = 36.80 ngày.
  - Nhóm giá trị thường ($n_2 = 609,054$ hóa đơn): Mean = 37.57 ngày.
- **Kết quả kiểm định**:
  - Thống kê kiểm định $U = 57,883,994,058.50$.
  - Giá trị $z = -42.5717$, $p\text{-value} < 10^{-100}$.
  - Kích thước hiệu ứng: $r = 0.0472$.
- **Kết luận**: Bác bỏ giả thuyết vô hiệu $H_0$. Thứ hạng thanh toán giữa hai nhóm giá trị khác biệt có ý nghĩa thống kê cao ($p < 0.001$).

#### Kiểm định 3: Tác động của Tháng 6 Chốt Sổ Tài Khóa lên Tỷ lệ Trễ Hạn
- **Phương pháp**: Bảng tiếp liên $2 \times 2$ (Tháng 6 vs Tháng khác $\times$ Đúng hạn vs Trễ hạn). Kiểm định Chi-Square của Pearson.
- **Dữ liệu**:
  - Tháng 6: 47,043 đúng hạn, 13,405 trễ hạn $\to$ Tỷ lệ trễ hạn = **22.18%**.
  - Các tháng khác: 563,845 đúng hạn, 187,621 trễ hạn $\to$ Tỷ lệ trễ hạn = **24.97%**.
- **Kết quả kiểm định**:
  - Chi-Square thống kê: $\chi^2 = 233.83, df = 1$.
  - Giá trị $p\text{-value} = 8.70 \times 10^{-53} < 0.001$.
  - Kích thước hiệu ứng Cramér's $V = \sqrt{\frac{\chi^2}{N \cdot (k-1)}} = \sqrt{\frac{233.83}{811,914}} = 0.0170$.
- **Kết luận**: Bác bỏ $H_0$. Tháng 6 có tỷ lệ trễ hạn thấp hơn các tháng khác trong năm một cách rõ rệt nhờ cơ chế ưu tiên tất toán năm tài khóa.

#### Kiểm định 4: Chênh lệch Rủi ro giữa các Đơn vị Vận hành (`operating_unit`)
- **Phương pháp**: Bảng tiếp liên $2 \times 2$ giữa `METRO_GG` (Chính quyền tổng hợp) và `METRO_MNPS` (Hệ thống trường công lập).
- **Dữ liệu**:
  - `METRO_GG`: 356,698 đúng hạn, 87,261 trễ hạn $\to$ Tỷ lệ trễ hạn = **19.66%**.
  - `METRO_MNPS`: 254,190 đúng hạn, 113,765 trễ hạn $\to$ Tỷ lệ trễ hạn = **30.92%**.
  - Độ chênh lệch: $\Delta = +11.26\%$ nguy cơ trễ hạn tại hệ thống giáo dục!
- **Kết quả kiểm định**:
  - Chi-Square thống kê: $\chi^2 = 13,700.04, df = 1$.
  - Giá trị $p\text{-value} < 10^{-100}$.
  - Kích thước hiệu ứng Cramér's $V = \sqrt{\frac{13700.04}{811,914}} = 0.1299$ (**Kích thước hiệu ứng đáng kể - Moderate Effect Size**).
- **Ý nghĩa thực tiễn**: Đây là một trong những đặc trưng đơn lẻ có sức mạnh phân loại mạnh nhất trong toàn bộ tập dữ liệu, khẳng định tầm quan trọng của việc đưa biến `operating_unit` vào làm đặc trưng cốt lõi trong mô hình Machine Learning.

---

### 3.6. Phân Tích Pareto & Mức Độ Tập Trung Của Nhà Cung Cấp

![06_vendor_pareto.png](file:///D:/Antigravity/projects/project_data_analysis/.worktrees/feat-ds-rebuild/artifacts/eda/06_vendor_pareto.png)

Định luật Pareto kinh điển dự báo rằng 80% kết quả xuất phát từ 20% nguyên nhân. Trong danh mục chi tiêu AP của Metro Nashville:
1. **Siêu tập trung ngân sách (Super-Pareto Spend)**:
   - Tổng danh mục gồm **19,688 nhà cung cấp độc lập**, thực hiện tổng chi tiêu **$12,944,954,204.07** (~$12.94 tỷ USD).
   - Top 10 nhà cung cấp lớn nhất nắm giữ **$5,199,578,169.30**, chiếm tới **40.17%** tổng ngân sách chi tiêu của toàn thành phố!
   - Top 20% nhà cung cấp (3,938 nhà thầu) chiếm tới **99.73%** tổng giá trị thanh toán ($12.91 tỷ USD). 80% nhà cung cấp còn lại chỉ chia sẻ chưa đầy 0.3% tổng ngân sách.
2. **Siêu tập trung rủi ro trễ hạn (Late Invoices Concentration)**:
   - Top 20% nhà cung cấp cũng là nguồn phát sinh của **96.78%** tổng số hóa đơn bị thanh toán trễ (194,553 trên tổng số 201,026 hóa đơn chậm trả).
   - **Hàm ý cho Giai đoạn Xây dựng Đặc trưng (Task 5)**:
     - Mức độ tập trung nhà cung cấp cực cao đồng nghĩa với việc lịch sử thanh toán trong quá khứ của nhà cung cấp (`vendor_historical_late_rate`, `vendor_total_invoices_paid_prior`) sẽ là những biến số có năng lực giải thích vượt trội.
     - Đồng thời, cần áp dụng cơ chế Out-of-Fold Target Encoding hoặc tính toán lịch sử theo trục thời gian đóng (Point-in-time calculation) để tránh hiện tượng rò rỉ dữ liệu (Data Leakage).

---

## 4. Danh Mục Đồ Thị Đã Xuất Bản & Tệp Đính Kèm

Toàn bộ biểu đồ đã được kết xuất ở độ phân giải cao (300 DPI) và lưu trữ vĩnh viễn tại thư mục artifacts dự án:

1. `artifacts/eda/01_data_audit_funnel.png`: Biểu đồ thanh ngang biểu diễn 5 bước lọc phễu từ 1.3M dòng thô xuống 811k hóa đơn sạch.
2. `artifacts/eda/02_distributions_skew.png`: Biểu đồ kép thể hiện phân phối lệch phải của `days_to_pay` (đỉnh Net 30) và phân phối logarit của `invoice_amount`.
3. `artifacts/eda/03_right_censoring_verification.png`: Đồ thị trục kép chuỗi thời gian 2025 chứng minh sự sụp đổ giả tạo trong tháng 11-12/2025 và vị trí vùng đệm 61 ngày kết thúc tại 31/10/2025.
4. `artifacts/eda/04_threshold_and_seasonality.png`: Đường cong độ nhạy ngưỡng thanh toán và biểu đồ mùa vụ 12 tháng làm nổi bật đỉnh tháng 5 và đáy giải ngân tháng 6.
5. `artifacts/eda/05_hypothesis_tests.png`: Biểu đồ so sánh tỷ lệ trễ hạn giữa `METRO_GG` vs `METRO_MNPS` và hàm phân phối tích lũy (CDF) giữa hóa đơn đa phòng ban vs đơn phòng ban.
6. `artifacts/eda/06_vendor_pareto.png`: Đường cong Lorenz biểu diễn mức độ siêu tập trung của chi tiêu (99.73%) và rủi ro trễ hạn (96.78%) trong nhóm 20% nhà thầu hàng đầu.

---

## 5. Kết Luận & Chuyển Giao Cho Giai Đoạn Tiếp Theo

1. **Kiến trúc dữ liệu đã được xác nhận thực tế**: Tập dữ liệu 811,914 dòng tại `data/interim/invoices_clean.parquet` đạt độ sạch 100%, không còn bản ghi âm, không còn lỗi ngày tháng bất thường.
2. **Khung thời gian phân chia mô hình được chứng minh bằng toán học**:
   - `Train`: `2022-07-01` đến `2024-06-30` (2 năm tài khóa hoàn chỉnh).
   - `Validation`: `2024-07-01` đến `2024-12-31` (Nửa đầu năm tài khóa 2025).
   - `Test`: `2025-01-01` đến `2025-10-31` (10 tháng, bảo toàn vùng đệm 61 ngày chống cắt phải).
3. **Sẵn sàng bước vào Task 5**: Tiến hành trích xuất đặc trưng lịch sử nhà cung cấp (Vendor Historical Rolling Features), đặc trưng lịch sử phòng ban (Department Risk), và đặc trưng lịch thời gian (Calendar & Fiscal Features) với cơ chế bảo vệ nghiêm ngặt chống rò rỉ thông tin tương lai.
