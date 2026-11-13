# Huấn Luyện GBDT Trên Kaggle, Hiệu Chuẩn Xác Suất & Đánh Giá Kép (Dual-Track Modeling & Evaluation)

Tài liệu này tổng hợp toàn bộ kết quả thực nghiệm huấn luyện **LightGBM** và **XGBoost** trên Kaggle (`bangta205/nashville-ap-dual-track-gbdt`), đối chiếu trực tiếp với hai tầng mốc chuẩn cục bộ (**Dummy/Median** và **Linear Logistic/Ridge** từ `artifacts/baseline_metrics.json`), kèm giải thích SHAP và phân tích sai số thực tế trên **195,889** hóa đơn kiểm thử (`artifacts/kaggle_results/test_predictions.parquet`).

---

## 1. Tóm Tắt Điều Hành (Executive Summary - Minto Pyramid)

- **Kết luận then chốt 1: Mô hình GBDT (đặc biệt XGBoost) vượt trội toàn diện mô hình tuyến tính ở cả hai bài toán xếp hạng rủi ro trễ hạn và ước lượng số ngày thanh toán.**
  - Trên tập Test hold-out (`2025-01-01` .. `2025-10-31`, `n = 195,889`, tỷ lệ trễ nền `20.76%`), **XGBoost Calibrated** đạt **PR-AUC = `0.4092`** (+37.1% tương đối so với Logistic `0.2984`, gần gấp đôi Dummy `0.2076`) và **ROC-AUC = `0.7252`** (so với Logistic `0.5853`).
  - Ở ngưỡng rà soát **Top 5% (9,794 hóa đơn)**, XGBoost đạt độ chính xác **58.81%** (`Lift@5% = 2.83x`); ở **Top 20% (39,178 hóa đơn)** với tỷ lệ chi phí phạt/kiểm tra `5:1`, điểm lợi ích ròng `net_benefit_score` đạt **43,292** (`ROI = 2.11x`) so với **23,622** (`ROI = 1.60x`) của Logistic (+83.3% giá trị ròng).
- **Kết luận then chốt 2: Biến đổi mục tiêu `log1p(days_to_pay)` kết hợp `expm1` giúp GBDT giảm hơn 20% sai số MAE và đạt MAPE kỳ hạn 14 ngày chỉ `0.65%`, nhưng cần cẩn trọng với hiệu ứng ngưỡng cứng Net-30.**
  - **XGBoost Regressor** đạt **MAE = `8.73` ngày** (giảm 20.4% so với Ridge `10.97` ngày, giảm 9.9% so với Median `9.69` ngày), **RMSE = `18.51` ngày** và **MedianAE = `3.97` ngày** (so với Ridge `6.64` ngày).
  - Dự báo dòng tiền kỳ hạn 14 ngày của XGBoost đạt **MAPE = `0.65%`** (`$1,769.56M` dự báo vs `$1,781.07M` thực chi). Tuy nhiên, tại kỳ hạn 30 ngày, dự báo trung vị của XGBoost nằm quanh `30.81` ngày khiến **75,498 hóa đơn** trả đúng ngày 30 (tổng giá trị **`$560.76M`**) bị đẩy sang phía `> 30` ngày, làm MAPE 30 ngày tăng lên `11.94%` so với `6.65%` của Ridge.
- **Kết luận then chốt 3: Tín hiệu hành vi lịch sử nhà cung cấp và phòng ban chi phối 75% sức mạnh dự báo, đồng thời bộc lộ hai điểm cần hiệu chỉnh ở vòng lặp tiếp theo.**
  - 5 đặc trưng đứng đầu theo giá trị SHAP tuyệt đối trung bình (`mean |SHAP|`) đều là đặc trưng Point-in-Time lịch sử và lịch tài khóa: `vendor_prior_late_rate_smoothed` (`0.4294`), `dept_prior_late_rate` (`0.2836`), `vendor_prior_paid_count` (`0.2153`), `vendor_prior_avg_days_90d` (`0.1782`), và `days_to_fiscal_year_end` (`0.1172`).
  - Phân tích sai số chỉ ra 3 vùng rủi ro thực tế: (1) nhóm **cold-start** (`vendor_prior_paid_count == 0`, 3,858 hóa đơn Test) có `late_rate = 29.47%` và MAE = `16.37` ngày (gần gấp đôi nhóm có lịch sử `8.57` ngày); (2) phòng ban **Planning Commission** có MAE = `24.11` ngày; và (3) hiệu chuẩn **Isotonic trên tập Val** giúp giảm Brier trên Val (`0.1452 -> 0.1405`) nhưng hơi mất hiệu quả ở đuôi trên tập Test do dịch chuyển tỷ lệ nền (`19.09% -> 20.76%`).

---

## 2. Tại Sao Chọn GBDT (LightGBM & XGBoost) Cho Dữ Liệu AP Dạng Bảng?

Bài toán Accounts Payable trên 811,914 hóa đơn có 4 đặc tính cấu trúc khiến mô hình tuyến tính (`LogisticRegression` / `Ridge`) gặp trần hiệu năng:

1. **Tương tác phi tuyến bậc cao (Non-linear Interactions):** Một nhà cung cấp mới (`vendor_prior_paid_count = 0..3`) có `vendor_prior_late_rate_smoothed = 0.25` mang mức độ bất định hoàn toàn khác với nhà cung cấp lâu năm (`vendor_prior_paid_count = 5,000`) có cùng tỷ lệ `0.25`. Cây quyết định (LightGBM leaf-wise / XGBoost depth-wise) tách tự nhiên theo cặp `(vendor_prior_paid_count, vendor_prior_late_rate_smoothed)` mà mô hình tuyến tính không thể biểu diễn nếu không nhân tay hàng trăm cặp biến.
2. **Mùa vụ không đơn điệu (`days_to_fiscal_year_end`, `inv_dayofweek`):** Áp lực giải ngân trước ngày khóa sổ 30/06 (`days_to_fiscal_year_end -> 0`) tạo đỉnh nghẽn tháng 5 và xả thanh toán cuối tháng 6; mối quan hệ dạng chữ U hoặc răng cưa theo tuần/tháng không thể khớp bằng một hệ số tuyến tính đơn lẻ.
3. **Bất biến với thang đo và phân phối lệch cực độ (Heavy-tailed Skewness):** Các biến như `vendor_prior_paid_count` (0 đến 37,782), `same_day_vendor_inv_count` (1 đến 399), hay `amount_vs_vendor_median_ratio` có độ lệch rất lớn. Cây GBDT chia ngưỡng theo phân vị (histogram binning `tree_method="hist"`) nên miễn nhiễm với outlier mà không phụ thuộc `StandardScaler`.
4. **Xử lý mã phân loại tần suất cao gọn nhẹ:** Với `OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)`, GBDT khai thác trực tiếp `primary_department` và `operating_unit` mà không bùng nổ chiều như One-Hot Encoding.

---

## 3. Cấu Hình Quét Tham Số & Điểm Dừng Sớm (Validation Sweep)

Toàn bộ quá trình huấn luyện được thực thi trên Kaggle (`notebooks/02_kaggle_training.ipynb`) với tập dữ liệu `bangta205/metro-nashville-ap-processed` (`train = 456,414`, `val = 115,974`, `test = 195,889`), `n_estimators = 1,500`, `early_stopping_rounds = 50`, `subsample = 0.8`, `colsample_bytree = 0.8`.

Tiêu chí chọn mô hình trên Validation:
- **Track 1 (Classification):** Tối đa hóa **Val PR-AUC** (phù hợp với lớp thiểu số `is_late ~ 19-21%`), sau đó khớp `CalibratedClassifierCV(..., method="isotonic")` trên `val` (`FrozenEstimator`).
- **Track 2 (Regression):** Tối thiểu hóa **Val log-RMSE** trên mục tiêu $\ln(1 + \text{days\_to\_pay})$.

| Mô hình & Track | Tham số tốt nhất (`best_params`) | Số cây dừng sớm (`best_iteration`) | Điểm Validation Sweep |
| :--- | :--- | :---: | :--- |
| **LightGBM Clf** | `num_leaves=127, lr=0.03, min_child_samples=100, reg_lambda=10.0` | **275** | Val log-sweep PR-AUC = `0.3524` (sau isotonic = `0.3485`) |
| **XGBoost Clf** | `max_depth=6, lr=0.05, min_child_weight=5, reg_lambda=1.0` | **287** | Val log-sweep PR-AUC = `0.3508` (sau isotonic = `0.3465`) |
| **LightGBM Reg** | `num_leaves=127, lr=0.03, min_child_samples=100, reg_lambda=10.0` | **258** | Val `log-RMSE` = `0.4952` |
| **XGBoost Reg** | `max_depth=8, lr=0.03, min_child_weight=20, reg_lambda=5.0` | **215** | Val `log-RMSE` = `0.4940` |

> [!NOTE]
> Cả 4 mô hình đều hội tụ sớm quanh **215–287 vòng lặp** (trên trần 1,500), chứng tỏ tốc độ học `0.03–0.05` kết hợp điều chuẩn `reg_lambda = 1.0–10.0` kiểm soát tốt hiện tượng quá khớp (overfitting) trên tập huấn luyện 456k dòng.

---

## 4. Track 1 — So Sánh Dự Báo Trễ Hạn (`is_late`) & Hiệu Chuẩn Xác Suất

### 4.1. Bảng Đối Chiếu Toàn Diện Trên Tập Validation & Test

Tỷ lệ trễ hạn thực tế (`base_rate`): **Train = `27.53%`**, **Val = `19.09%`**, **Test = `20.76%`** (tổng giá trị hóa đơn trễ hạn trên Test: **`$450,435,161.14`** / 40,665 hóa đơn).

| Tập đánh giá | Mô hình | PR-AUC | ROC-AUC | Brier (Cal / Raw) | Lift@5% | Lift@10% | Lift@20% | Precision@20% | Value Captured@20% |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Val** | Dummy (Train Prior `0.2753`) | `0.1909` | `0.5000` | `0.1616` | — | `0.97x` | — | — | `18.39%` |
| **Val** | LogisticRegression | `0.3182` | `0.6557` | `0.1501` | `2.37x` | `2.10x` | `1.71x` | `32.73%` | **`50.67%`** (`$133.94M`) |
| **Val** | LightGBM (Isotonic) | **`0.3485`** | **`0.7179`** | **`0.1398`** / `0.1442` | `2.35x` | **`2.23x`** | `1.97x` | `37.67%` | `43.75%` (`$115.66M`) |
| **Val** | XGBoost (Isotonic) | `0.3465` | `0.7110` | `0.1405` / `0.1452` | **`2.46x`** | `2.22x` | **`2.01x`** | **`38.33%`** | `42.44%` (`$112.18M`) |
| **Test** | Dummy (Train Prior `0.2753`) | `0.2076` | `0.5000` | `0.1691` | — | `1.00x` | — | — | `21.50%` |
| **Test** | LogisticRegression | `0.2984` | `0.5853` | `0.1701` | `2.16x` | `1.81x` | `1.54x` | `32.06%` | **`40.38%`** (`$181.89M`) |
| **Test** | LightGBM (Isotonic) | `0.3896` | `0.7131` | `0.1487` / `0.1484` | `2.64x` | `2.39x` | `1.94x` | `40.37%` | `42.94%` (`$193.43M`) |
| **Test** | **XGBoost (Isotonic)** | **`0.4092`** | **`0.7252`** | `0.1469` / **`0.1459`** | **`2.83x`** | **`2.43x`** | **`2.03x`** | **`42.10%`** | `39.99%` (`$180.13M`) |

### 4.2. Phân Tích Chuyên Sâu: Hiệu Chuẩn, Độ Bền Thời Gian & Tác Động Nghiệp Vụ

1. **Bước nhảy sức bền Out-of-Time của GBDT so với LogisticRegression:**
   - Khi chuyển từ Val (H2 2024: tháng 7–12) sang Test (2025: tháng 1–10), mô hình tuyến tính Logistic bị suy giảm mạnh (`ROC-AUC` tụt từ `0.6557` xuống `0.5853`, `PR-AUC` tụt từ `0.3182` xuống `0.2984`).
   - Ngược lại, **XGBoost tăng từ `0.7110 -> 0.7252` ROC-AUC và `0.3465 -> 0.4092` PR-AUC** (LightGBM cũng giữ vững ROC-AUC `0.7131` và đạt PR-AUC `0.3896`). Nguyên nhân là tập Test có tỷ lệ nhà cung cấp mới thấp hơn (`1.97%` vs `2.17%` trên Val) và chiều sâu lịch sử tích lũy dài hơn, giúp cây GBDT tận dụng triệt để các biến hành vi nhà cung cấp mà không bị nhiễu bởi sự dịch chuyển mùa vụ giữa H2 và H1.
2. **Nghịch lý Calibration (`val_calibrated_in_sample = true`):**
   - Trên Val, hiệu chuẩn Isotonic làm giảm Brier Score thêm `+0.0044` cho LightGBM (`0.1442 -> 0.1398`) và `+0.0047` cho XGBoost (`0.1452 -> 0.1405`).
   - Tuy nhiên trên Test, xác suất thô (raw) của GBDT lại có Brier Score tốt hơn một chút so với sau Isotonic: LightGBM `0.1484` raw vs `0.1487` cal (`-0.0003`), XGBoost `0.1459` raw vs `0.1469` cal (`-0.0010`).
   - **Giải thích định lượng trên bảng 10 phân vị của `test_predictions.parquet`:** Vì tập Val (19.09% trễ) thấp hơn tập Test (20.76% trễ) và Isotonic được khớp trực tiếp trên chính `val` (vốn đã dùng cho early stopping), hàm bậc thang Isotonic bị nén bảo thủ ở phần đuôi cao: ở nhóm decile cao nhất của `xgb_prob_late` `(0.361, 0.629]` (`n = 15,596`), xác suất dự báo trung bình là **`0.4356`** trong khi tỷ lệ trễ thực tế lên tới **`0.5399`** (lệch `-10.43%`). Ngược lại, trong 5 nhóm phân vị đầu (`0.000..0.176`), độ lệch hiệu chuẩn chỉ dao động `0.10%` đến `1.97%`.
3. **Tại sao Logistic bắt được `40.38%` giá trị `$` tại Top 20% trên Test so với `39.99%` của XGBoost dù PR-AUC kém xa?**
   - Kiểm tra trực tiếp trên `test_predictions.parquet`: hóa đơn trung bình toàn tập Test là **`$17,639.06`**, nhưng nhóm Top 20% của XGBoost có giá trị trung bình **`$9,813.31`** (trung vị `$450.00` vs `$386.67`). Các siêu hóa đơn hàng triệu USD thường nằm ở `Administrative` (tổng `$994.9M`, tỷ lệ trễ chỉ `8.48%`) hoặc `Juvenile Court`. Mô hình không gán trọng số tiền (`sample_weight`) tối ưu hóa **số lượng hóa đơn trễ** (giúp XGBoost bắt được **16,494** ca trễ thực tế tại Top 20% so với **12,560** của Logistic, tức nhiều hơn **+3,934 hóa đơn trễ**!). Ở lát cắt hẹp hơn (**Top 10%**), XGBoost vẫn vượt hẳn về giá trị tiền giữ lại: **`23.03%` (`$103.75M`)** vs Logistic `17.86%` (`$80.44M`) và LightGBM `16.34%` (`$73.60M`).
4. **Bảng quét độ nhạy tác động nghiệp vụ (Test, `business_impact_sensitivity`):**

| Dung lượng rà soát (`K`) | Tỷ lệ chi phí Phạt/Kiểm tra | Logistic (`TP` / `Net Benefit` / `ROI`) | LightGBM (`TP` / `Net Benefit` / `ROI`) | XGBoost (`TP` / `Net Benefit` / `ROI`) |
| :---: | :---: | :---: | :---: | :---: |
| **5%** (`9,794`) | `5 : 1` | `4,391` / `12,161` / `2.24x` | `5,376` / `17,086` / `2.74x` | **`5,760` / `19,006` / `2.94x`** |
| **10%** (`19,589`) | `5 : 1` | `7,346` / `17,141` / `1.88x` | `9,725` / `29,036` / `2.48x` | **`9,882` / `29,821` / `2.52x`** |
| **20%** (`39,178`) | `5 : 1` | `12,560` / `23,622` / `1.60x` | `15,818` / `39,912` / `2.02x` | **`16,494` / `43,292` / `2.11x`** |
| **20%** (`39,178`) | `10 : 1` | `12,560` / `86,422` / `3.21x` | `15,818` / `119,002` / `4.04x` | **`16,494` / `125,762` / `4.21x`** |

### 4.3. Giải Thích Mô Hình Qua SHAP (`shap_top_features`, 1,000 mẫu Test)

Thứ hạng top 10 đặc trưng có ảnh hưởng lớn nhất lên log-odds dự báo trễ hạn của LightGBM:

1. `vendor_prior_late_rate_smoothed`: `mean |SHAP| = 0.4294` — chỉ báo mạnh nhất về thói quen hoặc nút thắt quy trình của nhà cung cấp.
2. `dept_prior_late_rate`: `0.2836` — đặc thù tốc độ phê duyệt và đối soát chứng từ của từng phòng ban.
3. `vendor_prior_paid_count`: `0.2153` — tách biệt nhà cung cấp quen (hồ sơ chuẩn hóa) với nhóm mới/ít giao dịch.
4. `vendor_prior_avg_days_90d`: `0.1782` — xung lượng (momentum) chậm trễ gần nhất trong 90 ngày.
5. `days_to_fiscal_year_end`: `0.1172` — hiệu ứng mùa vụ khóa sổ tài chính ngày 30/06.
6. `vendor_prior_avg_days_365d`: `0.1026` — đường cơ sở tốc độ thanh toán dài hạn 1 năm.
7. `same_day_vendor_inv_count`: `0.0875` — hiệu ứng nộp lô hóa đơn cùng ngày.
8. `primary_department`: `0.0843` — hiệu ứng cố định riêng từng phòng ban chưa phản ánh hết qua tỷ lệ trễ trung bình.
9. `inv_dayofweek`: `0.0786` — chu kỳ chốt lệnh chi trả theo ngày trong tuần.
10. `dept_inv_count_7d`: `0.0772` — tắc nghẽn ngắn hạn (backlog) tại phòng ban trong 7 ngày trước đó.

> [!TIP]
> Nhóm đặc trưng quy mô tiền (`amount_vs_vendor_median_ratio`: `0.0406`, `log_invoice_amount`: `0.0400`) đứng ở hạng 14–15. Điều này khẳng định nguyên nhân trễ hạn của Metro Nashville đến từ **ma sát quy trình phòng ban & lịch sử nhà cung cấp** nhiều hơn là giá trị hóa đơn lớn hay nhỏ.

---

## 5. Track 2 — Dự Báo Số Ngày Thanh Toán (`days_to_pay`) & Dòng Tiền (Cashflow Horizons)

### 5.1. Tại Sao Phải Biến Đổi `log1p(days_to_pay)` và Nghịch Đảo `clip(expm1(y), 0)`?

Phân phối `days_to_pay` có độ lệch phải cực lớn (trung vị 30 ngày, `p90 = 61` ngày, cực đại lên tới hàng nghìn ngày). Nếu huấn luyện hồi quy trực tiếp trên thang ngày thô với hàm mất mát bình phương (MSE/RMSE), chỉ một vài hóa đơn ngoại lai >300 ngày sẽ tạo gradient lớn gấp hàng trăm lần các hóa đơn thông thường, kéo dự báo chung lệch hẳn khỏi vùng Net-30.
- Huấn luyện trên $z = \ln(1 + \text{days\_to\_pay})$ nén đuôi dài về thang gần chuẩn, giúp GBDT tập trung phân biệt chính xác vùng 5–45 ngày.
- Khi dự báo, nghịch đảo qua $\hat{y} = \max(0, \exp(\hat{z}) - 1)$ đảm bảo thời gian thanh toán không bao giờ âm.

### 5.2. Bảng So Sánh Hồi Quy & Dự Báo Dòng Tiền Trên Val và Test

| Tập | Mô hình | MAE (ngày) | RMSE (ngày) | MedianAE (ngày) | Cashflow 7d MAPE | Cashflow 14d MAPE | Cashflow 30d MAPE |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Val** | Median (`30.00` ngày) | `11.72` | `30.09` | **`2.00`** | `100.00%` | `100.00%` | `19.74%` |
| **Val** | Ridge (`alpha=10.0`) | `12.76` | `29.57` | `6.14` | `37.74%` | `31.88%` | **`9.64%`** |
| **Val** | LightGBM Regressor | `11.32` | `28.73` | `4.46` | `25.06%` | **`6.89%`** | `15.30%` |
| **Val** | **XGBoost Regressor** | **`11.27`** | **`28.71`** | `4.30` | **`22.61%`** | `7.93%` | `14.47%` |
| **Test** | Median (`30.00` ngày) | `9.69` | `20.37` | **`2.00`** | `100.00%` | `100.00%` | `14.99%` |
| **Test** | Ridge (`alpha=10.0`) | `10.97` | `20.14` | `6.64` | `46.37%` | `23.67%` | **`6.65%`** |
| **Test** | LightGBM Regressor | `8.75` | `18.58` | **`3.94`** | `38.82%` | `7.06%` | `11.06%` |
| **Test** | **XGBoost Regressor** | **`8.73`** | **`18.51`** | `3.97` | **`29.86%`** | **`0.65%`** | `11.94%` |

*(Lưu ý về MedianAE = `2.00` của Median Baseline: Vì phân phối có đỉnh Dirac rất nhọn tại đúng 30 ngày — 25%–50% hóa đơn trả quanh 28–30 ngày — nên dự báo hằng số `30.00` đạt MedianAE = 2 ngày, nhưng **hoàn toàn vô dụng** cho dòng tiền ngắn hạn khi cho sai số `100.00%` ở khung 7 ngày và 14 ngày do không dự đoán bất kỳ khoản chi nào trước ngày 30!)*

### 5.3. Giải Mã Hiện Tượng "Kỳ Hạn 14d Gần Tuyệt Đối (`0.65%`) Nhưng 30d Kém Hơn Ridge (`11.94%` vs `6.65%`)"

Kiểm tra chi tiết dòng tiền thực tế trên tập Test:
- **7 ngày:** Thực chi `$1,538.72M` — XGBoost dự báo `$1,079.33M` (`MAPE = 29.86%`), vượt xa Ridge (`$825.19M`, `MAPE = 46.37%`).
- **14 ngày:** Thực chi `$1,781.07M` — **XGBoost dự báo `$1,769.56M` (`MAPE = 0.65%`)** và LightGBM dự báo `$1,655.30M` (`MAPE = 7.06%`), bỏ xa Ridge (`$1,359.45M`, `MAPE = 23.67%`).
- **30 ngày:** Thực chi `$3,004.86M` — XGBoost dự báo `$2,646.08M` (`MAPE = 11.94%`), LightGBM `$2,672.52M` (`11.06%`), trong khi Ridge đạt `$2,805.03M` (`6.65%`).

**Nguyên nhân gốc rễ đã được kiểm chứng trên `test_predictions.parquet`:**
- Phân vị 50% (trung vị) của `xgb_pred_days` trên Test là **`30.8051` ngày**.
- Do chính sách Net-30 tạo ra một bức tường dày đặc hóa đơn thanh toán đúng ngày thứ `30`, hàng loạt hóa đơn thực tế trả ở mốc $\le 30$ ngày được GBDT dự báo làm tròn ở mức `31–35` ngày (`mean = 34.49` ngày). Phép đếm ngưỡng cứng `pred_diff <= 30` loại bỏ hoàn toàn **75,498 hóa đơn** loại này với tổng giá trị **`$560,758,125.78`** (`$560.76M`), tạo ra mức thiếu hụt `$358.79M` ở kỳ hạn 30 ngày.
- Trong khi đó, do sử dụng biến đổi `log1p`, giá trị trung bình của $\exp(\mathbb{E}[\ln(1+Y)]) - 1$ về mặt toán học là **trung vị hình học**, luôn nhỏ hơn kỳ vọng số học (Bất đẳng thức Jensen), và bị lệch nhẹ quanh nút cắt 30 ngày khi chuyển thành hàm chỉ báo nhị phân `pred_days <= 30`.

---

## 6. Phân Tích Sai Số Thực Tế (Real Error Analysis trên `test_predictions.parquet`)

### 6.1. Sai Số Theo Phòng Ban (`primary_department`, lọc các phòng ban `n >= 500`)

| Phòng ban (`primary_department`) | Số lượng (`n`) | Tổng tiền (`$`) | Tỷ lệ trễ thực tế | Prob XGB TB | Ngày trả TB thực / dự báo | MAE (ngày) | RMSE (ngày) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Planning Commission** | `593` | `$39.12M` | `33.73%` | `15.69%` | `46.53` / `26.54` | **`24.11`** | `49.04` |
| **Metro Action Commission** | `4,746` | `$15.46M` | `16.48%` | `17.48%` | `20.05` / `16.97` | `13.83` | `26.38` |
| **Water and Sewer** | `12,608` | `$478.59M` | `21.79%` | `20.65%` | `34.28` / `33.94` | `13.27` | `25.12` |
| **Information Technology Service** | `2,169` | `$52.91M` | `31.30%` | `19.20%` | `39.23` / `29.16` | `12.90` | `26.53` |
| **Law** | `1,347` | `$31.77M` | `17.07%` | `11.79%` | `20.30` / `14.43` | `12.55` | `23.09` |
| *(Tham chiếu Top Volume)* **MNPS** | `91,152` | `$615.65M` | `25.42%` | `22.53%` | `33.45` / `32.11` | **`8.08`** | `17.19` |
| *(Tham chiếu Top Volume)* **General Services** | `15,676` | `$88.22M` | `9.63%` | `9.40%` | `27.20` / `25.83` | **`6.17`** | `15.00` |
| *(Tham chiếu Top Volume)* **Administrative** | `7,797` | `$994.93M` | `8.48%` | `7.65%` | `16.13` / `14.14` | **`6.49`** | `12.61` |

- **Nhận xét:** Các phòng ban có khối lượng lớn và quy trình chuẩn hóa như `MNPS` (`8.08` ngày), `General Services` (`6.17` ngày), `Administrative` (`6.49` ngày) được dự báo rất sát. Ngược lại, `Planning Commission` (`33.73%` trễ vs mô hình dự báo `15.69%`, MAE `24.11` ngày) và `Information Technology Service` (`31.30%` trễ vs dự báo `19.20%`) có độ trễ thực tế tăng vọt trong năm 2025 so với giai đoạn Train/Val trước đó (concept drift cục bộ ở phòng ban quy mô vừa/nhỏ).

### 6.2. Nhóm Nhà Cung Cấp Mới (Cold-Start) vs Quen (Warm Vendors)

- **Warm vendors (`vendor_prior_paid_count > 0`, `n = 192,031`):** Tỷ lệ trễ thực tế = `20.58%`, xác suất dự báo TB = `18.60%`, **MAE = `8.57` ngày**.
- **Cold-start vendors (`vendor_prior_paid_count == 0`, `n = 3,858`):** Tỷ lệ trễ thực tế = **`29.47%`**, xác suất dự báo TB = `24.59%`, **MAE = `16.37` ngày** (gấp **1.91 lần** nhóm có lịch sử). Việc làm mịn Laplace về `0.25` phản ánh tốt xu hướng rủi ro cao hơn của cold-start, nhưng độ bất định từng cá nhân vẫn rất lớn.

### 6.3. Các Ca Bỏ Sót Giá Trị Lớn (High-Value Misses) & Ngoại Lai Cực Đại

- Trong **40,665** hóa đơn trễ hạn trên Test (`$450.44M`), có **20,885** hóa đơn được XGBoost gán xác suất `< 0.25` (tổng `$226.54M`).
- 3 hóa đơn giá trị cao bị bỏ sót tiêu biểu:
  1. `Juvenile Court` — Supplier `2272`: **`$11,724,561.37`**, thực tế trả `35` ngày (trễ nhẹ 5 ngày), dự báo `29.83` ngày (`prob = 0.1763`).
  2. `Juvenile Court` — Supplier `2272`: **`$10,428,109.51`**, thực tế trả `36` ngày (trễ 6 ngày), dự báo `29.45` ngày (`prob = 0.1866`).
  3. `Administrative` — Supplier `13191`: **`$6,618,938.00`**, thực tế trả `41` ngày, dự báo `14.18` ngày (`prob = 0.1378`).
- Các sai số hồi quy lớn nhất (`316–326` ngày) đều là các hóa đơn nhỏ (`$147` – `$1,626` tại `Metro Action Commission`, `MNPS`, `State Trial Courts`) bị treo tranh chấp chứng từ tới `315–350` ngày — sự kiện bất thường không thể suy ra từ đặc trưng lịch sử trước khi phát hành hóa đơn.

---

## 7. Hướng Cải Tiến Tiếp Theo & Ghi Nhớ Phỏng Vấn

### Đề xuất cải tiến v2 (Future Roadmap)
1. **Huấn luyện có trọng số giá trị hoặc hàm mục tiêu Risk-Weighted:** Thêm tùy chọn `sample_weight = log1p(invoice_amount)` hoặc xếp hạng theo **Expected Late Value** ($\hat{p}_{\text{late}} \times \text{invoice\_amount}$) ở tầng ra quyết định để tối đa hóa đồng thời Precision và `$` thu hồi cho các ca như `$11.72M` tại `Juvenile Court`.
2. **Tách riêng Calibrator ngoài mẫu (Out-of-fold / Time-holdout Calibration):** Không khớp Isotonic trên chính tập Val dùng cho early stopping; thay vào đó dùng **Platt Scaling (Logistic Calibration)** mượt hơn hoặc chia nửa cuối Val để tránh hiệu ứng bậc thang làm lệch đuôi trên tập Test.
3. **Dự báo xác suất ngưỡng 30 ngày cho Cashflow 30d:** Thay vì chỉ dùng điểm cắt cứng `round(pred_days) <= 30`, kết hợp xác suất `1 - P(is_late)` nhân với `invoice_amount` sẽ giúp dòng tiền 30 ngày không bị hiệu ứng vách đá (cliff effect) quanh mốc 30.8 ngày.

### Ghi Nhớ Phỏng Vấn (Interview Takeaways)
1. **"Tại sao PR-AUC của GBDT tăng trên Test trong khi LogisticRegression lại giảm?"**
   - LogisticRegression phụ thuộc vào các hệ số tuyến tính cố định theo mùa vụ H2 của Val, còn GBDT học được các tương tác phi tuyến của lịch sử nhà cung cấp (vốn dồi dào hơn ở tập Test năm 2025 với tỷ lệ cold-start giảm xuống `1.97%`).
2. **"Tại sao Brier Score sau hiệu chuẩn Isotonic lại tệ hơn một chút trên Test (`0.1469` vs `0.1459`)?"**
   - Do hai yếu tố: (1) tập Val dùng cho Isotonic cũng chính là tập chạy early stopping (`val_calibrated_in_sample`), và (2) base rate dịch từ `19.09%` (Val) lên `20.76%` (Test), khiến hàm bậc thang Isotonic bị trần nhẹ ở phân vị cao nhất (dự báo TB `0.4356` vs thực tế `0.5399`).
3. **"Làm sao giải thích việc XGBoost đạt MAPE 14d = `0.65%` nhưng MAPE 30d = `11.94%` (kém hơn Ridge `6.65%`)?"**
   - Kiểm chứng thực tế trên `test_predictions.parquet` cho thấy trung vị `xgb_pred_days = 30.81` ngày. Do đỉnh Net-30 tập trung dày đặc ở đúng ngày 30, có tới **75,498 hóa đơn** (`$560.76M`) thực trả $\le 30$ ngày nhưng bị dự báo lệch sang `31–35` ngày, rơi khỏi ngưỡng đếm cứng `pred_diff <= 30`.
