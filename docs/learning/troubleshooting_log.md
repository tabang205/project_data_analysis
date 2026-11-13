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
