"""
Lớp kiểm thử dữ liệu (Data Contract) — chèn giữa 2 file đã có:

    paddleocr_qwen_extraction.py  --(df_ocr thô, có thể bẩn)-->  [FILE NÀY]  --(df sạch)-->  ap_analysis.analyze()

Lý do PHẢI đặt ở vị trí này, không phải nơi khác:
- Validate ngược vào df_clean của ap_analysis.py là vô nghĩa: dữ liệu đó sinh ra bằng code,
  chắc chắn sạch theo định nghĩa -> schema sẽ luôn pass, không chứng minh được gì.
- Điểm duy nhất dữ liệu thực sự "bẩn" (do OCR đọc sai số, Qwen suy luận thiếu, ảnh mờ...)
  là ngay sau bước trích xuất. Đây mới là nơi cần cổng kiểm thử.

6 lỗ hổng đã sửa so với bản gốc:
1. Sai tên cột: schema validate "Total_Amount" nhưng pipeline thực tế trả về Gross_Amount /
   Net_Amount / Credit_Note_Amount -> validate() luôn raise ngay ở dòng đầu, không chạy nổi.
2. Thiếu lazy=True: .validate() mặc định fail-fast (dừng ở lỗi ĐẦU TIÊN) và raise
   pandera.errors.SchemaError (số ít) — nhưng code bắt except SchemaErrors (số nhiều, dạng
   tổng hợp). Hai kiểu exception khác nhau -> except không bắt được gì, chương trình crash.
3. Payment_Terms / Due_Date bị nullable=False cứng: OCR/Qwen có thể không đọc được các
   trường này (đã thiết kế để trả None thay vì đoán bừa ở clean_extracted_record) — schema
   gốc sẽ chặn cứng toàn bộ các hóa đơn hợp lệ nhưng thiếu 1 trường phụ.
4. Không có ràng buộc Invoice_ID duy nhất — trùng lặp (đã nêu ở review trước) không được
   pandera bắt, dồn hết trách nhiệm sang detect thủ công.
5. Không kiểm tra quan hệ Credit_Note_Amount <= Gross_Amount và Net_Amount phải khớp phép trừ
   — dữ liệu điều chỉnh sai vẫn lọt qua.
6. Check due_date >= invoice_date không loại trừ NaT trước khi so sánh -> với Due_Date nullable,
   phép so sánh với NaT sẽ luôn False và làm rớt oan những dòng thiếu Due_Date hợp lệ.
"""

from __future__ import annotations

import pandas as pd
import pandera.pandas as pa
from pandera.pandas import Check, Column, DataFrameSchema

invoice_schema = DataFrameSchema(
    {
        "Invoice_ID": Column(str, nullable=False, unique=True),
        "Vendor_Name": Column(str, nullable=False),
        "Invoice_Date": Column(pa.DateTime, nullable=False),
        "Payment_Terms": Column(
            str, Check.isin(["Net 30", "Net 60", "2/10 Net 30"]), nullable=True
        ),
        "Due_Date": Column(pa.DateTime, nullable=True),
        "Gross_Amount": Column(
            float, Check(lambda s: s > 0, error="Gross_Amount phải > 0"), nullable=False
        ),
        "Credit_Note_Amount": Column(
            float, Check(lambda s: s >= 0, error="Credit_Note_Amount không được âm"), nullable=False
        ),
        "Net_Amount": Column(float, Check(lambda s: s >= 0, error="Net_Amount không được âm"), nullable=False),
    },
    checks=[
        Check(
            lambda df: df["Due_Date"].isna() | (df["Due_Date"] >= df["Invoice_Date"]),
            name="check_due_date_after_invoice_date",
            error="Ngày đến hạn nhỏ hơn ngày phát hành hóa đơn",
        ),
        Check(
            lambda df: df["Credit_Note_Amount"] <= df["Gross_Amount"],
            name="check_credit_note_not_exceed_gross",
            error="Số tiền giảm trừ lớn hơn tổng tiền hóa đơn gốc",
        ),
        Check(
            lambda df: (df["Gross_Amount"] - df["Credit_Note_Amount"] - df["Net_Amount"]).abs() < 1,
            name="check_net_amount_consistency",
            error="Net_Amount không khớp Gross_Amount - Credit_Note_Amount",
        ),
    ],
    strict=False,  # cho phép các cột nghiệp vụ khác (Payments, Date_Anomaly...) đi kèm, không chặn
)


def validate_extracted_data(df_invoices: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Trả về (df_hop_le, df_loi). Không raise ra ngoài — batch OCR luôn có khả năng lẫn
    dòng lỗi, nhiệm vụ của hàm này là TÁCH chứ không phải chặn đứng cả pipeline."""
    try:
        invoice_schema.validate(df_invoices, lazy=True)
        print("✅ Kiểm thử thành công: toàn bộ dữ liệu sạch.")
        return df_invoices, pd.DataFrame()
    except pa.errors.SchemaErrors as err:
        failure_cases = err.failure_cases
        bad_indices = set(failure_cases["index"].dropna().astype(int))
        df_bad = df_invoices.loc[df_invoices.index.isin(bad_indices)]
        df_good = df_invoices.loc[~df_invoices.index.isin(bad_indices)]
        print(f"❌ {len(df_bad)}/{len(df_invoices)} hóa đơn lỗi — đẩy sang hàng chờ kế toán soát lại:")
        print(failure_cases[["column", "check", "failure_case"]].to_string(index=False))
        return df_good, df_bad


# =============================================================
# DEMO DEPLOYMENT: OCR (giả lập) -> cổng validate -> báo cáo công nợ
# =============================================================
if __name__ == "__main__":
    from ap_analysis import analyze

    # Giả lập output thô của paddleocr_qwen_extraction.py: trộn hóa đơn sạch + các lỗi
    # điển hình thực tế của OCR/LLM (đọc sai số âm, suy luận sai ngày, trùng mã, thiếu field)
    df_ocr_raw = pd.DataFrame(
        [
            dict(Invoice_ID="OCR-001", Vendor_Name="Công ty CP Thép Hòa An",
                 Invoice_Date=pd.Timestamp("2026-02-01"), Payment_Terms="Net 30",
                 Due_Date=pd.Timestamp("2026-03-03"), Gross_Amount=120_000_000.0,
                 Credit_Note_Amount=0.0, Net_Amount=120_000_000.0, Payments=[]),
            dict(Invoice_ID="OCR-002", Vendor_Name="Bao bì Toàn Cầu",
                 Invoice_Date=pd.Timestamp("2026-02-10"), Payment_Terms=None,  # Qwen không đọc được
                 Due_Date=pd.NaT, Gross_Amount=45_000_000.0,
                 Credit_Note_Amount=0.0, Net_Amount=45_000_000.0, Payments=[]),
            dict(Invoice_ID="OCR-003", Vendor_Name="Hóa chất Việt Đức",
                 Invoice_Date=pd.Timestamp("2026-02-15"), Payment_Terms="Net 30",
                 Due_Date=pd.Timestamp("2026-01-01"),  # PaddleOCR đọc nhầm ngày -> due < invoice
                 Gross_Amount=80_000_000.0, Credit_Note_Amount=0.0, Net_Amount=80_000_000.0, Payments=[]),
            dict(Invoice_ID="OCR-001", Vendor_Name="Công ty CP Thép Hòa An",  # mã trùng
                 Invoice_Date=pd.Timestamp("2026-02-01"), Payment_Terms="Net 30",
                 Due_Date=pd.Timestamp("2026-03-03"), Gross_Amount=120_000_000.0,
                 Credit_Note_Amount=0.0, Net_Amount=120_000_000.0, Payments=[]),
            dict(Invoice_ID="OCR-005", Vendor_Name="Điện máy Công nghiệp Phúc Hưng",
                 Invoice_Date=pd.Timestamp("2026-02-20"), Payment_Terms="Net 60",
                 Due_Date=pd.Timestamp("2026-04-21"), Gross_Amount=-15_000_000.0,  # OCR đọc nhầm dấu trừ
                 Credit_Note_Amount=0.0, Net_Amount=-15_000_000.0, Payments=[]),
        ]
    )

    df_good, df_bad = validate_extracted_data(df_ocr_raw)
    print(f"\n-> {len(df_good)} hóa đơn được nạp vào báo cáo, {len(df_bad)} hóa đơn bị giữ lại.\n")

    if not df_good.empty:
        result = analyze(df_good, as_of_date="2026-04-15")
        print(f"DPO (dollar-weighted) trên batch OCR sạch: {result['dpo_dollar_weighted']}")
