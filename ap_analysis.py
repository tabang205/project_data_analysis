import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# =============================================================
# 1. SINH DỮ LIỆU MẪU (mô phỏng hệ thống kế toán thực tế hơn)
# =============================================================
np.random.seed(42)
n = 50

vendors = [
    'Công ty CP Thép Hòa An',
    'Nhà cung cấp Nhựa Tân Phát',
    'Bao bì Toàn Cầu',
    'Hóa chất Việt Đức',
    'Điện máy Công nghiệp Phúc Hưng'
]
terms_list = ['Net 30', 'Net 60', '2/10 Net 30']
start_date = datetime(2026, 1, 1)

def generate_invoices(n=50):
    rows = []
    for i in range(1, n + 1):
        vendor = np.random.choice(vendors)
        term = np.random.choice(terms_list, p=[0.4, 0.3, 0.3])
        inv_date = start_date + timedelta(days=int(np.random.randint(0, 90)))
        term_days = 60 if term == 'Net 60' else 30
        due_date = inv_date + timedelta(days=term_days)
        amount = float(np.random.randint(20, 350) * 1_000_000)

        # Ghi chú (2): số hóa đơn gốc/ký hiệu để đối chiếu pháp lý, tách khỏi mã nội bộ
        symbol = f"1C26T{np.random.randint(1,4)}"
        legal_no = f"{np.random.randint(1, 99999):07d}"

        # Ghi chú (3): hóa đơn điều chỉnh/giảm giá (credit note) ~15% số hóa đơn
        credit_note = round(amount * 0.05, 0) if np.random.rand() < 0.15 else 0.0
        net_amount = amount - credit_note

        # Ghi chú (1): thanh toán từng đợt thay vì nhị phân trả hết/chưa trả
        n_installments = np.random.choice([1, 2, 3], p=[0.6, 0.3, 0.1])
        remaining = net_amount
        payments = []
        cursor_date = inv_date
        for k in range(n_installments):
            if np.random.rand() < 0.25 and k == n_installments - 1:
                break  # đợt cuối chưa thanh toán -> hóa đơn còn dư nợ
            pct = 1.0 if n_installments == 1 else np.random.uniform(0.3, 0.6)
            pay_amt = round(min(remaining, net_amount * pct), 0)
            if pay_amt <= 0:
                continue
            gap = max(3, int(np.random.normal(loc=term_days / n_installments, scale=10)))
            cursor_date = cursor_date + timedelta(days=gap)
            dispute = np.random.rand() < 0.08          # (4) tranh chấp -> mất quyền chiết khấu
            wrong_channel = np.random.rand() < 0.05     # (4) sai kênh thỏa thuận -> mất quyền chiết khấu
            payments.append({
                'pay_date': cursor_date,
                'pay_amount': pay_amt,
                'dispute': dispute,
                'wrong_channel': wrong_channel
            })
            remaining -= pay_amt

        rows.append({
            'Invoice_ID': f"INV-2026-{i:03d}",
            'Legal_Invoice_No': legal_no,
            'Invoice_Symbol': symbol,
            'Vendor_Name': vendor,
            'Invoice_Date': inv_date,
            'Payment_Terms': term,
            'Due_Date': due_date,
            'Gross_Amount': amount,
            'Credit_Note_Amount': credit_note,
            'Net_Amount': net_amount,
            'Payments': payments,   # list thay vì 1 giá trị đơn -> hỗ trợ trả nhiều đợt
        })
    return pd.DataFrame(rows)

df = generate_invoices(n)

# =============================================================
# 2. LÀM SẠCH DỮ LIỆU (Data Integrity)
# =============================================================

# (5) Ngày hóa đơn phải <= ngày đến hạn — hóa đơn nào vi phạm bị gắn cờ để loại khỏi báo cáo
df['Date_Anomaly'] = df['Invoice_Date'] > df['Due_Date']

# (5) Phát hiện trùng lặp: cùng NCC + cùng số tiền + hóa đơn cách nhau <=1 ngày
# (thường do quét OCR trùng file) -> gắn cờ, không tự xóa để kế toán xác nhận thủ công
df_sorted = df.sort_values(['Vendor_Name', 'Gross_Amount', 'Invoice_Date'])
df['Possible_Duplicate'] = df_sorted.duplicated(subset=['Vendor_Name', 'Gross_Amount'], keep=False) & \
    (df_sorted.groupby(['Vendor_Name', 'Gross_Amount'])['Invoice_Date']
     .transform(lambda s: (s.max() - s.min()).days <= 1))
df['Possible_Duplicate'] = df['Possible_Duplicate'].reindex(df.index, fill_value=False)

df_clean = df[~df['Date_Anomaly']].copy()

# =============================================================
# 3. PHÂN TÍCH POINT-IN-TIME theo mốc chốt sổ (không hardcode)
# =============================================================

REQUIRED_COLUMNS = [
    "Invoice_ID", "Vendor_Name", "Invoice_Date", "Payment_Terms",
    "Due_Date", "Net_Amount", "Payments",
]


def analyze(df_in, as_of_date=None):
    """Toàn bộ trạng thái công nợ phải được tính LẠI theo as_of_date,
    không dùng trạng thái 'hiện tại' của Payments (tránh Data Leakage)."""
    if df_in.empty:
        raise ValueError(
            "analyze() nhận vào DataFrame rỗng (0 hóa đơn) — không có gì để phân tích. "
            "Kiểm tra lại bước trích xuất/lọc trước đó: có thể thư mục ảnh trống, "
            "hoặc toàn bộ hóa đơn bị clean_extracted_record()/validate_extracted_data() loại bỏ."
        )
    missing = [c for c in REQUIRED_COLUMNS if c not in df_in.columns]
    if missing:
        raise ValueError(
            f"analyze() thiếu cột bắt buộc: {missing}. "
            f"DataFrame truyền vào có các cột: {list(df_in.columns)}. "
            "Kiểm tra lại pipeline trích xuất — schema đầu ra phải khớp REQUIRED_COLUMNS."
        )
    as_of_date = pd.to_datetime(as_of_date or datetime.now().date())
    out = df_in.copy()

    def paid_by(payments):
        return sum(p['pay_amount'] for p in payments if p['pay_date'] <= as_of_date)

    def last_pay_date(payments):
        past = [p['pay_date'] for p in payments if p['pay_date'] <= as_of_date]
        return max(past) if past else pd.NaT

    def discount_ok(payments, net_amount):
        # (4) Điều kiện chiết khấu thật: trả đủ trong <=10 ngày, không tranh chấp, đúng kênh
        past = [p for p in payments if p['pay_date'] <= as_of_date]
        if not past:
            return False
        total_paid = sum(p['pay_amount'] for p in past)
        if total_paid < net_amount * 0.999:
            return False
        if any(p['dispute'] or p['wrong_channel'] for p in past):
            return False
        return max(p['pay_date'] for p in past) <= out.loc[0, 'Invoice_Date']  # placeholder, overwritten below

    out['Paid_To_Date'] = out['Payments'].apply(paid_by)
    out['Outstanding'] = (out['Net_Amount'] - out['Paid_To_Date']).round(0)
    out['Is_Fully_Paid'] = out['Outstanding'] <= 0
    out['Last_Payment_Date'] = out['Payments'].apply(last_pay_date)
    out['Days_To_Pay'] = (out['Last_Payment_Date'] - out['Invoice_Date']).dt.days

    # điều kiện chiết khấu tính đúng (không dùng hàm placeholder ở trên)
    def eligible_discount(row):
        if row['Payment_Terms'] != '2/10 Net 30' or not row['Is_Fully_Paid']:
            return False
        past = [p for p in row['Payments'] if p['pay_date'] <= as_of_date]
        if any(p['dispute'] or p['wrong_channel'] for p in past):
            return False
        return (row['Last_Payment_Date'] - row['Invoice_Date']).days <= 10
    out['Discount_Eligible'] = out.apply(eligible_discount, axis=1)
    out['Missed_Discount_VND'] = np.where(
        (out['Payment_Terms'] == '2/10 Net 30') & out['Is_Fully_Paid'] & ~out['Discount_Eligible'],
        out['Net_Amount'] * 0.02, 0.0)

    # (6) DPO theo trọng số giá trị tiền (dollar-weighted), không phải trung bình cộng số ngày
    paid_rows = out[out['Is_Fully_Paid'] & out['Days_To_Pay'].notna()]
    dpo_dollar_weighted = (
        (paid_rows['Days_To_Pay'] * paid_rows['Net_Amount']).sum() / paid_rows['Net_Amount'].sum()
        if len(paid_rows) else np.nan
    )

    # Aging schedule cho phần dư nợ chưa trả hết tính đến as_of_date
    open_rows = out[~out['Is_Fully_Paid']].copy()
    open_rows['Overdue_Days'] = (as_of_date - open_rows['Due_Date']).dt.days

    def bucket(d):
        if d <= 0: return '1. Chưa đến hạn'
        if d <= 30: return '2. Quá hạn 1-30 ngày'
        if d <= 60: return '3. Quá hạn 31-60 ngày'
        return '4. Quá hạn >60 ngày (rủi ro cao)'
    open_rows['Aging_Bucket'] = open_rows['Overdue_Days'].apply(bucket)

    aging_report = open_rows.groupby('Aging_Bucket')['Outstanding'].agg(
        So_luong_HD='count', Tong_no_VND='sum').reset_index()
    if aging_report['Tong_no_VND'].sum() > 0:
        aging_report['Ty_le_%'] = (aging_report['Tong_no_VND'] / aging_report['Tong_no_VND'].sum() * 100).round(1)

    # (7) Mức độ tập trung nhà cung cấp trên phần dư nợ quá hạn
    overdue_rows = open_rows[open_rows['Overdue_Days'] > 0]
    vendor_concentration = overdue_rows.groupby('Vendor_Name')['Outstanding'].sum().sort_values(ascending=False)
    vendor_concentration_pct = (vendor_concentration / vendor_concentration.sum() * 100).round(1) \
        if vendor_concentration.sum() > 0 else vendor_concentration

    # (8) Dự báo dòng tiền ra: hóa đơn CHƯA quá hạn nhưng sẽ đến hạn trong 7/14/30 ngày tới
    upcoming = open_rows[open_rows['Overdue_Days'] <= 0]
    forecast = {}
    for horizon in (7, 14, 30):
        mask = upcoming['Due_Date'] <= as_of_date + timedelta(days=horizon)
        forecast[horizon] = upcoming.loc[mask, 'Outstanding'].sum()

    return {
        'as_of_date': as_of_date,
        'dpo_dollar_weighted': dpo_dollar_weighted,
        'missed_discount_total': out['Missed_Discount_VND'].sum(),
        'aging_report': aging_report,
        'vendor_concentration_pct': vendor_concentration_pct,
        'cash_outflow_forecast': forecast,
        'flags': {
            'date_anomalies': int(df['Date_Anomaly'].sum()),
            'possible_duplicates': int(df['Possible_Duplicate'].sum()),
        }
    }

# =============================================================
# 4. BÁO CÁO (as_of_date truyền tham số, không hardcode)
# =============================================================
# Chỉ chạy demo report khi gọi trực tiếp "python ap_analysis.py" — không chạy khi bị
# import bởi paddleocr_qwen_extraction.py / invoice_validation.py / streamlit_app.py,
# tránh in báo cáo dữ liệu GIẢ LẬP đè lên output thật của các file đó mỗi lần import.
if __name__ == "__main__":
    result = analyze(df_clean, as_of_date='2026-04-15')

    print("=" * 60)
    print(f"BÁO CÁO CÔNG NỢ & VỐN LƯU ĐỘNG (tính đến {result['as_of_date'].strftime('%d/%m/%Y')})")
    print("=" * 60)
    print(f"1. DPO (dollar-weighted): {result['dpo_dollar_weighted']:.1f} ngày")
    print(f"2. Chiết khấu sớm bị bỏ lỡ: {result['missed_discount_total']:,.0f} VNĐ")
    print(f"3. Cảnh báo dữ liệu: {result['flags']['date_anomalies']} ngày bất thường, "
          f"{result['flags']['possible_duplicates']} hóa đơn nghi trùng lặp\n")

    print("4. AGING SCHEDULE (phần dư nợ, không phải tổng hóa đơn):")
    for _, row in result['aging_report'].iterrows():
        print(f" - {row['Aging_Bucket']}: {row['So_luong_HD']} hóa đơn | {row['Tong_no_VND']:,.0f} VNĐ ({row.get('Ty_le_%', 0)}%)")

    print("\n5. TẬP TRUNG RỦI RO NHÀ CUNG CẤP (trên dư nợ quá hạn):")
    for vendor, pct in result['vendor_concentration_pct'].items():
        print(f" - {vendor}: {pct}%")

    print("\n6. DỰ BÁO DÒNG TIỀN RA (hóa đơn CHƯA quá hạn, sắp đến hạn):")
    for horizon, amt in result['cash_outflow_forecast'].items():
        print(f" - {horizon} ngày tới: {amt:,.0f} VNĐ")
