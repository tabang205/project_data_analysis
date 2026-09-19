"""
Demo tương tác cho khách hàng/người dùng cuối (không cần biết code).
Chạy: streamlit run streamlit_app.py

Có 2 CHẾ ĐỘ tách biệt rõ ràng — xem giải thích Web vs Local ở cuối file:

- "Demo mẫu"   : dữ liệu giả lập (ap_analysis.generate_invoices) -> AN TOÀN để deploy công khai
                 lên web (Streamlit Cloud / VPS) cho khách xem trước khi ký hợp đồng.
- "Dữ liệu thật": upload hóa đơn thật (ảnh -> OCR/Qwen, hoặc CSV đã trích xuất sẵn) ->
                 BẮT BUỘC chạy local/on-prem, không public lên internet.
"""

import io
from datetime import datetime

import pandas as pd
import plotly.express as px
import streamlit as st

from ap_analysis import generate_invoices, analyze
from invoice_validation import validate_extracted_data

st.set_page_config(page_title="Quản trị Công nợ Nhà cung cấp", layout="wide")
st.title("📊 Dashboard Quản trị Công nợ & Vốn lưu động")

mode = st.sidebar.radio(
    "Chọn chế độ",
    ["🧪 Demo mẫu (dữ liệu giả lập)", "📁 Dữ liệu thật (upload CSV đã trích xuất)"],
)
as_of = st.sidebar.date_input("Chốt sổ tính đến ngày", value=datetime(2026, 4, 15))

# -------------------------------------------------------------
# Chế độ 1: Demo mẫu — luôn chạy được, không đụng dữ liệu thật
# -------------------------------------------------------------
if mode.startswith("🧪"):
    st.info("Đang xem **dữ liệu mẫu** để minh họa, không phải số liệu thật của công ty bạn.")
    n = st.sidebar.slider("Số hóa đơn mẫu", 20, 200, 50)
    df = generate_invoices(n)
    df_valid = df  # dữ liệu sinh sẵn, không cần qua cổng validate

# -------------------------------------------------------------
# Chế độ 2: Dữ liệu thật — bắt buộc qua cổng kiểm thử trước khi hiển thị
# -------------------------------------------------------------
else:
    st.warning(
        "⚠️ Chế độ này xử lý dữ liệu công nợ thật — chỉ nên chạy trên máy nội bộ/on-prem, "
        "KHÔNG dùng bản deploy công khai cho chế độ này (xem ghi chú cuối trang)."
    )
    uploaded = st.file_uploader(
        "Upload file CSV hóa đơn (cột: Invoice_ID, Vendor_Name, Invoice_Date, Payment_Terms, "
        "Due_Date, Gross_Amount, Credit_Note_Amount, Net_Amount)",
        type=["csv"],
    )
    if uploaded is None:
        st.stop()

    raw = pd.read_csv(uploaded, parse_dates=["Invoice_Date", "Due_Date"])
    raw["Payments"] = [[] for _ in range(len(raw))]  # placeholder, nạp riêng từ sổ quỹ nếu có
    df_valid, df_bad = validate_extracted_data(raw)

    if not df_bad.empty:
        with st.expander(f"❌ {len(df_bad)} hóa đơn bị giữ lại — cần kế toán soát tay"):
            st.dataframe(df_bad, use_container_width=True)

    if df_valid.empty:
        st.error("Không có hóa đơn nào qua được kiểm thử. Vui lòng kiểm tra lại file CSV.")
        st.stop()
    st.success(f"✅ {len(df_valid)} hóa đơn hợp lệ, sẵn sàng phân tích.")

# -------------------------------------------------------------
# Phân tích & hiển thị (dùng chung cho cả 2 chế độ)
# -------------------------------------------------------------
result = analyze(df_valid, as_of_date=as_of.isoformat())

col1, col2, col3 = st.columns(3)
col1.metric("DPO (dollar-weighted)", f"{result['dpo_dollar_weighted']:.1f} ngày")
col2.metric("Chiết khấu sớm bị bỏ lỡ", f"{result['missed_discount_total']:,.0f} VNĐ")
col3.metric("Hóa đơn nghi trùng lặp", result['flags']['possible_duplicates'])

st.subheader("Bảng phân bổ tuổi nợ (Aging Schedule)")
aging = result["aging_report"]
if not aging.empty:
    fig_aging = px.bar(aging, x="Aging_Bucket", y="Tong_no_VND", text="So_luong_HD",
                        labels={"Tong_no_VND": "Dư nợ (VNĐ)", "Aging_Bucket": ""})
    st.plotly_chart(fig_aging, use_container_width=True)
else:
    st.write("Không có dư nợ mở tại thời điểm chốt sổ.")

col4, col5 = st.columns(2)
with col4:
    st.subheader("Tập trung rủi ro nhà cung cấp")
    vc = result["vendor_concentration_pct"]
    if len(vc):
        st.plotly_chart(px.pie(values=vc.values, names=vc.index), use_container_width=True)
    else:
        st.write("Không có dư nợ quá hạn.")

with col5:
    st.subheader("Dự báo dòng tiền ra sắp tới")
    forecast = result["cash_outflow_forecast"]
    st.plotly_chart(
        px.bar(x=[f"{h} ngày" for h in forecast], y=list(forecast.values()),
               labels={"x": "", "y": "VNĐ cần chuẩn bị"}),
        use_container_width=True,
    )

with st.expander("Xem dữ liệu chi tiết"):
    st.dataframe(df_valid, use_container_width=True)

st.divider()
st.caption(
    "Demo mẫu dùng dữ liệu giả lập — an toàn để chia sẻ link công khai cho khách xem trước. "
    "Chế độ dữ liệu thật xử lý số liệu công nợ nội bộ — chỉ chạy trên máy/server nội bộ, "
    "không public lên internet."
)
