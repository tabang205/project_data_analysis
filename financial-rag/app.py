import os
import time
import tempfile
import streamlit as st

from src.ingestion.pdf_extractor import extract_document_data
from src.ingestion.chunker import build_semantic_chunks
from src.core.contextualizer import generate_contextualized_chunks
from src.core.vector_store import HybridChromaStore
from src.core.generator import generate_financial_answer
from src.llm.providers import TransformersLLM
from src.config import default_config

st.set_page_config(
    page_title="Hệ Thống Phân Tích Dữ Liệu Doanh Nghiệp",
    page_icon="📑",
    layout="wide"
)

# Custom Styling cho giao diện
st.markdown("""
<style>
    .info-badge {
        background-color: #1e293b;
        border-left: 4px solid #38bdf8;
        padding: 14px 18px;
        border-radius: 8px;
        margin-bottom: 16px;
        font-size: 14px;
    }
    .source-box {
        background-color: #0f172a;
        border: 1px solid #334155;
        border-radius: 8px;
        padding: 14px;
        margin-bottom: 12px;
        font-size: 13.5px;
        line-height: 1.6;
        color: #e2e8f0;
    }
    .page-tag {
        background-color: #0369a1;
        color: #ffffff;
        font-weight: 600;
        padding: 3px 8px;
        border-radius: 4px;
        font-size: 12px;
        margin-bottom: 6px;
        display: inline-block;
    }
    .metric-tag {
        color: #94a3b8;
        font-size: 12px;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_resource(show_spinner=False)
def get_vector_store():
    return HybridChromaStore()


@st.cache_resource(show_spinner=False)
def get_shared_llm():
    return TransformersLLM(
        model_name_or_path=default_config.llm_model_name,
        device=default_config.device,
        load_in_4bit=default_config.load_in_4bit,
        max_new_tokens=350
    )


vector_store = get_vector_store()
llm = get_shared_llm()

# Khởi tạo trạng thái phiên làm việc (Session State)
if "processed_file_id" not in st.session_state:
    st.session_state["processed_file_id"] = None
if "ready_to_chat" not in st.session_state:
    st.session_state["ready_to_chat"] = False
if "messages" not in st.session_state:
    st.session_state.messages = []

# --- THANH BÊN (SIDEBAR): TỰ ĐỘNG XỬ LÝ KHI CÓ FILE ---
with st.sidebar:
    st.title("📂 Quản Lý Hồ Sơ Tài Liệu")
    st.markdown("---")

    doc_title = st.text_input("Tên đơn vị / Tên tài liệu", value="Báo cáo phân tích")
    doc_period = st.selectbox("Kỳ báo cáo / Phiên bản", ["2025", "2024", "2023", "Bản cập nhật"])
    uploaded_file = st.file_uploader("Tải lên tài liệu (PDF hoặc Ảnh)", type=["pdf", "png", "jpg", "jpeg"])

    # 1. TỰ ĐỘNG KÍCH HOẠT XỬ LÝ KHI PHÁT HIỆN FILE MỚI ĐƯỢC TẢI LÊN
    if uploaded_file is not None:
        file_id = f"{uploaded_file.name}_{uploaded_file.size}"

        # Chỉ xử lý nếu đây là file mới tải lên (tránh xử lý lặp lại khi chat)
        if st.session_state["processed_file_id"] != file_id:
            pdf_bytes = uploaded_file.getvalue()
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                tmp.write(pdf_bytes)
                tmp_path = tmp.name

            try:
                with st.status("⚡ Tự động phân tích và nạp dữ liệu...", expanded=True) as status:
                    pages_data = extract_document_data(tmp_path)

                    if not pages_data or not any(p.get("text", "").strip() for p in pages_data):
                        status.update(label="Không thể xử lý tài liệu", state="error")
                        st.error("⚠️ Định dạng tài liệu không hợp lệ hoặc không có nội dung văn bản.")
                        st.session_state["ready_to_chat"] = False
                        st.stop()

                    meta = {
                        "company_name": doc_title,
                        "fiscal_year": doc_period,
                        "filename": uploaded_file.name
                    }
                    raw_chunks = build_semantic_chunks(pages_data, meta)

                    whole_doc = "\n".join([p["text"] for p in pages_data if p.get("text")])
                    augmented_chunks = generate_contextualized_chunks(whole_doc, raw_chunks, llm=llm)

                    vector_store.add_chunks(augmented_chunks)

                    # Ghi nhớ trạng thái hoàn tất
                    st.session_state["processed_file_id"] = file_id
                    st.session_state["ready_to_chat"] = True
                    st.session_state["active_doc"] = {
                        "name": doc_title,
                        "period": doc_period,
                        "total_pages": len(pages_data),
                        "total_segments": len(augmented_chunks),
                        "file_name": uploaded_file.name
                    }
                    status.update(label="✅ Đã xử lý xong và sẵn sàng đối soát!", state="complete", expanded=False)
                    st.toast("Tài liệu đã sẵn sàng! Khung câu hỏi đã được mở khóa.", icon="🎉")
            except Exception as e:
                st.error("⚠️ Đã có sự cố trong quá trình xử lý file.")
                st.session_state["ready_to_chat"] = False
            finally:
                if os.path.exists(tmp_path):
                    try:
                        os.remove(tmp_path)
                    except Exception:
                        pass
    else:
        # Khi người dùng xóa file
        st.session_state["processed_file_id"] = None
        st.session_state["ready_to_chat"] = False
        if "active_doc" in st.session_state:
            del st.session_state["active_doc"]

# 2. KHÓA / MỞ KHÓA KHUNG HỎI ĐÁP
is_ready = st.session_state.get("ready_to_chat", False)

# Khởi tạo biến lưu câu hỏi trong session
if "current_query" not in st.session_state:
    st.session_state["current_query"] = None

# Khung nhập câu hỏi (Streamlit luôn ghim cố định ở đáy màn hình)
user_input = st.chat_input(
    "Nhập nội dung cần tra cứu từ báo cáo tại đây..." if is_ready else "Vui lòng tải lên tài liệu ở thanh bên trái để mở khóa...",
    disabled=not is_ready
)

# Nếu người dùng vừa gõ câu hỏi mới
if user_input:
    st.session_state["current_query"] = user_input

# ==============================================================================
# TRƯỜNG HỢP A: CHƯA NHẬP PROMPT -> HIỂN THỊ VÙNG KHOANH ĐỎ (HERO BANNER & GỢI Ý)
# ==============================================================================
if not st.session_state["current_query"]:
    st.title("📊 Trợ Lý Phân Tích & Đối Soát Báo Cáo")
    st.caption("Nền tảng tra cứu số liệu tự động, hỗ trợ đối chiếu thông tin minh bạch từ báo cáo gốc.")

    if is_ready and "active_doc" in st.session_state:
        doc = st.session_state["active_doc"]
        st.markdown(f"""
        <div class="info-badge">
            <b>Tài liệu đang mở:</b> {doc['file_name']} | <b>Hồ sơ:</b> {doc['name']} ({doc['period']}) |
            <b>Quy mô:</b> {doc['total_pages']} trang (Đã đồng bộ {doc['total_segments']} phần dữ liệu)
        </div>
        """, unsafe_allow_html=True)
    else:
        st.info("👈 **Hướng dẫn:** Hãy tải lên tài liệu ở cột bên trái. Hệ thống sẽ **tự động xử lý ngay lập tức** và mở khóa khung đặt câu hỏi khi hoàn tất.")

    # Các nút gợi ý câu hỏi nhanh
    if is_ready:
        st.markdown("##### 💡 Gợi ý câu hỏi kiểm tra nhanh:")
        c1, c2, c3, c4 = st.columns(4)
        doc_period = st.session_state.get("active_doc", {}).get("period", "mới nhất")
        doc_name = st.session_state.get("active_doc", {}).get("name", "tài liệu")

        if c1.button("📈 Doanh thu & Thu nhập", use_container_width=True):
            st.session_state["current_query"] = f"Tổng thu nhập hoạt động và kết quả kinh doanh năm {doc_period} của {doc_name} đạt bao nhiêu?"
            st.rerun()
        if c2.button("📉 Chi phí & Vận hành", use_container_width=True):
            st.session_state["current_query"] = f"Tình hình chi phí hoạt động và các khoản chi phí lớn nhất trong kỳ là gì?"
            st.rerun()
        if c3.button("🏦 Tài sản & Dư nợ", use_container_width=True):
            st.session_state["current_query"] = f"Tổng tài sản và quy mô cho vay / đầu tư biến động như thế nào đến cuối kỳ?"
            st.rerun()
        if c4.button("🔍 Đánh giá rủi ro & Lưu ý", use_container_width=True):
            st.session_state["current_query"] = f"Báo cáo có những lưu ý hoặc thuyết minh trọng yếu nào về rủi ro và nợ khó đòi?"
            st.rerun()

# ==============================================================================
# TRƯỜNG HỢP B: ĐÃ CÓ PROMPT -> ẨN TOÀN BỘ VÙNG ĐỎ, CHỈ HIỆN 2 CỘT KẾT QUẢ
# ==============================================================================
else:
    active_query = st.session_state["current_query"]

    # Thanh tiêu đề nhỏ gọn kèm nút hỏi câu khác
    top_col1, top_col2 = st.columns(2)
    with top_col1:
        st.markdown(f"### ❓ **Câu hỏi:** *{active_query}*")
    with top_col2:
        if st.button("🔄 Hỏi câu khác", use_container_width=True):
            st.session_state["current_query"] = None
            st.rerun()

    st.markdown("---")

    # Hiển thị chữ "Đang suy nghĩ..." trong lúc AI đang tìm kiếm và tính toán
    with st.spinner("Đang suy nghĩ..."):
        t_start = time.time()
        retrieved_chunks = vector_store.search(active_query, top_k=4)
        retrieve_duration = time.time() - t_start

        t_gen_start = time.time()
        result = generate_financial_answer(active_query, retrieved_chunks, llm=llm)
        gen_duration = time.time() - t_gen_start

    # HIỂN THỊ CÂU TRẢ LỜI TẬP TRUNG
    st.subheader("💡 Kết Quả Phân Tích")
    st.markdown(f"<span class='metric-tag'>Thời gian phản hồi: {gen_duration:.2f} giây &bull; Đối soát xong trong {retrieve_duration:.2f} giây</span>", unsafe_allow_html=True)
    st.markdown("---")

    # 1. Nội dung câu trả lời phân tích chính (Chiếm trọn vẹn màn hình, dễ đọc)
    st.markdown(result["answer"])

    # 2. Đính kèm căn cứ số liệu ngay dưới câu trả lời
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("##### 📌 Căn cứ dữ liệu & Nguồn trích dẫn:")

    citations = result.get("citations", [])
    has_valid_citation = False
    for cit in citations:
        if cit.get("page"):
            has_valid_citation = True
            st.markdown(f"- **Vị trí Trang {cit['page']}** *({cit.get('company', 'Tài liệu')})*: `{cit['snippet']}`")

    if not has_valid_citation:
        st.caption("*(Dữ liệu được tổng hợp từ các đoạn ngữ cảnh liên quan trong tài liệu)*")

    # 3. Hộp gập/mở đối chiếu nguyên văn (chỉ bung ra khi người dùng cần kiểm chứng kỹ)
    if retrieved_chunks:
        with st.expander("📑 Xem lại nguyên văn các đoạn trích từ tài liệu gốc (Đối chứng)", expanded=False):
            for i, c in enumerate(retrieved_chunks, 1):
                meta = c.get("metadata", {})
                page = meta.get("page_number", meta.get("page", "?"))
                comp = meta.get("company_name", "")
                content = c.get("augmented_content", c.get("content", ""))
                cleaned_display = content.split("]\n\n")[-1] if "]\n\n" in content else content

                st.markdown(f"""
                <div class="source-box">
                    <span class="page-tag">Đoạn đối chiếu {i} &bull; Trang {page} ({comp})</span><br>
                    {cleaned_display}
                </div>
                """, unsafe_allow_html=True)
