# Financial Data Analysis & Intelligent Document Processing (RAG + OCR)

Bộ giải pháp phân tích dữ liệu kế toán - tài chính kết hợp **Production Financial RAG** (Contextual Retrieval + Hybrid Search) và **Intelligent Document Processing (IDP)** trích xuất, đối soát hóa đơn tự động bằng PaddleOCR & Qwen-VL/LLM.

---

## Cấu trúc Dự án

```text
project_data_analysis/
├── financial-rag/                      # Hệ thống RAG Phân tích Báo cáo Tài chính (ChromaDB + BM25)
│   ├── src/
│   │   ├── core/                       # Vector Store, Hybrid Retriever & Chunking
│   │   ├── ingestion/                  # Trích xuất PDF & Contextual Prepending
│   │   ├── llm/                        # Kết nối LLM & Sinh câu trả lời kèm trích dẫn
│   │   └── evaluation/                 # Đánh giá độ chính xác (Ground Truth Eval)
│   ├── app.py                          # Giao diện Streamlit cho Financial Copilot
│   ├── financial_rag_modular_pipeline.ipynb
│   └── requirements.txt
├── invoices_scan/                      # Thư mục lưu trữ hóa đơn đầu vào (được ignore dữ liệu thô)
├── paddleocr_qwen_extraction.py        # Pipeline OCR & LLM bóc tách thông tin hóa đơn
├── invoice_validation.py               # Kiểm tra tính hợp lệ, đối soát chéo số liệu hóa đơn
├── ap_analysis.py                      # Phân tích Công nợ Phải trả (Accounts Payable Analysis)
├── streamlit_app.py                    # Dashboard tương tác kiểm duyệt hóa đơn & phân tích AP
├── data-analysis.ipynb                 # Notebook EDA & Phân tích dữ liệu tổng hợp
├── colab_learning.ipynb                # Notebook thực nghiệm huấn luyện / chạy trên Google Colab
└── kaggle_production.ipynb             # Notebook triển khai pipeline trên Kaggle GPU
```

---

## Tính năng Chính

1. **Financial RAG Copilot (`financial-rag/`)**
   - Trích xuất bảng biểu báo cáo tài chính PDF sang Markdown bảo toàn cấu trúc.
   - Áp dụng **Contextual Retrieval** kết hợp **Hybrid Search (ChromaDB Dense Vector + BM25)**.
   - Trích dẫn nguồn chính xác, chống ảo giác số liệu (Zero Calculation Hallucination).

2. **Bóc tách & Đối soát Hóa đơn Tự động (`paddleocr_qwen_extraction.py`, `invoice_validation.py`)**
   - Nhận diện ký tự quang học tiếng Việt độ chính xác cao với **PaddleOCR** kết hợp mô hình ngôn ngữ lớn **Qwen**.
   - Tự động kiểm tra chéo tổng tiền trước thuế, thuế suất VAT và tổng thanh toán.

3. **Phân tích Công nợ Phải trả (`ap_analysis.py`, `streamlit_app.py`)**
   - Thống kê dòng tiền, phân tích tuổi nợ (Aging Schedule), phát hiện hóa đơn bất thường hoặc trùng lặp.

---

## Hướng dẫn Cài đặt & Chạy

```bash
# 1. Tạo và kích hoạt môi trường ảo
python -m venv .venv
.venv\Scripts\activate

# 2. Cài đặt thư viện
pip install -r financial-rag/requirements.txt

# 3. Chạy ứng dụng Streamlit
streamlit run streamlit_app.py
# Hoặc chạy Financial RAG Copilot:
streamlit run financial-rag/app.py
```
