# 📊 Financial & Business Intelligence Copilot (Production RAG)

> Hệ thống Trợ lý Phân tích Báo cáo Tài chính tự động hóa dựa trên kiến trúc **Contextual Retrieval của Anthropic**, loại bỏ hoàn toàn hiện tượng vỡ bảng số liệu và ảo giác tính toán.

## 🏗 Kiến trúc Hệ thống (Architecture)

```mermaid
graph TD
    A[Báo cáo tài chính PDF] --> B[pdfplumber: Trích xuất Bảng sang Markdown]
    B --> C[Semantic Chunking 500-800 tokens]
    C --> D[Claude 3 Haiku: Contextual Prepending]
    D --> E[(ChromaDB + BM25 Hybrid Store)]
    
    F[User Query: CFO / BI] --> G[Hybrid Retrieval]
    E --> G
    G --> H[Context Injection via XML Tags]
    H --> I[Claude 3.5 Sonnet: Reasoning & Citation]
    I --> J[Streamlit UI + Verified Citations]