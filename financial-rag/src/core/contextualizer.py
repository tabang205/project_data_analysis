from typing import List, Dict, Any, Optional
from src.config import default_config

def generate_contextualized_chunks(
    whole_doc: str,
    chunks: List[Dict[str, Any]],
    llm: Optional[Any] = None
) -> List[Dict[str, Any]]:
    """Tối ưu hóa: gắn Header Context tĩnh để không làm quá tải GPU laptop/Kaggle."""
    contextualized_chunks = []

    for chunk in chunks:
        meta = chunk.get("metadata", {})
        comp = meta.get("company_name", "Doanh nghiệp")
        year = meta.get("fiscal_year", "N/A")
        page = meta.get("page_number", 0)

        header_context = f"[Tài liệu: BCTC {comp} ({year}) | Trang: {page}]"
        augmented_content = f"{header_context}\n\n{chunk['content']}"

        contextualized_chunks.append({
            "original_content": chunk["content"],
            "augmented_content": augmented_content,
            "metadata": chunk["metadata"]
        })

    return contextualized_chunks

