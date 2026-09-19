from typing import List, Dict, Any


def build_semantic_chunks(
    pages_data: List[Dict[str, Any]],
    meta_info: Dict[str, str],
    max_chunk_chars: int = 1500
) -> List[Dict[str, Any]]:
    chunks = []

    for item in pages_data:
        p_num = item.get("page_number", 0)
        text_content = item.get("text", "").strip()
        tables = item.get("tables", [])

        parts = []
        if text_content:
            parts.append(text_content)
        if tables:
            parts.append("### BẢNG BIỂU TÀI CHÍNH:\n" + "\n\n".join(tables))

        full_content = "\n\n".join(parts)
        if not full_content:
            continue

        paragraphs = full_content.split("\n\n")
        current_chunk = ""

        def add_chunk(content: str):
            chunks.append({
                "content": content.strip(),
                "metadata": {
                    **meta_info,
                    "page_number": p_num
                }
            })

        for para in paragraphs:
            para = para.strip()
            if not para:
                continue

            if len(para) > max_chunk_chars:
                if current_chunk:
                    add_chunk(current_chunk)
                    current_chunk = ""
                add_chunk(para)
                continue

            if len(current_chunk) + len(para) <= max_chunk_chars:
                current_chunk += para + "\n\n"
            else:
                if current_chunk:
                    add_chunk(current_chunk)
                current_chunk = para + "\n\n"

        if current_chunk:
            add_chunk(current_chunk)

    return chunks

