from typing import List, Dict, Any, Optional
from src.llm.providers import TransformersLLM
from src.config import default_config

_generator_llm: Optional[TransformersLLM] = None

def get_generator_llm() -> TransformersLLM:
    global _generator_llm
    if _generator_llm is None:
        _generator_llm = TransformersLLM(
            model_name_or_path=default_config.llm_model_name,
            device=default_config.device,
            load_in_4bit=default_config.load_in_4bit,
            max_new_tokens=350,
            temperature=0.1
        )
    return _generator_llm

SYSTEM_PROMPT = """Bạn là trợ lý Chuyên gia Phân tích Tài chính C-Level (Financial Copilot).
Nhiệm vụ của bạn là trả lời các câu hỏi dựa CHÍNH XÁC trên ngữ cảnh (Context) được cung cấp.

Quy tắc:
1. KHÔNG tự suy đoán, KHÔNG bịa đặt số liệu ngoài tài liệu.
2. Với các bảng số liệu, nếu cần tính tỷ lệ (như biên lợi nhuận), hãy giải thích rõ công thức và số liệu thô.
3. Sử dụng thẻ <thinking> trước khi đưa ra câu trả lời chính thức để:
   - Phân tích nguồn thông tin và tính toán.
   - Ghi nhận trang tài liệu.
4. Trả lời rõ ràng, chuyên nghiệp bằng Markdown."""

USER_TEMPLATE = """Dưới đây là tài liệu liên quan:
<context>
{CONTEXT}
</context>

Câu hỏi của người dùng:
<query>
{QUERY}
</query>

Hãy trả lời theo định dạng:
<thinking>
[Phân tích nguồn thông tin và logic tính toán tại đây]
</thinking>

[Nội dung trả lời trực tiếp cho lãnh đạo doanh nghiệp]
"""


def generate_financial_answer(
    query: str,
    retrieved_chunks: List[Dict[str, Any]],
    llm: Optional[TransformersLLM] = None
) -> Dict[str, Any]:
    active_llm = llm or get_generator_llm()

    context_str = ""
    for idx, c in enumerate(retrieved_chunks[:2]):
        meta = c.get("metadata", {})
        page = meta.get("page_number", meta.get("page", "N/A"))
        company = meta.get("company_name", meta.get("company", "N/A"))
        context_str += f"\n--- TÀI LIỆU {idx+1} (Trang {page}, {company}) ---\n"
        raw_content = c.get("augmented_content", c.get("content", ""))[:800]
        context_str += raw_content + "\n"

    user_prompt = USER_TEMPLATE.format(CONTEXT=context_str, QUERY=query)
    full_prompt = f"{SYSTEM_PROMPT}\n\n{user_prompt}"

    raw_text = active_llm.generate(full_prompt, max_new_tokens=350)

    thinking = ""
    answer = raw_text
    if "<thinking>" in raw_text and "</thinking>" in raw_text:
        parts = raw_text.split("</thinking>")
        thinking = parts[0].replace("<thinking>", "").strip()
        answer = parts[-1].strip()

    return {
        "answer": answer,
        "thinking": thinking,
        "citations": [
            {
                "page": c.get("metadata", {}).get("page_number"),
                "company": c.get("metadata", {}).get("company_name"),
                "snippet": c.get("original_content", c.get("content", ""))[:200] + "..."
            }
            for c in retrieved_chunks[:2]
        ]
    }

