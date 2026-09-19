"""
Pipeline trích xuất hóa đơn: PaddleOCR (layout/text) -> Qwen (structuring) -> DataFrame
tương thích schema đầu vào của analyze() trong ap_analysis.py.

Phiên bản khuyến nghị:
- PaddleOCR PP-OCRv4 (lang='vi' hoặc 'vietnamese' tùy version) — nhẹ, hỗ trợ tiếng Việt tốt,
  dùng PP-StructureV2 nếu cần giữ layout bảng (nhiều hóa đơn có bảng chi tiết dòng hàng).
  KHÔNG dùng PP-OCRv5 nếu máy không có GPU đời mới — v4 ổn định hơn cho CPU/Colab free tier.
- Qwen2.5-3B-Instruct (bản gốc, KHÔNG quant) để cấu trúc hóa văn bản OCR thành JSON.
  Đã thử AWQ/bitsandbytes cho bản 7B nhưng cả 2 đều gãy trên máy VRAM/RAM thấp:
  bitsandbytes cần load tạm bản fp16 đầy đủ (~15GB) qua RAM trước khi nén -> tràn RAM;
  AutoAWQ (thư viện quant AWQ) đã NGỪNG BẢO TRÌ chính thức, không còn tương thích các bản
  torch/transformers mới. Bản 3B ở fp16 chỉ cần ~6GB VRAM, vừa khít RTX 3050 Ti, không cần
  quant gì cả -> tránh toàn bộ nhóm lỗi trên. Đủ tốt cho tác vụ structuring JSON (không phải
  sinh văn bản dài/sáng tạo, không cần model lớn).
  Nếu cần đọc trực tiếp ảnh hóa đơn mất khung/nghiêng nặng, đổi sang Qwen2.5-VL-3B-Instruct
  (bỏ qua bước OCR text, đưa thẳng ảnh vào model).
  => Pipeline dưới đây dùng cấu hình: PaddleOCR trích text -> Qwen2.5-3B-Instruct structuring,
     phù hợp GPU 6-8GB VRAM như RTX 3050 Ti (laptop), không cần quant/bitsandbytes/AWQ.

Giữ đúng 4 bước Extract/Clean/Map/Export như doc-normalization pipeline đã dùng trước đó.
"""

import json
import re
from datetime import datetime
from pathlib import Path

import pandas as pd
from paddleocr import PaddleOCR
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

# Import lại schema/hàm phân tích đã có sẵn để pipeline này "kế thừa" trực tiếp
from ap_analysis import analyze, terms_list  # noqa: F401  (terms_list dùng để validate Payment_Terms)

QWEN_MODEL_NAME = "Qwen/Qwen2.5-3B-Instruct"

# =============================================================
# BƯỚC 1 — EXTRACT: OCR ảnh/PDF hóa đơn bằng PaddleOCR
# =============================================================

def build_ocr_engine(lang: str = "vi") -> PaddleOCR:
    """Khởi tạo PaddleOCR 1 lần, tái sử dụng cho nhiều file (tránh load lại model mỗi lần gọi).
    PaddleOCR 3.x (từ 04/2025) đổi API không tương thích ngược với 2.x:
    - use_angle_cls / show_log đã bị XÓA khỏi constructor -> dùng use_textline_orientation thay thế
    - Muốn tắt log thì cấu hình qua logging chuẩn của Python, không truyền tham số vào PaddleOCR nữa"""
    import logging
    logging.getLogger("ppocr").setLevel(logging.ERROR)
    return PaddleOCR(lang=lang, use_textline_orientation=True)


def ocr_extract_text(ocr_engine: PaddleOCR, image_path: str) -> str:
    """Chạy OCR trên 1 ảnh/scan hóa đơn, trả về text thô theo thứ tự đọc trên xuống dưới.
    Không tự parse field ở đây — nhiệm vụ của bước này CHỈ là lấy text, tránh lẫn logic.
    PaddleOCR 3.x đổi .ocr(cls=True) -> .predict(); output không còn dạng [box, (text, conf)]
    mà là list các dict có 'rec_texts' / 'rec_scores'."""
    result = ocr_engine.predict(image_path)
    lines = []
    for page in result:
        texts = page.get("rec_texts", [])
        scores = page.get("rec_scores", [])
        for text, score in zip(texts, scores):
            if score >= 0.5:  # bỏ text OCR quá nhiễu, độ tin cậy thấp
                lines.append(text)
    return "\n".join(lines)


# =============================================================
# BƯỚC 2 — MAP: dùng Qwen cấu trúc hóa text OCR thành JSON theo schema
# =============================================================

EXTRACTION_SCHEMA_PROMPT = """Bạn là công cụ trích xuất dữ liệu hóa đơn kế toán tiếng Việt.
Từ văn bản OCR dưới đây, trả về DUY NHẤT một JSON object (không thêm chữ nào khác) với các khóa:
- "legal_invoice_no": số hóa đơn gốc (chuỗi số, không lấy mã nội bộ)
- "invoice_symbol": ký hiệu mẫu số hóa đơn (vd "1C26TAA")
- "vendor_name": tên nhà cung cấp
- "invoice_date": ngày hóa đơn, định dạng YYYY-MM-DD
- "payment_terms": một trong ["Net 30", "Net 60", "2/10 Net 30"], suy luận từ điều khoản ghi trên hóa đơn,
  nếu không thấy ghi rõ thì để null (KHÔNG tự đoán bừa)
- "due_date": ngày đến hạn nếu có ghi rõ, không thì null
- "gross_amount": tổng tiền trước điều chỉnh (số, đơn vị VNĐ, không có dấu phẩy/chấm phân cách)
- "credit_note_amount": số tiền giảm trừ/hàng trả lại nếu hóa đơn có ghi chú, không thì 0

Nếu một trường không xác định được từ text, để null — KHÔNG bịa số liệu.

Văn bản OCR:
---
{ocr_text}
---
JSON:"""


def build_qwen(model_name: str = QWEN_MODEL_NAME):
    """Load Qwen 1 lần. Bản 3B ở fp16 chỉ cần ~6GB VRAM — không cần quant (bitsandbytes/AWQ),
    tránh toàn bộ nhóm lỗi liên quan (tràn RAM lúc quant runtime, package quant ngừng bảo trì)."""
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        dtype="auto",  # "torch_dtype" đã deprecated trong transformers bản mới, dùng "dtype"
        device_map="auto",
    )
    return tokenizer, model


def qwen_structure_invoice(tokenizer, model, ocr_text: str) -> dict:
    """Gọi Qwen với prompt ràng buộc schema, parse JSON output.
    Trả về dict rỗng nếu model trả về JSON không hợp lệ (để bước Clean xử lý tiếp).
    
    Lưu ý: apply_chat_template() có thể trả về tensor thuần HOẶC BatchEncoding
    (object dict-like của transformers, subscriptable qua ["input_ids"] nhưng
    KHÔNG có .shape trực tiếp) tùy phiên bản/tham số -> check bằng hasattr("shape"),
    không dùng isinstance(..., dict) vì BatchEncoding không kế thừa dict thuần."""
    prompt = EXTRACTION_SCHEMA_PROMPT.format(ocr_text=ocr_text[:4000])  # cắt bớt tránh vượt context
    messages = [{"role": "user", "content": prompt}]
    chat_output = tokenizer.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt")

    if hasattr(chat_output, "shape"):
        inputs = chat_output.to(model.device)
    else:
        inputs = chat_output["input_ids"].to(model.device)

    output = model.generate(inputs, max_new_tokens=512, do_sample=False)
    raw = tokenizer.decode(output[0][inputs.shape[-1]:], skip_special_tokens=True)

    # Model đôi khi bọc JSON trong ```json ... ``` dù đã cấm — chỉ lấy phần {...} đầu tiên cho an toàn
    match = re.search(r"\{.*\}", raw, re.DOTALL)
    if not match:
        return {}
    try:
        return json.loads(match.group(0))
    except json.JSONDecodeError:
        return {}


# =============================================================
# BƯỚC 3 — CLEAN: validate output của Qwen trước khi đưa vào phân tích
# =============================================================

def clean_extracted_record(record: dict, invoice_id: str) -> dict | None:
    """Loại bỏ hoặc gắn cờ record thiếu field bắt buộc — không để lọt dữ liệu rác
    vào pipeline phân tích công nợ (kế thừa nguyên tắc Data Integrity của ap_analysis.py)."""
    required = ["vendor_name", "invoice_date", "gross_amount"]
    if any(record.get(k) in (None, "", 0) for k in required if k != "gross_amount") or record.get("gross_amount") in (None, 0):
        return None  # thiếu field bắt buộc -> đẩy sang hàng chờ soát thủ công, không auto-nhét vào báo cáo

    try:
        invoice_date = pd.to_datetime(record["invoice_date"])
    except (ValueError, TypeError):
        return None

    payment_terms = record.get("payment_terms")
    if payment_terms not in terms_list:
        payment_terms = None  # để trống thay vì đoán, tránh sai lệch DPO/aging về sau

    due_date = pd.to_datetime(record["due_date"]) if record.get("due_date") else None
    if payment_terms and due_date is None:
        term_days = 60 if payment_terms == "Net 60" else 30
        due_date = invoice_date + pd.Timedelta(days=term_days)

    gross = float(record["gross_amount"])
    credit_note = float(record.get("credit_note_amount") or 0)

    return {
        "Invoice_ID": invoice_id,
        "Legal_Invoice_No": record.get("legal_invoice_no"),
        "Invoice_Symbol": record.get("invoice_symbol"),
        "Vendor_Name": record["vendor_name"],
        "Invoice_Date": invoice_date,
        "Payment_Terms": payment_terms,
        "Due_Date": due_date,
        "Gross_Amount": gross,
        "Credit_Note_Amount": credit_note,
        "Net_Amount": gross - credit_note,
        "Payments": [],  # OCR hóa đơn không có thông tin thanh toán — nạp riêng từ sổ quỹ/ngân hàng
        "Date_Anomaly": bool(due_date is not None and invoice_date > due_date),
        "Possible_Duplicate": False,  # đối chiếu trùng lặp thực hiện ở cấp toàn bộ batch, xem bước Export
    }


# =============================================================
# BƯỚC 4 — EXPORT: gộp batch, phát hiện trùng lặp, trả DataFrame cho analyze()
# =============================================================

def extract_invoices_from_folder(folder: str, lang: str = "vi") -> pd.DataFrame:
    """Chạy toàn bộ pipeline trên 1 thư mục ảnh/scan hóa đơn, trả về DataFrame
    có thể truyền thẳng vào analyze() của ap_analysis.py."""
    ocr_engine = build_ocr_engine(lang)
    tokenizer, model = build_qwen()

    records = []
    skipped = []
    for i, path in enumerate(sorted(Path(folder).glob("*.*")), start=1):
        invoice_id = f"OCR-{datetime.now():%Y%m}-{i:03d}"
        text = ocr_extract_text(ocr_engine, str(path))
        raw_record = qwen_structure_invoice(tokenizer, model, text)
        clean = clean_extracted_record(raw_record, invoice_id)
        if clean is None:
            skipped.append(path.name)  # đưa vào hàng chờ soát thủ công, không âm thầm bỏ qua
            continue
        records.append(clean)

    df_out = pd.DataFrame(records)
    if not df_out.empty:
        df_out["Possible_Duplicate"] = df_out.duplicated(
            subset=["Vendor_Name", "Gross_Amount", "Invoice_Date"], keep=False
        )

    if skipped:
        print(f"[!] {len(skipped)} file cần soát thủ công (thiếu field bắt buộc): {skipped}")

    return df_out


if __name__ == "__main__":
    df_ocr = extract_invoices_from_folder("./invoices_scan")
    result = analyze(df_ocr, as_of_date=datetime.now().date().isoformat())
    print(f"Trích xuất {len(df_ocr)} hóa đơn hợp lệ, DPO: {result['dpo_dollar_weighted']:.1f} ngày")
