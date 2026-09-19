import os
import sys
import re
import json
import subprocess
import tempfile
from typing import List, Dict, Any
from PIL import Image
import pdfplumber


def clean_financial_number(text: str) -> str:
    """Chuyển đổi (1.234.567) thành -1.234.567 cho bảng tài chính."""
    if not text:
        return ""
    return re.sub(r"\(\s*([0-9\.,]+)\s*\)", r"-\1", text.strip())


def run_paddle_ocr_batch(pil_images: List[Image.Image]) -> List[str]:
    worker_script = os.path.join(os.path.dirname(__file__), "paddle_worker.py")
    temp_dir = tempfile.mkdtemp()
    image_paths = []

    try:
        for idx, img in enumerate(pil_images):
            img_path = os.path.join(temp_dir, f"page_{idx}.jpg")
            img.convert("RGB").save(img_path, format="JPEG", quality=95)
            image_paths.append(img_path)

        input_json = os.path.join(temp_dir, "input.json")
        output_json = os.path.join(temp_dir, "output.json")

        with open(input_json, "w", encoding="utf-8") as f:
            json.dump(image_paths, f)

        cmd = [sys.executable, worker_script, input_json, output_json]
        proc = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8")

        if proc.returncode != 0:
            print(f"[!] Lỗi tiến trình PaddleOCR: {proc.stderr}")
            return ["" for _ in pil_images]

        if not os.path.exists(output_json):
            return ["" for _ in pil_images]

        with open(output_json, "r", encoding="utf-8") as f:
            res_dict = json.load(f)

        return [res_dict.get(p, "") for p in image_paths]
    finally:
        for p in image_paths:
            if os.path.exists(p):
                try:
                    os.remove(p)
                except Exception:
                    pass
        for jf in [os.path.join(temp_dir, "input.json"), os.path.join(temp_dir, "output.json")]:
            if os.path.exists(jf):
                try:
                    os.remove(jf)
                except Exception:
                    pass
        try:
            os.rmdir(temp_dir)
        except Exception:
            pass


def extract_document_data(file_path: str, progress_callback=None) -> List[Dict[str, Any]]:
    file_path_str = str(file_path)
    ext = "." + file_path_str.split(".")[-1].lower() if "." in file_path_str else ""
    pages_data = []

    if ext in [".png", ".jpg", ".jpeg"]:
        if progress_callback:
            progress_callback(1, 1)
        img = Image.open(file_path)
        texts = run_paddle_ocr_batch([img])
        return [{
            "page_number": 1,
            "text": texts[0].strip() if texts else "",
            "tables": []
        }]

    with pdfplumber.open(file_path) as pdf:
        total_pages = len(pdf.pages)
        has_any_text = False

        for idx, page in enumerate(pdf.pages):
            page_num = idx + 1
            if progress_callback:
                progress_callback(page_num, total_pages)

            raw_text = page.extract_text(layout=False) or ""
            if raw_text.strip():
                has_any_text = True

            markdown_tables = []
            try:
                tables = page.extract_tables()
                for tbl in tables:
                    if not tbl or len(tbl) < 2:
                        continue
                    cleaned_rows = [
                        [clean_financial_number(cell).replace("\n", " ").strip() if cell else "-" for cell in row]
                        for row in tbl
                    ]
                    headers = [h if h != "-" else f"Col_{i}" for i, h in enumerate(cleaned_rows[0])]
                    md = "| " + " | ".join(headers) + " |\n"
                    md += "| " + " | ".join(["---"] * len(headers)) + " |\n"
                    for row in cleaned_rows[1:]:
                        md += "| " + " | ".join(row) + " |\n"
                    markdown_tables.append(md)
            except Exception:
                pass

            pages_data.append({
                "page_number": page_num,
                "text": raw_text.strip(),
                "tables": markdown_tables
            })

    if not has_any_text:
        print("[!] Phát hiện PDF Scan. Đang gọi PaddleOCR qua tiến trình con độc lập...")
        import pypdfium2 as pdfium
        pdf_doc = pdfium.PdfDocument(file_path)
        pages_data = []

        try:
            max_scan_pages = min(len(pdf_doc), 25)
            pil_images = []

            for idx in range(max_scan_pages):
                if progress_callback:
                    progress_callback(idx + 1, max_scan_pages)
                page = pdf_doc[idx]
                pil_images.append(page.render(scale=140 / 72).to_pil())

            ocr_results = run_paddle_ocr_batch(pil_images)

            for idx, text in enumerate(ocr_results):
                pages_data.append({
                    "page_number": idx + 1,
                    "text": text.strip(),
                    "tables": []
                })
        finally:
            pdf_doc.close()

    return pages_data

extract_pdf_with_tables = extract_document_data

