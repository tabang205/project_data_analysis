import sys
import json
from paddleocr import PaddleOCR


def process_images(image_paths):
    ocr = PaddleOCR(use_angle_cls=True, lang="vi")
    results_map = {}

    for img_path in image_paths:
        try:
            result = ocr.ocr(img_path)
            lines = []
            if result:
                for res in result:
                    if not res:
                        continue
                    if isinstance(res, dict):
                        if "rec_text" in res:
                            lines.append(str(res["rec_text"]))
                        elif "rec_texts" in res:
                            lines.extend([str(t) for t in res["rec_texts"]])
                    elif isinstance(res, (list, tuple)):
                        for line in res:
                            if isinstance(line, dict) and "rec_text" in line:
                                lines.append(str(line["rec_text"]))
                            elif isinstance(line, (list, tuple)) and len(line) >= 2:
                                _, text_payload = line[:2]
                                if isinstance(text_payload, (list, tuple)):
                                    lines.append(str(text_payload[0]))
                                else:
                                    lines.append(str(text_payload))
            results_map[img_path] = "\n".join(lines)
        except Exception as e:
            print(f"[!] Lỗi khi OCR ảnh {img_path}: {e}", file=sys.stderr)
            results_map[img_path] = ""

    return results_map


if __name__ == "__main__":
    if len(sys.argv) >= 3:
        _, json_input_path, json_output_path = sys.argv[:3]

        with open(json_input_path, "r", encoding="utf-8") as f:
            paths = json.load(f)

        output_data = process_images(paths)

        with open(json_output_path, "w", encoding="utf-8") as f:
            json.dump(output_data, f, ensure_ascii=False)

