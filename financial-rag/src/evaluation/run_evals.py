import os
import json
import re
import pandas as pd
from src.core.vector_store import HybridChromaStore
from src.core.generator import generate_financial_answer
from src.llm.providers import TransformersLLM
from src.config import default_config


def compute_keyword_overlap(prediction: str, ground_truth: str) -> float:
    pred_words = set(re.findall(r"\w+", prediction.lower()))
    gt_words = set(re.findall(r"\w+", ground_truth.lower()))
    if not gt_words:
        return 0.0
    overlap = pred_words.intersection(gt_words)
    return round(len(overlap) / len(gt_words), 2)


def create_sample_ground_truth_if_not_exists(file_path: str = "data/ground_truth_eval.json"):
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    if not os.path.exists(file_path):
        sample_data = [
            {
                "question": "Tổng thu nhập hoạt động của Techcombank năm 2025 là bao nhiêu?",
                "ground_truth": "Tổng thu nhập hoạt động năm 2025 đạt 53.391.125 triệu đồng."
            }
        ]
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(sample_data, f, ensure_ascii=False, indent=2)


def run_evaluation_pipeline():
    eval_path = "data/ground_truth_eval.json"
    create_sample_ground_truth_if_not_exists(eval_path)

    with open(eval_path, "r", encoding="utf-8") as f:
        qa_pairs = json.load(f)

    print("=== Khởi tạo Vector Store và Generator ===")
    store = HybridChromaStore()

    llm = TransformersLLM(
        model_name_or_path=default_config.llm_model_name,
        device=default_config.device,
        load_in_4bit=default_config.load_in_4bit,
        max_new_tokens=350
    )

    results = []
    print(f"\nĐang đánh giá {len(qa_pairs)} câu hỏi...")

    for idx, item in enumerate(qa_pairs, 1):
        q = item["question"]
        gt = item["ground_truth"]

        retrieved = store.search(q, top_k = 4)
        res = generate_financial_answer(q, retrieved, llm=llm)
        pred_ans = res["answer"]
        score = compute_keyword_overlap(pred_ans, gt)

        results.append({
            "Câu hỏi": q,
            "Đáp án chuẩn (Ground Truth)": gt,
            "Qwen 2.5 Trả lời": pred_ans,
            "Độ khớp": score
        })
        print(f" -> Xong [{idx}/{len(qa_pairs)}] - Điểm: {score}")

    df = pd.DataFrame(results)
    print("\n=== KẾT QUẢ ĐÁNH GIÁ OFFLINE ===")
    print(df[["Câu hỏi", "Độ khớp"]].to_string(index=False))


if __name__ == "__main__":
    run_evaluation_pipeline()

