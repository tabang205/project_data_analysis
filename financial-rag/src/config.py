import os
from dataclasses import dataclass
import torch


@dataclass
class RAGConfig:
    # --- Cấu hình Đường dẫn Dữ liệu (Paths) ---
    data_raw_dir: str = "./data/raw"
    vector_store_path: str = "./data/vector_db"
    eval_data_path: str = "./data/ground_truth_eval.json"

    # --- Cấu hình Text Chunking & Embedding ---
    max_chunk_chars: int = 1500
    chunk_overlap: int = 100
    embedding_provider: str = "sentence-transformers"
    embedding_model_name: str = "BAAI/bge-m3"
    top_k: int = 4

    # --- Cấu hình LLM ---
    llm_provider: str = "transformers"
    llm_model_name: str = "Qwen/Qwen2.5-3B-Instruct"
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    max_new_tokens: int = 512
    temperature: float = 0.1
    torch_dtype: str = "bfloat16" if torch.cuda.is_bf16_supported() else "float16"
    load_in_4bit: bool = True


default_config = RAGConfig()

# Aliases tương thích ngược
CHROMA_PERSIST_DIR = default_config.vector_store_path
EMBEDDING_MODEL = default_config.embedding_model_name
DATA_RAW_DIR = default_config.data_raw_dir
EVAL_DATA_PATH = default_config.eval_data_path

