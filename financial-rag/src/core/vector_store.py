import os
import re
from typing import List, Dict, Any
import chromadb
from chromadb.utils import embedding_functions
from rank_bm25 import BM25Okapi

try:
    from src.config import CHROMA_PERSIST_DIR, EMBEDDING_MODEL
except ImportError:
    from src.config import default_config
    CHROMA_PERSIST_DIR = default_config.vector_store_path
    EMBEDDING_MODEL = default_config.embedding_model_name


def simple_tokenize(text: str) -> List[str]:
    return re.findall(r"\w+", text.lower())


class HybridChromaStore:
    def __init__(self, collection_name: str = "financial_reports"):
        os.makedirs(CHROMA_PERSIST_DIR, exist_ok=True)
        self.client = chromadb.PersistentClient(path=CHROMA_PERSIST_DIR)
        self.embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name=EMBEDDING_MODEL
        )
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            embedding_function=self.embedding_fn
        )
        self.bm25: BM25Okapi = None
        self.corpus_chunks: List[Dict[str, Any]] = []

    def add_chunks(self, chunks: List[Dict[str, Any]]):
        if not chunks:
            return

        ids = []
        documents = []
        metadatas = []

        for idx, c in enumerate(chunks):
            meta = c["metadata"].copy()
            page = meta.get("page_number", meta.get("page", 0))
            company = meta.get("company_name", meta.get("company", "corp"))

            chunk_id = f"{company}_p{page}_idx{idx}_{abs(hash(c['augmented_content'])) % 100000}"
            ids.append(chunk_id)
            documents.append(c["augmented_content"])

            cleaned_meta = {}
            for k, v in meta.items():
                if v is not None and isinstance(v, (str, int, float, bool)):
                    cleaned_meta[k] = v
                else:
                    cleaned_meta[k] = str(v) if v is not None else ""
            cleaned_meta["original_content"] = c.get("original_content", c.get("content", ""))
            metadatas.append(cleaned_meta)

        self.collection.add(
            ids=ids,
            documents=documents,
            metadatas=metadatas
        )

        self.corpus_chunks.extend(chunks)
        tokenized_corpus = [simple_tokenize(c["augmented_content"]) for c in self.corpus_chunks]
        self.bm25 = BM25Okapi(tokenized_corpus)

    def search(self, query: str, top_k: int = 4) -> List[Dict[str, Any]]:
        chroma_res = self.collection.query(
            query_texts=[query],
            n_results=min(top_k * 4, max(self.collection.count(), 1))
        )

        dense_ranks = {}
        dense_docs = {}
        if chroma_res and chroma_res["documents"] and chroma_res["documents"][0]:
            for rank, doc_id in enumerate(chroma_res["ids"][0]):
                dense_ranks[doc_id] = rank + 1
                dense_docs[doc_id] = {
                    "augmented_content": chroma_res["documents"][0][rank],
                    "original_content": chroma_res["metadatas"][0][rank].get("original_content", ""),
                    "metadata": chroma_res["metadatas"][0][rank]
                }

        sparse_ranks = {}
        if self.bm25 and self.corpus_chunks:
            query_tokens = simple_tokenize(query)
            bm25_scores = self.bm25.get_scores(query_tokens)
            top_bm25_indices = sorted(range(len(bm25_scores)), key=lambda i: bm25_scores[i], reverse=True)[:top_k * 2]

            for rank, idx in enumerate(top_bm25_indices):
                if bm25_scores[idx] > 0:
                    c = self.corpus_chunks[idx]
                    content_key = c["augmented_content"]
                    sparse_ranks[content_key] = rank + 1

        k_const = 60
        combined_scores = {}

        for doc_id, doc_data in dense_docs.items():
            dense_rank = dense_ranks.get(doc_id, 100)
            content_key = doc_data["augmented_content"]
            sparse_rank = sparse_ranks.get(content_key, 100)

            rrf_score = (1.0 / (k_const + dense_rank)) + (1.0 / (k_const + sparse_rank))
            combined_scores[doc_id] = (rrf_score, doc_data)

        sorted_results = sorted(combined_scores.values(), key=lambda x: x[0], reverse=True)
        return [item[1] for item in sorted_results[:top_k]]

