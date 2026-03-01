from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Tuple

from langchain_community.vectorstores import FAISS
from langchain_ollama import OllamaEmbeddings
from langchain_core.documents import Document

DB_PATH = Path("../vectorstore")
EMBEDDING_MODEL = "qwen3-embedding"


@dataclass
class RetrievalConfig:
    top_k: int = 4
    min_similarity: float = 0.3


def _normalize_scores(raw_scores: Iterable[float]) -> List[float]:
    scores = list(raw_scores)
    if not scores:
        return []
    min_score = min(scores)
    max_score = max(scores)
    if max_score == min_score:
        return [1.0 for _ in scores]
    return [(score - min_score) / (max_score - min_score) for score in scores]


def load_vectorstore() -> FAISS:
    embeddings = OllamaEmbeddings(model=EMBEDDING_MODEL)
    return FAISS.load_local(
        str(DB_PATH),
        embeddings,
        allow_dangerous_deserialization=True,
    )


def retrieve_documents(
    query: str,
    config: RetrievalConfig,
) -> Tuple[List[Document], List[float]]:
    db = load_vectorstore()
    results = db.similarity_search_with_score(query, k=config.top_k)
    if not results:
        return [], []

    documents, raw_scores = zip(*results)
    normalized = _normalize_scores(raw_scores)

    filtered_docs: List[Document] = []
    filtered_scores: List[float] = []
    for doc, score in zip(documents, normalized):
        if score >= config.min_similarity:
            filtered_docs.append(doc)
            filtered_scores.append(score)

    return filtered_docs, filtered_scores