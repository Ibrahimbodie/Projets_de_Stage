from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

from langchain_community.vectorstores import FAISS
from langchain_ollama import OllamaEmbeddings
from langchain_core.documents import Document

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DB_PATH = PROJECT_ROOT / "vectorstore"

EMBEDDING_MODEL = "qwen3-embedding"


# --------------------------------------------------
# CONFIGURATION RETRIEVAL
# --------------------------------------------------

@dataclass
class RetrievalConfig:

    top_k: int = 6

    max_distance: float = 1.5


# --------------------------------------------------
# LOAD VECTORSTORE
# --------------------------------------------------

def load_vectorstore() -> FAISS:

    if not DB_PATH.exists():

        raise FileNotFoundError(
            f"Index vectoriel introuvable : "
            f"{DB_PATH}"
        )

    embeddings = OllamaEmbeddings(
        model=EMBEDDING_MODEL
    )

    return FAISS.load_local(
        str(DB_PATH),
        embeddings,
        allow_dangerous_deserialization=True,
    )


# --------------------------------------------------
# DISTANCE -> SIMILARITE
# --------------------------------------------------

def distance_to_similarity(
    distance: float,
) -> float:

    similarity = 1 / (1 + distance)

    return round(similarity, 4)


# --------------------------------------------------
# RETRIEVE DOCUMENTS
# --------------------------------------------------

def retrieve_documents(
    query: str,
    config: RetrievalConfig,
) -> Tuple[List[Document], List[float]]:

    db = load_vectorstore()

    results = db.similarity_search_with_score(
        query,
        k=config.top_k,
    )

    if not results:

        return [], []

    filtered_docs: List[Document] = []

    filtered_scores: List[float] = []

    for doc, distance in results:

        if distance <= config.max_distance:

            similarity = (
                distance_to_similarity(
                    distance
                )
            )

            filtered_docs.append(doc)

            filtered_scores.append(
                similarity
            )

    return filtered_docs, filtered_scores
