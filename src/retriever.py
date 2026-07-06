from __future__ import annotations

from typing import List, Tuple

from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document

from config import PROJECT_ROOT, load_config, RetrievalConfig, EmbeddingConfig
from embeddings import create_embeddings


def load_vectorstore(embedding_cfg: EmbeddingConfig | None = None):
    if embedding_cfg is None:
        config = load_config()
        embedding_cfg = config.embedding
        db_path = config.vectorstore_path
    else:
        config = load_config()
        if (
            embedding_cfg.provider == config.embedding.provider
            and embedding_cfg.model == config.embedding.model
        ):
            db_path = config.vectorstore_path
        else:
            model_slug = embedding_cfg.model.replace("/", "_")
            db_path = PROJECT_ROOT / "vectorstore" / model_slug

    if not db_path.exists():
        raise FileNotFoundError(
            f"Index vectoriel introuvable : {db_path}\n"
            f"Lancez d'abord ingest.py pour le modèle "
            f"{embedding_cfg.model}."
        )

    embeddings = create_embeddings(embedding_cfg)
    return FAISS.load_local(
        str(db_path),
        embeddings,
        allow_dangerous_deserialization=True,
    )


def distance_to_similarity(distance: float) -> float:
    similarity = 1 / (1 + distance)
    return round(similarity, 4)


def retrieve_documents(
    query: str,
    config: RetrievalConfig,
    embedding_cfg: EmbeddingConfig | None = None,
) -> Tuple[List[Document], List[float]]:
    db = load_vectorstore(embedding_cfg)
    results = db.similarity_search_with_score(query, k=config.top_k)

    if not results:
        return [], []

    filtered_docs: List[Document] = []
    filtered_scores: List[float] = []

    for doc, distance in results:
        if distance <= config.max_distance:
            similarity = distance_to_similarity(distance)
            filtered_docs.append(doc)
            filtered_scores.append(similarity)

    return filtered_docs, filtered_scores
