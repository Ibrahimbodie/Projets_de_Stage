from __future__ import annotations

from pathlib import Path

from langchain_community.document_loaders import (
    PyPDFLoader,
    TextLoader,
    Docx2txtLoader,
)
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS

from config import PROJECT_ROOT, load_config
from embeddings import create_embeddings

DATA_PATH = PROJECT_ROOT / "data"

CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200


def load_document(file_path: Path):
    suffix = file_path.suffix.lower()

    if suffix == ".pdf":
        print(f"Chargement PDF : {file_path.name}")
        loader = PyPDFLoader(str(file_path))
    elif suffix == ".txt":
        print(f"Chargement TXT : {file_path.name}")
        loader = TextLoader(str(file_path), encoding="utf-8")
    elif suffix == ".docx":
        print(f"Chargement DOCX : {file_path.name}")
        loader = Docx2txtLoader(str(file_path))
    else:
        print(f"Format ignoré : {file_path.name}")
        return []

    return loader.load()


def create_vectorstore() -> None:
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Dossier introuvable : {DATA_PATH}")

    all_documents = []

    for file_path in DATA_PATH.iterdir():
        if file_path.is_file():
            try:
                documents = load_document(file_path)
                all_documents.extend(documents)
            except Exception as exc:
                print(f"Erreur avec {file_path.name} : {exc}")

    if not all_documents:
        raise ValueError("Aucun document valide trouvé.")

    print("Découpage en chunks...")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
    )
    texts = text_splitter.split_documents(all_documents)

    config = load_config()
    model_slug = config.embedding.model.replace("/", "_")
    db_path = PROJECT_ROOT / "vectorstore" / model_slug

    print(
        f"Création des embeddings avec "
        f"{config.embedding.provider}:{config.embedding.model}..."
    )
    embeddings = create_embeddings(config.embedding)

    print("Création de l'index FAISS...")
    try:
        db = FAISS.from_documents(texts, embeddings)
    except ConnectionError as exc:
        raise RuntimeError(
            "Impossible de contacter le fournisseur d'embedding. "
            "Vérifiez la configuration."
        ) from exc

    db_path.mkdir(parents=True, exist_ok=True)
    db.save_local(str(db_path))
    print(f"Index sauvegardé : {db_path}")


if __name__ == "__main__":
    try:
        create_vectorstore()
    except Exception as exc:
        print(f"Erreur ingestion : {exc}")
