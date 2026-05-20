from pathlib import Path

from langchain_community.document_loaders import (
    PyPDFLoader,
    TextLoader,
    Docx2txtLoader,
)

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_ollama import OllamaEmbeddings

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_PATH = PROJECT_ROOT / "data"

DB_PATH = PROJECT_ROOT / "vectorstore"

EMBEDDING_MODEL = "qwen3-embedding"

CHUNK_SIZE = 800
CHUNK_OVERLAP = 150


def load_document(file_path: Path):

    suffix = file_path.suffix.lower()

    if suffix == ".pdf":

        print(f"Chargement PDF : {file_path.name}")

        loader = PyPDFLoader(str(file_path))

    elif suffix == ".txt":

        print(f"Chargement TXT : {file_path.name}")

        loader = TextLoader(
            str(file_path),
            encoding="utf-8",
        )

    elif suffix == ".docx":

        print(f"Chargement DOCX : {file_path.name}")

        loader = Docx2txtLoader(str(file_path))

    else:

        print(f"Format ignoré : {file_path.name}")

        return []

    return loader.load()


def create_vectorstore() -> None:

    if not DATA_PATH.exists():

        raise FileNotFoundError(
            f"Dossier introuvable : {DATA_PATH}"
        )

    all_documents = []

    # Parcours de tous les fichiers
    for file_path in DATA_PATH.iterdir():

        if file_path.is_file():

            try:

                documents = load_document(file_path)

                all_documents.extend(documents)

            except Exception as exc:

                print(
                    f"Erreur avec {file_path.name} : {exc}"
                )

    if not all_documents:

        raise ValueError(
            "Aucun document valide trouvé."
        )

    print("Découpage en chunks...")

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
    )

    texts = text_splitter.split_documents(
        all_documents
    )

    print(
        f"Création des embeddings avec "
        f"{EMBEDDING_MODEL}..."
    )

    embeddings = OllamaEmbeddings(
        model=EMBEDDING_MODEL
    )

    print("Création de l’index FAISS...")

    try:

        db = FAISS.from_documents(
            texts,
            embeddings,
        )

    except ConnectionError as exc:

        raise RuntimeError(
            "Impossible de contacter Ollama. "
            "Démarre Ollama puis relance "
            "l'ingestion."
        ) from exc

    DB_PATH.mkdir(
        parents=True,
        exist_ok=True,
    )

    db.save_local(str(DB_PATH))

    print("Index sauvegardé avec succès.")


if __name__ == "__main__":

    try:

        create_vectorstore()

    except Exception as exc:

        print(f"Erreur ingestion : {exc}")