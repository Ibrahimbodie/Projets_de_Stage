from pathlib import Path

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_ollama import OllamaEmbeddings

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = PROJECT_ROOT / "data" / "Contrat_apprentissage_SIM.pdf"
DB_PATH = PROJECT_ROOT / "vectorstore"
EMBEDDING_MODEL = "qwen3-embedding"
CHUNK_SIZE = 800
CHUNK_OVERLAP = 150


def create_vectorstore() -> None:
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"PDF introuvable: {DATA_PATH}")

    print("Chargement du PDF...")
    loader = PyPDFLoader(str(DATA_PATH))
    documents = loader.load()

    print("Découpage en chunks...")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
    )
    texts = text_splitter.split_documents(documents)

    print(f"Création des embeddings avec {EMBEDDING_MODEL}...")
    embeddings = OllamaEmbeddings(model=EMBEDDING_MODEL)

    print("Création de l’index FAISS...")
    try:
        db = FAISS.from_documents(texts, embeddings)
    except ConnectionError as exc:
        raise RuntimeError(
            "Impossible de contacter Ollama. Démarre Ollama puis relance l'ingestion."
        ) from exc

    DB_PATH.mkdir(parents=True, exist_ok=True)
    db.save_local(str(DB_PATH))
    print("Index sauvegardé avec succès.")


if __name__ == "__main__":
    try:
        create_vectorstore()
    except Exception as exc:
        print(f"Erreur ingestion: {exc}")
