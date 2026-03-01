from pathlib import Path

from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_ollama import OllamaEmbeddings

DATA_PATH = Path("../data/Contrat_apprentissage_SIM_P27.pdf")
DB_PATH = Path("../vectorstore")
EMBEDDING_MODEL = "qwen3-embedding"
CHUNK_SIZE = 800
CHUNK_OVERLAP = 150


def create_vectorstore() -> None:
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
    db = FAISS.from_documents(texts, embeddings)

    DB_PATH.mkdir(parents=True, exist_ok=True)
    db.save_local(str(DB_PATH))
    print("Index sauvegardé avec succès.")


if __name__ == "__main__":
    create_vectorstore()