from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_ollama import OllamaEmbeddings

DATA_PATH = "../data/Contrat_apprentissage_SIM_P27.pdf"
DB_PATH = "../vectorstore"

def create_vectorstore():
    print("Chargement du PDF...")
    loader = PyPDFLoader(DATA_PATH)
    documents = loader.load()

    print("Découpage en chunks...")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=150
    )
    texts = text_splitter.split_documents(documents)

    print("Création des embeddings avec qwen3-embedding...")
    embeddings = OllamaEmbeddings(model="qwen3-embedding")

    print("Création de l’index FAISS...")
    db = FAISS.from_documents(texts, embeddings)

    db.save_local(DB_PATH)
    print("Index sauvegardé avec succès.")

if __name__ == "__main__":
    create_vectorstore()