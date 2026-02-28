from langchain_community.vectorstores import FAISS
from langchain_ollama import OllamaEmbeddings, OllamaLLM
from langchain.chains import RetrievalQA

DB_PATH = "../vectorstore"

def load_qa_chain():
    embeddings = OllamaEmbeddings(model="qwen3-embedding")

    db = FAISS.load_local(
        DB_PATH,
        embeddings,
        allow_dangerous_deserialization=True
    )

    retriever = db.as_retriever(search_kwargs={"k": 3})

    llm = OllamaLLM(model="qwen3.5:27b")

    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        return_source_documents=True
    )

    return qa_chain


def main():
    qa = load_qa_chain()

    print("Assistant prêt. Tape 'exit' pour quitter.\n")

    while True:
        query = input("Question: ")

        if query.lower() == "exit":
            break

        result = qa.invoke({"query": query})
        print("\nRéponse:\n", result["result"])
        print("-" * 60)


if __name__ == "__main__":
    main()