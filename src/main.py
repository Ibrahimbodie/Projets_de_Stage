from __future__ import annotations

import streamlit as st

from llm_chain import ChainConfig, format_sources, generate_answer
from retriever import RetrievalConfig, retrieve_documents

def run_app() -> None:
    st.set_page_config(page_title="Assistant Master SIM", layout="centered")
    st.title("Assistant académique Master SIM")
    st.write(
        "Je réponds uniquement à partir du contexte officiel du Master SIM. "
        "Si la réponse n'est pas dans le document, je l'indiquerai clairement."
    )

    with st.sidebar:
        st.header("Réglages")
        top_k = st.slider("Top K", min_value=1, max_value=8, value=4)
        min_similarity = st.slider(
            "Seuil de similarité (0-1)",
            min_value=0.0,
            max_value=1.0,
            value=0.3,
            step=0.05,
        )
        llm_model = st.text_input("Modèle LLM", value="qwen3.5:9b")

    question = st.text_area("Question de l’étudiant", placeholder="Tape la question ici")

    if st.button("Répondre", type="primary"):
        if not question.strip():
            st.warning("Merci de saisir une question.")
            return

        try:
            retrieval_config = RetrievalConfig(top_k=top_k, min_similarity=min_similarity)
            documents, _ = retrieve_documents(question, retrieval_config)
            answer = generate_answer(question, documents, ChainConfig(llm_model=llm_model))
            sources = format_sources(documents)
        except FileNotFoundError as exc:
            st.error(str(exc))
            return
        except ConnectionError:
            st.error(
                "Connexion Ollama impossible. Vérifie que Ollama est démarré "
                "et que les modèles nécessaires sont disponibles."
            )
            return
        except Exception as exc:
            st.error(f"Erreur inattendue: {exc}")
            return

        st.subheader("Réponse")
        st.write(answer)

        if sources:
            st.subheader("Sources")
            st.write(", ".join(sources))


if __name__ == "__main__":
    run_app()
