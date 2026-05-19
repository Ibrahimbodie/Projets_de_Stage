from __future__ import annotations

import streamlit as st

from llm_chain import ChainConfig, format_sources, generate_answer
from retriever import (
    RetrievalConfig,
    retrieve_documents,
    load_vectorstore,
)

# --------------------------------------------------
# CONFIGURATION PAGE
# --------------------------------------------------

st.set_page_config(
    page_title="Assistant Master SIM",
    page_icon="🎓",
    layout="wide",
)

# --------------------------------------------------
# CACHE VECTORSTORE
# --------------------------------------------------

@st.cache_resource
def get_vectorstore():
    return load_vectorstore()


# Charger une seule fois au démarrage
db = get_vectorstore()

# --------------------------------------------------
# SESSION STATE
# --------------------------------------------------

if "messages" not in st.session_state:
    st.session_state.messages = []

# --------------------------------------------------
# HEADER
# --------------------------------------------------

st.title("🎓 Assistant Virtuel Académique - Master SIM")

st.markdown(
    """
Cet assistant répond uniquement à partir des règlements académiques officiels du Master SIM.

Posez votre question en langage naturel.
"""
)

# --------------------------------------------------
# SIDEBAR
# --------------------------------------------------

with st.sidebar:

    st.header("⚙️ Paramètres")

    top_k = st.slider(
        "Top K",
        min_value=1,
        max_value=10,
        value=4,
    )

    min_similarity = st.slider(
        "Seuil de similarité",
        min_value=0.0,
        max_value=1.0,
        value=0.3,
        step=0.05,
    )

    llm_model = st.selectbox(
        "Modèle Ollama",
        [
            "qwen2.5:3b",
            "gemma3:4b",
            "qwen3.5:9b",
        ],
        index=0,
    )

    st.markdown("---")

    if st.button("🗑️ Effacer la conversation"):
        st.session_state.messages = []
        st.rerun()

# --------------------------------------------------
# AFFICHAGE HISTORIQUE
# --------------------------------------------------

for message in st.session_state.messages:

    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# --------------------------------------------------
# INPUT UTILISATEUR
# --------------------------------------------------

question = st.chat_input(
    "Posez votre question sur les règlements académiques..."
)

# --------------------------------------------------
# TRAITEMENT QUESTION
# --------------------------------------------------

if question:

    # Affichage message utilisateur
    st.session_state.messages.append(
        {
            "role": "user",
            "content": question,
        }
    )

    with st.chat_message("user"):
        st.markdown(question)

    # Réponse assistant
    with st.chat_message("assistant"):

        with st.spinner("Recherche des informations en cours..."):

            try:

                # Retrieval config
                retrieval_config = RetrievalConfig(
                    top_k=top_k,
                    min_similarity=min_similarity,
                )

                # Recherche documents
                documents, scores = retrieve_documents(
                    question,
                    retrieval_config,
                )

                # Génération réponse
                answer = generate_answer(
                    question,
                    documents,
                    ChainConfig(
                        llm_model=llm_model,
                        request_timeout=180,
                    ),
                )

                # Sources
                sources = format_sources(documents)

                # Affichage réponse
                st.markdown(answer)

                # Affichage sources
                if sources:

                    st.markdown("### 📚 Sources")

                    for source in sources:
                        st.markdown(f"- {source}")

                # Sauvegarde historique
                final_answer = answer

                if sources:
                    final_answer += "\n\nSources:\n"
                    final_answer += "\n".join(
                        [f"- {s}" for s in sources]
                    )

                st.session_state.messages.append(
                    {
                        "role": "assistant",
                        "content": final_answer,
                    }
                )

            except FileNotFoundError as exc:

                st.error(str(exc))

            except ConnectionError:

                st.error(
                    "Impossible de contacter Ollama.\n\n"
                    "Vérifie que Ollama est lancé."
                )

            except Exception as exc:

                st.error(f"Erreur inattendue : {exc}")