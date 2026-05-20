from __future__ import annotations

import streamlit as st

from llm_chain import (
    ChainConfig,
    format_sources,
    generate_answer,
)

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


# Chargement unique du vectorstore
db = get_vectorstore()

# --------------------------------------------------
# SESSION STATE
# --------------------------------------------------

if "messages" not in st.session_state:

    st.session_state.messages = []

# --------------------------------------------------
# HEADER
# --------------------------------------------------

st.title(
    "🎓 Assistant Virtuel Académique - Master SIM"
)

st.markdown(
    """
Cet assistant répond uniquement à partir
des règlements académiques officiels
du Master SIM.

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
        value=2,
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
        ],
        index=0,
    )

    st.markdown("---")

    # --------------------------------------------------
    # UPLOAD DOCUMENTS
    # --------------------------------------------------

    st.subheader("📂 Documents utilisateur")

    uploaded_files = st.file_uploader(
        "Ajoutez vos documents",
        type=["pdf", "txt", "docx"],
        accept_multiple_files=True,
    )

    if uploaded_files:

        st.success(
            f"{len(uploaded_files)} document(s) chargé(s)."
        )

        for file in uploaded_files:

            st.write(f"• {file.name}")

    st.markdown("---")

    # --------------------------------------------------
    # RESET CHAT
    # --------------------------------------------------

    if st.button("🗑️ Effacer la conversation"):

        st.session_state.messages = []

        st.rerun()

# --------------------------------------------------
# HISTORIQUE CHAT
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

    # --------------------------------------------------
    # PETITES CONVERSATIONS
    # --------------------------------------------------

    small_talk = {

        "merci": "Je vous en prie 😊",

        "merci beaucoup": "Avec plaisir 😊",

        "bonjour": (
            "Bonjour 👋 "
            "Comment puis-je vous aider "
            "concernant les règlements académiques ?"
        ),

        "salut": (
            "Salut 👋 "
            "Comment puis-je vous aider ?"
        ),

        "ok": "Très bien 👍",

        "d'accord": "Parfait 👍",

        "au revoir": "Au revoir 👋",
    }

    normalized_question = (
        question.lower().strip()
    )

    # --------------------------------------------------
    # REPONSE PETITES CONVERSATIONS
    # --------------------------------------------------

    if normalized_question in small_talk:

        response = small_talk[
            normalized_question
        ]

        # Message utilisateur
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

            st.markdown(response)

        st.session_state.messages.append(
            {
                "role": "assistant",
                "content": response,
            }
        )

        st.stop()

    # --------------------------------------------------
    # MESSAGE UTILISATEUR
    # --------------------------------------------------

    st.session_state.messages.append(
        {
            "role": "user",
            "content": question,
        }
    )

    with st.chat_message("user"):

        st.markdown(question)

    # --------------------------------------------------
    # REPONSE ASSISTANT
    # --------------------------------------------------

    with st.chat_message("assistant"):

        with st.spinner(
            "Recherche des informations en cours..."
        ):

            try:

                # Configuration retrieval
                retrieval_config = RetrievalConfig(
                    top_k=top_k,
                    min_similarity=min_similarity,
                )

                # Recherche documents
                documents, scores = (
                    retrieve_documents(
                        question,
                        retrieval_config,
                    )
                )

                # Génération réponse
                answer = generate_answer(
                    question,
                    documents,
                    ChainConfig(
                        llm_model=llm_model,
                        request_timeout=300,
                    ),
                )

                # Sources
                sources = format_sources(
                    documents
                )

                # Affichage réponse
                st.markdown(answer)

                # --------------------------------------------------
                # SOURCES
                # --------------------------------------------------

                if sources:

                    st.markdown(
                        "### 📚 Sources"
                    )

                    for source in sources:

                        st.markdown(
                            f"- {source}"
                        )

                # --------------------------------------------------
                # SAUVEGARDE HISTORIQUE
                # --------------------------------------------------

                final_answer = answer

                if sources:

                    final_answer += (
                        "\n\nSources:\n"
                    )

                    final_answer += "\n".join(
                        [
                            f"- {s}"
                            for s in sources
                        ]
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

                st.error(
                    f"Erreur inattendue : {exc}"
                )