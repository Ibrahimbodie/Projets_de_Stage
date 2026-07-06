from __future__ import annotations

import sys
import time
from pathlib import Path

import streamlit as st

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from config import load_config
from llm_chain import format_sources, generate_answer
from retriever import retrieve_documents


st.set_page_config(
    page_title="Assistant Master SIM",
    page_icon="🎓",
    layout="wide",
)


@st.cache_resource
def get_config():
    return load_config()


cfg = get_config()

OLLAMA_CHAT_MODELS = ["qwen2.5:3b", "llama3.2", "mistral:7b", "qwen3.5:9b"]
OPENROUTER_CHAT_MODELS = [
    "google/gemini-2.5-flash",
    "google/gemini-2.5-pro",
    "deepseek/deepseek-chat",
    "qwen/qwen2.5-72b-instruct",
    "meta-llama/llama-3.3-70b-instruct",
]
OLLAMA_EMBED_MODELS = ["nomic-embed-text", "qwen3-embedding", "nomic-embed-text-v2-moe"]
HF_EMBED_MODELS = [
    "BAAI/bge-m3",
    "intfloat/multilingual-e5-large",
    "jinaai/jina-embeddings-v3",
]


def has_vectorstore(provider: str, model: str) -> bool:
    slug = model.replace("/", "_")
    return (ROOT / "vectorstore" / slug).exists()

if "messages" not in st.session_state:
    st.session_state.messages = []

st.title("🎓 Assistant Virtuel Académique - Master SIM")
st.markdown(
    "Cet assistant répond uniquement à partir "
    "des règlements académiques officiels du Master SIM.\n\n"
    "Posez votre question en langage naturel."
)

with st.sidebar:
    st.header("⚙️ Paramètres")

    st.subheader("🤖 LLM")
    llm_provider = st.selectbox(
        "Provider", ["openrouter", "ollama"],
        index=0 if cfg.llm.provider == "openrouter" else 1,
        key="llm_provider",
    )

    llm_models = (
        OPENROUTER_CHAT_MODELS if llm_provider == "openrouter"
        else OLLAMA_CHAT_MODELS
    )
    default_llm = (
        cfg.llm.model if cfg.llm.model in llm_models
        else llm_models[0]
    )
    llm_model = st.selectbox(
        "Modèle", llm_models,
        index=llm_models.index(default_llm),
        key="llm_model",
    )

    st.subheader("🔤 Embedding")
    emb_provider = st.selectbox(
        "Provider", ["huggingface", "ollama"],
        index=0 if cfg.embedding.provider == "huggingface" else 1,
        key="emb_provider",
    )

    emb_models = (
        HF_EMBED_MODELS if emb_provider == "huggingface"
        else OLLAMA_EMBED_MODELS
    )
    default_emb = (
        cfg.embedding.model if cfg.embedding.model in emb_models
        else emb_models[0]
    )
    emb_model = st.selectbox(
        "Modèle", emb_models,
        index=emb_models.index(default_emb),
        key="emb_model",
    )

    if not has_vectorstore(emb_provider, emb_model):
        st.warning(
            f"Aucun index FAISS pour {emb_provider}:{emb_model}. "
            f"Lancez ingest.py ou changez de modèle."
        )

    top_k = st.slider(
        "Top K",
        min_value=1,
        max_value=10,
        value=cfg.retrieval.top_k,
    )

    max_distance = st.slider(
        "Distance maximale",
        min_value=0.5,
        max_value=3.0,
        value=cfg.retrieval.max_distance,
        step=0.1,
    )

    st.markdown("---")
    st.subheader("📂 Documents utilisateur")

    uploaded_files = st.file_uploader(
        "Ajoutez vos documents",
        type=["pdf", "txt", "docx"],
        accept_multiple_files=True,
    )

    if uploaded_files:
        st.success(f"{len(uploaded_files)} document(s) chargé(s).")
        for file in uploaded_files:
            st.write(f"• {file.name}")

    st.markdown("---")

    if st.button("🗑️ Effacer la conversation"):
        st.session_state.messages = []
        st.rerun()

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

question = st.chat_input(
    "Posez votre question sur les règlements académiques..."
)

if question:
    small_talk = {
        "merci": "Je vous en prie 😊",
        "merci beaucoup": "Avec plaisir 😊",
        "bonjour": "Bonjour 👋 Comment puis-je vous aider concernant les règlements académiques ?",
        "salut": "Salut 👋 Comment puis-je vous aider ?",
        "ok": "Très bien 👍",
        "d'accord": "Parfait 👍",
        "au revoir": "Au revoir 👋",
    }

    normalized_question = question.lower().strip()

    if normalized_question in small_talk:
        response = small_talk[normalized_question]
        st.session_state.messages.append({
            "role": "user",
            "content": question,
        })
        with st.chat_message("user"):
            st.markdown(question)
        with st.chat_message("assistant"):
            st.markdown(response)
        st.session_state.messages.append({
            "role": "assistant",
            "content": response,
        })
        st.stop()

    st.session_state.messages.append({
        "role": "user",
        "content": question,
    })

    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.spinner("Recherche des informations..."):
            try:
                from config import RetrievalConfig, LLMConfig, EmbeddingConfig

                retrieval_cfg = RetrievalConfig(
                    top_k=top_k,
                    max_distance=max_distance,
                )
                llm_cfg = LLMConfig(
                    provider=st.session_state.llm_provider,
                    model=st.session_state.llm_model,
                    temperature=cfg.llm.temperature,
                    num_predict=cfg.llm.num_predict,
                    request_timeout=cfg.llm.request_timeout,
                )
                embedding_cfg = EmbeddingConfig(
                    provider=st.session_state.emb_provider,
                    model=st.session_state.emb_model,
                    device=cfg.embedding.device,
                )

                start_time = time.time()
                documents, scores = retrieve_documents(
                    question,
                    retrieval_cfg,
                    embedding_cfg,
                )
                answer = generate_answer(
                    question,
                    documents,
                    llm_cfg,
                )
                end_time = time.time()
                response_time = end_time - start_time

                sources = format_sources(documents)

                st.markdown(answer)
                st.markdown("---")

                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("⏱️ Temps réponse", f"{response_time:.2f} s")
                with col2:
                    st.metric("📄 Chunks", len(documents))
                with col3:
                    if scores:
                        avg_score = sum(scores) / len(scores)
                        confidence = (
                            "Élevée" if avg_score > 0.7
                            else "Moyenne" if avg_score > 0.5
                            else "Faible"
                        )
                        st.metric("🎯 Confiance", confidence)

                if sources:
                    with st.expander("📚 Sources utilisées"):
                        for idx, source in enumerate(sources):
                            st.markdown(f"### Source {idx+1}")
                            st.write(source)
                            if idx < len(scores):
                                st.write(
                                    f"Score similarité : {scores[idx]:.4f}"
                                )

                with st.expander("🧠 Contexte récupéré"):
                    for idx, doc in enumerate(documents):
                        st.markdown(f"### Chunk {idx+1}")
                        st.write(doc.page_content[:700])
                        st.markdown("---")

                final_answer = answer
                if sources:
                    final_answer += "\n\nSources:\n" + "\n".join(
                        [f"- {s}" for s in sources]
                    )

                st.session_state.messages.append({
                    "role": "assistant",
                    "content": final_answer,
                })

            except FileNotFoundError as exc:
                st.error(str(exc))
            except ConnectionError:
                st.error(
                    "Impossible de contacter Ollama.\n\n"
                    "Vérifie que Ollama est lancé."
                )
            except Exception as exc:
                st.error(f"Erreur inattendue : {exc}")
