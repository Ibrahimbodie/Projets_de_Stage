from __future__ import annotations

from dataclasses import dataclass
from typing import List

from langchain_ollama import OllamaLLM
from langchain_core.documents import Document
from langchain_core.prompts import PromptTemplate

FALLBACK_MESSAGE = (
    "Information non trouvée dans "
    "les règlements officiels."
)


# --------------------------------------------------
# CONFIGURATION LLM
# --------------------------------------------------

@dataclass
class ChainConfig:

    llm_model: str = "qwen2.5:3b"

    request_timeout: float = 300.0

    num_predict: int = 512


# --------------------------------------------------
# PROMPT
# --------------------------------------------------

def build_prompt() -> PromptTemplate:

    template = (
        "Tu es un assistant académique intelligent "
        "du Master SIM à VNU-IS.\n\n"

        "Ton rôle est d'aider les étudiants "
        "à comprendre facilement les règlements "
        "académiques.\n\n"

        "Tu dois répondre en français SIMPLE, "
        "naturel et pédagogique.\n\n"

        "RÈGLES IMPORTANTES :\n"
        "- Utilise uniquement les informations "
        "présentes dans le contexte.\n"
        "- Reformule avec tes propres mots.\n"
        "- N'affiche jamais directement "
        "les paragraphes du document.\n"
        "- Explique les règles comme à un étudiant.\n"
        "- Donne des réponses claires "
        "et faciles à comprendre.\n"
        "- Si l'information n'existe pas dans "
        "le contexte, répond exactement :\n"
        "'Information non trouvée dans "
        "les règlements officiels.'\n\n"

        "Contexte :\n"
        "{context}\n\n"

        "Question de l'étudiant :\n"
        "{question}\n\n"

        "Réponse pédagogique :"
    )

    return PromptTemplate.from_template(template)


# --------------------------------------------------
# Filter conversationnel
# --------------------------------------------------



# --------------------------------------------------
# FORMAT CONTEXTE
# --------------------------------------------------

def format_context(
    documents: List[Document],
) -> str:

    parts = []

    for doc in documents:

        page = doc.metadata.get("page")

        page_label = (
            f"Page {page + 1}"
            if isinstance(page, int)
            else "Page inconnue"
        )

        clean_text = " ".join(
            doc.page_content.split()
        )

        parts.append(
            f"[{page_label}]\n{clean_text}"
        )

    return "\n\n".join(parts)


# --------------------------------------------------
# FORMAT SOURCES
# --------------------------------------------------

def format_sources(
    documents: List[Document],
) -> List[str]:

    pages = []

    for doc in documents:

        page = doc.metadata.get("page")

        if isinstance(page, int):

            pages.append(page + 1)

    pages = sorted(set(pages))

    return [
        f"Page {page}"
        for page in pages
    ]


# --------------------------------------------------
# GENERATION REPONSE
# --------------------------------------------------

def generate_answer(
    question: str,
    documents: List[Document],
    config: ChainConfig,
) -> str:

    if not documents:

        return FALLBACK_MESSAGE

    prompt = build_prompt()

    context = format_context(documents)

    llm = OllamaLLM(
        model=config.llm_model,
        temperature=0.2,
        num_predict=config.num_predict,
        sync_client_kwargs={
            "timeout": config.request_timeout
        },
    )

    try:

        answer = llm.invoke(
            prompt.format(
                context=context,
                question=question,
            )
        )

        if isinstance(answer, str):

            clean_answer = answer.strip()

            if clean_answer:

                return clean_answer

        return (
            "Je n'ai pas réussi à générer "
            "une réponse claire."
        )

    except Exception as exc:

        details = (
            str(exc).strip()
            or exc.__class__.__name__
        )

        raise RuntimeError(
            "Erreur génération Ollama : "
            f"{details}"
        ) from exc