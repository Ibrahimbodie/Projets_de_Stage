from __future__ import annotations

from dataclasses import dataclass
from typing import List

from langchain_ollama import OllamaLLM
from langchain_core.documents import Document
from langchain_core.prompts import PromptTemplate

FALLBACK_MESSAGE = "Information non trouvée dans les règlements officiels."


@dataclass
class ChainConfig:
    llm_model: str = "qwen3.5:9b"
    request_timeout: float = 120.0
    num_predict: int = 256


def build_prompt() -> PromptTemplate:
    template = (
        "Tu es un assistant académique officiel du programme Master SIM à VNU-IS.\n"
        "Ta mission est de répondre aux questions des étudiants en utilisant exclusivement les informations contenues dans le contexte fourni.\n\n"
        "Règles strictes :\n"
        "- N'utilise que les informations présentes dans le contexte.\n"
        "- N'ajoute aucune information provenant de tes connaissances générales.\n"
        "- Si l'information n'est pas explicitement présente dans le contexte, réponds exactement :\n"
        f"  '{FALLBACK_MESSAGE}'\n\n"
        "Reformule les informations de manière claire et structurée.\n"
        "Lorsque c’est possible, indique la page ou la section mentionnée dans le contexte.\n\n"
        "Contexte :\n{context}\n\n"
        "Question :\n{question}\n\n"
        "Réponse :"
    )
    return PromptTemplate.from_template(template)


def format_context(documents: List[Document]) -> str:
    parts = []
    for doc in documents:
        page = doc.metadata.get("page")
        page_label = f"Page {page + 1}" if isinstance(page, int) else "Page inconnue"
        parts.append(f"[{page_label}]\n{doc.page_content}")
    return "\n\n".join(parts)


def format_sources(documents: List[Document]) -> List[str]:
    pages = []
    for doc in documents:
        page = doc.metadata.get("page")
        if isinstance(page, int):
            pages.append(page + 1)
    pages = sorted(set(pages))
    return [f"Page {page}" for page in pages]


def _build_excerpts_answer(documents: List[Document], max_docs: int = 3, max_chars: int = 350) -> str:
    excerpts: List[str] = []
    for doc in documents[:max_docs]:
        page = doc.metadata.get("page")
        page_label = f"Page {page + 1}" if isinstance(page, int) else "Page inconnue"
        text = " ".join(doc.page_content.split())
        if len(text) > max_chars:
            text = text[:max_chars].rstrip() + "..."
        excerpts.append(f"[{page_label}] {text}")
    return "\n\n".join(excerpts)


def generate_answer(question: str, documents: List[Document], config: ChainConfig) -> str:
    if not documents:
        return FALLBACK_MESSAGE

    prompt = build_prompt()
    context = format_context(documents)
    llm = OllamaLLM(
        model=config.llm_model,
        temperature=0,
        num_predict=config.num_predict,
        sync_client_kwargs={"timeout": config.request_timeout},
    )
    try:
        answer = llm.invoke(prompt.format(context=context, question=question))
        if isinstance(answer, str) and answer.strip():
            return answer
        return _build_excerpts_answer(documents)
    except Exception as exc:
        details = str(exc).strip() or exc.__class__.__name__
        raise RuntimeError(
            "Le modèle Ollama n'a pas répondu à temps ou a échoué. "
            "Réessaie avec un timeout plus grand (--llm-timeout) "
            "ou un modèle plus léger."
            f" Détail: {details}"
        ) from exc
