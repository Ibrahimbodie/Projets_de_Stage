from __future__ import annotations

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama import OllamaEmbeddings


def create_embeddings(
    provider="huggingface",
    model="BAAI/bge-m3",
    device="cpu",
):
    if hasattr(provider, "provider"):
        device = getattr(provider, "device", "cpu")
        model = getattr(provider, "model", "BAAI/bge-m3")
        provider = provider.provider

    if provider == "huggingface":
        return HuggingFaceEmbeddings(
            model_name=model,
            model_kwargs={"device": device},
            encode_kwargs={"normalize_embeddings": True},
        )

    if provider == "ollama":
        return OllamaEmbeddings(model=model)

    raise ValueError(
        f"Fournisseur d'embedding inconnu : {provider}. "
        f"Supportés : huggingface, ollama."
    )
