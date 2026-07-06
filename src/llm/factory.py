from __future__ import annotations

from typing import Optional

from .base import BaseLLMProvider
from .ollama import OllamaProvider
from .openrouter import OpenRouterProvider


def create_llm(
    provider: str = "ollama",
    model: str = "qwen2.5:3b",
    temperature: float = 0.2,
    num_predict: int = 512,
    request_timeout: float = 300.0,
    api_key: Optional[str] = None,
) -> BaseLLMProvider:
    if provider == "ollama":
        return OllamaProvider(
            model=model,
            temperature=temperature,
            num_predict=num_predict,
            request_timeout=request_timeout,
        )

    if provider == "openrouter":
        return OpenRouterProvider(
            model=model,
            api_key=api_key,
            temperature=temperature,
            max_tokens=num_predict,
            timeout=request_timeout,
        )

    raise ValueError(
        f"Fournisseur LLM inconnu : {provider}. "
        f"Supportés : ollama, openrouter."
    )


def llm_invoke(llm: BaseLLMProvider, prompt: str) -> str:
    return llm.invoke(prompt)
