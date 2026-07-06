from __future__ import annotations

from .base import BaseLLMProvider
from langchain_ollama import OllamaLLM


class OllamaProvider(BaseLLMProvider):
    def __init__(
        self,
        model: str = "qwen2.5:3b",
        temperature: float = 0.2,
        num_predict: int = 512,
        request_timeout: float = 300.0,
    ):
        self._model = model
        self._llm = OllamaLLM(
            model=model,
            temperature=temperature,
            num_predict=num_predict,
            sync_client_kwargs={"timeout": request_timeout},
        )

    def invoke(self, prompt: str) -> str:
        result = self._llm.invoke(prompt)
        if hasattr(result, "content"):
            return result.content
        return str(result)

    def get_model_name(self) -> str:
        return self._model
