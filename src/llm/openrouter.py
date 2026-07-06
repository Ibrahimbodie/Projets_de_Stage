from __future__ import annotations

import os

import requests

from .base import BaseLLMProvider


OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"


class OpenRouterProvider(BaseLLMProvider):
    def __init__(
        self,
        model: str = "google/gemini-2.5-flash",
        api_key: str | None = None,
        temperature: float = 0.2,
        max_tokens: int = 512,
        timeout: float = 120.0,
    ):
        self._model = model
        self._api_key = api_key or os.getenv("OPENROUTER_API_KEY")
        self._temperature = temperature
        self._max_tokens = max_tokens
        self._timeout = timeout

        if not self._api_key:
            raise RuntimeError(
                "OPENROUTER_API_KEY non définie. "
                "Ajoutez-la dans .env ou exportez-la."
            )

    def invoke(self, prompt: str) -> str:
        response = requests.post(
            OPENROUTER_URL,
            headers={
                "Authorization": f"Bearer {self._api_key}",
                "Content-Type": "application/json",
            },
            json={
                "model": self._model,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": self._temperature,
                "max_tokens": self._max_tokens,
            },
            timeout=self._timeout,
        )
        response.raise_for_status()
        data = response.json()
        return data["choices"][0]["message"]["content"]

    def get_model_name(self) -> str:
        return self._model
