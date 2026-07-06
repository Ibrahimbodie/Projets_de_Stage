from __future__ import annotations

import os

import requests

from deepeval.models.base_model import DeepEvalBaseLLM


class OllamaJudge(DeepEvalBaseLLM):
    def __init__(self, model_name: str = "qwen2.5:3b"):
        self._model_name = model_name

    def load_model(self):
        return self._model_name

    def generate(self, prompt: str) -> str:
        try:
            response = requests.post(
                "http://localhost:11434/api/generate",
                json={
                    "model": self._model_name,
                    "prompt": prompt,
                    "stream": False,
                },
                timeout=120,
            )
            return response.json().get("response", "")
        except Exception as exc:
            return f"Error connecting to Ollama: {exc}"

    async def a_generate(self, prompt: str) -> str:
        return self.generate(prompt)

    def get_model_name(self) -> str:
        return self._model_name


class OpenRouterJudge(DeepEvalBaseLLM):
    def __init__(self, model_name: str = "google/gemini-2.5-flash"):
        self._model_name = model_name
        self._api_key = os.getenv("OPENROUTER_API_KEY")
        if not self._api_key:
            raise RuntimeError("OPENROUTER_API_KEY non définie.")

    def load_model(self):
        return self._model_name

    def generate(self, prompt: str) -> str:
        try:
            response = requests.post(
                "https://openrouter.ai/api/v1/chat/completions",
                headers={
                    "Authorization": f"Bearer {self._api_key}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": self._model_name,
                    "messages": [{"role": "user", "content": prompt}],
                },
                timeout=120,
            )
            response.raise_for_status()
            return response.json()["choices"][0]["message"]["content"]
        except Exception as exc:
            return f"Error connecting to OpenRouter: {exc}"

    async def a_generate(self, prompt: str) -> str:
        return self.generate(prompt)

    def get_model_name(self) -> str:
        return self._model_name


def create_judge(
    provider: str = "ollama",
    model: str = "qwen2.5:3b",
):
    if provider == "ollama":
        return OllamaJudge(model_name=model)

    if provider == "openrouter":
        return OpenRouterJudge(model_name=model)

    raise ValueError(
        f"Fournisseur de juge inconnu : {provider}. "
        f"Supportés : ollama, openrouter."
    )
