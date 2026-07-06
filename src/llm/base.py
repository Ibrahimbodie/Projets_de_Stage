from __future__ import annotations

from abc import ABC, abstractmethod


class BaseLLMProvider(ABC):
    @abstractmethod
    def invoke(self, prompt: str) -> str:
        ...

    @abstractmethod
    def get_model_name(self) -> str:
        ...
