from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import yaml


PROJECT_ROOT = Path(__file__).resolve().parent.parent


def load_env() -> None:
    env_path = PROJECT_ROOT / ".env"
    if env_path.exists():
        with open(env_path) as f:
            for line in f:
                stripped = line.strip()
                if not stripped or stripped.startswith("#") or "=" not in stripped:
                    continue
                key, value = stripped.split("=", 1)
                key, value = key.strip(), value.strip().strip("\"'")
                os.environ.setdefault(key, value)


@dataclass
class EmbeddingConfig:
    provider: str = "huggingface"
    model: str = "BAAI/bge-m3"
    device: str = "cpu"


@dataclass
class LLMConfig:
    provider: str = "ollama"
    model: str = "qwen2.5:3b"
    temperature: float = 0.2
    num_predict: int = 512
    request_timeout: float = 300.0


@dataclass
class EvaluationConfig:
    provider: str = "ollama"
    model: str = "qwen2.5:3b"
    threshold: float = 0.75


@dataclass
class RetrievalConfig:
    top_k: int = 6
    max_distance: float = 1.5


@dataclass
class AppConfig:
    embedding: EmbeddingConfig = field(default_factory=EmbeddingConfig)
    llm: LLMConfig = field(default_factory=LLMConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)
    retrieval: RetrievalConfig = field(default_factory=RetrievalConfig)

    @property
    def vectorstore_path(self) -> Path:
        model_slug = self.embedding.model.replace("/", "_")
        return PROJECT_ROOT / "vectorstore" / model_slug


load_env()


def load_config(path: Optional[str] = None) -> AppConfig:
    if path is None:
        path = str(PROJECT_ROOT / "config.yaml")

    cfg_path = Path(path)

    if not cfg_path.exists():
        return AppConfig()

    with open(cfg_path, "r") as f:
        raw = yaml.safe_load(f) or {}

    embedding_raw = raw.get("embedding", {})
    llm_raw = raw.get("llm", {})
    evaluation_raw = raw.get("evaluation", {})
    retrieval_raw = raw.get("retrieval", {})

    return AppConfig(
        embedding=EmbeddingConfig(
            provider=embedding_raw.get("provider", "huggingface"),
            model=embedding_raw.get("model", "BAAI/bge-m3"),
            device=embedding_raw.get("device", "cpu"),
        ),
        llm=LLMConfig(
            provider=llm_raw.get("provider", "ollama"),
            model=llm_raw.get("model", "qwen2.5:3b"),
            temperature=llm_raw.get("temperature", 0.2),
            num_predict=llm_raw.get("num_predict", 512),
            request_timeout=llm_raw.get("request_timeout", 300.0),
        ),
        evaluation=EvaluationConfig(
            provider=evaluation_raw.get("provider", "ollama"),
            model=evaluation_raw.get("model", "qwen2.5:3b"),
            threshold=evaluation_raw.get("threshold", 0.75),
        ),
        retrieval=RetrievalConfig(
            top_k=retrieval_raw.get("top_k", 6),
            max_distance=retrieval_raw.get("max_distance", 1.5),
        ),
    )
