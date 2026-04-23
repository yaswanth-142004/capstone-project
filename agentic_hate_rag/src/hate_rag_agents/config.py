from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv


load_dotenv()


@dataclass(frozen=True)
class AppConfig:
    llm_provider: str = field(default_factory=lambda: os.getenv("LLM_PROVIDER", "ollama").strip().lower())
    ollama_model: str = field(default_factory=lambda: os.getenv("OLLAMA_MODEL", "llama3.1:8b"))
    ollama_base_url: str = field(default_factory=lambda: os.getenv("OLLAMA_BASE_URL", "http://127.0.0.1:11434"))
    ollama_timeout_seconds: float = field(default_factory=lambda: float(os.getenv("OLLAMA_TIMEOUT_SECONDS", "180")))
    vllm_model: str = field(default_factory=lambda: os.getenv("VLLM_MODEL", "google/gemma-2-9b-it"))
    vllm_base_url: str = field(default_factory=lambda: os.getenv("VLLM_BASE_URL", "http://127.0.0.1:8090"))
    vllm_api_key: str = field(default_factory=lambda: os.getenv("VLLM_API_KEY", ""))
    vllm_timeout_seconds: float = field(default_factory=lambda: float(os.getenv("VLLM_TIMEOUT_SECONDS", "180")))
    muril_model: str = field(default_factory=lambda: os.getenv("MURIL_MODEL", "google/muril-base-cased"))
    chroma_dir: Path = field(default_factory=lambda: Path(os.getenv("CHROMA_DIR", "./storage/chroma")))
    chroma_collection: str = field(default_factory=lambda: os.getenv("CHROMA_COLLECTION", "code_mixed_hate_memory"))
    rag_top_k: int = field(default_factory=lambda: int(os.getenv("RAG_TOP_K", "5")))
    confidence_threshold: float = field(default_factory=lambda: float(os.getenv("CONFIDENCE_THRESHOLD", "0.50")))
    hitl_queue: Path = field(default_factory=lambda: Path(os.getenv("HITL_QUEUE", "./outputs/hitl_review_queue.csv")))
    app_log: Path = field(default_factory=lambda: Path(os.getenv("APP_LOG", "./outputs/app.log")))
    temperature: float = field(default_factory=lambda: float(os.getenv("OLLAMA_TEMPERATURE", "0")))

    @property
    def active_model(self) -> str:
        return self.vllm_model if self.llm_provider == "vllm" else self.ollama_model

    @property
    def active_base_url(self) -> str:
        return self.vllm_base_url if self.llm_provider == "vllm" else self.ollama_base_url

    @property
    def active_timeout_seconds(self) -> float:
        return self.vllm_timeout_seconds if self.llm_provider == "vllm" else self.ollama_timeout_seconds
    # Reflection loop settings
    reflection_enabled: bool = os.getenv("REFLECTION_ENABLED", "true").lower() in {"true", "1", "yes"}
    max_reflection_retries: int = int(os.getenv("MAX_REFLECTION_RETRIES", "1"))
    # Smart retrieval settings
    max_retrieval_distance: float = float(os.getenv("MAX_RETRIEVAL_DISTANCE", "1.2"))
    # Auto-eval and auto-ingest settings
    auto_ingest_threshold: float = float(os.getenv("AUTO_INGEST_THRESHOLD", "0.85"))
    eval_batch_size: int = int(os.getenv("EVAL_BATCH_SIZE", "50"))
    lessons_path: Path = Path(os.getenv("LESSONS_PATH", "./outputs/lessons.json"))


def resolve_path(path: str | Path) -> Path:
    return Path(path).expanduser().resolve()
