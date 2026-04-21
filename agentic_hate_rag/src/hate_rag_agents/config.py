from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv


load_dotenv()


@dataclass(frozen=True)
class AppConfig:
    ollama_model: str = os.getenv("OLLAMA_MODEL", "llama3.1:8b")
    ollama_base_url: str = os.getenv("OLLAMA_BASE_URL", "http://127.0.0.1:11434")
    muril_model: str = os.getenv("MURIL_MODEL", "google/muril-base-cased")
    chroma_dir: Path = Path(os.getenv("CHROMA_DIR", "./storage/chroma"))
    chroma_collection: str = os.getenv("CHROMA_COLLECTION", "code_mixed_hate_memory")
    rag_top_k: int = int(os.getenv("RAG_TOP_K", "5"))
    confidence_threshold: float = float(os.getenv("CONFIDENCE_THRESHOLD", "0.68"))
    hitl_queue: Path = Path(os.getenv("HITL_QUEUE", "./outputs/hitl_review_queue.csv"))
    temperature: float = float(os.getenv("OLLAMA_TEMPERATURE", "0"))


def resolve_path(path: str | Path) -> Path:
    return Path(path).expanduser().resolve()
