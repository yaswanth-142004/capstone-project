from __future__ import annotations

import os
import math
from typing import Iterable

import numpy as np
from langchain_core.embeddings import Embeddings

from .logging_utils import get_app_logger, log_timing


class MuRILEmbeddings(Embeddings):
    """LangChain embeddings adapter for MuRIL-style Hugging Face encoders."""

    def __init__(
        
        self,
       
        model_name: str = "google/muril-base-cased",
       
        batch_size: int = 16,
        device: str | None = None,
        show_progress: bool | None = None,
    ,
        device: str | None = None,
        show_progress: bool | None = None,
    ):
        self.model_name = model_name
        self.batch_size = batch_size
        self.device = device or os.getenv("MURIL_DEVICE", "auto")
        self.show_progress = _env_flag("MURIL_PROGRESS", default=True) if show_progress is None else show_progress
        self._tokenizer = None
        self._model = None
        self._device = None

    def _load(self) -> None:
        if self._model is not None:
            return

        import torch
        from transformers import AutoModel, AutoTokenizer

        logger = get_app_logger()
        self._torch = torch
        self._device = self._resolve_device(torch)
        if self.show_progress:
            print(f"Loading MuRIL embedding model on {self._device}: {self.model_name}", flush=True)
            if self.device.strip().lower() == "auto" and self._device == "cpu" and not torch.cuda.is_available():
                print("MuRIL device note: CUDA is not available in this Python environment; using CPU.", flush=True)
        self._device = self._resolve_device(torch)
        if self.show_progress:
            print(f"Loading MuRIL embedding model on {self._device}: {self.model_name}", flush=True)
            if self.device.strip().lower() == "auto" and self._device == "cpu" and not torch.cuda.is_available():
                print("MuRIL device note: CUDA is not available in this Python environment; using CPU.", flush=True)
        logger.info("muril_load_start model=%s device=%s", self.model_name, self._device)
        with log_timing("muril_tokenizer_load", model=self.model_name):
            self._tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        with log_timing("muril_model_load", model=self.model_name, device=self._device):
            self._model = AutoModel.from_pretrained(self.model_name).to(self._device).to(self._device)
        self._model.eval()

    def _resolve_device(self, torch) -> str:
        requested = self.device.strip().lower()
        if requested == "auto":
            return "cuda" if torch.cuda.is_available() else "cpu"
        if requested == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("MURIL_DEVICE=cuda was requested, but CUDA is not available in this PyTorch install.")
        return requested

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        self._load()
        vectors: list[list[float]] = []
        total = len(texts)
        total_batches = math.ceil(total / self.batch_size) if self.batch_size else 0
        progress_enabled = self.show_progress and total > self.batch_size
        if progress_enabled:
            print(f"Embedding {total} texts in {total_batches} batches...", flush=True)
        for batch_number, batch in enumerate(_batches(texts, self.batch_size), start=1):
            vectors.extend(self._embed_batch(batch))
            processed = min(batch_number * self.batch_size, total)
            if progress_enabled and (batch_number == 1 or batch_number == total_batches or batch_number % 10 == 0):
                print(f"Embedding progress: {processed}/{total} texts ({batch_number}/{total_batches} batches)", flush=True)
        return vectors

    def embed_query(self, text: str) -> list[float]:
        return self.embed_documents([text])[0]

    def _embed_batch(self, texts: list[str]) -> list[list[float]]:
        encoded = self._tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=256,
            return_tensors="pt",
        )
        encoded = {key: value.to(self._device) for key, value in encoded.items()}
        with self._torch.no_grad():
            output = self._model(**encoded)

        token_embeddings = output.last_hidden_state
        attention_mask = encoded["attention_mask"].unsqueeze(-1)
        masked = token_embeddings * attention_mask
        summed = masked.sum(dim=1)
        counts = attention_mask.sum(dim=1).clamp(min=1)
        mean_pooled = summed / counts
        normalized = mean_pooled / mean_pooled.norm(dim=1, keepdim=True).clamp(min=1e-12)
        return normalized.cpu().numpy().astype(np.float32).tolist()


def _batches(items: list[str], size: int) -> Iterable[list[str]]:
    for start in range(0, len(items), size):
        yield items[start : start + size]


def _env_flag(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() not in {"0", "false", "no", "off"}
