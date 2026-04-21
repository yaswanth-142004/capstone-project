from __future__ import annotations

from typing import Iterable

import numpy as np
from langchain_core.embeddings import Embeddings


class MuRILEmbeddings(Embeddings):
    """LangChain embeddings adapter for MuRIL-style Hugging Face encoders."""

    def __init__(self, model_name: str = "google/muril-base-cased", batch_size: int = 16):
        self.model_name = model_name
        self.batch_size = batch_size
        self._tokenizer = None
        self._model = None

    def _load(self) -> None:
        if self._model is not None:
            return

        import torch
        from transformers import AutoModel, AutoTokenizer

        self._torch = torch
        self._tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self._model = AutoModel.from_pretrained(self.model_name)
        self._model.eval()

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        self._load()
        vectors: list[list[float]] = []
        for batch in _batches(texts, self.batch_size):
            vectors.extend(self._embed_batch(batch))
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
