from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

from langchain_chroma import Chroma
from langchain_core.documents import Document

from .embeddings import MuRILEmbeddings


def make_vector_store(
    persist_directory: Path,
    collection_name: str,
    embedding_model: str,
) -> Chroma:
    persist_directory.mkdir(parents=True, exist_ok=True)
    embeddings = MuRILEmbeddings(model_name=embedding_model)
    return Chroma(
        collection_name=collection_name,
        embedding_function=embeddings,
        persist_directory=str(persist_directory),
    )


def stable_id(source: str, row_index: Any, text: str) -> str:
    payload = f"{source}|{row_index}|{text}".encode("utf-8", errors="ignore")
    return hashlib.sha1(payload).hexdigest()


def documents_from_rows(rows: list[dict[str, Any]]) -> tuple[list[Document], list[str]]:
    documents: list[Document] = []
    ids: list[str] = []
    for row in rows:
        text = str(row["text"]).strip()
        source = str(row.get("source", "unknown"))
        row_index = row.get("row_index", "")
        ids.append(stable_id(source, row_index, text))
        documents.append(
            Document(
                page_content=text,
                metadata={
                    "label": str(row.get("label", "")),
                    "source": source,
                    "row_index": str(row_index),
                    "original_text": str(row.get("original_text", "")),
                },
            )
        )
    return documents, ids


def retrieve_examples(vector_store: Chroma, query: str, top_k: int) -> list[dict[str, Any]]:
    results = vector_store.similarity_search_with_score(query, k=top_k)
    examples: list[dict[str, Any]] = []
    for document, distance in results:
        examples.append(
            {
                "text": document.page_content,
                "label": document.metadata.get("label", ""),
                "source": document.metadata.get("source", ""),
                "row_index": document.metadata.get("row_index", ""),
                "distance": float(distance),
            }
        )
    return examples
