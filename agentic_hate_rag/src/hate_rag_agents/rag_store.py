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


def retrieve_examples(
    vector_store: Chroma,
    query: str,
    top_k: int,
    max_distance: float | None = None,
) -> list[dict[str, Any]]:
    results = vector_store.similarity_search_with_score(query, k=top_k)
    examples: list[dict[str, Any]] = []
    for document, distance in results:
        if max_distance is not None and float(distance) > max_distance:
            continue
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


def retrieve_diverse_examples(
    vector_store: Chroma,
    query: str,
    total: int = 5,
    fetch_k: int = 10,
    max_distance: float | None = None,
) -> list[dict[str, Any]]:
    """Retrieve examples with label diversity: fetch more than needed, then balance by label.

    This prevents the LLM from being biased by seeing only one label class.
    """
    results = vector_store.similarity_search_with_score(query, k=fetch_k)
    by_label: dict[str, list[dict[str, Any]]] = {"0": [], "1": [], "other": []}
    for document, distance in results:
        if max_distance is not None and float(distance) > max_distance:
            continue
        item = {
            "text": document.page_content,
            "label": document.metadata.get("label", ""),
            "source": document.metadata.get("source", ""),
            "row_index": document.metadata.get("row_index", ""),
            "distance": float(distance),
        }
        label_key = item["label"] if item["label"] in {"0", "1"} else "other"
        by_label[label_key].append(item)

    # Take up to ceil(total/2) from each label, prioritizing closest first
    per_label = max(1, (total + 1) // 2)
    diverse: list[dict[str, Any]] = []
    for label_key in ["0", "1", "other"]:
        diverse.extend(by_label[label_key][:per_label])

    # Sort by distance and trim to total
    diverse.sort(key=lambda x: x["distance"])
    return diverse[:total]


def auto_ingest_row(
    vector_store: Chroma,
    text: str,
    original_text: str,
    label: int,
    source: str = "auto_ingest",
    row_index: Any = "",
) -> str:
    """Ingest a single high-confidence verified row into ChromaDB and return its ID."""
    doc_id = stable_id(source, row_index, text)
    document = Document(
        page_content=text,
        metadata={
            "label": str(label),
            "source": source,
            "row_index": str(row_index),
            "original_text": original_text,
        },
    )
    vector_store.add_documents([document], ids=[doc_id])
    return doc_id
