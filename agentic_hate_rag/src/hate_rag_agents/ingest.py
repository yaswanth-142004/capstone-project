from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import pandas as pd

from .config import AppConfig, resolve_path
from .io_utils import DEFAULT_LABEL_COLUMNS, DEFAULT_TEXT_COLUMNS, detect_column, discover_tables, read_table
from .labels import normalize_label
from .normalization import normalize_for_analysis
from .rag_store import documents_from_rows, make_vector_store


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Ingest labelled code-mixed examples into Chroma RAG memory.")
    parser.add_argument("--input", required=True, help="CSV/XLSX file or folder to ingest.")
    parser.add_argument("--text-column", help="Column containing the comment text.")
    parser.add_argument("--label-column", help="Column containing hate/offensive labels.")
    parser.add_argument("--recursive", action="store_true", help="Scan folders recursively.")
    parser.add_argument("--limit", type=int, help="Optional row limit for smoke tests.")
    parser.add_argument("--chroma-dir", help="Override Chroma persistence directory.")
    parser.add_argument("--collection", help="Override Chroma collection name.")
    parser.add_argument("--embedding-model", help="Override MuRIL/HF embedding model.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = AppConfig()
    chroma_dir = resolve_path(args.chroma_dir) if args.chroma_dir else config.chroma_dir
    collection = args.collection or config.chroma_collection
    embedding_model = args.embedding_model or config.muril_model

    vector_store = make_vector_store(
        persist_directory=chroma_dir,
        collection_name=collection,
        embedding_model=embedding_model,
    )

    files = discover_tables(resolve_path(args.input), recursive=args.recursive)
    total = 0
    for file_path in files:
        rows = rows_from_file(file_path, args.text_column, args.label_column, args.limit)
        if not rows:
            print(f"Skipped {file_path}: no labelled rows.")
            continue
        documents, ids = documents_from_rows(rows)
        vector_store.add_documents(documents, ids=ids)
        total += len(documents)
        print(f"Ingested {len(documents)} rows from {file_path}")

    print(f"Finished. Total rows ingested: {total}")
    print(f"Chroma directory: {chroma_dir}")
    print(f"Collection: {collection}")


def rows_from_file(
    file_path: Path,
    text_column: str | None,
    label_column: str | None,
    limit: int | None,
) -> list[dict[str, Any]]:
    df = read_table(file_path)
    if limit:
        df = df.head(limit).copy()

    text_col = detect_column(df, text_column, DEFAULT_TEXT_COLUMNS, "text")
    label_col = detect_column(df, label_column, DEFAULT_LABEL_COLUMNS, "label")

    rows: list[dict[str, Any]] = []
    for row_index, row in df.iterrows():
        label = normalize_label(row.get(label_col))
        if label is None:
            continue
        text = row.get(text_col, "")
        normalized = normalize_for_analysis(text).transliterated_text
        if not normalized.strip():
            continue
        rows.append(
            {
                "text": normalized,
                "original_text": "" if pd.isna(text) else str(text),
                "label": label,
                "source": str(file_path),
                "row_index": row_index,
            }
        )
    return rows


if __name__ == "__main__":
    main()
