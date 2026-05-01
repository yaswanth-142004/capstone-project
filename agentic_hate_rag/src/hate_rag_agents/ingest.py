from __future__ import annotations

import argparse
import shutil
from pathlib import Path
from typing import Any

import pandas as pd

from .config import AppConfig, resolve_path
from .io_utils import DEFAULT_LABEL_COLUMNS, DEFAULT_TEXT_COLUMNS, detect_column, discover_tables, read_table
from .labels import normalize_label
from .logging_utils import get_app_logger, log_timing, setup_app_logging
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
    parser.add_argument("--reset-store", action="store_true", help="Delete the existing Chroma store before ingesting.")
    parser.add_argument("--add-batch-size", type=int, default=5000, help="Maximum documents to add to Chroma per upsert.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = AppConfig()
    logger = setup_app_logging(config.app_log)
    chroma_dir = resolve_path(args.chroma_dir) if args.chroma_dir else config.chroma_dir
    collection = args.collection or config.chroma_collection
    embedding_model = args.embedding_model or config.muril_model
    logger.info("ingest_start input=%s chroma_dir=%s collection=%s reset=%s add_batch_size=%s", args.input, chroma_dir, collection, args.reset_store, args.add_batch_size)

    if args.reset_store:
        with log_timing("ingest_reset_store", chroma_dir=chroma_dir):
            reset_store(chroma_dir)

    with log_timing("ingest_make_vector_store", chroma_dir=chroma_dir, collection=collection):
        vector_store = make_vector_store(
            persist_directory=chroma_dir,
            collection_name=collection,
            embedding_model=embedding_model,
        )

    with log_timing("ingest_discover_tables", input=args.input, recursive=args.recursive):
        files = discover_tables(resolve_path(args.input), recursive=args.recursive)
    total = 0
    for file_path in files:
        with log_timing("ingest_rows_from_file", path=file_path):
            rows = rows_from_file(file_path, args.text_column, args.label_column, args.limit)
        if not rows:
            print(f"Skipped {file_path}: no labelled rows.")
            logger.info("ingest_file_skipped path=%s reason=no_labelled_rows", file_path)
            continue
        documents, ids = documents_from_rows(rows)
        with log_timing("ingest_add_documents", path=file_path, rows=len(documents)):
            add_documents_in_batches(vector_store, documents, ids, args.add_batch_size)
        total += len(documents)
        print(f"Ingested {len(documents)} rows from {file_path}")
        logger.info("ingest_file_done path=%s rows=%s total=%s", file_path, len(documents), total)

    print(f"Finished. Total rows ingested: {total}")
    print(f"Chroma directory: {chroma_dir}")
    print(f"Collection: {collection}")
    logger.info("ingest_done total=%s chroma_dir=%s collection=%s", total, chroma_dir, collection)


def reset_store(chroma_dir: Path) -> None:
    chroma_dir = chroma_dir.resolve()
    if not chroma_dir.exists():
        print(f"No existing Chroma store found at {chroma_dir}")
        return
    if not chroma_dir.is_dir():
        raise ValueError(f"Refusing to reset non-directory Chroma path: {chroma_dir}")
    if chroma_dir.anchor == str(chroma_dir):
        raise ValueError(f"Refusing to reset filesystem root: {chroma_dir}")
    shutil.rmtree(chroma_dir)
    print(f"Deleted existing Chroma store: {chroma_dir}")


def add_documents_in_batches(vector_store, documents: list, ids: list[str], batch_size: int) -> None:
    if batch_size < 1:
        raise ValueError("--add-batch-size must be at least 1.")

    total = len(documents)
    if total <= batch_size:
        vector_store.add_documents(documents, ids=ids)
        return

    total_batches = (total + batch_size - 1) // batch_size
    print(f"Adding {total} embedded documents to Chroma in {total_batches} batches...", flush=True)
    logger = get_app_logger()
    logger.info("chroma_add_start total=%s batch_size=%s batches=%s", total, batch_size, total_batches)
    for batch_number, start in enumerate(range(0, total, batch_size), start=1):
        end = min(start + batch_size, total)
        with log_timing("chroma_add_batch", batch=batch_number, batches=total_batches, start=start, end=end):
            vector_store.add_documents(documents[start:end], ids=ids[start:end])
        print(f"Chroma add progress: {end}/{total} documents ({batch_number}/{total_batches} batches)", flush=True)
        logger.info("chroma_add_progress end=%s total=%s batch=%s batches=%s", end, total, batch_number, total_batches)


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
