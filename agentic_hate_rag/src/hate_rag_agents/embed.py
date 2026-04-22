from __future__ import annotations

import argparse
import hashlib
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .config import AppConfig, resolve_path
from .embeddings import MuRILEmbeddings
from .io_utils import DEFAULT_LABEL_COLUMNS, DEFAULT_TEXT_COLUMNS, detect_column, discover_tables, read_table
from .labels import label_tag, normalize_label
from .normalization import normalize_for_analysis
from .syntax import analyze_syntax


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run only the MuRIL embedding job for CSV/XLSX data and write vectors plus metadata."
    )
    parser.add_argument("--input", required=True, help="CSV/XLSX file or folder to embed.")
    parser.add_argument("--output-dir", default="./outputs/embeddings", help="Folder for .npz and metadata CSV outputs.")
    parser.add_argument("--text-column", help="Column containing the comment text.")
    parser.add_argument("--label-column", help="Optional label column. Auto-detected when present.")
    parser.add_argument("--id-column", help="Optional stable ID column from your CSV.")
    parser.add_argument("--recursive", action="store_true", help="Scan folders recursively.")
    parser.add_argument("--limit", type=int, help="Optional row limit for smoke tests.")
    parser.add_argument("--embedding-model", help="Override MuRIL/Hugging Face embedding model.")
    parser.add_argument("--batch-size", type=int, default=16, help="MuRIL embedding batch size.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = AppConfig()
    input_path = resolve_path(args.input)
    output_dir = resolve_path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    embedding_model = args.embedding_model or config.muril_model
    embedder = MuRILEmbeddings(model_name=embedding_model, batch_size=args.batch_size,device=)

    files = discover_tables(input_path, recursive=args.recursive)
    total = 0
    for file_path in files:
        records = records_from_file(
            file_path=file_path,
            text_column=args.text_column,
            label_column=args.label_column,
            id_column=args.id_column,
            limit=args.limit,
        )
        if not records:
            print(f"Skipped {file_path}: no embeddable rows.")
            continue

        texts = [record["normalized_text"] for record in records]
        vectors = np.asarray(embedder.embed_documents(texts), dtype=np.float32)
        metadata_df = pd.DataFrame(records)

        output_stem = safe_stem(file_path)
        vector_path = output_dir / f"{output_stem}_muril_vectors.npz"
        metadata_path = output_dir / f"{output_stem}_muril_metadata.csv"

        np.savez_compressed(
            vector_path,
            embeddings=vectors,
            ids=metadata_df["id"].to_numpy(dtype=str),
            row_indices=metadata_df["row_index"].to_numpy(dtype=str),
            normalized_texts=metadata_df["normalized_text"].to_numpy(dtype=object),
            tags=metadata_df["tags"].to_numpy(dtype=object),
        )
        metadata_df.to_csv(metadata_path, index=False, encoding="utf-8-sig")
        total += len(records)

        print(f"Embedded {len(records)} rows from {file_path}")
        print(f"Vectors: {vector_path}")
        print(f"Metadata: {metadata_path}")

    print(f"Finished MuRIL embedding job. Total rows embedded: {total}")


def records_from_file(
    file_path: Path,
    text_column: str | None,
    label_column: str | None,
    id_column: str | None,
    limit: int | None,
) -> list[dict[str, Any]]:
    df = read_table(file_path)
    if limit:
        df = df.head(limit).copy()

    text_col = detect_column(df, text_column, DEFAULT_TEXT_COLUMNS, "text")
    label_col = detect_optional_column(df, label_column, DEFAULT_LABEL_COLUMNS, "label")
    id_col = detect_optional_column(df, id_column, [], "id")

    records: list[dict[str, Any]] = []
    for row_index, row in df.iterrows():
        original = row.get(text_col, "")
        normalized = normalize_for_analysis(original)
        normalized_text = normalized.transliterated_text
        if not normalized_text.strip():
            continue

        label = normalize_label(row.get(label_col)) if label_col else None
        syntax = analyze_syntax(normalized_text)
        tags = build_tags(file_path=file_path, label=label, syntax_report=syntax.as_dict())
        item_id = str(row.get(id_col)) if id_col else stable_id(file_path, row_index, normalized_text)

        records.append(
            {
                "id": item_id,
                "row_index": row_index,
                "source": str(file_path),
                "original_text": "" if pd.isna(original) else str(original),
                "cleaned_text": normalized.cleaned_text,
                "normalized_text": normalized_text,
                "transliteration_backend": normalized.transliteration_backend,
                "label": "" if label is None else label,
                "tags": "|".join(tags),
                "token_count": syntax.token_count,
                "latin_tokens": syntax.latin_tokens,
                "telugu_tokens": syntax.telugu_tokens,
                "code_mix_ratio": syntax.code_mix_ratio,
                "repetition_ratio": syntax.repetition_ratio,
                "short_token_ratio": syntax.short_token_ratio,
                "suspicious_word_salad": syntax.suspicious_word_salad,
            }
        )
    return records


def detect_optional_column(
    df: pd.DataFrame,
    requested: str | None,
    candidates: list[str],
    purpose: str,
) -> str | None:
    if requested:
        if requested not in df.columns:
            raise ValueError(f"{purpose} column {requested!r} was not found.")
        return requested
    for column in candidates:
        if column in df.columns:
            return column
    return None


def build_tags(file_path: Path, label: int | None, syntax_report: dict[str, Any]) -> list[str]:
    telugu_tokens = int(syntax_report.get("telugu_tokens", 0))
    latin_tokens = int(syntax_report.get("latin_tokens", 0))
    if telugu_tokens and latin_tokens:
        script_tag = "script:code_mixed"
    elif telugu_tokens:
        script_tag = "script:telugu"
    elif latin_tokens:
        script_tag = "script:latin"
    else:
        script_tag = "script:other"

    tags = [
        f"source:{file_path.stem}",
        label_tag(label),
        script_tag,
    ]
    if syntax_report.get("suspicious_word_salad"):
        tags.append("quality:suspicious_word_salad")
    return tags


def stable_id(file_path: Path, row_index: Any, text: str) -> str:
    payload = f"{file_path}|{row_index}|{text}".encode("utf-8", errors="ignore")
    return hashlib.sha1(payload).hexdigest()


def safe_stem(file_path: Path) -> str:
    return "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in file_path.stem)


if __name__ == "__main__":
    main()
