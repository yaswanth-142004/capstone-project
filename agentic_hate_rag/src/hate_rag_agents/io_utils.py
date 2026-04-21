from __future__ import annotations

from pathlib import Path
from typing import Optional

import pandas as pd


DEFAULT_TEXT_COLUMNS = [
    "normalized_text",
    "Comment",
    "comment",
    "text",
    "Text",
    "sentence",
    "Sentence",
    "content",
    "Content",
]

DEFAULT_LABEL_COLUMNS = [
    "hate_label",
    "label",
    "Label",
    "class",
    "Class",
    "category",
    "Category",
    "reviewer_label",
]


def read_table(path: Path) -> pd.DataFrame:
    suffix = path.suffix.lower()
    if suffix == ".csv":
        return pd.read_csv(path, encoding="utf-8-sig")
    if suffix in {".xlsx", ".xls"}:
        return pd.read_excel(path)
    raise ValueError(f"Unsupported file type: {suffix}")


def write_table(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    suffix = path.suffix.lower()
    if suffix == ".csv":
        df.to_csv(path, index=False, encoding="utf-8-sig")
        return
    if suffix in {".xlsx", ".xls"}:
        df.to_excel(path, index=False)
        return
    raise ValueError(f"Unsupported output file type: {suffix}")


def is_supported_table(path: Path) -> bool:
    return path.is_file() and path.suffix.lower() in {".csv", ".xlsx", ".xls"}


def discover_tables(path: Path, recursive: bool) -> list[Path]:
    if path.is_file():
        if not is_supported_table(path):
            raise ValueError(f"Unsupported input file: {path}")
        return [path]

    if not path.is_dir():
        raise FileNotFoundError(f"Input path not found: {path}")

    pattern = "**/*" if recursive else "*"
    files = [item for item in path.glob(pattern) if is_supported_table(item)]
    files.sort()
    if not files:
        raise ValueError(f"No CSV/XLSX files found in folder: {path}")
    return files


def detect_column(
    df: pd.DataFrame,
    requested: Optional[str],
    candidates: list[str],
    purpose: str,
) -> str:
    if requested:
        if requested not in df.columns:
            raise ValueError(f"{purpose} column {requested!r} was not found.")
        return requested

    for column in candidates:
        if column in df.columns:
            return column

    raise ValueError(f"Could not infer {purpose} column. Pass it explicitly.")
