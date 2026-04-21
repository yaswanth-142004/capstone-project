from __future__ import annotations

from typing import Any

import pandas as pd


def normalize_label(value: Any) -> int | None:
    if pd.isna(value):
        return None
    if isinstance(value, str):
        cleaned = value.strip().lower()
        if cleaned in {"1", "hate", "hateful", "offensive", "toxic", "abusive"}:
            return 1
        if cleaned in {"0", "non-hate", "non hate", "non-offensive", "normal", "neutral"}:
            return 0
        try:
            numeric = float(cleaned)
        except ValueError:
            return None
        if numeric == 1.0:
            return 1
        if numeric == 0.0:
            return 0
        return None

    numeric = float(value)
    if numeric == 1.0:
        return 1
    if numeric == 0.0:
        return 0
    return None


def label_tag(label: int | None) -> str:
    if label == 1:
        return "label:offensive"
    if label == 0:
        return "label:non_offensive"
    return "label:unknown"
