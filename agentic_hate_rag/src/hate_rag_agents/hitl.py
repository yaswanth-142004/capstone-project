from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd


def append_review_item(path: Path, item: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    row = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "text": item.get("text", ""),
        "normalized_text": item.get("normalized_text", ""),
        "predicted_label": item.get("label", ""),
        "predicted_label_name": item.get("label_name", ""),
        "confidence": item.get("confidence", ""),
        "reason": item.get("review_reason", ""),
        "explanation": item.get("explanation", ""),
        "retrieved_examples": json.dumps(item.get("retrieved_examples", []), ensure_ascii=False),
        "reviewer_label": "",
        "reviewer_notes": "",
    }

    new_df = pd.DataFrame([row])
    if path.exists():
        existing = pd.read_csv(path, encoding="utf-8-sig")
        new_df = pd.concat([existing, new_df], ignore_index=True)
    new_df.to_csv(path, index=False, encoding="utf-8-sig")
