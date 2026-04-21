from __future__ import annotations

import math
import re
import unicodedata
from dataclasses import dataclass
from typing import Any


TELUGU_RE = re.compile(r"[\u0C00-\u0C7F]")
LATIN_WORD_RE = re.compile(r"\b[a-zA-Z][a-zA-Z.'-]{2,}\b")


@dataclass
class NormalizationResult:
    original_text: str
    cleaned_text: str
    transliterated_text: str
    transliteration_backend: str


def normalize_text(text: Any) -> str:
    if text is None or (isinstance(text, float) and math.isnan(text)):
        return ""

    value = str(text)
    value = unicodedata.normalize("NFKC", value)
    value = value.replace("\r\n", "\n").replace("\r", "\n")
    value = re.sub(r"https?://\S+|www\.\S+", " [URL] ", value, flags=re.IGNORECASE)
    value = re.sub(r"@\w+", " [MENTION] ", value)
    value = value.replace("&amp;", "&")
    value = expand_emojis(value)
    value = re.sub(r"[ \t]+", " ", value)
    value = re.sub(r"\n{2,}", "\n", value)
    value = re.sub(r"\s+([,.!?;:])", r"\1", value)
    value = re.sub(r"([!?.,])\1{2,}", r"\1\1", value)
    return value.strip()


def expand_emojis(text: str) -> str:
    parts: list[str] = []
    for ch in text:
        if is_emoji_like(ch):
            name = emoji_name(ch)
            parts.append(f" <emoji: {name}> " if name else " <emoji> ")
        else:
            parts.append(ch)
    return re.sub(r"\s+", " ", "".join(parts)).strip()


def is_emoji_like(ch: str) -> bool:
    if ch in {"\u200d", "\ufe0f"}:
        return False
    codepoint = ord(ch)
    return (
        0x1F000 <= codepoint <= 0x1FAFF
        or 0x2600 <= codepoint <= 0x27BF
        or unicodedata.category(ch) == "So"
    )


def emoji_name(ch: str) -> str:
    try:
        return unicodedata.name(ch).lower().replace("_", " ")
    except ValueError:
        return ""


def normalize_for_analysis(text: Any) -> NormalizationResult:
    original = "" if text is None else str(text)
    cleaned = normalize_text(original)
    transliterated, backend = transliterate_romanized_telugu(cleaned)
    return NormalizationResult(
        original_text=original,
        cleaned_text=cleaned,
        transliterated_text=transliterated,
        transliteration_backend=backend,
    )


def transliterate_romanized_telugu(text: str) -> tuple[str, str]:
    if not text or TELUGU_RE.search(text):
        return text, "not-needed"

    try:
        from indic_transliteration import sanscript
        from indic_transliteration.sanscript import transliterate
    except Exception:
        return text, "unavailable"

    def convert(match: re.Match[str]) -> str:
        token = match.group(0)
        if token.lower() in {"url", "mention", "emoji"}:
            return token
        try:
            return transliterate(token, sanscript.ITRANS, sanscript.TELUGU)
        except Exception:
            return token

    converted = LATIN_WORD_RE.sub(convert, text)
    if converted == text:
        return text, "indic-transliteration-noop"
    return converted, "indic-transliteration"
