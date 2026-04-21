from __future__ import annotations

import re
from dataclasses import dataclass


TELUGU_TOKEN_RE = re.compile(r"[\u0C00-\u0C7F]+")
LATIN_TOKEN_RE = re.compile(r"[A-Za-z]+")
TOKEN_RE = re.compile(r"[\u0C00-\u0C7FA-Za-z]+")


@dataclass
class SyntaxReport:
    token_count: int
    latin_tokens: int
    telugu_tokens: int
    code_mix_ratio: float
    repetition_ratio: float
    short_token_ratio: float
    suspicious_word_salad: bool
    notes: list[str]

    def as_dict(self) -> dict:
        return {
            "token_count": self.token_count,
            "latin_tokens": self.latin_tokens,
            "telugu_tokens": self.telugu_tokens,
            "code_mix_ratio": self.code_mix_ratio,
            "repetition_ratio": self.repetition_ratio,
            "short_token_ratio": self.short_token_ratio,
            "suspicious_word_salad": self.suspicious_word_salad,
            "notes": self.notes,
        }


def analyze_syntax(text: str) -> SyntaxReport:
    tokens = TOKEN_RE.findall(text or "")
    token_count = len(tokens)
    if token_count == 0:
        return SyntaxReport(0, 0, 0, 0.0, 0.0, 0.0, False, ["empty text"])

    latin_count = sum(1 for token in tokens if LATIN_TOKEN_RE.fullmatch(token))
    telugu_count = sum(1 for token in tokens if TELUGU_TOKEN_RE.search(token))
    code_mix_ratio = min(latin_count, telugu_count) / max(token_count, 1)
    unique_count = len({token.lower() for token in tokens})
    repetition_ratio = 1.0 - (unique_count / token_count)
    short_token_ratio = sum(1 for token in tokens if len(token) <= 2) / token_count

    notes: list[str] = []
    if latin_count and telugu_count:
        notes.append("contains both Latin and Telugu script tokens")
    if repetition_ratio >= 0.45:
        notes.append("high token repetition")
    if short_token_ratio >= 0.55 and token_count >= 6:
        notes.append("many very short tokens")

    suspicious = repetition_ratio >= 0.55 or (short_token_ratio >= 0.65 and token_count >= 8)
    if suspicious:
        notes.append("possible low-structure evasion text")

    return SyntaxReport(
        token_count=token_count,
        latin_tokens=latin_count,
        telugu_tokens=telugu_count,
        code_mix_ratio=round(code_mix_ratio, 3),
        repetition_ratio=round(repetition_ratio, 3),
        short_token_ratio=round(short_token_ratio, 3),
        suspicious_word_salad=suspicious,
        notes=notes,
    )
