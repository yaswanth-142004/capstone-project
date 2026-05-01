"""Reflection and self-improvement engine: manages learned lessons and error analysis."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class LessonStore:
    """Manages a growing list of lessons learned from classification errors."""

    lessons: list[str] = field(default_factory=list)
    path: Path | None = None

    @classmethod
    def load(cls, path: Path) -> "LessonStore":
        """Load lessons from a JSON file, or return an empty store if the file doesn't exist."""
        store = cls(path=path)
        if path.exists():
            try:
                with open(path, "r", encoding="utf-8") as fh:
                    data = json.load(fh)
                if isinstance(data, list):
                    store.lessons = [str(item) for item in data]
                elif isinstance(data, dict) and "lessons" in data:
                    store.lessons = [str(item) for item in data["lessons"]]
            except (json.JSONDecodeError, OSError):
                pass
        return store

    def save(self) -> None:
        """Persist the current lessons to the configured path."""
        if self.path is None:
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.path, "w", encoding="utf-8") as fh:
            json.dump({"lessons": self.lessons, "count": len(self.lessons)}, fh, ensure_ascii=False, indent=2)

    def add_lessons(self, new_lessons: list[str]) -> None:
        """Add new lessons (deduplicating by exact string match) and persist."""
        existing = set(self.lessons)
        for lesson in new_lessons:
            stripped = lesson.strip()
            if stripped and stripped not in existing:
                self.lessons.append(stripped)
                existing.add(stripped)
        self.save()

    def as_prompt_block(self) -> str:
        """Format all lessons as a prompt-injectable text block."""
        if not self.lessons:
            return ""
        numbered = "\n".join(f"{i + 1}. {lesson}" for i, lesson in enumerate(self.lessons))
        return (
            "IMPORTANT - Lessons learned from previous classification errors on this dataset. "
            "Apply these when classifying the current comment:\n"
            f"{numbered}\n"
        )


def analyze_errors(llm: Any, misclassified_rows: list[dict[str, Any]]) -> list[str]:
    """Send misclassified rows to the LLM and ask it to identify error patterns.

    Returns a list of lesson strings extracted from the LLM response.
    """
    if not misclassified_rows:
        return []

    examples_text = json.dumps(misclassified_rows[:15], ensure_ascii=False, indent=2)
    prompt = f"""You are reviewing your own classification errors on Telugu-English and Indian code-mixed social media comments.

Below are comments where your predicted label was WRONG compared to the human-annotated ground truth.

For each mistake, analyze WHY the prediction was wrong: cultural context you missed, Telugu slang or idioms you misread, sarcasm you didn't detect, political criticism you incorrectly flagged as hate, or offensive language you incorrectly marked as benign.

Return a JSON object with a single key "lessons" containing an array of SHORT, ACTIONABLE lesson strings.
Each lesson should be one concrete pattern or rule you should remember for future classifications.
Limit to 3-5 lessons maximum.
Each lesson must be a single plain-English sentence.
Each lesson must be under 160 characters.
Do not repeat words or fragments.
Do not output Markdown, TeX, escaped formulas, or long quoted examples.
Do not invent placeholder tokens like X or Y.
If no high-quality lesson can be formed, return {{"lessons": []}}.

Example output:
{{"lessons": [
  "Political criticism without a slur or threat should not be labeled hate speech.",
  "Direct insults targeting a person are offensive even when no protected group is mentioned.",
  "Sarcastic praise followed by abusive wording often signals offensive intent."
]}}

Misclassified examples:
{examples_text}
"""

    try:
        response = llm.invoke(prompt)
        raw = str(response.content)
        parsed = _parse_json_response(raw)
        lessons = parsed.get("lessons", [])
        if isinstance(lessons, list):
            return _sanitize_lessons(lessons)
    except Exception as exc:
        print(f"Warning: Error analysis LLM call failed: {type(exc).__name__}: {exc}", flush=True)

    return []


def build_reflection_prompt(
    original_text: str,
    normalized_text: str,
    first_label: int,
    first_label_name: str,
    first_confidence: float,
    first_explanation: str,
    new_examples: list[dict[str, Any]],
    syntax_report: dict[str, Any],
) -> str:
    """Build a reflection prompt that asks the LLM to reconsider its initial classification."""
    examples_text = json.dumps(new_examples, ensure_ascii=False, indent=2)
    syntax_text = json.dumps(syntax_report, ensure_ascii=False, indent=2)
    return f"""You are a safety auditor for Telugu-English and Indian code-mixed social media comments.

You previously classified this comment but had LOW CONFIDENCE ({first_confidence:.2f}).

Your initial assessment:
- Label: {first_label} ({first_label_name})
- Confidence: {first_confidence:.2f}
- Rationale: {first_explanation}

Now reconsider your classification using these ADDITIONAL similar examples that include both offensive (label=1) and non-offensive (label=0) comments for contrast.

Be more careful this time. Consider:
- Is this genuine hate/abuse, or just strong political opinion?
- Are there Telugu slang words that are actually insults?
- Could this be sarcasm or coded language?

Return ONLY valid JSON with this schema:
{{
  "label": 0,
  "label_name": "Non-Offensive",
  "confidence": 0.0,
  "languages": ["Telugu", "English"],
  "primary_topic": "politics",
  "topic_tags": ["politics"],
  "rationale": "brief explanation of your revised reasoning",
  "signals": ["short signal strings"],
  "reflection_note": "what changed in your assessment and why"
}}

Target original text:
{original_text}

Target normalized text:
{normalized_text}

Syntax/code-mixing report:
{syntax_text}

Additional diverse examples for reconsideration:
{examples_text}
"""


def _parse_json_response(raw: str) -> dict[str, Any]:
    """Parse JSON from LLM response, handling wrapped/dirty output."""
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        start = raw.find("{")
        end = raw.rfind("}")
        if start != -1 and end != -1 and end > start:
            try:
                return json.loads(raw[start : end + 1])
            except json.JSONDecodeError:
                pass
    return {}


def _sanitize_lessons(lessons: list[Any]) -> list[str]:
    cleaned: list[str] = []
    seen: set[str] = set()
    for lesson in lessons:
        text = _normalize_lesson_text(lesson)
        if not text:
            continue
        key = text.casefold()
        if key in seen:
            continue
        seen.add(key)
        cleaned.append(text)
    return cleaned[:5]


def _normalize_lesson_text(lesson: Any) -> str:
    text = str(lesson).strip()
    if not text:
        return ""
    text = text.replace("\r", " ").replace("\n", " ")
    text = re.sub(r"\s+", " ", text).strip()
    text = re.sub(r"[*_`#]", "", text)
    if text.startswith(("'", '"')) and text.endswith(("'", '"')) and len(text) >= 2:
        text = text[1:-1].strip()
    if len(text) > 160:
        return ""
    if "$" in text or "\\text{" in text or "\\frac" in text:
        return ""
    if re.search(r"(qual(?:/qual)+|quant(?:/qual)+)", text, flags=re.IGNORECASE):
        return ""
    if _has_repetitive_fragment(text):
        return ""
    if text.count(" ") < 3:
        return ""
    if text[-1] not in ".!?":
        text += "."
    return text


def _has_repetitive_fragment(text: str) -> bool:
    repeated_word = re.search(r"\b(\w{3,})\b(?:\W+\1\b){3,}", text, flags=re.IGNORECASE)
    if repeated_word:
        return True
    repeated_chunk = re.search(r"(.{4,20}?)(?:\1){3,}", text, flags=re.IGNORECASE)
    if repeated_chunk:
        return True
    slash_segments = text.split("/")
    if len(slash_segments) >= 8:
        normalized = [segment.strip().casefold() for segment in slash_segments if segment.strip()]
        if normalized:
            most_common = max(normalized.count(item) for item in set(normalized))
            if most_common >= 6:
                return True
    return False
