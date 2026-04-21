from __future__ import annotations

import json


def build_reasoning_prompt(
    original_text: str,
    normalized_text: str,
    syntax_report: dict,
    retrieved_examples: list[dict],
) -> str:
    examples_text = json.dumps(retrieved_examples, ensure_ascii=False, indent=2)
    syntax_text = json.dumps(syntax_report, ensure_ascii=False, indent=2)
    return f"""You are a safety auditor for Telugu-English and Indian code-mixed social media comments.

Classify the target comment as hate/offensive speech.

Labels:
- 1 = hate or offensive speech, including slurs, dehumanization, targeted abuse, threats, or attacks on protected or political/social groups.
- 0 = non-hate or not offensive, including criticism, ordinary disagreement, news, jokes without abuse, or unclear benign comments.

Use the retrieved examples only as cultural and linguistic context. Do not copy their labels blindly.
Return only valid JSON with this schema:
{{
  "label": 0,
  "label_name": "Non-Offensive",
  "confidence": 0.0,
  "languages": ["Telugu", "English"],
  "rationale": "brief explanation grounded in the text",
  "signals": ["short signal strings"]
}}

Target original text:
{original_text}

Target normalized text:
{normalized_text}

Syntax/code-mixing report:
{syntax_text}

Retrieved historical examples:
{examples_text}
"""
