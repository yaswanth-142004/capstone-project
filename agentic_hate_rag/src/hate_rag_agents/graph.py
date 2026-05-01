from __future__ import annotations

import json
from typing import Any, TypedDict

from langgraph.graph import END, StateGraph

from .config import AppConfig
from .hitl import append_review_item
from .llm_clients import build_llm
from .logging_utils import get_app_logger, log_timing
from .normalization import normalize_for_analysis
from .prompts import build_reasoning_prompt
from .rag_store import make_vector_store, retrieve_diverse_examples, retrieve_examples
from .reflection import build_reflection_prompt
from .syntax import analyze_syntax


class ClassificationState(TypedDict, total=False):
    row_id: str
    text: str
    normalized_text: str
    cleaned_text: str
    transliteration_backend: str
    syntax_report: dict[str, Any]
    retrieved_examples: list[dict[str, Any]]
    raw_response: str
    label: int
    label_name: str
    confidence: float
    languages: list[str]
    primary_topic: str
    topic_tags: list[str]
    explanation: str
    signals: list[str]
    parse_error: str
    needs_review: bool
    review_reason: str
    reflection_count: int
    first_pass_label: int
    first_pass_confidence: float
    first_pass_explanation: str
    reflection_used: bool
    reflection_note: str
    lessons_block: str


def build_graph(config: AppConfig):
    logger = get_app_logger()
    vector_store = make_vector_store(
        persist_directory=config.chroma_dir,
        collection_name=config.chroma_collection,
        embedding_model=config.muril_model,
    )
    llm = build_llm(config)

    def normalize_node(state: ClassificationState) -> ClassificationState:
        row_id = state.get("row_id", "")
        with log_timing("node_normalization", row_id=row_id, chars=len(state.get("text", ""))):
            result = normalize_for_analysis(state["text"])
            return {
                **state,
                "cleaned_text": result.cleaned_text,
                "normalized_text": result.transliterated_text or result.cleaned_text,
                "transliteration_backend": result.transliteration_backend,
            }

    def syntax_node(state: ClassificationState) -> ClassificationState:
        row_id = state.get("row_id", "")
        text = state.get("normalized_text", "")
        with log_timing("node_syntax", row_id=row_id, chars=len(text)):
            report = analyze_syntax(text)
            return {**state, "syntax_report": report.as_dict()}

    def retrieve_node(state: ClassificationState) -> ClassificationState:
        row_id = state.get("row_id", "")
        query = state.get("normalized_text") or state["text"]
        with log_timing("node_retrieval", row_id=row_id, chars=len(query), top_k=config.rag_top_k):
            examples = retrieve_examples(
                vector_store,
                query=query,
                top_k=config.rag_top_k,
                max_distance=config.max_retrieval_distance,
            )
            logger.info("retrieval_done row_id=%s examples=%s", row_id, len(examples))
            return {**state, "retrieved_examples": examples}

    def reason_node(state: ClassificationState) -> ClassificationState:
        row_id = state.get("row_id", "")
        with log_timing("node_reasoning", row_id=row_id, examples=len(state.get("retrieved_examples", []))):
            prompt = build_reasoning_prompt(
                original_text=state["text"],
                normalized_text=state.get("normalized_text", ""),
                syntax_report=state.get("syntax_report", {}),
                retrieved_examples=state.get("retrieved_examples", []),
                lessons_block=state.get("lessons_block", ""),
            )
            logger.info(
                "llm_invoke_start row_id=%s prompt_chars=%s provider=%s model=%s timeout_s=%s",
                row_id,
                len(prompt),
                config.llm_provider,
                config.active_model,
                config.active_timeout_seconds,
            )
            with log_timing("llm_invoke", row_id=row_id, provider=config.llm_provider, model=config.active_model):
                response = llm.invoke(prompt)
            raw = str(response.content)
            logger.info("llm_invoke_done row_id=%s provider=%s response_chars=%s", row_id, config.llm_provider, len(raw))
            try:
                parsed = _parse_llm_json(raw)
            except (json.JSONDecodeError, ValueError, TypeError) as exc:
                logger.warning("llm_json_parse_failed row_id=%s error=%s raw_prefix=%s", row_id, exc, raw[:240].replace("\n", "\\n"))
                return {
                    **state,
                    "raw_response": raw,
                    "label": 0,
                    "label_name": "Unclear",
                    "confidence": 0.0,
                    "languages": [],
                    "primary_topic": "unclear",
                    "topic_tags": ["unclear"],
                    "explanation": f"LLM returned invalid JSON: {type(exc).__name__}",
                    "signals": ["invalid_llm_json"],
                    "parse_error": str(exc),
                    "reflection_count": state.get("reflection_count", 0),
                    "reflection_used": False,
                }

            label = _coerce_label(parsed.get("label", 0))
            confidence = _coerce_confidence(parsed.get("confidence", 0.0))
            logger.info("reasoning_done row_id=%s label=%s confidence=%s topic=%s", row_id, label, confidence, parsed.get("primary_topic", ""))
            return {
                **state,
                "raw_response": raw,
                "label": label,
                "label_name": parsed.get("label_name") or ("Offensive" if label == 1 else "Non-Offensive"),
                "confidence": confidence,
                "languages": parsed.get("languages", []),
                "primary_topic": _clean_topic(parsed.get("primary_topic", "unclear")),
                "topic_tags": _clean_topic_tags(parsed.get("topic_tags", [])),
                "explanation": parsed.get("rationale", ""),
                "signals": parsed.get("signals", []),
                "parse_error": "",
                "reflection_count": state.get("reflection_count", 0),
                "reflection_used": False,
            }

    def should_reflect(state: ClassificationState) -> str:
        if not config.reflection_enabled:
            return "hitl_routing"
        if state.get("reflection_count", 0) >= config.max_reflection_retries:
            return "hitl_routing"
        if state.get("confidence", 0.0) < config.confidence_threshold:
            return "reflection"
        return "hitl_routing"

    def reflection_node(state: ClassificationState) -> ClassificationState:
        row_id = state.get("row_id", "")
        query = state.get("normalized_text") or state["text"]
        first_label = state.get("label", 0)
        first_confidence = state.get("confidence", 0.0)
        first_explanation = state.get("explanation", "")

        with log_timing("node_reflection", row_id=row_id, chars=len(query), top_k=config.rag_top_k):
            new_examples = retrieve_diverse_examples(
                vector_store,
                query=query,
                total=config.rag_top_k,
                fetch_k=max(config.rag_top_k * 2, config.rag_top_k),
                max_distance=config.max_retrieval_distance,
            )

            prompt = build_reflection_prompt(
                original_text=state["text"],
                normalized_text=state.get("normalized_text", ""),
                first_label=first_label,
                first_label_name=state.get("label_name", ""),
                first_confidence=first_confidence,
                first_explanation=first_explanation,
                new_examples=new_examples,
                syntax_report=state.get("syntax_report", {}),
            )
            with log_timing("llm_reflection_invoke", row_id=row_id, provider=config.llm_provider, model=config.active_model):
                response = llm.invoke(prompt)
            raw = str(response.content)
            try:
                parsed = _parse_llm_json(raw)
            except (json.JSONDecodeError, ValueError, TypeError) as exc:
                logger.warning("reflection_json_parse_failed row_id=%s error=%s raw_prefix=%s", row_id, exc, raw[:240].replace("\n", "\\n"))
                return {
                    **state,
                    "raw_response": raw,
                    "parse_error": f"reflection JSON parse failed: {exc}",
                    "reflection_count": state.get("reflection_count", 0) + 1,
                    "reflection_used": True,
                    "reflection_note": "Reflection was attempted, but the LLM returned invalid JSON.",
                    "retrieved_examples": new_examples,
                }

            new_label = _coerce_label(parsed.get("label", 0))
            new_confidence = _coerce_confidence(parsed.get("confidence", 0.0))
            logger.info("reflection_done row_id=%s label=%s confidence=%s", row_id, new_label, new_confidence)
            return {
                **state,
                "raw_response": raw,
                "label": new_label,
                "label_name": parsed.get("label_name") or ("Offensive" if new_label == 1 else "Non-Offensive"),
                "confidence": new_confidence,
                "languages": parsed.get("languages", []),
                "primary_topic": _clean_topic(parsed.get("primary_topic", state.get("primary_topic", "unclear"))),
                "topic_tags": _clean_topic_tags(parsed.get("topic_tags", state.get("topic_tags", []))),
                "explanation": parsed.get("rationale", ""),
                "signals": parsed.get("signals", []),
                "first_pass_label": first_label,
                "first_pass_confidence": first_confidence,
                "first_pass_explanation": first_explanation,
                "reflection_count": state.get("reflection_count", 0) + 1,
                "reflection_used": True,
                "reflection_note": parsed.get("reflection_note", ""),
                "retrieved_examples": new_examples,
                "parse_error": "",
            }

    def route_node(state: ClassificationState) -> ClassificationState:
        row_id = state.get("row_id", "")
        with log_timing("node_hitl_routing", row_id=row_id):
            reasons: list[str] = []
            if state.get("confidence", 0.0) < config.confidence_threshold:
                reasons.append("low confidence")
            if state.get("syntax_report", {}).get("suspicious_word_salad"):
                reasons.append("possible evasion or low-structure text")
            if state.get("parse_error"):
                reasons.append("invalid LLM JSON")
            if not state.get("retrieved_examples"):
                reasons.append("no retrieved context")

            needs_review = bool(reasons)
            updated = {
                **state,
                "needs_review": needs_review,
                "review_reason": "; ".join(reasons),
            }
            logger.info("hitl_route row_id=%s needs_review=%s reason=%s", row_id, needs_review, "; ".join(reasons))
            if needs_review:
                append_review_item(
                    config.hitl_queue,
                    {
                        "text": state["text"],
                        "normalized_text": state.get("normalized_text", ""),
                        "label": state.get("label", ""),
                        "label_name": state.get("label_name", ""),
                        "confidence": state.get("confidence", ""),
                        "primary_topic": state.get("primary_topic", ""),
                        "topic_tags": state.get("topic_tags", []),
                        "review_reason": "; ".join(reasons),
                        "explanation": state.get("explanation", ""),
                        "retrieved_examples": state.get("retrieved_examples", []),
                    },
                )
            return updated

    workflow = StateGraph(ClassificationState)
    workflow.add_node("normalization", normalize_node)
    workflow.add_node("syntax", syntax_node)
    workflow.add_node("retrieval", retrieve_node)
    workflow.add_node("reasoning", reason_node)
    workflow.add_node("reflection", reflection_node)
    workflow.add_node("hitl_routing", route_node)

    workflow.set_entry_point("normalization")
    workflow.add_edge("normalization", "syntax")
    workflow.add_edge("syntax", "retrieval")
    workflow.add_edge("retrieval", "reasoning")
    workflow.add_conditional_edges("reasoning", should_reflect, {"reflection": "reflection", "hitl_routing": "hitl_routing"})
    workflow.add_edge("reflection", "hitl_routing")
    workflow.add_edge("hitl_routing", END)
    return workflow.compile()


def classify_text(text: str, config: AppConfig | None = None) -> ClassificationState:
    graph = build_graph(config or AppConfig())
    return graph.invoke({"text": text})


def _parse_llm_json(raw: str) -> dict[str, Any]:
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        start = raw.find("{")
        end = raw.rfind("}")
        if start == -1 or end == -1 or end <= start:
            raise
        parsed = json.loads(raw[start : end + 1])
    if not isinstance(parsed, dict):
        raise ValueError("LLM returned JSON, but it was not an object.")
    return parsed


def _coerce_label(value: Any) -> int:
    try:
        return 1 if int(value) == 1 else 0
    except (TypeError, ValueError):
        return 0


def _coerce_confidence(value: Any) -> float:
    try:
        confidence = float(value)
    except (TypeError, ValueError):
        return 0.0
    return max(0.0, min(1.0, confidence))


def _clean_topic(value: Any) -> str:
    allowed = {
        "politics",
        "government",
        "religion",
        "sports",
        "entertainment",
        "caste_or_community",
        "gender_or_sexuality",
        "regional_or_nationality",
        "personal_abuse",
        "general",
        "unclear",
    }
    topic = str(value or "").strip().lower().replace(" ", "_").replace("-", "_")
    return topic if topic in allowed else "unclear"


def _clean_topic_tags(value: Any) -> list[str]:
    if isinstance(value, str):
        raw_tags = [item.strip() for item in value.split(",")]
    elif isinstance(value, list):
        raw_tags = value
    else:
        raw_tags = []

    tags: list[str] = []
    for item in raw_tags:
        tag = _clean_topic(item)
        if tag and tag not in tags:
            tags.append(tag)
    return tags or ["unclear"]
