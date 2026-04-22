from __future__ import annotations

import json
from typing import Any, Literal, TypedDict

from langchain_ollama import ChatOllama
from langgraph.graph import END, StateGraph

from .config import AppConfig
from .hitl import append_review_item
from .normalization import normalize_for_analysis
from .prompts import build_reasoning_prompt
from .rag_store import (
    auto_ingest_row,
    make_vector_store,
    retrieve_diverse_examples,
    retrieve_examples,
)
from .reflection import build_reflection_prompt
from .syntax import analyze_syntax


class ClassificationState(TypedDict, total=False):
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
    explanation: str
    signals: list[str]
    needs_review: bool
    review_reason: str
    # Reflection loop fields
    reflection_count: int
    first_pass_label: int
    first_pass_confidence: float
    first_pass_explanation: str
    reflection_used: bool
    reflection_note: str
    # Lessons injected from error analysis
    lessons_block: str


def build_graph(config: AppConfig):
    vector_store = make_vector_store(
        persist_directory=config.chroma_dir,
        collection_name=config.chroma_collection,
        embedding_model=config.muril_model,
    )
    llm = ChatOllama(
        model=config.ollama_model,
        base_url=config.ollama_base_url,
        temperature=config.temperature,
        format="json",
    )

    def normalize_node(state: ClassificationState) -> ClassificationState:
        result = normalize_for_analysis(state["text"])
        return {
            **state,
            "cleaned_text": result.cleaned_text,
            "normalized_text": result.transliterated_text or result.cleaned_text,
            "transliteration_backend": result.transliteration_backend,
        }

    def syntax_node(state: ClassificationState) -> ClassificationState:
        report = analyze_syntax(state.get("normalized_text", ""))
        return {**state, "syntax_report": report.as_dict()}

    def retrieve_node(state: ClassificationState) -> ClassificationState:
        query = state.get("normalized_text") or state["text"]
        examples = retrieve_examples(
            vector_store,
            query=query,
            top_k=config.rag_top_k,
            max_distance=config.max_retrieval_distance,
        )
        return {**state, "retrieved_examples": examples}

    def reason_node(state: ClassificationState) -> ClassificationState:
        prompt = build_reasoning_prompt(
            original_text=state["text"],
            normalized_text=state.get("normalized_text", ""),
            syntax_report=state.get("syntax_report", {}),
            retrieved_examples=state.get("retrieved_examples", []),
            lessons_block=state.get("lessons_block", ""),
        )
        response = llm.invoke(prompt)
        raw = str(response.content)
        parsed = _parse_llm_json(raw)
        label = int(parsed.get("label", 0))
        confidence = float(parsed.get("confidence", 0.0))
        return {
            **state,
            "raw_response": raw,
            "label": 1 if label == 1 else 0,
            "label_name": parsed.get("label_name") or ("Offensive" if label == 1 else "Non-Offensive"),
            "confidence": max(0.0, min(1.0, confidence)),
            "languages": parsed.get("languages", []),
            "explanation": parsed.get("rationale", ""),
            "signals": parsed.get("signals", []),
            "reflection_count": state.get("reflection_count", 0),
            "reflection_used": False,
        }

    def should_reflect(state: ClassificationState) -> str:
        """Conditional edge: decide whether to reflect or proceed to HITL routing."""
        if not config.reflection_enabled:
            return "hitl_routing"
        reflection_count = state.get("reflection_count", 0)
        if reflection_count >= config.max_reflection_retries:
            return "hitl_routing"
        if state.get("confidence", 0.0) < config.confidence_threshold:
            return "reflection"
        return "hitl_routing"

    def reflection_node(state: ClassificationState) -> ClassificationState:
        """Re-retrieve with diversity and ask the LLM to reconsider its initial answer."""
        query = state.get("normalized_text") or state["text"]
        new_examples = retrieve_diverse_examples(
            vector_store,
            query=query,
            total=config.rag_top_k,
            fetch_k=config.rag_top_k * 2,
            max_distance=config.max_retrieval_distance,
        )

        first_label = state.get("label", 0)
        first_confidence = state.get("confidence", 0.0)
        first_explanation = state.get("explanation", "")

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
        response = llm.invoke(prompt)
        raw = str(response.content)
        parsed = _parse_llm_json(raw)
        new_label = int(parsed.get("label", 0))
        new_confidence = float(parsed.get("confidence", 0.0))

        return {
            **state,
            "raw_response": raw,
            "label": 1 if new_label == 1 else 0,
            "label_name": parsed.get("label_name") or ("Offensive" if new_label == 1 else "Non-Offensive"),
            "confidence": max(0.0, min(1.0, new_confidence)),
            "languages": parsed.get("languages", []),
            "explanation": parsed.get("rationale", ""),
            "signals": parsed.get("signals", []),
            "first_pass_label": first_label,
            "first_pass_confidence": first_confidence,
            "first_pass_explanation": first_explanation,
            "reflection_count": state.get("reflection_count", 0) + 1,
            "reflection_used": True,
            "reflection_note": parsed.get("reflection_note", ""),
            "retrieved_examples": new_examples,
        }

    def route_node(state: ClassificationState) -> ClassificationState:
        reasons: list[str] = []
        if state.get("confidence", 0.0) < config.confidence_threshold:
            reasons.append("low confidence")
        if state.get("syntax_report", {}).get("suspicious_word_salad"):
            reasons.append("possible evasion or low-structure text")
        if not state.get("retrieved_examples"):
            reasons.append("no retrieved context")

        needs_review = bool(reasons)
        updated = {
            **state,
            "needs_review": needs_review,
            "review_reason": "; ".join(reasons),
        }
        if needs_review:
            append_review_item(
                config.hitl_queue,
                {
                    "text": state["text"],
                    "normalized_text": state.get("normalized_text", ""),
                    "label": state.get("label", ""),
                    "label_name": state.get("label_name", ""),
                    "confidence": state.get("confidence", ""),
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
    # Conditional edge: after reasoning, either reflect or go to HITL routing
    workflow.add_conditional_edges("reasoning", should_reflect, {"reflection": "reflection", "hitl_routing": "hitl_routing"})
    # After reflection, go back to HITL routing (not another reasoning pass)
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
