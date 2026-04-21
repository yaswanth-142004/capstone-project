from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from .config import AppConfig, resolve_path
from .graph import build_graph
from .io_utils import DEFAULT_TEXT_COLUMNS, detect_column, read_table, write_table


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Classify code-mixed comments with the LangGraph RAG agent.")
    parser.add_argument("--text", help="Classify a single comment.")
    parser.add_argument("--input", help="CSV/XLSX file to classify.")
    parser.add_argument("--output", help="Output CSV/XLSX path for file mode.")
    parser.add_argument("--text-column", help="Text column for file mode.")
    parser.add_argument("--limit", type=int, help="Optional row limit for smoke tests.")
    parser.add_argument("--model", help="Override Ollama model.")
    parser.add_argument("--ollama-base-url", help="Override Ollama base URL.")
    parser.add_argument("--chroma-dir", help="Override Chroma persistence directory.")
    parser.add_argument("--collection", help="Override Chroma collection name.")
    parser.add_argument("--confidence-threshold", type=float, help="Override HITL threshold.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.text and not args.input:
        raise SystemExit("Pass either --text or --input.")

    config = make_config(args)
    graph = build_graph(config)

    if args.text:
        result = graph.invoke({"text": args.text})
        print(json.dumps(public_result(result), ensure_ascii=False, indent=2))
        return

    input_path = resolve_path(args.input)
    output_path = resolve_path(args.output) if args.output else input_path.with_name(f"{input_path.stem}_agentic_classified.csv")
    df = read_table(input_path)
    if args.limit:
        df = df.head(args.limit).copy()
    text_col = detect_column(df, args.text_column, DEFAULT_TEXT_COLUMNS, "text")

    records = []
    for idx, row in df.iterrows():
        text = "" if pd.isna(row.get(text_col)) else str(row.get(text_col))
        result = graph.invoke({"text": text})
        records.append(public_result(result))
        print(f"Classified row {idx}: label={result.get('label')} confidence={result.get('confidence')}")

    output_df = df.copy()
    output_df["agent_normalized_text"] = [item["normalized_text"] for item in records]
    output_df["agent_hate_label"] = [item["label"] for item in records]
    output_df["agent_label_name"] = [item["label_name"] for item in records]
    output_df["agent_confidence"] = [item["confidence"] for item in records]
    output_df["agent_needs_review"] = [item["needs_review"] for item in records]
    output_df["agent_explanation"] = [item["explanation"] for item in records]
    output_df["agent_retrieved_examples"] = [
        json.dumps(item["retrieved_examples"], ensure_ascii=False) for item in records
    ]
    output_df["agent_raw_response"] = [item["raw_response"] for item in records]
    write_table(output_df, output_path)
    print(f"Wrote {len(output_df)} rows to {output_path}")


def make_config(args: argparse.Namespace) -> AppConfig:
    base = AppConfig()
    values = {
        "ollama_model": args.model or base.ollama_model,
        "ollama_base_url": args.ollama_base_url or base.ollama_base_url,
        "muril_model": base.muril_model,
        "chroma_dir": resolve_path(args.chroma_dir) if args.chroma_dir else base.chroma_dir,
        "chroma_collection": args.collection or base.chroma_collection,
        "rag_top_k": base.rag_top_k,
        "confidence_threshold": args.confidence_threshold if args.confidence_threshold is not None else base.confidence_threshold,
        "hitl_queue": base.hitl_queue,
        "temperature": base.temperature,
    }
    return AppConfig(**values)


def public_result(result: dict) -> dict:
    return {
        "text": result.get("text", ""),
        "normalized_text": result.get("normalized_text", ""),
        "label": result.get("label", 0),
        "label_name": result.get("label_name", ""),
        "confidence": result.get("confidence", 0.0),
        "languages": result.get("languages", []),
        "needs_review": result.get("needs_review", False),
        "review_reason": result.get("review_reason", ""),
        "explanation": result.get("explanation", ""),
        "signals": result.get("signals", []),
        "syntax_report": result.get("syntax_report", {}),
        "retrieved_examples": result.get("retrieved_examples", []),
        "raw_response": result.get("raw_response", ""),
    }


if __name__ == "__main__":
    main()
