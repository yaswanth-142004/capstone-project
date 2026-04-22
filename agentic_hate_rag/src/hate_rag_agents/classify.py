from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import requests
import pandas as pd
from httpx import ConnectError as HttpxConnectError, RequestError as HttpxRequestError

from .config import AppConfig, resolve_path
from .graph import build_graph
from .io_utils import DEFAULT_LABEL_COLUMNS, DEFAULT_TEXT_COLUMNS, detect_column, read_table, write_table
from .labels import normalize_label
from .ollama_health import check_ollama, format_ollama_error


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Classify code-mixed comments with the LangGraph RAG agent.")
    parser.add_argument("--text", help="Classify a single comment.")
    parser.add_argument("--input", help="CSV/XLSX file to classify.")
    parser.add_argument("--output", help="Output CSV/XLSX path for file mode.")
    parser.add_argument("--text-column", help="Text column for file mode.")
    parser.add_argument("--label-column", help="Optional gold label column to copy and compare in file mode.")
    parser.add_argument("--limit", type=int, help="Optional row limit for smoke tests.")
    parser.add_argument("--model", help="Override Ollama model.")
    parser.add_argument("--ollama-base-url", help="Override Ollama base URL.")
    parser.add_argument("--skip-ollama-check", action="store_true", help="Skip the startup Ollama health check.")
    parser.add_argument("--chroma-dir", help="Override Chroma persistence directory.")
    parser.add_argument("--collection", help="Override Chroma collection name.")
    parser.add_argument("--confidence-threshold", type=float, help="Override HITL threshold.")
    parser.add_argument("--save-every", type=int, default=1, help="Write output progress after this many classified rows.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.text and not args.input:
        raise SystemExit("Pass either --text or --input.")

    config = make_config(args)
    print_classification_config(config, args)
    if not args.skip_ollama_check:
        health = check_ollama(config.ollama_base_url, config.ollama_model)
        if not health.reachable or not health.model_available:
            raise SystemExit(format_ollama_error(config.ollama_base_url, config.ollama_model, health))

    graph = build_graph(config)

    try:
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
        label_col = detect_optional_column(df, args.label_column, DEFAULT_LABEL_COLUMNS, "label")

        output_df = initialize_output_df(df, label_col)
        total_rows = len(output_df)
        save_every = max(1, args.save_every)

        for completed, (idx, row) in enumerate(df.iterrows(), start=1):
            text = "" if pd.isna(row.get(text_col)) else str(row.get(text_col))
            result = graph.invoke({"text": text})
            item = public_result(result)
            write_result_to_row(output_df, idx, item)
            if label_col:
                gold_label = output_df.at[idx, "gold_hate_label"]
                output_df.at[idx, "agent_label_matches_gold"] = (
                    item["label"] == gold_label if gold_label is not None and not pd.isna(gold_label) else ""
                )
            print(
                f"Classified {completed}/{total_rows} row {idx}: "
                f"label={item['label']} confidence={item['confidence']} topic={item['primary_topic']}",
                flush=True,
            )

            if completed % save_every == 0 or completed == total_rows:
                write_table(output_df, output_path)
                print(f"Saved progress: {completed}/{total_rows} rows to {output_path}", flush=True)

        write_table(output_df, output_path)
        print(f"Wrote {len(output_df)} rows to {output_path}")
    except (HttpxConnectError, HttpxRequestError, requests.RequestException) as exc:
        raise SystemExit(
            format_ollama_error(config.ollama_base_url, config.ollama_model)
            + f"\n\nRuntime error: {type(exc).__name__}: {exc}"
        ) from exc


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


def print_classification_config(config: AppConfig, args: argparse.Namespace) -> None:
    requested_device = os.getenv("MURIL_DEVICE", "auto")
    resolved_device, cuda_note = resolve_muril_device_for_display(requested_device)
    input_mode = "single text" if args.text else "file"
    print("Classification configuration:", flush=True)
    print(f"  Input mode: {input_mode}", flush=True)
    if args.input:
        print(f"  Input: {resolve_path(args.input)}", flush=True)
    if args.output:
        print(f"  Output: {resolve_path(args.output)}", flush=True)
    print(f"  Ollama model: {config.ollama_model}", flush=True)
    print(f"  Ollama base URL: {config.ollama_base_url}", flush=True)
    print(f"  MuRIL model: {config.muril_model}", flush=True)
    print(f"  MuRIL device requested: {requested_device}", flush=True)
    print(f"  MuRIL device resolved: {resolved_device}", flush=True)
    if cuda_note:
        print(f"  MuRIL device note: {cuda_note}", flush=True)
    print(f"  Chroma directory: {config.chroma_dir}", flush=True)
    print(f"  Chroma collection: {config.chroma_collection}", flush=True)
    print(f"  RAG top-k: {config.rag_top_k}", flush=True)
    print(f"  Confidence threshold: {config.confidence_threshold}", flush=True)
    print(f"  HITL queue: {config.hitl_queue}", flush=True)
    print(f"  Save every: {max(1, args.save_every)} row(s)", flush=True)


def resolve_muril_device_for_display(requested_device: str) -> tuple[str, str]:
    try:
        import torch
    except ImportError:
        return "unknown", "PyTorch is not installed."

    requested = requested_device.strip().lower()
    cuda_available = torch.cuda.is_available()
    if requested == "auto":
        if cuda_available:
            return f"cuda ({torch.cuda.get_device_name(0)})", ""
        return "cpu", "CUDA is not available in this Python environment, so MuRIL will run on CPU."
    if requested == "cuda" and not cuda_available:
        return "unavailable", "MURIL_DEVICE=cuda was requested, but this PyTorch install cannot see CUDA."
    if requested == "cuda":
        return f"cuda ({torch.cuda.get_device_name(0)})", ""
    return requested, ""


def initialize_output_df(df: pd.DataFrame, label_col: str | None) -> pd.DataFrame:
    output_df = df.copy()
    output_columns = [
        "agent_normalized_text",
        "agent_hate_label",
        "agent_label_name",
        "agent_confidence",
        "agent_primary_topic",
        "agent_topic_tags",
        "agent_needs_review",
        "agent_review_reason",
        "agent_explanation",
        "agent_retrieved_examples",
        "agent_raw_response",
    ]
    for column in output_columns:
        output_df[column] = pd.Series([None] * len(output_df), index=output_df.index, dtype="object")
    if label_col:
        output_df["gold_hate_label"] = pd.Series(
            [normalize_label(value) for value in df[label_col]],
            index=output_df.index,
            dtype="object",
        )
        output_df["agent_label_matches_gold"] = pd.Series([None] * len(output_df), index=output_df.index, dtype="object")
    return output_df


def write_result_to_row(output_df: pd.DataFrame, idx, item: dict) -> None:
    output_df.at[idx, "agent_normalized_text"] = item["normalized_text"]
    output_df.at[idx, "agent_hate_label"] = item["label"]
    output_df.at[idx, "agent_label_name"] = item["label_name"]
    output_df.at[idx, "agent_confidence"] = item["confidence"]
    output_df.at[idx, "agent_primary_topic"] = item["primary_topic"]
    output_df.at[idx, "agent_topic_tags"] = "|".join(item["topic_tags"])
    output_df.at[idx, "agent_needs_review"] = item["needs_review"]
    output_df.at[idx, "agent_review_reason"] = item["review_reason"]
    output_df.at[idx, "agent_explanation"] = item["explanation"]
    output_df.at[idx, "agent_retrieved_examples"] = json.dumps(item["retrieved_examples"], ensure_ascii=False)
    output_df.at[idx, "agent_raw_response"] = item["raw_response"]


def public_result(result: dict) -> dict:
    return {
        "text": result.get("text", ""),
        "normalized_text": result.get("normalized_text", ""),
        "label": result.get("label", 0),
        "label_name": result.get("label_name", ""),
        "confidence": result.get("confidence", 0.0),
        "languages": result.get("languages", []),
        "primary_topic": result.get("primary_topic", "unclear"),
        "topic_tags": result.get("topic_tags", ["unclear"]),
        "needs_review": result.get("needs_review", False),
        "review_reason": result.get("review_reason", ""),
        "explanation": result.get("explanation", ""),
        "signals": result.get("signals", []),
        "syntax_report": result.get("syntax_report", {}),
        "retrieved_examples": result.get("retrieved_examples", []),
        "raw_response": result.get("raw_response", ""),
    }


def detect_optional_column(
    df: pd.DataFrame,
    requested: str | None,
    candidates: list[str],
    purpose: str,
) -> str | None:
    if requested:
        if requested not in df.columns:
            raise ValueError(f"{purpose} column {requested!r} was not found.")
        return requested
    for column in candidates:
        if column in df.columns:
            return column
    return None


if __name__ == "__main__":
    main()
