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
from .logging_utils import get_app_logger, log_timing, setup_app_logging
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
    parser.add_argument("--log-file", help="Path to app log file. Defaults to APP_LOG or ./outputs/app.log.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.text and not args.input:
        raise SystemExit("Pass either --text or --input.")

    config = make_config(args)
    logger = setup_app_logging(config.app_log)
    logger.info("classify_start input=%s output=%s", args.input or "", args.output or "")
    print_classification_config(config, args)
    if not args.skip_ollama_check:
        with log_timing("ollama_health_check", model=config.ollama_model, base_url=config.ollama_base_url):
            health = check_ollama(config.ollama_base_url, config.ollama_model)
        if not health.reachable or not health.model_available:
            logger.error("ollama_health_failed reachable=%s model_available=%s", health.reachable, health.model_available)
            raise SystemExit(format_ollama_error(config.ollama_base_url, config.ollama_model, health))

    with log_timing("build_graph", chroma_dir=config.chroma_dir, collection=config.chroma_collection):
        graph = build_graph(config)

    try:
        if args.text:
            with log_timing("classify_single_text", chars=len(args.text)):
                result = graph.invoke({"text": args.text, "row_id": "single"})
            print(json.dumps(public_result(result), ensure_ascii=False, indent=2))
            logger.info("classify_done mode=single_text")
            return

        input_path = resolve_path(args.input)
        output_path = resolve_path(args.output) if args.output else input_path.with_name(f"{input_path.stem}_agentic_classified.csv")
        with log_timing("read_input", path=input_path):
            df = read_table(input_path)
        if args.limit:
            df = df.head(args.limit).copy()
        text_col = detect_column(df, args.text_column, DEFAULT_TEXT_COLUMNS, "text")
        label_col = detect_optional_column(df, args.label_column, DEFAULT_LABEL_COLUMNS, "label")

        output_df = initialize_output_df(df, label_col)
        total_rows = len(output_df)
        save_every = max(1, args.save_every)
        logger.info("classify_file_ready rows=%s text_column=%s label_column=%s save_every=%s", total_rows, text_col, label_col or "", save_every)

        for completed, (idx, row) in enumerate(df.iterrows(), start=1):
            text = "" if pd.isna(row.get(text_col)) else str(row.get(text_col))
            logger.info("row_start completed=%s total=%s row_index=%s chars=%s", completed, total_rows, idx, len(text))
            try:
                with log_timing("row_graph_invoke", completed=completed, total=total_rows, row_index=idx, chars=len(text)):
                    result = graph.invoke({"text": text, "row_id": str(idx)})
                item = public_result(result)
            except Exception as exc:
                logger.exception("row_failed completed=%s row_index=%s", completed, idx)
                item = error_result(text, exc)
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
            logger.info(
                "row_done completed=%s total=%s row_index=%s label=%s confidence=%s topic=%s review=%s error=%s",
                completed,
                total_rows,
                idx,
                item["label"],
                item["confidence"],
                item["primary_topic"],
                item["needs_review"],
                bool(item["error"]),
            )

            if completed % save_every == 0 or completed == total_rows:
                with log_timing("save_progress", completed=completed, total=total_rows, path=output_path):
                    write_table(output_df, output_path)
                print(f"Saved progress: {completed}/{total_rows} rows to {output_path}", flush=True)

        with log_timing("write_final_output", rows=len(output_df), path=output_path):
            write_table(output_df, output_path)
        print(f"Wrote {len(output_df)} rows to {output_path}")
        logger.info("classify_done mode=file rows=%s output=%s", len(output_df), output_path)
    except (HttpxConnectError, HttpxRequestError, requests.RequestException) as exc:
        logger.exception("classify_http_error")
        raise SystemExit(
            format_ollama_error(config.ollama_base_url, config.ollama_model)
            + f"\n\nRuntime error: {type(exc).__name__}: {exc}"
        ) from exc


def make_config(args: argparse.Namespace) -> AppConfig:
    base = AppConfig()
    values = {
        "ollama_model": args.model or base.ollama_model,
        "ollama_base_url": args.ollama_base_url or base.ollama_base_url,
        "ollama_timeout_seconds": base.ollama_timeout_seconds,
        "muril_model": base.muril_model,
        "chroma_dir": resolve_path(args.chroma_dir) if args.chroma_dir else base.chroma_dir,
        "chroma_collection": args.collection or base.chroma_collection,
        "rag_top_k": base.rag_top_k,
        "confidence_threshold": args.confidence_threshold if args.confidence_threshold is not None else base.confidence_threshold,
        "hitl_queue": base.hitl_queue,
        "app_log": resolve_path(args.log_file) if args.log_file else base.app_log,
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
    print(f"  Ollama timeout: {config.ollama_timeout_seconds}s", flush=True)
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
    print(f"  App log: {config.app_log}", flush=True)
    print(f"  Save every: {max(1, args.save_every)} row(s)", flush=True)
    logger = get_app_logger()
    for line in [
        f"config input_mode={input_mode}",
        f"config ollama_model={config.ollama_model}",
        f"config ollama_base_url={config.ollama_base_url}",
        f"config ollama_timeout_seconds={config.ollama_timeout_seconds}",
        f"config muril_model={config.muril_model}",
        f"config muril_device_requested={requested_device}",
        f"config muril_device_resolved={resolved_device}",
        f"config chroma_dir={config.chroma_dir}",
        f"config chroma_collection={config.chroma_collection}",
        f"config rag_top_k={config.rag_top_k}",
        f"config confidence_threshold={config.confidence_threshold}",
        f"config hitl_queue={config.hitl_queue}",
        f"config app_log={config.app_log}",
        f"config save_every={max(1, args.save_every)}",
    ]:
        logger.info(line)


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
        "agent_error",
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
    output_df.at[idx, "agent_error"] = item["error"]


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
        "error": result.get("parse_error", ""),
    }


def error_result(text: str, exc: Exception) -> dict:
    return {
        "text": text,
        "normalized_text": text,
        "label": 0,
        "label_name": "Unclear",
        "confidence": 0.0,
        "languages": [],
        "primary_topic": "unclear",
        "topic_tags": ["unclear"],
        "needs_review": True,
        "review_reason": "classification runtime error",
        "explanation": f"Classification failed for this row: {type(exc).__name__}",
        "signals": ["classification_error"],
        "syntax_report": {},
        "retrieved_examples": [],
        "raw_response": "",
        "error": f"{type(exc).__name__}: {exc}",
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
