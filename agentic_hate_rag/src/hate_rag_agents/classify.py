from __future__ import annotations

import argparse
import json
import os
import os
from pathlib import Path

import requests
import pandas as pd
from dotenv import load_dotenv
from httpx import ConnectError as HttpxConnectError, RequestError as HttpxRequestError

from .config import AppConfig, resolve_path
from .eval import EvalTracker, generate_eval_summary, print_eval_summary
from .graph import build_graph
from .io_utils import DEFAULT_LABEL_COLUMNS, DEFAULT_TEXT_COLUMNS, detect_column, read_table, write_table
from .labels import normalize_label
from .logging_utils import get_app_logger, log_timing, setup_app_logging
from .ollama_health import check_ollama, format_ollama_error
from .rag_store import auto_ingest_row, make_vector_store
from .reflection import LessonStore, analyze_errors

from langchain_ollama import ChatOllama
from .vllm_health import check_vllm, format_vllm_error


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Classify code-mixed comments with the LangGraph RAG agent.")
    parser.add_argument("--text", help="Classify a single comment.")
    parser.add_argument("--input", help="CSV/XLSX file to classify.")
    parser.add_argument("--output", help="Output CSV/XLSX path for file mode.")
    parser.add_argument("--env-file", help="Optional env file to load before building config.")
    parser.add_argument("--text-column", help="Text column for file mode.")
    parser.add_argument("--label-column", help="Optional gold label column to copy and compare in file mode.")
    parser.add_argument("--limit", type=int, help="Optional row limit for smoke tests.")
    parser.add_argument("--llm-provider", choices=["ollama", "vllm"], help="Which LLM backend to use.")
    parser.add_argument("--model", help="Override the active LLM model name.")
    parser.add_argument("--ollama-base-url", help="Override Ollama base URL.")
    parser.add_argument("--vllm-base-url", help="Override vLLM base URL.")
    parser.add_argument("--vllm-api-key", help="Optional bearer token for vLLM OpenAI-compatible serving.")
    parser.add_argument("--skip-llm-check", action="store_true", help="Skip the startup LLM health check.")
    parser.add_argument("--skip-ollama-check", action="store_true", help="Backward-compatible alias for --skip-llm-check.")
    parser.add_argument("--chroma-dir", help="Override Chroma persistence directory.")
    parser.add_argument("--collection", help="Override Chroma collection name.")
    parser.add_argument("--confidence-threshold", type=float, help="Override HITL threshold.")
    parser.add_argument("--save-every", type=int, default=1, help="Write output progress after this many classified rows.")
    # New flags for the auto-eval reflection loop
    parser.add_argument("--no-reflection", action="store_true", help="Disable the reflection loop for low-confidence rows.")
    parser.add_argument("--no-auto-ingest", action="store_true", help="Disable auto-ingestion of high-confidence verified rows.")
    parser.add_argument("--clear-lessons", action="store_true", help="Clear learned lessons from previous runs before starting.")
    parser.add_argument("--eval-batch-size", type=int, help="Override batch size for error analysis (default: 50).")
    parser.add_argument("--save-every", type=int, default=1, help="Write output progress after this many classified rows.")
    parser.add_argument("--log-file", help="Path to app log file. Defaults to APP_LOG or ./outputs/app.log.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.text and not args.input:
        raise SystemExit("Pass either --text or --input.")
    if args.env_file:
        load_dotenv(args.env_file, override=True)

    config = make_config(args)
    print_classification_config(config, args)
    logger = setup_app_logging(config.app_log)
    logger.info("classify_start input=%s output=%s", args.input or "", args.output or "")
    print_classification_config(config, args)
    if not (args.skip_llm_check or args.skip_ollama_check):
        ensure_llm_health(config, logger)

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

        # --- Auto-eval setup ---
        has_ground_truth = label_col is not None
        tracker = EvalTracker() if has_ground_truth else None
        eval_batch_size = args.eval_batch_size or config.eval_batch_size

        # --- Lesson store setup ---
        lessons_path = resolve_path(config.lessons_path)
        lesson_store = LessonStore.load(lessons_path)
        if args.clear_lessons:
            lesson_store.lessons.clear()
            lesson_store.save()
            print("Cleared all previously learned lessons.", flush=True)
        if lesson_store.lessons:
            print(f"Loaded {len(lesson_store.lessons)} lessons from previous runs.", flush=True)

        # --- Auto-ingest setup ---
        auto_ingest_enabled = has_ground_truth and not args.no_auto_ingest
        vector_store = None
        if auto_ingest_enabled:
            vector_store = make_vector_store(
                persist_directory=config.chroma_dir,
                collection_name=config.chroma_collection,
                embedding_model=config.muril_model,
            )
        auto_ingested_count = 0

        # --- LLM for error analysis ---
        analysis_llm = None
        if has_ground_truth:
            analysis_llm = ChatOllama(
                model=config.ollama_model,
                base_url=config.ollama_base_url,
                temperature=0,
                format="json",
            )

        batch_start_idx = 0

        for completed, (idx, row) in enumerate(df.iterrows(), start=1):
            text = "" if pd.isna(row.get(text_col)) else str(row.get(text_col))

            # Inject current lessons into the graph invocation
            invoke_state = {"text": text, "lessons_block": lesson_store.as_prompt_block()}
            result = graph.invoke(invoke_state)
            item = public_result(result)
            write_result_to_row(output_df, idx, item)

            # --- Auto-eval: compare with ground truth ---
            if has_ground_truth and label_col and tracker is not None:
                gold_label = output_df.at[idx, "gold_hate_label"]
                if gold_label != "" and not pd.isna(gold_label):
                    gold_int = int(gold_label)
                    match = item["label"] == gold_int
                    output_df.at[idx, "agent_label_matches_gold"] = match
                    tracker.add(
                        gold_label=gold_int,
                        predicted_label=item["label"],
                        confidence=item["confidence"],
                        text=text,
                        normalized_text=item["normalized_text"],
                        explanation=item["explanation"],
                        row_index=idx,
                    )

                    # --- Auto-ingest: high confidence + correct ---
                    if auto_ingest_enabled and vector_store is not None and match:
                        if item["confidence"] >= config.auto_ingest_threshold:
                            auto_ingest_row(
                                vector_store,
                                text=item["normalized_text"],
                                original_text=text,
                                label=item["label"],
                                source=f"auto_ingest:{input_path.stem}",
                                row_index=str(idx),
                            )
                            auto_ingested_count += 1

            print(
                f"Classified {completed}/{total_rows} row {idx}: "
                f"label={item['label']} confidence={item['confidence']:.2f} "
                f"topic={item['primary_topic']}"
                f"{' [reflected]' if item.get('reflection_used') else ''}",
                flush=True,
            )

            # --- Periodic save ---
            if completed % save_every == 0 or completed == total_rows:
                write_table(output_df, output_path)
                print(f"Saved progress: {completed}/{total_rows} rows to {output_path}", flush=True)

            # --- Batch error analysis: learn from mistakes ---
            if (
                has_ground_truth
                and tracker is not None
                and analysis_llm is not None
                and completed % eval_batch_size == 0
                and completed > 0
            ):
                batch_misclassified = [
                    m for m in tracker.misclassified
                    if isinstance(m.get("row_index"), int) and m["row_index"] >= batch_start_idx
                ]
                if batch_misclassified:
                    print(
                        f"\n--- Batch error analysis (rows {batch_start_idx}-{idx}, "
                        f"{len(batch_misclassified)} errors) ---",
                        flush=True,
                    )
                    new_lessons = analyze_errors(analysis_llm, batch_misclassified)
                    if new_lessons:
                        lesson_store.add_lessons(new_lessons)
                        print(f"Learned {len(new_lessons)} new lessons:", flush=True)
                        for lesson in new_lessons:
                            print(f"  • {lesson}", flush=True)
                    else:
                        print("No new lessons extracted from this batch.", flush=True)

                    # Print running metrics
                    running = tracker.metrics()
                    print(
                        f"Running metrics: accuracy={running['accuracy']:.2%} "
                        f"F1={running['f1']:.2%} "
                        f"precision={running['precision']:.2%} "
                        f"recall={running['recall']:.2%}",
                        flush=True,
                    )
                    print("---\n", flush=True)
                batch_start_idx = idx + 1 if isinstance(idx, int) else batch_start_idx + eval_batch_size

        # --- Final save ---
        write_table(output_df, output_path)
        print(f"\nWrote {len(output_df)} rows to {output_path}")

        # --- Final eval summary ---
        if tracker is not None and tracker.total > 0:
            eval_summary_path = output_path.with_name(f"{output_path.stem}_eval_summary.json")
            summary = generate_eval_summary(tracker, eval_summary_path)
            print_eval_summary(summary)
            print(f"Eval summary written to {eval_summary_path}")

        if auto_ingested_count > 0:
            print(f"Auto-ingested {auto_ingested_count} high-confidence verified rows into ChromaDB.")

        if lesson_store.lessons:
            print(f"Total lessons learned: {len(lesson_store.lessons)} (saved to {lessons_path})")

    except (HttpxConnectError, HttpxRequestError, requests.RequestException) as exc:
        logger.exception("classify_http_error")
        error_message = (
            format_vllm_error(config.vllm_base_url, config.vllm_model)
            if config.llm_provider == "vllm"
            else format_ollama_error(config.ollama_base_url, config.ollama_model)
        )
        raise SystemExit(error_message + f"\n\nRuntime error: {type(exc).__name__}: {exc}") from exc


def make_config(args: argparse.Namespace) -> AppConfig:
    base = AppConfig()
    llm_provider = (args.llm_provider or base.llm_provider).strip().lower()
    if llm_provider not in {"ollama", "vllm"}:
        raise SystemExit(f"Unsupported --llm-provider {llm_provider!r}. Use 'ollama' or 'vllm'.")
    values = {
        "llm_provider": llm_provider,
        "ollama_model": (args.model if llm_provider == "ollama" and args.model else base.ollama_model),
        "ollama_base_url": args.ollama_base_url or base.ollama_base_url,
        "ollama_timeout_seconds": base.ollama_timeout_seconds,
        "vllm_model": (args.model if llm_provider == "vllm" and args.model else base.vllm_model),
        "vllm_base_url": args.vllm_base_url or base.vllm_base_url,
        "vllm_api_key": args.vllm_api_key or base.vllm_api_key,
        "vllm_timeout_seconds": base.vllm_timeout_seconds,
        "muril_model": base.muril_model,
        "chroma_dir": resolve_path(args.chroma_dir) if args.chroma_dir else base.chroma_dir,
        "chroma_collection": args.collection or base.chroma_collection,
        "rag_top_k": base.rag_top_k,
        "confidence_threshold": args.confidence_threshold if args.confidence_threshold is not None else base.confidence_threshold,
        "hitl_queue": base.hitl_queue,
        "app_log": resolve_path(args.log_file) if args.log_file else base.app_log,
        "temperature": base.temperature,
        "reflection_enabled": base.reflection_enabled and not args.no_reflection,
        "max_reflection_retries": base.max_reflection_retries,
        "max_retrieval_distance": base.max_retrieval_distance,
        "auto_ingest_threshold": base.auto_ingest_threshold,
        "eval_batch_size": args.eval_batch_size or base.eval_batch_size,
        "lessons_path": base.lessons_path,
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
    print(f"  Reflection enabled: {config.reflection_enabled}", flush=True)
    print(f"  Max reflection retries: {config.max_reflection_retries}", flush=True)
    print(f"  Max retrieval distance: {config.max_retrieval_distance}", flush=True)
    print(f"  Auto-ingest threshold: {config.auto_ingest_threshold}", flush=True)
    print(f"  Auto-ingest enabled: {not args.no_auto_ingest}", flush=True)
    print(f"  Eval batch size: {config.eval_batch_size}", flush=True)
    print(f"  Lessons path: {config.lessons_path}", flush=True)


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
        "agent_reflection_used",
        "agent_reflection_note",
    ]
    for column in output_columns:
        output_df[column] = ""
    if label_col:
        output_df["gold_hate_label"] = [normalize_label(value) for value in df[label_col]]
        output_df["agent_label_matches_gold"] = ""
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
    output_df.at[idx, "agent_reflection_used"] = item.get("reflection_used", False)
    output_df.at[idx, "agent_reflection_note"] = item.get("reflection_note", "")


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
        "reflection_used": result.get("reflection_used", False),
        "reflection_note": result.get("reflection_note", ""),
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
