from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import requests
from dotenv import load_dotenv
from httpx import ConnectError as HttpxConnectError, RequestError as HttpxRequestError

from .classify import ensure_llm_health
from .config import AppConfig, resolve_path
from .eval import EvalTracker, generate_eval_summary, print_eval_summary
from .io_utils import read_table
from .labels import normalize_label
from .llm_clients import build_llm
from .logging_utils import setup_app_logging
from .reflection import LessonStore, analyze_errors


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run auto-eval and lesson extraction on an already-classified file.")
    parser.add_argument("--input", required=True, help="Classified CSV/XLSX file containing predicted and gold labels.")
    parser.add_argument("--output-summary", help="Optional JSON path for evaluation summary.")
    parser.add_argument("--env-file", help="Optional env file to load before building config.")
    parser.add_argument("--gold-column", help="Gold label column. Defaults to gold_hate_label or hate_label.")
    parser.add_argument("--pred-column", help="Predicted label column. Defaults to agent_hate_label.")
    parser.add_argument("--confidence-column", help="Confidence column. Defaults to agent_confidence.")
    parser.add_argument("--text-column", help="Original text column. Defaults to Comment, text, or normalized_text.")
    parser.add_argument("--normalized-text-column", help="Normalized text column. Defaults to agent_normalized_text or normalized_text.")
    parser.add_argument("--explanation-column", help="Explanation column. Defaults to agent_explanation.")
    parser.add_argument("--llm-provider", choices=["ollama", "vllm"], help="Which LLM backend to use for lesson extraction.")
    parser.add_argument("--model", help="Override the active LLM model name.")
    parser.add_argument("--ollama-base-url", help="Override Ollama base URL.")
    parser.add_argument("--vllm-base-url", help="Override vLLM base URL.")
    parser.add_argument("--vllm-api-key", help="Optional bearer token for vLLM OpenAI-compatible serving.")
    parser.add_argument("--skip-llm-check", action="store_true", help="Skip the startup LLM health check.")
    parser.add_argument("--skip-lessons", action="store_true", help="Only compute the eval summary without lesson extraction.")
    parser.add_argument("--clear-lessons", action="store_true", help="Clear learned lessons before extracting new ones.")
    parser.add_argument("--eval-batch-size", type=int, help="Override batch size for error analysis.")
    parser.add_argument("--lessons-path", help="Override where learned lessons are written.")
    parser.add_argument("--log-file", help="Path to app log file. Defaults to APP_LOG or ./outputs/app.log.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.env_file:
        load_dotenv(args.env_file, override=True)

    config = make_autoloop_config(args)

    logger = setup_app_logging(config.app_log)
    input_path = resolve_path(args.input)
    logger.info("autoloop_start input=%s", input_path)

    df = read_table(input_path)
    gold_col = detect_required_column(df, args.gold_column, ["gold_hate_label", "hate_label"], "gold label")
    pred_col = detect_required_column(df, args.pred_column, ["agent_hate_label"], "predicted label")
    confidence_col = detect_required_column(df, args.confidence_column, ["agent_confidence"], "confidence")
    text_col = detect_optional_column(df, args.text_column, ["Comment", "comment", "text", "Text", "normalized_text"])
    normalized_text_col = detect_optional_column(df, args.normalized_text_column, ["agent_normalized_text", "normalized_text"])
    explanation_col = detect_optional_column(df, args.explanation_column, ["agent_explanation"])

    tracker = build_tracker(
        df=df,
        gold_col=gold_col,
        pred_col=pred_col,
        confidence_col=confidence_col,
        text_col=text_col,
        normalized_text_col=normalized_text_col,
        explanation_col=explanation_col,
    )
    if tracker.total == 0:
        raise SystemExit("No valid rows found with both gold and predicted labels.")

    summary_path = resolve_summary_path(input_path, args.output_summary)
    summary = generate_eval_summary(tracker, summary_path)
    print_eval_summary(summary)
    print(f"Eval summary written to {summary_path}", flush=True)

    if args.skip_lessons:
        logger.info("autoloop_done mode=summary_only total=%s", tracker.total)
        return

    if not args.skip_llm_check:
        ensure_llm_health(config, logger)
    llm = build_llm(config)

    lesson_store = LessonStore.load(resolve_path(config.lessons_path))
    if args.clear_lessons:
        lesson_store.lessons.clear()
        lesson_store.save()
        print("Cleared all previously learned lessons.", flush=True)

    batch_size = args.eval_batch_size or config.eval_batch_size
    total_batches = (tracker.total + batch_size - 1) // batch_size
    print(
        f"Starting lesson extraction across {total_batches} batch(es) with batch size {batch_size}.",
        flush=True,
    )
    for batch_index, start in enumerate(range(0, tracker.total, batch_size), start=1):
        end = min(start + batch_size, tracker.total)
        batch_misclassified = tracker.pop_recent_misclassified(start)
        batch_misclassified = [item for item in batch_misclassified if item.get("tracker_index", -1) < end]
        if not batch_misclassified:
            print(
                f"Batch {batch_index}/{total_batches} (tracker rows {start}-{end - 1}): no misclassifications, skipping.",
                flush=True,
            )
            continue
        print(
            f"\n--- Batch {batch_index}/{total_batches} error analysis "
            f"(tracker rows {start}-{end - 1}, {len(batch_misclassified)} errors) ---",
            flush=True,
        )
        new_lessons = analyze_errors(llm, batch_misclassified)
        if new_lessons:
            lesson_store.add_lessons(new_lessons)
            print(f"Learned {len(new_lessons)} new lessons:", flush=True)
            for lesson in new_lessons:
                print(f"  - {lesson}", flush=True)
        else:
            print("No new lessons extracted from this batch.", flush=True)
        print(
            f"Finished batch {batch_index}/{total_batches}. Running lesson count: {len(lesson_store.lessons)}",
            flush=True,
        )

    print(f"Total lessons learned: {len(lesson_store.lessons)} (saved to {lesson_store.path})", flush=True)
    logger.info("autoloop_done total=%s lessons=%s summary=%s", tracker.total, len(lesson_store.lessons), summary_path)


def build_tracker(
    *,
    df: pd.DataFrame,
    gold_col: str,
    pred_col: str,
    confidence_col: str,
    text_col: str | None,
    normalized_text_col: str | None,
    explanation_col: str | None,
) -> EvalTracker:
    tracker = EvalTracker()
    for idx, row in df.iterrows():
        gold_label = normalize_label(row.get(gold_col))
        pred_label = normalize_label(row.get(pred_col))
        if gold_label is None or pred_label is None:
            continue
        confidence = safe_float(row.get(confidence_col))
        text = "" if text_col is None or pd.isna(row.get(text_col)) else str(row.get(text_col))
        normalized_text = "" if normalized_text_col is None or pd.isna(row.get(normalized_text_col)) else str(row.get(normalized_text_col))
        explanation = "" if explanation_col is None or pd.isna(row.get(explanation_col)) else str(row.get(explanation_col))
        tracker.add(
            gold_label=gold_label,
            predicted_label=pred_label,
            confidence=confidence,
            text=text,
            normalized_text=normalized_text,
            explanation=explanation,
            row_index=idx,
        )
    return tracker


def make_autoloop_config(args: argparse.Namespace) -> AppConfig:
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
        "chroma_dir": base.chroma_dir,
        "chroma_collection": base.chroma_collection,
        "rag_top_k": base.rag_top_k,
        "confidence_threshold": base.confidence_threshold,
        "hitl_queue": base.hitl_queue,
        "app_log": resolve_path(args.log_file) if args.log_file else base.app_log,
        "temperature": base.temperature,
        "reflection_enabled": base.reflection_enabled,
        "max_reflection_retries": base.max_reflection_retries,
        "max_retrieval_distance": base.max_retrieval_distance,
        "auto_ingest_threshold": base.auto_ingest_threshold,
        "eval_batch_size": args.eval_batch_size or base.eval_batch_size,
        "lessons_path": resolve_path(args.lessons_path) if args.lessons_path else base.lessons_path,
    }
    return AppConfig(**values)


def detect_required_column(df: pd.DataFrame, requested: str | None, candidates: list[str], purpose: str) -> str:
    if requested:
        if requested not in df.columns:
            raise SystemExit(f"{purpose.title()} column {requested!r} was not found.")
        return requested
    for column in candidates:
        if column in df.columns:
            return column
    raise SystemExit(f"Could not infer {purpose} column. Pass it explicitly.")


def detect_optional_column(df: pd.DataFrame, requested: str | None, candidates: list[str]) -> str | None:
    if requested:
        if requested not in df.columns:
            raise SystemExit(f"Column {requested!r} was not found.")
        return requested
    for column in candidates:
        if column in df.columns:
            return column
    return None


def resolve_summary_path(input_path: Path, output_summary: str | None) -> Path:
    if output_summary:
        return resolve_path(output_summary)
    return input_path.with_name(f"{input_path.stem}_eval_summary.json")


def safe_float(value) -> float:
    try:
        if pd.isna(value):
            return 0.0
        return float(value)
    except (TypeError, ValueError):
        return 0.0


if __name__ == "__main__":
    try:
        main()
    except (HttpxConnectError, HttpxRequestError, requests.RequestException) as exc:
        raise SystemExit(f"Auto-loop failed due to LLM connectivity error: {type(exc).__name__}: {exc}") from exc
