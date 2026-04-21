import argparse
import json
import math
import re
import time
import unicodedata
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import pandas as pd
import requests


DEFAULT_TEXT_COLUMNS = [
    "Comment",
    "comment",
    "text",
    "Text",
    "sentence",
    "Sentence",
    "content",
    "Content",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Classify Telugu or mixed-language comments into hate/non-hate using a local Ollama model."
    )
    parser.add_argument(
        "--input",
        required=True,
        help="Path to an input CSV/XLSX file or to a folder containing CSV/XLSX files.",
    )
    parser.add_argument(
        "--output",
        help=(
            "Path to the output CSV file for single-file mode, or an output folder for folder mode. "
            "Defaults to <input_stem>_classified.csv for files and <input>/classified_outputs for folders."
        ),
    )
    parser.add_argument(
        "--text-column",
        help="Name of the text column. If omitted, the script tries common column names.",
    )
    parser.add_argument(
        "--model",
        required=True,
        help="Ollama model name, for example llama3.1:8b or mistral:latest.",
    )
    parser.add_argument(
        "--ollama-url",
        default="http://127.0.0.1:11434/api/generate",
        help="Ollama generate endpoint.",
    )
    parser.add_argument(
        "--batch-char-budget",
        type=int,
        default=2200,
        help="Approximate total normalized characters to include in each model call.",
    )
    parser.add_argument(
        "--max-batch-size",
        type=int,
        default=20,
        help="Maximum rows to send in one model call.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="Sampling temperature for Ollama.",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=180,
        help="HTTP timeout in seconds for each Ollama request.",
    )
    parser.add_argument(
        "--sleep-seconds",
        type=float,
        default=0.25,
        help="Pause between model requests.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        help="Optional limit for quick testing.",
    )
    parser.add_argument(
        "--recursive",
        action="store_true",
        help="When --input is a folder, include files from subfolders as well.",
    )
    return parser.parse_args()


def read_table(path: Path) -> pd.DataFrame:
    suffix = path.suffix.lower()
    if suffix == ".csv":
        return pd.read_csv(path, encoding="utf-8-sig")
    if suffix in {".xlsx", ".xls"}:
        return pd.read_excel(path)
    raise ValueError(f"Unsupported file type: {suffix}")


def is_supported_input_file(path: Path) -> bool:
    return path.is_file() and path.suffix.lower() in {".csv", ".xlsx", ".xls"}


def discover_input_files(path: Path, recursive: bool) -> List[Path]:
    if path.is_file():
        if not is_supported_input_file(path):
            raise ValueError(f"Unsupported input file: {path}")
        return [path]

    if not path.is_dir():
        raise FileNotFoundError(f"Input path not found: {path}")

    pattern = "**/*" if recursive else "*"
    files = [item for item in path.glob(pattern) if is_supported_input_file(item)]
    files.sort()

    if not files:
        raise ValueError(f"No CSV/XLSX files found in folder: {path}")

    return files


def detect_text_column(df: pd.DataFrame, requested: Optional[str]) -> str:
    if requested:
        if requested not in df.columns:
            raise ValueError(f"Text column '{requested}' was not found in the input file.")
        return requested

    for column in DEFAULT_TEXT_COLUMNS:
        if column in df.columns:
            return column

    if len(df.columns) == 1:
        return str(df.columns[0])

    raise ValueError(
        "Could not infer the text column. Pass --text-column with the correct column name."
    )


def normalize_text(text: Any) -> str:
    if text is None or (isinstance(text, float) and math.isnan(text)):
        return ""

    value = str(text)
    value = unicodedata.normalize("NFKC", value)
    value = value.replace("\r\n", "\n").replace("\r", "\n")
    value = re.sub(r"https?://\S+|www\.\S+", " [URL] ", value, flags=re.IGNORECASE)
    value = re.sub(r"@\w+", " [MENTION] ", value)
    value = value.replace("&amp;", "&")
    value = expand_emojis(value)
    value = re.sub(r"[ \t]+", " ", value)
    value = re.sub(r"\n{2,}", "\n", value)
    value = re.sub(r"\s+([,.!?;:])", r"\1", value)
    value = re.sub(r"([!?.,])\1{2,}", r"\1\1", value)
    return value.strip()


def expand_emojis(text: str) -> str:
    parts: List[str] = []
    for ch in text:
        if is_emoji_like(ch):
            name = emoji_name(ch)
            if name:
                parts.append(f" <emoji: {name}> ")
            else:
                parts.append(" <emoji> ")
        else:
            parts.append(ch)
    expanded = "".join(parts)
    expanded = re.sub(r"\s+", " ", expanded)
    return expanded.strip()


def is_emoji_like(ch: str) -> bool:
    if ch in {"\u200d", "\ufe0f"}:
        return False
    codepoint = ord(ch)
    return (
        0x1F000 <= codepoint <= 0x1FAFF
        or 0x2600 <= codepoint <= 0x27BF
        or unicodedata.category(ch) == "So"
    )


def emoji_name(ch: str) -> str:
    try:
        raw_name = unicodedata.name(ch)
    except ValueError:
        return ""

    cleaned = raw_name.lower().replace("_", " ")
    cleaned = cleaned.replace("face with", "")
    cleaned = cleaned.replace("black", "")
    cleaned = cleaned.replace("white", "")
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    return cleaned


def build_prompt(batch: Sequence[Dict[str, Any]]) -> str:
    instructions = """
You are a hate-speech classification system for Telugu and Telugu-mixed social media comments.

Task:
- Classify each comment as `1` if it contains hateful, abusive, threatening, violent, dehumanizing, or group-targeted hate speech.
- Classify each comment as `0` if it is non-hate, including disagreement, criticism, sarcasm, political opinion, frustration, or offensive language that does not target a protected or vulnerable group.

Important guidance:
- The comments may contain Telugu script, English, transliterated Telugu, slang, spelling variation, repeated letters, and emojis.
- Emojis were converted into short English expressions like `<emoji: smiling face with hearts>` before sending them to you. Use those emoji meanings as additional context.
- Consider the whole meaning, not just individual bad words.
- Do not over-label. Return `1` only when the text clearly expresses hate speech or targeted hateful abuse.
- If the meaning is unclear or not explicitly hateful, prefer `0`.

Output rules:
- Return valid JSON only.
- Return one top-level object with a single key named `results`.
- `results` must be an array with one object per input comment.
- Each object must contain:
  - `id`: the exact input id string
  - `label`: integer 0 or 1

Example output:
{"results":[{"id":"1","label":0},{"id":"2","label":1}]}
""".strip()

    payload = []
    for item in batch:
        payload.append(
            {
                "id": str(item["id"]),
                "comment": item["normalized_text"],
            }
        )

    return f"{instructions}\n\nInput comments:\n{json.dumps(payload, ensure_ascii=False, indent=2)}"


def create_batches(records: Sequence[Dict[str, Any]], char_budget: int, max_batch_size: int) -> List[List[Dict[str, Any]]]:
    batches: List[List[Dict[str, Any]]] = []
    current: List[Dict[str, Any]] = []
    current_chars = 0

    for record in records:
        size = max(1, len(record["normalized_text"]))
        would_overflow = current and (
            len(current) >= max_batch_size or current_chars + size > char_budget
        )
        if would_overflow:
            batches.append(current)
            current = []
            current_chars = 0

        current.append(record)
        current_chars += size

    if current:
        batches.append(current)

    return batches


def call_ollama(
    url: str,
    model: str,
    prompt: str,
    temperature: float,
    timeout: int,
) -> Dict[str, Any]:
    response = requests.post(
        url,
        json={
            "model": model,
            "prompt": prompt,
            "stream": False,
            "format": "json",
            "options": {
                "temperature": temperature,
            },
        },
        timeout=timeout,
    )
    response.raise_for_status()
    return response.json()


def parse_model_results(raw_response: Dict[str, Any], expected_ids: Sequence[str]) -> Dict[str, int]:
    model_text = raw_response.get("response", "")
    if not isinstance(model_text, str) or not model_text.strip():
        raise ValueError("Ollama response did not contain a non-empty JSON string in `response`.")

    parsed = json.loads(model_text)
    results = parsed.get("results")
    if not isinstance(results, list):
        raise ValueError("Model JSON does not contain a `results` array.")

    label_map: Dict[str, int] = {}
    for item in results:
        if not isinstance(item, dict):
            raise ValueError("Each result item must be a JSON object.")

        item_id = str(item.get("id", "")).strip()
        label = item.get("label")

        if item_id == "":
            raise ValueError("A result item is missing `id`.")
        if label not in (0, 1):
            raise ValueError(f"Invalid label for id {item_id!r}: {label!r}")

        label_map[item_id] = int(label)

    missing = [item_id for item_id in expected_ids if item_id not in label_map]
    extra = [item_id for item_id in label_map if item_id not in expected_ids]

    if missing or extra:
        raise ValueError(
            f"Model output ids do not match input ids. Missing={missing}, Extra={extra}"
        )

    return label_map


def classify_batches(
    batches: Sequence[Sequence[Dict[str, Any]]],
    model: str,
    url: str,
    temperature: float,
    timeout: int,
    sleep_seconds: float,
) -> Tuple[Dict[str, int], Dict[str, str]]:
    labels: Dict[str, int] = {}
    raw_outputs: Dict[str, str] = {}

    total_batches = len(batches)
    for index, batch in enumerate(batches, start=1):
        prompt = build_prompt(batch)
        expected_ids = [str(item["id"]) for item in batch]

        last_error: Optional[Exception] = None
        raw_response: Optional[Dict[str, Any]] = None

        for attempt in range(2):
            try:
                raw_response = call_ollama(
                    url=url,
                    model=model,
                    prompt=prompt,
                    temperature=temperature,
                    timeout=timeout,
                )
                batch_labels = parse_model_results(raw_response, expected_ids)
                labels.update(batch_labels)
                raw_text = raw_response.get("response", "")
                for item_id in expected_ids:
                    raw_outputs[item_id] = raw_text
                print(f"Processed batch {index}/{total_batches} with {len(batch)} rows.")
                break
            except Exception as exc:
                last_error = exc
                if attempt == 1:
                    timeout_hint = ""
                    if "Read timed out" in str(exc) or "timed out" in str(exc).lower():
                        timeout_hint = (
                            " Ollama is reachable, but the model did not respond before the timeout. "
                            "Try a smaller model, increase --timeout, or reduce --max-batch-size "
                            "and --batch-char-budget."
                        )
                    raise RuntimeError(
                        f"Batch {index}/{total_batches} failed after retry: {exc}.{timeout_hint}"
                    ) from exc
                time.sleep(1.0)

        if raw_response is None and last_error is not None:
            raise last_error

        if sleep_seconds > 0:
            time.sleep(sleep_seconds)

    return labels, raw_outputs


def prepare_records(df: pd.DataFrame, text_column: str) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    for idx, row in df.iterrows():
        original = row.get(text_column, "")
        normalized = normalize_text(original)
        records.append(
            {
                "id": str(idx),
                "row_index": idx,
                "original_text": "" if pd.isna(original) else str(original),
                "normalized_text": normalized,
            }
        )
    return records


def split_empty_records(
    records: Sequence[Dict[str, Any]],
) -> Tuple[List[Dict[str, Any]], Dict[str, int], Dict[str, str]]:
    non_empty: List[Dict[str, Any]] = []
    preset_labels: Dict[str, int] = {}
    preset_raw_outputs: Dict[str, str] = {}

    for record in records:
        item_id = str(record["id"])
        if record["normalized_text"].strip():
            non_empty.append(record)
        else:
            preset_labels[item_id] = 0
            preset_raw_outputs[item_id] = '{"results":[{"id":"' + item_id + '","label":0}]}'

    return non_empty, preset_labels, preset_raw_outputs


def main() -> None:
    args = parse_args()
    input_path = Path(args.input)
    input_files = discover_input_files(input_path, recursive=args.recursive)

    if input_path.is_file():
        output_paths = [
            Path(args.output)
            if args.output
            else input_path.with_name(f"{input_path.stem}_classified.csv")
        ]
    else:
        output_dir = (
            Path(args.output)
            if args.output
            else input_path / "classified_outputs"
        )
        output_dir.mkdir(parents=True, exist_ok=True)
        output_paths = [output_dir / f"{file_path.stem}_classified.csv" for file_path in input_files]

    total_rows = 0
    total_batches = 0

    for file_index, (source_path, output_path) in enumerate(zip(input_files, output_paths), start=1):
        print(f"Starting file {file_index}/{len(input_files)}: {source_path}")

        df = read_table(source_path)
        if args.limit:
            df = df.head(args.limit).copy()

        text_column = detect_text_column(df, args.text_column)
        records = prepare_records(df, text_column)
        records_for_model, preset_labels, preset_raw_outputs = split_empty_records(records)
        batches = create_batches(
            records_for_model,
            char_budget=args.batch_char_budget,
            max_batch_size=args.max_batch_size,
        )

        labels, raw_outputs = classify_batches(
            batches=batches,
            model=args.model,
            url=args.ollama_url,
            temperature=args.temperature,
            timeout=args.timeout,
            sleep_seconds=args.sleep_seconds,
        )
        labels.update(preset_labels)
        raw_outputs.update(preset_raw_outputs)

        result_df = df.copy()
        result_df["normalized_text"] = [record["normalized_text"] for record in records]
        result_df["hate_label"] = [labels[str(record["id"])] for record in records]
        result_df["ollama_raw_response"] = [raw_outputs[str(record["id"])] for record in records]
        result_df.to_csv(output_path, index=False, encoding="utf-8-sig")

        total_rows += len(result_df)
        total_batches += len(batches)

        print(f"Wrote {len(result_df)} rows to {output_path}")
        print(f"Detected text column: {text_column}")
        print(f"Total batches sent for this file: {len(batches)}")

    print(f"Finished processing {len(input_files)} file(s).")
    print(f"Total rows written: {total_rows}")
    print(f"Total model batches sent: {total_batches}")


if __name__ == "__main__":
    main()
