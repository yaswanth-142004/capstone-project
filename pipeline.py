"""
End-to-end Telugu-English code-mixed ASR data pipeline.

The pipeline can process an existing WAV + Telugu VTT pair, a single YouTube
URL, or a CSV of YouTube URLs through download, subtitle cleanup,
transliteration, loanword tagging, tag refinement, and audio chunk export.
"""
import argparse
import csv
import json
import os
import re
from pathlib import Path

import webvtt
from aksharamukha import transliterate as aksh
from pydub import AudioSegment


def time_to_ms(ts):
    """HH:MM:SS.mmm -> milliseconds"""
    h, m, rest = ts.split(":")
    if "." in rest:
        s, ms = rest.split(".")
    else:
        s, ms = rest.split(",")
    return int(h) * 3600000 + int(m) * 60000 + int(s) * 1000 + int(ms)


# Step 0: Clean raw YouTube VTT
def _caption_duration_ms(caption):
    return time_to_ms(caption.end) - time_to_ms(caption.start)


def _strip_youtube_markup(text):
    text = re.sub(r"<[^>]+>", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _remove_repeated_prefix(text, previous_text):
    """
    YouTube auto-captions often keep the previous caption as visual context:
    "old words\nnew words". For ASR labels, keep only the newly spoken part.
    """
    if not previous_text:
        return text

    if text == previous_text:
        return ""

    if text.startswith(previous_text + " "):
        return text[len(previous_text):].strip()

    previous_words = previous_text.split()
    words = text.split()
    max_overlap = min(len(previous_words), len(words))
    for size in range(max_overlap, 0, -1):
        if words[:size] == previous_words[-size:]:
            return " ".join(words[size:]).strip()

    return text


def clean_vtt(input_vtt, output_vtt, min_duration_ms=250):
    """
    Strip YouTube auto-caption markup, short captions, and repeated context text.
    """
    vtt = webvtt.read(input_vtt)
    cleaned = []
    prev_text = ""

    for caption in vtt:
        if _caption_duration_ms(caption) < min_duration_ms:
            continue

        text = _strip_youtube_markup(caption.text)
        text = _remove_repeated_prefix(text, prev_text)

        if not text or text == prev_text:
            continue

        cleaned.append({
            "start": caption.start,
            "end": caption.end,
            "text": text,
        })
        prev_text = text

    with open(output_vtt, "w", encoding="utf-8") as f:
        f.write("WEBVTT\n\n")
        for i, caption in enumerate(cleaned):
            f.write(f"{i + 1}\n")
            f.write(f"{caption['start']} --> {caption['end']}\n")
            f.write(f"{caption['text']}\n\n")

    print(f"[Clean] {len(cleaned)} unique captions -> {output_vtt}")
    return output_vtt


# Step 1: Transliterate Telugu -> Latin
def transliterate_vtt(input_vtt, output_vtt, target_scheme="ISO"):
    """Convert Telugu script in VTT to a Latin transliteration scheme."""
    vtt = webvtt.read(input_vtt)
    for caption in vtt:
        caption.text = aksh.process("Telugu", target_scheme, caption.text)
    vtt.save(output_vtt)
    print(f"[Transliterate] Saved -> {output_vtt}")
    return output_vtt


_TOKEN_RE = re.compile(r"^([^\w\[]*)([\w\u00c0-\u024f]+)([^\w\]]*)$")
_TAGGED_TOKEN_RE = re.compile(r"^([^\[]*)\[([^\]]+)\]([^\]]*)$")


def _tag_token(token):
    if token.startswith("[") and token.endswith("]"):
        return token, False

    match = _TOKEN_RE.match(token)
    if not match:
        return token, False

    prefix, core, suffix = match.groups()

    from fuzzy_loanword_detector import detect_loanword
    from loanword_dict import lookup

    english = lookup(core) or detect_loanword(core)
    if not english:
        return token, False

    return f"{prefix}[{english}]{suffix}", True


# Step 2: Normalize and tag English loanwords
def normalize_and_tag_vtt(input_vtt, output_vtt):
    """
    Identify English loanwords using dictionary lookup plus the fuzzy detector.
    Tags detected English words with brackets: [word].
    """
    vtt = webvtt.read(input_vtt)
    tagged_count = 0

    for caption in vtt:
        tagged = []
        for word in caption.text.split():
            tagged_word, was_tagged = _tag_token(word)
            tagged.append(tagged_word)
            if was_tagged:
                tagged_count += 1
        caption.text = " ".join(tagged)

    vtt.save(output_vtt)
    print(f"[Tag] {tagged_count} English loanwords tagged -> {output_vtt}")
    return output_vtt


# False positives: English words that are common Telugu words or artifacts.
_FALSE_POSITIVES = {
    "trot", "van", "inn", "pal", "dam", "den", "fin", "fur",
    "hen", "hut", "mud", "nun", "ore", "peg", "rag", "sap",
    "tar", "urn", "vat", "wad", "yam", "pan", "mat", "pat",
    "nit", "pun", "cot", "dip", "jot", "pod", "rot",
    "rut", "sod", "tot", "vet", "wit",
}

_CORRECTIONS = {
    "praises": "prices",
    "said": "side",
    "pas": "pass",
    "fails": "files",
    "fast": "first",
    "trot": None,
}


def _refine_tagged_token(token):
    match = _TAGGED_TOKEN_RE.match(token)
    if not match:
        return token, "kept"

    prefix, english, suffix = match.groups()
    normalized = english.lower()

    if normalized in _FALSE_POSITIVES:
        return f"{prefix}{english}{suffix}", "removed"

    if normalized in _CORRECTIONS:
        correction = _CORRECTIONS[normalized]
        if correction is None:
            return f"{prefix}{english}{suffix}", "removed"
        return f"{prefix}[{correction}]{suffix}", "corrected"

    return token, "kept"


def refine_tags_vtt(input_vtt, output_vtt):
    """
    Remove known false-positive tags and apply known transliteration corrections.
    """
    vtt = webvtt.read(input_vtt)
    removed = 0
    corrected = 0

    for caption in vtt:
        refined = []
        for word in caption.text.split():
            refined_word, action = _refine_tagged_token(word)
            refined.append(refined_word)
            if action == "removed":
                removed += 1
            elif action == "corrected":
                corrected += 1
        caption.text = " ".join(refined)

    vtt.save(output_vtt)
    print(
        f"[Refine] {removed} false positives removed, "
        f"{corrected} corrections applied -> {output_vtt}"
    )
    return output_vtt


# Step 3: Chunk audio and text into dataset
def _write_jsonl(path, records):
    with open(path, "w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def _write_csv(path, records):
    if not records:
        return
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(records[0].keys()))
        writer.writeheader()
        writer.writerows(records)


def _save_hf_dataset(path, records):
    if not records:
        return
    try:
        from datasets import Audio, Dataset
    except ImportError:
        print("[Dataset] Hugging Face datasets is not installed; skipped hf_dataset export")
        return

    hf_records = [
        {
            "audio": record["audio_path"],
            "text": record["text"],
            "duration_s": record["duration_s"],
            "video_id": record["video_id"],
            "start": record["start"],
            "end": record["end"],
        }
        for record in records
    ]
    dataset = Dataset.from_list(hf_records)
    dataset = dataset.cast_column("audio", Audio(sampling_rate=16000))
    dataset.save_to_disk(path)
    print(f"[Dataset] Hugging Face dataset saved to {path}")


def chunk_and_export(
    audio_path,
    vtt_path,
    output_dir,
    min_duration_ms=500,
    max_duration_ms=30000,
    video_id=None,
):
    """Slice audio by VTT timestamps and export dataset manifests."""
    os.makedirs(output_dir, exist_ok=True)
    chunk_dir = os.path.join(output_dir, "audio_chunks")
    os.makedirs(chunk_dir, exist_ok=True)

    audio = AudioSegment.from_wav(audio_path)
    vtt = webvtt.read(vtt_path)

    records = []
    chunk_idx = 0
    for caption in vtt:
        start = time_to_ms(caption.start)
        end = time_to_ms(caption.end)
        duration_ms = end - start
        text = caption.text.strip()
        if not text or duration_ms < min_duration_ms or duration_ms > max_duration_ms:
            continue

        chunk = audio[start:end]
        fname = f"chunk_{chunk_idx:05d}.wav"
        fpath = os.path.join(chunk_dir, fname)
        chunk.export(fpath, format="wav")

        records.append({
            "audio": os.path.abspath(fpath),
            "audio_path": os.path.abspath(fpath),
            "audio_filepath": os.path.abspath(fpath),
            "text": text,
            "sentence": text,
            "start": caption.start,
            "end": caption.end,
            "duration_s": round(duration_ms / 1000.0, 2),
            "video_id": video_id or Path(output_dir).name,
        })
        chunk_idx += 1

    meta_path = os.path.join(output_dir, "metadata.json")
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(records, f, indent=2, ensure_ascii=False)

    _write_jsonl(os.path.join(output_dir, "metadata.jsonl"), records)
    _write_csv(os.path.join(output_dir, "metadata.csv"), records)
    _save_hf_dataset(os.path.join(output_dir, "hf_dataset"), records)

    print(f"[Chunk] {len(records)} chunks saved to {chunk_dir}")
    print(f"[Chunk] Metadata at {meta_path}")
    return records


def run_pipeline(audio_path, vtt_path, output_dir="data/processed", target_scheme="ISO"):
    os.makedirs(output_dir, exist_ok=True)
    base = os.path.splitext(os.path.basename(vtt_path))[0]
    video_id = Path(audio_path).stem

    print("=" * 60)
    print("  Telugu-English Code-Mixed ASR Pipeline")
    print("=" * 60)

    cleaned_vtt = os.path.join(output_dir, f"{base}_clean.vtt")
    clean_vtt(vtt_path, cleaned_vtt)

    translit_vtt = os.path.join(output_dir, f"{base}_translit.vtt")
    transliterate_vtt(cleaned_vtt, translit_vtt, target_scheme=target_scheme)

    tagged_vtt = os.path.join(output_dir, f"{base}_tagged.vtt")
    normalize_and_tag_vtt(translit_vtt, tagged_vtt)

    refined_vtt = os.path.join(output_dir, f"{base}_refined.vtt")
    refine_tags_vtt(tagged_vtt, refined_vtt)

    records = chunk_and_export(audio_path, refined_vtt, output_dir, video_id=video_id)

    print("=" * 60)
    print(f"  Pipeline complete! {len(records)} samples generated.")
    print("=" * 60)
    return records


def _read_urls(csv_path, url_column="url"):
    with open(csv_path, "r", encoding="utf-8-sig", newline="") as f:
        first_line = f.readline()
        f.seek(0)
        first_cells = next(csv.reader([first_line])) if first_line else []
        has_header = bool(first_cells and first_cells[0].strip() == url_column)
        if has_header:
            reader = csv.DictReader(f)
            if url_column not in (reader.fieldnames or []):
                raise ValueError(
                    f"CSV does not contain a '{url_column}' column. "
                    f"Found: {reader.fieldnames}"
                )
            for row in reader:
                url = (row.get(url_column) or "").strip()
                if url:
                    yield url
        else:
            for row in csv.reader(f):
                if row and row[0].strip():
                    yield row[0].strip()


def process_youtube_url(url, raw_dir="data/raw", processed_dir="data/processed"):
    from data_collection import download_youtube_video

    all_records = []
    assets = download_youtube_video(url, raw_dir)
    if not assets:
        print(f"[URL] Skipping {url}: no WAV + Telugu VTT pair found")
        return all_records

    for asset in assets:
        video_id = asset["video_id"]
        out_dir = processed_dir
        if len(assets) > 1:
            out_dir = os.path.join(processed_dir, video_id)
        records = run_pipeline(asset["audio_path"], asset["vtt_path"], out_dir)
        all_records.extend(records)

    return all_records


def process_youtube_csv(csv_path, raw_dir="data/raw", processed_dir="data/processed", url_column="url"):
    from data_collection import download_youtube_video

    all_records = []
    for url in _read_urls(csv_path, url_column=url_column):
        print(f"[CSV] Processing {url}")
        assets = download_youtube_video(url, raw_dir)
        if not assets:
            print(f"[CSV] Skipping {url}: no WAV + Telugu VTT pair found")
            continue

        for asset in assets:
            video_id = asset["video_id"]
            out_dir = os.path.join(processed_dir, video_id)
            records = run_pipeline(asset["audio_path"], asset["vtt_path"], out_dir)
            all_records.extend(records)

    os.makedirs(processed_dir, exist_ok=True)
    combined_json = os.path.join(processed_dir, "all_metadata.json")
    with open(combined_json, "w", encoding="utf-8") as f:
        json.dump(all_records, f, indent=2, ensure_ascii=False)
    _write_jsonl(os.path.join(processed_dir, "all_metadata.jsonl"), all_records)
    _write_csv(os.path.join(processed_dir, "all_metadata.csv"), all_records)
    _save_hf_dataset(os.path.join(processed_dir, "hf_dataset"), all_records)

    print(f"[CSV] Combined dataset records: {len(all_records)}")
    print(f"[CSV] Combined manifest: {combined_json}")
    return all_records


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run the full Telugu-English code-mixed ASR pipeline"
    )
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument("--url", help="YouTube video or playlist URL")
    input_group.add_argument("--csv", help="CSV of YouTube URLs")
    input_group.add_argument("--audio", help="Path to WAV audio file")
    parser.add_argument("--vtt", help="Path to Telugu VTT subtitle file")
    parser.add_argument("--url-column", default="url", help="URL column name for --csv files")
    parser.add_argument("--raw-dir", default="data/raw", help="Raw download directory")
    parser.add_argument("--out", default="data/processed", help="Output directory")
    parser.add_argument(
        "--target-scheme",
        default="ISO",
        help="Aksharamukha target transliteration scheme",
    )
    args = parser.parse_args()

    if args.url:
        process_youtube_url(args.url, raw_dir=args.raw_dir, processed_dir=args.out)
    elif args.csv:
        process_youtube_csv(
            args.csv,
            raw_dir=args.raw_dir,
            processed_dir=args.out,
            url_column=args.url_column,
        )
    else:
        if not args.vtt:
            parser.error("--vtt is required when using --audio")
        run_pipeline(args.audio, args.vtt, args.out, target_scheme=args.target_scheme)
