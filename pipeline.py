"""
End-to-end pipeline for Telugu-English Code-Mixed ASR data preparation.

Usage:
  # From a YouTube URL (downloads audio + subtitles automatically):
  python pipeline.py --url "https://www.youtube.com/watch?v=VIDEO_ID" --out data/processed

  # From pre-downloaded files:
  python pipeline.py --audio data/raw/video.wav --vtt data/raw/video.te.vtt --out data/processed
"""
import os
import re
import json
import argparse
import webvtt
from pydub import AudioSegment
from aksharamukha import transliterate as aksh


# ─── Step 0: Download from YouTube ────────────────────────────────────────────
def download_from_youtube(url, raw_dir="data/raw"):
    """Download audio + Telugu subtitles using yt-dlp. Returns (wav_path, vtt_path)."""
    from data_collection import download_youtube_video
    return download_youtube_video(url, raw_dir)


# ─── Step 1: Clean raw YouTube VTT ────────────────────────────────────────────
def clean_vtt(input_vtt, output_vtt):
    """
    YouTube auto-generated VTT has duplicate lines and inline <c> tags.
    Strips them and merges sequential duplicate captions.
    """
    vtt = webvtt.read(input_vtt)
    cleaned = []
    prev_text = ""

    for caption in vtt:
        # Strip HTML-like tags  e.g. <c>, </c>, <00:00:02.159>
        text = re.sub(r'<[^>]+>', '', caption.text).strip()
        # Collapse whitespace
        text = re.sub(r'\s+', ' ', text).strip()

        if not text or text == prev_text:
            continue

        cleaned.append({
            "start": caption.start,
            "end": caption.end,
            "text": text
        })
        prev_text = text

    # Write cleaned VTT
    with open(output_vtt, 'w', encoding='utf-8') as f:
        f.write("WEBVTT\n\n")
        for i, c in enumerate(cleaned):
            f.write(f"{i+1}\n")
            f.write(f"{c['start']} --> {c['end']}\n")
            f.write(f"{c['text']}\n\n")

    print(f"[Clean] {len(cleaned)} unique captions → {output_vtt}")
    return output_vtt


# ─── Step 2: Transliterate Telugu → Latin (Roman Colloquial) ──────────────────
def transliterate_vtt(input_vtt, output_vtt):
    """Converts Telugu script in VTT to Roman Colloquial (natural Tenglish) transliteration."""
    vtt = webvtt.read(input_vtt)
    for caption in vtt:
        caption.text = aksh.process('Telugu', 'RomanColloquial', caption.text)
    vtt.save(output_vtt)
    print(f"[Transliterate] Saved → {output_vtt}")
    return output_vtt


# ─── Step 3a: Normalize & Tag English loanwords (fuzzy detector) ──────────────
def normalize_and_tag_vtt(input_vtt, output_vtt):
    """
    Identifies English loanwords using the fuzzy detector:
      1. Strips ISO diacritics + applies phonetic rules
      2. Matches against a high-frequency English corpus
    Tags detected English words with brackets: [word]
    """
    from fuzzy_loanword_detector import detect_loanword
    vtt = webvtt.read(input_vtt)

    tagged_count = 0

    for caption in vtt:
        words = caption.text.split()
        tagged = []
        for w in words:
            eng = detect_loanword(w)
            if eng:
                tagged.append(f"[{eng}]")
                tagged_count += 1
            else:
                tagged.append(w)

        caption.text = " ".join(tagged)

    vtt.save(output_vtt)
    print(f"[Tag] {tagged_count} English loanwords tagged → {output_vtt}")
    return output_vtt


# ─── Step 3b: Context-aware refinement of tagged loanwords ────────────────────

# False positives: English words that the fuzzy detector matches but are actually
# common Telugu words or transliteration artifacts, NOT real English loanwords.
_FALSE_POSITIVES = {
    "trot", "van", "inn", "pal", "dam", "den", "fin", "fur",
    "hen", "hut", "mud", "nun", "ore", "peg", "rag", "sap",
    "tar", "urn", "vat", "wad", "yam", "pan", "mat", "pat",
    "nit", "pun", "cot", "dip", "jot", "pod", "rot",
    "rut", "sod", "tot", "vet", "wit",
}

# Common transliteration corrections: fuzzy detector maps to wrong English word
_CORRECTIONS = {
    "praises": "prices",    # praisēs → prices (Telugu pronunciation of "prices")
    "said": "side",         # saiḍ → side (not "said")
    "pas": "pass",          # pās → pass
    "fails": "files",       # phails → files (Telugu pronunciation)
    "fast": "first",        # phasṭ → could be "first" in context
    "trot": None,           # trōṭ → throat infection context, not English "trot"
}


def refine_tags_vtt(input_vtt, output_vtt):
    """
    Context-aware refinement pass over tagged VTT:
      1. Removes false-positive tags (Telugu words matching English by accident)
      2. Applies known transliteration corrections
      3. Removes tags on very short words that are likely Telugu particles
    """
    vtt = webvtt.read(input_vtt)

    removed = 0
    corrected = 0

    for caption in vtt:
        words = caption.text.split()
        refined = []
        for w in words:
            # Check if this is a tagged word: [word]
            if w.startswith("[") and w.endswith("]"):
                eng = w[1:-1]

                # 1. Remove false positives
                if eng in _FALSE_POSITIVES:
                    # Restore the original untagged form (just the word without brackets)
                    refined.append(eng)
                    removed += 1
                    continue

                # 2. Apply known corrections
                if eng in _CORRECTIONS:
                    correction = _CORRECTIONS[eng]
                    if correction is None:
                        # Explicitly marked as "not English" — remove tag
                        refined.append(eng)
                        removed += 1
                    else:
                        refined.append(f"[{correction}]")
                        corrected += 1
                    continue

                # 3. Keep the tag as-is
                refined.append(w)
            else:
                refined.append(w)

        caption.text = " ".join(refined)

    vtt.save(output_vtt)
    print(f"[Refine] {removed} false positives removed, {corrected} corrections applied → {output_vtt}")
    return output_vtt


# ─── Step 4: Chunk audio + text into dataset ─────────────────────────────────
def time_to_ms(ts):
    """HH:MM:SS.mmm → milliseconds"""
    h, m, rest = ts.split(':')
    s, ms = rest.split('.')
    return int(h)*3600000 + int(m)*60000 + int(s)*1000 + int(ms)


def chunk_and_export(audio_path, vtt_path, output_dir):
    """Slices audio by VTT timestamps and exports dataset (audio chunks + metadata.json)."""
    os.makedirs(output_dir, exist_ok=True)
    chunk_dir = os.path.join(output_dir, "audio_chunks")
    os.makedirs(chunk_dir, exist_ok=True)

    audio = AudioSegment.from_wav(audio_path)
    vtt = webvtt.read(vtt_path)

    records = []
    chunk_idx = 0  # Sequential counter for chunk naming
    for cap in vtt:
        start = time_to_ms(cap.start)
        end = time_to_ms(cap.end)
        text = cap.text.strip()
        if not text or (end - start) < 500:
            continue

        chunk = audio[start:end]
        fname = f"chunk_{chunk_idx:05d}.wav"
        fpath = os.path.join(chunk_dir, fname)
        chunk.export(fpath, format="wav")

        records.append({
            "audio_path": os.path.abspath(fpath),
            "text": text,
            "start": cap.start,
            "end": cap.end,
            "duration_s": round((end - start) / 1000.0, 2)
        })
        chunk_idx += 1

    meta_path = os.path.join(output_dir, "metadata.json")
    with open(meta_path, 'w', encoding='utf-8') as f:
        json.dump(records, f, indent=2, ensure_ascii=False)

    print(f"[Chunk] {len(records)} chunks → {chunk_dir}")
    print(f"[Chunk] Metadata → {meta_path}")
    return records


# ─── Main ─────────────────────────────────────────────────────────────────────
def run_pipeline(audio_path, vtt_path, output_dir="data/processed"):
    os.makedirs(output_dir, exist_ok=True)
    base = os.path.splitext(os.path.basename(vtt_path))[0]

    print("=" * 60)
    print("  Telugu-English Code-Mixed ASR Pipeline")
    print("=" * 60)

    # Step 1: Clean VTT
    cleaned_vtt = os.path.join(output_dir, f"{base}_clean.vtt")
    clean_vtt(vtt_path, cleaned_vtt)

    # Step 2: Transliterate
    translit_vtt = os.path.join(output_dir, f"{base}_translit.vtt")
    transliterate_vtt(cleaned_vtt, translit_vtt)

    # Step 3a: Normalize & tag (fuzzy detector)
    tagged_vtt = os.path.join(output_dir, f"{base}_tagged.vtt")
    normalize_and_tag_vtt(translit_vtt, tagged_vtt)

    # Step 3b: Refine tags (context-aware corrections)
    refined_vtt = os.path.join(output_dir, f"{base}_refined.vtt")
    refine_tags_vtt(tagged_vtt, refined_vtt)

    # Step 4: Chunk and export (uses refined VTT)
    records = chunk_and_export(audio_path, refined_vtt, output_dir)

    print("=" * 60)
    print(f"  Pipeline complete! {len(records)} samples generated.")
    print("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Telugu-English Code-Mixed ASR data preparation pipeline"
    )

    # Input: either a YouTube URL or pre-downloaded audio+vtt
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument(
        "--url", help="YouTube video URL (downloads audio + Telugu subtitles automatically)"
    )
    input_group.add_argument(
        "--audio", help="Path to pre-downloaded WAV audio file (use with --vtt)"
    )

    parser.add_argument("--vtt", help="Path to Telugu VTT subtitle file (required with --audio)")
    parser.add_argument("--out", default="data/processed", help="Output directory (default: data/processed)")

    args = parser.parse_args()

    if args.url:
        # Download from YouTube first
        raw_dir = os.path.join(os.path.dirname(args.out), "raw")
        audio_path, vtt_path = download_from_youtube(args.url, raw_dir)
        run_pipeline(audio_path, vtt_path, args.out)
    else:
        if not args.vtt:
            parser.error("--vtt is required when using --audio")
        run_pipeline(args.audio, args.vtt, args.out)
