"""
End-to-end pipeline processor for a single YouTube video.
Takes a raw WAV + auto-generated VTT, cleans, transliterates, tags, and chunks.
"""
import os
import re
import json
import argparse
import webvtt
from pydub import AudioSegment
from aksharamukha import transliterate as aksh
from spellchecker import SpellChecker

# ─── Step 0: Clean raw YouTube VTT ────────────────────────────────────────────
def clean_vtt(input_vtt, output_vtt):
    """
    YouTube auto-generated VTT has lots of duplicate lines and inline <c> tags.
    This strips them and merges sequential duplicate captions.
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

    print(f"[Clean] {len(cleaned)} unique captions written to {output_vtt}")
    return output_vtt

# ─── Step 1: Transliterate Telugu → Latin (ISO) ──────────────────────────────
def transliterate_vtt(input_vtt, output_vtt):
    """Converts Telugu script in VTT to ISO Latin transliteration."""
    vtt = webvtt.read(input_vtt)
    for caption in vtt:
        # Only transliterate parts that contain Telugu characters
        # English words (already Latin) will pass through unchanged
        caption.text = aksh.process('Telugu', 'ISO', caption.text)
    vtt.save(output_vtt)
    print(f"[Transliterate] Saved to {output_vtt}")
    return output_vtt

# ─── Step 2: Normalize & Tag English loanwords ───────────────────────────────
def normalize_and_tag_vtt(input_vtt, output_vtt):
    """
    Identifies English loanwords using the generalized fuzzy detector:
      1. Strips ISO diacritics + applies phonetic rules
      2. Generates multiple spelling variants
      3. Fuzzy-matches against a high-frequency English corpus
    No hardcoded dictionary needed.
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
    print(f"[Tag] Saved to {output_vtt}")
    print(f"[Tag] Fuzzy detector tagged {tagged_count} English loanwords")
    return output_vtt

# ─── Step 3: Chunk audio + text into dataset ─────────────────────────────────
def time_to_ms(ts):
    """HH:MM:SS.mmm → milliseconds"""
    h, m, rest = ts.split(':')
    s, ms = rest.split('.')
    return int(h)*3600000 + int(m)*60000 + int(s)*1000 + int(ms)

def chunk_and_export(audio_path, vtt_path, output_dir):
    """Slices audio by VTT timestamps and exports dataset."""
    os.makedirs(output_dir, exist_ok=True)
    chunk_dir = os.path.join(output_dir, "audio_chunks")
    os.makedirs(chunk_dir, exist_ok=True)

    audio = AudioSegment.from_wav(audio_path)
    vtt = webvtt.read(vtt_path)

    records = []
    for i, cap in enumerate(vtt):
        start = time_to_ms(cap.start)
        end = time_to_ms(cap.end)
        text = cap.text.strip()
        if not text or (end - start) < 500:
            continue

        chunk = audio[start:end]
        fname = f"chunk_{i:05d}.wav"
        fpath = os.path.join(chunk_dir, fname)
        chunk.export(fpath, format="wav")

        records.append({
            "audio_path": os.path.abspath(fpath),
            "text": text,
            "start": cap.start,
            "end": cap.end,
            "duration_s": round((end - start) / 1000.0, 2)
        })

    meta_path = os.path.join(output_dir, "metadata.json")
    with open(meta_path, 'w', encoding='utf-8') as f:
        json.dump(records, f, indent=2, ensure_ascii=False)

    print(f"[Chunk] {len(records)} chunks saved to {chunk_dir}")
    print(f"[Chunk] Metadata at {meta_path}")
    return records

# ─── Main ─────────────────────────────────────────────────────────────────────
def run_pipeline(audio_path, vtt_path, output_dir="data/processed"):
    os.makedirs(output_dir, exist_ok=True)
    base = os.path.splitext(os.path.basename(vtt_path))[0]

    print("=" * 60)
    print("  Telugu-English Code-Mixed ASR Pipeline")
    print("=" * 60)

    # Step 0: Clean VTT
    cleaned_vtt = os.path.join(output_dir, f"{base}_clean.vtt")
    clean_vtt(vtt_path, cleaned_vtt)

    # Step 1: Transliterate
    translit_vtt = os.path.join(output_dir, f"{base}_translit.vtt")
    transliterate_vtt(cleaned_vtt, translit_vtt)

    # Step 2: Normalize & tag
    tagged_vtt = os.path.join(output_dir, f"{base}_tagged.vtt")
    normalize_and_tag_vtt(translit_vtt, tagged_vtt)./venv/bin/python pipeline.py --audio data/raw/ve8XqZ3bRcM.wav --vtt data/raw/ve8XqZ3bRcM.te.vtt --out data/processed/ve8XqZ3bRcM


    # Step 3: Chunk and export
    records = chunk_and_export(audio_path, tagged_vtt, output_dir)

    print("=" * 60)
    print(f"  Pipeline complete! {len(records)} samples generated.")
    print("=" * 60)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run full ASR pipeline on a YouTube video")
    parser.add_argument("--audio", required=True, help="Path to WAV audio file")
    parser.add_argument("--vtt", required=True, help="Path to Telugu VTT subtitle file")
    parser.add_argument("--out", default="data/processed", help="Output directory")
    args = parser.parse_args()
    run_pipeline(args.audio, args.vtt, args.out)
