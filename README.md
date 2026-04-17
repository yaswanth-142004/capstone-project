# Telugu-English Code-Mixed ASR Dataset Pipeline

Enhancing Telugu-English code-mixed ASR via transliteration, lexical
normalization, explicit language tagging, and timestamp-aligned audio chunks.

## Overview

This project builds weakly supervised ASR fine-tuning data from Telugu YouTube
videos with subtitles.

Pipeline stages:

1. Download YouTube audio as 16 kHz mono WAV and Telugu VTT subtitles.
2. Clean YouTube auto-caption markup and remove repeated context text.
3. Transliterate Telugu subtitles to Latin/Tinglish with Aksharamukha.
4. Normalize English loanwords and tag them as bracketed code-switch tokens,
   for example `[battery]`.
5. Refine known false-positive tags and transliteration corrections.
6. Split the audio by subtitle timestamps and export ASR manifests.

## Setup

### Prerequisites

- Python 3.9+
- FFmpeg, or `static-ffmpeg` from `requirements.txt`

### Install

```bash
pip install -r requirements.txt
```

## Process One YouTube URL

```bash
python pipeline.py \
  --url "https://www.youtube.com/watch?v=VIDEO_ID" \
  --raw-dir data/raw \
  --out data/processed/VIDEO_ID
```

## Process One Existing WAV/VTT Pair

```bash
python pipeline.py \
  --audio data/raw/VIDEO_ID.wav \
  --vtt data/raw/VIDEO_ID.te.vtt \
  --out data/processed/VIDEO_ID
```

## Process a CSV of YouTube URLs

Create a CSV with a `url` column:

```csv
url
https://www.youtube.com/watch?v=ve8XqZ3bRcM
```

Run the full download and dataset pipeline:

```bash
python pipeline.py \
  --csv urls.csv \
  --raw-dir data/raw \
  --out data/processed
```

If your CSV uses another column name:

```bash
python pipeline.py --csv urls.csv --url-column youtube_url
```

## Download Only

```bash
python data_collection.py "https://www.youtube.com/watch?v=VIDEO_ID" --out data/raw
```

For CSV download only:

```bash
python data_collection.py --csv urls.csv --out data/raw
```

## Outputs

Each processed video directory contains:

- `*_clean.vtt`: cleaned native Telugu subtitles
- `*_translit.vtt`: Latin/Tinglish transliteration
- `*_tagged.vtt`: Tinglish with normalized English tokens in brackets
- `*_refined.vtt`: corrected tagged subtitles used for chunk export
- `audio_chunks/*.wav`: timestamp-aligned audio clips
- `metadata.json`, `metadata.jsonl`, `metadata.csv`: ASR manifests
- `hf_dataset/`: Hugging Face dataset with `audio` and `text` columns, when
  the optional `datasets` package is installed

For CSV runs, combined manifests are also written to the output root:

- `all_metadata.json`
- `all_metadata.jsonl`
- `all_metadata.csv`
- `hf_dataset/`

Each manifest entry includes paths and ASR-friendly text fields:

```json
{
  "audio_path": "/absolute/path/to/chunk_00000.wav",
  "text": "mīru [phone] lō [battery] [life] cālā [important]",
  "start": "00:00:01.200",
  "end": "00:00:04.800",
  "duration_s": 3.6,
  "video_id": "VIDEO_ID"
}
```

## Project Structure

```text
├── pipeline.py                 # Main end-to-end pipeline
├── data_collection.py          # YouTube download utility
├── loanword_dict.py            # Known Tinglish-to-English loanword mapping
├── fuzzy_loanword_detector.py  # English loanword detection engine
├── finetune_whisper_peft.py    # Whisper PEFT fine-tuning
├── urls.csv                    # Example CSV input
├── requirements.txt            # Python dependencies
└── experiments/                # Experiment notebooks
```
