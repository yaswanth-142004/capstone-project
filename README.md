# Telugu-English Code-Mixed ASR Pipeline

Enhancing Telugu-English Code-Mixed ASR via **Transliteration**, **Lexical Normalization**, and **Explicit Language Tagging**.

## Overview

This pipeline transforms raw Telugu YouTube videos into a structured code-mixed ASR training dataset:

```
YouTube URL → WAV + VTT → Clean VTT → Transliterate (Telugu → Latin) → Tag English Loanwords → Audio Chunks + Metadata
```

### Pipeline Steps

| Step | What it does |
|------|-------------|
| **0. Download** | Fetches 16 kHz WAV audio and Telugu auto-subtitles via `yt-dlp` |
| **1. Clean VTT** | Strips duplicate lines, HTML tags, and collapses whitespace |
| **2. Transliterate** | Converts Telugu script → ISO Latin ("Tinglish") via `aksharamukha` |
| **3. Tag Loanwords** | Detects English words using diacritic stripping + phonetic rules, wraps them in `[brackets]` |
| **4. Chunk & Export** | Slices audio by subtitle timestamps, exports `audio_chunks/` + `metadata.json` |

## Setup

### Prerequisites
- Python 3.9+
- [FFmpeg](https://ffmpeg.org/download.html) on PATH

### Install
```bash
pip install -r requirements.txt
```

## Usage

### From YouTube URL (end-to-end)
```bash
python pipeline.py --url "https://www.youtube.com/watch?v=VIDEO_ID" --out data/processed
```

### From pre-downloaded files
```bash
python pipeline.py --audio data/raw/video.wav --vtt data/raw/video.te.vtt --out data/processed
```

### Download only (no processing)
```bash
python data_collection.py "https://www.youtube.com/watch?v=VIDEO_ID" --out data/raw
```

## Output Format

```
data/processed/
├── audio_chunks/
│   ├── chunk_00000.wav
│   ├── chunk_00001.wav
│   └── ...
├── *_clean.vtt        # Cleaned subtitles
├── *_translit.vtt     # Transliterated subtitles
├── *_tagged.vtt       # Tagged subtitles (with [english] markers)
└── metadata.json      # Dataset manifest
```

Each entry in `metadata.json`:
```json
{
  "audio_path": "/absolute/path/to/chunk_00000.wav",
  "text": "mīru [phone] lō [battery] [life] cālā [important]",
  "start": "00:00:01.200",
  "end": "00:00:04.800",
  "duration_s": 3.6
}
```

## Project Structure

```
├── pipeline.py                 # Main end-to-end pipeline
├── data_collection.py          # YouTube download utility
├── fuzzy_loanword_detector.py  # English loanword detection engine
├── finetune_whisper_peft.py    # (Future) Whisper PEFT fine-tuning
├── requirements.txt            # Python dependencies
└── .gitignore
```