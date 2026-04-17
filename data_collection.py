"""
Download YouTube video audio (16 kHz mono WAV) and Telugu subtitles (VTT).
Can be used standalone or called from the pipeline.
"""
import argparse
import csv
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path


YOUTUBE_ID_RE = re.compile(
    r"(?:v=|youtu\.be/|shorts/|embed/)([A-Za-z0-9_-]{11})"
)


def extract_video_id(url):
    match = YOUTUBE_ID_RE.search(url)
    return match.group(1) if match else None


def _asset_pairs(output_dir, video_id=None):
    output = Path(output_dir)
    if video_id:
        audio = output / f"{video_id}.wav"
        subtitles = sorted(output.glob(f"{video_id}.te*.vtt"))
        if audio.exists() and subtitles:
            return [{
                "video_id": video_id,
                "audio_path": str(audio),
                "vtt_path": str(subtitles[0]),
            }]
        return []

    pairs = []
    for audio in sorted(output.glob("*.wav")):
        subtitles = sorted(output.glob(f"{audio.stem}.te*.vtt"))
        if subtitles:
            pairs.append({
                "video_id": audio.stem,
                "audio_path": str(audio),
                "vtt_path": str(subtitles[0]),
            })
    return pairs


def _find_ffmpeg():
    """
    Locate an ffmpeg binary directory for yt-dlp.

    Search order:
      1. static-ffmpeg pip package
      2. Conda environment binaries
      3. System PATH
    """
    try:
        import static_ffmpeg

        ffmpeg_path, _ = static_ffmpeg.run.get_or_fetch_platform_executables_else_raise()
        ffmpeg_dir = os.path.dirname(ffmpeg_path)
        print(f"[Download] Found ffmpeg via static-ffmpeg: {ffmpeg_dir}")
        return ffmpeg_dir
    except Exception:
        pass

    conda_prefix = os.environ.get("CONDA_PREFIX") or os.path.dirname(sys.executable)

    conda_bin_win = os.path.join(conda_prefix, "Library", "bin")
    ffmpeg_conda_win = os.path.join(conda_bin_win, "ffmpeg.exe")
    if os.path.isfile(ffmpeg_conda_win):
        try:
            result = subprocess.run(
                [ffmpeg_conda_win, "-version"],
                capture_output=True,
                timeout=5,
                check=False,
            )
            if result.returncode == 0 and result.stdout:
                print(f"[Download] Found ffmpeg in conda env: {conda_bin_win}")
                return conda_bin_win
        except Exception:
            pass
        print(f"[Download] Conda ffmpeg at {conda_bin_win} appears broken, skipping")

    conda_bin_unix = os.path.join(conda_prefix, "bin")
    ffmpeg_conda_unix = os.path.join(conda_bin_unix, "ffmpeg")
    if os.path.isfile(ffmpeg_conda_unix):
        print(f"[Download] Found ffmpeg in conda env: {conda_bin_unix}")
        return conda_bin_unix

    ffmpeg_on_path = shutil.which("ffmpeg")
    if ffmpeg_on_path:
        ffmpeg_dir = os.path.dirname(ffmpeg_on_path)
        print(f"[Download] Found ffmpeg on PATH: {ffmpeg_dir}")
        return ffmpeg_dir

    print("[Download] WARNING: ffmpeg not found. Install via: pip install static-ffmpeg")
    return None


def download_youtube_video(url, output_dir="data/raw"):
    """
    Download audio as 16 kHz mono WAV and Telugu VTT subtitles.

    Returns a list of discovered WAV/VTT asset dictionaries. A single video
    usually returns one item; playlists can return multiple items.
    """
    os.makedirs(output_dir, exist_ok=True)

    video_id = extract_video_id(url)
    before = {asset["video_id"] for asset in _asset_pairs(output_dir)}
    ffmpeg_dir = _find_ffmpeg()

    command = [
        "yt-dlp",
        "-x",
        "--audio-format", "wav",
        "--audio-quality", "0",
        "--postprocessor-args", "ffmpeg:-ar 16000 -ac 1",
        "--write-sub",
        "--sub-lang", "te",
        "--write-auto-sub",
        "--sub-format", "vtt",
        "--no-write-thumbnail",
        "--output", os.path.join(output_dir, "%(id)s.%(ext)s"),
    ]
    if ffmpeg_dir:
        command.extend(["--ffmpeg-location", ffmpeg_dir])
    command.append(url)

    print(f"[Download] Executing: {' '.join(command)}")
    try:
        subprocess.run(command, check=True)
    except subprocess.CalledProcessError as e:
        print(f"[Download] Error downloading {url}: {e}")
        return []

    print(f"[Download] Successfully downloaded from {url} to {output_dir}")
    pairs = _asset_pairs(output_dir, video_id=video_id) if video_id else []
    if pairs:
        return pairs

    after_pairs = _asset_pairs(output_dir)
    new_pairs = [pair for pair in after_pairs if pair["video_id"] not in before]
    return new_pairs or after_pairs


def read_urls_from_csv(csv_path, url_column="url"):
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


def download_from_csv(csv_path, output_dir="data/raw", url_column="url"):
    assets = []
    for url in read_urls_from_csv(csv_path, url_column=url_column):
        assets.extend(download_youtube_video(url, output_dir))
    return assets


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download YouTube videos for ASR dataset")
    parser.add_argument("url", nargs="?", help="YouTube video or playlist URL")
    parser.add_argument("--csv", help="CSV of YouTube URLs")
    parser.add_argument("--url-column", default="url", help="URL column name for --csv files")
    parser.add_argument("--out", type=str, default="data/raw", help="Output directory")
    args = parser.parse_args()

    if args.csv:
        download_from_csv(args.csv, args.out, url_column=args.url_column)
    elif args.url:
        download_youtube_video(args.url, args.out)
    else:
        parser.error("Provide a URL or --csv")
