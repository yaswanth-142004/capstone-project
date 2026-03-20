"""
Download YouTube video audio (16 kHz WAV) and Telugu subtitles (VTT).
Can be used standalone or called from the pipeline.
"""
import os
import sys
import glob
import shutil
import subprocess
import argparse


def _find_ffmpeg():
    """
    Locate ffmpeg binary directory. Always returns an explicit path when possible.
    
    Search order:
      1. static-ffmpeg pip package (most reliable — self-contained binaries)
      2. Conda environment's Library/bin (Windows) or bin/ (Unix)
      3. shutil.which() fallback (system PATH)
    """
    # --- 1. Check static-ffmpeg (pip package with working standalone binaries) ---
    try:
        import static_ffmpeg
        ffmpeg_path, _ = static_ffmpeg.run.get_or_fetch_platform_executables_else_raise()
        ffmpeg_dir = os.path.dirname(ffmpeg_path)
        print(f"[Download] Found ffmpeg via static-ffmpeg: {ffmpeg_dir}")
        return ffmpeg_dir
    except (ImportError, Exception):
        pass

    # --- 2. Check conda environment ---
    conda_prefix = os.environ.get("CONDA_PREFIX")
    if not conda_prefix:
        conda_prefix = os.path.dirname(sys.executable)

    # Windows: conda puts ffmpeg in Library/bin
    conda_bin_win = os.path.join(conda_prefix, "Library", "bin")
    ffmpeg_conda_win = os.path.join(conda_bin_win, "ffmpeg.exe")
    if os.path.isfile(ffmpeg_conda_win):
        # Verify it actually works (conda ffmpeg can be broken due to missing DLLs)
        try:
            result = subprocess.run(
                [ffmpeg_conda_win, "-version"],
                capture_output=True, timeout=5
            )
            if result.returncode == 0 and result.stdout:
                print(f"[Download] Found ffmpeg in conda env: {conda_bin_win}")
                return conda_bin_win
        except Exception:
            pass
        print(f"[Download] Conda ffmpeg found at {conda_bin_win} but appears broken, skipping")

    # Unix/macOS: conda puts ffmpeg in bin/
    conda_bin_unix = os.path.join(conda_prefix, "bin")
    ffmpeg_conda_unix = os.path.join(conda_bin_unix, "ffmpeg")
    if os.path.isfile(ffmpeg_conda_unix):
        print(f"[Download] Found ffmpeg in conda env: {conda_bin_unix}")
        return conda_bin_unix

    # --- 3. Fallback: check system PATH ---
    ffmpeg_on_path = shutil.which("ffmpeg")
    if ffmpeg_on_path:
        ffmpeg_dir = os.path.dirname(ffmpeg_on_path)
        print(f"[Download] Found ffmpeg on PATH: {ffmpeg_dir}")
        return ffmpeg_dir

    print("[Download] WARNING: ffmpeg not found! Install via: pip install static-ffmpeg")
    return None


def download_youtube_video(url, output_dir="data/raw"):
    """
    Downloads audio as 16 kHz WAV and Telugu (.vtt) subtitles from a YouTube video.
    Returns (wav_path, vtt_path) on success, or raises RuntimeError on failure.
    """
    os.makedirs(output_dir, exist_ok=True)

    # Locate ffmpeg
    ffmpeg_dir = _find_ffmpeg()

    command = [
        "yt-dlp",
        "-x",                                    # Extract audio
        "--audio-format", "wav",                  # Convert to WAV
        "--audio-quality", "0",                   # Best audio quality
        "--postprocessor-args", "-ar 16000",      # Resample to 16 kHz for ASR
        "--write-sub",                            # Write subtitle file
        "--sub-lang", "te",                       # Download Telugu subtitles
        "--write-auto-sub",                       # Fallback to auto-generated
        "--sub-format", "vtt",                    # VTT format
        "--output", os.path.join(output_dir, "%(id)s.%(ext)s"),
    ]

    # Point yt-dlp to ffmpeg if it's not on PATH
    if ffmpeg_dir:
        command.extend(["--ffmpeg-location", ffmpeg_dir])

    command.append(url)

    print(f"[Download] Executing: {' '.join(command)}")
    try:
        subprocess.run(command, check=True)
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"yt-dlp failed for {url}: {e}")

    # --- Discover downloaded files ---
    wav_files = glob.glob(os.path.join(output_dir, "*.wav"))
    vtt_files = glob.glob(os.path.join(output_dir, "*.vtt"))

    if not wav_files:
        raise RuntimeError(f"No .wav file found in {output_dir} after download")
    if not vtt_files:
        raise RuntimeError(f"No .vtt file found in {output_dir} — video may lack Telugu subtitles")

    # Pick the most recently created files (handles the case of multiple downloads)
    wav_path = max(wav_files, key=os.path.getmtime)
    vtt_path = max(vtt_files, key=os.path.getmtime)

    print(f"[Download] Audio: {wav_path}")
    print(f"[Download] Subtitles: {vtt_path}")
    return wav_path, vtt_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download YouTube videos for ASR dataset")
    parser.add_argument("url", type=str, help="YouTube video or playlist URL")
    parser.add_argument("--out", type=str, default="data/raw", help="Output directory")
    args = parser.parse_args()

    wav, vtt = download_youtube_video(args.url, args.out)
    print(f"\nReady for pipeline:\n  python pipeline.py --audio \"{wav}\" --vtt \"{vtt}\"")
