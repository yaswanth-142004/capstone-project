import os
import subprocess
import argparse

def download_youtube_video(url, output_dir="data/raw"):
    """
    Downloads audio as wav and Telugu (.vtt) subtitles from a YouTube video or playlist.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # We want to extract audio as wav 16kHz (standard for ASR)
    # And grab the te (Telugu) subtitles
    
    command = [
        "yt-dlp",
        "-x",                            # Extract audio
        "--audio-format", "wav",         # Convert to wav
        "--audio-quality", "0",          # Best audio quality
        "--postprocessor-args", "-ar 16000",  # Resample to 16kHz for ASR
        "--write-sub",                   # Write subtitle file
        "--sub-lang", "te",              # Download Telugu subtitles
        "--write-auto-sub",              # Fallback to auto-generated if manual not available
        "--sub-format", "vtt",           # Request VTT format specifically
        "--output", f"{output_dir}/%(id)s.%(ext)s", # Save as video_id.ext
        url
    ]
    
    print(f"Executing: {' '.join(command)}")
    try:
        subprocess.run(command, check=True)
        print(f"Successfully downloaded from {url} to {output_dir}")
    except subprocess.CalledProcessError as e:
        print(f"Error downloading {url}: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download YouTube videos for ASR dataset")
    parser.add_argument("url", type=str, help="YouTube video or playlist URL")
    parser.add_argument("--out", type=str, default="data/raw", help="Output directory")
    args = parser.parse_args()
    
    download_youtube_video(args.url, args.out)
