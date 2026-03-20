import os
import argparse
import json
import webvtt
from pydub import AudioSegment
from datasets import Dataset

def time_to_ms(time_str):
    """ Converts VTT timestamp HH:MM:SS.mmm to milliseconds """
    h, m, s_ms = time_str.split(':')
    s, ms = s_ms.split('.')
    return (int(h) * 3600000) + (int(m) * 60000) + (int(s) * 1000) + int(ms)

def format_audio_and_text(audio_file, vtt_file, output_dir):
    """
    Slices the audio file based on VTT timestamps, pairs them with text,
    and exports a HuggingFace Dataset structure.
    """
    os.makedirs(output_dir, exist_ok=True)
    audio_dir = os.path.join(output_dir, "audio_chunks")
    os.makedirs(audio_dir, exist_ok=True)
    
    try:
        audio = AudioSegment.from_wav(audio_file)
        vtt = webvtt.read(vtt_file)
        
        data_records = []
        
        for i, caption in enumerate(vtt):
            start_ms = time_to_ms(caption.start)
            end_ms = time_to_ms(caption.end)
            text = caption.text.strip()
            
            # Skip empty captions or very short/long clips if needed
            if not text or (end_ms - start_ms) < 500:
                continue
                
            # Slice audio chunk
            chunk = audio[start_ms:end_ms]
            chunk_filename = f"chunk_{i:04d}.wav"
            chunk_path = os.path.join(audio_dir, chunk_filename)
            chunk.export(chunk_path, format="wav")
            
            data_records.append({
                "audio_path": os.path.abspath(chunk_path),
                "text": text,
                "duration_s": (end_ms - start_ms) / 1000.0
            })
            
        # Create HuggingFace dataset
        ds = Dataset.from_list(data_records)
        ds.save_to_disk(os.path.join(output_dir, "hf_dataset"))
        
        # Save as JSON for manual inspection
        with open(os.path.join(output_dir, "metadata.json"), "w", encoding="utf-8") as f:
            json.dump(data_records, f, indent=4)
            
        print(f"Successfully processed and saved dataset to {output_dir}")
        print(f"Total chunks created: {len(data_records)}")
        
    except Exception as e:
        print(f"Error during alignment and formatting: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Slice audio and format as HF Dataset")
    parser.add_argument("--audio", type=str, required=True, help="Input WAV file")
    parser.add_argument("--vtt", type=str, required=True, help="Input VTT file")
    parser.add_argument("--out", type=str, default="data/processed", help="Output directory")
    args = parser.parse_args()
    
    format_audio_and_text(args.audio, args.vtt, args.out)
