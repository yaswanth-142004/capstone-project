import argparse
import webvtt
from aksharamukha import transliterate

def process_vtt_file(input_vtt, output_vtt):
    """
    Reads a VTT subtitle file in Telugu and transliterates it to ISO (Latin-based).
    """
    try:
        vtt = webvtt.read(input_vtt)
        for caption in vtt:
            # We use ISO transliteration scheme for a consistent Latin representation (Tinglish)
            transliterated_text = transliterate.process('Telugu', 'ISO', caption.text)
            caption.text = transliterated_text
            
        vtt.save(output_vtt)
        print(f"Transliterated VTT saved to: {output_vtt}")
    except Exception as e:
        print(f"Error processing {input_vtt}: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Transliterate Telugu VTT files to Latin script")
    parser.add_argument("input", type=str, help="Input VTT file (.vtt)")
    parser.add_argument("--output", type=str, default=None, help="Output VTT file")
    args = parser.parse_args()
    
    out_file = args.output if args.output else args.input.replace(".vtt", "_en.vtt")
    process_vtt_file(args.input, out_file)
