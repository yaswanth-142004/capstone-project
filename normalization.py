import argparse
import re
import webvtt
from spellchecker import SpellChecker

def normalize_and_tag(input_vtt, output_vtt):
    """
    Reads a transliterated VTT file, attempts to identify English loanwords,
    normalizes their spelling, and tags them with explicit brackets [].
    """
    spell = SpellChecker()
    # We can load custom dictionaries of common English loanwords used in Telugu here
    # spell.word_frequency.load_words(['battery', 'processor', 'camera', 'display', 'screen'])

    try:
        vtt = webvtt.read(input_vtt)
        for caption in vtt:
            words = caption.text.split()
            tagged_words = []
            
            for w in words:
                # Remove basic punctuation for checking
                clean_w = re.sub(r'[^\w\s]', '', w).lower()
                
                # Skip numerics or empty
                if not clean_w or clean_w.isnumeric():
                    tagged_words.append(w)
                    continue
                
                # Basic check: if the word is exactly an English word, tag it.
                # In a full production system, this would use a more robust
                # cross-lingual dictionary or LLM to catch transliterated English words (e.g. bāṭarī -> battery).
                if clean_w in spell:
                    # Tag the English word
                    tagged_words.append(f"[{clean_w}]")
                else:
                    # Keep as Tinglish (Telugu in Latin script)
                    tagged_words.append(w)
            
            caption.text = " ".join(tagged_words)
            
        vtt.save(output_vtt)
        print(f"Normalized & Tagged VTT saved to: {output_vtt}")
    except Exception as e:
        print(f"Error processing {input_vtt}: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Normalize and Tag English words in Tinglish VTT")
    parser.add_argument("input", type=str, help="Input VTT file (.vtt)")
    parser.add_argument("--output", type=str, default=None, help="Output VTT file")
    args = parser.parse_args()
    
    out_file = args.output if args.output else args.input.replace(".vtt", "_tagged.vtt")
    normalize_and_tag(args.input, out_file)
