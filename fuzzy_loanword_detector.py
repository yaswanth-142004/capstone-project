"""
Ultra-fast English loanword detector for ISO-transliterated Telugu.
Uses simple diacritic stripping + direct set lookup. No fuzzy matching.
Runs in seconds, not minutes.
"""
import re
from functools import lru_cache
from unidecode import unidecode

# ─── English word set ─────────────────────────────────────────────────────────
from spellchecker import SpellChecker
_spell = SpellChecker()
_all = sorted(
    [(w, f) for w, f in _spell.word_frequency.dictionary.items() if len(w) >= 3 and w.isalpha()],
    key=lambda x: x[1], reverse=True
)
_ENG = set(w for w, _ in _all[:15000])
_ENG.update({
    "laptop", "gaming", "battery", "charging", "camera", "display", "phone",
    "processor", "graphic", "graphics", "magnetic", "support", "rating",
    "memory", "foldable", "telephoto", "touch", "screen", "discount",
    "selling", "price", "description", "information", "concept", "project",
    "budget", "model", "order", "complete", "discontinue", "alphabetical",
    "overtake", "successor", "release", "record", "records", "breaking",
    "total", "views", "final", "semifinal", "dominate", "install",
    "news", "next", "first", "same", "deal", "leak", "leaks", "series",
    "report", "break", "link", "tweet", "clarity", "layer", "fold",
    "unfold", "flat", "crease", "creaseless", "printing", "glass",
    "special", "compact", "ultra", "titanium", "hinge", "gap",
    "refresh", "resolution", "refrigerator", "double", "door", "star",
    "normal", "day", "top", "half", "work", "reel", "short", "time",
    "date", "life", "check", "run", "add", "game", "games", "third",
    "party", "fill", "mostly", "already", "exact", "overall",
    "feature", "features", "update", "updates", "version", "storage",
    "wireless", "bluetooth", "speaker", "speakers", "microphone",
    "sensor", "fingerprint", "power", "performance", "benchmark",
    "software", "hardware", "design", "premium", "flagship",
    "laptop", "brand", "launch", "market", "compare", "review",
})
del _all

# Telugu stopwords (normalized) to skip
_STOP = {
    "ki","lo","to","ni","ga","idi","adi","ivi","avi","tho",
    "mana","maku","miku","naku","vari","dani","oka","anni",
    "ante","ayite","kuda","kani","mari","inka","emi",
    "undi","vundi","chesi","kosam","nunchi",
    "eppudu","appudu","ipudu","akkada","ikkada",
    "ela","enta","entha","enduku",
    "konni","rendu","moodu","nalugu",
    "okkati","okati","chala","antha","inta",
    "avutundi","vastundi","untadi","untundi",
    "chudochu","cheyochu",
    "ledu","leedu","raledu",
    "kaadu","kakapote","lekkapote",
    "annatu","cheppesi","chesaru","chestunnaru",
    "untayi","vastayi","chestaaru",
    "varaku","daaka","meeda",
    "gurinchi","vaccesariki","vachchesariki",
    "vastunna","vasthundi","vastadi",
    "kanipistadi","kanipinchadu","kanipistundi",
    "teesukostaaru","tisukoni","teesukunna",
    "matladaadan","matladdam",
    "ayindi","ayyindi","vellipoindi","vellindi",
    "pedataru","paddaru","pedallu",
}

print(f"[Detector] {len(_ENG)} English words loaded.")

@lru_cache(maxsize=None)
def detect_loanword(word: str) -> str | None:
    """Check if a Tinglish word maps to English via diacritic stripping."""
    clean = re.sub(r'[^\w]', '', word).lower()
    if len(clean) < 3 or clean.isnumeric():
        return None
    
    # Already English?
    if clean in _ENG:
        return clean

    # Strip diacritics with unidecode
    stripped = unidecode(clean)
    stripped = re.sub(r'[^a-z]', '', stripped.lower())
    
    if not stripped or len(stripped) < 3 or stripped in _STOP:
        return None
    if stripped in _ENG:
        return stripped
    
    # Try ph→f substitution (most common Telugu→English phonetic shift)
    pf = stripped.replace('ph', 'f')
    if pf != stripped and pf in _ENG:
        return pf
    
    # Try c→ch (Telugu c = English ch)
    ch_variant = stripped.replace('c', 'ch')
    if ch_variant != stripped and ch_variant in _ENG:
        return ch_variant

    # Try v→w
    vw = stripped.replace('v', 'w')
    if vw != stripped and vw in _ENG:
        return vw
    
    return None


def tag_sentence(sentence: str) -> str:
    words = sentence.split()
    return " ".join(f"[{e}]" if (e := detect_loanword(w)) else w for w in words)


if __name__ == "__main__":
    tests = [
        "prāsesar", "ḍisplē", "byāṭarī", "cārjiṁg", "kemerā",
        "phōn", "lāpṭāp", "gēmiṁg", "nyūs", "neksṭ", "brēk",
        "ṭōṭal", "grāphik", "māgneṭik", "selliṁg",
        "vaccēsariki", "vastuṁdi", "guriṁci", "māṭlāḍadāṁ",
    ]
    for w in tests:
        result = detect_loanword(w)
        s = unidecode(re.sub(r'[^\w]', '', w).lower())
        print(f"{w:<20} → {s:<20} → {result or '(Telugu)'}")
