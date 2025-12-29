import re

VOWELS = set("aeiouy")

def _words(text: str):
    return re.findall(r"[a-zA-Z']+", text)

def _sentences(text: str):
    s = re.split(r"[.!?]+", text)
    return [x.strip() for x in s if x.strip()]

def _syllables_in_word(word: str) -> int:
    w = re.sub(r"[^a-z]", "", word.lower())
    if not w:
        return 0
    count = 0
    prev_vowel = False
    for ch in w:
        is_vowel = ch in VOWELS
        if is_vowel and not prev_vowel:
            count += 1
        prev_vowel = is_vowel
    if w.endswith("e") and count > 1:
        count -= 1
    return max(count, 1)

def flesch_reading_ease(text: str) -> float:
    words = _words(text)
    sentences = _sentences(text)
    if not words or not sentences:
        return 0.0
    syllables = sum(_syllables_in_word(w) for w in words)
    wps = len(words) / max(len(sentences), 1)
    spw = syllables / max(len(words), 1)
    return 206.835 - 1.015 * wps - 84.6 * spw

def flesch_kincaid_grade(text: str) -> float:
    words = _words(text)
    sentences = _sentences(text)
    if not words or not sentences:
        return 0.0
    syllables = sum(_syllables_in_word(w) for w in words)
    wps = len(words) / max(len(sentences), 1)
    spw = syllables / max(len(words), 1)
    return 0.39 * wps + 11.8 * spw - 15.59
