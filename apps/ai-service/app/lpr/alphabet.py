"""Canonical dataset alphabet. ژ is the dataset's accessibility-symbol code.

Tokens follow physical plate order: two digits, letter/symbol, three digits,
then two region digits. الف is ONE token. Never reverse the stored string.
"""

import math
import re
import unicodedata

LETTERS = (
    "الف",
    "ب",
    "پ",
    "ت",
    "ث",
    "ج",
    "د",
    "ز",
    "ژ",
    "س",
    "ش",
    "ص",
    "ط",
    "ظ",
    "ع",
    "ف",
    "ق",
    "ل",
    "م",
    "ن",
    "ه",
    "و",
    "ی",
)
TOKENS = ("<blank>", *"0123456789", *LETTERS)
TOKEN_IDS = {token: index for index, token in enumerate(TOKENS)}
_DIGITS = str.maketrans("۰۱۲۳۴۵۶۷۸۹٠١٢٣٤٥٦٧٨٩", "01234567890123456789")
_PATTERN = re.compile(r"([0-9]{2})(الف|[" + "".join(LETTERS[1:]) + r"])([0-9]{5})\Z")


def normalize(text: str) -> str:
    text = unicodedata.normalize("NFKC", text).translate(_DIGITS)
    text = "".join(c for c in text if not c.isspace() and unicodedata.category(c) != "Cf")
    return text.replace("ي", "ی").replace("ك", "ک").replace("آ", "الف")


def tokenize(text: str) -> list[str]:
    match = _PATTERN.fullmatch(normalize(text))
    if match is None:
        raise ValueError(f"invalid Iranian plate label: {text!r}")
    first, letter, last = match.groups()
    return [*first, letter, *last]


def encode(text: str) -> list[int]:
    return [TOKEN_IDS[token] for token in tokenize(text)]


def decode_ctc(indices, probabilities, *, min_confidence: float = 0.0) -> tuple[str, float]:
    """Collapse adjacent repeats BEFORE removing blanks (11 and 1-blank-1 differ)."""
    output, scores, previous = [], [], None
    for index, score in zip(indices, probabilities, strict=True):
        index, score = int(index), float(score)
        if index and index != previous:
            output.append(TOKENS[index])
            scores.append(max(1e-12, score))
        previous = index
    text = "".join(output)
    confidence = math.exp(sum(map(math.log, scores)) / len(scores)) if scores else 0.0
    try:
        tokenize(text)
    except ValueError:
        return "", 0.0
    return (text, confidence) if confidence >= min_confidence else ("", 0.0)
