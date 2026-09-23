from functools import lru_cache
from pathlib import Path

import arabic_reshaper
from bidi.algorithm import get_display
from PIL import ImageFont

from .lpr.alphabet import tokenize


@lru_cache(maxsize=16)
def annotation_font(size):
    path = Path(__file__).parent / "assets" / "NotoSansArabic.ttf"
    # Shaping is done explicitly; do not run a second bidi pass in libraqm.
    return ImageFont.truetype(str(path), size, layout_engine=ImageFont.Layout.BASIC)


def display_plate(text):
    if not text:
        return "unreadable"
    try:
        tokens = tokenize(text)
    except ValueError:  # explicit demo backend uses Latin labels
        return text
    letter = get_display(arabic_reshaper.reshape(tokens[2]), base_dir="R")
    # Physical plate order is LTR, while the isolated multi-letter word is RTL.
    return f"{''.join(tokens[:2])} {letter} {''.join(tokens[3:6])} {''.join(tokens[6:])}"
