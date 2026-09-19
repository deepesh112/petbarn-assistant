"""Text hygiene shared by the scraper, the review client and the analyser.

Real retail copy is messy. Petbarn's product descriptions contain stray
``U+0092`` characters where apostrophes belong -- a Windows-1252 smart quote
that was transcoded into the Unicode C1 control block somewhere upstream. Left
alone it reaches the model as an invisible control character and comes back out
in answers as ``large dogs particular needs``.

Cleaning here, once, keeps that repair out of every call site.
"""

from __future__ import annotations

import html
import re
import unicodedata

#: Map the C1 control block back through Windows-1252, which is what the bytes
#: were meant to be. Codepoints genuinely undefined in cp1252 are dropped.
_C1_REPAIRS: dict[int, str | None] = {}
for _code in range(0x80, 0xA0):
    try:
        _C1_REPAIRS[_code] = bytes([_code]).decode("cp1252")
    except UnicodeDecodeError:
        _C1_REPAIRS[_code] = None

#: Zero-width and directional marks that survive copy-paste and add nothing.
_INVISIBLES = {0x200B: None, 0x200C: None, 0x200D: None, 0x200E: None, 0x200F: None, 0xFEFF: None}

_TRANSLATION = {**_C1_REPAIRS, **_INVISIBLES}

_WHITESPACE_RE = re.compile(r"[ \t ]+")
_NEWLINES_RE = re.compile(r"\n{3,}")

#: Split on sentence-ending punctuation or a line break. Deliberately simple:
#: aspect matching only needs roughly-sentence-sized spans, and a dependency-free
#: splitter cannot fail to install on a deployment host.
_SENTENCE_RE = re.compile(r"(?<=[.!?])\s+|\n+|\s+[-–—]\s+")


def clean_text(value: str | None, *, collapse_newlines: bool = True) -> str:
    """Repair, unescape and tidy a string scraped from the web.

    Returns an empty string for ``None`` so callers can treat the result as text
    unconditionally.
    """
    if not value:
        return ""
    text = html.unescape(str(value))
    text = text.translate(_TRANSLATION)
    # Normalise compatibility forms so "ﬁ" and friends compare as plain letters,
    # without folding away meaningful characters like ° or ½.
    text = unicodedata.normalize("NFC", text)
    text = _WHITESPACE_RE.sub(" ", text)
    if collapse_newlines:
        text = text.replace("\r\n", "\n").replace("\r", "\n")
        text = _NEWLINES_RE.sub("\n\n", text)
    return text.strip()


def split_sentences(text: str) -> list[str]:
    """Break review prose into sentence-sized spans for aspect matching."""
    if not text:
        return []
    parts = (part.strip() for part in _SENTENCE_RE.split(text))
    return [part for part in parts if len(part) > 1]


def truncate(text: str, limit: int, *, suffix: str = "…") -> str:
    """Shorten ``text`` to ``limit`` characters, breaking on a word boundary."""
    text = text.strip()
    if len(text) <= limit:
        return text
    cut = text[: max(limit - len(suffix), 0)]
    # Prefer cutting at the last space so a quote never ends mid-word.
    pivot = cut.rfind(" ")
    if pivot > limit * 0.6:
        cut = cut[:pivot]
    return cut.rstrip(" ,;:.-") + suffix
