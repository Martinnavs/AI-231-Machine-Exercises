"""Text normalization for the Option B grammar.

Deliberately separate from `me2_voicegen.vcm.text.normalize_text`: that
normalizer drops digits (they are outside the 29-token VCM CTC alphabet),
which would collapse several Option B numeric slot values into ambiguous
shared buckets once digits are stripped -- e.g. digit-form "10 seconds" and
"30 seconds" would both lose their leading digit and become indistinguishable.
Option B's slot vocabulary depends on digits surviving normalization, so this
module keeps its own copy rather than importing vcm's.
"""

from __future__ import annotations

import string

_ALLOWED_CHARS = set(string.ascii_lowercase) | set(string.digits) | {" ", "'"}
_CURLY_APOSTROPHES = {"‘", "’", "ʼ"}


def normalize_text(text: str) -> str:
    """Lowercase; fold curly apostrophes to `'`; keep `a-z`, `0-9`, `'`,
    space; every other character becomes a word boundary; collapse
    whitespace; strip."""
    text = text.lower()
    chars = []
    for ch in text:
        if ch in _CURLY_APOSTROPHES:
            chars.append("'")
        elif ch in _ALLOWED_CHARS:
            chars.append(ch)
        else:
            chars.append(" ")
    return " ".join("".join(chars).split())
