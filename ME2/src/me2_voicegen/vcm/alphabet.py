"""29-token CTC alphabet for the toy voice-command acoustic model.

Token layout (see docs/VCM-CONTRACT.md, which this module is the source of
truth for):
    id 0        -> CTC blank
    id 1..26    -> 'a'..'z'
    id 27       -> ' ' (space)
    id 28       -> "'" (apostrophe)
29 tokens total.

The original spec text this project derives from describes "32 tokens" for
the alphabet, but its own enumeration (blank + 26 letters + space +
apostrophe) only lists 29 distinct symbols. That is a discrepancy in the
spec itself, not an omission here: this module implements the 29-token
alphabet the spec's own enumeration actually describes, and does not pad
with 3 unused/reserved tokens to force the number 32.
"""

from __future__ import annotations

import string

BLANK_ID = 0

_SYMBOLS: tuple[str, ...] = tuple(string.ascii_lowercase) + (" ", "'")
"""26 letters + space + apostrophe, in id order starting at 1."""

ID_TO_CHAR: dict[int, str] = {0: ""} | {
    i + 1: ch for i, ch in enumerate(_SYMBOLS)
}
CHAR_TO_ID: dict[str, int] = {ch: i for i, ch in ID_TO_CHAR.items() if i != BLANK_ID}

ALPHABET_SIZE = len(ID_TO_CHAR)
assert ALPHABET_SIZE == 29, f"expected 29-token alphabet, got {ALPHABET_SIZE}"


def encode(text: str) -> list[int]:
    """Map each character of `text` to its alphabet id.

    Raises ValueError on any character outside the 29-token alphabet (run
    `vcm.text.normalize_text` first to guarantee this never happens).
    """
    try:
        return [CHAR_TO_ID[ch] for ch in text]
    except KeyError as exc:
        raise ValueError(
            f"character {exc.args[0]!r} is not in the 29-token VCM alphabet"
        ) from exc


def decode(ids: list[int]) -> str:
    """Map alphabet ids back to text. Blank ids (0) decode to '' (no char)."""
    try:
        return "".join(ID_TO_CHAR[i] for i in ids)
    except KeyError as exc:
        raise ValueError(
            f"id {exc.args[0]!r} is not a valid id in the 29-token VCM alphabet"
        ) from exc


def collapse(ids: list[int]) -> list[int]:
    """CTC-style collapse: merge consecutive duplicate ids, then drop blanks.

    This is the standard CTC decoding collapse applied to a raw per-frame
    argmax id sequence (e.g. from a (T, 29) posterior array), not to
    already-clean token ids from `encode`.
    """
    collapsed: list[int] = []
    prev: int | None = None
    for i in ids:
        if i != prev:
            collapsed.append(i)
        prev = i
    return [i for i in collapsed if i != BLANK_ID]
