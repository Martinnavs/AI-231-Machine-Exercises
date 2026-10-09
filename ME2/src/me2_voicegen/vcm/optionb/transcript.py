"""Digit-safe transcript preparation for Option B rows heading into the VCM
CTC pipeline.

`vcm.text.normalize_text` drops every character outside `a-z`/space/
apostrophe -- digits included (see `docs/VCM-CONTRACT.md` section 2). Real
Option B transcripts carry digit-bearing slot values (e.g. "Alarm 6 AM",
"Brightness 100 percent"); feeding them through `normalize_text` unchanged
silently corrupts the target ("alarm am", "brightness percent"). This module
spells digit runs out as words *before* that normalization happens, so no
numeric content is lost. It does not otherwise touch casing or punctuation
-- `vcm.text.normalize_text` downstream already handles that.
"""

from __future__ import annotations

import re

from me2_voicegen.vcm.optionb.numbers import spell_integer

_DIGIT_RUN = re.compile(r"\d+")
_ON_THE_HOUR = re.compile(r"(\d):00\b")


def prepare_ctc_transcript(text: str) -> str:
    """Replace every run of digits in `text` with its spelled-out English
    words (via `vcm.optionb.numbers.spell_integer`). Raises `ValueError` if a
    digit run falls outside `spell_integer`'s 0-100 range, rather than
    silently mis-spelling an unexpected number."""
    text = _ON_THE_HOUR.sub(r"\1", text)  # "6:00 AM" is the same spoken time as "6 AM"
    return _DIGIT_RUN.sub(lambda m: spell_integer(int(m.group())), text)
