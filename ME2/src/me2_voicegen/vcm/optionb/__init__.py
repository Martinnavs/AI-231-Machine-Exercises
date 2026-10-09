"""Option B closed-world spoken-command grammar.

See docs/OPTIONB-GRAMMAR-CONTRACT.md for the full contract this package
implements.
"""

from __future__ import annotations

from .grammar import KNOWN_README_DIVERGENCES, OPTIONB_GRAMMAR
from .numbers import spell_integer
from .text import normalize_text

__all__ = ["KNOWN_README_DIVERGENCES", "OPTIONB_GRAMMAR", "normalize_text", "spell_integer"]
