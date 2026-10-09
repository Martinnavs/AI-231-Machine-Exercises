"""`spell_integer(n)`: 0-100 English number-to-words, no hyphens.

Bounded to 0-100 (not a general number-to-words engine) because that is the
entire range Option B's numeric slots ever need (README slot values: 10, 30,
1, 6, 8, 9, 18, 22, 26, 20, 60, 100) -- anything beyond that is out of scope
for this grammar and out of scope for D11. This lives under `optionb/`
rather than `grammar_core` because it is Option-B-specific vocabulary
generation, not a reusable grammar combinator.
"""

from __future__ import annotations

_ONES = [
    "zero", "one", "two", "three", "four", "five", "six", "seven", "eight",
    "nine",
]
_TEENS = [
    "ten", "eleven", "twelve", "thirteen", "fourteen", "fifteen", "sixteen",
    "seventeen", "eighteen", "nineteen",
]
_TENS = [
    "", "", "twenty", "thirty", "forty", "fifty", "sixty", "seventy",
    "eighty", "ninety",
]


def spell_integer(n: int) -> str:
    if not isinstance(n, int) or isinstance(n, bool) or not (0 <= n <= 100):
        raise ValueError(f"spell_integer only supports 0-100, got {n!r}")

    if n == 100:
        return "one hundred"
    if n < 10:
        return _ONES[n]
    if n < 20:
        return _TEENS[n - 10]

    tens, ones = divmod(n, 10)
    if ones == 0:
        return _TENS[tens]
    return f"{_TENS[tens]} {_ONES[ones]}"
