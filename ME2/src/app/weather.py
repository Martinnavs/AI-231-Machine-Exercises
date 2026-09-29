"""Weather seam: zero-cloud by design.

The UI site makes **no network calls** (feature `ui-site` decision:
zero-cloud), so the WEATHER intent's read-out is produced by a local
`WeatherProvider` implemented as a stub. A real (still local) provider
can be swapped in later without touching the indicator, dispatch, or API
layers -- this module is the only place the seam exists.
"""

from __future__ import annotations

from typing import Protocol


class WeatherProvider(Protocol):
    def current(self) -> str:
        """A short, human-readable current-conditions string."""
        ...


class StubWeatherProvider:
    """The default: a fixed, offline conditions string (no I/O)."""

    def __init__(self, text: str = "sunny, 24 degrees") -> None:
        self._text = text

    def current(self) -> str:
        return self._text
