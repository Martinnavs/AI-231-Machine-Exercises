"""Indicator service: listening state, detected-word line, status line.

The listening state is a **pure receiver** (feature `ui-site`, decision:
option A -- driven by the model's listening-state push via
`POST /api/listening`, not a local simulation): `passive` (waiting for
the wakeword) is the default, `active` (gate open, waiting for the
command) triggers the UI's 3-second semicircle animation.

The detected-word line shows the canonical intent label plus the slot's
canonical value if present -- never the raw spoken variant (decision
Q5: the "output" is the canonical intent label + slot if present).

The status line carries the TIME/WEATHER read-out. It is a **temporary**
display (decision: "disappears after 5s, overridable by the next
TIME/WEATHER"): shown for `STATUS_TTL_S` app-clock ticks, then cleared;
a new TIME/WEATHER overrides it immediately and restarts the 5-second
timer.
"""

from __future__ import annotations

from app.intents import format_detected

LISTENING_STATES = ("passive", "active")
STATUS_TTL_S = 5


class IndicatorService:
    """Display-only state: no `(state, event)` table, no side effects."""

    def __init__(self) -> None:
        self.listening = "passive"
        self.detected: str | None = None
        self.status_line: str | None = None
        self._status_remaining = 0

    def set_listening(self, state: str) -> tuple[bool, str]:
        """Set the listening state from the model's push."""
        if state not in LISTENING_STATES:
            return False, f"Ignored: unknown listening state {state!r}"
        changed = self.listening != state
        self.listening = state
        return changed, f"listening -> {state}"

    def on_command(self, intent: str, slots: dict[str, str]) -> str:
        """Update the detected-word line for a processed command."""
        self.detected = format_detected(intent, slots)
        return self.detected

    def set_status(self, text: str) -> None:
        """Show a temporary status line (TIME/WEATHER), 5 s TTL."""
        self.status_line = text
        self._status_remaining = STATUS_TTL_S

    def tick(self) -> None:
        """One second of app time: count the status line down to gone."""
        if self._status_remaining > 0:
            self._status_remaining -= 1
            if self._status_remaining == 0:
                self.status_line = None

    def to_dict(self) -> dict:
        return {
            "listening": self.listening,
            "detected": self.detected,
            "status_line": self.status_line,
        }
