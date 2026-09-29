"""Timer service: a countdown with three fixed presets.

Restricted to the grammar's three `DURATION` values -- 10 seconds,
30 seconds, 1 minute (feature `ui-site`, decision Q7). Counts down
`MM:SS` once per second (the shared app clock's `tick`); at 00:00 it
shows "Timer done" and stays there -- **no auto-reset**: a reset button
(voice has no reset intent, so UI-only) returns it to idle. Re-issuing a
start while the timer is running is **ignored** (the user waits it out);
starting from idle *or* from done is fine.
"""

from __future__ import annotations

from enum import Enum, auto


class TimerState(Enum):
    IDLE = auto()
    RUNNING = auto()
    DONE = auto()


class TimerEvent(Enum):
    # Values are the preset durations in seconds (the grammar's DURATIONs).
    START_10 = 10
    START_30 = 30
    START_60 = 60
    RESET = auto()


class TimerService:
    """Explicit `(state, event) -> next_state` table + ignore-on-unhandled."""

    PRESETS = (10, 30, 60)

    TRANSITIONS = {
        # Start a preset from idle or from done (a finished timer restarts).
        (TimerState.IDLE, TimerEvent.START_10): TimerState.RUNNING,
        (TimerState.IDLE, TimerEvent.START_30): TimerState.RUNNING,
        (TimerState.IDLE, TimerEvent.START_60): TimerState.RUNNING,
        (TimerState.DONE, TimerEvent.START_10): TimerState.RUNNING,
        (TimerState.DONE, TimerEvent.START_30): TimerState.RUNNING,
        (TimerState.DONE, TimerEvent.START_60): TimerState.RUNNING,
        # Reset (UI-only) from running or done; stays put at idle.
        (TimerState.RUNNING, TimerEvent.RESET): TimerState.IDLE,
        (TimerState.DONE, TimerEvent.RESET): TimerState.IDLE,
        # Ignored (no entry): any START_* while RUNNING (wait it out),
        # RESET while IDLE (nothing to reset).
    }

    def __init__(self) -> None:
        self.state = TimerState.IDLE
        self.duration_s = 0
        self.remaining_s = 0

    @property
    def display(self) -> str:
        """The `MM:SS` countdown read-out."""
        return f"{self.remaining_s // 60:02d}:{self.remaining_s % 60:02d}"

    def dispatch(self, event: TimerEvent) -> tuple[bool, str]:
        """Apply `event` if a transition is defined; else ignore."""
        next_state = self.TRANSITIONS.get((self.state, event))
        if next_state is None:
            return False, (
                f"Ignored: cannot {event.name.lower()} while {self.state.name.lower()}"
            )
        if event is not TimerEvent.RESET:
            self.duration_s = event.value
            self.remaining_s = self.duration_s
        else:
            self.duration_s = 0
            self.remaining_s = 0
        self.state = next_state
        return True, f"[{event.name}] -> {self.state.name} ({self.display})"

    def start(self, seconds: int | None) -> tuple[bool, str]:
        """Start from a preset duration (grammar `DURATION` seconds)."""
        if seconds not in self.PRESETS:
            return False, (
                f"Ignored: timer must be one of {list(self.PRESETS)} seconds, "
                f"got {seconds!r}"
            )
        event = {10: TimerEvent.START_10, 30: TimerEvent.START_30, 60: TimerEvent.START_60}[seconds]
        return self.dispatch(event)

    def reset(self) -> tuple[bool, str]:
        return self.dispatch(TimerEvent.RESET)

    def tick(self) -> None:
        """One second of app time: decrement the countdown."""
        if self.state is not TimerState.RUNNING:
            return
        self.remaining_s = max(0, self.remaining_s - 1)
        if self.remaining_s == 0:
            self.state = TimerState.DONE  # stays done; no auto-reset

    def to_dict(self) -> dict:
        return {
            "state": self.state.name,
            "duration_s": self.duration_s,
            "remaining_s": self.remaining_s,
            "display": self.display,
        }
