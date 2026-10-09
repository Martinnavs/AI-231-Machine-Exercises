"""Alarm service: a settable, replaceable, one-shot alarm.

Set times are restricted to the grammar's three `ALARM_TIME` values
(6 AM / 8 AM / 9 PM -- feature `ui-site`, decision Q12). The panel reads
"Alarm set for: HH:MM AM/PM". Setting a time is valid from **any** state
(including DUE) -- a new invocation replaces the old time, and there is
no daily repeat: once it has gone due it stays due until the user resets
or replaces it.

The due check is lightweight (feature `ui-site`, decision Q12): each app
clock tick, a SET alarm compares the current 12-hour hour + minute +
AM/PM against its set time; on a match it flips to DUE, where it persists
until `reset()` (UI-only -- there is no alarm-reset voice intent).
"""

from __future__ import annotations

import re
from datetime import datetime
from enum import Enum, auto
from typing import Callable

Now = Callable[[], datetime]

_ALARM_TIME_RE = re.compile(r"^(\d{1,2})\s*(AM|PM)$", re.IGNORECASE)


class AlarmState(Enum):
    UNSET = auto()
    SET = auto()
    DUE = auto()


class AlarmEvent(Enum):
    RESET = auto()


class AlarmService:
    """A SET/DUE flag machine (the set time is carried on the service).

    `set(time)` is intentionally not in the `(state, event)` table: it is
    valid from every state (replaceable, including DUE), so it is a named
    method like the lights' `set_color`; `reset` follows the table.
    """

    ALARM_TIMES = ("6 AM", "8 AM", "9 PM")

    TRANSITIONS = {
        # Reset (UI-only) from set or due; stays put when unset.
        (AlarmState.SET, AlarmEvent.RESET): AlarmState.UNSET,
        (AlarmState.DUE, AlarmEvent.RESET): AlarmState.UNSET,
        # Ignored (no entry): RESET while UNSET.
    }

    def __init__(self, now: Now | None = None) -> None:
        self._now = now if now is not None else datetime.now
        self.state = AlarmState.UNSET
        self._time: str | None = None  # canonical value, e.g. "6 AM"

    @property
    def time(self) -> str | None:
        """The canonical set time (grammar `ALARM_TIME` value) or None."""
        return self._time

    @property
    def display(self) -> str | None:
        """The panel read-out value, "HH:MM AM/PM" (e.g. "06:00 AM")."""
        if self._time is None:
            return None
        match = _ALARM_TIME_RE.match(self._time)
        hour, ampm = int(match.group(1)), match.group(2).upper()
        return f"{hour:02d}:00 {ampm}"

    def set(self, time_str: str) -> tuple[bool, str]:
        """Set (or replace) the alarm time; valid from any state."""
        if time_str not in self.ALARM_TIMES:
            return False, (
                f"Ignored: alarm must be one of {list(self.ALARM_TIMES)}, "
                f"got {time_str!r}"
            )
        self._time = time_str
        self.state = AlarmState.SET
        return True, f"alarm set for {self.display}"

    def reset(self) -> tuple[bool, str]:
        """Clear the alarm (UI-only)."""
        next_state = self.TRANSITIONS.get((self.state, AlarmEvent.RESET))
        if next_state is None:
            return False, (
                f"Ignored: cannot reset while {self.state.name.lower()}"
            )
        self.state = next_state
        self._time = None
        return True, "alarm cleared"

    def tick(self) -> None:
        """One second of app time: the lightweight due check."""
        if self.state is not AlarmState.SET or self._time is None:
            return
        now = self._now()
        hour12 = now.hour % 12 or 12
        ampm = "AM" if now.hour < 12 else "PM"
        match = _ALARM_TIME_RE.match(self._time)
        set_hour, set_ampm = int(match.group(1)), match.group(2).upper()
        if hour12 == set_hour and now.minute == 0 and ampm == set_ampm:
            self.state = AlarmState.DUE  # persists until reset/replace

    def to_dict(self) -> dict:
        return {
            "state": self.state.name,
            "time": self._time,
            "display": self.display,
        }
