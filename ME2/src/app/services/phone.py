"""Phone service: a simulated call + message log.

A small hardcoded contact book (eight Filipino names + numbers). `CALL`
picks a random person and enters the in-call state with an elapsed ticker
(advanced once per second by the shared app clock). Re-issuing `CALL`
mid-call starts a *new* call -- it replaces the person and restarts the
ticker (feature `ui-site`, decision Q10). `HANG_UP` is the only way out
of a call and is **UI-only**: there is no hang-up voice intent in the
Option B grammar. `MESSAGE` picks a random person and a random canned
text and appends it to a visible, timestamped log; it is a log action
valid in any call state (like the lights' color) and never changes the
call state.
"""

from __future__ import annotations

import random
from datetime import datetime
from enum import Enum, auto
from typing import Callable

Now = Callable[[], datetime]

# The simulated contact book (decision Q10: ~8 hardcoded Filipino names).
CONTACTS = (
    ("Maria Santos", "0917 111 2201"),
    ("Juan Dela Cruz", "0917 222 3302"),
    ("Ana Reyes", "0918 333 4403"),
    ("Jose Garcia", "0917 444 5504"),
    ("Liza Mendoza", "0919 555 6605"),
    ("Rico Tan", "0917 666 7706"),
    ("Bea Lim", "0918 777 8807"),
    ("Carlo Flores", "0917 888 9908"),
)

CANNED_MESSAGES = (
    "Hey, are you free later?",
    "Did you see the game last night?",
    "Can you grab groceries on your way home?",
    "Talk soon!",
    "Miss you -- call me back.",
    "Are we still on for dinner?",
    "Happy birthday!",
    "Let me know when you're around.",
)


class PhoneState(Enum):
    IDLE = auto()
    IN_CALL = auto()


class PhoneEvent(Enum):
    CALL = auto()
    HANG_UP = auto()


class PhoneService:
    """Explicit `(state, event) -> next_state` table + ignore-on-unhandled.

    `MESSAGE` is intentionally not in the table: it is an orthogonal log
    action (like lights' color), not a call-state transition.
    """

    TRANSITIONS = {
        # Call from idle, or a fresh call that replaces the current one.
        (PhoneState.IDLE, PhoneEvent.CALL): PhoneState.IN_CALL,
        (PhoneState.IN_CALL, PhoneEvent.CALL): PhoneState.IN_CALL,
        # Hang up only makes sense mid-call (UI-only; no voice intent).
        (PhoneState.IN_CALL, PhoneEvent.HANG_UP): PhoneState.IDLE,
        # Ignored (no entry): HANG_UP while IDLE.
    }

    def __init__(
        self,
        rng: random.Random | None = None,
        now: Now | None = None,
    ) -> None:
        self._rng = rng if rng is not None else random.Random()
        self._now = now if now is not None else datetime.now
        self.state = PhoneState.IDLE
        self.person: dict | None = None  # {"name": ..., "number": ...}
        self.elapsed_s = 0
        self.messages: list[dict] = []  # chronological

    def dispatch(self, event: PhoneEvent) -> tuple[bool, str]:
        """Apply `event` if a transition is defined; else ignore."""
        next_state = self.TRANSITIONS.get((self.state, event))
        if next_state is None:
            return False, (
                f"Ignored: cannot {event.name.lower()} while {self.state.name.lower()}"
            )
        if event is PhoneEvent.CALL:
            was_in_call = self.state is PhoneState.IN_CALL
            name, number = self._rng.choice(list(CONTACTS))
            self.person = {"name": name, "number": number}
            self.elapsed_s = 0
            self.state = next_state
            verb = "new call to" if was_in_call else "call to"
            return True, f"{verb} {name} ({number})"
        # HANG_UP
        hung_up = self.person["name"] if self.person else "?"
        self.state = next_state
        self.person = None
        self.elapsed_s = 0
        return True, f"hang up (was {hung_up})"

    def call(self) -> tuple[bool, str]:
        return self.dispatch(PhoneEvent.CALL)

    def hang_up(self) -> tuple[bool, str]:
        return self.dispatch(PhoneEvent.HANG_UP)

    def message(self) -> tuple[bool, str]:
        """Append a canned text to a random contact (any call state)."""
        name, _number = self._rng.choice(list(CONTACTS))
        text = self._rng.choice(list(CANNED_MESSAGES))
        at = self._now().strftime("%H:%M:%S")
        self.messages.append({"name": name, "text": text, "at": at})
        return True, f"message to {name}: {text}"

    def tick(self) -> None:
        """One second of app time: advance the call elapsed ticker."""
        if self.state is PhoneState.IN_CALL:
            self.elapsed_s += 1

    def to_dict(self) -> dict:
        return {
            "state": self.state.name,
            "person": self.person,
            "elapsed_s": self.elapsed_s,
            "messages": list(reversed(self.messages)),  # newest first
        }
