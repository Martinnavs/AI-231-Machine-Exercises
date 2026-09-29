"""Reminders service: an append-only, log-and-list task list.

Tasks are restricted to the grammar's three `TASK` values (drink water /
study / exercise) -- for voice *and* for the UI buttons (feature
`ui-site`, decision Q11), so the vocabulary stays grammar-derived
(`app.intents.REMINDER_TASKS`) rather than re-listed here. Each reminder
is stamped with the machine clock's `HH:MM:SS` at the moment it is logged
(for a voice command, that is when the command was spoken, relative to
this machine's clock). Items are stored, listed, and never fired and
never deleted (no reminders fire in this simulation).

The panel shows only the latest item (or nothing) until `LIST_REMINDERS`
expands it to the full newest-first list; a collapse button re-collapses
it (UI-only -- there is no collapse voice intent, mirroring hang-up).
"""

from __future__ import annotations

from datetime import datetime
from typing import Callable

from app.intents import REMINDER_TASKS

Now = Callable[[], datetime]


class ReminderService:
    """Append-only list + collapsed/expanded panel flag.

    Not a `(state, event)` machine in the lights sense: `add`/`expand`/
    `collapse` are valid from any state; the only "ignored" case is an
    unknown task (the task vocabulary is the grammar's, closed world).
    """

    def __init__(self, now: Now | None = None) -> None:
        self._now = now if now is not None else datetime.now
        self._items: list[dict] = []  # chronological
        self._next_id = 1
        self.expanded = False

    @property
    def items(self) -> list[dict]:
        """All reminders, newest first."""
        return list(reversed(self._items))

    @property
    def latest(self) -> dict | None:
        """The most recently logged reminder, or None if none yet."""
        return self.items[0] if self.items else None

    def add(self, task: str) -> tuple[bool, str]:
        """Log a reminder stamped with the current machine time."""
        if task not in REMINDER_TASKS:
            return False, f"Ignored: unknown task {task!r}"
        item = {
            "id": self._next_id,
            "task": task,
            "logged_at": self._now().strftime("%H:%M:%S"),
        }
        self._next_id += 1
        self._items.append(item)
        return True, f"reminder logged: {task}"

    def expand(self) -> tuple[bool, str]:
        self.expanded = True
        return True, "reminders list expanded"

    def collapse(self) -> tuple[bool, str]:
        self.expanded = False
        return True, "reminders list collapsed"

    def to_dict(self) -> dict:
        return {
            "expanded": self.expanded,
            "items": self.items,
            "latest": self.latest,
        }
