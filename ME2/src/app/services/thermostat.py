"""Thermostat service: discrete set-point selection.

The thermostat only understands three set points -- 18/22/26 degrees
(feature `ui-site`, decision Q4: the grammar's `DEGREES` values), default
22. Every set point is reachable from every current set point (a full
3x3 transition table); any other value is ignored (strict closed-world
validation, same stance as the lights' brightness scale).
"""

from __future__ import annotations

from enum import Enum, auto


class ThermoState(Enum):
    T18 = 18
    T22 = 22
    T26 = 26


class ThermoEvent(Enum):
    SET_18 = auto()
    SET_22 = auto()
    SET_26 = auto()


class ThermostatService:
    """Explicit `(state, event) -> next_state` table + ignore-on-unhandled."""

    DEGREES = (18, 22, 26)

    # Full table: any set point is reachable from any other.
    TRANSITIONS = {
        (_from, event): target
        for _from in ThermoState
        for event, target in (
            (ThermoEvent.SET_18, ThermoState.T18),
            (ThermoEvent.SET_22, ThermoState.T22),
            (ThermoEvent.SET_26, ThermoState.T26),
        )
    }

    def __init__(self) -> None:
        self.state = ThermoState.T22

    @property
    def degrees(self) -> int:
        return self.state.value

    def dispatch(self, event: ThermoEvent) -> tuple[bool, str]:
        """Apply `event` if a transition is defined; else ignore."""
        next_state = self.TRANSITIONS.get((self.state, event))
        if next_state is None:
            return False, (
                f"Ignored: cannot {event.name.lower()} while {self.state.name.lower()}"
            )
        self.state = next_state
        return True, f"[{event.name}] -> {self.state.name} ({self.state.value} degrees)"

    def set(self, degrees: int | str | None) -> tuple[bool, str]:
        """Direct set from a canonical degrees value (grammar `DEGREES`,
        e.g. "18 degrees", or a bare int)."""
        n = None
        if isinstance(degrees, str):
            digits = "".join(ch for ch in degrees if ch.isdigit())
            n = int(digits) if digits else None
        elif isinstance(degrees, int):
            n = degrees
        if n not in self.DEGREES:
            return False, f"Ignored: thermostat must be 18/22/26 degrees, got {degrees!r}"
        event = {18: ThermoEvent.SET_18, 22: ThermoEvent.SET_22, 26: ThermoEvent.SET_26}[n]
        return self.dispatch(event)

    def to_dict(self) -> dict:
        return {"degrees": self.degrees}
