"""Lights service: power + brightness state machine, with an orthogonal color.

Patterned on the reference `LightStateMachine` (feature `ui-site`). The
brightness scale is **OFF/20/60/100** (not the reference sketch's
25/50/75/100): it is rescaled so the Option B grammar's `BRIGHTNESS` intent
(which can only ever say 20/60/100 percent) lands exactly on a valid state, and
it matches the requirement's own UI line "Brightness (0, 20, 60, 100%)"
(decision Q3). INCREASE/DECREASE step 20 -> 60 -> 100 and cap at both ends;
POWER_ON always lands on the lowest on-level (20) and never resumes a previous
level; a `BRIGHTNESS` command while OFF is **ignored** (strict state-machine
validation -- decision Q6).

Color (red/green/blue from the grammar; white is the default/UI-only) is an
**orthogonal attribute**: it is stored on the service, settable while on or off,
and retained across power cycles. The brightness state machine stays
power/brightness-only (decision Q3).
"""

from __future__ import annotations

from enum import Enum, auto


class LightState(Enum):
    OFF = 0
    ON_20 = 20
    ON_60 = 60
    ON_100 = 100


class LightEvent(Enum):
    POWER_ON = auto()
    POWER_OFF = auto()
    INCREASE = auto()
    DECREASE = auto()
    SET_20 = auto()
    SET_60 = auto()
    SET_100 = auto()


# Map a canonical BRIGHTNESS percent value ("20 percent"/"60 percent"/
# "100 percent", or a bare int) to the SET_* event.
def brightness_event(percent: int | str) -> LightEvent | None:
    if isinstance(percent, str):
        digits = "".join(ch for ch in percent if ch.isdigit())
        percent = int(digits) if digits else None
    if percent == 20:
        return LightEvent.SET_20
    if percent == 60:
        return LightEvent.SET_60
    if percent == 100:
        return LightEvent.SET_100
    return None


class LightService:
    """Explicit `(state, event) -> next_state` table + ignore-on-unhandled."""

    TRANSITIONS = {
        # Power on defaults to the lowest on-level (20); never resumes.
        (LightState.OFF, LightEvent.POWER_ON): LightState.ON_20,
        # Power off from any active state.
        (LightState.ON_20, LightEvent.POWER_OFF): LightState.OFF,
        (LightState.ON_60, LightEvent.POWER_OFF): LightState.OFF,
        (LightState.ON_100, LightEvent.POWER_OFF): LightState.OFF,
        # Increase (caps at 100).
        (LightState.ON_20, LightEvent.INCREASE): LightState.ON_60,
        (LightState.ON_60, LightEvent.INCREASE): LightState.ON_100,
        (LightState.ON_100, LightEvent.INCREASE): LightState.ON_100,
        # Decrease (caps at 20).
        (LightState.ON_100, LightEvent.DECREASE): LightState.ON_60,
        (LightState.ON_60, LightEvent.DECREASE): LightState.ON_20,
        (LightState.ON_20, LightEvent.DECREASE): LightState.ON_20,
        # Direct set, from any active state (ignored while OFF -- no entry).
        (LightState.ON_20, LightEvent.SET_20): LightState.ON_20,
        (LightState.ON_60, LightEvent.SET_20): LightState.ON_20,
        (LightState.ON_100, LightEvent.SET_20): LightState.ON_20,
        (LightState.ON_20, LightEvent.SET_60): LightState.ON_60,
        (LightState.ON_60, LightEvent.SET_60): LightState.ON_60,
        (LightState.ON_100, LightEvent.SET_60): LightState.ON_60,
        (LightState.ON_20, LightEvent.SET_100): LightState.ON_100,
        (LightState.ON_60, LightEvent.SET_100): LightState.ON_100,
        (LightState.ON_100, LightEvent.SET_100): LightState.ON_100,
    }

    # red/green/blue are voice-settable (the grammar's COLOR slot); white is the
    # default and the UI-only extra.
    COLORS = ("white", "red", "green", "blue")

    def __init__(self) -> None:
        self.state = LightState.OFF
        self.color = "white"

    def dispatch(self, event: LightEvent) -> tuple[bool, str]:
        """Apply `event` if a transition is defined; else ignore. Returns
        (changed, message) mirroring the reference `dispatch`'s output."""
        next_state = self.TRANSITIONS.get((self.state, event))
        if next_state is None:
            return False, (
                f"Ignored: cannot {event.name.lower()} while {self.state.name.lower()}"
            )
        self.state = next_state
        return True, f"[{event.name}] -> {self.state.name} ({self.state.value}%)"

    def set_brightness(self, percent: int | str) -> tuple[bool, str]:
        """Direct set from a canonical percent value (grammar `PERCENT`)."""
        event = brightness_event(percent)
        if event is None:
            return False, f"Ignored: brightness {percent!r} is not 20/60/100"
        return self.dispatch(event)

    def set_color(self, color: str) -> tuple[bool, str]:
        """Orthogonal color attribute; valid on or off, retained across cycles."""
        if color not in self.COLORS:
            return False, f"Ignored: unknown color {color!r}"
        self.color = color
        return True, f"color -> {color}"

    @property
    def brightness(self) -> int:
        return self.state.value

    def to_dict(self) -> dict:
        return {
            "state": self.state.name,
            "on": self.state is not LightState.OFF,
            "brightness": self.brightness,
            "color": self.color,
        }
