"""Intent -> device-action mapping (feature `ui-site`).

The 19-intent vocabulary comes from `app.intents` (derived from the
production Option B grammar -- single source of truth). Each valid intent
maps to exactly one device action. The model side has already
grammar-decoded each command into `(intent, slots)`, so this layer only
validates the intent name, reads the (single) slot when the intent
carries one, and applies the action.

Two distinct failure modes, mirroring the services' policy:

- an **unknown intent** or a **missing required slot** is a malformed
  command -> `ok: False` (the API layer answers 400);
- a valid intent whose slot value the device does not accept is
  **ignored** -> `ok: True, changed: False` (the API layer answers 200
  with the unchanged state).

Every processed command also updates the indicator's detected-word line
(canonical label + slot, never the raw variant -- decision Q5); TIME and
WEATHER additionally set the indicator's temporary 5-second status line.
"""

from __future__ import annotations

from .intents import INTENTS, SLOT_FOR_INTENT
from .services.lights import LightEvent
from .state import AppState


def _leading_int(value: str) -> int | None:
    """The first run of digits in a canonical slot value, or None."""
    digits = "".join(ch for ch in value if ch.isdigit())
    return int(digits) if digits else None


def _duration_seconds(value: str) -> int | None:
    """Grammar `DURATION` -> seconds ("10 seconds"->10, "1 minute"->60)."""
    n = _leading_int(value)
    if n is None:
        return None
    return n * 60 if "minute" in value.lower() else n


# --- one handler per intent: (AppState, slots) -> (changed, message) -----

def _h_play_music(st: AppState, sl: dict) -> tuple[bool, str]:
    return st.music.play()


def _h_pause(st: AppState, sl: dict) -> tuple[bool, str]:
    return st.music.pause()


def _h_stop(st: AppState, sl: dict) -> tuple[bool, str]:
    return st.music.stop()


def _h_next(st: AppState, sl: dict) -> tuple[bool, str]:
    return st.music.next_track()


def _h_volume_up(st: AppState, sl: dict) -> tuple[bool, str]:
    return st.music.volume_up()


def _h_volume_down(st: AppState, sl: dict) -> tuple[bool, str]:
    return st.music.volume_down()


def _h_light_on(st: AppState, sl: dict) -> tuple[bool, str]:
    return st.lights.dispatch(LightEvent.POWER_ON)


def _h_light_off(st: AppState, sl: dict) -> tuple[bool, str]:
    return st.lights.dispatch(LightEvent.POWER_OFF)


def _h_brightness(st: AppState, sl: dict) -> tuple[bool, str]:
    return st.lights.set_brightness(_leading_int(sl["PERCENT"]))


def _h_color(st: AppState, sl: dict) -> tuple[bool, str]:
    return st.lights.set_color(sl["COLOR"])


def _h_temperature(st: AppState, sl: dict) -> tuple[bool, str]:
    return st.thermostat.set(_leading_int(sl["DEGREES"]))


def _h_time(st: AppState, sl: dict) -> tuple[bool, str]:
    text = f"Current time is {st.now().strftime('%H:%M')}"
    st.indicator.set_status(text)
    return True, text


def _h_weather(st: AppState, sl: dict) -> tuple[bool, str]:
    text = f"Weather is {st.weather.current()}"
    st.indicator.set_status(text)
    return True, text


def _h_timer(st: AppState, sl: dict) -> tuple[bool, str]:
    return st.timer.start(_duration_seconds(sl["DURATION"]))


def _h_alarm(st: AppState, sl: dict) -> tuple[bool, str]:
    return st.alarm.set(sl["ALARM_TIME"])


def _h_call(st: AppState, sl: dict) -> tuple[bool, str]:
    return st.phone.call()


def _h_message(st: AppState, sl: dict) -> tuple[bool, str]:
    return st.phone.message()


def _h_create_reminder(st: AppState, sl: dict) -> tuple[bool, str]:
    return st.reminders.add(sl["TASK"])


def _h_list_reminders(st: AppState, sl: dict) -> tuple[bool, str]:
    return st.reminders.expand()


# All 19 intents, every one mapped (unknown intents 400 before reaching
# the table; this is a safety net, not a dispatch path).
_ACTION_HANDLERS = {
    "PLAY_MUSIC": _h_play_music,
    "PAUSE": _h_pause,
    "STOP": _h_stop,
    "NEXT": _h_next,
    "VOLUME_UP": _h_volume_up,
    "VOLUME_DOWN": _h_volume_down,
    "LIGHT_ON": _h_light_on,
    "LIGHT_OFF": _h_light_off,
    "BRIGHTNESS": _h_brightness,
    "COLOR": _h_color,
    "TEMPERATURE": _h_temperature,
    "WEATHER": _h_weather,
    "TIME": _h_time,
    "TIMER": _h_timer,
    "ALARM": _h_alarm,
    "CALL": _h_call,
    "MESSAGE": _h_message,
    "CREATE_REMINDER": _h_create_reminder,
    "LIST_REMINDERS": _h_list_reminders,
}


def handle_command(
    state: AppState, intent: str, slots: dict[str, str]
) -> dict:
    """Process one decoded command. Returns the API result dict:
    `{"ok": True, "intent", "detected", "changed", "message", "state"}`
    or `{"ok": False, "error", "state"}` for a malformed command."""
    if intent not in INTENTS:
        return {
            "ok": False,
            "error": f"unknown intent {intent!r}",
            "state": state.snapshot(),
        }

    slot_name = SLOT_FOR_INTENT.get(intent)
    if slot_name is not None and slot_name not in slots:
        return {
            "ok": False,
            "error": f"intent {intent!r} is missing required slot {slot_name!r}",
            "state": state.snapshot(),
        }

    changed, message = _ACTION_HANDLERS[intent](state, slots)
    detected = state.indicator.on_command(intent, slots)
    return {
        "ok": True,
        "intent": intent,
        "detected": detected,
        "changed": changed,
        "message": message,
        "state": state.snapshot(),
    }
