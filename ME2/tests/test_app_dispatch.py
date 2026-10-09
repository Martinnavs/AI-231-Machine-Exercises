"""Intent -> device-action mapping: every one of the 19 grammar intents
reaches exactly the right service; malformed commands 400; bad slot
values are ignored (200, unchanged) (feature `ui-site`)."""

import random
from datetime import datetime

from app.dispatch import handle_command
from app.state import AppState


def make_state() -> AppState:
    return AppState(
        now=lambda: datetime(2026, 9, 29, 14, 32, 7),
        rng=random.Random(42),
    )


def test_all_nineteen_intents_reach_their_services():
    st = make_state()

    r = handle_command(st, "PLAY_MUSIC", {})
    assert r["ok"] and r["changed"]
    assert st.music.state.name == "PLAYING"

    r = handle_command(st, "PAUSE", {})
    assert r["changed"] and st.music.state.name == "PAUSED"

    r = handle_command(st, "STOP", {})
    assert r["changed"] and st.music.state.name == "STOPPED"

    st.music.play()
    r = handle_command(st, "NEXT", {})
    assert r["changed"] and st.music.track_index == 1

    r = handle_command(st, "VOLUME_UP", {})
    assert st.music.volume == 60
    r = handle_command(st, "VOLUME_DOWN", {})
    assert st.music.volume == 50

    r = handle_command(st, "LIGHT_ON", {})
    assert st.lights.state.name == "ON_20"
    r = handle_command(st, "LIGHT_OFF", {})
    assert st.lights.state.name == "OFF"
    handle_command(st, "LIGHT_ON", {})
    r = handle_command(st, "BRIGHTNESS", {"PERCENT": "60 percent"})
    assert r["changed"] and st.lights.brightness == 60
    r = handle_command(st, "COLOR", {"COLOR": "red"})
    assert r["changed"] and st.lights.color == "red"

    r = handle_command(st, "TEMPERATURE", {"DEGREES": "18 degrees"})
    assert r["changed"] and st.thermostat.degrees == 18

    r = handle_command(st, "TIME", {})
    assert r["changed"]
    assert st.indicator.status_line == "Current time is 14:32"
    assert st.indicator.detected == "Time"

    r = handle_command(st, "WEATHER", {})
    assert r["changed"]
    assert st.indicator.status_line == "Weather is sunny, 24 degrees"
    assert st.indicator.detected == "Weather"

    r = handle_command(st, "TIMER", {"DURATION": "1 minute"})
    assert r["changed"]
    assert st.timer.state.name == "RUNNING"
    assert st.timer.remaining_s == 60

    r = handle_command(st, "ALARM", {"ALARM_TIME": "6 AM"})
    assert r["changed"] and st.alarm.state.name == "SET"

    r = handle_command(st, "CALL", {})
    assert r["changed"] and st.phone.state.name == "IN_CALL"

    r = handle_command(st, "MESSAGE", {})
    assert r["changed"] and len(st.phone.messages) == 1

    r = handle_command(st, "CREATE_REMINDER", {"TASK": "study"})
    assert r["changed"] and st.reminders.latest["task"] == "study"

    r = handle_command(st, "LIST_REMINDERS", {})
    assert r["changed"] and st.reminders.expanded is True


def test_unknown_intent_rejected():
    r = handle_command(make_state(), "DANCE", {})
    assert not r["ok"]
    assert "unknown intent" in r["error"]


def test_missing_required_slot_rejected():
    r = handle_command(make_state(), "BRIGHTNESS", {})
    assert not r["ok"]
    assert "PERCENT" in r["error"]


def test_bad_slot_value_is_ignored_not_rejected():
    st = make_state()
    r = handle_command(st, "TEMPERATURE", {"DEGREES": "25 degrees"})
    assert r["ok"]
    assert not r["changed"]
    assert st.thermostat.degrees == 22  # unchanged


def test_detected_line_updates_on_every_command():
    st = make_state()
    handle_command(st, "PLAY_MUSIC", {})
    assert st.indicator.detected == "Play Music"
    handle_command(st, "LIGHT_ON", {})
    assert st.indicator.detected == "Lights On"
    handle_command(st, "ALARM", {"ALARM_TIME": "9 PM"})
    assert st.indicator.detected == "Alarm 9 PM"


def test_other_commands_do_not_clear_the_status_line():
    st = make_state()
    handle_command(st, "TIME", {})
    status = st.indicator.status_line
    handle_command(st, "PLAY_MUSIC", {})
    assert st.indicator.status_line == status  # only TIME/WEATHER touch it


def test_result_carries_the_full_snapshot():
    r = handle_command(make_state(), "PLAY_MUSIC", {})
    assert set(r["state"]) == {
        "lights", "music", "phone", "reminders",
        "thermostat", "timer", "alarm", "indicator",
    }
    assert r["state"]["music"]["state"] == "PLAYING"
    assert r["state"]["indicator"]["detected"] == "Play Music"
