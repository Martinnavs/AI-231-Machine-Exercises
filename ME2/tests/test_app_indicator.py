"""Indicator service: pure-receiver listening state, canonical
detected-word line, 5-second temporary status line (decisions: option A
driver; Q5 output; 5s temporary display)."""

from app.services.indicator import (
    LISTENING_STATES,
    STATUS_TTL_S,
    IndicatorService,
)


def test_defaults_passive_blank():
    svc = IndicatorService()
    assert svc.listening == "passive"
    assert svc.detected is None
    assert svc.status_line is None
    assert svc.to_dict() == {
        "listening": "passive",
        "detected": None,
        "status_line": None,
    }


def test_listening_states_are_binary():
    assert LISTENING_STATES == ("passive", "active")


def test_set_listening_reports_only_real_changes():
    svc = IndicatorService()
    assert svc.set_listening("active") == (True, "listening -> active")
    assert svc.set_listening("active") == (False, "listening -> active")
    assert svc.set_listening("passive")[0] is True


def test_unknown_listening_state_ignored():
    svc = IndicatorService()
    changed, msg = svc.set_listening("bogus")
    assert not changed
    assert msg.startswith("Ignored")
    assert svc.listening == "passive"


def test_on_command_sets_the_canonical_detected_line():
    svc = IndicatorService()
    assert svc.on_command("BRIGHTNESS", {"PERCENT": "60 percent"}) == (
        "Brightness 60 percent"
    )
    assert svc.on_command("PLAY_MUSIC", {}) == "Play Music"
    assert svc.detected == "Play Music"


def test_status_line_expires_after_five_ticks():
    svc = IndicatorService()
    svc.set_status("Current time is 14:32")
    for _ in range(4):
        svc.tick()
    assert svc.status_line == "Current time is 14:32"
    svc.tick()  # fifth tick
    assert svc.status_line is None


def test_new_status_overrides_and_restarts_the_ttl():
    svc = IndicatorService()
    svc.set_status("Current time is 14:32")
    for _ in range(4):
        svc.tick()
    svc.set_status("Weather is sunny, 24 degrees")
    for _ in range(4):
        svc.tick()
    assert svc.status_line == "Weather is sunny, 24 degrees"
    svc.tick()
    assert svc.status_line is None


def test_status_ttl_is_five_seconds():
    assert STATUS_TTL_S == 5
