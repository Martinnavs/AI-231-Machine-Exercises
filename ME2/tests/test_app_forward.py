"""Forwarder: JSONL parsing, event mapping, synthesis, and replay
planning (the pure, network-free parts -- feature `ui-site`)."""

import json

from app.forward import (
    LISTENING_LEAD_S,
    build_events,
    parse_line,
    record_events,
)


def make_trigger(t: float, intent: str = "CALL", slots: dict | None = None) -> dict:
    return {
        "event": "trigger",
        "t_seconds": t,
        "window_index": 1,
        "intent": intent,
        "slots": slots or {},
        "text": "call",
        "confidence": -0.5,
        "policy_reason": "intent='CALL' confidence=-0.5 >= threshold -1.0",
    }


def test_parse_line_valid():
    assert parse_line(json.dumps({"event": "window"})) == {"event": "window"}


def test_parse_line_blank_and_malformed_and_non_dict():
    assert parse_line("") is None
    assert parse_line("   ") is None
    assert parse_line("not json {") is None
    assert parse_line("[1, 2]") is None
    assert parse_line('"just a string"') is None


def test_trigger_maps_to_one_command_post():
    events = record_events(make_trigger(4.5))
    assert events == [(4.5, "/api/command", {"intent": "CALL", "slots": {}})]


def test_trigger_passes_slots_through():
    events = record_events(make_trigger(2.0, "BRIGHTNESS", {"PERCENT": "60 percent"}))
    assert events[0][2] == {
        "intent": "BRIGHTNESS",
        "slots": {"PERCENT": "60 percent"},
    }


def test_trigger_with_null_slots_defaults_to_empty():
    record = make_trigger(1.0)
    record["slots"] = None
    events = record_events(record)
    assert events[0][2]["slots"] == {}


def test_listening_record_maps_to_a_listening_post():
    events = record_events(
        {"event": "listening", "t_seconds": 1.0, "window_index": 4, "state": "active"}
    )
    assert events == [(1.0, "/api/listening", {"state": "active"})]


def test_window_and_unknown_records_map_to_nothing():
    assert record_events({"event": "window", "t_seconds": 0.5}) == []
    assert record_events({"event": "mystery", "t_seconds": 0.5}) == []


def test_synthesis_adds_active_and_passive_around_the_trigger():
    events = record_events(make_trigger(10.0), synthesize=True)
    assert (10.0 - LISTENING_LEAD_S, "/api/listening", {"state": "active"}) in events
    assert (10.0, "/api/listening", {"state": "passive"}) in events
    assert (10.0, "/api/command", {"intent": "CALL", "slots": {}}) in events
    assert len(events) == 3


def test_synthesis_lead_is_three_seconds():
    assert LISTENING_LEAD_S == 3.0


def test_build_events_sorts_by_time_and_filters_junk():
    lines = [
        json.dumps(make_trigger(5.0)),
        "",
        "not json",
        json.dumps({"event": "window", "t_seconds": 1.0}),
        json.dumps(make_trigger(2.0, "PAUSE")),
    ]
    events = build_events(lines)
    assert [t for t, _ep, _pl in events] == [2.0, 5.0]
    assert [ep for _t, ep, _pl in events] == ["/api/command", "/api/command"]


def test_build_events_default_has_no_listening_events():
    events = build_events([json.dumps(make_trigger(10.0))])
    assert all(ep == "/api/command" for _t, ep, _pl in events)


def test_build_events_with_synthesis_includes_listening_events():
    events = build_events([json.dumps(make_trigger(10.0))], synthesize=True)
    endpoints = [ep for _t, ep, _pl in events]
    assert endpoints.count("/api/listening") == 2
    assert endpoints.count("/api/command") == 1
    # active strictly before the trigger, passive at the trigger's time
    times = [t for t, ep, _pl in events if ep == "/api/listening"]
    assert times == [7.0, 10.0]
