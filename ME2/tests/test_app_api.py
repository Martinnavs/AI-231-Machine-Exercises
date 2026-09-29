"""API surface: pages, intake endpoints, every device REST action,
vocab, and the WebSocket initial snapshot (feature `ui-site`)."""

import json

from fastapi.testclient import TestClient

from app.main import create_app


def make_client() -> TestClient:
    return TestClient(create_app())


def test_index_serves_the_dashboard():
    with make_client() as client:
        r = client.get("/")
        assert r.status_code == 200
        assert "voice dashboard" in r.text


def test_static_assets_are_served():
    with make_client() as client:
        assert client.get("/static/style.css").status_code == 200
        assert client.get("/static/app.js").status_code == 200


def test_state_shape():
    with make_client() as client:
        data = client.get("/api/state").json()
        assert set(data) == {
            "lights", "music", "phone", "reminders",
            "thermostat", "timer", "alarm", "indicator",
        }


def test_vocab_lists_match_the_closed_wordlists():
    with make_client() as client:
        v = client.get("/api/vocab").json()
        assert v["reminder_tasks"] == ["drink water", "exercise", "study"]
        assert v["alarm_times"] == ["6 AM", "8 AM", "9 PM"]
        assert v["timer_presets_s"] == [10, 30, 60]
        assert v["thermostat_degrees"] == [18, 22, 26]
        assert v["light_levels"] == [20, 60, 100]
        assert v["light_colors"] == ["white", "red", "green", "blue"]
        assert len(v["tracks"]) == 5


def test_command_endpoint_mutates_state():
    with make_client() as client:
        client.post("/api/light/power_on")  # lights start OFF
        r = client.post(
            "/api/command",
            json={"intent": "BRIGHTNESS", "slots": {"PERCENT": "60 percent"}},
        )
        assert r.status_code == 200
        body = r.json()
        assert body["ok"] and body["changed"]
        assert body["detected"] == "Brightness 60 percent"
        assert body["state"]["lights"]["brightness"] == 60


def test_ignored_command_returns_200_with_unchanged_state():
    with make_client() as client:
        # BRIGHTNESS while OFF is ignored by the lights' state machine.
        r = client.post(
            "/api/command",
            json={"intent": "BRIGHTNESS", "slots": {"PERCENT": "60 percent"}},
        )
        assert r.status_code == 200
        body = r.json()
        assert body["ok"] and not body["changed"]
        assert body["state"]["lights"]["state"] == "OFF"


def test_unknown_intent_is_400():
    with make_client() as client:
        r = client.post("/api/command", json={"intent": "DANCE", "slots": {}})
        assert r.status_code == 400
        assert "unknown intent" in r.json()["error"]


def test_missing_required_slot_is_400():
    with make_client() as client:
        r = client.post("/api/command", json={"intent": "BRIGHTNESS", "slots": {}})
        assert r.status_code == 400
        assert "PERCENT" in r.json()["error"]


def test_non_string_intent_is_422():
    with make_client() as client:
        r = client.post("/api/command", json={"intent": 123, "slots": {}})
        assert r.status_code == 422


def test_listening_endpoint_sets_the_indicator():
    with make_client() as client:
        r = client.post("/api/listening", json={"state": "active"})
        assert r.status_code == 200
        body = r.json()
        assert body["changed"]
        assert body["state"]["indicator"]["listening"] == "active"
        r = client.post("/api/listening", json={"state": "passive"})
        assert r.json()["state"]["indicator"]["listening"] == "passive"


def test_listening_endpoint_rejects_unknown_states():
    with make_client() as client:
        r = client.post("/api/listening", json={"state": "bogus"})
        assert r.status_code == 400


def test_light_action_endpoints():
    with make_client() as client:
        assert client.post("/api/light/power_on").json()["state"]["lights"]["brightness"] == 20
        assert client.post("/api/light/increase").json()["state"]["lights"]["brightness"] == 60
        assert client.post("/api/light/set_brightness", json={"level": 100}).json()["state"]["lights"]["brightness"] == 100
        assert client.post("/api/light/set_color", json={"color": "blue"}).json()["state"]["lights"]["color"] == "blue"
        assert client.post("/api/light/power_off").json()["state"]["lights"]["on"] is False
        body = client.post("/api/light/decrease").json()
        assert body["changed"] is False  # OFF: no transition


def test_music_action_endpoints():
    with make_client() as client:
        assert client.post("/api/music/play").json()["state"]["music"]["state"] == "PLAYING"
        assert client.post("/api/music/pause").json()["state"]["music"]["state"] == "PAUSED"
        assert client.post("/api/music/next").json()["state"]["music"]["track_index"] == 1
        assert client.post("/api/music/volume_up").json()["state"]["music"]["volume"] == 60
        assert client.post("/api/music/volume_down").json()["state"]["music"]["volume"] == 50
        assert client.post("/api/music/stop").json()["state"]["music"]["state"] == "STOPPED"
        assert client.post("/api/music/stop").json()["state"]["music"]["track_index"] == 0


def test_phone_action_endpoints():
    with make_client() as client:
        body = client.post("/api/phone/call").json()
        assert body["state"]["phone"]["state"] == "IN_CALL"
        assert body["state"]["phone"]["person"] is not None
        body = client.post("/api/phone/message").json()
        assert len(body["state"]["phone"]["messages"]) == 1
        body = client.post("/api/phone/hang_up").json()
        assert body["state"]["phone"]["state"] == "IDLE"
        body = client.post("/api/phone/hang_up").json()
        assert body["changed"] is False


def test_reminder_endpoints():
    with make_client() as client:
        body = client.post("/api/reminders", json={"task": "exercise"}).json()
        assert body["state"]["reminders"]["latest"]["task"] == "exercise"
        body = client.post("/api/reminders/expand").json()
        assert body["state"]["reminders"]["expanded"] is True
        body = client.post("/api/reminders/collapse").json()
        assert body["state"]["reminders"]["expanded"] is False
        body = client.post("/api/reminders", json={"task": "walk the dog"}).json()
        assert body["changed"] is False  # not one of the grammar's tasks


def test_thermostat_endpoints():
    with make_client() as client:
        assert client.post("/api/thermostat", json={"degrees": 26}).json()["state"]["thermostat"]["degrees"] == 26
        body = client.post("/api/thermostat", json={"degrees": 25}).json()
        assert body["changed"] is False


def test_timer_endpoints():
    with make_client() as client:
        body = client.post("/api/timer/start", json={"duration_s": 30}).json()
        assert body["state"]["timer"]["state"] == "RUNNING"
        body = client.post("/api/timer/start", json={"duration_s": 30}).json()
        assert body["changed"] is False  # re-issue while running: ignored
        body = client.post("/api/timer/reset").json()
        assert body["state"]["timer"]["state"] == "IDLE"


def test_alarm_endpoints():
    with make_client() as client:
        body = client.post("/api/alarm/set", json={"time": "9 PM"}).json()
        assert body["state"]["alarm"]["state"] == "SET"
        assert body["state"]["alarm"]["display"] == "09:00 PM"
        body = client.post("/api/alarm/set", json={"time": "7 AM"}).json()
        assert body["changed"] is False
        body = client.post("/api/alarm/reset").json()
        assert body["state"]["alarm"]["state"] == "UNSET"
        body = client.post("/api/alarm/reset").json()
        assert body["changed"] is False


def test_websocket_sends_the_initial_snapshot():
    with make_client() as client:
        with client.websocket_connect("/ws") as websocket:
            data = json.loads(websocket.receive_text())
            assert "lights" in data
            assert "indicator" in data
            assert data["indicator"]["listening"] == "passive"
