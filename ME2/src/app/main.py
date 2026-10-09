"""FastAPI app for the UI site (feature `ui-site`).

Run it in its own terminal window (it is a **separate service** from the
voice pipeline -- the model side reaches it one-way over this HTTP API):

    make app                        # uvicorn app.main:app, 127.0.0.1:8000

In terminal 1 the model-side forwarder (`python -m app.forward`, piped
from the streaming runner or replaying a JSONL file) posts
`POST /api/command` and `POST /api/listening`. The browser talks to this
same app over REST (local clicks) and one WebSocket (pushed state,
including the 1-second tick). Single global in-memory state, no
persistence, no auth (decisions Q7/Q8).
"""

from __future__ import annotations

import asyncio
import json
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Callable

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from . import dispatch
from .intents import REMINDER_TASKS
from .services.alarm import AlarmService
from .services.lights import LightEvent, LightService, LightState
from .services.music_player import default_music
from .services.thermostat import ThermostatService
from .services.timer import TimerService
from .services.indicator import LISTENING_STATES
from .state import AppState

STATIC_DIR = Path(__file__).parent / "static"
TICK_INTERVAL_S = 1.0


# --- request bodies -------------------------------------------------------

class CommandBody(BaseModel):
    intent: str
    slots: dict[str, str] = {}


class ListeningBody(BaseModel):
    state: str


class LightColorBody(BaseModel):
    color: str


class LightLevelBody(BaseModel):
    level: int


class ThermostatBody(BaseModel):
    degrees: int


class TimerStartBody(BaseModel):
    duration_s: int


class AlarmSetBody(BaseModel):
    time: str


class ReminderBody(BaseModel):
    task: str


# --- websocket fan-out -----------------------------------------------------

class _Connections:
    """The set of live browser sockets; one broadcast per state change."""

    def __init__(self) -> None:
        self._clients: set[WebSocket] = set()

    async def connect(self, ws: WebSocket) -> None:
        await ws.accept()
        self._clients.add(ws)

    def disconnect(self, ws: WebSocket) -> None:
        self._clients.discard(ws)

    async def broadcast(self, payload: str) -> None:
        for ws in list(self._clients):
            try:
                await ws.send_text(payload)
            except Exception:
                self._clients.discard(ws)


def create_app(state: AppState | None = None) -> FastAPI:
    """Build the FastAPI app around `state` (a fresh AppState by default).

    Tests pass in a state with an injected clock/RNG; `uvicorn
    app.main:app` uses the module-level `app` below, which builds a real
    one.
    """
    state = state if state is not None else AppState()
    conns = _Connections()

    @asynccontextmanager
    async def lifespan(_app: FastAPI):
        async def tick_loop():
            while True:
                await asyncio.sleep(TICK_INTERVAL_S)
                state.tick()
                await conns.broadcast(json.dumps(state.snapshot()))

        task = asyncio.create_task(tick_loop())
        try:
            yield
        finally:
            task.cancel()
            if state.music.player is not None:
                state.music.player.stop()

    app = FastAPI(title="UI Site", lifespan=lifespan)

    def _respond(changed: bool, message: str) -> dict:
        return {"changed": changed, "message": message, "state": state.snapshot()}

    async def _push() -> None:
        await conns.broadcast(json.dumps(state.snapshot()))

    # --- pages -------------------------------------------------------------

    @app.get("/", response_class=HTMLResponse)
    async def index() -> str:
        return (STATIC_DIR / "index.html").read_text()

    # --- model-facing intake (one-way: the forwarder posts here) -----------

    @app.post("/api/command")
    async def api_command(body: CommandBody):
        result = dispatch.handle_command(state, body.intent, body.slots)
        if not result["ok"]:
            return JSONResponse(status_code=400, content=result)
        await _push()
        return result

    @app.post("/api/listening")
    async def api_listening(body: ListeningBody):
        if body.state not in LISTENING_STATES:
            return JSONResponse(
                status_code=400,
                content={"ok": False, "error": f"unknown listening state {body.state!r}"},
            )
        changed, message = state.indicator.set_listening(body.state)
        state.music.on_listening(body.state)
        if state.music.soft_paused and body.state == "passive":
            # resume promptly after the grace instead of waiting for the 1 s tick
            asyncio.get_running_loop().call_later(
                state.music.RESUME_GRACE_S + 0.05,
                lambda: (state.music.tick(), asyncio.ensure_future(_push())),
            )
        await _push()
        return _respond(changed, message)

    # --- local UI clicks (REST actions) -------------------------------------

    def _no_body_route(path: str, action: Callable[[], tuple[bool, str]]) -> None:
        async def handler():
            changed, message = action()
            await _push()
            return _respond(changed, message)

        handler.__name__ = path.strip("/").replace("/", "_")
        app.add_api_route(path, handler, methods=["POST"])

    _no_body_route("/api/light/power_on", lambda: state.lights.dispatch(LightEvent.POWER_ON))
    _no_body_route("/api/light/power_off", lambda: state.lights.dispatch(LightEvent.POWER_OFF))
    _no_body_route("/api/light/increase", lambda: state.lights.dispatch(LightEvent.INCREASE))
    _no_body_route("/api/light/decrease", lambda: state.lights.dispatch(LightEvent.DECREASE))
    _no_body_route("/api/music/play", state.music.play)
    _no_body_route("/api/music/pause", state.music.pause)
    _no_body_route("/api/music/next", state.music.next_track)
    _no_body_route("/api/music/stop", state.music.stop)
    _no_body_route("/api/music/volume_up", state.music.volume_up)
    _no_body_route("/api/music/volume_down", state.music.volume_down)
    _no_body_route("/api/phone/call", state.phone.call)
    _no_body_route("/api/phone/hang_up", state.phone.hang_up)
    _no_body_route("/api/phone/message", state.phone.message)
    _no_body_route("/api/reminders/expand", state.reminders.expand)
    _no_body_route("/api/reminders/collapse", state.reminders.collapse)
    _no_body_route("/api/timer/reset", state.timer.reset)
    _no_body_route("/api/alarm/reset", state.alarm.reset)

    @app.post("/api/light/set_color")
    async def api_light_color(body: LightColorBody):
        changed, message = state.lights.set_color(body.color)
        await _push()
        return _respond(changed, message)

    @app.post("/api/light/set_brightness")
    async def api_light_brightness(body: LightLevelBody):
        changed, message = state.lights.set_brightness(body.level)
        await _push()
        return _respond(changed, message)

    @app.post("/api/thermostat")
    async def api_thermostat(body: ThermostatBody):
        changed, message = state.thermostat.set(body.degrees)
        await _push()
        return _respond(changed, message)

    @app.post("/api/timer/start")
    async def api_timer_start(body: TimerStartBody):
        changed, message = state.timer.start(body.duration_s)
        await _push()
        return _respond(changed, message)

    @app.post("/api/alarm/set")
    async def api_alarm_set(body: AlarmSetBody):
        changed, message = state.alarm.set(body.time)
        await _push()
        return _respond(changed, message)

    @app.post("/api/reminders")
    async def api_reminder_add(body: ReminderBody):
        changed, message = state.reminders.add(body.task)
        await _push()
        return _respond(changed, message)

    # --- read models ---------------------------------------------------------

    @app.get("/api/state")
    async def api_state():
        return state.snapshot()

    @app.get("/api/vocab")
    async def api_vocab():
        """The closed choice lists the UI buttons/dropdowns render from
        (grammar-derived where the device's vocabulary *is* the grammar's)."""
        return {
            "reminder_tasks": sorted(REMINDER_TASKS),
            "alarm_times": sorted(AlarmService.ALARM_TIMES),
            "timer_presets_s": list(TimerService.PRESETS),
            "light_colors": list(LightService.COLORS),
            "light_levels": [s.value for s in LightState if s is not LightState.OFF],
            "thermostat_degrees": sorted(ThermostatService.DEGREES),
            "tracks": list(state.music.tracks),
        }

    # --- websocket (push) -----------------------------------------------------

    @app.websocket("/ws")
    async def ws_endpoint(ws: WebSocket):
        await conns.connect(ws)
        try:
            await ws.send_text(json.dumps(state.snapshot()))
            while True:
                # The UI is a pure receiver over WS; ignore inbound text.
                await ws.receive_text()
        except WebSocketDisconnect:
            pass
        finally:
            conns.disconnect(ws)

    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
    return app


app = create_app(AppState(music=default_music()))
