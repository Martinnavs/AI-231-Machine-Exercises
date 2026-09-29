"""The single global in-memory app state (feature `ui-site`).

One `AppState` holds every device service plus the shared clock and the
weather seam. There is no persistence and no auth: state lives for the
process lifetime of the UI service (decisions Q7/Q8). `tick()` advances
the time-based services (timer countdown, phone elapsed, alarm due
check, indicator status TTL) once per second; the FastAPI app runs it on
a 1-second background loop and fans the resulting snapshot out over the
WebSocket.

Everything here is pure in-process state -- no I/O, no model imports,
no audio. The clock and RNG are injectable so tests are deterministic.
"""

from __future__ import annotations

import random
from datetime import datetime
from typing import Callable

from .services.alarm import AlarmService
from .services.indicator import IndicatorService
from .services.lights import LightService
from .services.music import MusicService
from .services.phone import PhoneService
from .services.reminders import ReminderService
from .services.thermostat import ThermostatService
from .services.timer import TimerService
from .weather import StubWeatherProvider, WeatherProvider

Now = Callable[[], datetime]


class AppState:
    def __init__(
        self,
        now: Now | None = None,
        rng: random.Random | None = None,
        weather: WeatherProvider | None = None,
    ) -> None:
        self.now: Now = now if now is not None else datetime.now
        self.weather: WeatherProvider = (
            weather if weather is not None else StubWeatherProvider()
        )
        self.rng: random.Random = rng if rng is not None else random.Random()
        self.lights = LightService()
        self.music = MusicService()
        self.phone = PhoneService(rng=self.rng, now=self.now)
        self.reminders = ReminderService(now=self.now)
        self.thermostat = ThermostatService()
        self.timer = TimerService()
        self.alarm = AlarmService(now=self.now)
        self.indicator = IndicatorService()

    def snapshot(self) -> dict:
        """The full board state as JSON-serializable dicts."""
        return {
            "lights": self.lights.to_dict(),
            "music": self.music.to_dict(),
            "phone": self.phone.to_dict(),
            "reminders": self.reminders.to_dict(),
            "thermostat": self.thermostat.to_dict(),
            "timer": self.timer.to_dict(),
            "alarm": self.alarm.to_dict(),
            "indicator": self.indicator.to_dict(),
        }

    def tick(self) -> None:
        """One second of app time: advance the time-based services."""
        self.timer.tick()
        self.phone.tick()
        self.alarm.tick()
        self.indicator.tick()
