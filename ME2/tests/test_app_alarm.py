"""Alarm service: set/replace/reset + the lightweight due check
(decision Q12; round-3: DUE persists until reset, replaceable from any
state, no daily repeat)."""

from datetime import datetime, timedelta

from app.services.alarm import AlarmService, AlarmState


class Clock:
    def __init__(self, t: datetime) -> None:
        self.t = t

    def __call__(self) -> datetime:
        return self.t

    def advance(self, seconds: int) -> None:
        self.t += timedelta(seconds=seconds)


def test_default_unset():
    svc = AlarmService(now=lambda: datetime(2026, 9, 29, 12, 0, 0))
    assert svc.state is AlarmState.UNSET
    assert svc.display is None
    assert svc.to_dict()["time"] is None


def test_set_and_display_format():
    clock = Clock(datetime(2026, 9, 29, 12, 0, 0))
    svc = AlarmService(now=clock)
    changed, msg = svc.set("6 AM")
    assert changed
    assert svc.state is AlarmState.SET
    assert svc.time == "6 AM"
    assert svc.display == "06:00 AM"
    assert "06:00 AM" in msg


def test_replace_from_set():
    svc = AlarmService(now=lambda: datetime(2026, 9, 29, 12, 0, 0))
    svc.set("6 AM")
    changed, _msg = svc.set("9 PM")
    assert changed
    assert svc.state is AlarmState.SET
    assert svc.display == "09:00 PM"


def test_replace_from_due_drops_back_to_set():
    clock = Clock(datetime(2026, 9, 29, 8, 0, 0))
    svc = AlarmService(now=clock)
    svc.set("8 AM")
    svc.tick()  # -> DUE at 08:00
    assert svc.state is AlarmState.DUE
    changed, _msg = svc.set("6 AM")
    assert changed
    assert svc.state is AlarmState.SET
    assert svc.display == "06:00 AM"


def test_invalid_time_ignored():
    svc = AlarmService(now=lambda: datetime(2026, 9, 29, 12, 0, 0))
    changed, msg = svc.set("7 AM")
    assert not changed
    assert msg.startswith("Ignored")
    assert svc.state is AlarmState.UNSET


def test_due_on_the_matching_minute():
    clock = Clock(datetime(2026, 9, 29, 7, 59, 59))
    svc = AlarmService(now=clock)
    svc.set("8 AM")
    svc.tick()
    assert svc.state is AlarmState.SET  # 07:59:59 -> not yet
    clock.advance(1)  # 08:00:00
    svc.tick()
    assert svc.state is AlarmState.DUE


def test_due_requires_ampm_match():
    clock = Clock(datetime(2026, 9, 29, 9, 0, 0))  # 9 AM
    svc = AlarmService(now=clock)
    svc.set("9 PM")
    svc.tick()
    assert svc.state is AlarmState.SET  # same 12-hour hour, wrong AM/PM
    clock.advance(12 * 3600)  # 21:00
    svc.tick()
    assert svc.state is AlarmState.DUE


def test_due_only_fires_on_the_zero_minute():
    clock = Clock(datetime(2026, 9, 29, 8, 1, 0))  # 08:01, past the minute
    svc = AlarmService(now=clock)
    svc.set("8 AM")
    svc.tick()
    assert svc.state is AlarmState.SET  # the minute already passed


def test_due_persists_until_reset():
    clock = Clock(datetime(2026, 9, 29, 8, 0, 0))
    svc = AlarmService(now=clock)
    svc.set("8 AM")
    svc.tick()
    assert svc.state is AlarmState.DUE
    clock.advance(30)
    svc.tick()
    assert svc.state is AlarmState.DUE  # no auto-clear, no re-arm


def test_reset_from_set_and_due():
    for target in ("SET", "DUE"):
        clock = Clock(datetime(2026, 9, 29, 8, 0, 0))
        svc = AlarmService(now=clock)
        svc.set("8 AM")
        if target == "DUE":
            svc.tick()
        changed, _msg = svc.reset()
        assert changed
        assert svc.state is AlarmState.UNSET
        assert svc.display is None


def test_reset_while_unset_ignored():
    svc = AlarmService(now=lambda: datetime(2026, 9, 29, 12, 0, 0))
    changed, msg = svc.reset()
    assert not changed
    assert msg.startswith("Ignored")
