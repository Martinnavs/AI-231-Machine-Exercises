"""Timer service: 10/30/60-second presets, MM:SS countdown, done-stays
done (no auto-reset), re-issue while running ignored (decision Q7)."""

from app.services.timer import TimerEvent, TimerService, TimerState


def test_defaults_idle_zero():
    svc = TimerService()
    assert svc.state is TimerState.IDLE
    assert svc.remaining_s == 0
    assert svc.display == "00:00"


def test_start_each_preset():
    cases = ((10, "00:10"), (30, "00:30"), (60, "01:00"))
    for seconds, display in cases:
        svc = TimerService()
        changed, _msg = svc.start(seconds)
        assert changed
        assert svc.state is TimerState.RUNNING
        assert svc.duration_s == seconds
        assert svc.remaining_s == seconds
        assert svc.display == display


def test_start_rejects_non_preset_durations():
    for bad in (5, 45, 90, None):
        svc = TimerService()
        changed, msg = svc.start(bad)
        assert not changed, bad
        assert msg.startswith("Ignored")
        assert svc.state is TimerState.IDLE


def test_countdown_reaches_done_at_zero():
    svc = TimerService()
    svc.start(10)
    for _ in range(9):
        svc.tick()
    assert svc.state is TimerState.RUNNING
    assert svc.remaining_s == 1
    svc.tick()
    assert svc.state is TimerState.DONE
    assert svc.remaining_s == 0


def test_done_stays_done_no_auto_reset():
    svc = TimerService()
    svc.start(10)
    for _ in range(15):
        svc.tick()
    assert svc.state is TimerState.DONE
    assert svc.remaining_s == 0


def test_restart_from_done():
    svc = TimerService()
    svc.start(10)
    for _ in range(10):
        svc.tick()
    assert svc.state is TimerState.DONE
    changed, _msg = svc.start(30)
    assert changed
    assert svc.state is TimerState.RUNNING
    assert svc.remaining_s == 30


def test_reissue_while_running_is_ignored():
    svc = TimerService()
    svc.start(60)
    changed, msg = svc.start(10)
    assert not changed
    assert msg.startswith("Ignored")
    assert svc.duration_s == 60
    assert svc.remaining_s == 60


def test_reset_from_running_and_done():
    for setup in ("running", "done"):
        svc = TimerService()
        svc.start(10)
        if setup == "done":
            for _ in range(10):
                svc.tick()
        changed, _msg = svc.reset()
        assert changed
        assert svc.state is TimerState.IDLE
        assert svc.remaining_s == 0
        assert svc.duration_s == 0


def test_reset_while_idle_ignored():
    svc = TimerService()
    changed, msg = svc.reset()
    assert not changed
    assert msg.startswith("Ignored")


def test_tick_does_nothing_when_idle():
    svc = TimerService()
    svc.tick()
    assert svc.state is TimerState.IDLE
    assert svc.remaining_s == 0


def test_every_unmapped_state_event_pair_is_ignored():
    for st in TimerState:
        for ev in (
            TimerEvent.START_10,
            TimerEvent.START_30,
            TimerEvent.START_60,
            TimerEvent.RESET,
        ):
            svc = TimerService()
            svc.state = st
            if (st, ev) in TimerService.TRANSITIONS:
                continue
            changed, msg = svc.dispatch(ev)
            assert not changed, (st, ev)
            assert msg.startswith("Ignored")
            assert svc.state is st
