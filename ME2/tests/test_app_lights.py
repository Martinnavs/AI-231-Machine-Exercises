"""Lights service: OFF/20/60/100 brightness machine + orthogonal color,
patterned on the reference LightStateMachine (rescaled, decision Q3)."""

from app.services.lights import LightEvent, LightService, LightState


def test_defaults_off_white():
    svc = LightService()
    assert svc.state is LightState.OFF
    assert svc.color == "white"
    assert svc.brightness == 0


def test_power_on_defaults_to_20_never_resumes():
    svc = LightService()
    svc.state = LightState.ON_100  # pretend it was at 100 before a cycle
    svc.dispatch(LightEvent.POWER_OFF)
    changed, _msg = svc.dispatch(LightEvent.POWER_ON)
    assert changed
    assert svc.state is LightState.ON_20


def test_power_off_from_each_on_state():
    for on in (LightState.ON_20, LightState.ON_60, LightState.ON_100):
        svc = LightService()
        svc.state = on
        changed, _msg = svc.dispatch(LightEvent.POWER_OFF)
        assert changed
        assert svc.state is LightState.OFF


def test_increase_steps_and_caps_at_100():
    svc = LightService()
    svc.dispatch(LightEvent.POWER_ON)
    assert svc.dispatch(LightEvent.INCREASE)[0]
    assert svc.state is LightState.ON_60
    assert svc.dispatch(LightEvent.INCREASE)[0]
    assert svc.state is LightState.ON_100
    changed, _msg = svc.dispatch(LightEvent.INCREASE)  # capped
    assert changed
    assert svc.state is LightState.ON_100


def test_decrease_steps_and_caps_at_20():
    svc = LightService()
    svc.state = LightState.ON_100
    assert svc.dispatch(LightEvent.DECREASE)[0]
    assert svc.state is LightState.ON_60
    assert svc.dispatch(LightEvent.DECREASE)[0]
    assert svc.state is LightState.ON_20
    changed, _msg = svc.dispatch(LightEvent.DECREASE)  # capped
    assert changed
    assert svc.state is LightState.ON_20


def test_direct_set_from_any_on_state():
    cases = (
        (LightEvent.SET_20, LightState.ON_20),
        (LightEvent.SET_60, LightState.ON_60),
        (LightEvent.SET_100, LightState.ON_100),
    )
    for event, target in cases:
        for on in (LightState.ON_20, LightState.ON_60, LightState.ON_100):
            svc = LightService()
            svc.state = on
            changed, _msg = svc.dispatch(event)
            assert changed
            assert svc.state is target


def test_set_brightness_from_canonical_percent():
    svc = LightService()
    svc.dispatch(LightEvent.POWER_ON)
    assert svc.set_brightness("60 percent")[0]
    assert svc.brightness == 60
    assert svc.set_brightness(100)[0]
    assert svc.brightness == 100


def test_brightness_while_off_is_ignored():
    svc = LightService()
    changed, msg = svc.set_brightness(60)
    assert not changed
    assert msg.startswith("Ignored")
    assert svc.state is LightState.OFF


def test_brightness_outside_the_scale_is_ignored():
    svc = LightService()
    svc.dispatch(LightEvent.POWER_ON)
    changed, msg = svc.set_brightness(50)
    assert not changed
    assert msg.startswith("Ignored")
    assert svc.brightness == 20


def test_color_orthogonal_set_on_and_off_and_persists_across_cycles():
    svc = LightService()
    assert svc.set_color("red")[0]
    svc.dispatch(LightEvent.POWER_ON)
    svc.dispatch(LightEvent.POWER_OFF)
    assert svc.color == "red"
    assert svc.set_color("blue")[0]
    assert svc.color == "blue"


def test_unknown_color_ignored():
    svc = LightService()
    changed, msg = svc.set_color("purple")
    assert not changed
    assert msg.startswith("Ignored")
    assert svc.color == "white"


def test_every_unmapped_state_event_pair_is_ignored():
    for st in LightState:
        for ev in LightEvent:
            svc = LightService()
            svc.state = st
            if (st, ev) in LightService.TRANSITIONS:
                continue
            changed, msg = svc.dispatch(ev)
            assert not changed, (st, ev)
            assert msg.startswith("Ignored")
            assert svc.state is st
