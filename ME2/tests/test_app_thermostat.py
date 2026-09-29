"""Thermostat service: discrete 18/22/26 set points, default 22
(decision Q4)."""

from app.services.thermostat import ThermoEvent, ThermoState, ThermostatService


def test_default_22():
    svc = ThermostatService()
    assert svc.state is ThermoState.T22
    assert svc.to_dict() == {"degrees": 22}


def test_every_set_point_reachable_from_every_state():
    cases = (
        (ThermoEvent.SET_18, ThermoState.T18, 18),
        (ThermoEvent.SET_22, ThermoState.T22, 22),
        (ThermoEvent.SET_26, ThermoState.T26, 26),
    )
    for _event, target, target_degrees in cases:
        for st in ThermoState:
            svc = ThermostatService()
            svc.state = st
            changed, _msg = svc.dispatch(_event)
            assert changed
            assert svc.state is target
            assert svc.degrees == target_degrees


def test_set_from_canonical_grammar_string():
    svc = ThermostatService()
    assert svc.set("18 degrees")[0]
    assert svc.degrees == 18
    assert svc.set("26 degrees")[0]
    assert svc.degrees == 26


def test_set_rejects_values_outside_18_22_26():
    for bad in (17, 20, 25, 30, None):
        svc = ThermostatService()
        changed, msg = svc.set(bad)
        assert not changed, bad
        assert msg.startswith("Ignored")
        assert svc.degrees == 22  # unchanged


def test_set_from_bare_int():
    svc = ThermostatService()
    assert svc.set(26)[0]
    assert svc.degrees == 26
