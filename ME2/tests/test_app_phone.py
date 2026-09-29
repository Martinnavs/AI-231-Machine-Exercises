"""Phone service: random-call simulation with elapsed ticker, UI-only
hang-up, append-only message log (decision Q10)."""

import random
from datetime import datetime

from app.services.phone import (
    CANNED_MESSAGES,
    CONTACTS,
    PhoneEvent,
    PhoneService,
    PhoneState,
)


def _now() -> datetime:
    return datetime(2026, 9, 29, 9, 5, 3)


def test_defaults_idle_empty():
    svc = PhoneService(rng=random.Random(0), now=_now)
    assert svc.state is PhoneState.IDLE
    assert svc.person is None
    assert svc.elapsed_s == 0
    assert svc.messages == []


def test_call_picks_a_contact_and_enters_in_call():
    svc = PhoneService(rng=random.Random(0), now=_now)
    changed, msg = svc.call()
    assert changed
    assert svc.state is PhoneState.IN_CALL
    assert svc.person in [{"name": n, "number": num} for n, num in CONTACTS]
    assert svc.elapsed_s == 0
    assert svc.person["name"] in msg


def test_call_is_deterministic_under_a_seed():
    a = PhoneService(rng=random.Random(7), now=_now).call()
    b = PhoneService(rng=random.Random(7), now=_now).call()
    assert a[1].split(" to ")[1] == b[1].split(" to ")[1]


def test_reissuing_call_replaces_and_restarts_ticker():
    svc = PhoneService(rng=random.Random(1), now=_now)
    first = svc.call()
    svc.elapsed_s = 45
    second = svc.call()
    assert second[0]
    assert svc.state is PhoneState.IN_CALL
    assert svc.elapsed_s == 0
    assert second[1].startswith("new call to")
    assert first[1].startswith("call to")


def test_hang_up_returns_to_idle_and_clears():
    svc = PhoneService(rng=random.Random(0), now=_now)
    svc.call()
    changed, msg = svc.hang_up()
    assert changed
    assert svc.state is PhoneState.IDLE
    assert svc.person is None
    assert svc.elapsed_s == 0


def test_hang_up_while_idle_ignored():
    svc = PhoneService(rng=random.Random(0), now=_now)
    changed, msg = svc.hang_up()
    assert not changed
    assert msg.startswith("Ignored")


def test_message_appends_a_timestamped_entry():
    svc = PhoneService(rng=random.Random(1), now=_now)
    changed, _msg = svc.message()
    assert changed
    assert len(svc.messages) == 1
    item = svc.messages[0]
    assert item["at"] == "09:05:03"
    assert item["name"] in [name for name, _n in CONTACTS]
    assert item["text"] in CANNED_MESSAGES


def test_message_is_valid_while_in_call_and_keeps_call_state():
    svc = PhoneService(rng=random.Random(2), now=_now)
    svc.call()
    assert svc.message()[0]
    assert svc.state is PhoneState.IN_CALL
    assert svc.elapsed_s == 0  # a message does not restart the call


def test_tick_advances_elapsed_only_in_call():
    svc = PhoneService(rng=random.Random(0), now=_now)
    svc.tick()
    assert svc.elapsed_s == 0  # idle: no tick
    svc.call()
    svc.tick()
    svc.tick()
    svc.tick()
    assert svc.elapsed_s == 3
    svc.hang_up()
    svc.tick()
    assert svc.elapsed_s == 0


def test_to_dict_lists_messages_newest_first():
    svc = PhoneService(rng=random.Random(3), now=_now)
    svc.message()
    svc.message()
    svc.message()
    listed = svc.to_dict()["messages"]
    assert [m["text"] for m in listed] == [m["text"] for m in reversed(svc.messages)]
    assert len(listed) == 3


def test_every_unmapped_state_event_pair_is_ignored():
    for st in PhoneState:
        for ev in (PhoneEvent.CALL, PhoneEvent.HANG_UP):
            svc = PhoneService(rng=random.Random(0), now=_now)
            svc.state = st
            if (st, ev) in PhoneService.TRANSITIONS:
                continue
            changed, msg = svc.dispatch(ev)
            assert not changed, (st, ev)
            assert msg.startswith("Ignored")
            assert svc.state is st
