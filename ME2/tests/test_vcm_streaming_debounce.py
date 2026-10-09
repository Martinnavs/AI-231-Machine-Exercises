"""Fast, CPU-only tests for `vcm.streaming.debounce.Debouncer`.

Not listed under ticket 01's `## Files`, which named only
`test_vcm_streaming_buffer.py` and `test_vcm_streaming_policy.py` -- added
because the ticket's own Acceptance Criteria explicitly requires
"`Debouncer` unit-tested for arm/suppress/re-arm and the `refractory=0`
pass-through case; `reset()` tested", and no other listed file is a
sensible home for it. See ticket 01's Execution Log for this note.
"""

from __future__ import annotations

from me2_voicegen.vcm.streaming.debounce import Debouncer


def test_debouncer_starts_armed():
    d = Debouncer(refractory_samples=100)
    assert d.armed is True
    assert d.cooldown_samples_remaining == 0


def test_debouncer_emits_on_first_match_then_suppresses_within_cooldown():
    d = Debouncer(refractory_samples=30)

    assert d.gate(matched=True) is True
    assert d.suppressed_count == 0
    assert d.armed is False

    d.tick(10)
    assert d.gate(matched=True) is False
    assert d.suppressed_count == 1

    d.tick(10)
    assert d.gate(matched=True) is False
    assert d.suppressed_count == 2


def test_debouncer_rearms_once_cooldown_fully_elapsed():
    d = Debouncer(refractory_samples=30)
    assert d.gate(matched=True) is True

    d.tick(10)
    d.tick(10)
    d.tick(10)  # 30 samples of stride ticked -> cooldown fully elapsed
    assert d.armed is True
    assert d.gate(matched=True) is True
    assert d.suppressed_count == 0


def test_debouncer_non_match_is_noop_and_does_not_tick_itself():
    d = Debouncer(refractory_samples=30)
    assert d.gate(matched=False) is False
    assert d.suppressed_count == 0
    assert d.armed is True


def test_debouncer_refractory_zero_is_pass_through():
    d = Debouncer(refractory_samples=0)
    for _ in range(5):
        d.tick(10)
        assert d.gate(matched=True) is True
    assert d.suppressed_count == 0


def test_debouncer_reset_clears_cooldown_and_suppressed_count():
    d = Debouncer(refractory_samples=30)
    d.gate(matched=True)
    d.tick(5)
    d.gate(matched=True)  # suppressed
    assert d.suppressed_count == 1
    assert d.cooldown_samples_remaining > 0

    d.reset()
    assert d.suppressed_count == 0
    assert d.cooldown_samples_remaining == 0
    assert d.armed is True
