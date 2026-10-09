"""Fast, CPU-only tests for `vcm.streaming.policy` -- `WindowObservation`,
`PolicyDecision`, the `ThresholdPolicy` acceptance policy, and the
`ModePeriodPolicy` listening-period policy (against a scripted fake gate).
"""

from __future__ import annotations

import pytest

import numpy as np

from me2_voicegen.common.features import SAMPLE_RATE
from me2_voicegen.vcm.decoder import DecodeResult
from me2_voicegen.vcm.evaluate import RowResult, _accepted
from me2_voicegen.vcm.streaming.gate import GateState
from me2_voicegen.vcm.streaming.policy import (
    ModePeriodPolicy,
    PolicyDecision,
    ThresholdPolicy,
    WindowObservation,
)


def _result(intent, confidence) -> DecodeResult:
    return DecodeResult(
        intent=intent,
        slots={},
        text="whatever",
        confidence=confidence,
        no_match=intent is None,
        out_of_grammar_gap=0.0,
    )


def _obs(intent, confidence, window_index=0, samples_seen=0) -> WindowObservation:
    return WindowObservation(
        window_index=window_index,
        samples_seen=samples_seen,
        result=_result(intent, confidence),
    )


def test_threshold_policy_accepts_above_threshold():
    policy = ThresholdPolicy(threshold=-1.0)
    decision = policy.observe(_obs("CALL", -0.5))
    assert decision.accept is True
    assert isinstance(decision, PolicyDecision)
    assert decision.reason


def test_threshold_policy_accepts_at_threshold():
    policy = ThresholdPolicy(threshold=-1.0)
    decision = policy.observe(_obs("CALL", -1.0))
    assert decision.accept is True


def test_threshold_policy_rejects_below_threshold():
    policy = ThresholdPolicy(threshold=-1.0)
    decision = policy.observe(_obs("CALL", -1.5))
    assert decision.accept is False
    assert "below threshold" in decision.reason


def test_threshold_policy_rejects_when_intent_is_none():
    policy = ThresholdPolicy(threshold=-100.0)
    decision = policy.observe(_obs(None, float("-inf")))
    assert decision.accept is False
    assert "intent is None" in decision.reason


def test_threshold_policy_reset_is_a_noop_and_does_not_error():
    policy = ThresholdPolicy(threshold=-1.0)
    policy.observe(_obs("CALL", -0.5))
    policy.reset()
    # still behaves identically after reset -- it's stateless.
    decision = policy.observe(_obs("CALL", -0.5))
    assert decision.accept is True


def test_threshold_policy_reduces_exactly_to_evaluate_accepted():
    """Behavior-preservation: `ThresholdPolicy.observe(...).accept` must
    agree with `vcm.evaluate._accepted` for every (intent, confidence,
    threshold) combination -- proving the reduction claim, not just
    asserting it in a docstring."""
    cases = [
        (None, float("-inf"), -1.0),
        ("CALL", -0.5, -1.0),
        ("CALL", -1.0, -1.0),
        ("CALL", -1.5, -1.0),
        ("ALARM", 0.0, 0.0),
        ("ALARM", -0.0001, 0.0),
    ]
    for intent, confidence, threshold in cases:
        policy = ThresholdPolicy(threshold=threshold)
        decision = policy.observe(_obs(intent, confidence))

        row = RowResult(
            index=0,
            bucket="target_commands",
            label="whatever",
            text="whatever",
            intent=intent,
            confidence=None if intent is None else confidence,
        )
        expected = _accepted(row, threshold)
        assert decision.accept == expected, (intent, confidence, threshold)


# ---------------------------------------------------------------------------
# ModePeriodPolicy -- scripted fake gate (file-local by design; no conftest)

_STRIDE = 4000  # 250 ms at SAMPLE_RATE=16000, the default streaming stride


class _ScriptedGate:
    """Scripted `ListeningGate` with `SpacebarGate`'s poll ordering: the
    period auto-closes first, then a queued space press opens (or
    re-opens) it at the polled `samples_seen`. `press()` queues an
    operator keypress consumed at the next poll, mirroring
    `SpacebarGate`'s poll-boundary drain (a double press within one
    stride yields the same `open_at_samples`)."""

    def __init__(self, period_s: float) -> None:
        self._period = int(period_s * SAMPLE_RATE)
        self._open_at = None
        self._pending_presses = 0
        self.polls = []  # (samples_seen, waveform) per poll, for assertions

    def press(self) -> None:
        self._pending_presses += 1

    def poll(self, samples_seen, window=None):
        self.polls.append((samples_seen, window))
        if self._open_at is not None and samples_seen >= self._open_at + self._period:
            self._open_at = None
        if self._pending_presses:
            self._pending_presses = 0
            self._open_at = samples_seen
        return GateState(is_open=self._open_at is not None, open_at_samples=self._open_at)

    def close(self) -> None:
        self._open_at = None


def _obs_slots(intent, confidence, slots, window_index=0, samples_seen=0) -> WindowObservation:
    return WindowObservation(
        window_index=window_index,
        samples_seen=samples_seen,
        result=DecodeResult(
            intent=intent,
            slots=slots,
            text="whatever",
            confidence=confidence,
            no_match=intent is None,
            out_of_grammar_gap=0.0,
        ),
    )


def _period_setup(period_s=1.0, threshold=-1.0):
    gate = _ScriptedGate(period_s)
    policy = ModePeriodPolicy(threshold, gate=gate, period_s=period_s)
    return policy, gate


def _run_period(decodes, *, period_s=1.0, threshold=-1.0):
    """Open a period at samples_seen=0, feed one decode per stride, then the
    flush observation; return (flush_decision, policy, gate). `decodes` are
    (intent, confidence[, slots]) and must fill exactly one period."""
    policy, gate = _period_setup(period_s=period_s, threshold=threshold)
    gate.press()
    n = int(period_s * SAMPLE_RATE) // _STRIDE
    assert len(decodes) == n
    for i, spec in enumerate(decodes):
        slots = spec[2] if len(spec) > 2 else {}
        policy.observe(_obs_slots(spec[0], spec[1], slots, window_index=i, samples_seen=i * _STRIDE))
    flush = policy.observe(_obs(None, float("-inf"), window_index=n, samples_seen=n * _STRIDE))
    return flush, policy, gate


def test_backlog_ac1_ten_identical_decodes_collapse_to_single_accept_at_flush():
    """Backlog AC1 (lingering failure mode): 10 identical correct decodes
    across one open period yield exactly one accept, at the flush, with
    the mode's intent/slots/text preserved."""
    policy, gate = _period_setup(period_s=2.5)  # 40000 samples -> 10 strides
    gate.press()
    decisions = [
        policy.observe(_obs("TIME", -0.2, window_index=i, samples_seen=i * _STRIDE))
        for i in range(10)
    ]
    assert all(not d.accept for d in decisions)
    assert all(d.result is None for d in decisions)
    assert decisions[2].reason == "collecting (3 obs, running mode 'TIME' 3/3)"

    flush = policy.observe(_obs(None, float("-inf"), window_index=10, samples_seen=40000))
    assert flush.accept is True
    assert flush.result is not None
    assert flush.result.intent == "TIME"
    assert flush.result.slots == {}
    assert flush.result.text == "whatever"
    assert flush.result.confidence == pytest.approx(-0.2)  # mean of the ten confidences
    assert ">= threshold" in flush.reason
    assert "10 obs" in flush.reason

    # exactly one accept overall; the gate has closed again afterwards.
    assert sum(d.accept for d in decisions + [flush]) == 1
    closed = policy.observe(_obs("TIME", -0.2, window_index=11, samples_seen=44000))
    assert closed.accept is False
    assert closed.reason == "gate closed (not in a listening period)"


def test_backlog_ac2_intermittent_noise_in_mostly_none_period_never_triggers():
    """Backlog AC2 (ambient failure mode): intermittent low-confidence
    noise in a mostly-`None` period yields zero accepts -- the `None`
    class wins the mode."""
    policy, gate = _period_setup(period_s=2.5, threshold=-1.0)
    gate.press()
    decisions = []
    for i in range(10):
        if i in (3, 7):
            decisions.append(policy.observe(_obs("TIME", -2.0, window_index=i, samples_seen=i * _STRIDE)))
        else:
            decisions.append(policy.observe(_obs(None, float("-inf"), window_index=i, samples_seen=i * _STRIDE)))
    flush = policy.observe(_obs(None, float("-inf"), window_index=10, samples_seen=40000))
    assert all(not d.accept for d in decisions + [flush])
    assert flush.reason == "mode_period: silence dominated the period (None 8/10)"


def test_silence_dominated_all_none_period_rejects_with_none_over_none_reason():
    policy, gate = _period_setup()  # period 16000 -> 4 strides, flush at 16000
    gate.press()
    for i in range(4):
        policy.observe(_obs(None, float("-inf"), window_index=i, samples_seen=i * _STRIDE))
    flush = policy.observe(_obs(None, float("-inf"), window_index=4, samples_seen=16000))
    assert flush.accept is False
    assert flush.result is None
    assert flush.reason == "mode_period: silence dominated the period (None 4/4)"


def test_mode_tie_break_count_beats_confidence_sum():
    flush, _, _ = _run_period([("A", 0.1), ("B", 0.9), ("A", 0.1), ("A", 0.1)])
    assert flush.accept is True
    assert flush.result.intent == "A"  # 3 votes beat B's higher sum
    assert flush.result.confidence == pytest.approx(0.1)


def test_mode_tie_break_confidence_sum_beats_earlier_first_seen():
    # B is seen first; A has the same count and the higher confidence sum.
    flush, _, _ = _run_period([("B", 0.9), ("A", 0.6), ("B", 0.2), ("A", 0.6)])
    assert flush.accept is True
    assert flush.result.intent == "A"
    assert flush.result.confidence == pytest.approx(0.6)


def test_mode_tie_break_earliest_first_seen_when_counts_and_sums_equal():
    # A and B tie on count (2) and sum (1.0); A is seen first.
    flush, _, _ = _run_period([("A", 0.5), ("B", 0.3), ("A", 0.5), ("B", 0.7)])
    assert flush.accept is True
    assert flush.result.intent == "A"
    assert flush.result.confidence == pytest.approx(0.5)


def test_flush_rejects_when_winning_class_mean_below_threshold():
    flush, _, _ = _run_period([("TIME", -2.0)] * 4, threshold=-1.0)
    assert flush.accept is False
    assert flush.result is not None  # the mode result still rides along for logging
    assert flush.result.intent == "TIME"
    assert flush.result.confidence == -2.0
    assert "below threshold" in flush.reason
    assert "-2.0" in flush.reason and "-1.0" in flush.reason and "4 obs" in flush.reason


def test_confident_decode_while_gate_closed_rejects_and_collects_nothing():
    policy, gate = _period_setup()  # the press lands at the stride-4000 poll
    early = policy.observe(_obs("TIME", -0.2, window_index=0, samples_seen=0))
    assert early.accept is False
    assert early.result is None
    assert early.reason == "gate closed (not in a listening period)"

    gate.press()  # period [4000, 20000): the earlier decode must not leak in
    policy.observe(_obs(None, float("-inf"), window_index=1, samples_seen=4000))
    policy.observe(_obs(None, float("-inf"), window_index=2, samples_seen=8000))
    policy.observe(_obs(None, float("-inf"), window_index=3, samples_seen=12000))
    flush = policy.observe(_obs(None, float("-inf"), window_index=4, samples_seen=20000))
    assert flush.accept is False
    assert flush.reason == "mode_period: silence dominated the period (None 3/3)"


def test_repress_discards_in_flight_and_restarts_at_new_position():
    policy, gate = _period_setup()  # period 16000
    gate.press()
    policy.observe(_obs("TIME", -0.2, window_index=0, samples_seen=0))
    policy.observe(_obs("TIME", -0.2, window_index=1, samples_seen=4000))

    # re-press at the stride-8000 poll: in-flight TIME obs are discarded and
    # a fresh period opens at 8000 (edge 3) -- without the discard the reason
    # below would read "(3 obs, running mode 'TIME' 2/3)".
    gate.press()
    d = policy.observe(_obs(None, float("-inf"), window_index=2, samples_seen=8000))
    assert d.accept is False
    assert d.reason == "collecting (1 obs, running mode None 1/1)"
    policy.observe(_obs(None, float("-inf"), window_index=3, samples_seen=12000))
    policy.observe(_obs(None, float("-inf"), window_index=4, samples_seen=16000))
    policy.observe(_obs(None, float("-inf"), window_index=5, samples_seen=20000))
    flush = policy.observe(_obs(None, float("-inf"), window_index=6, samples_seen=24000))
    assert flush.accept is False
    # (None 4/4) -- not (None 4/6): the 2 in-flight TIME obs are gone.
    assert flush.reason == "mode_period: silence dominated the period (None 4/4)"


def test_reset_mid_period_clears_in_flight_collection():
    policy, gate = _period_setup()
    gate.press()
    policy.observe(_obs("TIME", -0.2, window_index=0, samples_seen=0))
    policy.observe(_obs("TIME", -0.2, window_index=1, samples_seen=4000))

    policy.reset()

    # the gate is still open (reset clears the policy's period, not the
    # gate's): the next observation re-enters collection from scratch.
    d = policy.observe(_obs(None, float("-inf"), window_index=2, samples_seen=8000))
    assert d.accept is False
    assert d.reason == "collecting (1 obs, running mode None 1/1)"
    policy.observe(_obs(None, float("-inf"), window_index=3, samples_seen=12000))
    flush = policy.observe(_obs(None, float("-inf"), window_index=4, samples_seen=16000))
    # without the reset the 2 TIME obs would tie the 2 None obs and win
    # (finite sum beats -inf), producing an accept instead.
    assert flush.accept is False
    assert flush.reason == "mode_period: silence dominated the period (None 2/2)"


def test_negative_inf_confidence_in_winning_class_rejected_at_finite_threshold():
    # SPEC Assumption 4, defensive: -inf confidences inside the winning
    # class make its mean -inf, below any finite threshold.
    flush, _, _ = _run_period(
        [
            ("TIME", float("-inf")),
            ("TIME", float("-inf")),
            (None, float("-inf")),
            ("TIME", float("-inf")),
        ],
        threshold=-1.0,
    )
    assert flush.accept is False
    assert flush.result is not None
    assert flush.result.intent == "TIME"
    assert flush.result.confidence == float("-inf")
    assert "below threshold" in flush.reason
    assert flush.reason.startswith("mode_period: intent='TIME'")


def test_flush_with_simultaneous_reopen_returns_flush_and_seeds_new_period():
    # A space press landing on the exact stride a period ends (the gate
    # auto-closes, then the press re-opens at the same samples_seen): the
    # flush decision is returned for that observation, AND the same
    # observation seeds the new period (settled at plan confirmation
    # 2026-09-20 -- the SPEC observe algorithm is binding).
    policy, gate = _period_setup()  # period 16000 -> flush at 16000
    gate.press()
    for i in range(4):
        policy.observe(_obs("TIME", -0.2, window_index=i, samples_seen=i * _STRIDE))

    gate.press()  # the operator re-presses exactly at the flush stride
    flush = policy.observe(_obs(None, float("-inf"), window_index=4, samples_seen=16000))

    # (a) the decision is the first period's mode:
    assert flush.accept is True
    assert flush.result.intent == "TIME"
    assert flush.result.confidence == pytest.approx(-0.2)

    # (b) the same observation seeded the new period [16000, 32000):
    d = policy.observe(_obs(None, float("-inf"), window_index=5, samples_seen=20000))
    assert d.accept is False
    assert d.reason == "collecting (2 obs, running mode None 2/2)"
    policy.observe(_obs(None, float("-inf"), window_index=6, samples_seen=24000))
    policy.observe(_obs(None, float("-inf"), window_index=7, samples_seen=28000))
    flush2 = policy.observe(_obs(None, float("-inf"), window_index=8, samples_seen=32000))
    assert flush2.accept is False
    # (None 4/4) -- the 4th obs is the flush observation itself; had it not
    # seeded the new period this would read (None 3/3).
    assert flush2.reason == "mode_period: silence dominated the period (None 4/4)"


def test_same_intent_with_different_slots_counts_as_distinct_classes():
    # the mode key is (intent, sorted-slots): same intent, different slots,
    # is a different class -- and the winner result carries its slots.
    flush, _, _ = _run_period(
        [
            ("TIME", -0.4, {"hour": "8"}),
            ("TIME", -0.4, {"hour": "8"}),
            ("TIME", -0.1, {"hour": "9"}),
            ("CALL", -0.2),
        ],
        threshold=-1.0,
    )
    assert flush.accept is True
    assert flush.result.intent == "TIME"
    assert flush.result.slots == {"hour": "8"}
    assert flush.result.confidence == pytest.approx(-0.4)


def test_double_press_within_one_stride_opens_one_period():
    # `SpacebarGate` drains all pending bytes at a poll boundary, so a
    # double press within one stride yields the same `open_at_samples`:
    # the policy treats it as the same open period (no re-press discard).
    policy, gate = _period_setup()
    gate.press()
    gate.press()
    d0 = policy.observe(_obs("TIME", -0.2, window_index=0, samples_seen=0))
    d1 = policy.observe(_obs("TIME", -0.2, window_index=1, samples_seen=4000))
    assert d0.reason == "collecting (1 obs, running mode 'TIME' 1/1)"
    assert d1.reason == "collecting (2 obs, running mode 'TIME' 2/2)"
    policy.observe(_obs("TIME", -0.2, window_index=2, samples_seen=8000))
    policy.observe(_obs("TIME", -0.2, window_index=3, samples_seen=12000))
    flush = policy.observe(_obs(None, float("-inf"), window_index=4, samples_seen=16000))
    assert flush.accept is True
    assert flush.result.confidence == pytest.approx(-0.2)


def test_waveform_is_forwarded_to_the_gate_on_every_poll():
    policy, gate = _period_setup()
    wave = np.zeros(160, dtype=np.float32)
    obs = WindowObservation(
        window_index=0, samples_seen=0, result=_result(None, float("-inf")), waveform=wave
    )
    policy.observe(obs)
    assert gate.polls[0][1] is wave

    gate.press()
    obs2 = WindowObservation(
        window_index=1, samples_seen=4000, result=_result(None, float("-inf"))
    )
    policy.observe(obs2)
    assert gate.polls[1][1] is None


# ---------------------------------------------------------------------------
# ModePeriodPolicy -- optional on_period_event sink (--log-periods digest)
# ---------------------------------------------------------------------------


def _events_setup(period_s=1.0, threshold=-1.0):
    gate = _ScriptedGate(period_s)
    events: list = []
    policy = ModePeriodPolicy(
        threshold,
        gate=gate,
        period_s=period_s,
        on_period_event=lambda event, samples_seen, decision: events.append(
            (event, samples_seen, decision)
        ),
    )
    return policy, gate, events


def test_period_event_sink_reports_open_and_closed_with_flush_decision():
    policy, gate, events = _events_setup()
    gate.press()
    for i in range(4):
        policy.observe(
            _obs_slots("TIME", -0.2, {}, window_index=i, samples_seen=i * _STRIDE)
        )
    flush = policy.observe(
        _obs(None, float("-inf"), window_index=4, samples_seen=4 * _STRIDE)
    )

    assert [event for event, _, _ in events] == ["open", "closed"]
    assert events[0][1] == 0 and events[0][2] is None
    assert events[1][1] == 4 * _STRIDE
    assert events[1][2] is flush  # the same decision object the runner sees
    assert flush.accept is True


def test_period_event_sink_silent_while_collecting_and_when_gate_closed():
    policy, gate, events = _events_setup()

    # no press: confident decodes while the gate is closed emit nothing
    policy.observe(_obs("TIME", -0.2, window_index=0, samples_seen=0))
    assert events == []

    gate.press()
    for i in range(4):
        policy.observe(
            _obs_slots("TIME", -0.2, {}, window_index=i, samples_seen=i * _STRIDE)
        )
    policy.observe(_obs(None, float("-inf"), window_index=4, samples_seen=4 * _STRIDE))
    assert [event for event, _, _ in events] == ["open", "closed"]

    # after the period: closed-gate observations add no events
    policy.observe(_obs("TIME", -0.2, window_index=5, samples_seen=5 * _STRIDE))
    assert [event for event, _, _ in events] == ["open", "closed"]


def test_period_event_sink_reports_reopened_on_mid_period_repress():
    policy, gate, events = _events_setup(period_s=2.0)
    gate.press()
    for i in range(3):
        policy.observe(
            _obs_slots("TIME", -0.2, {}, window_index=i, samples_seen=i * _STRIDE)
        )
    gate.press()  # mid-period re-press
    for i in range(3, 11):
        policy.observe(
            _obs_slots("TIME", -0.2, {}, window_index=i, samples_seen=i * _STRIDE)
        )
    flush = policy.observe(
        _obs(None, float("-inf"), window_index=11, samples_seen=11 * _STRIDE)
    )

    assert [event for event, _, _ in events] == ["open", "reopened", "closed"]
    assert events[0][1] == 0
    assert events[1][1] == 3 * _STRIDE  # the re-press stride
    assert events[2][1] == 11 * _STRIDE
    # the pre-press observations were discarded: 8 collected, not 11
    assert "over 8 obs" in flush.reason


def test_period_event_sink_flush_and_simultaneous_reopen_reports_closed_then_open():
    policy, gate, events = _events_setup()
    gate.press()
    for i in range(4):
        policy.observe(
            _obs_slots("TIME", -0.2, {}, window_index=i, samples_seen=i * _STRIDE)
        )
    gate.press()  # lands exactly on the flush stride
    flush = policy.observe(
        _obs(None, float("-inf"), window_index=4, samples_seen=4 * _STRIDE)
    )

    assert [event for event, _, _ in events] == ["open", "closed", "open"]
    assert events[1][2] is flush
    # the new period is collecting: the next observation reports 2 obs
    next_decision = policy.observe(
        _obs_slots("TIME", -0.2, {}, window_index=5, samples_seen=5 * _STRIDE)
    )
    assert next_decision.reason.startswith("collecting (2 obs")


def test_period_event_sink_defaults_to_none():
    policy, _gate = _period_setup()
    assert policy._on_period_event is None
