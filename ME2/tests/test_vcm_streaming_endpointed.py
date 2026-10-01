"""Tests for `EndpointedPeriodPolicy` (wake word + sliding window, fire on the
first confident, stable, ended command). Fake gates and synthetic one-hot
posteriors only; no checkpoint, ONNX file or microphone."""

from __future__ import annotations

import io
from typing import Optional

import numpy as np
import pytest

from me2_voicegen.common.features import SAMPLE_RATE
from me2_voicegen.vcm import alphabet
from me2_voicegen.vcm.decoder import DecodeResult
from me2_voicegen.vcm.optionb import OPTIONB_GRAMMAR
from me2_voicegen.vcm.streaming.config import resolve_policy
from me2_voicegen.vcm.streaming.gate import GateState
from me2_voicegen.vcm.streaming.policy import EndpointedPeriodPolicy, WindowObservation
from me2_voicegen.vcm.streaming.runner import StreamingRunner

STRIDE = int(0.25 * SAMPLE_RATE)
HOP = 160  # one 10 ms posterior frame per hop for these stride-1 fakes


class _Gate:
    """Open from `open_at`; with `rearm`, reports a later open time on every
    poll (what the wake-word gate does while the wake word is still in its
    trailing window)."""

    def __init__(self, open_at: Optional[int], rearm: bool = False, later_open: Optional[int] = None):
        self.open_at = open_at
        self.rearm = rearm
        self.later_open = later_open

    def poll(self, samples_seen: int, window=None) -> GateState:
        if self.later_open is not None and samples_seen >= self.later_open:
            return GateState(is_open=True, open_at_samples=self.later_open)
        if self.open_at is None or samples_seen < self.open_at:
            return GateState(is_open=False, open_at_samples=None)
        return GateState(is_open=True, open_at_samples=samples_seen if self.rearm else self.open_at)

    def close(self) -> None:
        pass


def _logp(phrase: str, n_frames: int, trailing_blank: int) -> np.ndarray:
    """One-hot posteriors: `phrase` (3 frames per character) placed so that
    exactly `trailing_blank` blank frames end the array."""
    chars: list[int] = []
    for cid in alphabet.encode(phrase):
        chars.extend([cid] * 3)
    ids = [alphabet.BLANK_ID] * n_frames
    end = n_frames - trailing_blank
    start = max(0, end - len(chars))
    ids[start:end] = chars[len(chars) - (end - start):]
    logits = np.zeros((n_frames, alphabet.ALPHABET_SIZE), dtype=np.float32)
    logits[np.arange(n_frames), ids] = 12.0
    return logits - np.log(np.exp(logits).sum(axis=-1, keepdims=True))


def _result(intent: Optional[str], confidence: float = -0.01, slots: Optional[dict] = None) -> DecodeResult:
    return DecodeResult(intent=intent, slots=slots or {}, text="", confidence=confidence,
                        no_match=intent is None, out_of_grammar_gap=0.0)


def _obs(i: int, result: DecodeResult, logp: np.ndarray) -> WindowObservation:
    return WindowObservation(window_index=i, samples_seen=i * STRIDE, result=result,
                             waveform=np.zeros(logp.shape[0] * HOP, dtype=np.float32), logp=logp)


def _policy(gate, **kw) -> EndpointedPeriodPolicy:
    return EndpointedPeriodPolicy(-0.5, gate=gate, period_s=3.0, **kw)


# --- anchor / period lifecycle -------------------------------------------------


def test_anchor_does_not_drift_on_gate_rearm():
    policy = _policy(_Gate(open_at=STRIDE, rearm=True))
    requests = [policy.period_request(n * STRIDE, np.zeros(1)) for n in range(1, 8)]
    starts = {r.start_samples for r in requests if r is not None}
    assert starts == {STRIDE}  # always the first open time, never a re-armed one


def test_min_audio_skips_early_strides_and_window_slides_after_cap():
    policy = _policy(_Gate(open_at=0), window_s=1.0, min_audio_s=0.3)
    assert policy.period_request(STRIDE, np.zeros(1)) is None  # 0.25 s < 0.3 s
    r = policy.period_request(2 * STRIDE, np.zeros(1))
    assert (r.start_samples, r.end_samples) == (0, 2 * STRIDE)
    r = policy.period_request(6 * STRIDE, np.zeros(1))  # 1.5 s in, window capped at 1.0 s
    assert (r.start_samples, r.end_samples) == (2 * STRIDE, 6 * STRIDE)


def test_timeout_closes_silently_and_stale_open_is_not_reused():
    events = []
    policy = _policy(_Gate(open_at=0), on_period_event=lambda e, s, d: events.append((e, d)))
    for n in range(1, 14):  # through 3.25 s
        policy.period_request(n * STRIDE, np.zeros(1))
    assert [e for e, _ in events] == ["open", "closed"]
    assert events[-1][1].accept is False and "timed out" in events[-1][1].reason
    # The gate still reports the old open time: no new period.
    assert policy.period_request(14 * STRIDE, np.zeros(1)) is None
    assert [e for e, _ in events] == ["open", "closed"]


def test_new_period_after_fresh_gate_open():
    policy = _policy(_Gate(open_at=0, later_open=20 * STRIDE))
    for n in range(1, 14):
        policy.period_request(n * STRIDE, np.zeros(1))  # first period times out
    r = None
    for n in range(20, 23):
        r = policy.period_request(n * STRIDE, np.zeros(1)) or r
    assert r is not None and r.start_samples == 20 * STRIDE


# --- accept rule -----------------------------------------------------------------


def test_fires_only_after_stable_and_ended():
    policy = _policy(_Gate(open_at=0))
    ended = _logp("stop", 60, trailing_blank=40)
    assert not policy.observe(_obs(1, _result("STOP"), ended)).accept  # stable 1/2
    assert policy.observe(_obs(2, _result("STOP"), ended)).accept


def test_does_not_fire_while_speech_continues():
    policy = _policy(_Gate(open_at=0))
    talking = _logp("stop", 60, trailing_blank=5)  # 50 ms of blank < 300 ms hold
    for i in range(1, 5):
        decision = policy.observe(_obs(i, _result("STOP"), talking))
        assert not decision.accept
    assert "not ended" in decision.reason


def test_time_then_timer_does_not_fire_time():
    """Growing window over 'timer ten seconds': an early stride decodes TIME,
    later strides decode TIMER. TIME must never be emitted."""
    policy = _policy(_Gate(open_at=0))
    seq = [
        ("TIME", {}, _logp("time", 50, trailing_blank=35)),  # looks ended, but only once
        ("TIMER", {"duration": "10 seconds"}, _logp("timer ten seconds", 90, trailing_blank=2)),
        ("TIMER", {"duration": "10 seconds"}, _logp("timer ten seconds", 110, trailing_blank=35)),
    ]
    fired = [policy.observe(_obs(i, _result(intent, slots=slots), lp)) for i, (intent, slots, lp) in enumerate(seq)]
    assert [d.accept for d in fired] == [False, False, True]
    assert fired[2].reason.startswith("endpointed:")


def test_slot_change_resets_stability():
    policy = _policy(_Gate(open_at=0))
    lp = _logp("brightness twenty percent", 120, trailing_blank=40)
    assert not policy.observe(_obs(1, _result("BRIGHTNESS", slots={"percent": "20"}), lp)).accept
    assert not policy.observe(_obs(2, _result("BRIGHTNESS", slots={"percent": "60"}), lp)).accept
    assert policy.observe(_obs(3, _result("BRIGHTNESS", slots={"percent": "60"}), lp)).accept


def test_low_confidence_and_no_match_reset_streak():
    policy = _policy(_Gate(open_at=0))
    lp = _logp("stop", 60, trailing_blank=40)
    policy.observe(_obs(1, _result("STOP"), lp))
    assert not policy.observe(_obs(2, _result("STOP", confidence=-5.0), lp)).accept
    assert not policy.observe(_obs(3, _result("STOP"), lp)).accept  # streak restarted at 1
    assert not policy.observe(_obs(4, _result(None), lp)).accept


def test_endpoint_uses_frame_duration_from_window_and_logp():
    """A stride-2 model emits one frame per 20 ms: 15 trailing blank frames
    are 300 ms there, but only 150 ms at 10 ms per frame."""
    policy = _policy(_Gate(open_at=0), stable_strides=1)
    lp = _logp("stop", 50, trailing_blank=15)
    stride2 = WindowObservation(0, 0, _result("STOP"), waveform=np.zeros(50 * 320, np.float32), logp=lp)
    stride1 = WindowObservation(0, 0, _result("STOP"), waveform=np.zeros(50 * 160, np.float32), logp=lp)
    assert policy.observe(stride2).accept
    assert not policy.observe(stride1).accept


@pytest.mark.parametrize("kw", [{"stable_strides": 0}, {"blank_floor": 0.0}, {"hold_ms": -1}])
def test_invalid_options_rejected(kw):
    with pytest.raises(ValueError):
        _policy(_Gate(open_at=0), **kw)


def test_resolve_policy_requires_gate_and_passes_options():
    with pytest.raises(SystemExit, match="requires a listening gate"):
        resolve_policy("endpointed", -0.1, gate=None, period_s=3.0)
    policy = resolve_policy("endpointed", -0.1, gate=_Gate(open_at=0), period_s=4.0,
                            endpoint_options={"hold_ms": 200, "stable_strides": 3})
    assert isinstance(policy, EndpointedPeriodPolicy)
    assert policy.stable_strides == 3 and policy.hold_s == pytest.approx(0.2)
    assert policy.period_samples == 4 * SAMPLE_RATE


# --- through the real StreamingRunner ------------------------------------------


class _GrowingStopBackend:
    """Posteriors sized to the requested audio (one frame per 10 ms): 'stop'
    in the first 120 ms, blank afterwards. Records each waveform it gets."""

    def __init__(self) -> None:
        self.lengths: list[int] = []

    def logp_for_waveform(self, waveform: np.ndarray) -> np.ndarray:
        self.lengths.append(waveform.shape[-1])
        n = max(1, waveform.shape[-1] // HOP)
        lp = _logp("stop", n, trailing_blank=0)
        ids = [alphabet.BLANK_ID] * n
        chars = [c for cid in alphabet.encode("stop") for c in [cid] * 3]
        ids[: min(n, len(chars))] = chars[: min(n, len(chars))]
        logits = np.zeros((n, alphabet.ALPHABET_SIZE), dtype=np.float32)
        logits[np.arange(n), ids] = 12.0
        lp = logits - np.log(np.exp(logits).sum(axis=-1, keepdims=True))
        return lp


class _Source:
    is_realtime = False

    def __init__(self, n: int) -> None:
        self.n = n

    def blocks(self):
        for start in range(0, self.n, STRIDE):
            yield np.zeros(min(STRIDE, self.n - start), dtype=np.float32)

    def close(self) -> None:
        pass


def test_runner_emits_once_as_soon_as_command_is_stable_and_ended():
    open_at = 4 * STRIDE  # wake word at 1.0 s
    backend = _GrowingStopBackend()
    policy = _policy(_Gate(open_at=open_at, rearm=True))
    out = io.StringIO()
    runner = StreamingRunner(
        source=_Source(9 * SAMPLE_RATE), backend=backend, grammar=OPTIONB_GRAMMAR, policy=policy,
        window_s=2.5, stride_s=0.25, refractory_s=1.5, beam_width=25, out=out,
    )
    summary = runner.run()
    assert summary.events == 1
    assert runner.triggers[0].intent == "STOP"
    # First decode at anchor + 0.5 s (0.3 s minimum rounds up to two strides),
    # stable on the second: fires at anchor + 0.75 s, then the period closes.
    assert runner.triggers[0].t_seconds == pytest.approx(open_at / SAMPLE_RATE + 0.75)
    assert backend.lengths == [2 * STRIDE, 3 * STRIDE]


def test_same_wake_word_does_not_reopen_after_fast_accept():
    """The wake word stays in the gate's trailing window for a while after a
    command fires; the gate keeps re-arming. No second period until it stops
    and a new detection arrives."""
    policy = _policy(_Gate(open_at=0, rearm=True), stable_strides=1)
    lp = _logp("stop", 60, trailing_blank=40)
    req = policy.period_request(2 * STRIDE, np.zeros(1))
    decision = policy.observe(_obs(2, _result("STOP"), lp))
    policy.period_closed(2 * STRIDE, decision)
    assert decision.accept
    assert all(policy.period_request(n * STRIDE, np.zeros(1)) is None for n in range(3, 20))


def test_fresh_wake_word_after_fast_accept_opens_new_period():
    class _TwoEpisodes:
        def poll(self, s, w=None):
            if s < 6 * STRIDE:  # first wake word, re-arming until 1.5 s
                return GateState(True, s)
            if s < 12 * STRIDE:  # silence: gate holds the last detection time
                return GateState(True, 5 * STRIDE)
            return GateState(True, 12 * STRIDE)  # second wake word at 3.0 s

        def close(self):
            pass

    policy = _policy(_TwoEpisodes(), stable_strides=1)
    lp = _logp("stop", 60, trailing_blank=40)
    policy.period_request(2 * STRIDE, np.zeros(1))
    policy.period_closed(2 * STRIDE, policy.observe(_obs(2, _result("STOP"), lp)))
    reqs = [policy.period_request(n * STRIDE, np.zeros(1)) for n in range(3, 16)]
    starts = {r.start_samples for r in reqs if r is not None}
    assert starts == {12 * STRIDE}


def test_timeout_counts_from_last_wake_word_detection():
    """Window start stays at the first detection, but the time-out runs from
    the gate's latest re-arm (as SinglePeriodPolicy's period does)."""

    class _RearmUntil:
        def poll(self, s, w=None):
            return GateState(True, min(s, 4 * STRIDE))  # re-arms until 1.0 s

        def close(self):
            pass

    policy = _policy(_RearmUntil())
    reqs = {n: policy.period_request(n * STRIDE, np.zeros(1)) for n in range(0, 18)}
    assert reqs[15] is not None  # 3.75 s: still inside 1.0 s + 3.0 s
    assert reqs[16] is None  # 4.0 s: timed out
    starts = {r.start_samples for r in reqs.values() if r is not None}
    assert min(starts) == 0  # anchored at the first detection


def test_replay_summary_scoring():
    from me2_voicegen.vcm.streaming.replay_eval import summarize

    results = [
        {"label": "STOP", "speech_end_s": 2.0, "returncode": 0, "decodes": 6,
         "triggers": [{"t": 2.4, "intent": "STOP"}]},
        {"label": "TIMER", "speech_end_s": 3.0, "returncode": 0, "decodes": 8,
         "triggers": [{"t": 2.5, "intent": "TIME"}, {"t": 3.5, "intent": "TIMER"}]},
        {"label": "PAUSE", "speech_end_s": 1.5, "returncode": 0, "decodes": 10, "triggers": []},
    ]
    s = summarize(results)
    assert (s["correct_first_trigger"], s["wrong_intent_triggers"], s["missed"]) == (1, 1, 1)
    assert s["extra_triggers"] == 1 and s["early_first_triggers"] == 1
    assert s["latency_s"]["p50"] == pytest.approx(0.4)
    assert s["per_focus"]["STOP"] == [1, 1] and s["per_focus"]["PAUSE"] == [0, 1]
    assert s["decodes_per_session_mean"] == 8.0
