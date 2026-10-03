"""Cross-cutting integration tests for the mode-period-gate feature:
runner + gate + policy composed through the real pipeline, spanning the
four developer tasks none of which owns these seams. Fakes are file-local
by design (no conftest coupling), mirroring the repo's self-contained
streaming test files.

Coverage owned here (SPEC edges not fully pinned by the task-owned tests):
- gate auto-close boundary and policy flush agree from one `period_s`
  (real `SpacebarGate` + real `ModePeriodPolicy`, plan Risks loop-closer)
- `--gate-period` flows to BOTH the gate and the policy from
  `main()`'s single `cfg.gate_period_s` source
- re-press near the period boundary and exactly at it, through the full
  `StreamingRunner` (discard-&-restart; the confirmed flush-obs-also-
  seeds-the-new-period behavior)
- edge 14: a second flush inside the refractory window is suppressed by
  the `Debouncer` and counted in `suppressed`
- edge 11: `--log-all-windows` window records carry the mode policy's
  collection reasons through the real runner, no JSONL schema change
- edge 7: run ending mid-period with CONFIDENT decodes in flight emits
  nothing (interrupted and EOF paths, through `main()`), and the
  `atexit` backstop closes a gate whose pre-runner failure path bypasses
  `main()`'s `try/finally`
- edge 13: realtime skip-and-count drops -- the gate is polled once per
  evaluated stride (never per loop tick) and the flush lands on an
  evaluated stride position
- banner + summary lines through the real `main()` flow with explicit I/O
  sinks: where the import-time-bound stderr lands under pytest capture
  depends on when the module was first imported (a collection-time import
  binds the capture's own buffer), so capture-based assertions on these
  lines are not deterministic here; JSONL is asserted at the runner level
  for the same reason
"""

from __future__ import annotations

import io
import json
from pathlib import Path
from typing import Optional

import numpy as np
import pytest
import termios

from me2_voicegen.common.features import SAMPLE_RATE
from me2_voicegen.vcm import alphabet
from me2_voicegen.vcm.decoder import DecodeResult, NEG_INF, decode
from me2_voicegen.vcm.optionb import OPTIONB_GRAMMAR
from me2_voicegen.vcm.streaming.gate import GateState, SpacebarGate
from me2_voicegen.vcm.streaming.policy import ModePeriodPolicy, WindowObservation
from me2_voicegen.vcm.streaming.runner import StreamingRunner
from me2_voicegen.vcm.streaming.sources import MicrophoneUnavailableError

PHRASE_A = "adjust brightness to one hundred percent"
PHRASE_B = "volume up"

_STRIDE_S = 0.25
_STRIDE = int(round(_STRIDE_S * SAMPLE_RATE))  # 4000 samples


# ---------------------------------------------------------------------------
# Fakes (file-local; the runner test file's house fakes, re-declared)
# ---------------------------------------------------------------------------


class _ArrayAudioSource:
    """Yields a fixed numpy array in `block_samples`-sized blocks;
    `sleep_s` (if set) is slept between blocks -- models a real-time
    capture when it equals the block's audio duration."""

    def __init__(
        self,
        samples: np.ndarray,
        block_samples: int,
        is_realtime: bool = False,
        sleep_s: float = 0.0,
    ) -> None:
        self.is_realtime = is_realtime
        self._samples = np.asarray(samples, dtype=np.float32)
        self._block_samples = block_samples
        self._sleep_s = sleep_s
        self.closed = False

    def blocks(self):
        import time

        n = self._samples.shape[0]
        pos = 0
        while pos < n:
            end = min(pos + self._block_samples, n)
            if self._sleep_s:
                time.sleep(self._sleep_s)
            yield self._samples[pos:end]
            pos = end

    def close(self) -> None:
        self.closed = True


class _RaisingAudioSource:
    """Yields `n_ok_blocks` zero blocks, then raises `exc` -- models an
    interrupted capture."""

    def __init__(self, exc: BaseException, n_ok_blocks: int, block_samples: int) -> None:
        self.is_realtime = False
        self._exc = exc
        self._n_ok_blocks = n_ok_blocks
        self._block_samples = block_samples
        self.closed = False

    def blocks(self):
        for _ in range(self._n_ok_blocks):
            yield np.zeros(self._block_samples, dtype=np.float32)
        raise self._exc

    def close(self) -> None:
        self.closed = True


class _FixedLogpBackend:
    def __init__(self, logp: np.ndarray) -> None:
        self._logp = logp

    def logp_for_waveform(self, waveform: np.ndarray) -> np.ndarray:
        return self._logp


class _SlowBackend:
    def __init__(self, logp: np.ndarray, sleep_s: float) -> None:
        self._logp = logp
        self._sleep_s = sleep_s

    def logp_for_waveform(self, waveform: np.ndarray) -> np.ndarray:
        import time

        time.sleep(self._sleep_s)
        return self._logp


class _SequenceLogpBackend:
    """Returns precomputed `(T, 29)` log-prob arrays in call order,
    cycling the last one."""

    def __init__(self, logp_sequence: list[np.ndarray]) -> None:
        self._sequence = logp_sequence
        self._calls = 0

    def logp_for_waveform(self, waveform: np.ndarray) -> np.ndarray:
        logp = self._sequence[min(self._calls, len(self._sequence) - 1)]
        self._calls += 1
        return logp


class _PressGate:
    """Scripted `ListeningGate` for full-runner tests: `SpacebarGate`'s
    poll ordering (auto-close first, then keypresses), with presses
    schedulable either at the next poll (`press`) or at an exact
    `samples_seen` (`press_at` -- needed because a synchronous lockstep
    run() cannot be interrupted between evaluations). `fixed_windows`
    model a gate already open on arrival as `(open_at, close_at)`
    pairs. Records every poll position and `close()` call."""

    def __init__(self, period_s: float, *, fixed_windows=()) -> None:
        self.period_samples = int(period_s * SAMPLE_RATE)
        self._open_at: Optional[int] = None
        self._pending_presses = 0
        self._press_schedule: list[int] = []
        self._fixed = list(fixed_windows)
        self.poll_positions: list[int] = []
        self.close_calls = 0

    def press(self) -> None:
        self._pending_presses += 1

    def press_at(self, samples_seen: int) -> None:
        self._press_schedule.append(samples_seen)

    def poll(self, samples_seen: int, window=None) -> GateState:
        self.poll_positions.append(samples_seen)
        if self._open_at is not None and samples_seen >= (
            self._open_at + self.period_samples
        ):
            self._open_at = None
        if self._pending_presses or samples_seen in self._press_schedule:
            self._pending_presses = 0
            if samples_seen in self._press_schedule:
                self._press_schedule.remove(samples_seen)
            self._open_at = samples_seen
        if self._open_at is None:
            for open_at, close_at in self._fixed:
                if open_at <= samples_seen < close_at:
                    self._open_at = open_at
                    break
        return GateState(is_open=self._open_at is not None, open_at_samples=self._open_at)

    def close(self) -> None:
        self.close_calls += 1
        self._open_at = None


def _forced_ids(phrase: str, repeat: int = 3, pad_to: int = 200) -> list[int]:
    ids: list[int] = []
    for char_id in alphabet.encode(phrase):
        ids.extend([char_id] * repeat)
    while len(ids) < pad_to:
        ids.append(alphabet.BLANK_ID)
    return ids


def _one_hot_logp(ids: list[int], alphabet_size: int = 29, peak: float = 12.0) -> np.ndarray:
    T = len(ids)
    logits = np.zeros((T, alphabet_size), dtype=np.float32)
    for t, char_id in enumerate(ids):
        logits[t, char_id] = peak
    shifted = logits - logits.max(axis=-1, keepdims=True)
    return shifted - np.log(np.exp(shifted).sum(axis=-1, keepdims=True))


def _blank_logp() -> np.ndarray:
    return _one_hot_logp([alphabet.BLANK_ID] * 200)


def _logp_for_phrase(phrase: str) -> np.ndarray:
    return _one_hot_logp(_forced_ids(phrase))


def _result(intent: Optional[str], confidence: float) -> DecodeResult:
    return DecodeResult(
        intent=intent,
        slots={},
        text="whatever",
        confidence=confidence,
        no_match=intent is None,
        out_of_grammar_gap=0.0,
    )


def _obs(intent: Optional[str], confidence: float, samples_seen: int) -> WindowObservation:
    return WindowObservation(
        window_index=samples_seen // _STRIDE,
        samples_seen=samples_seen,
        result=_result(intent, confidence),
    )


# ---------------------------------------------------------------------------
# Gate auto-close and policy flush agree from a single period_s
# ---------------------------------------------------------------------------


def _install_gate_fakes(monkeypatch, pressed: bytes = b""):
    """Monkeypatches termios/atexit/select/os.read for the REAL
    `SpacebarGate` (same technique as the task-owned gate tests);
    returns (setattr_calls, atexit_registrations)."""
    setattr_calls = []
    registered = []
    monkeypatch.setattr(
        termios, "tcgetattr", lambda fd: [0, 0, 0, termios.ICANON | termios.ECHO | termios.ISIG]
    )
    monkeypatch.setattr(
        termios, "tcsetattr", lambda fd, when, attrs: setattr_calls.append((fd, when, list(attrs)))
    )
    monkeypatch.setattr("atexit.register", registered.append)

    queue = [pressed[i : i + 1] for i in range(len(pressed))]

    def fake_select(rlist, wlist, xlist, timeout=None):
        return ([rlist[0]] if queue else [], [], [])

    def fake_read(fd, size):
        return queue.pop(0) if queue else b""

    import me2_voicegen.vcm.streaming.gate as gate_mod

    monkeypatch.setattr(gate_mod.select, "select", fake_select)
    monkeypatch.setattr(gate_mod.os, "read", fake_read)
    return setattr_calls, registered


def test_gate_auto_close_and_policy_flush_agree_on_same_stride_from_single_period_s(monkeypatch):
    """The plan's Risks loop-closer: `SpacebarGate`'s auto-close boundary
    and `ModePeriodPolicy`'s flush boundary are both
    `open_at + int(period_s * SAMPLE_RATE)` from the SAME `period_s`
    value -- probe one sample before and one observation at the
    boundary: the gate is still open at `close_at - 1` and BOTH classes
    transition on the same observation at `close_at`."""
    _install_gate_fakes(monkeypatch, pressed=b" ")
    period_s = 1.0
    close_at = int(period_s * SAMPLE_RATE)  # 16000

    gate = SpacebarGate(_FilelikeStdin(), period_s=period_s)
    policy = ModePeriodPolicy(threshold=-1.0, gate=gate, period_s=period_s)

    decisions = []
    for pos in (0, _STRIDE, 2 * _STRIDE, 3 * _STRIDE, close_at - 1):
        decisions.append(policy.observe(_obs("TIME", -0.2, pos)))
    # still collecting one sample before the boundary
    assert decisions[-1].reason == (
        f"collecting (5 obs, running mode 'TIME' 5/5)"
    )

    flush = policy.observe(_obs("TIME", -0.2, close_at))
    assert flush.accept is True
    assert flush.result is not None
    assert flush.result.intent == "TIME"
    assert flush.result.confidence == pytest.approx(-0.2)
    assert "over 5 obs" in flush.reason

    # the gate auto-closed on that same poll: the next observation is a
    # plain "gate closed" reject, not a collecting decision
    after = policy.observe(_obs("TIME", -0.2, close_at + _STRIDE))
    assert after.accept is False
    assert after.reason == "gate closed (not in a listening period)"


class _FilelikeStdin:
    def fileno(self) -> int:
        return 3


# ---------------------------------------------------------------------------
# main() passes its single cfg.gate_period_s to gate AND policy
# ---------------------------------------------------------------------------


def _capture_main_runner(monkeypatch, *, out=None, summary_out=None):
    """Wraps the real `StreamingRunner` used by `main()` and records each
    constructed runner and its `RunSummary` (the summary line goes to the
    import-time-bound stderr, which this sandbox does not expose to
    capsys; recording the object keeps the assertion non-vacuous). When
    `out`/`summary_out` are given, they are injected as the runner's I/O
    sinks -- the documented kwargs `main()` leaves at their import-time
    defaults."""
    import me2_voicegen.vcm.streaming.__main__ as main_mod

    real_cls = main_mod.StreamingRunner
    runners: list = []
    summaries: list = []

    def _factory(*args, **kwargs):
        if out is not None:
            kwargs.setdefault("out", out)
        if summary_out is not None:
            kwargs.setdefault("summary_out", summary_out)
        runner = real_cls(*args, **kwargs)
        original_run = runner.run

        def _run():
            summary = original_run()
            summaries.append(summary)
            return summary

        runner.run = _run
        runners.append(runner)
        return runner

    monkeypatch.setattr(main_mod, "StreamingRunner", _factory)
    return runners, summaries


def _wire_fake_main(monkeypatch, backend, source):
    """Point `main()`'s model/backend/mic at fakes; returns the dict a
    `resolve_gate` stand-in should record kwargs into plus the gate
    object it hands back."""
    import me2_voicegen.vcm.streaming.__main__ as main_mod

    gate_holder: dict = {}
    captured: dict = {}

    def _resolve_gate(name, **kwargs):
        captured.update({"name": name, "kwargs": kwargs})
        return gate_holder["gate"]

    monkeypatch.setattr(main_mod, "resolve_gate", _resolve_gate)
    monkeypatch.setattr(main_mod, "resolve_model", lambda *a, **k: Path("/fake/model.onnx"))
    monkeypatch.setattr(main_mod, "OnnxBackend", lambda *a, **k: backend)
    monkeypatch.setattr(main_mod, "open_microphone_source", lambda *a, **k: source)
    return captured, gate_holder


def test_main_passes_single_gate_period_to_gate_and_policy(monkeypatch):
    import me2_voicegen.vcm.streaming.__main__ as main_mod

    gate = _PressGate(period_s=5.0)
    captured, gate_holder = _wire_fake_main(
        monkeypatch,
        backend=_FixedLogpBackend(_blank_logp()),
        source=_ArrayAudioSource(np.zeros(8000, dtype=np.float32), block_samples=4000),
    )
    gate_holder["gate"] = gate
    runners, summaries = _capture_main_runner(monkeypatch)

    main_mod.main(
        [
            "--policy", "mode_period",
            "--gate", "spacebar",
            "--gate-period", "1.5",
            "--threshold", "-0.1",
        ]
    )

    # `--gate-period` reaches the gate via resolve_gate's period_s kwarg
    assert captured["name"] == "spacebar"
    assert captured["kwargs"]["period_s"] == 1.5
    # ...and the SAME value reaches the policy main() constructs: the
    # single cfg.gate_period_s source, no second knob to drift.
    assert len(runners) == 1
    policy = runners[0].policy
    assert isinstance(policy, ModePeriodPolicy)
    assert policy._gate is gate
    assert policy._period_samples == int(1.5 * SAMPLE_RATE)
    assert summaries[0].exit_reason == "source_eof"


# ---------------------------------------------------------------------------
# Re-press through the full runner: near the boundary and exactly at it
# ---------------------------------------------------------------------------


def _make_runner(
    backend,
    policy,
    total_samples: int,
    *,
    refractory_s: float = 0.0,
    log_all_windows: bool = False,
):
    out = io.StringIO()
    runner = StreamingRunner(
        source=_ArrayAudioSource(
            np.arange(total_samples, dtype=np.float32), block_samples=_STRIDE
        ),
        backend=backend,
        grammar=OPTIONB_GRAMMAR,
        policy=policy,
        window_s=1.0,
        stride_s=_STRIDE_S,
        refractory_s=refractory_s,
        beam_width=25,
        log_all_windows=log_all_windows,
        out=out,
        summary_out=io.StringIO(),
    )
    return runner, out


def test_repress_near_boundary_discards_in_flight_through_real_runner():
    """Edge 3 through the real pipeline: a re-press on the last stride
    before the period end discards the in-flight (PHRASE A) decodes and
    restarts; the single emitted event carries PHRASE B (the new
    period's mode) at the new period's flush position. Without the
    discard, the first period would have flushed at its own boundary
    with PHRASE A."""
    period_s = 1.0
    open_at = _STRIDE  # first evaluated stride
    close_at = open_at + int(period_s * SAMPLE_RATE)  # 20000

    logp_a = _logp_for_phrase(PHRASE_A)
    logp_b = _logp_for_phrase(PHRASE_B)
    # 8 evaluated windows: 4000..32000
    backend = _SequenceLogpBackend([logp_a, logp_a, logp_a, logp_b, logp_b, logp_b, logp_b, logp_b])

    gate = _PressGate(period_s=period_s)
    gate.press()  # opens at the first poll (4000)
    gate.press_at(close_at - _STRIDE)  # re-press at 16000, near the boundary

    policy = ModePeriodPolicy(threshold=-1e6, gate=gate, period_s=period_s)
    runner, _ = _make_runner(backend, policy, total_samples=32000)
    summary = runner.run()

    result_b = decode(logp_b, OPTIONB_GRAMMAR, threshold=NEG_INF, beam_width=25)
    assert result_b.intent is not None

    assert summary.exit_reason == "source_eof"
    assert summary.events == 1
    event = runner.triggers[0]
    assert event.intent == result_b.intent  # new period's mode, not the discarded A
    assert event.window_index == 8  # flush of the restarted period (32000), not 5
    assert event.t_seconds == pytest.approx(32000 / SAMPLE_RATE)
    assert "over 4 obs" in event.policy_reason


def test_repress_at_boundary_flushes_and_seeds_next_period_through_real_runner():
    """The confirmed flush-vs-simultaneous-reopen edge through the real
    pipeline: a press landing exactly on the stride a period expires
    returns the first period's flush decision for that observation AND
    seeds the new period with it. Both flushes emit; the second period's
    flush counts the seeding observation (`over 4 obs`, not 3)."""
    period_s = 1.0
    open_at = _STRIDE
    close_at = open_at + int(period_s * SAMPLE_RATE)  # 20000

    logp_a = _logp_for_phrase(PHRASE_A)
    logp_b = _logp_for_phrase(PHRASE_B)
    # 9 evaluated windows: 4000..36000
    backend = _SequenceLogpBackend(
        [logp_a, logp_a, logp_a, logp_a, logp_b, logp_b, logp_b, logp_b, logp_b]
    )

    gate = _PressGate(period_s=period_s)
    gate.press()
    gate.press_at(close_at)  # re-press exactly on the flush stride

    policy = ModePeriodPolicy(threshold=-1e6, gate=gate, period_s=period_s)
    runner, _ = _make_runner(backend, policy, total_samples=36000)
    summary = runner.run()

    result_a = decode(logp_a, OPTIONB_GRAMMAR, threshold=NEG_INF, beam_width=25)
    result_b = decode(logp_b, OPTIONB_GRAMMAR, threshold=NEG_INF, beam_width=25)
    assert result_a.intent is not None and result_b.intent is not None
    assert result_a.intent != result_b.intent

    assert summary.exit_reason == "source_eof"
    assert summary.events == 2
    first, second = runner.triggers

    assert first.intent == result_a.intent
    assert first.window_index == 5  # the flush observation (20000)
    assert first.t_seconds == pytest.approx(close_at / SAMPLE_RATE)
    assert "over 4 obs" in first.policy_reason

    assert second.intent == result_b.intent
    assert second.window_index == 9  # the new period's flush (36000)
    assert second.t_seconds == pytest.approx(36000 / SAMPLE_RATE)
    # 4 B obs = the seeding flush observation + 3 collected: without the
    # seed this would read `over 3 obs`.
    assert "over 4 obs" in second.policy_reason


# ---------------------------------------------------------------------------
# Edge 14: a debouncer-suppressed flush counts in `suppressed`
# ---------------------------------------------------------------------------


def test_edge_14_second_flush_within_refractory_is_suppressed_and_counted():
    """Two consecutive periods (possible with short `--gate-period`)
    whose flushes are < `refractory_s` apart: the second flush is a real
    policy accept that the `Debouncer` rate-limits -- it is counted in
    `suppressed`, and with `--log-all-windows` it still records as a
    window with the flush reason."""
    period_s = 0.5  # 8000 samples: two periods fit inside the 1.5s refractory
    refractory_s = 1.5
    open_at = _STRIDE
    first_close_at = open_at + int(period_s * SAMPLE_RATE)  # 12000

    backend = _FixedLogpBackend(_logp_for_phrase(PHRASE_A))
    gate = _PressGate(period_s=period_s)
    gate.press()  # period 1: [4000, 12000)
    gate.press_at(first_close_at)  # period 2: [12000, 20000), flush 8000 later

    policy = ModePeriodPolicy(threshold=-1e6, gate=gate, period_s=period_s)
    runner, out = _make_runner(
        backend, policy, total_samples=20000,
        refractory_s=refractory_s, log_all_windows=True,
    )
    summary = runner.run()

    assert summary.exit_reason == "source_eof"
    assert summary.events == 1
    assert summary.suppressed == 1  # the second flush, not a missing one

    records = [json.loads(line) for line in out.getvalue().strip().splitlines()]
    assert [r["event"] for r in records] == ["window", "window", "trigger", "window", "window"]
    # first flush emitted at its boundary
    assert records[2]["window_index"] == 3
    assert records[2]["policy_reason"].startswith("mode_period:")
    assert "over 2 obs" in records[2]["policy_reason"]
    # second flush: policy accepted (reason carries the mean), debouncer
    # suppressed the emission
    suppressed = records[4]
    assert suppressed["window_index"] == 5
    assert ">= threshold" in suppressed["policy_reason"]
    assert "over 2 obs" in suppressed["policy_reason"]
    assert suppressed["intent"] is not None


# ---------------------------------------------------------------------------
# Edge 11: --log-all-windows collection reasons through the real runner
# ---------------------------------------------------------------------------


def test_edge_11_log_all_windows_shows_collection_state_through_real_runner():
    """Every record a `--log-all-windows` mode-policy run emits carries
    the collection state in `policy_reason` exactly as the SPEC
    prescribes, the flush records as usual, and the key set is the
    pre-feature schema (no new JSONL fields)."""
    period_s = 1.0
    # two fixed gate windows: [8000, 24000) and [24000, 40000)
    gate = _PressGate(
        period_s=period_s,
        fixed_windows=[
            (8000, 8000 + int(period_s * SAMPLE_RATE)),
            (24000, 24000 + int(period_s * SAMPLE_RATE)),
        ],
    )
    logp_a = _logp_for_phrase(PHRASE_A)
    blank = _blank_logp()
    # 10 evaluated windows: 4000..40000; 5 A decodes then silence
    backend = _SequenceLogpBackend([logp_a] * 5 + [blank] * 5)

    policy = ModePeriodPolicy(threshold=-0.1, gate=gate, period_s=period_s)
    runner, out = _make_runner(backend, policy, total_samples=40000, log_all_windows=True)
    summary = runner.run()

    result_a = decode(logp_a, OPTIONB_GRAMMAR, threshold=NEG_INF, beam_width=25)
    assert result_a.intent is not None

    # gate polled exactly once per evaluated window, never per tick
    assert gate.poll_positions == [_STRIDE * i for i in range(1, 11)]

    records = [json.loads(line) for line in out.getvalue().strip().splitlines()]
    assert summary.events == 1  # only the first period's flush triggers
    assert [r["event"] for r in records] == (
        ["window"] * 5 + ["trigger"] + ["window"] * 4
    )

    assert records[0]["policy_reason"] == "gate closed (not in a listening period)"
    for i, count in enumerate((1, 2, 3, 4), start=1):
        assert records[i]["policy_reason"] == (
            f"collecting ({count} obs, running mode {result_a.intent!r} {count}/{count})"
        )

    flush = records[5]
    assert flush["intent"] == result_a.intent
    assert flush["slots"] == result_a.slots
    assert flush["text"] == result_a.text
    assert flush["confidence"] == pytest.approx(result_a.confidence)
    assert flush["window_index"] == 6  # the flush observation (24000)
    assert f"intent={result_a.intent!r}" in flush["policy_reason"]
    assert ">= threshold" in flush["policy_reason"]
    assert "over 4 obs" in flush["policy_reason"]

    # second (silence-dominated) period: collecting None, then the
    # silence-dominated flush records as a window -- no trigger (edge 4)
    for i, count in enumerate((2, 3, 4), start=6):
        assert records[i]["policy_reason"] == (
            f"collecting ({count} obs, running mode None {count}/{count})"
        )
    silence_flush = records[9]
    assert silence_flush["intent"] is None
    assert silence_flush["policy_reason"] == (
        "mode_period: silence dominated the period (None 4/4)"
    )

    # no JSONL schema change: the pre-feature key set on every record
    expected_keys = frozenset(
        ("event", "t_seconds", "window_index", "intent", "slots", "text", "confidence", "policy_reason")
    )
    for record in records:
        assert set(record) == expected_keys


# ---------------------------------------------------------------------------
# Edge 7: run ends with a CONFIDENT period open -> nothing emitted
# (through the real main(), interrupted and EOF paths)
# ---------------------------------------------------------------------------


def _wire_confident_main(monkeypatch, source):
    """main() with a fake gate open from t=0 for 5s and a backend that
    confidently decodes PHRASE A on every window -- the in-flight period
    is meaningful: a wrong implementation that flushes on exit would
    emit an event."""
    import me2_voicegen.vcm.streaming.__main__ as main_mod

    gate = _PressGate(period_s=5.0, fixed_windows=[(0, 5.0 * SAMPLE_RATE)])
    captured, gate_holder = _wire_fake_main(
        monkeypatch,
        backend=_FixedLogpBackend(_logp_for_phrase(PHRASE_A)),
        source=source,
    )
    gate_holder["gate"] = gate
    runners, summaries = _capture_main_runner(monkeypatch)
    return captured, gate, runners, summaries


def test_keyboard_interrupt_with_open_period_and_confident_decodes_emits_nothing(monkeypatch):
    import me2_voicegen.vcm.streaming.__main__ as main_mod

    source = _RaisingAudioSource(KeyboardInterrupt(), n_ok_blocks=2, block_samples=4000)
    _, gate, runners, summaries = _wire_confident_main(monkeypatch, source)

    main_mod.main(
        ["--policy", "mode_period", "--gate", "spacebar", "--threshold", "-0.1"]
    )  # must not raise

    assert summaries[0].exit_reason == "keyboard_interrupt"
    assert summaries[0].events == 0  # in-flight confident period discarded
    assert summaries[0].windows == 2
    assert gate.close_calls == 1  # main()'s finally restored the gate
    assert runners[0].triggers == []


def test_run_ends_with_open_period_and_confident_decodes_emits_nothing(monkeypatch):
    import me2_voicegen.vcm.streaming.__main__ as main_mod

    source = _ArrayAudioSource(np.zeros(8000, dtype=np.float32), block_samples=4000)
    _, gate, runners, summaries = _wire_confident_main(monkeypatch, source)

    main_mod.main(
        ["--policy", "mode_period", "--gate", "spacebar", "--threshold", "-0.1"]
    )

    assert summaries[0].exit_reason == "source_eof"
    assert summaries[0].events == 0
    assert summaries[0].windows == 2
    assert gate.close_calls == 1
    assert runners[0].triggers == []


def test_atexit_backstop_closes_gate_on_pre_runner_failure(monkeypatch):
    """A failure between gate construction and the runner (here: the
    microphone) bypasses main()'s try/finally -- the `atexit` backstop
    registered by `SpacebarGate` is what restores the TTY. Drives the
    REAL `resolve_gate`/`SpacebarGate` (termios monkeypatched), then
    simulates process exit by invoking the registered handler."""
    import me2_voicegen.vcm.streaming.__main__ as main_mod

    original_attrs = [0, 0, 0, termios.ICANON | termios.ECHO | termios.ISIG]
    setattr_calls = []
    registered = []
    monkeypatch.setattr(termios, "tcgetattr", lambda fd: list(original_attrs))
    monkeypatch.setattr(
        termios, "tcsetattr", lambda fd, when, attrs: setattr_calls.append((fd, when, list(attrs)))
    )
    monkeypatch.setattr("atexit.register", registered.append)

    # `resolve_gate`'s default stdin is `sys.stdin` bound at import time;
    # under pytest's capture that is a DontReadFromInput whose fileno()
    # raises, so point the real resolve_gate at the file-like fake (the
    # same technique the task-owned gate tests use) -- main()'s call path
    # itself is unchanged.
    monkeypatch.setattr(main_mod.resolve_gate, "__kwdefaults__", {"stdin": _FilelikeStdin()})

    real_resolve = main_mod.resolve_gate
    gates: list = []

    def _resolve(name, **kwargs):
        gate = real_resolve(name, **kwargs)
        gates.append(gate)
        return gate

    monkeypatch.setattr(main_mod, "resolve_gate", _resolve)
    monkeypatch.setattr(main_mod, "resolve_model", lambda *a, **k: Path("/fake/model.onnx"))

    class _DummyBackend:
        def logp_for_waveform(self, waveform):
            raise NotImplementedError

    monkeypatch.setattr(main_mod, "OnnxBackend", lambda *a, **k: _DummyBackend())

    def _raise_unavailable(*args, **kwargs):
        raise MicrophoneUnavailableError(
            "microphone command ['arecord'] not found. Use --source <wav> to replay "
            "a file instead, or pass a working --mic-command."
        )

    monkeypatch.setattr(main_mod, "open_microphone_source", _raise_unavailable)

    with pytest.raises(SystemExit) as exc_info:
        main_mod.main(["--policy", "mode_period", "--gate", "spacebar"])

    # actionable, never a traceback -- and it is the MIC failure (pre-runner,
    # bypassing main()'s try/finally), not a gate failure:
    message = str(exc_info.value)
    assert "--source" in message
    assert "microphone" in message

    assert len(gates) == 1
    assert registered == [gates[0].close]  # the backstop was registered
    assert len(setattr_calls) == 1  # raw mode entered; finally never ran

    # simulate process exit: the atexit handler restores the saved TTY
    registered[0]()
    assert len(setattr_calls) == 2
    fd, when, attrs = setattr_calls[-1]
    assert when == termios.TCSANOW
    assert attrs == original_attrs


# ---------------------------------------------------------------------------
# Edge 13: realtime skip-and-count drops spanning the period
# ---------------------------------------------------------------------------


def test_realtime_drops_spanning_period_keep_gate_polls_on_evaluated_strides():
    """The realtime loop skip-and-counts dropped strides while a gate
    period is open and flushes: the source paces at 10x real time
    (320-sample blocks, 2 ms apart) and the backend is slower than one
    paced stride (0.03 s > 0.025 s), so the loop deterministically falls
    behind and skips boundaries. The gate is polled once per evaluated
    stride (never per loop tick), windows+dropped accounts for every
    stride, and the consolidated flush lands on an evaluated stride
    position >= the period end -- sample-accurate across drops."""
    period_s = 2.0
    close_at = int(period_s * SAMPLE_RATE)  # 32000
    total_samples = 64000

    backend = _SlowBackend(_logp_for_phrase(PHRASE_A), sleep_s=0.03)
    gate = _PressGate(period_s=period_s, fixed_windows=[(0, close_at)])
    policy = ModePeriodPolicy(threshold=-1e6, gate=gate, period_s=period_s)

    runner = StreamingRunner(
        source=_ArrayAudioSource(
            np.zeros(total_samples, dtype=np.float32),
            block_samples=320,
            is_realtime=True,
            sleep_s=0.002,
        ),
        backend=backend,
        grammar=OPTIONB_GRAMMAR,
        policy=policy,
        window_s=1.0,
        stride_s=_STRIDE_S,
        refractory_s=0.0,
        beam_width=25,
        out=io.StringIO(),
        summary_out=io.StringIO(),
        poll_interval_s=0.001,
    )
    summary = runner.run()

    assert summary.exit_reason == "source_eof"
    # the drop path was actually exercised, and no stride is lost: every
    # stride boundary was either evaluated or skipped-and-counted
    assert summary.dropped > 0
    assert summary.windows + summary.dropped == total_samples // _STRIDE

    # gate polled once per evaluated stride, at stride-aligned positions
    assert len(gate.poll_positions) == summary.windows
    assert all(pos % _STRIDE == 0 for pos in gate.poll_positions)

    assert summary.events == 1
    event = runner.triggers[0]
    result_a = decode(backend._logp, OPTIONB_GRAMMAR, threshold=NEG_INF, beam_width=25)
    assert event.intent == result_a.intent
    # the flush is the first evaluated stride >= the period end
    assert event.t_seconds >= close_at / SAMPLE_RATE
    assert event.t_seconds / _STRIDE_S == pytest.approx(round(event.t_seconds / _STRIDE_S))


# ---------------------------------------------------------------------------
# Banner + summary lines through the real main() flow (explicit I/O sinks)
# ---------------------------------------------------------------------------


def test_main_banner_and_summary_lines_end_to_end(monkeypatch):
    """The `listening gate:` banner line and the `streaming run
    complete:` summary are the operator-visible contract of the CLI,
    asserted through real `main()` runs. The production code prints to
    stderr objects bound at import time; under pytest capture where those
    land depends on when the module was first imported (a collection-time
    import binds the capture's own buffer), so capture-based assertions
    on these lines are not deterministic -- the real `_print_banner` is
    run with a forced `out`, and the real runner's documented
    `out`/`summary_out` kwargs are injected, which `main()` leaves at
    their import-time defaults."""
    import me2_voicegen.vcm.streaming.__main__ as main_mod

    banner_out = io.StringIO()
    summary_out = io.StringIO()
    jsonl_out = io.StringIO()

    real_banner = main_mod._print_banner

    def _banner(*args, **kwargs):
        kwargs["out"] = banner_out
        return real_banner(*args, **kwargs)

    monkeypatch.setattr(main_mod, "_print_banner", _banner)

    _, _gate_holder = _wire_fake_main(
        monkeypatch,
        backend=_FixedLogpBackend(_blank_logp()),
        source=_ArrayAudioSource(np.zeros(8000, dtype=np.float32), block_samples=4000),
    )
    _gate_holder["gate"] = _PressGate(period_s=5.0, fixed_windows=[(0, 5.0 * SAMPLE_RATE)])
    _, summaries = _capture_main_runner(
        monkeypatch, out=jsonl_out, summary_out=summary_out
    )

    main_mod.main(
        [
            "--policy", "mode_period",
            "--gate", "spacebar",
            "--threshold", "-0.1",
        ]
    )

    assert "listening gate: spacebar (press SPACE to open a 5.0 s listening period)" in (
        banner_out.getvalue()
    )
    assert (
        "streaming run complete: windows=2 events=0 suppressed=0 dropped=0 "
        "exit_reason=source_eof"
    ) in summary_out.getvalue()
    assert summaries[0].exit_reason == "source_eof"

    banner_out.truncate(0)
    banner_out.seek(0)
    summary_out.truncate(0)
    summary_out.seek(0)
    main_mod.main(
        ["--policy", "threshold", "--gate", "none", "--threshold", "-0.1"]
    )
    assert "listening gate: none" in banner_out.getvalue()
    assert (
        "streaming run complete: windows=2 events=0 suppressed=0 dropped=0 "
        "exit_reason=source_eof"
    ) in summary_out.getvalue()


def test_main_warms_up_beam_search_once_before_run(monkeypatch, capsys):
    import me2_voicegen.vcm.streaming.__main__ as main_mod

    gate_holder: dict = {}
    _wire_fake_main(
        monkeypatch,
        backend=_FixedLogpBackend(_blank_logp()),
        source=_ArrayAudioSource(np.zeros(8000, dtype=np.float32), block_samples=4000),
    )
    gate_holder["gate"] = _PressGate(period_s=5.0)
    monkeypatch.setattr(main_mod, "resolve_gate", lambda name, **kw: gate_holder["gate"])
    events: list = []
    monkeypatch.setattr(
        main_mod.beam_decoder, "warm_up", lambda root, bw: (events.append(("warm_up", bw)), ("python", 12.0))[1]
    )
    runners, _ = _capture_main_runner(monkeypatch)
    real_factory = main_mod.StreamingRunner

    def _factory(*a, **k):
        events.append(("runner_built", None))
        return real_factory(*a, **k)

    monkeypatch.setattr(main_mod, "StreamingRunner", _factory)

    main_mod.main(["--policy", "mode_period", "--gate", "spacebar", "--threshold", "-0.1", "--beam-width", "7"])

    assert events == [("warm_up", 7), ("runner_built", None)]
    assert "beam search: backend=python warm-up=12 ms" in capsys.readouterr().err
