"""Tests for `vcm.streaming.runner`/`vcm.streaming.__main__`.

Fast tests use fake `AudioSource`s and either a fixed precomputed log-prob
array or the shared `vcm_stub_model_factory` fixture -- never a real
checkpoint, `.onnx` artifact, or microphone. The two real-artifact checks
(the completion-only smoke test and the torch/onnx parity check) are
`@pytest.mark.slow`.
"""

from __future__ import annotations

import dataclasses
import io
import inspect
import json
import math
from pathlib import Path
from typing import Optional

import numpy as np
import pytest
import torch
import torchaudio

from me2_voicegen.common.features import SAMPLE_RATE, LogMelFeatureExtractor
from me2_voicegen.vcm import alphabet
from me2_voicegen.vcm.decoder import NEG_INF, decode
from me2_voicegen.vcm.optionb import OPTIONB_GRAMMAR
from me2_voicegen.vcm.pipeline import logp_for_waveform
from me2_voicegen.vcm.streaming.backends import OnnxBackend, TorchBackend
from me2_voicegen.vcm.streaming.config import resolve_grammar, resolve_policy, resolve_threshold
from me2_voicegen.vcm.streaming.gate import GateState
from me2_voicegen.vcm.streaming.policy import ModePeriodPolicy, SinglePeriodPolicy, ThresholdPolicy
from me2_voicegen.vcm.streaming.runner import StreamingRunner, TriggerEvent
from me2_voicegen.vcm.streaming.sources import WavFileSource

PROJECT_ROOT = Path(__file__).resolve().parents[1]
OPTIONC_RUN_DIR = PROJECT_ROOT / "out" / "vcm" / "optionb-optionc"

TARGET_PHRASE = "change the brightness to one hundred percent"


# ---------------------------------------------------------------------------
# Fakes
# ---------------------------------------------------------------------------


class _ArrayAudioSource:
    """Yields a fixed numpy array in `block_samples`-sized blocks, no
    pacing. `sleep_s` (if set) sleeps between blocks -- used to model a
    genuinely slow/paced realtime source without a real device."""

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

    def __enter__(self):
        return self

    def __exit__(self, *exc_info) -> None:
        self.close()


class _RaisingAudioSource:
    is_realtime = True

    def __init__(self, exc: BaseException, n_ok_blocks: int = 0, block_samples: int = 1600) -> None:
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

    def __enter__(self):
        return self

    def __exit__(self, *exc_info) -> None:
        self.close()


class _FlushTrackingStringIO(io.StringIO):
    """Records explicit flushes from the JSONL writer."""

    def __init__(self) -> None:
        super().__init__()
        self.flush_count = 0

    def flush(self) -> None:
        self.flush_count += 1
        super().flush()


class _InfiniteAudioSource:
    """Never ends on its own -- like a real microphone. Used for
    `--listen-for` expiry tests."""

    is_realtime = True

    def __init__(self, block_samples: int = 1600, sleep_s: float = 0.001) -> None:
        self._block_samples = block_samples
        self._sleep_s = sleep_s
        self.closed = False

    def blocks(self):
        import time

        while True:
            time.sleep(self._sleep_s)
            yield np.zeros(self._block_samples, dtype=np.float32)

    def close(self) -> None:
        self.closed = True

    def __enter__(self):
        return self

    def __exit__(self, *exc_info) -> None:
        self.close()


class _FixedLogpBackend:
    """Ignores its input waveform entirely and always returns the same
    precomputed `(T, 29)` log-prob array -- models "the same command held
    across many overlapping windows" without depending on real feature
    extraction timing."""

    def __init__(self, logp: np.ndarray) -> None:
        self._logp = logp

    def logp_for_waveform(self, waveform: np.ndarray) -> np.ndarray:
        return self._logp


class _RecordingLogpBackend(_FixedLogpBackend):
    def __init__(self, logp: np.ndarray) -> None:
        super().__init__(logp)
        self.waveforms: list[np.ndarray] = []

    def logp_for_waveform(self, waveform: np.ndarray) -> np.ndarray:
        self.waveforms.append(waveform.copy())
        return super().logp_for_waveform(waveform)


class _KeyframeStubBackend:
    """Wraps a `_StubCTCModel` (see `tests/conftest.py`) behind the
    `InferenceBackend` protocol, going through the real
    `LogMelFeatureExtractor`/`vcm.pipeline.logp_for_waveform` path -- for
    tests that need `T` (frame count) to genuinely depend on window
    duration, unlike `_FixedLogpBackend`."""

    def __init__(self, model) -> None:
        self._model = model
        self._feature_extractor = LogMelFeatureExtractor()

    def logp_for_waveform(self, waveform: np.ndarray) -> np.ndarray:
        wav_t = torch.as_tensor(waveform, dtype=torch.float32)
        return logp_for_waveform(self._model, self._feature_extractor, wav_t, device="cpu")


class _SlowBackend:
    def __init__(self, logp: np.ndarray, sleep_s: float) -> None:
        self._logp = logp
        self._sleep_s = sleep_s

    def logp_for_waveform(self, waveform: np.ndarray) -> np.ndarray:
        import time

        time.sleep(self._sleep_s)
        return self._logp


class _SequenceLogpBackend:
    """Returns precomputed `(T, 29)` log-prob arrays in call order, cycling
    the last one -- models a window whose decode changes over time (needed
    to prove the policy's authoritative result overrides the flush window's
    own decode)."""

    def __init__(self, logp_sequence: list[np.ndarray]) -> None:
        self._sequence = logp_sequence
        self._calls = 0

    def logp_for_waveform(self, waveform: np.ndarray) -> np.ndarray:
        logp = self._sequence[min(self._calls, len(self._sequence) - 1)]
        self._calls += 1
        return logp


class _ScriptedFakeGate:
    """Deterministic `ListeningGate` for runner/CLI tests: open at
    `open_at` for `period_s` (auto-closing at the period end, like
    `SpacebarGate`), recording the waveforms handed to `poll()` and
    `close()` calls. No TTY, no timing."""

    def __init__(self, open_at: Optional[int], period_s: float) -> None:
        self.open_at = open_at
        self.period_samples = int(period_s * SAMPLE_RATE)
        self.poll_waveforms: list = []
        self.close_calls = 0

    def poll(self, samples_seen: int, window=None) -> GateState:
        self.poll_waveforms.append(window)
        if (
            self.open_at is not None
            and self.open_at <= samples_seen < self.open_at + self.period_samples
        ):
            return GateState(is_open=True, open_at_samples=self.open_at)
        return GateState(is_open=False, open_at_samples=None)

    def close(self) -> None:
        self.close_calls += 1


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


def _fixed_target_phrase_backend() -> _FixedLogpBackend:
    logp = _one_hot_logp(_forced_ids(TARGET_PHRASE, repeat=3, pad_to=200))
    return _FixedLogpBackend(logp)


# ---------------------------------------------------------------------------
# Lockstep (file) mode: debounce dedup + determinism.
# ---------------------------------------------------------------------------


def _make_lockstep_runner(
    refractory_s: float, total_samples: int = 24000, policy=None
) -> StreamingRunner:
    source = _ArrayAudioSource(np.zeros(total_samples, dtype=np.float32), block_samples=4000)
    backend = _fixed_target_phrase_backend()
    if policy is None:
        policy = ThresholdPolicy(threshold=-1e6)
    return StreamingRunner(
        source=source,
        backend=backend,
        grammar=OPTIONB_GRAMMAR,
        policy=policy,
        window_s=1.0,
        stride_s=0.25,
        refractory_s=refractory_s,
        beam_width=25,
        out=io.StringIO(),
    )


def test_lockstep_one_command_produces_exactly_one_event():
    runner = _make_lockstep_runner(refractory_s=1.5)
    summary = runner.run()

    assert summary.exit_reason == "source_eof"
    assert summary.events == 1
    assert len(runner.triggers) == 1
    assert summary.suppressed == summary.windows - 1


def test_lockstep_zero_refractory_produces_more_than_one_event():
    runner = _make_lockstep_runner(refractory_s=0.0)
    summary = runner.run()

    assert summary.events > 1
    assert summary.suppressed == 0


class _AlwaysAcceptPolicy:
    """Throwaway test-only `AcceptancePolicy`: accepts every window
    regardless of decoded evidence. Used only to prove the policy seam is a
    real swap point -- constructing `StreamingRunner` with this in place of
    `ThresholdPolicy` requires zero changes to `runner.py`."""

    def observe(self, obs):
        from me2_voicegen.vcm.streaming.policy import PolicyDecision

        return PolicyDecision(accept=True, reason="always_accept_policy")

    def reset(self) -> None:
        return None


class _NConsecutivePolicy:
    """Throwaway test-only `AcceptancePolicy`: accepts only once the same
    intent has been observed on `n` consecutive windows in a row (any
    non-matching or differing-intent window resets the streak). Strictly
    stricter than `ThresholdPolicy`, which accepts on the first
    above-threshold window."""

    def __init__(self, n: int) -> None:
        self.n = n
        self._streak_intent = None
        self._streak_len = 0

    def observe(self, obs):
        from me2_voicegen.vcm.streaming.policy import PolicyDecision

        intent = obs.result.intent
        if intent is None:
            self._streak_intent = None
            self._streak_len = 0
            return PolicyDecision(accept=False, reason="no_match: intent is None")
        if intent == self._streak_intent:
            self._streak_len += 1
        else:
            self._streak_intent = intent
            self._streak_len = 1
        if self._streak_len >= self.n:
            return PolicyDecision(
                accept=True, reason=f"{self._streak_len} consecutive windows agreeing on {intent!r}"
            )
        return PolicyDecision(
            accept=False, reason=f"only {self._streak_len}/{self.n} consecutive windows on {intent!r}"
        )

    def reset(self) -> None:
        self._streak_intent = None
        self._streak_len = 0


def test_acceptance_policy_is_swappable_through_the_constructor_with_no_runner_changes():
    """Plan-required proof (R2-2): injecting a different `AcceptancePolicy`
    through `StreamingRunner`'s constructor changes the emitted event
    sequence, using ONLY the two throwaway policies above -- neither
    `runner.py` nor any other production file was touched to support
    either policy, which is the entire point of the seam. Deliberately
    driven through the deterministic lockstep path (not realtime/threaded)
    so trigger timing is stable and independent of the realtime-path
    refractory behavior fixed under R2-1.
    """
    # The fixed backend always decodes the same intent at the same
    # confidence (~-0.053) on every window, so a threshold ABOVE that
    # (here 0.0) makes `ThresholdPolicy` reject every window -- the
    # baseline against which `AlwaysAcceptPolicy` (which ignores
    # confidence entirely) must accept strictly more.
    strict_threshold_summary = _make_lockstep_runner(
        refractory_s=0.0, policy=ThresholdPolicy(threshold=0.0)
    ).run()
    always_accept_summary = _make_lockstep_runner(
        refractory_s=0.0, policy=_AlwaysAcceptPolicy()
    ).run()
    assert always_accept_summary.events > strict_threshold_summary.events
    assert strict_threshold_summary.events == 0

    # Against the permissive threshold (-1e6, accepts every window
    # immediately -- ticket 01/04's own baseline), `NConsecutivePolicy(n=3)`
    # must accept strictly fewer windows since it withholds acceptance
    # until the 3rd consecutive agreeing window.
    permissive_threshold_summary = _make_lockstep_runner(refractory_s=0.0).run()
    n_consecutive_summary = _make_lockstep_runner(
        refractory_s=0.0, policy=_NConsecutivePolicy(n=3)
    ).run()
    assert n_consecutive_summary.events < permissive_threshold_summary.events


def test_lockstep_deterministic_across_runs_byte_for_byte():
    def run_once() -> str:
        source = _ArrayAudioSource(np.zeros(24000, dtype=np.float32), block_samples=4000)
        backend = _fixed_target_phrase_backend()
        policy = ThresholdPolicy(threshold=-1e6)
        out = io.StringIO()
        StreamingRunner(
            source=source,
            backend=backend,
            grammar=OPTIONB_GRAMMAR,
            policy=policy,
            window_s=1.0,
            stride_s=0.25,
            refractory_s=1.5,
            beam_width=25,
            out=out,
        ).run()
        return out.getvalue()

    first = run_once()
    second = run_once()
    assert first == second
    assert first  # sanity: something was actually emitted


# ---------------------------------------------------------------------------
# R2-1 regression: the Debouncer's cooldown must track real elapsed samples
# even when the realtime loop skips-and-counts dropped strides, not tick by
# a fixed one stride per evaluation (which would inflate the effective
# refractory period by the drop factor).
# ---------------------------------------------------------------------------


def test_realtime_refractory_period_not_inflated_by_dropped_windows():
    stride_s = 0.25
    strides_per_block = 3  # forces strides_elapsed=3 on (almost) every evaluation
    block_samples = int(round(stride_s * SAMPLE_RATE)) * strides_per_block
    n_blocks = 12
    total_samples = block_samples * n_blocks

    source = _ArrayAudioSource(
        np.zeros(total_samples, dtype=np.float32),
        block_samples=block_samples,
        is_realtime=True,
        sleep_s=0.05,
    )
    backend = _fixed_target_phrase_backend()
    policy = ThresholdPolicy(threshold=-1e6)  # always-accept once intent is found

    runner = StreamingRunner(
        source=source,
        backend=backend,
        grammar=OPTIONB_GRAMMAR,
        policy=policy,
        window_s=1.0,
        stride_s=stride_s,
        refractory_s=1.5,  # an exact multiple of strides_per_block * stride_s (0.75s)
        beam_width=25,
        out=io.StringIO(),
        poll_interval_s=0.001,
    )
    summary = runner.run()

    assert summary.dropped > 0  # sanity: the drop path was actually exercised
    assert len(runner.triggers) >= 3

    gaps = [
        b.t_seconds - a.t_seconds
        for a, b in zip(runner.triggers, runner.triggers[1:])
    ]
    # Pre-fix, ticking by a fixed one stride regardless of drops inflated
    # every gap to strides_per_block * refractory_s (4.5s here). Post-fix,
    # gaps should track the real elapsed time and land close to
    # refractory_s (rounded up to the nearest evaluation granularity).
    for gap in gaps:
        assert gap < 2.0 * 1.5, gaps  # nowhere near the 3x-inflated 4.5s bug


# ---------------------------------------------------------------------------
# Realtime (mic) mode: backpressure/skip-and-count, listen-for, EOF,
# KeyboardInterrupt, capture-thread exception propagation.
# ---------------------------------------------------------------------------


def test_realtime_backpressure_skips_and_counts_dropped_windows():
    total_samples = 64000
    samples = np.arange(total_samples, dtype=np.float32)
    source = _ArrayAudioSource(samples, block_samples=320, is_realtime=True, sleep_s=0.002)
    logp = _one_hot_logp([alphabet.BLANK_ID] * 5)
    backend = _SlowBackend(logp, sleep_s=0.03)
    policy = ThresholdPolicy(threshold=-1e6)

    runner = StreamingRunner(
        source=source,
        backend=backend,
        grammar=OPTIONB_GRAMMAR,
        policy=policy,
        window_s=0.1,
        stride_s=0.01,
        refractory_s=0.0,
        beam_width=5,
        out=io.StringIO(),
        poll_interval_s=0.001,
    )
    summary = runner.run()

    assert summary.exit_reason == "source_eof"
    assert summary.dropped > 0
    assert summary.windows > 0
    # Exact accounting invariant: every stride boundary crossed was either
    # evaluated (windows) or skipped-and-counted (dropped) -- never lost,
    # never double-counted.
    total_strides_covered = runner._buffer.samples_written // runner.stride_samples
    assert summary.windows + summary.dropped == total_strides_covered

    # Capture boundary never drops audio -- only inference opportunities.
    assert runner._buffer.samples_written == total_samples
    tail = runner._buffer.snapshot()
    expected_tail = samples[-tail.shape[0] :]
    assert np.array_equal(tail, expected_tail)


def test_realtime_listen_for_expiry_exits_cleanly_with_summary():
    source = _InfiniteAudioSource(block_samples=1600, sleep_s=0.001)
    logp = _one_hot_logp([alphabet.BLANK_ID] * 5)
    backend = _FixedLogpBackend(logp)
    policy = ThresholdPolicy(threshold=-1e6)

    runner = StreamingRunner(
        source=source,
        backend=backend,
        grammar=OPTIONB_GRAMMAR,
        policy=policy,
        window_s=0.1,
        stride_s=0.05,
        refractory_s=0.0,
        beam_width=5,
        listen_for_s=0.2,
        out=io.StringIO(),
        poll_interval_s=0.001,
    )
    summary = runner.run()

    assert summary.exit_reason == "listen_for_expired"
    assert source.closed


def test_realtime_source_eof_exits_cleanly_with_summary():
    source = _ArrayAudioSource(
        np.zeros(8000, dtype=np.float32), block_samples=1600, is_realtime=True
    )
    logp = _one_hot_logp([alphabet.BLANK_ID] * 5)
    backend = _FixedLogpBackend(logp)
    policy = ThresholdPolicy(threshold=-1e6)

    runner = StreamingRunner(
        source=source,
        backend=backend,
        grammar=OPTIONB_GRAMMAR,
        policy=policy,
        window_s=0.1,
        stride_s=0.05,
        refractory_s=0.0,
        beam_width=5,
        out=io.StringIO(),
        poll_interval_s=0.001,
    )
    summary = runner.run()

    assert summary.exit_reason == "source_eof"


def test_lockstep_keyboard_interrupt_exits_cleanly_with_summary():
    source = _RaisingAudioSource(KeyboardInterrupt(), n_ok_blocks=1, block_samples=4000)
    source.is_realtime = False
    logp = _one_hot_logp([alphabet.BLANK_ID] * 5)
    backend = _FixedLogpBackend(logp)
    policy = ThresholdPolicy(threshold=-1e6)

    runner = StreamingRunner(
        source=source,
        backend=backend,
        grammar=OPTIONB_GRAMMAR,
        policy=policy,
        window_s=1.0,
        stride_s=0.25,
        refractory_s=0.0,
        beam_width=5,
        out=io.StringIO(),
    )
    summary = runner.run()  # must not raise

    assert summary.exit_reason == "keyboard_interrupt"
    assert source.closed


def test_realtime_capture_thread_exception_propagates_not_hangs():
    source = _RaisingAudioSource(RuntimeError("boom"), n_ok_blocks=0)
    logp = _one_hot_logp([alphabet.BLANK_ID] * 5)
    backend = _FixedLogpBackend(logp)
    policy = ThresholdPolicy(threshold=-1e6)

    runner = StreamingRunner(
        source=source,
        backend=backend,
        grammar=OPTIONB_GRAMMAR,
        policy=policy,
        window_s=0.1,
        stride_s=0.05,
        refractory_s=0.0,
        beam_width=5,
        out=io.StringIO(),
        poll_interval_s=0.001,
    )
    with pytest.raises(RuntimeError, match="boom"):
        runner.run()


# ---------------------------------------------------------------------------
# Must Verify 2: window length is load-bearing (negative half).
# ---------------------------------------------------------------------------


def test_window_length_load_bearing_negative_half(vcm_stub_model_factory):
    model = vcm_stub_model_factory(forced_ids=_forced_ids(TARGET_PHRASE, repeat=3, pad_to=260))
    backend = _KeyframeStubBackend(model)
    policy = ThresholdPolicy(threshold=-1e6)

    def run_at_window(window_s: float) -> int:
        n_samples = int(round(window_s * SAMPLE_RATE))
        source = _ArrayAudioSource(np.zeros(n_samples, dtype=np.float32), block_samples=n_samples)
        runner = StreamingRunner(
            source=source,
            backend=backend,
            grammar=OPTIONB_GRAMMAR,
            policy=policy,
            window_s=window_s,
            stride_s=window_s,
            refractory_s=0.0,
            beam_width=25,
            out=io.StringIO(),
        )
        summary = runner.run()
        return summary.events

    # Positive half already established (real model, real posterior) --
    # this only checks our stub-model wiring doesn't contradict it.
    assert run_at_window(2.5) == 1
    # Negative half (Must Verify 2): too short a window can't fit the
    # phrase at all, regardless of confidence threshold.
    assert run_at_window(0.5) == 0


# ---------------------------------------------------------------------------
# Must Verify 3: decode(-inf) + AcceptancePolicy == decode(threshold)
# directly, at the runner level.
# ---------------------------------------------------------------------------


def test_decode_at_inf_plus_policy_equivalent_to_direct_threshold_decode(vcm_stub_model_factory):
    model = vcm_stub_model_factory(forced_ids=_forced_ids(TARGET_PHRASE, repeat=3, pad_to=260))
    feature_extractor = LogMelFeatureExtractor()
    threshold = -0.1  # the shipped real operating point

    waveform = torch.zeros(int(round(2.5 * SAMPLE_RATE)))
    logp = logp_for_waveform(model, feature_extractor, waveform, device="cpu")
    direct = decode(logp, OPTIONB_GRAMMAR, threshold=threshold, beam_width=25)

    source = _ArrayAudioSource(waveform.numpy(), block_samples=waveform.shape[0])
    backend = _KeyframeStubBackend(model)
    policy = ThresholdPolicy(threshold=threshold)
    runner = StreamingRunner(
        source=source,
        backend=backend,
        grammar=OPTIONB_GRAMMAR,
        policy=policy,
        window_s=2.5,
        stride_s=2.5,
        refractory_s=0.0,
        beam_width=25,
        out=io.StringIO(),
    )
    runner.run()

    assert (len(runner.triggers) == 1) == (not direct.no_match)
    if not direct.no_match:
        triggered = runner.triggers[0]
        assert triggered.intent == direct.intent
        assert triggered.slots == direct.slots
        assert triggered.text == direct.text


# ---------------------------------------------------------------------------
# Mode-period policy through the full runner (SPEC Proof 4): the ~6-line
# `_evaluate_window` diff (approved deviation 1) must carry the policy's
# authoritative result into the emitted event without changing the JSONL
# schema or `StreamingRunner.__init__`.
# ---------------------------------------------------------------------------

PHRASE_B = "volume up"


def test_streaming_runner_init_signature_unchanged_by_mode_period_wiring():
    # Approved deviation 1 confines the runner change to `_evaluate_window`:
    # a gate-aware policy goes in through the existing `policy` parameter,
    # so this signature stays exactly as it was.
    #
    # Intentional contract update (ticket 03 of
    # .scratch/incomplete-grammar-rejection/tickets): `required_command_margin`
    # was appended after `log_all_windows` (before `out`) for the
    # incomplete-prefix rejection gate; it defaults to None (gate disabled)
    # so every existing caller is unchanged.
    params = list(inspect.signature(StreamingRunner.__init__).parameters)
    assert params == [
        "self",
        "source",
        "backend",
        "grammar",
        "policy",
        "window_s",
        "stride_s",
        "refractory_s",
        "beam_width",
        "listen_for_s",
        "log_all_windows",
        "required_command_margin",
        "out",
        "summary_out",
        "poll_interval_s",
    ]


def test_mode_period_policy_emits_one_consolidated_event_overriding_flush_window():
    """Full runner + fake source + scripted fake gate + real
    `ModePeriodPolicy`: four windows decode PHRASE A (two each at two
    confidence levels) and one decodes PHRASE B inside a gate period; the
    flush window decodes PHRASE B itself. The single emitted event must
    carry PHRASE A (the mode) -- not the flush window's own decode -- with
    the mean confidence and the flush observation's window_index/t_seconds."""
    stride_s = 0.25
    stride_samples = int(round(stride_s * SAMPLE_RATE))
    open_at = stride_samples  # first evaluated stride
    period_s = 1.25
    close_at = open_at + int(period_s * SAMPLE_RATE)
    total_samples = close_at + 2 * stride_samples  # 8 evaluated windows
    samples = np.arange(total_samples, dtype=np.float32)

    logp_a_high = _one_hot_logp(_forced_ids(TARGET_PHRASE, repeat=3, pad_to=200), peak=12.0)
    logp_a_low = _one_hot_logp(_forced_ids(TARGET_PHRASE, repeat=3, pad_to=200), peak=8.0)
    logp_b = _one_hot_logp(_forced_ids(PHRASE_B, repeat=3, pad_to=200), peak=12.0)
    backend = _SequenceLogpBackend(
        [logp_a_high, logp_a_high, logp_a_low, logp_a_low, logp_b, logp_b]
    )

    gate = _ScriptedFakeGate(open_at=open_at, period_s=period_s)
    policy = ModePeriodPolicy(threshold=-1e6, gate=gate, period_s=period_s)

    runner = StreamingRunner(
        source=_ArrayAudioSource(samples, block_samples=stride_samples),
        backend=backend,
        grammar=OPTIONB_GRAMMAR,
        policy=policy,
        window_s=1.0,
        stride_s=stride_s,
        refractory_s=0.0,
        beam_width=25,
        out=io.StringIO(),
    )
    summary = runner.run()

    assert summary.exit_reason == "source_eof"
    assert summary.windows == total_samples // stride_samples
    assert summary.events == 1
    assert summary.suppressed == 0
    assert len(runner.triggers) == 1

    result_a_high = decode(logp_a_high, OPTIONB_GRAMMAR, threshold=NEG_INF, beam_width=25)
    result_a_low = decode(logp_a_low, OPTIONB_GRAMMAR, threshold=NEG_INF, beam_width=25)
    result_b = decode(logp_b, OPTIONB_GRAMMAR, threshold=NEG_INF, beam_width=25)
    assert result_a_high.intent is not None
    assert result_b.intent is not None
    assert result_a_high.intent != result_b.intent

    event = runner.triggers[0]
    # The mode (PHRASE A) overrides the flush window's own decode (PHRASE B).
    assert event.intent == result_a_high.intent
    assert event.intent != result_b.intent
    assert event.slots == result_a_high.slots
    assert event.text == result_a_high.text
    # Mean confidence across the four mode-class obs -- neither the latest
    # mode obs's own confidence nor the flush window's.
    assert event.confidence == pytest.approx(
        (result_a_high.confidence + result_a_low.confidence) / 2
    )
    # Flush observation's position, not the mode obs'.
    assert event.window_index == close_at // stride_samples
    assert event.t_seconds == pytest.approx(close_at / SAMPLE_RATE)
    assert event.policy_reason.startswith("mode_period:")
    assert "4 obs" in event.policy_reason

    # The window audio actually reached the gate (SPEC's audio-carrying
    # seam): the ring-buffer snapshot, never None.
    assert len(gate.poll_waveforms) == summary.windows
    assert np.array_equal(gate.poll_waveforms[0], samples[:stride_samples])
    assert all(isinstance(w, np.ndarray) and w.size > 0 for w in gate.poll_waveforms)


def test_single_period_runs_one_exact_gate_interval_inference():
    stride = int(0.25 * SAMPLE_RATE)
    period_s = 3.0
    open_at = stride
    close_at = open_at + int(period_s * SAMPLE_RATE)
    samples = np.arange(close_at + stride, dtype=np.float32)
    backend = _RecordingLogpBackend(_one_hot_logp(_forced_ids(TARGET_PHRASE, pad_to=200)))
    policy = SinglePeriodPolicy(-1e6, gate=_ScriptedFakeGate(open_at, period_s), period_s=period_s)
    runner = StreamingRunner(
        source=_ArrayAudioSource(samples, block_samples=stride), backend=backend,
        grammar=OPTIONB_GRAMMAR, policy=policy, window_s=2.5, stride_s=0.25,
        refractory_s=0.0, beam_width=25, out=io.StringIO(),
    )
    summary = runner.run()
    assert summary.events == 1
    assert len(backend.waveforms) == 1
    assert np.array_equal(backend.waveforms[0], samples[open_at:close_at])


def test_single_period_requires_exactly_three_seconds():
    gate = _ScriptedFakeGate(open_at=0, period_s=3.0)
    with pytest.raises(SystemExit, match="--gate-period 3"):
        resolve_policy("single_period", -0.1, gate=gate, period_s=5.0)


def test_single_period_discards_incomplete_period_without_inference():
    stride = int(0.25 * SAMPLE_RATE)
    open_at = stride
    close_at = open_at + int(3.0 * SAMPLE_RATE)
    backend = _RecordingLogpBackend(_one_hot_logp(_forced_ids(TARGET_PHRASE, pad_to=200)))
    runner = StreamingRunner(
        source=_ArrayAudioSource(np.zeros(close_at - stride, dtype=np.float32), block_samples=stride),
        backend=backend, grammar=OPTIONB_GRAMMAR,
        policy=SinglePeriodPolicy(-1e6, gate=_ScriptedFakeGate(open_at, 3.0), period_s=3.0),
        window_s=2.5, stride_s=0.25, refractory_s=0.0, beam_width=25, out=io.StringIO(),
    )
    summary = runner.run()
    assert summary.events == 0
    assert backend.waveforms == []


def test_jsonl_payload_key_set_unchanged():
    # No new JSONL fields (SPEC contract section 5): every record's key set
    # is exactly the pre-feature schema, 'window' and 'trigger' alike.
    source = _ArrayAudioSource(np.zeros(24000, dtype=np.float32), block_samples=4000)
    out = _FlushTrackingStringIO()
    StreamingRunner(
        source=source,
        backend=_fixed_target_phrase_backend(),
        grammar=OPTIONB_GRAMMAR,
        policy=ThresholdPolicy(threshold=-1e6),
        window_s=1.0,
        stride_s=0.25,
        refractory_s=1.5,
        beam_width=25,
        log_all_windows=True,
        out=out,
    ).run()

    expected_keys = frozenset(
        (
            "event",
            "t_seconds",
            "window_index",
            "intent",
            "slots",
            "text",
            "confidence",
            "policy_reason",
        )
    )
    # The TriggerEvent dataclass and the JSONL payload must stay in lockstep.
    assert frozenset(f.name for f in dataclasses.fields(TriggerEvent)) == expected_keys

    lines = out.getvalue().strip().splitlines()
    assert [json.loads(line)["event"] for line in lines] == [
        "trigger",
        "window",
        "window",
        "window",
        "window",
        "window",
    ]
    for line in lines:
        assert set(json.loads(line)) == expected_keys
    assert out.flush_count == len(lines)


def _broken_play_logp(total_frames: int = 200) -> np.ndarray:
    """Broken-`play` evidence: greedy text `play`, weak `time` terminal,
    high-confidence trailing blanks (duplicate of the fixture in
    tests/test_vcm_decoder_incomplete_prefix.py -- tests/ is not a package,
    so duplication is the repo idiom for test helpers). With the
    incomplete-prefix margin enabled, this decodes as a gate rejection.

    `play` substitutes for the spec's original `color` example: the
    live-upstream grammar refresh rewrote COLOR's phrasings so bare "color"
    is no longer a designated incomplete prefix (docs/OPTIONB-GRAMMAR-CONTRACT.md
    §6). `play` is a genuine `OPTIONB_GRAMMAR.incomplete_prefixes` member and,
    unlike `color`/`alarm`, shares no characters with `time`
    (see tests/test_vcm_decoder_incomplete_prefix.py for the full rationale
    and the frame-layout math this mirrors)."""
    neq = float("-inf")
    play_len = 4  # len("play")
    q_play, q_blank = 0.56, 0.44
    q_time = math.exp((-6.98 - play_len * math.log(q_blank)) / 4.0)
    logp = np.full((total_frames, 29), neq, dtype=np.float64)
    for t, ch in enumerate("play"):
        logp[t, alphabet.CHAR_TO_ID[ch]] = math.log(q_play)
        logp[t, alphabet.BLANK_ID] = math.log(q_blank)
    for i, ch in enumerate("time"):
        logp[play_len + i, alphabet.CHAR_TO_ID[ch]] = math.log(q_time)
        logp[play_len + i, alphabet.BLANK_ID] = math.log(1.0 - q_time)
    logp[play_len + 4 :, alphabet.BLANK_ID] = 0.0
    return logp


def test_jsonl_keeps_key_set_and_surfaces_gate_rejection_via_existing_fields_only():
    """Ticket 03 E8 / audit E4: with the margin enabled the JSONL payload
    keeps exactly its documented key set (the new DecodeResult fields do
    not leak), and a gate rejection surfaces only as `intent: null` /
    `confidence: null`; with the margin off the same run is the documented
    false accept, same key set."""
    expected_keys = frozenset(
        (
            "event",
            "t_seconds",
            "window_index",
            "intent",
            "slots",
            "text",
            "confidence",
            "policy_reason",
        )
    )

    def run_lockstep(margin) -> list[dict]:
        out = io.StringIO()
        StreamingRunner(
            source=_ArrayAudioSource(np.zeros(24000, dtype=np.float32), block_samples=4000),
            backend=_FixedLogpBackend(_broken_play_logp()),
            grammar=OPTIONB_GRAMMAR,
            policy=ThresholdPolicy(threshold=-1e6),
            window_s=1.0,
            stride_s=0.25,
            refractory_s=1.5,
            beam_width=25,
            log_all_windows=True,
            required_command_margin=margin,
            out=out,
        ).run()
        return [json.loads(line) for line in out.getvalue().strip().splitlines()]

    # Gate enabled: every window is rejected by the margin gate; the
    # rejection is observable only through the existing fields.
    gated = run_lockstep(0.0)
    assert gated
    assert all(set(record) == expected_keys for record in gated)
    assert all(record["event"] == "window" for record in gated)
    assert all(record["intent"] is None for record in gated)
    assert all(record["confidence"] is None for record in gated)

    # Gate disabled: the documented false accept, identical key set.
    baseline = run_lockstep(None)
    assert baseline
    assert all(set(record) == expected_keys for record in baseline)
    assert any(
        record["intent"] == "TIME" and record["confidence"] is not None
        for record in baseline
    )


# ---------------------------------------------------------------------------
# CLI: lazy mic construction, mic-unavailable error, startup banner,
# gate flags / cross-validation / gate lifecycle.
# ---------------------------------------------------------------------------


def test_cli_help_exits_cleanly_without_touching_microphone(monkeypatch):
    import me2_voicegen.vcm.streaming.__main__ as main_mod

    def _boom(*args, **kwargs):
        raise AssertionError("--help must never open the microphone")

    monkeypatch.setattr(main_mod, "open_microphone_source", _boom)

    with pytest.raises(SystemExit) as exc_info:
        main_mod.build_arg_parser().parse_args(["--help"])
    assert exc_info.value.code == 0


def test_cli_default_config_is_mic_but_construction_does_not_touch_device():
    from me2_voicegen.vcm.streaming.config import StreamingConfig

    cfg = StreamingConfig()
    assert cfg.source == "mic"
    assert cfg.realtime is True


def test_cli_mic_unavailable_raises_actionable_systemexit_not_traceback(monkeypatch):
    import me2_voicegen.vcm.streaming.__main__ as main_mod
    from me2_voicegen.vcm.streaming.sources import MicrophoneUnavailableError

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
        main_mod.main([])

    message = str(exc_info.value)
    assert "--source" in message
    assert "--mic-command" in message


def test_cli_startup_banner_reports_all_required_fields():
    import me2_voicegen.vcm.streaming.__main__ as main_mod
    from me2_voicegen.vcm.streaming.config import StreamingConfig

    cfg = StreamingConfig()
    out = io.StringIO()
    main_mod._print_banner(
        cfg,
        model_path=Path("out/vcm/optionb-optionc/export/vcm_model.fp32.onnx"),
        run_dir=Path("out/vcm/optionb-optionc"),
        threshold=-0.1,
        grammar_label="OPTIONB_GRAMMAR",
        out=out,
    )
    banner = out.getvalue()

    assert "optionb-optionc" in banner  # checkpoint/run dir
    assert "onnx" in banner  # backend
    assert cfg.onnx_variant in banner  # onnx variant
    assert f"window_s={cfg.window_s}" in banner
    assert f"stride_s={cfg.stride_s}" in banner
    assert f"refractory_s={cfg.refractory_s}" in banner
    assert f"beam_width={cfg.beam_width}" in banner
    assert "-0.1" in banner  # resolved threshold
    assert "CC-BY-NC-SA-4.0" in banner


def test_cli_banner_reports_listening_gate():
    import me2_voicegen.vcm.streaming.__main__ as main_mod
    from me2_voicegen.vcm.streaming.config import StreamingConfig

    def banner_for(cfg) -> str:
        out = io.StringIO()
        main_mod._print_banner(
            cfg,
            model_path=Path("out/vcm/optionb-optionc/export/vcm_model.fp32.onnx"),
            run_dir=Path("out/vcm/optionb-optionc"),
            threshold=-0.1,
            grammar_label="OPTIONB_GRAMMAR",
            out=out,
        )
        return out.getvalue()

    assert "listening gate: none" in banner_for(StreamingConfig())

    banner = banner_for(StreamingConfig(gate="spacebar", gate_period_s=7.5))
    assert (
        "listening gate: spacebar (press SPACE to open a 7.5 s listening period)"
        in banner
    )


def test_cli_help_lists_gate_flags(capsys):
    import me2_voicegen.vcm.streaming.__main__ as main_mod

    with pytest.raises(SystemExit) as exc_info:
        main_mod.build_arg_parser().parse_args(["--help"])
    assert exc_info.value.code == 0
    help_text = capsys.readouterr().out
    assert "--gate" in help_text
    assert "--gate-period" in help_text


def test_cli_mode_policy_with_none_gate_fails_fast_before_side_effects(monkeypatch):
    import me2_voicegen.vcm.streaming.__main__ as main_mod

    def _boom(*args, **kwargs):
        raise AssertionError(
            "--policy mode_period with --gate none must be rejected before "
            "gate construction and before the microphone opens"
        )

    monkeypatch.setattr(main_mod, "resolve_gate", _boom)
    monkeypatch.setattr(main_mod, "open_microphone_source", _boom)

    with pytest.raises(SystemExit) as exc_info:
        main_mod.main(["--policy", "mode_period"])

    message = str(exc_info.value)
    assert "--gate spacebar" in message
    assert "--gate none" in message


def test_cli_spacebar_gate_with_threshold_policy_fails_fast_before_side_effects(monkeypatch):
    import me2_voicegen.vcm.streaming.__main__ as main_mod

    def _boom(*args, **kwargs):
        raise AssertionError(
            "--gate spacebar with --policy threshold must be rejected before "
            "gate construction and before the microphone opens"
        )

    monkeypatch.setattr(main_mod, "resolve_gate", _boom)
    monkeypatch.setattr(main_mod, "open_microphone_source", _boom)

    with pytest.raises(SystemExit) as exc_info:
        main_mod.main(["--gate", "spacebar"])

    message = str(exc_info.value)
    assert "--policy mode_period" in message
    assert "--gate none" in message


def test_cli_gate_unavailable_raises_actionable_systemexit_not_traceback(monkeypatch):
    import me2_voicegen.vcm.streaming.__main__ as main_mod
    from me2_voicegen.vcm.streaming.gate import GateUnavailableError

    def _boom(*args, **kwargs):
        raise AssertionError("gate construction must fail before the microphone opens")

    monkeypatch.setattr(main_mod, "open_microphone_source", _boom)

    def _raise_unavailable(*args, **kwargs):
        raise GateUnavailableError(
            "stdin is not a usable TTY, so the spacebar listening gate cannot "
            "read keypresses. Run in an interactive shell, pass --gate none, "
            "or replay a file with --source <wav>."
        )

    monkeypatch.setattr(main_mod, "resolve_gate", _raise_unavailable)

    with pytest.raises(SystemExit) as exc_info:
        main_mod.main(["--policy", "mode_period", "--gate", "spacebar"])

    message = str(exc_info.value)
    assert "--gate none" in message
    assert "--source" in message


def _capture_runner_summaries(monkeypatch):
    import me2_voicegen.vcm.streaming.__main__ as main_mod

    # Wrap the real `StreamingRunner` to record the `RunSummary` each
    # `main()` run returns: the summary line is printed to a stderr file
    # object bound at import time, which this environment's capture
    # plumbing does not expose to capsys/capfd.
    real_cls = main_mod.StreamingRunner
    summaries: list = []

    def _factory(*args, **kwargs):
        runner = real_cls(*args, **kwargs)
        original_run = runner.run

        def _run():
            summary = original_run()
            summaries.append(summary)
            return summary

        runner.run = _run
        return runner

    monkeypatch.setattr(main_mod, "StreamingRunner", _factory)
    return summaries


def test_cli_gate_closed_when_run_completes_cleanly(monkeypatch, tmp_path):
    import me2_voicegen.vcm.streaming.__main__ as main_mod

    gate = _ScriptedFakeGate(open_at=0, period_s=5.0)
    monkeypatch.setattr(main_mod, "resolve_gate", lambda *a, **k: gate)
    monkeypatch.setattr(main_mod, "resolve_model", lambda *a, **k: Path("/fake/model.onnx"))
    monkeypatch.setattr(
        main_mod,
        "OnnxBackend",
        lambda *a, **k: _FixedLogpBackend(_one_hot_logp([alphabet.BLANK_ID] * 5)),
    )
    summaries = _capture_runner_summaries(monkeypatch)

    wav_path = tmp_path / "clip.wav"
    _write_sine_wav(wav_path, duration_s=0.5)

    main_mod.main(
        [
            "--policy", "mode_period",
            "--gate", "spacebar",
            "--threshold", "-0.1",
            "--source", str(wav_path),
        ]
    )

    assert [s.exit_reason for s in summaries] == ["source_eof"]
    assert gate.close_calls == 1


def test_cli_gate_closed_when_run_keyboard_interrupted(monkeypatch):
    import me2_voicegen.vcm.streaming.__main__ as main_mod

    gate = _ScriptedFakeGate(open_at=0, period_s=5.0)
    monkeypatch.setattr(main_mod, "resolve_gate", lambda *a, **k: gate)
    monkeypatch.setattr(main_mod, "resolve_model", lambda *a, **k: Path("/fake/model.onnx"))
    monkeypatch.setattr(
        main_mod,
        "OnnxBackend",
        lambda *a, **k: _FixedLogpBackend(_one_hot_logp([alphabet.BLANK_ID] * 5)),
    )
    summaries = _capture_runner_summaries(monkeypatch)

    def _interrupting_source(*args, **kwargs):
        source = _RaisingAudioSource(KeyboardInterrupt(), n_ok_blocks=2, block_samples=4000)
        source.is_realtime = False
        return source

    monkeypatch.setattr(main_mod, "open_microphone_source", _interrupting_source)

    main_mod.main(
        ["--policy", "mode_period", "--gate", "spacebar", "--threshold", "-0.1"]
    )  # must not raise

    assert [s.exit_reason for s in summaries] == ["keyboard_interrupt"]
    assert gate.close_calls == 1


# ---------------------------------------------------------------------------
# Slow: real optionc artifacts.
# ---------------------------------------------------------------------------

_MISSING_ARTIFACTS = not (
    (OPTIONC_RUN_DIR / "checkpoints" / "checkpoint.pt").exists()
    and (OPTIONC_RUN_DIR / "export" / "vcm_model.fp32.onnx").exists()
)


def _write_sine_wav(path: Path, duration_s: float = 3.0, freq_hz: float = 440.0) -> None:
    n = int(round(duration_s * SAMPLE_RATE))
    t = torch.arange(n, dtype=torch.float32) / SAMPLE_RATE
    waveform = (0.05 * torch.sin(2 * math.pi * freq_hz * t)).unsqueeze(0)
    torchaudio.save(str(path), waveform, SAMPLE_RATE)


@pytest.mark.slow
@pytest.mark.skipif(_MISSING_ARTIFACTS, reason="no checked-in optionb-optionc checkpoint/onnx export")
def test_runner_completes_over_real_onnx_artifact_on_synthetic_wav(tmp_path):
    """A sine tone is not speech -- this asserts completion only, not
    recognition, per this ticket's acceptance criteria."""
    wav_path = tmp_path / "tone.wav"
    _write_sine_wav(wav_path)

    grammar, grammar_label = resolve_grammar("optionb")
    threshold = resolve_threshold(OPTIONC_RUN_DIR, override=None, grammar_label=grammar_label)
    backend = OnnxBackend(OPTIONC_RUN_DIR / "export" / "vcm_model.fp32.onnx", ort_threads=1)
    policy = ThresholdPolicy(threshold)
    source = WavFileSource(wav_path, block_samples=1600, realtime=False)

    runner = StreamingRunner(
        source=source,
        backend=backend,
        grammar=grammar,
        policy=policy,
        window_s=2.5,
        stride_s=0.25,
        refractory_s=1.5,
        beam_width=25,
        out=io.StringIO(),
    )
    summary = runner.run()

    assert summary.exit_reason == "source_eof"
    assert summary.windows > 0


@pytest.mark.slow
@pytest.mark.skipif(_MISSING_ARTIFACTS, reason="no checked-in optionb-optionc checkpoint/onnx export")
def test_torch_and_onnx_backends_produce_same_intent_slots_on_same_wav(tmp_path):
    wav_path = tmp_path / "tone.wav"
    _write_sine_wav(wav_path)

    grammar, grammar_label = resolve_grammar("optionb")
    threshold = resolve_threshold(OPTIONC_RUN_DIR, override=None, grammar_label=grammar_label)

    backends = {
        "torch": TorchBackend(OPTIONC_RUN_DIR / "checkpoints" / "checkpoint.pt", device="cpu"),
        "onnx": OnnxBackend(OPTIONC_RUN_DIR / "export" / "vcm_model.fp32.onnx", ort_threads=1),
    }

    results = {}
    logp_by_backend = {}
    for name, backend in backends.items():
        source = WavFileSource(wav_path, block_samples=1600, realtime=False)
        policy = ThresholdPolicy(threshold)
        runner = StreamingRunner(
            source=source,
            backend=backend,
            grammar=grammar,
            policy=policy,
            window_s=2.5,
            stride_s=0.25,
            refractory_s=1.5,
            beam_width=10,
            out=io.StringIO(),
        )
        summary = runner.run()
        results[name] = [(t.intent, t.slots) for t in runner.triggers]
        # `results["torch"] == results["onnx"]` alone would pass vacuously
        # if a sine tone never triggers acceptance on either backend (both
        # lists empty) without proving the two backends agree on anything.
        # Directly compare the raw per-window backend output on a real
        # window of audio so this test still fails if the ONNX export and
        # the torch checkpoint diverge numerically, independent of whether
        # the acceptance policy ever fires.
        assert summary.windows > 0
        # Feed a real slice of the synthetic tone itself for the parity probe
        # rather than silence, so the probe exercises the same model path
        # the run itself used.
        wav_signal, _ = torchaudio.load(str(wav_path))
        window_samples = int(2.5 * 16000)
        clip = wav_signal[0, :window_samples].numpy().astype(np.float32)
        if clip.shape[0] < window_samples:
            clip = np.pad(clip, (0, window_samples - clip.shape[0]))
        logp_by_backend[name] = np.asarray(backend.logp_for_waveform(clip))

    assert results["torch"] == results["onnx"]
    torch_logp = logp_by_backend["torch"]
    onnx_logp = logp_by_backend["onnx"]
    assert torch_logp.shape == onnx_logp.shape
    np.testing.assert_allclose(torch_logp, onnx_logp, atol=1e-3, rtol=1e-3)


# ---------------------------------------------------------------------------
# CLI: --log-periods digest (printer format, run-end logger, main() wiring).
# ---------------------------------------------------------------------------


def test_log_periods_printer_formats_all_digest_lines():
    import me2_voicegen.vcm.streaming.__main__ as main_mod
    from me2_voicegen.vcm.streaming.policy import PolicyDecision

    out = io.StringIO()
    main_mod._print_period_event("open", 2 * SAMPLE_RATE, None, out=out)
    main_mod._print_period_event("reopened", 3 * SAMPLE_RATE, None, out=out)
    accept = PolicyDecision(
        accept=True,
        reason=(
            "mode_period: intent='TIME' mean confidence 0.87 "
            ">= threshold 0.7 over 8 obs"
        ),
    )
    main_mod._print_period_event("closed", 5 * SAMPLE_RATE, accept, out=out)
    reject = PolicyDecision(
        accept=False,
        reason="mode_period: silence dominated the period (None 9/10)",
    )
    main_mod._print_period_event("closed", 7 * SAMPLE_RATE, reject, out=out)
    main_mod._print_period_event("run_ended", None, None, out=out)

    assert out.getvalue().splitlines() == [
        "gate: open      t=2.00s",
        "gate: reopened  t=3.00s",
        "gate: closed    t=5.00s  period: ACCEPT  intent='TIME' "
        "mean confidence 0.87 >= threshold 0.7 over 8 obs",
        "gate: closed    t=7.00s  period: REJECT  "
        "silence dominated the period (None 9/10)",
        "gate: closed    (run ended)",
    ]


def test_run_end_gate_logger_prints_only_when_a_period_was_open():
    import me2_voicegen.vcm.streaming.__main__ as main_mod

    class _ScriptedInnerGate:
        def __init__(self, open_states):
            self._states = list(open_states)
            self.windows = []
            self.close_calls = 0

        def poll(self, samples_seen, window=None):
            self.windows.append(window)
            is_open = self._states.pop(0)
            return GateState(is_open=is_open, open_at_samples=0 if is_open else None)

        def close(self):
            self.close_calls += 1

    # period already closed before the run ends: no digest line
    inner = _ScriptedInnerGate([True, False])
    out = io.StringIO()
    logger = main_mod._RunEndGateLogger(inner, out=out)
    wave = np.zeros(16, dtype=np.float32)
    assert logger.poll(0, window=wave).is_open is True
    assert logger.poll(1).is_open is False
    assert inner.windows == [wave, None]  # waveform passes through untouched
    logger.close()
    assert out.getvalue() == ""
    assert inner.close_calls == 1

    # run ends with the period still open: balanced with a close line
    inner2 = _ScriptedInnerGate([False, True])
    out2 = io.StringIO()
    logger2 = main_mod._RunEndGateLogger(inner2, out=out2)
    logger2.poll(0)
    logger2.poll(1)
    logger2.close()
    assert out2.getvalue().splitlines() == ["gate: closed    (run ended)"]
    assert inner2.close_calls == 1

    # a run that never polled an open period stays silent (no open was
    # ever printed, so nothing to balance)
    inner3 = _ScriptedInnerGate([False])
    out3 = io.StringIO()
    logger3 = main_mod._RunEndGateLogger(inner3, out=out3)
    logger3.poll(0)
    logger3.close()
    assert out3.getvalue() == ""


def _log_periods_cli_setup(monkeypatch, tmp_path, duration_s, gate_period_s, extra_argv):
    """Drive `main()` against a wav replay with a scripted gate (open from
    t=0) and a captured `_print_period_event`; returns
    (events, gate, summaries)."""
    import me2_voicegen.vcm.streaming.__main__ as main_mod

    gate = _ScriptedFakeGate(open_at=0, period_s=gate_period_s)
    monkeypatch.setattr(main_mod, "resolve_gate", lambda *a, **k: gate)
    monkeypatch.setattr(main_mod, "resolve_model", lambda *a, **k: Path("/fake/model.onnx"))
    monkeypatch.setattr(
        main_mod,
        "OnnxBackend",
        lambda *a, **k: _FixedLogpBackend(_one_hot_logp([alphabet.BLANK_ID] * 5)),
    )
    events: list = []

    def _capture(event, samples_seen, decision, out=None):
        events.append((event, samples_seen, decision))

    monkeypatch.setattr(main_mod, "_print_period_event", _capture)
    summaries = _capture_runner_summaries(monkeypatch)

    wav_path = tmp_path / "clip.wav"
    _write_sine_wav(wav_path, duration_s=duration_s)

    main_mod.main(
        [
            "--policy", "mode_period",
            "--gate", "spacebar",
            "--threshold", "-0.1",
            "--window-s", "0.5",
            "--stride-s", "0.25",
            "--source", str(wav_path),
        ]
        + [str(item) for item in extra_argv]
    )
    return events, gate, summaries


def test_cli_log_periods_wires_digest_events_through_main(monkeypatch, tmp_path):
    events, gate, summaries = _log_periods_cli_setup(
        monkeypatch,
        tmp_path,
        duration_s=2.0,
        gate_period_s=1.0,
        extra_argv=["--gate-period", "1.0", "--log-periods"],
    )

    # window_s=0.5/stride_s=0.25 -> lockstep evaluations at 4000, 8000, ...;
    # the 1.0 s period [0, 16000) collects the 4000/8000/12000 observations
    # and flushes at the first evaluation >= 16000 (all BLANK decodes).
    assert [event for event, _, _ in events] == ["open", "closed"]
    assert events[0][1] == 4000 and events[0][2] is None
    assert events[1][1] == 16000
    assert events[1][2].accept is False
    assert "silence dominated the period (None 3/3)" in events[1][2].reason
    assert [s.exit_reason for s in summaries] == ["source_eof"]
    assert gate.close_calls == 1


def test_cli_log_periods_reports_run_ended_when_period_still_open(monkeypatch, tmp_path):
    events, gate, summaries = _log_periods_cli_setup(
        monkeypatch,
        tmp_path,
        duration_s=2.0,
        gate_period_s=5.0,
        extra_argv=["--log-periods"],
    )

    # the 5.0 s period outlives the 2.0 s file: the in-flight period is
    # discarded (never flushed) and the digest balances the open line.
    assert [event for event, _, _ in events] == ["open", "run_ended"]
    assert events[0][1] == 4000
    assert events[1] == ("run_ended", None, None)
    assert [s.exit_reason for s in summaries] == ["source_eof"]
    assert gate.close_calls == 1


def test_cli_without_log_periods_builds_no_logger_and_emits_no_events(monkeypatch, tmp_path):
    import me2_voicegen.vcm.streaming.__main__ as main_mod

    def _boom(*args, **kwargs):
        raise AssertionError("digest plumbing must stay inert without --log-periods")

    monkeypatch.setattr(main_mod, "_print_period_event", _boom)
    monkeypatch.setattr(main_mod, "_RunEndGateLogger", _boom)

    events, gate, summaries = _log_periods_cli_setup(
        monkeypatch,
        tmp_path,
        duration_s=2.0,
        gate_period_s=1.0,
        extra_argv=["--gate-period", "1.0"],
    )

    assert events == []
    assert [s.exit_reason for s in summaries] == ["source_eof"]
    assert gate.close_calls == 1  # the raw gate closed, no wrapper around it


def test_cli_help_lists_log_periods_flag(capsys):
    import me2_voicegen.vcm.streaming.__main__ as main_mod

    with pytest.raises(SystemExit) as exc_info:
        main_mod.build_arg_parser().parse_args(["--help"])
    assert exc_info.value.code == 0
    help_text = capsys.readouterr().out
    assert "--log-periods" in help_text


def test_cli_banner_reports_log_periods_when_enabled():
    import me2_voicegen.vcm.streaming.__main__ as main_mod
    from me2_voicegen.vcm.streaming.config import StreamingConfig

    def banner_for(cfg) -> str:
        out = io.StringIO()
        main_mod._print_banner(
            cfg,
            model_path=Path("out/vcm/optionb-optionc/export/vcm_model.fp32.onnx"),
            run_dir=Path("out/vcm/optionb-optionc"),
            threshold=-0.1,
            grammar_label="OPTIONB_GRAMMAR",
            out=out,
        )
        return out.getvalue()

    assert "--log-periods" not in banner_for(StreamingConfig())
    assert (
        "logging: per-period digest on stderr (--log-periods)"
        in banner_for(StreamingConfig(log_periods=True))
    )


# ---------------------------------------------------------------------------
# required_command_margin threading (ticket 03 of
# .scratch/incomplete-grammar-rejection/tickets, docs/
# INCOMPLETE-GRAMMAR-REJECTION.md Step 3).
# ---------------------------------------------------------------------------


def test_runner_passes_required_command_margin_to_decode(monkeypatch):
    """The runner must hand its `required_command_margin` (default `None`)
    to `decode` on every window evaluation."""
    import me2_voicegen.vcm.streaming.runner as runner_mod

    from me2_voicegen.vcm.decoder import DecodeResult

    calls: list = []

    def _recording_decode(
        logp, grammar, threshold, beam_width, required_command_margin=None
    ):
        calls.append(required_command_margin)
        return DecodeResult(
            intent=None,
            slots={},
            text="",
            confidence=float("-inf"),
            no_match=True,
            out_of_grammar_gap=float("inf"),
        )

    monkeypatch.setattr(runner_mod, "decode", _recording_decode)

    for margin in (None, 0.0):
        calls.clear()
        source = _ArrayAudioSource(np.zeros(24000, dtype=np.float32), block_samples=4000)
        backend = _FixedLogpBackend(_one_hot_logp([alphabet.BLANK_ID] * 5))
        runner = StreamingRunner(
            source=source,
            backend=backend,
            grammar=OPTIONB_GRAMMAR,
            policy=ThresholdPolicy(threshold=-1e6),
            window_s=1.0,
            stride_s=0.25,
            refractory_s=1.5,
            beam_width=25,
            required_command_margin=margin,
            out=io.StringIO(),
        )
        runner.run()
        assert len(calls) > 0
        assert all(m == margin for m in calls)
