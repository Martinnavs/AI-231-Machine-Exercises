"""The streaming runner: wires `RingBuffer` -> `InferenceBackend` ->
`decode(threshold=-inf)` -> `AcceptancePolicy` -> `Debouncer` into the two
loop modes selected by `AudioSource.is_realtime`. See
`ME2/docs/STREAMING-CONTRACT.md` sections 1/4/5 for the fixed pipeline
order, the realtime-vs-lockstep backpressure contract, and the JSONL event
schema this module emits; see that doc's runner section (added by this
task) for the loop-mode/shutdown details below.

Per-window compute is dominated by grammar-constrained beam search, not the
inference backend (~97% beam search / ~3% ONNX forward at the shipped
defaults, T=251 beam=25) -- see this ticket's Execution Log for the
measurement. Nothing here changes that; this module only wires the already
sized/measured pieces together.
"""

from __future__ import annotations

import dataclasses
import json
import sys
import threading
import time
from typing import Optional, TextIO

import numpy as np

from me2_voicegen.common.features import SAMPLE_RATE
from me2_voicegen.common.grammar_core import Grammar
from me2_voicegen.vcm.decoder import decode
from me2_voicegen.vcm.streaming.backends import InferenceBackend
from me2_voicegen.vcm.streaming.buffer import RingBuffer
from me2_voicegen.vcm.streaming.debounce import Debouncer
from me2_voicegen.vcm.streaming.policy import AcceptancePolicy, WindowObservation
from me2_voicegen.vcm.streaming.sources import AudioSource

NEG_INF = float("-inf")


@dataclasses.dataclass(frozen=True)
class TriggerEvent:
    """One emitted (non-suppressed, policy-accepted) window. Matches the
    JSONL `"trigger"` schema (`docs/STREAMING-CONTRACT.md` section 5)."""

    event: str
    t_seconds: float
    window_index: int
    intent: Optional[str]
    slots: dict
    text: str
    confidence: Optional[float]
    policy_reason: str


@dataclasses.dataclass(frozen=True)
class RunSummary:
    windows: int
    events: int
    suppressed: int
    dropped: int
    exit_reason: str


class StreamingRunner:
    """Consumes one `AudioSource`, evaluates it one stride at a time
    through `backend`/`grammar`/`policy`, and writes JSONL events to `out`
    (default `sys.stdout`) plus a human-readable summary to `summary_out`
    (default `sys.stderr`) on exit.

    Loop mode is selected by `source.is_realtime` at `run()` time:
    threaded skip-and-count for a realtime source (e.g. the microphone),
    synchronous lockstep with no drops for a non-realtime source (e.g. a
    replayed wav file) -- see `docs/STREAMING-CONTRACT.md` section 1 for
    why these need different backpressure behavior.
    """

    def __init__(
        self,
        source: AudioSource,
        backend: InferenceBackend,
        grammar: Grammar,
        policy: AcceptancePolicy,
        *,
        window_s: float,
        stride_s: float,
        refractory_s: float,
        beam_width: int,
        listen_for_s: Optional[float] = None,
        log_all_windows: bool = False,
        required_command_margin: Optional[float] = None,
        score_mode: str = "mean_frame",
        out: TextIO = sys.stdout,
        summary_out: TextIO = sys.stderr,
        poll_interval_s: float = 0.005,
        log_timing: bool = False,
    ) -> None:
        self.source = source
        self.backend = backend
        self.grammar = grammar
        self.policy = policy
        self.beam_width = beam_width

        self.window_samples = int(round(window_s * SAMPLE_RATE))
        self.stride_samples = int(round(stride_s * SAMPLE_RATE))
        if self.stride_samples <= 0:
            raise ValueError(f"stride_s must be > 0, got {stride_s}")
        self.refractory_samples = max(0, int(round(refractory_s * SAMPLE_RATE)))
        self.listen_for_samples = (
            int(round(listen_for_s * SAMPLE_RATE)) if listen_for_s is not None else None
        )
        self.log_all_windows = log_all_windows
        self.log_timing = log_timing  # adds `gate_ms` and `decode_ms` to each emitted record (off by default)
        # Incomplete-prefix rejection gate margin (docs/
        # INCOMPLETE-GRAMMAR-REJECTION.md, Step 3); None = gate disabled.
        self.required_command_margin = required_command_margin
        # What `DecodeResult.confidence` measures ("mean_frame" | "per_char",
        # see vcm.decoder.decode_utterance); the policy threshold must be on
        # the matching scale.
        self.score_mode = score_mode
        self.poll_interval_s = poll_interval_s

        self._out = out
        self._summary_out = summary_out

        period_samples = int(getattr(policy, "period_samples", 0))
        # A single-period close may be observed after the realtime loop has
        # crossed several strides. Retain a full ordinary window beyond the
        # requested period so its exact [open, close) interval survives that
        # catch-up delay.
        self._buffer = RingBuffer(max(self.window_samples, period_samples + self.window_samples))
        self._debouncer = Debouncer(self.refractory_samples)

        self._window_index = 0
        self._events = 0
        self._dropped = 0
        self.triggers: list[TriggerEvent] = []

    def run(self) -> RunSummary:
        self.policy.reset()
        self._debouncer.reset()
        self._buffer.reset()
        self._window_index = 0
        self._events = 0
        self._dropped = 0
        self.triggers = []

        exit_reason = "unknown"
        try:
            if self.source.is_realtime:
                exit_reason = self._run_realtime()
            else:
                exit_reason = self._run_lockstep()
        except KeyboardInterrupt:
            exit_reason = "keyboard_interrupt"
        finally:
            self.source.close()

        summary = RunSummary(
            windows=self._window_index,
            events=self._events,
            suppressed=self._debouncer.suppressed_count,
            dropped=self._dropped,
            exit_reason=exit_reason,
        )
        print(
            f"streaming run complete: windows={summary.windows} "
            f"events={summary.events} suppressed={summary.suppressed} "
            f"dropped={summary.dropped} exit_reason={summary.exit_reason}",
            file=self._summary_out,
        )
        return summary

    def _run_lockstep(self) -> str:
        """No capture thread: pulls blocks synchronously and evaluates
        every stride, dropping nothing. Deterministic -- same source
        contents always produce the same event sequence."""
        last_stride_boundary = 0
        for block in self.source.blocks():
            self._buffer.write(block)
            while self._buffer.samples_written - last_stride_boundary >= self.stride_samples:
                last_stride_boundary += self.stride_samples
                self._evaluate_window(last_stride_boundary)
        return "source_eof"

    def _run_realtime(self) -> str:
        """A capture thread drains `source.blocks()` into the ring buffer
        continuously; the main loop evaluates once per elapsed stride and
        skips-and-counts (`dropped_windows`) any stride boundaries it
        falls behind on, rather than queueing -- see
        `docs/STREAMING-CONTRACT.md` section 1."""
        capture_error: list[BaseException] = []

        def _capture() -> None:
            try:
                for block in self.source.blocks():
                    self._buffer.write(block)
            except Exception as exc:  # noqa: BLE001 - must propagate, see below
                capture_error.append(exc)

        thread = threading.Thread(target=_capture, daemon=True)
        thread.start()

        last_stride_boundary = 0
        try:
            while True:
                if capture_error:
                    raise capture_error[0]

                samples_written = self._buffer.samples_written
                if (
                    self.listen_for_samples is not None
                    and samples_written >= self.listen_for_samples
                ):
                    return "listen_for_expired"

                delta = samples_written - last_stride_boundary
                if delta >= self.stride_samples:
                    strides_advanced = delta // self.stride_samples
                    if strides_advanced > 1:
                        self._dropped += strides_advanced - 1
                    last_stride_boundary += strides_advanced * self.stride_samples
                    self._evaluate_window(last_stride_boundary, strides_elapsed=strides_advanced)
                    continue

                if not thread.is_alive():
                    return "source_eof"

                time.sleep(self.poll_interval_s)
        finally:
            thread.join(timeout=2.0)

    def _evaluate_window(self, samples_seen: int, strides_elapsed: int = 1) -> None:
        """`strides_elapsed` is normally 1 (lockstep mode always advances
        exactly one stride per evaluation). Realtime mode passes the
        actual number of stride boundaries crossed since the last
        evaluation (>1 when the loop fell behind and skipped-and-counted)
        so the `Debouncer`'s sample-counted cooldown still tracks real
        elapsed time -- ticking by a fixed one stride regardless of drops
        would inflate the effective refractory period by the drop
        factor."""
        self._window_index += 1
        self._debouncer.tick(strides_elapsed * self.stride_samples)

        waveform = self._buffer.snapshot()
        period_request = getattr(self.policy, "period_request", None)
        t_gate = time.perf_counter()
        if period_request is not None:
            request = period_request(samples_seen, waveform)
            if request is None:
                return
            waveform = self._buffer.snapshot_range(request.start_samples, request.end_samples)
        t_decode = time.perf_counter()
        logp = np.asarray(self.backend.logp_for_waveform(waveform))
        result = decode(
            logp,
            self.grammar,
            threshold=NEG_INF,
            beam_width=self.beam_width,
            required_command_margin=self.required_command_margin,
            score_mode=self.score_mode,
        )

        obs = WindowObservation(
            window_index=self._window_index, samples_seen=samples_seen, result=result,
            waveform=waveform, logp=logp
        )
        decision = self.policy.observe(obs)
        t_done = time.perf_counter()
        period_closed = getattr(self.policy, "period_closed", None)
        if period_request is not None and period_closed is not None:
            period_closed(samples_seen, decision)
        authoritative = decision.result if decision.result is not None else result
        emitted = self._debouncer.gate(decision.accept)

        if not emitted and not self.log_all_windows:
            return

        confidence = None if authoritative.intent is None else authoritative.confidence
        payload = {
            "event": "trigger" if emitted else "window",
            "t_seconds": samples_seen / SAMPLE_RATE,
            "window_index": obs.window_index,
            "intent": authoritative.intent,
            "slots": authoritative.slots,
            "text": authoritative.text,
            "confidence": confidence,
            "policy_reason": decision.reason,
        }
        if self.log_timing:
            payload["gate_ms"] = round(1000 * (t_decode - t_gate), 3)
            payload["decode_ms"] = round(1000 * (t_done - t_decode), 3)
        # stdout is a pipe in the live app-pipeline. Without an explicit
        # flush, its block buffer can hold accepted commands until the
        # streaming process exits, so app.forward never sees them promptly.
        print(json.dumps(payload), file=self._out, flush=True)

        if emitted:
            self._events += 1
            self.triggers.append(TriggerEvent(**{k: v for k, v in payload.items() if k not in ("gate_ms", "decode_ms")}))
