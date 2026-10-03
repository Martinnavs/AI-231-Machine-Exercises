"""The acceptance-policy seam: decides whether one window's decoded
evidence is "real" before the debouncer ever gets a say. See
`ME2/docs/STREAMING-CONTRACT.md` for the fixed pipeline order this
composes into and the rationale for the evidence/emission-rate split.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Callable, Optional, Protocol

import numpy as np

from ...common.features import SAMPLE_RATE
from me2_voicegen.vcm.alphabet import BLANK_ID
from .gate import ListeningGate
from me2_voicegen.vcm.decoder import DecodeResult


@dataclass(frozen=True)
class WindowObservation:
    window_index: int
    samples_seen: int
    result: DecodeResult  # decoded at threshold=-inf
    waveform: Optional[np.ndarray] = None  # window audio (ring-buffer snapshot) for the gate
    logp: Optional[np.ndarray] = None  # (T, alphabet) log-posteriors the result was decoded from


@dataclass(frozen=True)
class PolicyDecision:
    accept: bool
    reason: str  # surfaced by --log-all-windows
    result: Optional[DecodeResult] = None  # authoritative override; None = window's own result


@dataclass(frozen=True)
class PeriodRequest:
    start_samples: int
    end_samples: int


# Optional period-lifecycle sink (the CLI's `--log-periods` digest wires a
# printer here): called with (event, samples_seen, decision), where event is
# "open" (a press), "reopened" (a mid-period re-press), or "closed" (the
# flush -- the decision is the period's consolidated result).
PeriodEventCallback = Callable[[str, int, Optional[PolicyDecision]], None]


class AcceptancePolicy(Protocol):
    def observe(self, obs: WindowObservation) -> PolicyDecision: ...
    def reset(self) -> None: ...


class ThresholdPolicy:
    """The ONE shipped `AcceptancePolicy` implementation: accept iff
    `result.intent is not None and result.confidence >= threshold` --
    exactly `vcm.evaluate._accepted`'s body. Stateless (ignores window
    history); `observe`/`reset` are still stateful-shaped per the
    `AcceptancePolicy` protocol so a future history-based policy (e.g.
    multi-window smoothing, N-consecutive voting, active-region scoring)
    can be swapped in without changing the protocol.
    """

    def __init__(self, threshold: float) -> None:
        self.threshold = threshold

    def observe(self, obs: WindowObservation) -> PolicyDecision:
        result = obs.result
        if result.intent is None:
            return PolicyDecision(accept=False, reason="no_match: intent is None")
        if result.confidence < self.threshold:
            return PolicyDecision(
                accept=False,
                reason=(
                    f"confidence {result.confidence!r} below threshold "
                    f"{self.threshold!r}"
                ),
            )
        return PolicyDecision(
            accept=True,
            reason=(
                f"intent={result.intent!r} confidence={result.confidence!r} "
                f">= threshold {self.threshold!r}"
            ),
        )

    def reset(self) -> None:
        return None


class ModePeriodPolicy:
    """`AcceptancePolicy` that accepts at most once per listening period:
    collects every observation the gate's period covers, then decides at
    the period's end from the mode (most frequent `(intent, slots)`
    decode) across it, comparing the mode's mean confidence to the
    operating threshold. Winner ties break by count, then confidence
    sum, then first-seen order -- deterministic, no randomness. A press
    landing on the exact stride a period ends returns the flush decision
    for that observation AND seeds the new period with the same
    observation (SPEC observe algorithm: step 2 and step 3 both apply to
    that observation). An optional `on_period_event` sink is called with
    the period-lifecycle events ("open" at a press, "reopened" at a
    mid-period re-press, "closed" at the flush with the decision) -- the
    CLI's `--log-periods` digest wires a printer here; default `None`
    keeps behavior identical."""

    def __init__(
        self,
        threshold: float,
        *,
        gate: ListeningGate,
        period_s: float,
        on_period_event: Optional[PeriodEventCallback] = None,
    ) -> None:
        self.threshold = threshold
        self._gate = gate
        self._period_samples = int(period_s * SAMPLE_RATE)
        self._collected: list[WindowObservation] = []
        self._open_at: Optional[int] = None
        self._on_period_event = on_period_event

    def observe(self, obs: WindowObservation) -> PolicyDecision:
        state = self._gate.poll(obs.samples_seen, obs.waveform)

        if self._collected and obs.samples_seen >= self._open_at + self._period_samples:
            decision = self._flush()
            self._emit_period_event("closed", obs.samples_seen, decision)
            if state.is_open:
                # A press that landed on this exact flush stride: the same
                # observation also seeds the new period's collection.
                self._open_at = state.open_at_samples
                self._collected = [obs]
                self._emit_period_event("open", obs.samples_seen, None)
            return decision

        if state.is_open:
            if self._collected and state.open_at_samples != self._open_at:
                self._collected = []  # re-press: discard in-flight, restart
                self._emit_period_event("reopened", obs.samples_seen, None)
            elif not self._collected:
                self._emit_period_event("open", obs.samples_seen, None)
            self._open_at = state.open_at_samples
            self._collected.append(obs)
            return self._collecting_decision()

        return PolicyDecision(accept=False, reason="gate closed (not in a listening period)")

    def reset(self) -> None:
        self._collected = []
        self._open_at = None

    def _emit_period_event(
        self, event: str, samples_seen: int, decision: Optional[PolicyDecision]
    ) -> None:
        if self._on_period_event is not None:
            self._on_period_event(event, samples_seen, decision)

    def _flush(self) -> PolicyDecision:
        collected = self._collected
        grouped = self._group_by_decode_class(collected)
        winner_key = self._winner_key(grouped)
        _first, count, total_conf, latest = grouped[winner_key]
        self._collected = []

        if winner_key[0] is None:
            return PolicyDecision(
                accept=False,
                reason=f"mode_period: silence dominated the period (None {count}/{len(collected)})",
            )

        mean = total_conf / count
        result = replace(latest.result, confidence=mean)
        if mean >= self.threshold:
            return PolicyDecision(
                accept=True,
                reason=(
                    f"mode_period: intent={winner_key[0]!r} mean confidence {mean!r} "
                    f">= threshold {self.threshold!r} over {count} obs"
                ),
                result=result,
            )
        return PolicyDecision(
            accept=False,
            reason=(
                f"mode_period: intent={winner_key[0]!r} mean confidence {mean!r} "
                f"below threshold {self.threshold!r} over {count} obs"
            ),
            result=result,
        )

    def _collecting_decision(self) -> PolicyDecision:
        grouped = self._group_by_decode_class(self._collected)
        winner_key = self._winner_key(grouped)
        _first, count, _total, _latest = grouped[winner_key]
        n = len(self._collected)
        return PolicyDecision(
            accept=False,
            reason=f"collecting ({n} obs, running mode {winner_key[0]!r} {count}/{n})",
        )

    @staticmethod
    def _group_by_decode_class(collected):
        """Group observations by their decode class `(intent, sorted-slots)`;
        returns key -> (first_seen_index, count, confidence_sum, latest_obs)."""
        grouped = {}
        for index, obs in enumerate(collected):
            key = (obs.result.intent, tuple(sorted(obs.result.slots.items())))
            first, count, total_conf, latest = grouped.get(key, (index, 0, 0.0, None))
            grouped[key] = (first, count + 1, total_conf + obs.result.confidence, obs)
        return grouped

    @staticmethod
    def _winner_key(grouped):
        return min(grouped.items(), key=lambda item: (-item[1][1], -item[1][2], item[1][0]))[0]


class SinglePeriodPolicy(ThresholdPolicy):
    """One exact gate-bounded waveform -> one decode -> threshold decision."""

    def __init__(self, threshold: float, *, gate: ListeningGate, period_s: float, on_period_event=None) -> None:
        super().__init__(threshold)
        self._gate = gate
        self.period_samples = int(period_s * SAMPLE_RATE)
        self._open_at: Optional[int] = None
        self._on_period_event = on_period_event

    def period_request(self, samples_seen: int, waveform: np.ndarray) -> Optional[PeriodRequest]:
        state = self._gate.poll(samples_seen, waveform)
        if self._open_at is not None and samples_seen >= self._open_at + self.period_samples:
            request = PeriodRequest(self._open_at, self._open_at + self.period_samples)
            self._open_at = None
            if state.is_open:
                self._open_at = state.open_at_samples
                if self._on_period_event is not None:
                    self._on_period_event("open", samples_seen, None)
            return request
        if state.is_open:
            event = "open" if self._open_at is None else "reopened"
            self._open_at = state.open_at_samples
            if self._on_period_event is not None:
                self._on_period_event(event, samples_seen, None)
        return None

    def period_closed(self, samples_seen: int, decision: PolicyDecision) -> None:
        if self._on_period_event is not None:
            self._on_period_event("closed", samples_seen, decision)

    def reset(self) -> None:
        self._open_at = None


class EndpointedPeriodPolicy(ThresholdPolicy):
    """Wake-word period with a sliding decode and an early, endpointed exit.

    The gate opening anchors a period. Every stride while it is active, the
    policy requests a decode of the audio from the anchor to now (growing to
    `window_s`, then sliding) and accepts the first window that is
    (1) confident under `ThresholdPolicy`, (2) the same `(intent, slots)` for
    `stable_strides` consecutive decodes, and (3) ended: the trailing `hold_ms`
    of posteriors are all blank with probability >= `blank_floor`. Accepting
    closes the period at once; reaching `period_s` after the last wake-word
    detection closes it with no output. The period length is therefore a
    time-out, not a wait, measured like `SinglePeriodPolicy`'s period (from
    the gate's latest open time), so it never ends earlier than that one.

    The anchor is the gate's first open time for an episode. Later re-arms
    (the wake-word gate re-fires while the wake word is still in its trailing
    window) are ignored so the window start does not drift into the command.
    After a period closes, a new one needs a fresh detection: the gate must
    first stop re-arming (the old wake word has left its window), then open
    again after the close. Without this, the same wake word reopens a period
    right after a fast accept and the command can fire twice.
    """

    def __init__(
        self,
        threshold: float,
        *,
        gate: ListeningGate,
        period_s: float,
        window_s: float = 2.5,
        min_audio_s: float = 0.3,
        stable_strides: int = 2,
        hold_ms: float = 300.0,
        blank_floor: float = 0.9,
        on_period_event: Optional[PeriodEventCallback] = None,
        fallback: Optional[Callable[[np.ndarray], Optional[DecodeResult]]] = None,
        fallback_hold_ms: float = 500.0,
        fallback_min_speech_ms: float = 200.0,
    ) -> None:
        super().__init__(threshold)
        if stable_strides < 1:
            raise ValueError(f"stable_strides must be >= 1, got {stable_strides}")
        if not 0.0 < blank_floor <= 1.0:
            raise ValueError(f"blank_floor must be in (0, 1], got {blank_floor}")
        if hold_ms < 0 or min_audio_s < 0:
            raise ValueError("hold_ms and min_audio_s must be >= 0")
        self._gate = gate
        self.period_samples = int(period_s * SAMPLE_RATE)
        self.window_samples = int(window_s * SAMPLE_RATE)
        self.min_audio_samples = int(min_audio_s * SAMPLE_RATE)
        self.stable_strides = stable_strides
        self.hold_s = hold_ms / 1000.0
        self.log_blank_floor = float(np.log(blank_floor))
        self._on_period_event = on_period_event
        # Hybrid (docs/AI231-FIL50.md): when the CTC decode has not accepted, the speech has ended (a longer hold than the
        # CTC's) and the window holds enough non-blank frames, ask `fallback` (the classifier heads) once per window.
        self._fallback = fallback
        self.fallback_hold_s = fallback_hold_ms / 1000.0
        self.fallback_min_speech_s = fallback_min_speech_ms / 1000.0
        self.reset()

    def reset(self) -> None:
        self._anchor: Optional[int] = None
        self._deadline_from: int = 0
        self._closed_at: int = -1
        self._armed = True
        self._last_open: Optional[int] = None
        self._last_key = None
        self._streak = 0

    def _event(self, event: str, samples_seen: int, decision: Optional[PolicyDecision]) -> None:
        if self._on_period_event is not None:
            self._on_period_event(event, samples_seen, decision)

    def _close(self, samples_seen: int, decision: PolicyDecision) -> None:
        self._anchor = None
        self._closed_at = samples_seen
        self._armed = False
        self._last_key = None
        self._streak = 0
        self._event("closed", samples_seen, decision)

    def period_request(self, samples_seen: int, waveform: np.ndarray) -> Optional[PeriodRequest]:
        state = self._gate.poll(samples_seen, waveform)
        opened = state.open_at_samples if state.is_open else None
        rearmed = opened is not None and opened != self._last_open
        self._last_open = opened
        if self._anchor is None:
            if not self._armed:
                if rearmed and not getattr(self._gate, "continuous", False):
                    # The closed period's wake word is still being detected:
                    # these detections belong to that period, not a new one.
                    self._closed_at = max(self._closed_at, opened)
                    return None
                self._armed = True
            if opened is None or opened <= self._closed_at:
                return None
            self._anchor = opened
            self._deadline_from = opened
            self._event("open", samples_seen, None)
        elif rearmed:
            self._deadline_from = opened  # wake word still heard: time-out counts from here
        if samples_seen >= self._deadline_from + self.period_samples:
            self._close(samples_seen, PolicyDecision(accept=False, reason="endpointed: timed out"))
            return None
        if samples_seen - self._anchor < self.min_audio_samples:
            return None
        return PeriodRequest(max(self._anchor, samples_seen - self.window_samples), samples_seen)

    def _ended(self, obs: WindowObservation, hold_s: Optional[float] = None) -> bool:
        if obs.logp is None or obs.waveform is None or obs.logp.shape[0] == 0:
            return False
        n_frames = obs.logp.shape[0]
        frame_s = obs.waveform.shape[-1] / SAMPLE_RATE / n_frames
        hold_frames = max(1, int(np.ceil((self.hold_s if hold_s is None else hold_s) / frame_s - 1e-9)))
        if hold_frames > n_frames:
            return False
        return bool(np.all(obs.logp[-hold_frames:, BLANK_ID] >= self.log_blank_floor))

    def _speech_s(self, obs: WindowObservation) -> float:
        if obs.logp is None or obs.waveform is None or obs.logp.shape[0] == 0:
            return 0.0
        frame_s = obs.waveform.shape[-1] / SAMPLE_RATE / obs.logp.shape[0]
        return float(np.sum(obs.logp[:, BLANK_ID] < self.log_blank_floor)) * frame_s

    def _try_fallback(self, obs: WindowObservation, base: PolicyDecision) -> PolicyDecision:
        if self._fallback is None or not self._ended(obs, self.fallback_hold_s) or self._speech_s(obs) < self.fallback_min_speech_s:
            return base
        result = self._fallback(obs.waveform)
        if result is None:
            return PolicyDecision(accept=False, reason=f"{base.reason}; classifier fallback rejected")
        return PolicyDecision(accept=True, reason=f"endpointed: classifier fallback ({base.reason})", result=result)

    def observe(self, obs: WindowObservation) -> PolicyDecision:
        base = super().observe(obs)
        if not base.accept:
            self._last_key, self._streak = None, 0
            return self._try_fallback(obs, base)
        key = (obs.result.intent, tuple(sorted(obs.result.slots.items())))
        self._streak = self._streak + 1 if key == self._last_key else 1
        self._last_key = key
        if self._streak < self.stable_strides:
            return PolicyDecision(
                accept=False, reason=f"endpointed: stable {self._streak}/{self.stable_strides}"
            )
        if not self._ended(obs):
            return PolicyDecision(accept=False, reason="endpointed: speech not ended")
        return PolicyDecision(accept=True, reason=f"endpointed: {base.reason}")

    def period_closed(self, samples_seen: int, decision: PolicyDecision) -> None:
        if decision.accept:
            self._close(samples_seen, decision)
