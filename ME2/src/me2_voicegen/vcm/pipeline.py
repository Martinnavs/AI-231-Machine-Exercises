"""Wav/waveform -> intent pipeline: log-mel features (`common.features`) ->
trained CTC model (`vcm.model`) -> grammar-constrained decoder
(`vcm.decoder`) -> a final `{intent, slots, confidence}` result or an
explicit rejection.

Two modes:
    - whole-clip (`infer_waveform`): one full waveform in, one
      `DecodeResult` out. This is what `vcm.evaluate` uses over the
      manifest's `test` split.
    - sliding-window (`SlidingWindowPipeline`): a minimal streaming mode --
      1.5s window, 100ms stride, simple ring-buffer semantics -- with a
      confidence threshold plus a refractory/debounce cooldown period so a
      single spoken command held across several overlapping windows isn't
      reported as several separate triggers. Sample-counted (not
      wall-clock-timed) so its debounce behavior is deterministic and
      testable without real time passing; a caller streaming real
      microphone audio at `common.features.SAMPLE_RATE` gets real-time
      behavior for free since sample count IS elapsed time at a fixed
      sample rate.

Checkpoint loading here must match the `{model_state_dict, preset, config,
alphabet_size, license, ...}` shape `vcm.train.main` writes to
`checkpoint.pt` -- see that module's `torch.save` call.
"""

from __future__ import annotations

import dataclasses
from pathlib import Path

import numpy as np
import torch

from me2_voicegen.common.features import HOP_LENGTH, SAMPLE_RATE, LogMelFeatureExtractor
from me2_voicegen.common.grammar_core import Grammar
from me2_voicegen.vcm.decoder import DecodeResult, decode
from me2_voicegen.vcm.model import MODEL_TYPE_KEY, build_model_from_config
from me2_voicegen.vcm.streaming.debounce import Debouncer

WINDOW_S = 1.5
STRIDE_S = 0.1
DEFAULT_REFRACTORY_S = 1.0

WINDOW_SAMPLES = int(round(WINDOW_S * SAMPLE_RATE))
STRIDE_SAMPLES = int(round(STRIDE_S * SAMPLE_RATE))


def frame_hop_samples(model: torch.nn.Module) -> int:
    """Waveform samples spanned by ONE output (posterior) frame of `model`.

    Every acoustic model emits one log-posterior frame per `total_stride`
    input feature frames, and one feature frame is `HOP_LENGTH` waveform
    samples (10 ms at 16 kHz). `QuartzNetCTC` exposes `total_stride`;
    `MatchboxNetCTC` has no such attribute and is stride 1, hence the
    `getattr(..., 1)`. Anything that turns a posterior frame index -- a
    `force_align` `start_frame`/`end_frame`, a `logp` row, a
    `crop_end_frame` -- into seconds or sample offsets must scale by this
    value (docs/VCM-CONTRACT.md, model-seam note)."""
    return HOP_LENGTH * getattr(model, "total_stride", 1)


def load_checkpoint(
    path: str | Path,
    device: str | torch.device = "cpu",
    weights_only: bool = False,
    allow_unsafe_load: bool = False,
) -> tuple[torch.nn.Module, dict]:
    """Load a `vcm.train`-written checkpoint. Returns `(model, checkpoint)`
    -- `model` is in `eval()` mode on `device`; `checkpoint` is the raw
    dict (carries `license`, `preset`, `val_loss`, `epoch`, etc. for report
    headers).

    `weights_only` defaults to `False`, preserving today's exact behavior
    for existing callers (`evaluate.py`, `benchmark.py`, `export_onnx.py`).
    Opt in with `weights_only=True` (as `vcm.streaming.backends.TorchBackend`
    does) to mitigate `torch.load`'s pickle-deserialization trust boundary on
    an untrusted/arbitrary `--model` path. `weights_only=True` is a
    mitigation, not a guarantee: it is a known-vulnerable configuration on
    this project's pinned `torch==2.3.1` (CVE-2025-32434 affects
    `weights_only=True` on torch <= 2.5.1; fixed in 2.6.0). Upgrading torch
    is explicitly out of scope here (a repo-wide pinned CUDA stack) and is
    tracked as a separate backlog item, not addressed by this function.

    A checkpoint that fails to load under `weights_only=True` is exactly
    the attack signature `weights_only=True` exists to catch, not merely an
    edge case -- so this does NOT automatically fall back to the unsafe
    `weights_only=False` path. It only does so if the caller also passes
    `allow_unsafe_load=True`, an explicit second opt-in on top of
    `weights_only=True` meaning "I trust this specific checkpoint enough to
    accept full pickle deserialization if the safe path rejects it."
    Without it, a `weights_only=True` failure is raised as an actionable
    `RuntimeError` naming the escape hatch, not silently downgraded.
    `vcm.streaming.backends.TorchBackend` calls with
    `allow_unsafe_load=False` (the default) -- a checkpoint failing
    `weights_only=True` there is a hard error, not a silent unsafe retry.
    See `docs/STREAMING-CONTRACT.md`."""
    if weights_only:
        try:
            checkpoint = torch.load(path, map_location=device, weights_only=True)
        except Exception as exc:
            import sys

            if not allow_unsafe_load:
                raise RuntimeError(
                    f"torch.load(..., weights_only=True) failed on {path} "
                    f"({exc!r}). This is refused rather than silently retried "
                    f"with weights_only=False, since a checkpoint failing the "
                    f"weights_only allow-list is the attack signature that "
                    f"mitigation exists to catch. If you produced this "
                    f"checkpoint yourself or otherwise trust it, re-call with "
                    f"allow_unsafe_load=True to explicitly accept full pickle "
                    f"deserialization."
                ) from exc

            print(
                f"warning: torch.load(..., weights_only=True) failed on {path} "
                f"({exc!r}); falling back to weights_only=False because "
                f"allow_unsafe_load=True was explicitly passed. Only load "
                f"checkpoints you produced yourself or otherwise trust.",
                file=sys.stderr,
            )
            checkpoint = torch.load(path, map_location=device)
    else:
        checkpoint = torch.load(path, map_location=device)
    model = build_model_from_config(checkpoint["config"], checkpoint.get(MODEL_TYPE_KEY))
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()
    return model, checkpoint


@torch.no_grad()
def logp_for_waveform(
    model: torch.nn.Module,
    feature_extractor: LogMelFeatureExtractor,
    waveform: torch.Tensor,
    device: str | torch.device = "cpu",
) -> np.ndarray:
    """waveform (samples,) -> (T, alphabet_size) log-posterior numpy array,
    the decoder's input contract (docs/VCM-CONTRACT.md section 7)."""
    features = feature_extractor(waveform).unsqueeze(0).to(device)
    logits = model(features)
    log_probs = logits.log_softmax(dim=-1)[0]
    return log_probs.cpu().numpy()


@torch.no_grad()
def infer_waveform(
    model: torch.nn.Module,
    feature_extractor: LogMelFeatureExtractor,
    waveform: torch.Tensor,
    grammar: Grammar,
    threshold: float,
    beam_width: int = 50,
    device: str | torch.device = "cpu",
    required_command_margin: float | None = None,
    score_mode: str = "mean_frame",
) -> DecodeResult:
    """Whole-clip mode: one waveform -> one `DecodeResult`. Used by
    `vcm.evaluate` over the manifest's `test`/`val` splits.
    `required_command_margin` (docs/INCOMPLETE-GRAMMAR-REJECTION.md, Step 3)
    is the incomplete-prefix rejection gate margin, forwarded verbatim to
    `decode`; `None` (default) leaves the gate disabled. `score_mode`
    (`"mean_frame"` default | `"per_char"`, see `decoder.decode_utterance`) is
    likewise forwarded verbatim."""
    logp = logp_for_waveform(model, feature_extractor, waveform, device=device)
    return decode(
        logp,
        grammar,
        threshold,
        beam_width=beam_width,
        required_command_margin=required_command_margin,
        score_mode=score_mode,
    )


@dataclasses.dataclass
class TriggerEvent:
    """One non-suppressed accepted trigger from `SlidingWindowPipeline.feed`."""

    window_index: int
    samples_seen: int
    result: DecodeResult


class SlidingWindowPipeline:
    """Minimal streaming wrapper: feed it waveform chunks of any length,
    get back any newly-triggered (non-suppressed) `TriggerEvent`s.

    Ring-buffer semantics: an internal buffer holds at most the last
    `window_samples` samples fed so far (shorter than that only at
    startup, before the buffer has filled once). Every `stride_samples`
    of newly-fed audio, the pipeline runs one whole-clip-style inference
    over the current buffer contents. An accepted (non-`no_match`) result
    only becomes a `TriggerEvent` if the refractory/debounce cooldown has
    fully elapsed since the last trigger; otherwise it is suppressed
    (counted, not emitted) -- this is what stops one held command spanning
    several overlapping windows from firing repeatedly.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        feature_extractor: LogMelFeatureExtractor,
        grammar: Grammar,
        threshold: float,
        beam_width: int = 50,
        window_s: float = WINDOW_S,
        stride_s: float = STRIDE_S,
        refractory_s: float = DEFAULT_REFRACTORY_S,
        device: str | torch.device = "cpu",
        required_command_margin: float | None = None,
        score_mode: str = "mean_frame",
    ) -> None:
        self.model = model
        self.feature_extractor = feature_extractor
        self.grammar = grammar
        self.threshold = threshold
        self.beam_width = beam_width
        self.window_samples = int(round(window_s * SAMPLE_RATE))
        self.stride_samples = int(round(stride_s * SAMPLE_RATE))
        self.refractory_samples = int(round(refractory_s * SAMPLE_RATE))
        self.device = device
        self.required_command_margin = required_command_margin
        self.score_mode = score_mode

        self._buffer = torch.zeros(0)
        self._samples_since_last_window = 0
        self._debouncer = Debouncer(self.refractory_samples)
        self._window_index = 0
        self._samples_seen = 0

    @property
    def suppressed_count(self) -> int:
        return self._debouncer.suppressed_count

    @property
    def _cooldown_samples_remaining(self) -> int:
        return self._debouncer.cooldown_samples_remaining

    def reset(self) -> None:
        self._buffer = torch.zeros(0)
        self._samples_since_last_window = 0
        self._debouncer.reset()
        self._window_index = 0
        self._samples_seen = 0

    def feed(self, chunk: torch.Tensor) -> list[TriggerEvent]:
        """Append `chunk` (1-D waveform tensor, any length > 0) to the ring
        buffer and evaluate once per `stride_samples` of newly-arrived
        audio. Returns the list of newly-emitted (non-suppressed)
        `TriggerEvent`s from this call (usually 0 or 1, but a large chunk
        spanning several strides can yield more than one)."""
        if chunk.numel() == 0:
            return []
        chunk = chunk.reshape(-1)

        events: list[TriggerEvent] = []
        self._buffer = torch.cat([self._buffer, chunk])
        if self._buffer.numel() > self.window_samples:
            self._buffer = self._buffer[-self.window_samples :]
        self._samples_since_last_window += chunk.numel()
        self._samples_seen += chunk.numel()

        while self._samples_since_last_window >= self.stride_samples:
            self._samples_since_last_window -= self.stride_samples
            self._window_index += 1
            self._debouncer.tick(self.stride_samples)

            result = infer_waveform(
                self.model,
                self.feature_extractor,
                self._buffer,
                self.grammar,
                self.threshold,
                beam_width=self.beam_width,
                device=self.device,
                required_command_margin=self.required_command_margin,
                score_mode=self.score_mode,
            )

            if self._debouncer.gate(not result.no_match):
                events.append(
                    TriggerEvent(
                        window_index=self._window_index,
                        samples_seen=self._samples_seen,
                        result=result,
                    )
                )

        return events
