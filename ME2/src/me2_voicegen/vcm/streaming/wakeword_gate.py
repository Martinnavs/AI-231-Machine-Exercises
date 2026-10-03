"""`WakeWordGate`: the trained DS-CNN wakeword classifier (feature
`wakeword-dscnn`) wired into the `ListeningGate` seam (feature
`wakeword-gate`, `feature-engineering/wakeword-gate/SPEC.md`), in place of
`SpacebarGate`'s manual keypress.

`WakewordInferenceBackend` is deliberately a separate protocol from
`backends.InferenceBackend`: the VCM backend is CTC-shaped
(`logp_for_waveform(waveform) -> (T, alphabet_size)`), while the wakeword
model is a 3-way clip classifier (`wakeword.model.DSCNN`, labels
`_wakeword_`/`_unknown_`/`_silence_`) -- one probability in, not a
per-frame sequence out.

SECURITY -- same trust boundary as `backends.py`: `WakewordTorchBackend`
loads its checkpoint via `torch.load(..., weights_only=True)` and refuses
outright rather than silently falling back to the unsafe
`weights_only=False` path (see `backends.py`'s module docstring for the
full rationale -- this mirrors it exactly, not `export_onnx.load_checkpoint`,
which does a plain `torch.load` and is not used here). `WakewordOnnxBackend`
loads via `onnxruntime.InferenceSession`, which is equally untrusted-input
parsing for a `.onnx` file you didn't produce or trust. Only ever point
either backend at a checkpoint/export you produced yourself or otherwise
trust.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Protocol

import numpy as np
import torch

from me2_voicegen.common.features import LogMelFeatureExtractor, SAMPLE_RATE
from me2_voicegen.wakeword.augment import center_window
from me2_voicegen.wakeword.dataset import WAKEWORD_WINDOW_SECONDS
from me2_voicegen.wakeword.model import DSCNN, DSCNNConfig, LABEL_TO_ID

from .gate import DEFAULT_WAKEWORD_THRESHOLD, GateState, ListeningGate

WAKEWORD_INDEX: int = LABEL_TO_ID["_wakeword_"]
WAKEWORD_WINDOW_SAMPLES: int = int(round(WAKEWORD_WINDOW_SECONDS * SAMPLE_RATE))


class WakewordInferenceBackend(Protocol):
    def wakeword_prob(self, waveform: np.ndarray) -> float:
        """`(samples,)` float32 waveform at `common.features.SAMPLE_RATE`,
        already cropped/padded to `WAKEWORD_WINDOW_SAMPLES` ->
        softmax probability of the `_wakeword_` class."""
        ...


def _softmax_wakeword_prob(logits: np.ndarray) -> float:
    """`logits`: `(3,)` raw DS-CNN output for one clip -> softmax
    probability at `WAKEWORD_INDEX`. Numerically-stable softmax, mirrors
    `backends._log_softmax`'s stability approach without the log."""
    shifted = logits - np.max(logits)
    exp = np.exp(shifted)
    probs = exp / np.sum(exp)
    return float(probs[WAKEWORD_INDEX])


class WakewordOnnxBackend:
    """`session.run` over a fixed-shape `(1, 3)`-output ONNX export
    (`vcm.export_onnx --model-family wakeword`)."""

    def __init__(self, model_path: str | Path, ort_threads: int = 1) -> None:
        import onnxruntime as ort

        so = ort.SessionOptions()
        so.intra_op_num_threads = ort_threads
        so.inter_op_num_threads = 1
        self._session = ort.InferenceSession(
            str(model_path), sess_options=so, providers=["CPUExecutionProvider"]
        )
        self._feature_extractor = LogMelFeatureExtractor()

    def wakeword_prob(self, waveform: np.ndarray) -> float:
        wav_t = torch.as_tensor(waveform, dtype=torch.float32)
        features = self._feature_extractor(wav_t).unsqueeze(0).numpy().astype(np.float32)
        (logits,) = self._session.run(None, {"features": features})
        return _softmax_wakeword_prob(logits[0])


class WakewordTorchBackend:
    """Wraps an in-process torch `DSCNN` built from the checkpoint's own
    `preset`/`config` metadata. `torch.load(..., weights_only=True)` --
    see this module's docstring; a checkpoint that fails to load under
    `weights_only=True` raises rather than silently retrying with full
    pickle deserialization."""

    def __init__(self, checkpoint_path: str | Path, device: str = "cpu") -> None:
        self.device = device
        checkpoint = torch.load(
            str(checkpoint_path), map_location=device, weights_only=True
        )
        config = DSCNNConfig(**checkpoint["config"])
        self.model = DSCNN(config)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.to(device)
        self.model.eval()
        self._feature_extractor = LogMelFeatureExtractor()

    def wakeword_prob(self, waveform: np.ndarray) -> float:
        wav_t = torch.as_tensor(waveform, dtype=torch.float32)  # the feature extractor lives on the CPU
        features = self._feature_extractor(wav_t).unsqueeze(0).to(self.device)
        with torch.no_grad():
            logits = self.model(features)
        return _softmax_wakeword_prob(logits[0].cpu().numpy())


def _trailing_wakeword_window(window: np.ndarray) -> np.ndarray:
    """The ring-buffer snapshot `poll()` receives can be several seconds
    long (up to `period_s + window_s`, see `runner.py`'s `RingBuffer`
    sizing) -- far more than the `WAKEWORD_WINDOW_SAMPLES` (1.5s) the
    classifier expects. Take the TRAILING slice (the most recently
    captured audio), never a centered crop of the whole buffer: a
    centered crop would classify stale audio from earlier in the period,
    silently breaking "did the user just say the wakeword right now".
    `wakeword.augment.center_window` is then applied only for its
    zero-pad branch (buffer not yet 1.5s full) -- a no-op once the input
    is already exactly `WAKEWORD_WINDOW_SAMPLES` long, since its
    `n >= window_samples` branch computes `start = 0` in that case."""
    if window.shape[-1] > WAKEWORD_WINDOW_SAMPLES:
        window = window[-WAKEWORD_WINDOW_SAMPLES:]
    wav_t = torch.as_tensor(window, dtype=torch.float32)
    padded = center_window(wav_t, WAKEWORD_WINDOW_SAMPLES)
    return padded.numpy()


class WakeWordGate:
    """`ListeningGate` whose open signal is the wakeword classifier
    crossing `threshold` on the trailing `WAKEWORD_WINDOW_SAMPLES` of the
    ring-buffer snapshot `poll()` receives. Same auto-close and
    discard-and-restart-on-renewed-detection semantics as `SpacebarGate`,
    so `ModePeriodPolicy`/`SinglePeriodPolicy` consume it identically."""

    def __init__(
        self,
        backend: WakewordInferenceBackend,
        *,
        threshold: float = DEFAULT_WAKEWORD_THRESHOLD,
        period_s: float = 5.0,
        poll_step_s: Optional[float] = None,
    ) -> None:
        self._backend = backend
        self._threshold = threshold
        self._period_s = period_s
        # Finer wake-word polling than the decode stride: between two `poll()` calls the classifier also scores the trailing
        # window ending every `poll_step_s` of audio, so a wake word that only fits the 1.5 s window between strides is still caught.
        self._poll_step = None if poll_step_s is None else max(1, int(poll_step_s * SAMPLE_RATE))
        self._last_poll: Optional[int] = None
        self._open_at_samples: Optional[int] = None
        self._closed = False

    def poll(self, samples_seen: int, window: Optional[np.ndarray] = None) -> GateState:
        if self._closed:
            return GateState(is_open=False, open_at_samples=None)

        if self._open_at_samples is not None and samples_seen >= (
            self._open_at_samples + int(self._period_s * SAMPLE_RATE)
        ):
            self._open_at_samples = None

        if window is not None and window.shape[-1] > 0:
            window = np.asarray(window, dtype=np.float32)
            ends = [samples_seen]
            if self._poll_step is not None and self._last_poll is not None:
                ends = list(range(self._last_poll + self._poll_step, samples_seen, self._poll_step)) + ends
            for end in ends:  # ascending: the latest detection wins, as with one poll per stride
                cut = window.shape[-1] - (samples_seen - end)
                if cut <= 0:
                    continue
                if self._backend.wakeword_prob(_trailing_wakeword_window(window[:cut])) >= self._threshold:
                    self._open_at_samples = end
            self._last_poll = samples_seen

        return GateState(
            is_open=self._open_at_samples is not None,
            open_at_samples=self._open_at_samples,
        )

    def close(self) -> None:
        """Idempotent; no OS resource held (unlike `SpacebarGate`'s
        termios), so this is just internal-state cleanup."""
        self._open_at_samples = None
        self._closed = True
