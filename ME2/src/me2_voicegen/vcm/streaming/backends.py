"""`InferenceBackend` adapters: "torch checkpoint" and "ONNX export" behind
one call shape (`docs/STREAMING-CONTRACT.md` section 3), so the runner loop
and CLI don't care which backend produced a window's `logp`.

SECURITY -- trust boundary, read before pointing either backend at a path
you didn't produce yourself:

- `TorchBackend` loads its checkpoint via `vcm.pipeline.load_checkpoint(...,
  weights_only=True, allow_unsafe_load=False)` -- a mitigation for
  `torch.load`'s pickle-based deserialization (arbitrary code execution on
  a malicious `.pt` file), not a full sandbox, and NOT a guarantee: this
  project's pinned `torch==2.3.1` is within CVE-2025-32434's affected range
  (a known bypass of `weights_only=True` on torch <= 2.5.1, fixed in 2.6.0
  -- upgrading torch is out of scope here, tracked separately). A
  checkpoint that fails `weights_only=True` is refused outright
  (`allow_unsafe_load=False` means no automatic silent fallback to the
  unsafe `weights_only=False` path -- see `load_checkpoint`'s docstring)
  rather than treated as a benign edge case, since failing the allow-list
  is itself the attack signature this mitigation exists to catch.
  `weights_only=True` is verified (see `docs/STREAMING-CONTRACT.md`/this
  ticket's Execution Log) to load both of this project's real checkpoints,
  which carry non-tensor metadata (`preset`, `config` dict, `license` str,
  `epoch`, `val_loss`) alongside their tensors, so this refusal path is not
  exercised by either shipped checkpoint today.
- `OnnxBackend` loads its model via `onnxruntime.InferenceSession`, which is
  ALSO parsing untrusted input if the `.onnx` file isn't one you produced or
  trust -- ORT's model parser is not a sandboxed/verified-safe format any
  more than pickle is. `providers=["CPUExecutionProvider"]` is a resource
  scoping choice (this project ships CPU-only), not a security boundary.

Neither backend applies path-containment (`resolve_under()`-style) to
`model_path` -- see `config.py`'s module docstring for why that pattern is
the wrong tool for this specific input. Only ever point either backend at a
checkpoint/`.onnx` file you produced yourself or otherwise trust.
"""

from __future__ import annotations

from pathlib import Path
from typing import Protocol

import numpy as np
import torch

from me2_voicegen.common.features import LogMelFeatureExtractor
from me2_voicegen.vcm.pipeline import load_checkpoint, logp_for_waveform


class InferenceBackend(Protocol):
    def logp_for_waveform(self, waveform: np.ndarray) -> np.ndarray:
        """(samples,) float32 waveform at `common.features.SAMPLE_RATE` ->
        (T, alphabet_size) log-posterior numpy array -- the same contract as
        `vcm.pipeline.logp_for_waveform` (`docs/VCM-CONTRACT.md` section 7)."""
        ...


def _log_softmax(logits: np.ndarray, axis: int = -1) -> np.ndarray:
    """Numerically-stable log-softmax in numpy. ONNX Runtime returns raw
    logits, not log-probs (`vcm.pipeline.logp_for_waveform` applies
    `log_softmax` on the torch side) -- easy to get wrong by silently
    feeding logits straight to the decoder, so this is applied explicitly,
    deliberately, here."""
    shifted = logits - np.max(logits, axis=axis, keepdims=True)
    log_sum_exp = np.log(np.sum(np.exp(shifted), axis=axis, keepdims=True))
    return shifted - log_sum_exp


class OnnxBackend:
    """`session.run` over a fixed-alphabet ONNX export
    (`vcm.export_onnx`). Preset metadata for the startup banner is not
    carried by the `.onnx` file itself -- pass the run dir's
    `eval_report.json`-derived metadata separately (see `config.py`)."""

    def __init__(self, model_path: str | Path, ort_threads: int = 1) -> None:
        import onnxruntime as ort

        so = ort.SessionOptions()
        so.intra_op_num_threads = ort_threads
        so.inter_op_num_threads = 1
        self._session = ort.InferenceSession(
            str(model_path), sess_options=so, providers=["CPUExecutionProvider"]
        )
        self._feature_extractor = LogMelFeatureExtractor()

    def logp_for_waveform(self, waveform: np.ndarray) -> np.ndarray:
        wav_t = torch.as_tensor(waveform, dtype=torch.float32)
        features = self._feature_extractor(wav_t).unsqueeze(0).numpy().astype(np.float32)
        (logits,) = self._session.run(None, {"features": features})
        return _log_softmax(logits[0], axis=-1)


class TorchBackend:
    """Wraps `vcm.pipeline.logp_for_waveform` over an in-process torch
    checkpoint. Uses `load_checkpoint(..., weights_only=True,
    allow_unsafe_load=False)` -- see this module's docstring. A checkpoint
    that fails to load under `weights_only=True` raises rather than
    silently retrying with full pickle deserialization."""

    def __init__(self, checkpoint_path: str | Path, device: str = "cpu") -> None:
        self.device = device
        self.model, self.checkpoint = load_checkpoint(
            checkpoint_path, device=device, weights_only=True, allow_unsafe_load=False
        )
        self._feature_extractor = LogMelFeatureExtractor()

    def logp_for_waveform(self, waveform: np.ndarray) -> np.ndarray:
        wav_t = torch.as_tensor(waveform, dtype=torch.float32)
        return logp_for_waveform(
            self.model, self._feature_extractor, wav_t, device=self.device
        )
