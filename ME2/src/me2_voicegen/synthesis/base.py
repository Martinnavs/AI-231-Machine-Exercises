from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import soundfile as sf


@dataclass(frozen=True)
class VoicePrompt:
    """A reference clip for voice-cloning backends. text is Optional because some
    backends (e.g. CosyVoice2's own inference_cross_lingual) don't need a transcript
    of the prompt audio, only the audio itself."""

    wav_path: Path
    text: str | None = None


@dataclass(frozen=True)
class SynthesisResult:
    audio: np.ndarray  # float32, shape (channels, samples) - never a torch.Tensor
    sample_rate: int


class Synthesizer(ABC):
    """Backend-agnostic contract for turning text into a SynthesisResult.

    Backend configuration (model dir, device, fp16, ...) is supplied at
    construction time via the concrete implementation's __init__, not here -
    synthesize() only ever needs text (+ optional prompt) so callers stay
    decoupled from any one backend's config shape.
    """

    @abstractmethod
    def synthesize(self, text: str, prompt: VoicePrompt | None = None) -> SynthesisResult:
        """prompt is Optional because a non-cloning backend (e.g. Piper, or
        CosyVoice's own inference_sft) has no reference clip to work from. A
        backend that requires one must raise a clear error when prompt is None,
        rather than silently degrading."""
        raise NotImplementedError


def save_wav(result: SynthesisResult, path: Path) -> None:
    """Write a SynthesisResult to disk via soundfile. Kept torch-free so it works
    for any future backend, cloning or not."""
    path.parent.mkdir(parents=True, exist_ok=True)
    audio = result.audio
    # soundfile wants (frames,) for mono or (frames, channels) for multi-channel -
    # the reverse layout of SynthesisResult.audio's (channels, samples).
    data = audio[0] if audio.ndim == 2 and audio.shape[0] == 1 else audio.T
    sf.write(str(path), data, result.sample_rate)
