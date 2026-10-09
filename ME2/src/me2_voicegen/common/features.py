"""Log-mel front-end for the toy VCM CTC acoustic model.

There is exactly ONE feature front-end in this feature: this module.
`vcm.dataset` (train/eval), `vcm.pipeline`-style eval/inference code (05),
and export (06) must all import `LogMelFeatureExtractor`/`log_mel_features`
from here rather than re-implementing MelSpectrogram parameters locally --
train/infer skew from two independently-tuned front-ends is the specific
risk this module exists to close off.

Per docs/VCM-CONTRACT.md section 5:
    - 40 mel bins
    - 480-sample analysis window (30ms @ 16kHz)
    - 160-sample hop (10ms)
    - log magnitude with an epsilon floor (no log(0))
    - output layout (batch, n_mels, frames), i.e. (B, 40, T)

Per-utterance mean/var normalization is applied after the log-mel step, on
each example independently (not batch statistics), so it is well-defined
for both single-utterance inference and batched training.
"""

from __future__ import annotations

import torch
import torchaudio

SAMPLE_RATE = 16000
N_FFT = 480
WIN_LENGTH = 480
HOP_LENGTH = 160
N_MELS = 40
LOG_EPS = 1e-6
NORM_EPS = 1e-5


class LogMelFeatureExtractor(torch.nn.Module):
    """Waveform -> per-utterance-normalized log-mel features.

    Input: waveform tensor of shape (samples,) or (channels, samples) at
    `SAMPLE_RATE`. Multi-channel input is averaged to mono first.
    Output: (n_mels, frames) float tensor, i.e. an unbatched (40, T)
    slice of the (B, 40, T) contract in docs/VCM-CONTRACT.md section 5.
    """

    def __init__(self) -> None:
        super().__init__()
        self._mel = torchaudio.transforms.MelSpectrogram(
            sample_rate=SAMPLE_RATE,
            n_fft=N_FFT,
            win_length=WIN_LENGTH,
            hop_length=HOP_LENGTH,
            n_mels=N_MELS,
        )

    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        if waveform.dim() == 2:
            waveform = waveform.mean(dim=0)
        elif waveform.dim() != 1:
            raise ValueError(
                f"expected waveform of shape (samples,) or (channels, samples), "
                f"got {tuple(waveform.shape)}"
            )

        mel = self._mel(waveform)
        log_mel = torch.log(mel + LOG_EPS)

        mean = log_mel.mean()
        std = log_mel.std()
        normalized = (log_mel - mean) / (std + NORM_EPS)
        return normalized


def log_mel_features(waveform: torch.Tensor) -> torch.Tensor:
    """Functional convenience wrapper around `LogMelFeatureExtractor`.

    Constructs a fresh extractor per call -- MelSpectrogram has no learned
    state, so this costs a small fixed filterbank recompute, not a model
    load. Prefer instantiating `LogMelFeatureExtractor` once and reusing it
    in any hot loop (e.g. `vcm.dataset`'s `__getitem__`).
    """
    return LogMelFeatureExtractor()(waveform)
