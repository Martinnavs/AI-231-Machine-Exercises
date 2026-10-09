"""Fixed-window fit for the wakeword DS-CNN (feature `wakeword-dscnn`,
feature-engineering/wakeword-dscnn/SPEC.md).

`WakewordDataset` crops each clip down to its speech span (plus a margin)
before calling into this module, which then fits that variable-length
segment into `WAKEWORD_WINDOW_SECONDS`'s fixed sample count: `shift_waveform`
for train (randomizes the fit position -- this is the "temporal shifting"
augmentation the handoff doc asked for, and doubles as a random-crop
safety net if the segment is longer than the window), `center_window` for
eval/export/calibration (deterministic, no randomness).

Noise mixing itself is not duplicated here -- `common.augment.Augmenter`
(RIR disabled via `p_rir=0.0`, per SPEC.md's Edges table) is reused
unmodified for that.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


def shift_waveform(
    waveform: torch.Tensor, window_samples: int, generator: torch.Generator
) -> torch.Tensor:
    """Fit `waveform` into exactly `window_samples`, at a random position,
    never moving any of `waveform`'s content outside `[0, window_samples)`.

    Shorter than the window: zero-padded, with the content placed at a
    random offset (the temporal-shift augmentation). Longer than or equal
    to the window: a random crop (a defensive case, not the common path --
    `WakewordDataset` normally hands this a segment already at or under
    the window length)."""
    n = waveform.shape[-1]
    if n >= window_samples:
        max_start = n - window_samples
        start = int(torch.randint(0, max_start + 1, (1,), generator=generator).item())
        return waveform[..., start : start + window_samples]

    max_pad = window_samples - n
    pad_left = int(torch.randint(0, max_pad + 1, (1,), generator=generator).item())
    pad_right = max_pad - pad_left
    return F.pad(waveform, (pad_left, pad_right))


def center_window(waveform: torch.Tensor, window_samples: int) -> torch.Tensor:
    """Deterministic counterpart to `shift_waveform`, for eval/export/
    calibration: center-crop if longer than the window, center-pad if
    shorter. Never randomized."""
    n = waveform.shape[-1]
    if n >= window_samples:
        start = (n - window_samples) // 2
        return waveform[..., start : start + window_samples]

    total_pad = window_samples - n
    pad_left = total_pad // 2
    pad_right = total_pad - pad_left
    return F.pad(waveform, (pad_left, pad_right))
