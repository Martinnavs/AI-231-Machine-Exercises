"""Dataset-free, deterministic training noise (feature ctc-attention).

The ESC-50 `background_noise` pool the earlier runs trained with is not rebuildable from the repo (its fetch/chunk
scripts are untracked and need a Kaggle login). This builds a noise pool from nothing but a seed, reusing the repo's
`wakeword.generate_silence.colored_noise` generator: white / pink / brown noise, each stationary or slowly amplitude-
modulated (1-4 Hz, a crude stand-in for fans, traffic and other non-stationary backgrounds). Same seed => same pool.
`Augmenter` rescales every entry to the requested SNR, so only the shape matters, not the level.
"""

from __future__ import annotations

import numpy as np
import torch

from me2_voicegen.wakeword.generate_silence import NOISE_COLORS, colored_noise

MODULATION_HZ = (1.0, 4.0)
MODULATION_DEPTH = 0.6


def synthetic_noise_pool(
    n_clips: int = 240, seconds: float = 3.0, sample_rate: int = 16000, seed: int = 0
) -> list[torch.Tensor]:
    """`n_clips` 1-D float32 clips, colours cycling white/pink/brown and every second one modulated."""
    n = int(seconds * sample_rate)
    t = np.arange(n) / sample_rate
    pool = []
    for i in range(n_clips):
        rng = np.random.default_rng([seed, i])
        noise = colored_noise(n, NOISE_COLORS[i % len(NOISE_COLORS)], rng)
        if (i // len(NOISE_COLORS)) % 2 == 1:
            hz = rng.uniform(*MODULATION_HZ)
            envelope = 1.0 + MODULATION_DEPTH * np.sin(2 * np.pi * hz * t + rng.uniform(0, 2 * np.pi))
            noise = noise * envelope
        pool.append(torch.from_numpy(noise.astype(np.float32)))
    return pool
