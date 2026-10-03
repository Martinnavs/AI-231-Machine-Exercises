"""Seedable on-the-fly waveform augmentations for VCM training.

Four augmentations, all OFF by default (an `Augmenter` built with the
default `p_*` probabilities of 0, or called in eval mode via
`vcm.dataset`, applies nothing):

1. **RIR (reverberation).** `torchaudio.prototype.functional.simulate_rir_ism`
   -- verified this session to import and run cleanly under this venv's
   pinned torch==2.3.1+cu121 / torchaudio==2.3.1+cu121 (no fallback needed;
   the ticket's synthetic exponentially-decaying-noise fallback was NOT
   used). One wrinkle worth recording: `simulate_rir_ism` takes an
   `absorption` coefficient, not RT60 directly -- there is no RT60
   convenience arg in this torchaudio version's API. We sample a target
   RT60 uniformly in [0.1, 0.5]s per room and convert it to a wall
   absorption coefficient via Sabine's formula
   (`absorption = 0.161 * volume / (surface_area * rt60)`, clamped to
   (0, 1)), which is the standard approach and the one torchaudio's own
   RIR tutorial uses.

   A pool of `rir_pool_size` (~200) impulse responses is generated once at
   `Augmenter.__init__` time (randomized room/source/mic geometry + RT60
   per pool entry), not written to disk. Each augmented example then draws
   one pool entry at random ("dynamic" per docs/VCM-CONTRACT.md's original
   spec language: random per-example/per-epoch selection from a
   pre-built in-memory pool, not a full from-scratch image-source
   simulation per training example).

2. **Additive noise.** `torchaudio.functional.add_noise` at an SNR sampled
   uniformly in [5, 25] dB, noise waveform supplied by the caller (drawn
   by `vcm.dataset` from the manifest's `background_noise` bucket).

3. **SpecAugment.** `torchaudio.transforms.TimeMasking` (1-2 masks, each
   <=25 frames) and `torchaudio.transforms.FrequencyMasking` (1-2 masks,
   each <=8 bins), applied to log-mel features (post `vcm.features`), not
   to the raw waveform.

4. **Time-stretch.** A stochastic resample-based stretch of the raw
   waveform, applied BEFORE log-mel extraction to break TTS prosodic
   homogeneity. The factor is drawn uniformly in [0.85, 1.15] (factor < 1
   -> shorter/faster/higher-pitch clip, factor > 1 -> longer/slower/
   lower-pitch clip) and the output length is exactly
   `round(N * factor)`. Resampling uses `torchaudio.functional.resample`
   with this venv's default backend (`resampling_method=
   'sinc_interp_hann'`, torch 2.3.1+cu121 / torchaudio 2.3.1+cu121) --
   verified this session on 16 kHz CPU tensors: finite, deterministic
   across calls, and exact `round(N * factor)` length for factors
   0.85/1.0/1.15 (no fallback needed). One wrinkle: this torchaudio
   version's `resample` takes integer `orig_freq`/`new_freq`, so the
   requested rate is `round(sample_rate * factor)`; on long clips that
   integer rounding can drift the raw resampled length by up to a few
   samples away from `round(N * factor)`, so the output is trimmed /
   edge-padded (repeat of the last sample) to the exact target length.

All randomness in this module is drawn from a `torch.Generator` the caller
supplies (or `Augmenter` creates internally, seedable via `seed=`), so a
fixed seed makes an `Augmenter`'s output fully reproducible.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from fractions import Fraction

import torch
import torchaudio
import torchaudio.functional as AF
from torchaudio.prototype.functional import simulate_rir_ism

RT60_MIN = 0.1
RT60_MAX = 0.5
SNR_MIN_DB = 5.0
SNR_MAX_DB = 25.0
# Speech-babble interference is mixed in more quietly than environmental noise: the target words must stay
# intelligible, since the transcript label is not re-checked after mixing.
BABBLE_SNR_MIN_DB = 12.0
BABBLE_SNR_MAX_DB = 25.0
TIME_MASK_MAX_FRAMES = 25
FREQ_MASK_MAX_BINS = 8
TSTRETCH_FACTOR_MIN = 0.85
TSTRETCH_FACTOR_MAX = 1.15


def _random_room_rir(generator: torch.Generator, sample_rate: int) -> torch.Tensor:
    """Build one RIR from a randomized shoebox room + RT60, via Sabine's
    formula for RT60 -> wall absorption (simulate_rir_ism takes absorption,
    not RT60, directly)."""
    room = (
        torch.tensor([4.0, 4.0, 2.5])
        + torch.rand(3, generator=generator) * torch.tensor([4.0, 4.0, 2.0])
    )
    volume = float(room.prod())
    w, l, h = room.tolist()
    surface = 2.0 * (w * l + w * h + l * h)

    rt60 = RT60_MIN + torch.rand((), generator=generator).item() * (RT60_MAX - RT60_MIN)
    absorption = 0.161 * volume / (surface * rt60)
    absorption = float(min(max(absorption, 0.01), 0.99))

    margin = 0.5
    source = margin + torch.rand(3, generator=generator) * (room - 2 * margin)
    mic = margin + torch.rand(3, generator=generator) * (room - 2 * margin)

    rir = simulate_rir_ism(
        room,
        source,
        mic.unsqueeze(0),
        max_order=10,
        absorption=absorption,
        sample_rate=float(sample_rate),
    )
    rir = rir[0]
    peak = rir.abs().max()
    if peak > 0:
        rir = rir / peak
    return rir


def build_rir_pool(
    pool_size: int = 200,
    sample_rate: int = 16000,
    seed: int | None = None,
) -> list[torch.Tensor]:
    """Pre-generate `pool_size` room impulse responses in memory."""
    generator = torch.Generator().manual_seed(seed) if seed is not None else torch.Generator()
    return [_random_room_rir(generator, sample_rate) for _ in range(pool_size)]


def apply_rir(waveform: torch.Tensor, rir: torch.Tensor) -> torch.Tensor:
    """Convolve `waveform` with `rir`, trimmed back to `waveform`'s length
    so RIR augmentation never changes an example's duration."""
    orig_len = waveform.shape[-1]
    wet = AF.fftconvolve(waveform.unsqueeze(0), rir.unsqueeze(0))[0]
    return wet[:orig_len]


def apply_noise(waveform: torch.Tensor, noise: torch.Tensor, snr_db: float, offset: float | None = None) -> torch.Tensor:
    """Mix `noise` into `waveform` at `snr_db` dB SNR, looping/trimming
    `noise` to match `waveform`'s length first. `offset` in [0, 1) picks where in the noise clip the window starts
    (None = its beginning, the original behaviour): a clip longer than the waveform is cut at that fraction of its
    spare length, a shorter one is rotated by that fraction before looping."""
    target_len = waveform.shape[-1]
    if offset is not None:
        if noise.shape[-1] > target_len:
            start = int(offset * (noise.shape[-1] - target_len))
            noise = noise[start : start + target_len]
        else:
            noise = torch.roll(noise, -int(offset * noise.shape[-1]), dims=-1)
    if noise.shape[-1] < target_len:
        repeats = math.ceil(target_len / noise.shape[-1])
        noise = noise.repeat(repeats)
    noise = noise[:target_len]
    if not bool((noise.pow(2).mean() > 0) & (waveform.pow(2).mean() > 0)):
        # `add_noise` divides by the noise power: an all-zero window (some clips open with digital silence)
        # or an all-zero waveform would give inf * 0 = NaN. There is nothing to mix, so leave the audio as is.
        return waveform
    snr = torch.tensor([snr_db])
    return AF.add_noise(waveform.unsqueeze(0), noise.unsqueeze(0), snr)[0]


TSTRETCH_MAX_DENOMINATOR = 200


def apply_timestretch(
    waveform: torch.Tensor, factor: float, sample_rate: int = 16000
) -> torch.Tensor:
    """Time-stretch `waveform` (samples,) by `factor`: the output length is
    exactly `round(len(waveform) * factor)`. Implemented as a resample via
    `torchaudio.functional.resample`, which only depends on the *ratio* of
    its two frequencies. That ratio is `factor` approximated by a fraction
    with denominator <= TSTRETCH_MAX_DENOMINATOR: a continuous factor mapped
    onto `round(sample_rate * factor)` Hz is almost always coprime with
    `sample_rate`, which makes the resampler build a ~16000-phase filter
    (measured: 5.3 s per 2 s clip at factor 1.1507, vs 2 ms at 1.15). The
    approximation error is <= 2.5e-5 in the ratio (about a sample on a 2 s
    clip); the resampled signal is then trimmed/edge-padded to the exact
    target length. `sample_rate` is kept for API compatibility and unused."""
    target_len = max(1, int(round(waveform.shape[-1] * factor)))
    ratio = Fraction(factor).limit_denominator(TSTRETCH_MAX_DENOMINATOR)
    stretched = AF.resample(waveform, ratio.denominator, ratio.numerator)
    if stretched.shape[-1] > target_len:
        stretched = stretched[..., :target_len]
    elif stretched.shape[-1] < target_len:
        pad = stretched[..., -1:].expand(*stretched.shape[:-1], target_len - stretched.shape[-1])
        stretched = torch.cat([stretched, pad], dim=-1)
    return stretched


@dataclass(frozen=True)
class Recipe:
    """One clip's waveform perturbations, fully specified (no randomness left): `None` = step skipped.
    Indices point into the pools handed to `apply_recipe`."""

    stretch: float | None = None
    rir: int | None = None
    noise: int | None = None
    noise_snr_db: float | None = None
    babble: int | None = None
    babble_snr_db: float | None = None
    noise_offset: float | None = None  # where in the noise / babble clip the mixed window starts (see apply_noise)
    babble_offset: float | None = None


def apply_recipe(
    waveform: torch.Tensor,
    recipe: Recipe,
    rir_pool: list[torch.Tensor],
    noise_pool: list[torch.Tensor],
    babble_pool: list[torch.Tensor],
    sample_rate: int = 16000,
) -> torch.Tensor:
    """Deterministic counterpart of `Augmenter.augment_waveform` (same order: stretch, RIR, noise, babble)."""
    if recipe.stretch is not None:
        waveform = apply_timestretch(waveform, recipe.stretch, sample_rate=sample_rate)
    if recipe.rir is not None:
        waveform = apply_rir(waveform, rir_pool[recipe.rir])
    if recipe.noise is not None:
        waveform = apply_noise(waveform, noise_pool[recipe.noise], recipe.noise_snr_db, recipe.noise_offset)
    if recipe.babble is not None:
        waveform = apply_noise(waveform, babble_pool[recipe.babble], recipe.babble_snr_db, recipe.babble_offset)
    return waveform


class Augmenter:
    """Seedable on-the-fly time-stretch + RIR + additive-noise + SpecAugment
    pipeline.

    All four augmentations default to probability 0 (OFF). `vcm.dataset`
    is expected to construct this with nonzero `p_*` only for its train
    split, and either omit it or leave the defaults for eval.
    """

    def __init__(
        self,
        sample_rate: int = 16000,
        rir_pool_size: int = 200,
        p_rir: float = 0.0,
        p_noise: float = 0.0,
        p_specaugment: float = 0.0,
        p_timestretch: float = 0.0,
        p_babble: float = 0.0,
        seed: int | None = None,
    ) -> None:
        self.sample_rate = sample_rate
        self.p_rir = p_rir
        self.p_noise = p_noise
        self.p_specaugment = p_specaugment
        self.p_timestretch = p_timestretch
        self.p_babble = p_babble
        self.generator = torch.Generator().manual_seed(seed) if seed is not None else torch.Generator()
        self._rir_pool: list[torch.Tensor] | None = None
        self._rir_pool_size = rir_pool_size

    @property
    def rir_pool(self) -> list[torch.Tensor]:
        if self._rir_pool is None:
            self._rir_pool = build_rir_pool(
                self._rir_pool_size,
                sample_rate=self.sample_rate,
                seed=int(torch.randint(0, 2**31 - 1, (1,), generator=self.generator).item()),
            )
        return self._rir_pool

    def _rand(self) -> float:
        return torch.rand((), generator=self.generator).item()

    def augment_waveform(
        self,
        waveform: torch.Tensor,
        noise_pool: list[torch.Tensor] | None = None,
        babble_pool: list[torch.Tensor] | None = None,
    ) -> torch.Tensor:
        """Apply time-stretch (if enabled) then RIR then additive-noise
        augmentation (each independently gated by its own probability) to a
        raw waveform. The stretch runs first so RIR/noise operate at the
        stretched duration; it is also the only step that may change an
        example's length (RIR trims back to, and noise matches, the
        waveform's current length)."""
        if self.p_timestretch > 0 and self._rand() < self.p_timestretch:
            factor = TSTRETCH_FACTOR_MIN + self._rand() * (TSTRETCH_FACTOR_MAX - TSTRETCH_FACTOR_MIN)
            waveform = apply_timestretch(waveform, factor, sample_rate=self.sample_rate)

        if self.p_rir > 0 and self._rand() < self.p_rir:
            idx = int(torch.randint(0, len(self.rir_pool), (1,), generator=self.generator).item())
            waveform = apply_rir(waveform, self.rir_pool[idx])

        if self.p_noise > 0 and noise_pool and self._rand() < self.p_noise:
            idx = int(torch.randint(0, len(noise_pool), (1,), generator=self.generator).item())
            snr_db = SNR_MIN_DB + self._rand() * (SNR_MAX_DB - SNR_MIN_DB)
            waveform = apply_noise(waveform, noise_pool[idx], snr_db)

        if self.p_babble > 0 and babble_pool and self._rand() < self.p_babble:
            idx = int(torch.randint(0, len(babble_pool), (1,), generator=self.generator).item())
            snr_db = BABBLE_SNR_MIN_DB + self._rand() * (BABBLE_SNR_MAX_DB - BABBLE_SNR_MIN_DB)
            waveform = apply_noise(waveform, babble_pool[idx], snr_db)

        return waveform

    def augment_features(self, log_mel: torch.Tensor) -> torch.Tensor:
        """Apply SpecAugment time/frequency masking to a (n_mels, T)
        log-mel feature tensor. No-op unless `p_specaugment > 0` and the
        per-call gate fires.

        `torchaudio.transforms.TimeMasking`/`FrequencyMasking` draw from
        torch's *global* RNG (they accept no generator argument), so full
        determinism under a fixed `Augmenter` seed requires seeding that
        global RNG too. We do so here from a value drawn off this
        `Augmenter`'s own generator, which keeps the sequence reproducible
        for a given `Augmenter(seed=...)` at the cost of mutating global
        torch RNG state as a side effect -- acceptable for this toy
        single-process training/eval scope, but callers relying on global
        RNG state elsewhere in the same process should be aware of it.
        """
        if self.p_specaugment <= 0 or self._rand() >= self.p_specaugment:
            return log_mel

        step_seed = int(torch.randint(0, 2**31 - 1, (1,), generator=self.generator).item())
        torch.manual_seed(step_seed)

        out = log_mel.unsqueeze(0)
        n_time_masks = 1 + int(self._rand() < 0.5)
        for _ in range(n_time_masks):
            tm = torchaudio.transforms.TimeMasking(time_mask_param=TIME_MASK_MAX_FRAMES)
            out = tm(out)

        n_freq_masks = 1 + int(self._rand() < 0.5)
        for _ in range(n_freq_masks):
            fm = torchaudio.transforms.FrequencyMasking(freq_mask_param=FREQ_MASK_MAX_BINS)
            out = fm(out)

        return out[0]
