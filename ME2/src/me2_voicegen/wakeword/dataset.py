"""Manifest-backed `torch.utils.data.Dataset` for the wakeword DS-CNN
(feature `wakeword-dscnn`, feature-engineering/wakeword-dscnn/SPEC.md).

Reads `out/conversions/v2/wakeword/manifest.csv` (ticket 04's final
assembled dataset -- see docs/WAKEWORD-DATASET-CONTRACT.md), filtered by
`split`. Each row's `path` already carries its owning subset as the first
path component (e.g. `positives_real/audio/picovoice/<uuid>.wav`), so
`audio_root` is just the manifest's own parent directory.

Windowing (SPEC.md's RECAP, Edges):
    - `_silence_` rows: no VAD at all (synthetic silence by construction).
    - Rows carrying precomputed `speech_start_s`/`speech_end_s`
      (`derive_speech_spans.py`, written for the `_wakeword_` chain and
      `common_voice_negative_sample`): crop to that span plus
      `derive_speech_spans.SPAN_MARGIN_SECONDS` of context per side.
    - Everything else (`adversaries`/`adversaries_noisy`, by design, plus
      any row where the precomputed columns are empty/missing): live
      per-sample `torchaudio.functional.vad` via
      `derive_speech_spans.detect_speech_span`, falling back to the whole
      clip if that also returns nothing.

Fallback-occurrence logging is asymmetric on purpose (SPEC.md's round-3
Edges row, refined once this pipeline actually ran against the real
dataset): `positives_real`/`positives_converted`/`positives_converted_noisy`
log one warning line per occurrence, since a fallback there is genuinely
rare (~2-3%, measured). `adversaries`/`adversaries_noisy` always take the
live-VAD path by design, and `common_voice_negative_sample` -- although it
does get a precomputed-span attempt -- turned out to have its own ~55%
natural VAD-empty rate once measured for real, just as high as
`adversaries`'; per-row logging for either would be pure noise, so both
are counted only, surfaced once via `log_fallback_summary()`.

The variable-length "content" segment (whatever VAD/precompute decided,
or the whole clip) is then fit into `WAKEWORD_WINDOW_SECONDS` by
`wakeword.augment.shift_waveform` (train, randomized -- the "temporal
shifting" augmentation) or `center_window` (eval/export, deterministic).
Dynamic SNR-mixing noise augmentation reuses `common.augment.Augmenter`
unmodified, with `p_rir` expected to stay 0.0 (SPEC.md Edges: the handoff
doc explicitly excludes RIR for this toy model).
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import NamedTuple

import torch
import torchaudio
from torch.utils.data import Dataset

from me2_voicegen.common.ambient_mix import AMBIENT_SUFFIX
from me2_voicegen.common.augment import Augmenter
from me2_voicegen.common.features import LogMelFeatureExtractor
from me2_voicegen.wakeword.augment import center_window, shift_waveform
from me2_voicegen.wakeword.derive_speech_spans import SPAN_MARGIN_SECONDS, detect_speech_span
from me2_voicegen.wakeword.model import LABEL_TO_ID

WAKEWORD_WINDOW_SECONDS: float = 1.5

# Subsets `derive_speech_spans.py` attempts to populate (join/precompute
# eligibility -- distinct from RARE_FALLBACK_SUBSETS below, which is about
# logging verbosity, not eligibility).
PRECOMPUTED_SUBSETS: frozenset[str] = frozenset(
    {"positives_real", "positives_converted", "positives_converted_noisy", "common_voice_negative_sample"}
)

# Subsets where a live-VAD fallback is a rare, actionable anomaly (~2-3%
# measured against the real dataset) and worth a per-row warning.
# `common_voice_negative_sample` is deliberately excluded even though it IS
# in `PRECOMPUTED_SUBSETS` above: running `derive_speech_spans.py` against
# the real dataset measured its own natural VAD-empty rate at ~55% --
# comparable to `adversaries`', not "rare" -- so per-row warnings there
# would be the exact log-noise problem this split exists to avoid (SPEC.md
# round 3's `adversaries` reasoning, discovered to also apply here only
# once the real numbers came in, logged as a minor deviation in SPEC.md).
# Everything else (adversaries/adversaries_noisy by design;
# common_voice_negative_sample; anything unrecognized) gets aggregate-only
# logging via `log_fallback_summary`.
RARE_FALLBACK_SUBSETS: frozenset[str] = frozenset(
    {"positives_real", "positives_converted", "positives_converted_noisy"}
)


class WakewordExample(NamedTuple):
    waveform: torch.Tensor
    label: int


class WakewordDataset(Dataset):
    """One row of the final assembled wakeword manifest per item.

    Args:
        manifest_path: path to `out/conversions/v2/wakeword/manifest.csv`
            (or a fixture with the same 10-column + extension schema).
        audio_root: directory `path` column entries are relative to
            (defaults to `manifest_path`'s parent).
        split: keep only rows whose `split` column equals this value.
        window_seconds: fixed window length every returned waveform is fit
            to (default `WAKEWORD_WINDOW_SECONDS`).
        augmenter: applied only when not None and `split == "train"`.
            Expected to have `p_rir=0.0` (SPEC.md Edges) -- RIR is not
            asserted off here, only documented as the intended usage.
        shift: when True and `split == "train"`, windowing uses
            `shift_waveform` (randomized fit / temporal-shift
            augmentation) instead of `center_window`.
        noise_root: directory of noise wavs for `augmenter`'s dynamic SNR
            mixing (e.g. `out/conversions/v2/background_noise`, whose wavs
            live under its `audio/` subdirectory -- searched recursively,
            not just top-level, so this corpus layout resolves correctly).
            Loaded and cached once, lazily, on first use -- never touched if
            `augmenter` is None or `augmenter.p_noise == 0`.
        generator: drives both windowing randomness and is handed to
            `augmenter`'s own calls; defaults to a fresh
            `torch.Generator()` if not supplied.
    """

    def __init__(
        self,
        manifest_path: str | Path,
        split: str | None = "train",
        audio_root: str | Path | None = None,
        window_seconds: float = WAKEWORD_WINDOW_SECONDS,
        augmenter: Augmenter | None = None,
        shift: bool = False,
        noise_root: str | Path | None = None,
        generator: torch.Generator | None = None,
    ) -> None:
        self.manifest_path = Path(manifest_path)
        self.audio_root = Path(audio_root) if audio_root is not None else self.manifest_path.parent
        self.split = split
        self.window_seconds = window_seconds
        self.augmenter = augmenter
        self.shift = shift
        self.noise_root = Path(noise_root) if noise_root is not None else None
        self.generator = generator if generator is not None else torch.Generator()

        with self.manifest_path.open(newline="", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))

        if split is not None:
            filtered = [r for r in rows if r.get("split") == split]
            if not filtered and any(r.get("split", "") == "" for r in rows):
                raise ValueError(
                    f"no rows with split={split!r} in {self.manifest_path}, and this manifest "
                    "has unassigned (empty-string) split rows -- this looks like a subset "
                    "manifest from before ticket 04's split assignment, not the final assembled "
                    "manifest.csv (WAKEWORD-DATASET-CONTRACT.md section 2)"
                )
            rows = filtered
        self.rows: list[dict] = rows

        self._noise_pool_cache: list[torch.Tensor] | None = None
        self.fallback_counts: dict[str, int] = {}

    def __len__(self) -> int:
        return len(self.rows)

    def _load_waveform(self, row: dict) -> tuple[torch.Tensor, int]:
        wav_path = self.audio_root / row["path"]
        waveform, sample_rate = torchaudio.load(str(wav_path))
        if waveform.dim() == 2:
            waveform = waveform.mean(dim=0)
        return waveform, sample_rate

    def _noise_pool(self) -> list[torch.Tensor]:
        if self._noise_pool_cache is None:
            pool: list[torch.Tensor] = []
            if self.noise_root is not None and self.noise_root.is_dir():
                for wav_path in sorted(self.noise_root.rglob("*.wav")):
                    waveform, _sr = torchaudio.load(str(wav_path))
                    if waveform.dim() == 2:
                        waveform = waveform.mean(dim=0)
                    pool.append(waveform)
            self._noise_pool_cache = pool
        return self._noise_pool_cache

    def _content_segment(self, row: dict, waveform: torch.Tensor, sample_rate: int) -> torch.Tensor:
        if row["label"] == "_silence_":
            return waveform

        start_raw = row.get("speech_start_s", "")
        end_raw = row.get("speech_end_s", "")
        subset = row["path"].split("/", 1)[0]

        if start_raw and end_raw:
            start_s, end_s = float(start_raw), float(end_raw)
        else:
            span = detect_speech_span(waveform, sample_rate)
            if span is None:
                if subset in RARE_FALLBACK_SUBSETS:
                    print(
                        f"WARNING: live VAD fallback (empty span) for {subset}/{row['filename']} "
                        "-- using whole clip"
                    )
                self.fallback_counts[subset] = self.fallback_counts.get(subset, 0) + 1
                return waveform
            start_s, end_s = span
            if subset in RARE_FALLBACK_SUBSETS:
                print(f"WARNING: live VAD fallback for {subset}/{row['filename']} (no precomputed span)")
            self.fallback_counts[subset] = self.fallback_counts.get(subset, 0) + 1

        start_sample = max(0, int((start_s - SPAN_MARGIN_SECONDS) * sample_rate))
        end_sample = min(waveform.numel(), int((end_s + SPAN_MARGIN_SECONDS) * sample_rate))
        if end_sample <= start_sample:
            return waveform
        return waveform[start_sample:end_sample]

    def log_fallback_summary(self) -> None:
        """Aggregate-only fallback counts for subsets not in
        `RARE_FALLBACK_SUBSETS` (adversaries/adversaries_noisy by design,
        plus common_voice_negative_sample -- its own natural VAD-empty rate
        turned out just as high once measured against the real dataset;
        per-row logging there would be just as much noise). Call once per
        epoch, not per batch."""
        for subset, count in sorted(self.fallback_counts.items()):
            if subset not in RARE_FALLBACK_SUBSETS:
                total = sum(1 for r in self.rows if r["path"].split("/", 1)[0] == subset)
                print(f"{subset}: {count}/{total} rows used live-VAD center-crop fallback this pass")

    def __getitem__(self, index: int) -> WakewordExample:
        row = self.rows[index]
        waveform, sample_rate = self._load_waveform(row)
        label = LABEL_TO_ID[row["label"]]

        content = self._content_segment(row, waveform, sample_rate)
        window_samples = int(round(self.window_seconds * sample_rate))

        if self.split == "train" and self.shift:
            windowed = shift_waveform(content, window_samples, self.generator)
        else:
            windowed = center_window(content, window_samples)

        if self.augmenter is not None and self.split == "train":
            # Rows whose source_dataset ends in `_ambient` are already
            # noised offline (ambient-noise-overlay feature): skip the
            # online noise step so no row is double-noised.
            is_ambient = row["source_dataset"].endswith(AMBIENT_SUFFIX)
            noise_pool = None if is_ambient else (self._noise_pool() if self.augmenter.p_noise > 0 else None)
            windowed = self.augmenter.augment_waveform(windowed, noise_pool=noise_pool)

        return WakewordExample(waveform=windowed, label=label)


def collate_fn(
    batch: list[WakewordExample], feature_extractor: LogMelFeatureExtractor
) -> dict[str, torch.Tensor]:
    """`(waveform, label)` examples, all already the same fixed window
    length -> `(B, 40, T)` log-mel features + `(B,)` labels."""
    features = torch.stack([feature_extractor(ex.waveform) for ex in batch])
    labels = torch.tensor([ex.label for ex in batch], dtype=torch.long)
    return {"features": features, "labels": labels}
