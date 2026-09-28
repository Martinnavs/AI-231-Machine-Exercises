"""Manifest-backed `torch.utils.data.Dataset` for the toy VCM CTC model.

Reads `out/conversions/v2/test_set/manifest.csv` (or any manifest with the
same column schema -- see `tests/conftest.py`'s `vcm_fake_manifest` fixture
for a synthetic stand-in), filtered by `split`, and resolves each row's
target transcript via `vcm.text.resolve_transcript` (per
docs/VCM-CONTRACT.md section 4).

Rows resolving to `None` (currently: all `filipino_speech_corpus` rows) are
excluded from `VCMDataset.loss_bearing_indices` -- the subset a training
loop should sample from -- but remain present and loadable by index for
eval-only use (e.g. rejection/false-accept probes). `__getitem__` returns
`(waveform, target_ids, target_len)` for every row regardless, with
`target_ids`/`target_len` set to an empty tensor / 0 for `None`-resolved
rows; callers that need to skip them for CTC loss should iterate
`loss_bearing_indices` (directly, or via a `Subset`/`Sampler`) rather than
filtering ad hoc.

Uses `common.features.LogMelFeatureExtractor` (the one and only feature
front-end for this whole feature, per docs/VCM-CONTRACT.md section 5) and,
optionally, a `common.augment.Augmenter` -- OFF by default, and expected to
stay off whenever `split != "train"` (eval-mode skew is exactly the thing
augmentation-on-by-default would risk).
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
from me2_voicegen.vcm import alphabet
from me2_voicegen.vcm.text import normalize_text, resolve_transcript


class VCMExample(NamedTuple):
    waveform: torch.Tensor
    target_ids: torch.Tensor
    target_len: int


class VCMDataset(Dataset):
    """One row of a VCM manifest per item.

    Args:
        manifest_path: path to a manifest.csv with the test_set schema
            (see docs/VCM-CONTRACT.md section 4's column table).
        audio_root: directory `path` column entries are relative to
            (defaults to `manifest_path`'s parent, matching
            `out/conversions/v2/test_set/`'s own layout).
        split: keep only rows whose `split` column equals this value.
        augmenter: applied only when not None and `split == "train"`.
    """

    def __init__(
        self,
        manifest_path: str | Path,
        audio_root: str | Path | None = None,
        split: str | None = "train",
        augmenter: Augmenter | None = None,
    ) -> None:
        self.manifest_path = Path(manifest_path)
        self.audio_root = Path(audio_root) if audio_root is not None else self.manifest_path.parent
        self.split = split
        self.augmenter = augmenter

        with self.manifest_path.open(newline="", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        if split is not None:
            rows = [r for r in rows if r["split"] == split]
        self.rows: list[dict] = rows

        self._resolved: list[str | None] = [resolve_transcript(r) for r in self.rows]
        self.loss_bearing_indices: list[int] = [
            i for i, t in enumerate(self._resolved) if t is not None
        ]

        self._noise_wave_paths = [
            self.audio_root / r["path"]
            for r in self.rows
            if r["source_dataset"] == "background_noise"
        ]

        self._features = LogMelFeatureExtractor()
        self._noise_pool_cache: list[torch.Tensor] | None = None

    def __len__(self) -> int:
        return len(self.rows)

    def _load_waveform(self, row: dict) -> torch.Tensor:
        wav_path = self.audio_root / row["path"]
        waveform, sample_rate = torchaudio.load(str(wav_path))
        if waveform.dim() == 2:
            waveform = waveform.mean(dim=0)
        if sample_rate != int(row["sample_rate"]):
            waveform = torchaudio.functional.resample(
                waveform, sample_rate, int(row["sample_rate"])
            )
        return waveform

    def _load_wav_path(self, wav_path: Path) -> torch.Tensor:
        waveform, _sample_rate = torchaudio.load(str(wav_path))
        if waveform.dim() == 2:
            waveform = waveform.mean(dim=0)
        return waveform

    def _noise_pool(self) -> list[torch.Tensor]:
        """Load every `background_noise`-source wav in this split once and
        cache it on the instance -- same spirit as `Augmenter.rir_pool`'s
        pre-generate-once-at-first-use pattern. Reloading these from disk
        per `__getitem__` call (the original bug here) caps real training
        throughput hard: ~135 extra disk reads per training example is the
        difference between ~17 epochs and ~90+ epochs in a fixed wall-clock
        budget (see this file's `docs/VCM-CONTRACT.md`-adjacent ticket's
        Execution Log for the diagnosis)."""
        if self._noise_pool_cache is None:
            self._noise_pool_cache = [self._load_wav_path(p) for p in self._noise_wave_paths]
        return self._noise_pool_cache

    def __getitem__(self, index: int) -> VCMExample:
        row = self.rows[index]
        waveform = self._load_waveform(row)
        transcript = self._resolved[index]

        if self.augmenter is not None and self.split == "train":
            # Rows whose source_dataset ends in `_ambient` are already
            # noised offline (ambient-noise-overlay feature): skip the
            # online noise step so no row is double-noised (RIR +
            # SpecAugment still apply).
            is_ambient = row["source_dataset"].endswith(AMBIENT_SUFFIX)
            noise_pool = None if is_ambient else (self._noise_pool() if self.augmenter.p_noise > 0 else None)
            waveform = self.augmenter.augment_waveform(waveform, noise_pool=noise_pool)

        if transcript is None:
            target_ids = torch.zeros(0, dtype=torch.long)
            target_len = 0
        else:
            ids = alphabet.encode(normalize_text(transcript))
            target_ids = torch.tensor(ids, dtype=torch.long)
            target_len = len(ids)

        return VCMExample(waveform=waveform, target_ids=target_ids, target_len=target_len)

    def features_for(self, index: int) -> torch.Tensor:
        """Convenience: `(waveform, ...) -> (n_mels, T)` log-mel for one
        item, applying SpecAugment (if this dataset's augmenter has it
        enabled and split == 'train')."""
        example = self[index]
        log_mel = self._features(example.waveform)
        if self.augmenter is not None and self.split == "train":
            log_mel = self.augmenter.augment_features(log_mel)
        return log_mel


def collate_fn(
    batch: list[VCMExample], feature_extractor: LogMelFeatureExtractor | None = None
) -> dict[str, torch.Tensor]:
    """Pad a list of `VCMExample`s into the CONTRACT's padded-batch shape
    (docs/VCM-CONTRACT.md section 6): `(B, 40, T)` log-mel + `input_lengths`
    + concatenated `target_ids` + `target_len`, i.e. exactly `nn.CTCLoss`'s
    expected input layout.
    """
    extractor = feature_extractor or LogMelFeatureExtractor()

    features = [extractor(ex.waveform) for ex in batch]
    input_lengths = torch.tensor([f.shape[-1] for f in features], dtype=torch.long)
    max_frames = int(input_lengths.max().item()) if len(features) else 0
    n_mels = features[0].shape[0] if features else 0

    padded = torch.zeros(len(batch), n_mels, max_frames)
    for i, f in enumerate(features):
        padded[i, :, : f.shape[-1]] = f

    target_ids = torch.cat([ex.target_ids for ex in batch]) if batch else torch.zeros(0, dtype=torch.long)
    target_len = torch.tensor([ex.target_len for ex in batch], dtype=torch.long)

    return {
        "features": padded,
        "input_lengths": input_lengths,
        "target_ids": target_ids,
        "target_len": target_len,
    }
