"""Real-data CTC feasibility of the QuartzNet subsampling (slow; skipped when
the production manifest is absent). Stride 2 must leave every loss-bearing row
with enough output frames; stride-4 counts are reported, not asserted."""

from __future__ import annotations

import math
from pathlib import Path

import pytest

from me2_voicegen.common.features import LogMelFeatureExtractor
from me2_voicegen.vcm import alphabet
from me2_voicegen.vcm.dataset import VCMDataset
from me2_voicegen.vcm.text import normalize_text
from me2_voicegen.vcm.train import ctc_min_frames

MANIFEST = (
    Path(__file__).resolve().parents[1]
    / "out/conversions/v2/optionb-v3-vcmx-fil50-ambient/manifest.csv"
)

pytestmark = [
    pytest.mark.slow,
    pytest.mark.skipif(not MANIFEST.exists(), reason="production manifest not present"),
]


def _frames(duration_s: float) -> int:
    return math.floor(duration_s * 16000 / 160) + 1


def test_frame_formula_matches_extractor_on_real_clips():
    ds = VCMDataset(MANIFEST, split="val")
    extractor = LogMelFeatureExtractor()
    for idx in ds.loss_bearing_indices[:3]:
        wave = ds[idx].waveform
        assert extractor(wave).shape[-1] == _frames(wave.numel() / 16000)


def test_no_ctc_infeasible_rows_at_stride_2(capsys):
    """Train/val (the loss-bearing splits) must have zero infeasible rows at
    stride 2. The test split has one known row (a 0.59 s Common Voice babble
    clip whose transcript needs 55 frames, `cv-valid-train__sample-121890_c04`);
    it is scored, never trained on, and is not filtered."""
    counts: dict[tuple[str, int], int] = {}
    for split in ("train", "val", "test"):
        ds = VCMDataset(MANIFEST, split=split)
        for idx in ds.loss_bearing_indices:
            ids = alphabet.encode(normalize_text(ds._resolved[idx]))
            t = _frames(float(ds.rows[idx]["duration"]))
            for stride in (2, 4):
                if ctc_min_frames(ids) > -(-t // stride):
                    counts[(split, stride)] = counts.get((split, stride), 0) + 1
    print(f"infeasible (split, stride) -> count: {counts}")
    assert counts.get(("train", 2), 0) == 0
    assert counts.get(("val", 2), 0) == 0
    assert counts.get(("test", 2), 0) <= 1
