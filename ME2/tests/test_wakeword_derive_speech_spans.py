import csv
import math
from pathlib import Path

import pytest
import torch
import torchaudio

from me2_voicegen.wakeword.derive_speech_spans import (
    ANCESTOR_SUBSETS,
    DERIVED_SUBSETS,
    SPAN_COLUMNS,
    SPAN_MARGIN_SECONDS,
    derive_for_ancestor,
    derive_for_join,
    detect_speech_span,
)

SR = 16000


def _silence_tone_silence_wav(path: Path, lead_s: float = 0.5, tone_s: float = 1.0, trail_s: float = 0.5) -> float:
    path.parent.mkdir(parents=True, exist_ok=True)
    lead = torch.zeros(int(lead_s * SR))
    t = torch.arange(int(tone_s * SR), dtype=torch.float32) / SR
    tone = 0.5 * torch.sin(2 * math.pi * 440.0 * t)
    trail = torch.zeros(int(trail_s * SR))
    waveform = torch.cat([lead, tone, trail]).unsqueeze(0)
    torchaudio.save(str(path), waveform, SR)
    return waveform.shape[-1] / SR


def test_detect_speech_span_locates_tone_within_silence_padding(tmp_path):
    wav_path = tmp_path / "clip.wav"
    _silence_tone_silence_wav(wav_path)
    waveform, sr = torchaudio.load(str(wav_path))
    span = detect_speech_span(waveform, sr)
    assert span is not None
    start_s, end_s = span
    assert 0.3 <= start_s <= 0.6  # true onset is 0.5s
    assert 1.4 <= end_s <= 1.7  # true offset is 1.5s


def test_detect_speech_span_returns_none_for_pure_silence(tmp_path):
    waveform = torch.zeros(1, int(1.0 * SR))
    span = detect_speech_span(waveform.squeeze(0), SR)
    assert span is None


def _write_manifest(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


BASE_FIELDS = [
    "filename",
    "path",
    "label",
    "duration",
    "sample_rate",
    "resampled",
    "source_dataset",
    "source_relpath",
    "group_id",
    "split",
]


def test_derive_for_ancestor_writes_span_columns(tmp_path):
    root = tmp_path / "wakeword"
    subset_dir = root / "positives_real"
    duration = _silence_tone_silence_wav(subset_dir / "audio" / "picovoice" / "clip1.wav")

    rows = [
        {
            "filename": "clip1.wav",
            "path": "audio/picovoice/clip1.wav",
            "label": "_wakeword_",
            "duration": f"{duration:.6f}",
            "sample_rate": "16000",
            "resampled": "False",
            "source_dataset": "picovoice",
            "source_relpath": "audio/computer/clip1.wav",
            "group_id": "clip1",
            "split": "train",
        }
    ]
    _write_manifest(subset_dir / "manifest.csv", rows, BASE_FIELDS)

    stats = derive_for_ancestor(root, "positives_real")
    assert stats == {"subset": "positives_real", "rows": 1, "empty": 0}

    written = list(csv.DictReader((subset_dir / "manifest.csv").open()))
    assert len(written) == 1
    assert SPAN_COLUMNS[0] in written[0] and SPAN_COLUMNS[1] in written[0]
    assert written[0][SPAN_COLUMNS[0]] != ""
    assert float(written[0][SPAN_COLUMNS[0]]) < float(written[0][SPAN_COLUMNS[1]])


def test_derive_for_join_propagates_span_via_group_id(tmp_path):
    root = tmp_path / "wakeword"
    ancestor_dir = root / "positives_real"
    child_dir = root / "positives_converted"

    duration = _silence_tone_silence_wav(ancestor_dir / "audio" / "picovoice" / "clip1.wav")
    ancestor_rows = [
        {
            "filename": "clip1.wav",
            "path": "audio/picovoice/clip1.wav",
            "label": "_wakeword_",
            "duration": f"{duration:.6f}",
            "sample_rate": "16000",
            "resampled": "False",
            "source_dataset": "picovoice",
            "source_relpath": "audio/computer/clip1.wav",
            "group_id": "clip1",
            "split": "train",
        }
    ]
    _write_manifest(ancestor_dir / "manifest.csv", ancestor_rows, BASE_FIELDS)
    derive_for_ancestor(root, "positives_real")

    child_fields = BASE_FIELDS + ["ref_voice"]
    child_rows = [
        {
            "filename": "clip1__voiceA.wav",
            "path": "audio/clip1__voiceA.wav",
            "label": "_wakeword_",
            "duration": f"{duration + 0.02:.6f}",  # small VC drift, within margin
            "sample_rate": "16000",
            "resampled": "False",
            "source_dataset": "cosyvoice_conversion",
            "source_relpath": "picovoice/audio/computer/clip1.wav",
            "group_id": "clip1",
            "split": "train",
            "ref_voice": "voiceA",
        }
    ]
    _write_manifest(child_dir / "manifest.csv", child_rows, child_fields)

    stats = derive_for_join(root, "positives_converted", "positives_real", ("group_id",))
    assert stats["rows"] == 1
    assert stats["mapped"] == 1
    assert stats["missing_join"] == 0
    assert stats["failed_sanity"] == 0

    written = list(csv.DictReader((child_dir / "manifest.csv").open()))[0]
    ancestor_written = list(csv.DictReader((ancestor_dir / "manifest.csv").open()))[0]
    assert written[SPAN_COLUMNS[0]] == ancestor_written[SPAN_COLUMNS[0]]
    assert written[SPAN_COLUMNS[1]] == ancestor_written[SPAN_COLUMNS[1]]


def test_derive_for_join_leaves_span_empty_on_missing_join_key(tmp_path):
    root = tmp_path / "wakeword"
    ancestor_dir = root / "positives_real"
    child_dir = root / "positives_converted"

    duration = _silence_tone_silence_wav(ancestor_dir / "audio" / "picovoice" / "clip1.wav")
    _write_manifest(
        ancestor_dir / "manifest.csv",
        [
            {
                "filename": "clip1.wav",
                "path": "audio/picovoice/clip1.wav",
                "label": "_wakeword_",
                "duration": f"{duration:.6f}",
                "sample_rate": "16000",
                "resampled": "False",
                "source_dataset": "picovoice",
                "source_relpath": "audio/computer/clip1.wav",
                "group_id": "clip1",
                "split": "train",
            }
        ],
        BASE_FIELDS,
    )
    derive_for_ancestor(root, "positives_real")

    child_fields = BASE_FIELDS + ["ref_voice"]
    _write_manifest(
        child_dir / "manifest.csv",
        [
            {
                "filename": "orphan__voiceB.wav",
                "path": "audio/orphan__voiceB.wav",
                "label": "_wakeword_",
                "duration": f"{duration:.6f}",
                "sample_rate": "16000",
                "resampled": "False",
                "source_dataset": "cosyvoice_conversion",
                "source_relpath": "picovoice/audio/computer/orphan.wav",
                "group_id": "no-such-ancestor",  # deliberately unmatched
                "split": "train",
                "ref_voice": "voiceB",
            }
        ],
        child_fields,
    )

    stats = derive_for_join(root, "positives_converted", "positives_real", ("group_id",))
    assert stats["mapped"] == 0
    assert stats["missing_join"] == 1

    written = list(csv.DictReader((child_dir / "manifest.csv").open()))[0]
    assert written[SPAN_COLUMNS[0]] == ""
    assert written[SPAN_COLUMNS[1]] == ""


def test_derive_for_join_sanity_gate_rejects_span_past_duration(tmp_path):
    root = tmp_path / "wakeword"
    ancestor_dir = root / "positives_real"
    child_dir = root / "positives_converted"

    duration = _silence_tone_silence_wav(ancestor_dir / "audio" / "picovoice" / "clip1.wav")
    _write_manifest(
        ancestor_dir / "manifest.csv",
        [
            {
                "filename": "clip1.wav",
                "path": "audio/picovoice/clip1.wav",
                "label": "_wakeword_",
                "duration": f"{duration:.6f}",
                "sample_rate": "16000",
                "resampled": "False",
                "source_dataset": "picovoice",
                "source_relpath": "audio/computer/clip1.wav",
                "group_id": "clip1",
                "split": "train",
            }
        ],
        BASE_FIELDS,
    )
    derive_for_ancestor(root, "positives_real")

    # Child row is implausibly short -- shorter than the ancestor's detected
    # span end minus the margin -- so the sanity gate must reject the map.
    child_fields = BASE_FIELDS + ["ref_voice"]
    _write_manifest(
        child_dir / "manifest.csv",
        [
            {
                "filename": "clip1__voiceA.wav",
                "path": "audio/clip1__voiceA.wav",
                "label": "_wakeword_",
                "duration": "0.05",
                "sample_rate": "16000",
                "resampled": "False",
                "source_dataset": "cosyvoice_conversion",
                "source_relpath": "picovoice/audio/computer/clip1.wav",
                "group_id": "clip1",
                "split": "train",
                "ref_voice": "voiceA",
            }
        ],
        child_fields,
    )

    stats = derive_for_join(root, "positives_converted", "positives_real", ("group_id",))
    assert stats["mapped"] == 0
    assert stats["failed_sanity"] == 1

    written = list(csv.DictReader((child_dir / "manifest.csv").open()))[0]
    assert written[SPAN_COLUMNS[0]] == ""
    assert written[SPAN_COLUMNS[1]] == ""


def test_adversaries_are_not_in_the_precompute_pipeline():
    """SPEC.md round 3: adversaries/adversaries_noisy are deliberately
    excluded -- this locks that decision in as a regression guard, not
    just documentation."""
    assert "adversaries" not in ANCESTOR_SUBSETS
    assert "adversaries_noisy" not in ANCESTOR_SUBSETS
    derived_children = {child for child, _ancestor, _keys in DERIVED_SUBSETS}
    assert "adversaries" not in derived_children
    assert "adversaries_noisy" not in derived_children
