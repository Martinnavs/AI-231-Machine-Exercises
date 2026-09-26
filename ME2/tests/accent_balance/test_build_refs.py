from __future__ import annotations

import csv
from pathlib import Path

import pytest

from me2_voicegen.accent_balance.build_refs import (
    ManifestValidationError,
    assign_splits,
    build,
)


def _write_fsc_manifest(tmp_path: Path, wav_factory, rows: list[dict]) -> Path:
    manifest = tmp_path / "fsc" / "manifest.csv"
    manifest.parent.mkdir(parents=True, exist_ok=True)
    audio_dir = manifest.parent / "audio"
    fields = ["filename", "sentence", "duration", "speech_type", "source_file", "speaker_id", "gender", "age_group"]
    with manifest.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            wav_factory(f"fsc/audio/{row['filename']}", duration_s=float(row["duration"]))
            writer.writerow(row)
    return manifest


def _fsc_rows_for_speaker(speaker_id: str, n_clips: int, gender: str, dur: float = 2.0) -> list[dict]:
    return [
        {
            "filename": f"{speaker_id}_{i}.wav",
            "sentence": f"sentence {i} for {speaker_id}",
            "duration": str(dur),
            "speech_type": "read",
            "source_file": "",
            "speaker_id": speaker_id,
            "gender": gender,
            "age_group": "20-27",
        }
        for i in range(n_clips)
    ]


def _write_refs_dir(tmp_path: Path) -> Path:
    import numpy as np
    import soundfile as sf

    refs_dir = tmp_path / "References"
    refs_dir.mkdir(parents=True)
    sr = 16000
    n = int(20 * sr)
    t = np.arange(n, dtype=np.float32) / sr
    for name in ("tagalog1", "tagalog2", "ilonggo1"):
        data = 0.1 * np.sin(2 * 3.14159 * 220.0 * t)
        sf.write(str(refs_dir / f"{name}.wav"), data.astype(np.float32), sr)
    return refs_dir


class _FakeWhisperResult(dict):
    pass


@pytest.fixture(autouse=True)
def _stub_whisper(monkeypatch):
    """Every reference cut in these tests must produce a real, distinct
    transcript without loading the actual Whisper model (slow, network/GPU)."""

    def fake_transcribe_cached(audio_path, model_size="base"):
        text = f"transcribed {Path(audio_path).stem}"
        Path(audio_path).with_suffix(".txt").write_text(text + "\n")
        return text

    monkeypatch.setattr("me2_voicegen.accent_balance.build_refs.transcribe_cached", fake_transcribe_cached)


def test_build_writes_wav_txt_and_voices_csv(tmp_path, vcm_wav_factory):
    rows = _fsc_rows_for_speaker("s1", 6, "female") + _fsc_rows_for_speaker("s2", 6, "male")
    manifest = _write_fsc_manifest(tmp_path, vcm_wav_factory, rows)
    refs_dir = _write_refs_dir(tmp_path)
    out_dir = tmp_path / "out"

    voice_rows = build(manifest, refs_dir, out_dir, seed=0, whisper_model="small")

    voice_ids = {r["voice_id"] for r in voice_rows}
    assert voice_ids == {"fsc_s1", "fsc_s2", "ref_tagalog1", "ref_tagalog2", "ref_ilonggo1"}
    for r in voice_rows:
        assert (out_dir / f"{r['voice_id']}.wav").is_file()
        assert (out_dir / f"{r['voice_id']}.txt").is_file()
        assert r["prompt_text"].strip()
        assert 0 < float(r["prompt_seconds"]) <= 25.0


def test_sapinsapin_speaker_below_minimum_is_skipped(tmp_path, vcm_wav_factory):
    # 2 short clips (1s each) never reach the 8s floor -- speaker should be dropped, not crash.
    rows = _fsc_rows_for_speaker("s_short", 2, "female", dur=1.0)
    rows += _fsc_rows_for_speaker("s_ok", 6, "female", dur=2.0)
    manifest = _write_fsc_manifest(tmp_path, vcm_wav_factory, rows)
    refs_dir = _write_refs_dir(tmp_path)
    out_dir = tmp_path / "out"

    voice_rows = build(manifest, refs_dir, out_dir, seed=0, whisper_model="small")

    sapinsapin_ids = {r["voice_id"] for r in voice_rows if r["prompt_source"] == "sapinsapin"}
    assert sapinsapin_ids == {"fsc_s_ok"}


def test_references_voices_are_all_train(tmp_path, vcm_wav_factory):
    rows = _fsc_rows_for_speaker("s1", 6, "female")
    manifest = _write_fsc_manifest(tmp_path, vcm_wav_factory, rows)
    refs_dir = _write_refs_dir(tmp_path)
    out_dir = tmp_path / "out"

    voice_rows = build(manifest, refs_dir, out_dir, seed=0, whisper_model="small")

    ref_rows = [r for r in voice_rows if r["prompt_source"] == "references"]
    assert ref_rows and all(r["split"] == "train" for r in ref_rows)


def test_references_prompt_text_is_not_the_stale_full_clip_sidecar(tmp_path, vcm_wav_factory):
    """A pre-existing <stem>.txt next to the SOURCE mp3/wav (the full clip's
    own transcript) must never leak into the cut prompt's transcript -- the
    cut is a different file at a different path."""
    rows = _fsc_rows_for_speaker("s1", 6, "female")
    manifest = _write_fsc_manifest(tmp_path, vcm_wav_factory, rows)
    refs_dir = _write_refs_dir(tmp_path)
    (refs_dir / "tagalog1.txt").write_text("Please call Stella, full clip transcript.\n")
    out_dir = tmp_path / "out"

    voice_rows = build(manifest, refs_dir, out_dir, seed=0, whisper_model="small")

    ref1 = next(r for r in voice_rows if r["voice_id"] == "ref_tagalog1")
    assert "Stella" not in ref1["prompt_text"]
    assert ref1["prompt_text"] == "transcribed ref_tagalog1"


def test_assign_splits_is_seed_deterministic_and_disjoint():
    genders = {f"v{i}": ("female" if i % 2 == 0 else "male") for i in range(20)}
    voice_ids = list(genders)

    a = assign_splits(voice_ids, genders, seed=42)
    b = assign_splits(voice_ids, genders, seed=42)
    c = assign_splits(voice_ids, genders, seed=1)

    assert a == b
    assert set(a) == set(voice_ids)
    assert set(a.values()) <= {"train", "val", "test"}
    assert a != c  # different seed, different assignment (statistically, and true for this fixture)


def test_assign_splits_roughly_70_15_15():
    genders = {f"v{i}": "female" for i in range(100)}
    result = assign_splits(list(genders), genders, seed=0)
    counts = {"train": 0, "val": 0, "test": 0}
    for split in result.values():
        counts[split] += 1
    assert 65 <= counts["train"] <= 75
    assert 10 <= counts["val"] <= 20
    assert 10 <= counts["test"] <= 20


def test_missing_refs_dir_fails_loudly(tmp_path, vcm_wav_factory):
    rows = _fsc_rows_for_speaker("s1", 6, "female")
    manifest = _write_fsc_manifest(tmp_path, vcm_wav_factory, rows)
    with pytest.raises(ManifestValidationError):
        build(manifest, tmp_path / "does_not_exist", tmp_path / "out", seed=0, whisper_model="small")
