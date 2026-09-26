from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import pytest

from me2_voicegen.accent_balance.generate import (
    GEN_FIELDS,
    load_prior_shard_manifest,
    merge_shards,
    parse_shard,
    run_shard,
)
from me2_voicegen.synthesis.base import SynthesisResult, Synthesizer
from me2_voicegen.synthesis.factory import _BACKENDS
from me2_voicegen.wakeword.fetch_positives import ManifestValidationError


class FakeSynthesizer(Synthesizer):
    calls: list[tuple] = []

    def __init__(self, voice_id: str = "default-voice") -> None:
        self.voice_id = voice_id

    def synthesize(self, text, prompt=None) -> SynthesisResult:
        FakeSynthesizer.calls.append((text, prompt))
        samples = 0.1 * np.sin(2 * np.pi * 220 * np.arange(4800, dtype=np.float32) / 16000)
        return SynthesisResult(audio=samples[np.newaxis, :], sample_rate=16000)


class RaisingSynthesizer(Synthesizer):
    def __init__(self, **kwargs) -> None:
        pass

    def synthesize(self, text, prompt=None):
        raise RuntimeError("synthesis backend exploded")


@pytest.fixture(autouse=True)
def _reset_fake():
    FakeSynthesizer.calls = []
    yield


@pytest.fixture
def registered_fake(monkeypatch):
    monkeypatch.setitem(_BACKENDS, "fake", FakeSynthesizer)


@pytest.fixture
def registered_raising(monkeypatch):
    monkeypatch.setitem(_BACKENDS, "fake", RaisingSynthesizer)


def _write_voice(refs_dir: Path, voice_id: str, wav_factory):
    refs_dir.mkdir(parents=True, exist_ok=True)
    wav_factory(refs_dir / f"{voice_id}.wav", duration_s=8.0)
    (refs_dir / f"{voice_id}.txt").write_text("hello there\n")


def _write_jobs(path: Path, jobs: list[dict]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    from me2_voicegen.accent_balance.plan_jobs import JOBS_FIELDS

    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=JOBS_FIELDS)
        writer.writeheader()
        writer.writerows(jobs)
    return path


def _job(job_id, model="vcm", voice_id="v1", text="Alarm 6 AM", label="ALARM"):
    return {
        "job_id": job_id, "model": model, "split": "train", "voice_id": voice_id,
        "text": text, "label": label, "source_row_ref": "", "noisy_target": "0",
    }


def test_parse_shard_valid_and_invalid():
    assert parse_shard("0/4") == (0, 4)
    assert parse_shard("3/4") == (3, 4)
    with pytest.raises(ValueError):
        parse_shard("bad")
    with pytest.raises(ValueError):
        parse_shard("4/4")
    with pytest.raises(ValueError):
        parse_shard("-1/4")


def test_load_prior_shard_manifest_missing_file_returns_empty(tmp_path):
    assert load_prior_shard_manifest(tmp_path / "nope.csv") == {}


def test_run_shard_synthesizes_only_its_own_jobs(tmp_path, vcm_wav_factory, registered_fake):
    refs_dir = tmp_path / "refs"
    _write_voice(refs_dir, "v1", vcm_wav_factory)
    jobs = [_job(f"job_{i}") for i in range(4)]
    jobs_csv = _write_jobs(tmp_path / "jobs.csv", jobs)
    out_dir = tmp_path / "out"

    manifest_path = run_shard(jobs_csv, out_dir, "0/2", refs_dir, device="cpu", backend="fake")

    rows = list(csv.DictReader(manifest_path.open()))
    assert {r["job_id"] for r in rows} == {"job_0", "job_2"}
    assert all(r["status"] == "ok" for r in rows)
    for r in rows:
        assert Path(r["path"]).is_file()
        assert float(r["duration"]) > 0


def test_run_shard_passes_prompt_text_and_target_text_to_synthesizer(tmp_path, vcm_wav_factory, registered_fake):
    refs_dir = tmp_path / "refs"
    _write_voice(refs_dir, "v1", vcm_wav_factory)
    jobs_csv = _write_jobs(tmp_path / "jobs.csv", [_job("job_0", text="Set a timer")])
    run_shard(jobs_csv, tmp_path / "out", "0/1", refs_dir, device="cpu", backend="fake")

    assert len(FakeSynthesizer.calls) == 1
    text, prompt = FakeSynthesizer.calls[0]
    assert text == "Set a timer"
    assert prompt.text == "hello there"
    assert prompt.wav_path == refs_dir / "v1.wav"


def test_run_shard_is_resumable_and_does_not_resynthesize(tmp_path, vcm_wav_factory, registered_fake):
    refs_dir = tmp_path / "refs"
    _write_voice(refs_dir, "v1", vcm_wav_factory)
    jobs_csv = _write_jobs(tmp_path / "jobs.csv", [_job("job_0")])
    out_dir = tmp_path / "out"

    run_shard(jobs_csv, out_dir, "0/1", refs_dir, device="cpu", backend="fake")
    assert len(FakeSynthesizer.calls) == 1

    run_shard(jobs_csv, out_dir, "0/1", refs_dir, device="cpu", backend="fake")
    assert len(FakeSynthesizer.calls) == 1  # second run: skipped, not re-synthesized


def test_run_shard_marks_failed_jobs_as_error_and_continues(tmp_path, vcm_wav_factory, registered_raising):
    refs_dir = tmp_path / "refs"
    _write_voice(refs_dir, "v1", vcm_wav_factory)
    jobs_csv = _write_jobs(tmp_path / "jobs.csv", [_job("job_0"), _job("job_1")])

    manifest_path = run_shard(jobs_csv, tmp_path / "out", "0/1", refs_dir, device="cpu", backend="fake")

    rows = {r["job_id"]: r for r in csv.DictReader(manifest_path.open())}
    assert rows["job_0"]["status"] == "error"
    assert rows["job_1"]["status"] == "error"
    assert rows["job_0"]["path"] == ""


def test_run_shard_missing_voice_prompt_marks_error_not_crash(tmp_path, vcm_wav_factory, registered_fake):
    refs_dir = tmp_path / "refs"  # no voice files written at all
    refs_dir.mkdir()
    jobs_csv = _write_jobs(tmp_path / "jobs.csv", [_job("job_0", voice_id="missing_voice")])

    manifest_path = run_shard(jobs_csv, tmp_path / "out", "0/1", refs_dir, device="cpu", backend="fake")
    rows = list(csv.DictReader(manifest_path.open()))
    assert rows[0]["status"] == "error"


def test_wakeword_job_gets_speech_span_columns(tmp_path, vcm_wav_factory, registered_fake):
    refs_dir = tmp_path / "refs"
    _write_voice(refs_dir, "v1", vcm_wav_factory)
    jobs_csv = _write_jobs(tmp_path / "jobs.csv", [_job("job_0", model="wakeword", text="Computer.", label="_wakeword_")])

    manifest_path = run_shard(jobs_csv, tmp_path / "out", "0/1", refs_dir, device="cpu", backend="fake")
    row = next(csv.DictReader(manifest_path.open()))
    assert row["status"] == "ok"
    # detect_speech_span may legitimately return None for a plain sine tone;
    # what matters is the columns are present and consistently both-empty or both-filled.
    assert (row["speech_start_s"] == "" ) == (row["speech_end_s"] == "")


def test_merge_shards_combines_disjoint_job_ids(tmp_path):
    out_dir = tmp_path / "out"
    out_dir.mkdir()
    with (out_dir / "gen_manifest.shard0.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=GEN_FIELDS)
        w.writeheader()
        w.writerow({k: "" for k in GEN_FIELDS} | {"job_id": "a", "status": "ok"})
    with (out_dir / "gen_manifest.shard1.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=GEN_FIELDS)
        w.writeheader()
        w.writerow({k: "" for k in GEN_FIELDS} | {"job_id": "b", "status": "ok"})

    dest = merge_shards(out_dir)
    rows = list(csv.DictReader(dest.open()))
    assert {r["job_id"] for r in rows} == {"a", "b"}


def test_merge_shards_raises_on_duplicate_job_id(tmp_path):
    out_dir = tmp_path / "out"
    out_dir.mkdir()
    for shard in (0, 1):
        with (out_dir / f"gen_manifest.shard{shard}.csv").open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=GEN_FIELDS)
            w.writeheader()
            w.writerow({k: "" for k in GEN_FIELDS} | {"job_id": "dup", "status": "ok"})

    with pytest.raises(ManifestValidationError):
        merge_shards(out_dir)


def test_merge_shards_raises_when_no_shards_found(tmp_path):
    out_dir = tmp_path / "empty_out"
    out_dir.mkdir()
    with pytest.raises(ManifestValidationError):
        merge_shards(out_dir)
