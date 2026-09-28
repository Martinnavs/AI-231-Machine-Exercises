"""Unit tests for common.ambient_mix (SPEC Proof #1).

Covers the plan/mix/selection core determinism and correctness contracts:
`plan_jobs` determinism under a fixed seed, distinct chunks per row, realized
p_mix and SNR-uniformity over 10k draws, the option-B `select_passing` matrix
(independent per-attempt gating, missing report == fail), v1-pair-unsafe
transcript rejection, and `mix_one` output format + measured SNR.
"""

from __future__ import annotations

import csv
import math
from pathlib import Path

import pytest
import torch
import torchaudio

from me2_voicegen.common import ambient_mix
from me2_voicegen.common.ambient_mix import (
    AmbientJob,
    AmbientNoiseError,
    Chunk,
    ambient_filename,
    expected_text_for,
    job_key,
    load_chunk_pool,
    mix_one,
    plan_jobs,
    select_passing,
    v1_pair_violations,
    validate_v1_pair_text,
)

SR = 16000


def _sine(duration_s: float = 1.0, freq_hz: float = 440.0, amp: float = 0.5) -> torch.Tensor:
    n = int(duration_s * SR)
    t = torch.arange(n, dtype=torch.float32) / SR
    return amp * torch.sin(2 * torch.pi * freq_hz * t)


def _white_noise(duration_s: float = 0.5, seed: int = 0) -> torch.Tensor:
    n = int(duration_s * SR)
    g = torch.Generator().manual_seed(seed)
    return torch.randn(n, generator=g)


def _write_wav(path: Path, wave: torch.Tensor, sr: int = SR) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if wave.dim() == 1:
        wave = wave.unsqueeze(0)
    torchaudio.save(str(path), wave, sr, bits_per_sample=16)


def _measure_snr_db(clean: torch.Tensor, noisy: torch.Tensor) -> float:
    noise_component = noisy - clean
    signal_power = clean.pow(2).mean()
    noise_power = noise_component.pow(2).mean()
    return 10.0 * math.log10((signal_power / noise_power).item())


# ---------------------------------------------------------------------------
# Fixtures: fake rows / chunks / pools (no filesystem, plan-time only).
# ---------------------------------------------------------------------------


def _vcm_row(filename: str, split: str = "train", transcript: str = "hello world") -> dict:
    return {
        "filename": filename,
        "split": split,
        "group_id": filename.rsplit(".", 1)[0],
        "transcript": transcript,
        "label": "",
        "source_dataset": "optionb",
    }


def _ww_row(filename: str, label: str, split: str = "train") -> dict:
    return {
        "filename": filename,
        "split": split,
        "group_id": filename.rsplit(".", 1)[0],
        "label": label,
        "transcript": "",
        "source_dataset": "adversaries_tts",
    }


def _chunk(chunk_id: str, split: str, path: Path | None = None) -> Chunk:
    return Chunk(
        chunk_id=chunk_id,
        path=path or Path(f"/fake/{chunk_id}.wav"),
        split=split,
        start_s=0.0,
        duration=2.0,
    )


def _pools_for(splits: list[str], per_split: int = 4) -> dict[str, list[Chunk]]:
    return {
        s: [_chunk(f"{s}-c{i}", s) for i in range(per_split)] for s in splits
    }


# ---------------------------------------------------------------------------
# plan_jobs: determinism, distinct chunks, p_mix, SNR uniformity.
# ---------------------------------------------------------------------------


def test_plan_jobs_deterministic_under_fixed_seed():
    rows = [_vcm_row(f"r{i:04d}.wav") for i in range(50)]
    pools = _pools_for(["train"])
    a = plan_jobs("vcm", rows, pools, p_mix=0.5, seed=7)
    b = plan_jobs("vcm", rows, pools, p_mix=0.5, seed=7)
    assert a == b
    assert len(a) > 0
    assert all(ja.chunk.chunk_id == jb.chunk.chunk_id for ja, jb in zip(a, b))
    assert all(ja.snr_db == jb.snr_db for ja, jb in zip(a, b))


def test_plan_jobs_different_seeds_diverge():
    rows = [_vcm_row(f"r{i:04d}.wav") for i in range(50)]
    pools = _pools_for(["train"])
    a = plan_jobs("vcm", rows, pools, p_mix=0.5, seed=1)
    b = plan_jobs("vcm", rows, pools, p_mix=0.5, seed=2)
    # Either the selected rows differ or a chunk/SNR draw differs.
    assert a != b


def test_plan_jobs_p_mix_realized_over_10k_rows():
    rows = [_vcm_row(f"r{i:05d}.wav") for i in range(10_000)]
    pools = _pools_for(["train"])
    jobs = plan_jobs("vcm", rows, pools, p_mix=0.5, seed=0)
    selected = {id(job.source_row) for job in jobs}
    frac = len(selected) / len(rows)
    assert abs(frac - 0.5) < 0.03  # ~6 sigma for n=10k, p=0.5


def test_plan_jobs_each_selected_row_has_two_jobs():
    rows = [_vcm_row(f"r{i:04d}.wav") for i in range(200)]
    pools = _pools_for(["train"])
    jobs = plan_jobs("vcm", rows, pools, p_mix=1.0, seed=3)
    assert len(jobs) == 2 * len(rows)
    by_row: dict[int, list[AmbientJob]] = {}
    for j in jobs:
        by_row.setdefault(id(j.source_row), []).append(j)
    for row_jobs in by_row.values():
        attempts = sorted(j.attempt for j in row_jobs)
        assert attempts == [1, 2]


def test_plan_jobs_distinct_chunks_per_row():
    rows = [_vcm_row(f"r{i:04d}.wav") for i in range(200)]
    pools = _pools_for(["train"], per_split=3)
    jobs = plan_jobs("vcm", rows, pools, p_mix=1.0, seed=5)
    by_row: dict[int, list[AmbientJob]] = {}
    for j in jobs:
        by_row.setdefault(id(j.source_row), []).append(j)
    for row_jobs in by_row.values():
        ids = {j.chunk.chunk_id for j in row_jobs}
        assert len(ids) == 2  # the two attempts never share a chunk


def test_plan_jobs_snr_uniform_over_10k_draws():
    rows = [_vcm_row(f"r{i:05d}.wav") for i in range(10_000)]
    pools = _pools_for(["train"])
    jobs = plan_jobs("vcm", rows, pools, p_mix=0.5, seed=0)
    snrs = [j.snr_db for j in jobs]
    assert len(snrs) >= 8_000
    assert min(snrs) >= ambient_mix.SNR_MIN_DB
    assert max(snrs) <= ambient_mix.SNR_MAX_DB
    mean = sum(snrs) / len(snrs)
    assert abs(mean - 15.0) < 0.5  # E[unif(0,30)] = 15.0
    frac_low = sum(1 for s in snrs if s < 5.0) / len(snrs)
    assert 0.12 < frac_low < 0.21  # P(snr<5) = 5/30 ~= 0.167


def test_plan_jobs_split_pool_respected():
    # Rows in each split must only draw from their own split's pool.
    rows = [_vcm_row(f"tr{i:03d}.wav", split="train") for i in range(50)] + [
        _vcm_row(f"va{i:03d}.wav", split="val") for i in range(50)
    ]
    pools = _pools_for(["train", "val"], per_split=3)
    jobs = plan_jobs("vcm", rows, pools, p_mix=1.0, seed=11)
    for j in jobs:
        assert j.chunk.split == j.source_row["split"]
        assert j.chunk.chunk_id.startswith(j.source_row["split"])


def test_plan_jobs_rejects_bad_params():
    rows = [_vcm_row("r.wav")]
    pools = _pools_for(["train"])
    with pytest.raises(AmbientNoiseError):
        plan_jobs("vcm", rows, pools, p_mix=1.5, seed=0)
    with pytest.raises(AmbientNoiseError):
        plan_jobs("vcm", rows, pools, p_mix=-0.1, seed=0)
    with pytest.raises(AmbientNoiseError):
        plan_jobs("vcm", rows, pools, snr_min_db=30.0, snr_max_db=-5.0, seed=0)


def test_plan_jobs_missing_split_pool_is_loud():
    rows = [_vcm_row("r.wav", split="test")]
    pools = _pools_for(["train"])  # no "test" pool
    with pytest.raises(AmbientNoiseError):
        plan_jobs("vcm", rows, pools, p_mix=1.0, seed=0)


# ---------------------------------------------------------------------------
# select_passing: option-B independent per-attempt gating.
# ---------------------------------------------------------------------------


def _two_jobs() -> list[AmbientJob]:
    row = _vcm_row("f.wav")
    c1 = _chunk("train-c0", "train")
    c2 = _chunk("train-c1", "train")
    return [
        AmbientJob("vcm", row, 1, c1, 10.0, "hello"),
        AmbientJob("vcm", row, 2, c2, 20.0, "hello"),
    ]


def test_select_passing_both_pass_keeps_two():
    jobs = _two_jobs()
    passed = {job_key(jobs[0]): True, job_key(jobs[1]): True}
    kept = select_passing(jobs, passed)
    assert len(kept) == 2


def test_select_passing_pass1_only():
    jobs = _two_jobs()
    kept = select_passing(jobs, {job_key(jobs[0]): True})
    assert len(kept) == 1
    assert kept[0].attempt == 1


def test_select_passing_pass2_only():
    jobs = _two_jobs()
    kept = select_passing(jobs, {job_key(jobs[1]): True})
    assert len(kept) == 1
    assert kept[0].attempt == 2


def test_select_passing_both_fail_keeps_zero():
    jobs = _two_jobs()
    kept = select_passing(jobs, {job_key(jobs[0]): False, job_key(jobs[1]): False})
    assert kept == []


def test_select_passing_missing_entry_never_passes():
    jobs = _two_jobs()
    assert select_passing(jobs, {}) == []
    # A missing entry for one attempt does not let the other through.
    assert select_passing(jobs, {job_key(jobs[0]): True}) == [jobs[0]]


def test_job_key_shape():
    row = _vcm_row("f.wav")
    j = AmbientJob("vcm", row, 2, _chunk("train-c0", "train"), 12.0, "hello")
    assert job_key(j) == "f|f.wav|a2"  # group_id "f" (stem), filename "f.wav", attempt 2


# ---------------------------------------------------------------------------
# expected_text_for + v1-pair naming validation.
# ---------------------------------------------------------------------------


def test_expected_text_for_vcm_returns_transcript():
    assert expected_text_for("vcm", _vcm_row("r.wav", transcript="open the door")) == "open the door"


def test_expected_text_for_vcm_rejects_empty_transcript():
    with pytest.raises(AmbientNoiseError):
        expected_text_for("vcm", _vcm_row("r.wav", transcript="   "))


def test_expected_text_for_wakeword():
    assert expected_text_for("wakeword", _ww_row("r.wav", "_wakeword_")) == "Sesame"
    assert expected_text_for("wakeword", _ww_row("r.wav", "_unknown_")) == ""


def test_expected_text_for_rejects_silence_and_unknown_model():
    with pytest.raises(AmbientNoiseError):
        expected_text_for("wakeword", _ww_row("r.wav", "_silence_"))
    with pytest.raises(AmbientNoiseError):
        expected_text_for("nope", _vcm_row("r.wav"))


@pytest.mark.parametrize("bad", ["a - b", "a/b", "a\\b"])
def test_validate_v1_pair_text_rejects_unsafe_markers(bad: str):
    with pytest.raises(AmbientNoiseError):
        validate_v1_pair_text(bad, "r.wav")


def test_validate_v1_pair_text_rejects_empty():
    with pytest.raises(AmbientNoiseError):
        validate_v1_pair_text("  ", "r.wav")


def test_validate_v1_pair_text_accepts_clean():
    validate_v1_pair_text("open the door", "r.wav")  # must not raise


def test_v1_pair_violations_collects_all():
    row_bad = _vcm_row("bad.wav", transcript="a - b")
    row_bad2 = _vcm_row("bad2.wav", transcript="c/d")
    row_ok = _vcm_row("ok.wav", transcript="fine")
    c = _chunk("train-c0", "train")
    jobs = [
        AmbientJob("vcm", row_bad, 1, c, 10.0, "a - b"),
        AmbientJob("vcm", row_bad2, 1, c, 10.0, "c/d"),
        AmbientJob("vcm", row_ok, 1, c, 10.0, "fine"),
    ]
    violations = v1_pair_violations(jobs)
    assert len(violations) == 2
    assert any("bad.wav" in v for v in violations)
    assert any("bad2.wav" in v for v in violations)
    assert not any("ok.wav" in v for v in violations)


# ---------------------------------------------------------------------------
# ambient_filename.
# ---------------------------------------------------------------------------


def test_ambient_filename_shape():
    row = _vcm_row("src clip.wav")  # group_id "src clip"
    job = AmbientJob("vcm", row, 2, _chunk("train-c3", "train"), 12.0, "hi")
    name = ambient_filename(job)
    # group slug "src-clip", source stem "src clip", attempt 2, chunk "train-c3"
    assert name == "src-clip__src-clip__amb2-train-c3.wav"
    assert name.endswith(".wav")


def test_ambient_filename_chunk_accepts_bare_stem_or_filename():
    row = _vcm_row("x.wav")
    j_stem = AmbientJob("vcm", row, 1, _chunk("abc", "train"), 1.0, "t")
    j_file = AmbientJob("vcm", row, 1, _chunk("abc", "train"), 1.0, "t")
    assert ambient_filename(j_stem) == ambient_filename_from("x", "x.wav", 1, "abc.wav")
    assert ambient_filename(j_file) == ambient_filename(j_stem)


def ambient_filename_from(group_id: str, filename: str, attempt: int, chunk_id: str) -> str:
    return ambient_mix.ambient_filename_from(group_id, filename, attempt, chunk_id)


# ---------------------------------------------------------------------------
# load_chunk_pool.
# ---------------------------------------------------------------------------


def _make_corpus(root: Path, chunks: list[tuple[str, str]]) -> Path:
    """chunks: (filename, split). Writes manifest.csv + audio/<filename>."""
    (root / "audio").mkdir(parents=True, exist_ok=True)
    with (root / "manifest.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(ambient_mix.CORPUS_MANIFEST_FIELDS)
        for name, split in chunks:
            _write_wav(root / "audio" / name, _white_noise(1.0, seed=hash(name) % 1000))
            w.writerow([name, "raw1.wav", split, "0.0", "1.0", "0.01"])
    return root


def test_load_chunk_pool_filters_by_split(tmp_path: Path):
    root = _make_corpus(
        tmp_path / "corpus",
        [("train-a.wav", "train"), ("train-b.wav", "train"), ("val-a.wav", "val")],
    )
    train = load_chunk_pool(root, "train")
    val = load_chunk_pool(root, "val")
    assert {c.chunk_id for c in train} == {"train-a", "train-b"}
    assert all(c.split == "train" for c in train)
    assert {c.chunk_id for c in val} == {"val-a"}


def test_load_chunk_pool_missing_manifest_raises(tmp_path: Path):
    empty = tmp_path / "empty"
    empty.mkdir()
    with pytest.raises(AmbientNoiseError):
        load_chunk_pool(empty, "train")


def test_load_chunk_pool_missing_wav_raises(tmp_path: Path):
    root = _make_corpus(tmp_path / "corpus", [("train-a.wav", "train")])
    (root / "audio" / "train-a.wav").unlink()
    with pytest.raises(AmbientNoiseError):
        load_chunk_pool(root, "train")


def test_load_chunk_pool_rejects_bad_format(tmp_path: Path):
    root = tmp_path / "corpus"
    (root / "audio").mkdir(parents=True, exist_ok=True)
    _write_wav(root / "audio" / "bad.wav", _sine(1.0), sr=44100)  # wrong sample rate
    with (root / "manifest.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(ambient_mix.CORPUS_MANIFEST_FIELDS)
        w.writerow(["bad.wav", "raw1.wav", "train", "0.0", "1.0", "0.01"])
    with pytest.raises(AmbientNoiseError):
        load_chunk_pool(root, "train")


# ---------------------------------------------------------------------------
# mix_one: output format, measured SNR, clamping, format rejection.
# ---------------------------------------------------------------------------


def _job_from_files(source: Path, chunk: Path, snr_db: float, attempt: int = 1) -> AmbientJob:
    row = _vcm_row(source.name)
    c = Chunk(Path(chunk.name).stem, chunk, "train", 0.0, 0.5)
    return AmbientJob("vcm", row, attempt, c, snr_db, "hello")


def test_mix_one_output_format_and_length(tmp_path: Path):
    source = tmp_path / "src.wav"
    _write_wav(source, _sine(1.0))
    chunk = tmp_path / "chunk.wav"
    _write_wav(chunk, _white_noise(0.5))  # shorter than source: must loop
    out = tmp_path / "out" / "mixed.wav"
    mix_one(_job_from_files(source, chunk, 20.0), source, out)

    info = torchaudio.info(str(out))
    assert info.sample_rate == 16000
    assert info.num_channels == 1
    assert info.bits_per_sample == 16
    wave, _ = torchaudio.load(str(out))
    assert wave.shape == (1, SR)  # length preserved (1 s)


def test_mix_one_measured_snr_within_tolerance(tmp_path: Path):
    source = tmp_path / "src.wav"
    _write_wav(source, _sine(1.0))
    chunk = tmp_path / "chunk.wav"
    _write_wav(chunk, _white_noise(0.5, seed=3))
    for requested in (10.0, 20.0):
        out = tmp_path / f"out_{requested}.wav"
        mix_one(_job_from_files(source, chunk, requested), source, out)
        clean, _ = torchaudio.load(str(source))
        mixed, _ = torchaudio.load(str(out))
        measured = _measure_snr_db(clean[0], mixed[0])
        assert abs(measured - requested) < 1.0


def test_mix_one_clamps_at_negative_snr(tmp_path: Path):
    source = tmp_path / "src.wav"
    _write_wav(source, _sine(1.0, amp=0.5))
    chunk = tmp_path / "chunk.wav"
    _write_wav(chunk, _white_noise(0.5, seed=3))
    out = tmp_path / "out_neg.wav"
    mix_one(_job_from_files(source, chunk, -5.0), source, out)
    mixed, _ = torchaudio.load(str(out))
    assert mixed.max().item() <= 1.0
    assert mixed.min().item() >= -1.0
    # At -5 dB the noise dominates, so clipping must actually occur.
    assert mixed.max().item() == pytest.approx(1.0, abs=1e-3)


def test_mix_one_rejects_nonconformant_source(tmp_path: Path):
    source = tmp_path / "src44.wav"
    _write_wav(source, _sine(1.0), sr=44100)
    chunk = tmp_path / "chunk.wav"
    _write_wav(chunk, _white_noise(0.5))
    out = tmp_path / "out.wav"
    with pytest.raises(AmbientNoiseError):
        mix_one(_job_from_files(source, chunk, 20.0), source, out)


def test_mix_one_rejects_missing_source(tmp_path: Path):
    chunk = tmp_path / "chunk.wav"
    _write_wav(chunk, _white_noise(0.5))
    job = _job_from_files(tmp_path / "nope.wav", chunk, 20.0)
    with pytest.raises(AmbientNoiseError):
        mix_one(job, tmp_path / "nope.wav", tmp_path / "out.wav")
