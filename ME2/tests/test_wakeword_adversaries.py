"""Tests for me2_voicegen.wakeword.generate_adversaries.

Fast tests here never load a real TTS model or Whisper -- `create_synthesizer`
and `transcribe_cached` are monkeypatched to a stub, mirroring
tests/test_wakeword_fetch_positives.py's offline-by-default convention. The
one real-model test is marked `@pytest.mark.slow` (this repo's pyproject.toml
already declares that marker and defaults `-m 'not slow'`), following
tests/test_personas_end_to_end_slow.py's skip-when-weights-absent pattern.
"""

from __future__ import annotations

import re
import struct
import wave
from pathlib import Path

import numpy as np
import pytest

from me2_voicegen.generation.download_model import missing_manifest_entries, target_dir
from me2_voicegen.synthesis.base import SynthesisResult
from me2_voicegen.synthesis.cosyvoice2_backend import CosyVoice2Synthesizer
from me2_voicegen.wakeword import fetch_positives as fp
from me2_voicegen.wakeword import generate_adversaries as ga


def write_wav(path: Path, *, sample_rate=16000, channels=1, sampwidth=2, n_frames=1600) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as w:
        w.setnchannels(channels)
        w.setsampwidth(sampwidth)
        w.setframerate(sample_rate)
        w.writeframes(struct.pack("<" + "h" * (n_frames * channels), *([0] * (n_frames * channels))))


def make_refs_dir(tmp_path: Path, names: list[str]) -> Path:
    refs_dir = tmp_path / "refs"
    for name in names:
        write_wav(refs_dir / name)
    return refs_dir


class StubSynthesizer:
    """Records every (text, prompt) call it receives and returns a fixed,
    short, silent SynthesisResult at a configurable sample rate."""

    def __init__(self, sample_rate: int = 22050, fail_on: set[str] | None = None):
        self.sample_rate = sample_rate
        self.fail_on = fail_on or set()
        self.calls: list[tuple[str, object]] = []

    def synthesize(self, text, prompt=None):
        self.calls.append((text, prompt))
        if text in self.fail_on:
            raise RuntimeError(f"stub failure for {text!r}")
        n = int(0.3 * self.sample_rate)
        audio = np.zeros((1, n), dtype=np.float32)
        return SynthesisResult(audio=audio, sample_rate=self.sample_rate)


# ---------------------------------------------------------------------------
# Phrase-list constant
# ---------------------------------------------------------------------------


def test_adversary_phrases_contain_no_hey_token():
    for phrase, _justification in ga.ADVERSARY_PHRASES:
        assert not re.search(r"\bhey\b", phrase, re.IGNORECASE), phrase


def test_adversary_phrases_have_nonempty_justification():
    assert len(ga.ADVERSARY_PHRASES) == 7
    seen = set()
    for phrase, justification in ga.ADVERSARY_PHRASES:
        assert phrase and isinstance(phrase, str)
        assert justification and isinstance(justification, str)
        assert phrase not in seen
        seen.add(phrase)


def test_adversary_phrases_only_keyword_is_computer():
    assert not any("alexa" in p.lower() or "jarvis" in p.lower() for p, _ in ga.ADVERSARY_PHRASES)


# ---------------------------------------------------------------------------
# Reference-voice pool discovery / selection
# ---------------------------------------------------------------------------


def test_list_reference_voices_filters_extensions_and_sorts(tmp_path):
    refs_dir = tmp_path / "refs"
    refs_dir.mkdir()
    write_wav(refs_dir / "b.wav")
    write_wav(refs_dir / "a.wav")
    (refs_dir / "notes.txt").write_text("not audio")
    (refs_dir / "c.mp3").write_bytes(b"not-a-real-mp3-but-has-the-right-suffix")

    files = ga.list_reference_voices(refs_dir)
    assert [p.name for p in files] == ["a.wav", "b.wav", "c.mp3"]


def test_list_reference_voices_raises_on_missing_dir(tmp_path):
    with pytest.raises(ValueError):
        ga.list_reference_voices(tmp_path / "does-not-exist")


def test_list_reference_voices_raises_on_empty_dir(tmp_path):
    refs_dir = tmp_path / "refs"
    refs_dir.mkdir()
    with pytest.raises(ValueError):
        ga.list_reference_voices(refs_dir)


def test_select_reference_voices_returns_full_pool_when_max_none(tmp_path):
    refs_dir = make_refs_dir(tmp_path, ["a.wav", "b.wav", "c.wav"])
    files = ga.list_reference_voices(refs_dir)
    assert ga.select_reference_voices(files, None, seed=0) == files


def test_select_reference_voices_returns_full_pool_when_max_exceeds_size(tmp_path):
    refs_dir = make_refs_dir(tmp_path, ["a.wav", "b.wav"])
    files = ga.list_reference_voices(refs_dir)
    assert ga.select_reference_voices(files, 10, seed=0) == files


def test_select_reference_voices_seeded_reproducible(tmp_path):
    refs_dir = make_refs_dir(tmp_path, [f"v{i}.wav" for i in range(10)])
    files = ga.list_reference_voices(refs_dir)

    first = ga.select_reference_voices(files, 3, seed=42)
    second = ga.select_reference_voices(files, 3, seed=42)
    assert first == second
    assert len(first) == 3
    assert set(first) <= set(files)


def test_select_reference_voices_rejects_zero_or_negative():
    with pytest.raises(ValueError):
        ga.select_reference_voices([Path("a.wav"), Path("b.wav")], 0, seed=0)


# ---------------------------------------------------------------------------
# Job planning
# ---------------------------------------------------------------------------


def test_build_jobs_group_id_is_phrase_and_count_is_cartesian(tmp_path):
    refs_dir = make_refs_dir(tmp_path, ["a.wav", "b.wav"])
    files = ga.list_reference_voices(refs_dir)
    jobs = ga.build_jobs(ga.ADVERSARY_PHRASES, files, refs_dir)

    assert len(jobs) == len(ga.ADVERSARY_PHRASES) * len(files)
    phrases = {p for p, _ in ga.ADVERSARY_PHRASES}
    for job in jobs:
        assert job.group_id == job.phrase
        assert job.phrase in phrases
        assert job.filename.endswith(".wav")
        assert job.source_relpath in ("a.wav", "b.wav")


def test_build_jobs_rejects_path_traversal(tmp_path):
    refs_dir = tmp_path / "refs"
    refs_dir.mkdir()
    outside = tmp_path / "outside.wav"
    write_wav(outside)

    with pytest.raises(fp.PathTraversalError):
        ga.build_jobs(ga.ADVERSARY_PHRASES, [outside], refs_dir)


def test_build_jobs_raises_on_duplicate_generated_filename(tmp_path):
    refs_dir = make_refs_dir(tmp_path, ["dup.wav"])
    dup_path = refs_dir / "dup.wav"
    with pytest.raises(fp.ManifestValidationError):
        ga.build_jobs(ga.ADVERSARY_PHRASES[:1], [dup_path, dup_path], refs_dir)


# ---------------------------------------------------------------------------
# run_job: format verification / resampling
# ---------------------------------------------------------------------------


def test_run_job_resamples_and_verifies_final_format(tmp_path):
    refs_dir = make_refs_dir(tmp_path, ["ref.wav"])
    ref_path = refs_dir / "ref.wav"
    jobs = ga.build_jobs(ga.ADVERSARY_PHRASES[:1], [ref_path], refs_dir)
    job = jobs[0]

    staging_root = tmp_path / "staging"
    staging_root.mkdir()
    synth = StubSynthesizer(sample_rate=22050)

    row = ga.run_job(job, synth, ref_path, "a reference transcript", staging_root)

    assert row["sample_rate"] == "16000"
    assert row["resampled"] == "True"
    assert row["label"] == ga.LABEL_UNKNOWN
    assert row["source_dataset"] == ga.SOURCE_DATASET
    assert row["group_id"] == job.phrase

    dest = staging_root / row["path"]
    assert dest.is_file()
    probe = fp.probe_wav(dest)
    assert (probe.sample_rate, probe.channels, probe.sampwidth) == (16000, 1, 2)

    # prompt actually carried through to the stub
    assert synth.calls[0][0] == job.phrase
    assert synth.calls[0][1].wav_path == ref_path
    assert synth.calls[0][1].text == "a reference transcript"


def test_run_job_no_resample_when_already_16k(tmp_path):
    refs_dir = make_refs_dir(tmp_path, ["ref.wav"])
    ref_path = refs_dir / "ref.wav"
    jobs = ga.build_jobs(ga.ADVERSARY_PHRASES[:1], [ref_path], refs_dir)
    job = jobs[0]

    staging_root = tmp_path / "staging"
    staging_root.mkdir()
    synth = StubSynthesizer(sample_rate=16000)

    row = ga.run_job(job, synth, ref_path, "text", staging_root)
    assert row["resampled"] == "False"
    assert row["sample_rate"] == "16000"


def test_run_job_raises_on_channel_mismatch(tmp_path):
    refs_dir = make_refs_dir(tmp_path, ["ref.wav"])
    ref_path = refs_dir / "ref.wav"
    jobs = ga.build_jobs(ga.ADVERSARY_PHRASES[:1], [ref_path], refs_dir)
    job = jobs[0]

    staging_root = tmp_path / "staging"
    staging_root.mkdir()

    class StereoStub:
        def synthesize(self, text, prompt=None):
            audio = np.zeros((2, 4800), dtype=np.float32)
            return SynthesisResult(audio=audio, sample_rate=16000)

    with pytest.raises(RuntimeError):
        ga.run_job(job, StereoStub(), ref_path, "text", staging_root)


# ---------------------------------------------------------------------------
# resolve_prompt_wav: trims prompts CosyVoice2's tokenizer would reject
# ---------------------------------------------------------------------------


def test_resolve_prompt_wav_passes_through_short_clip(tmp_path):
    import soundfile as sf

    path = tmp_path / "short.wav"
    sf.write(str(path), np.zeros(16000, dtype=np.float32), 16000)
    cache_dir = tmp_path / "cache"
    assert ga.resolve_prompt_wav(path, cache_dir) == path
    assert not cache_dir.exists()


def test_resolve_prompt_wav_trims_long_clip(tmp_path):
    import soundfile as sf

    path = tmp_path / "long.wav"
    sf.write(str(path), np.zeros(16000 * 40, dtype=np.float32), 16000)
    cache_dir = tmp_path / "cache"

    trimmed = ga.resolve_prompt_wav(path, cache_dir)
    assert trimmed != path
    assert trimmed.is_file()
    info = sf.info(str(trimmed))
    assert info.duration <= ga.MAX_PROMPT_SECONDS + 1e-6

    # cached on second call, no re-trim
    trimmed_again = ga.resolve_prompt_wav(path, cache_dir)
    assert trimmed_again == trimmed


# ---------------------------------------------------------------------------
# CLI: --dry-run and full main() flow, stubbed synthesizer/transcription
# ---------------------------------------------------------------------------


def test_dry_run_reports_planned_count_without_loading_model(tmp_path, monkeypatch, capsys):
    refs_dir = make_refs_dir(tmp_path, ["a.wav", "b.wav", "c.wav"])

    def _boom(*args, **kwargs):
        raise AssertionError("create_synthesizer must not be called on --dry-run")

    monkeypatch.setattr(ga, "create_synthesizer", _boom)
    monkeypatch.setattr(ga, "transcribe_cached", _boom)

    exit_code = ga.main(
        [
            "--refs-dir",
            str(refs_dir),
            "--out-root",
            str(tmp_path / "out"),
            "--dry-run",
        ]
    )
    assert exit_code == 0
    out = capsys.readouterr().out
    expected = len(ga.ADVERSARY_PHRASES) * 3
    assert str(expected) in out
    assert not (tmp_path / "out").exists()


def test_main_full_flow_writes_manifest_and_summary(tmp_path, monkeypatch):
    refs_dir = make_refs_dir(tmp_path, ["voice1.wav", "voice2.wav"])
    out_root = tmp_path / "out" / "adversaries"

    stub = StubSynthesizer(sample_rate=22050)
    monkeypatch.setattr(ga, "create_synthesizer", lambda name, **config: stub)
    monkeypatch.setattr(ga, "transcribe_cached", lambda path, model: "a fake transcript")

    exit_code = ga.main(
        [
            "--refs-dir",
            str(refs_dir),
            "--out-root",
            str(out_root),
        ]
    )
    assert exit_code == 0

    manifest_path = out_root / "manifest.csv"
    assert manifest_path.is_file()

    import csv

    with manifest_path.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    assert list(rows[0].keys()) == fp.MANIFEST_FIELDS
    assert len(rows) == len(ga.ADVERSARY_PHRASES) * 2
    assert {r["label"] for r in rows} == {ga.LABEL_UNKNOWN}
    assert {r["source_dataset"] for r in rows} == {ga.SOURCE_DATASET}
    assert {r["group_id"] for r in rows} == {p for p, _ in ga.ADVERSARY_PHRASES}
    for r in rows:
        assert (out_root / r["path"]).is_file()
        assert r["split"] == ""

    assert (out_root / "summary.md").is_file()
    summary = (out_root / "summary.md").read_text(encoding="utf-8")
    for phrase, _ in ga.ADVERSARY_PHRASES:
        assert phrase in summary

    staging = out_root.parent / f".{out_root.name}.staging"
    assert not staging.exists()


def test_main_partial_failure_writes_successful_rows_and_returns_nonzero(tmp_path, monkeypatch):
    refs_dir = make_refs_dir(tmp_path, ["voice1.wav"])
    out_root = tmp_path / "out" / "adversaries"

    stub = StubSynthesizer(sample_rate=16000, fail_on={"commuter"})
    monkeypatch.setattr(ga, "create_synthesizer", lambda name, **config: stub)
    monkeypatch.setattr(ga, "transcribe_cached", lambda path, model: "a fake transcript")

    exit_code = ga.main(["--refs-dir", str(refs_dir), "--out-root", str(out_root)])
    assert exit_code == 1

    import csv

    with (out_root / "manifest.csv").open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == len(ga.ADVERSARY_PHRASES) - 1
    assert "commuter" not in {r["group_id"] for r in rows}


def test_main_all_jobs_failing_writes_nothing(tmp_path, monkeypatch):
    refs_dir = make_refs_dir(tmp_path, ["voice1.wav"])
    out_root = tmp_path / "out" / "adversaries"

    stub = StubSynthesizer(sample_rate=16000, fail_on={p for p, _ in ga.ADVERSARY_PHRASES})
    monkeypatch.setattr(ga, "create_synthesizer", lambda name, **config: stub)
    monkeypatch.setattr(ga, "transcribe_cached", lambda path, model: "a fake transcript")

    exit_code = ga.main(["--refs-dir", str(refs_dir), "--out-root", str(out_root)])
    assert exit_code == 1
    assert not out_root.exists()
    staging = out_root.parent / f".{out_root.name}.staging"
    assert not staging.exists()


# ---------------------------------------------------------------------------
# Source-scan: no backend-specific identifier outside DEFAULT_BACKEND's value
# (mirrors tests/test_generate_sample_cli.py's
# test_no_backend_specific_identifiers_in_generate_sample_source)
# ---------------------------------------------------------------------------


def test_no_backend_specific_identifiers_in_generate_adversaries_source():
    source = Path(ga.__file__).read_text()
    lines = source.splitlines()

    def _occurrences(token: str) -> list[str]:
        return [
            line
            for line in lines
            if token.lower() in line.lower() and "DEFAULT_BACKEND" not in line
        ]

    assert _occurrences("cosyvoice2") == []
    assert _occurrences("AutoModel") == []
    assert _occurrences("inference_zero_shot") == []
    vendor_package_imports = [
        line
        for line in lines
        if ("import cosyvoice" in line or "from cosyvoice." in line or "from cosyvoice import" in line)
        and "cosyvoice_env" not in line
    ]
    assert vendor_package_imports == []


# ---------------------------------------------------------------------------
# Real-model coverage (slow, skipped when weights/vendor asset are absent --
# see this ticket's "Must Verify" re a cheap smoke test before a full run)
# ---------------------------------------------------------------------------


def _slow_skip_reasons() -> list[str]:
    reasons = []
    model_dir = target_dir()
    if not model_dir.is_dir():
        reasons.append("<entire model_dir missing>")
    else:
        reasons.extend(missing_manifest_entries(model_dir))
    if not CosyVoice2Synthesizer.DEFAULT_PROMPT_WAV.is_file():
        reasons.append(f"vendored asset missing: {CosyVoice2Synthesizer.DEFAULT_PROMPT_WAV}")
    return reasons


@pytest.mark.slow
@pytest.mark.skipif(
    bool(_slow_skip_reasons()),
    reason=(
        "real CosyVoice2-0.5B weights and/or the vendored CosyVoice asset are "
        f"not present/complete (missing: {_slow_skip_reasons()}); run "
        "`make download-model` and ensure vendor/CosyVoice is cloned first. "
        "Not mocked per this test's explicit scope."
    ),
)
def test_generate_adversaries_cli_real_backend_one_phrase_per_voice(tmp_path):
    """Real, end-to-end smoke test: the actual CosyVoice2 backend, the real
    vendored zero-shot prompt asset as the (single-clip) reference-voice
    pool, real Whisper transcription skipped via a pre-supplied transcript
    sidecar. Bounded to one voice via --max-voices-per-phrase so this stays
    a "cheap 1-clip-per-phrase" smoke test, not a full 245-clip run."""
    refs_dir = tmp_path / "refs"
    refs_dir.mkdir()
    ref_wav = refs_dir / "zero_shot_prompt.wav"
    ref_wav.write_bytes(CosyVoice2Synthesizer.DEFAULT_PROMPT_WAV.read_bytes())
    ref_wav.with_suffix(".txt").write_text(
        CosyVoice2Synthesizer.DEFAULT_PROMPT_TEXT, encoding="utf-8"
    )

    out_root = tmp_path / "out" / "adversaries"

    exit_code = ga.main(
        [
            "--refs-dir",
            str(refs_dir),
            "--out-root",
            str(out_root),
            "--max-voices-per-phrase",
            "1",
            "--backend",
            "cosyvoice2",
            "--opt",
            f"model_dir={target_dir()}",
            "--opt",
            "fp16=false",
        ]
    )

    assert exit_code == 0

    import csv

    with (out_root / "manifest.csv").open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == len(ga.ADVERSARY_PHRASES)
    for row in rows:
        dest = out_root / row["path"]
        assert dest.is_file()
        probe = fp.probe_wav(dest)
        assert (probe.sample_rate, probe.channels, probe.sampwidth) == (16000, 1, 2)
        assert probe.duration > 0.0
