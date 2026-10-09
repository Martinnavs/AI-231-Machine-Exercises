from __future__ import annotations

import csv
from pathlib import Path

import pytest

from me2_voicegen.accent_balance.qa import (
    build_shim,
    parse_reports,
    prompt_source_of,
    unit_name,
    validate_shim_text,
    write_qa_pass_csv,
    write_qa_summary,
)
from me2_voicegen.wakeword.fetch_positives import ManifestValidationError

FIXTURES = Path(__file__).parent.parent / "fixtures" / "accent_balance"


def _gen_row(job_id, model="vcm", voice_id="fsc_1", text="Alarm 6 AM", status="ok", path=None):
    return {
        "job_id": job_id, "model": model, "split": "train", "voice_id": voice_id,
        "text": text, "label": "ALARM", "source_row_ref": "", "noisy_target": "0",
        "path": str(path) if path else "", "duration": "1.0", "seconds_elapsed": "0.1",
        "status": status, "speech_start_s": "", "speech_end_s": "",
    }


def _write_gen_manifest(path: Path, rows: list[dict]) -> Path:
    from me2_voicegen.accent_balance.generate import GEN_FIELDS

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=GEN_FIELDS)
        w.writeheader()
        w.writerows(rows)
    return path


def test_prompt_source_of():
    assert prompt_source_of("fsc_94") == "sapinsapin"
    assert prompt_source_of("ref_tagalog3") == "references"
    with pytest.raises(ManifestValidationError):
        prompt_source_of("weird_voice")


def test_unit_name():
    assert unit_name("vcm", "sapinsapin") == "vcm_sapinsapin"


def test_validate_shim_text_rejects_dash_separator_and_slash():
    validate_shim_text("Alarm 6 AM", "job_0")  # fine, no raise
    with pytest.raises(ManifestValidationError):
        validate_shim_text("Set - a timer", "job_1")
    with pytest.raises(ManifestValidationError):
        validate_shim_text("A/B test", "job_2")
    with pytest.raises(ManifestValidationError):
        validate_shim_text("", "job_3")


def test_build_shim_creates_v1_pair_symlinks_and_index(tmp_path, vcm_wav_factory):
    wav_path = vcm_wav_factory("clip.wav", duration_s=1.0)
    rows = [_gen_row("job_0", model="vcm", voice_id="fsc_1", text="Alarm 6 AM", path=wav_path)]
    shim_dir = tmp_path / "shim"

    unit_dirs = build_shim(rows, shim_dir)

    assert set(unit_dirs) == {"vcm_sapinsapin"}
    unit_dir = unit_dirs["vcm_sapinsapin"]
    link = unit_dir / "Alarm 6 AM - job_0.wav"
    assert link.is_symlink()
    assert link.resolve() == wav_path.resolve()
    index_rows = list(csv.DictReader((unit_dir / "index.csv").open()))
    assert index_rows == [{"filename": "Alarm 6 AM - job_0.wav", "job_id": "job_0"}]


def test_build_shim_skips_error_rows(tmp_path, vcm_wav_factory):
    wav_path = vcm_wav_factory("clip.wav")
    rows = [
        _gen_row("job_0", path=wav_path, status="ok"),
        _gen_row("job_1", status="error"),
    ]
    unit_dirs = build_shim(rows, tmp_path / "shim")
    index_rows = list(csv.DictReader((next(iter(unit_dirs.values())) / "index.csv").open()))
    assert {r["job_id"] for r in index_rows} == {"job_0"}


def test_build_shim_raises_on_unsafe_text_not_silently_rewritten(tmp_path, vcm_wav_factory):
    wav_path = vcm_wav_factory("clip.wav")
    rows = [_gen_row("job_0", text="Set - a timer", path=wav_path)]
    with pytest.raises(ManifestValidationError):
        build_shim(rows, tmp_path / "shim")


def test_build_shim_groups_by_model_and_prompt_source(tmp_path, vcm_wav_factory):
    wav_path = vcm_wav_factory("clip.wav")
    rows = [
        _gen_row("job_0", model="vcm", voice_id="fsc_1", path=wav_path),
        _gen_row("job_1", model="vcm", voice_id="ref_tagalog3", path=wav_path),
        _gen_row("job_2", model="wakeword", voice_id="fsc_1", text="Computer.", path=wav_path),
    ]
    unit_dirs = build_shim(rows, tmp_path / "shim")
    assert set(unit_dirs) == {"vcm_sapinsapin", "vcm_references", "wakeword_sapinsapin"}


def test_parse_reports_against_a_real_transcriber_report(tmp_path):
    """Uses a report actually produced by ~/simple-audio-transcriber's
    audio_transcript_parser.qa (RECAP T4: "against a real report produced by
    the tool ... don't hand-write it"), on 4 real clips: 3 named
    "Computer. - job_N.wav" (correctly transcribed) and 1 deliberately
    mislabeled "Xylophone parade. - job_3.wav" (flagged)."""
    gen_rows = [
        _gen_row("job_0", model="wakeword", voice_id="fsc_1", text="Computer.", path="x0.wav"),
        _gen_row("job_1", model="wakeword", voice_id="fsc_1", text="Computer.", path="x1.wav"),
        _gen_row("job_2", model="wakeword", voice_id="fsc_1", text="Computer.", path="x2.wav"),
        _gen_row("job_3", model="wakeword", voice_id="fsc_1", text="Xylophone parade.", path="x3.wav"),
    ]
    gen_manifest = _write_gen_manifest(tmp_path / "gen_manifest.csv", gen_rows)

    rows = parse_reports(gen_manifest, FIXTURES / "shim_index", FIXTURES / "reports_dir")

    by_id = {r["job_id"]: r for r in rows}
    assert by_id["job_0"]["passed"] == "True"
    assert by_id["job_1"]["passed"] == "True"
    assert by_id["job_2"]["passed"] == "True"
    assert by_id["job_3"]["passed"] == "False"
    assert by_id["job_3"]["flag_reason"] == "qa_flagged"


def test_parse_reports_marks_generation_errors_without_a_report(tmp_path):
    gen_rows = [_gen_row("job_err", status="error")]
    gen_manifest = _write_gen_manifest(tmp_path / "gen_manifest.csv", gen_rows)

    rows = parse_reports(gen_manifest, tmp_path / "no_shim", tmp_path / "no_reports")

    assert rows == [{"job_id": "job_err", "passed": "False", "flag_reason": "generation_error"}]


def test_parse_reports_flags_ok_jobs_missing_from_any_report(tmp_path):
    gen_rows = [_gen_row("job_orphan", status="ok", path="x.wav")]
    gen_manifest = _write_gen_manifest(tmp_path / "gen_manifest.csv", gen_rows)

    rows = parse_reports(gen_manifest, tmp_path / "empty_shim", tmp_path / "empty_reports")

    assert rows == [{"job_id": "job_orphan", "passed": "False", "flag_reason": "missing_qa_report"}]


def test_parse_reports_raises_on_shim_index_mismatch(tmp_path):
    gen_rows = [_gen_row("job_0", model="wakeword", voice_id="fsc_1", text="Computer.", status="ok", path="x.wav")]
    gen_manifest = _write_gen_manifest(tmp_path / "gen_manifest.csv", gen_rows)

    with pytest.raises(ManifestValidationError):
        parse_reports(gen_manifest, tmp_path / "wrong_shim_dir", FIXTURES / "reports_dir")


def test_write_qa_pass_csv_and_summary(tmp_path, vcm_wav_factory):
    wav_path = vcm_wav_factory("clip.wav")
    gen_rows = [
        _gen_row("job_0", model="vcm", voice_id="fsc_1", status="ok", path=wav_path),
        _gen_row("job_1", model="vcm", voice_id="fsc_1", status="error"),
    ]
    gen_manifest = _write_gen_manifest(tmp_path / "gen_manifest.csv", gen_rows)
    qa_rows = [
        {"job_id": "job_0", "passed": "True", "flag_reason": ""},
        {"job_id": "job_1", "passed": "False", "flag_reason": "generation_error"},
    ]

    pass_csv = write_qa_pass_csv(tmp_path / "qa_pass.csv", qa_rows)
    summary = write_qa_summary(gen_manifest, qa_rows, tmp_path / "qa_summary.md")

    assert pass_csv.is_file()
    text = summary.read_text()
    assert "vcm_sapinsapin" in text
    assert "fsc_1" in text
