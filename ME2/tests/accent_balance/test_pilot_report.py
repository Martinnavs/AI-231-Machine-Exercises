from __future__ import annotations

import csv
from pathlib import Path

from me2_voicegen.accent_balance.pilot_report import (
    populate_listen_dir,
    project_full_run,
    throughput_by_model,
    write_report,
)

VCM_FIELDS = [
    "filename", "path", "bucket", "label", "duration", "sample_rate", "resampled",
    "source_dataset", "source_relpath", "group_id", "split", "transcript", "original_dataset",
]
WW_FIELDS = [
    "filename", "path", "label", "duration", "sample_rate", "resampled", "source_dataset",
    "source_relpath", "group_id", "split", "ref_voice", "noise_source_file", "snr_db",
    "speech_start_s", "speech_end_s",
]


def _write_csv(path: Path, fields: list[str], rows: list[dict]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
    return path


def test_throughput_by_model_averages_ok_rows_only():
    rows = [
        {"model": "vcm", "status": "ok", "seconds_elapsed": "4.0"},
        {"model": "vcm", "status": "ok", "seconds_elapsed": "6.0"},
        {"model": "vcm", "status": "error", "seconds_elapsed": ""},
        {"model": "wakeword", "status": "ok", "seconds_elapsed": "2.0"},
    ]
    result = throughput_by_model(rows)
    assert result["vcm"] == 5.0
    assert result["wakeword"] == 2.0


def test_project_full_run_scales_with_overgen_and_gpus(tmp_path):
    vcm_rows = (
        [{"filename": f"nf{i}.wav", "path": "", "bucket": "target_commands", "label": "ALARM", "duration": "1", "sample_rate": "16000", "resampled": "False", "source_dataset": "optionb", "source_relpath": "", "group_id": "s1", "split": "train", "transcript": "x", "original_dataset": ""} for i in range(10)]
    )
    ww_rows = (
        [{"filename": f"nf{i}.wav", "path": "", "label": "_wakeword_", "duration": "1", "sample_rate": "16000", "resampled": "False", "source_dataset": "picovoice", "source_relpath": "", "group_id": f"g{i}", "split": "train", "ref_voice": "", "noise_source_file": "", "snr_db": "", "speech_start_s": "", "speech_end_s": ""} for i in range(4)]
    )
    vcm_manifest = _write_csv(tmp_path / "vcm.csv", VCM_FIELDS, vcm_rows)
    ww_manifest = _write_csv(tmp_path / "ww.csv", WW_FIELDS, ww_rows)

    projection = project_full_run(
        vcm_manifest=vcm_manifest, wakeword_manifest=ww_manifest, overgen=1.0,
        throughput={"vcm": 5.0, "wakeword": 5.0}, n_gpus=2,
    )
    assert projection["vcm_jobs"] == 10
    assert projection["wakeword_jobs"] == 4
    assert projection["total_gpu_hours"] == (10 * 5.0 + 4 * 5.0) / 3600.0
    assert projection["wall_clock_hours"] == projection["total_gpu_hours"] / 2


def test_populate_listen_dir_copies_up_to_n_per_bucket(tmp_path):
    audio_dir = tmp_path / "audio"
    audio_dir.mkdir()
    gen_rows = []
    qa_rows = []
    for i in range(15):
        p = audio_dir / f"clip_{i}.wav"
        p.write_bytes(b"RIFF....WAVEfmt ")  # not a real wav, just needs to exist as a file
        gen_rows.append({"job_id": f"j{i}", "model": "vcm", "voice_id": "fsc_1", "status": "ok", "path": str(p)})
        qa_rows.append({"job_id": f"j{i}", "passed": "True"})
    listen_dir = tmp_path / "listen"

    populate_listen_dir(gen_rows, qa_rows, listen_dir, n_per_bucket=10, seed=0)

    passed_files = list((listen_dir / "sapinsapin" / "pass").glob("*.wav"))
    assert len(passed_files) == 10


def test_populate_listen_dir_separates_pass_and_flag_by_prompt_source(tmp_path):
    audio_dir = tmp_path / "audio"
    audio_dir.mkdir()
    gen_rows = []
    qa_rows = []
    for i, (voice_id, passed) in enumerate([("fsc_1", True), ("fsc_1", False), ("ref_tagalog1", True)]):
        p = audio_dir / f"clip_{i}.wav"
        p.write_bytes(b"data")
        gen_rows.append({"job_id": f"j{i}", "model": "vcm", "voice_id": voice_id, "status": "ok", "path": str(p)})
        qa_rows.append({"job_id": f"j{i}", "passed": "True" if passed else "False"})
    listen_dir = tmp_path / "listen"

    populate_listen_dir(gen_rows, qa_rows, listen_dir, n_per_bucket=10, seed=0)

    assert len(list((listen_dir / "sapinsapin" / "pass").glob("*.wav"))) == 1
    assert len(list((listen_dir / "sapinsapin" / "flag").glob("*.wav"))) == 1
    assert len(list((listen_dir / "references" / "pass").glob("*.wav"))) == 1
    assert not (listen_dir / "references" / "flag").exists()


def test_write_report_includes_projection_and_throughput(tmp_path):
    projection = {"vcm_jobs": 5, "wakeword_jobs": 2, "vcm_gpu_hours": 1.0, "wakeword_gpu_hours": 0.5, "total_gpu_hours": 1.5, "n_gpus": 2, "wall_clock_hours": 0.75}
    dest = write_report(
        out_path=tmp_path / "pilot_report.md", throughput={"vcm": 5.0}, projection=projection,
        qa_summary_path=tmp_path / "does_not_exist.md", listen_dir=tmp_path / "listen",
    )
    text = dest.read_text()
    assert "1.5 GPU-hours" in text
    assert "0.8 wall-clock hours" in text
    assert "5.00" in text
