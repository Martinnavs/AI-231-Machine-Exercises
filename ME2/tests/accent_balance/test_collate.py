from __future__ import annotations

import csv
from pathlib import Path

import pytest

from me2_voicegen.accent_balance.collate import (
    assert_base_rows_preserved,
    assert_fifty_fifty,
    assert_group_id_split_disjoint,
    collate_job_path,
    collate_vcm,
    collate_wakeword,
    mix_vcm_noise,
    mix_ww_noise,
    rebase_path,
    select_for_target_rows,
)
from me2_voicegen.vcm.vcmx_merge import MANIFEST_FIELDS as VCM_FIELDS
from me2_voicegen.wakeword.fetch_positives import ManifestValidationError
from me2_voicegen.wakeword.build_dataset import MANIFEST_FIELDS as WW_FIELDS

FILIPINO_ID = "s68"
NONFIL_ID = "s1"


def _write_csv(path: Path, fields: list[str], rows: list[dict]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
    return path


def _esc_row(filename, fold, source_file):
    return {"filename": filename, "category": "vacuum_cleaner", "esc_group": "interior_domestic", "fold": fold, "duration": "1.0", "rms": "0.2", "source_file": source_file}


def _write_esc50(tmp_path: Path) -> Path:
    esc_root = tmp_path / "background_noise"
    rows = []
    for i, fold in enumerate(["1", "2", "3", "4", "5"], start=1):
        # a real short noise wav via soundfile (no torch dependency needed here)
        import numpy as np
        import soundfile as sf

        (esc_root / "audio").mkdir(parents=True, exist_ok=True)
        sf.write(str(esc_root / "audio" / f"clip{i}.wav"), (0.05 * np.random.randn(16000)).astype("float32"), 16000)
        rows.append(_esc_row(f"clip{i}.wav", fold, f"{fold}-{1000 + i}-A-1.wav"))
    _write_csv(esc_root / "manifest.csv", list(rows[0].keys()), rows)
    return esc_root


# ---------------------------------------------------------------------------
# select_for_target_rows
# ---------------------------------------------------------------------------


def test_select_for_target_rows_exact_accounting_with_noisy_jobs():
    candidates = [
        {"job_id": f"j{i}", "noisy_target": "1" if i < 3 else "0"} for i in range(6)
    ]
    selection = select_for_target_rows(candidates, target_rows=5, seed=0)
    total = sum(2 if included else 1 for _job, included in selection)
    assert total == 5


def test_select_for_target_rows_insufficient_candidates_raises():
    candidates = [{"job_id": "j0", "noisy_target": "0"}]
    with pytest.raises(ManifestValidationError):
        select_for_target_rows(candidates, target_rows=3, seed=0)


def test_select_for_target_rows_zero_deficit_returns_empty():
    assert select_for_target_rows([{"job_id": "j0", "noisy_target": "0"}], target_rows=0, seed=0) == []


def test_select_for_target_rows_is_seed_deterministic():
    candidates = [{"job_id": f"j{i}", "noisy_target": "0"} for i in range(10)]
    a = select_for_target_rows(candidates, target_rows=4, seed=5)
    b = select_for_target_rows(candidates, target_rows=4, seed=5)
    assert [j["job_id"] for j, _ in a] == [j["job_id"] for j, _ in b]


# ---------------------------------------------------------------------------
# rebase_path / collate_job_path
# ---------------------------------------------------------------------------


def test_rebase_path_sibling_dirs(tmp_path):
    old_dir = tmp_path / "out" / "a"
    new_dir = tmp_path / "out" / "b"
    old_dir.mkdir(parents=True)
    new_dir.mkdir(parents=True)
    (tmp_path / "out" / "shared" / "clip.wav").parent.mkdir(parents=True, exist_ok=True)

    result = rebase_path("../shared/clip.wav", old_dir, new_dir)
    assert result == "../shared/clip.wav"


def test_collate_job_path_with_absolute_path(tmp_path):
    target = tmp_path / "audio" / "job_0.wav"
    new_dir = tmp_path / "manifest_dir"
    new_dir.mkdir()
    result = collate_job_path(str(target), new_dir)
    assert (new_dir / result).resolve() == target.resolve()


# ---------------------------------------------------------------------------
# assertions
# ---------------------------------------------------------------------------


def test_assert_fifty_fifty_passes_at_exact_parity():
    rows = (
        [{"bucket": "target_commands", "split": "train", "source_dataset": "optionb", "group_id": NONFIL_ID} for _ in range(5)]
        + [{"bucket": "target_commands", "split": "train", "source_dataset": "fil50_persona", "group_id": "fsc_1"} for _ in range(5)]
    )
    from me2_voicegen.accent_balance.collate import _is_filipino_vcm_row

    assert_fifty_fifty(rows, bucket_field="bucket", bucket_value="target_commands", split_field="split", is_filipino=_is_filipino_vcm_row)


def test_assert_fifty_fifty_raises_outside_tolerance():
    rows = (
        [{"bucket": "target_commands", "split": "train", "source_dataset": "optionb", "group_id": NONFIL_ID} for _ in range(9)]
        + [{"bucket": "target_commands", "split": "train", "source_dataset": "fil50_persona", "group_id": "fsc_1"} for _ in range(1)]
    )
    from me2_voicegen.accent_balance.collate import _is_filipino_vcm_row

    with pytest.raises(ManifestValidationError):
        assert_fifty_fifty(rows, bucket_field="bucket", bucket_value="target_commands", split_field="split", is_filipino=_is_filipino_vcm_row)


def test_assert_group_id_split_disjoint_raises_on_violation():
    rows = [{"group_id": "v1", "split": "train"}, {"group_id": "v1", "split": "test"}]
    with pytest.raises(ManifestValidationError):
        assert_group_id_split_disjoint(rows)


def test_assert_group_id_split_disjoint_passes_when_clean():
    rows = [{"group_id": "v1", "split": "train"}, {"group_id": "v2", "split": "test"}]
    assert_group_id_split_disjoint(rows) is None


def test_assert_group_id_split_disjoint_ref_prefix_exempt_across_splits():
    # Phase-1 scoped exception: ref_ group ids (references voices) may
    # legitimately appear in more than one split.
    rows = [
        {"group_id": "ref_tagalog1", "split": "train"},
        {"group_id": "ref_tagalog1", "split": "test"},
        {"group_id": "fsc_1", "split": "train"},
    ]
    assert assert_group_id_split_disjoint(rows) is None


@pytest.mark.parametrize("gid", ["fsc_1", "s68", "v1"])
def test_assert_group_id_split_disjoint_still_raises_for_non_ref_ids(gid):
    # Regression guard: the exemption is scoped to the ref_ prefix only.
    rows = [{"group_id": gid, "split": "train"}, {"group_id": gid, "split": "test"}]
    with pytest.raises(ManifestValidationError):
        assert_group_id_split_disjoint(rows)


def test_assert_base_rows_preserved_allows_only_ignored_fields_to_change():
    original = [{"filename": "a.wav", "path": "old/a.wav", "label": "ALARM"}]
    rebased = [{"filename": "a.wav", "path": "new/a.wav", "label": "ALARM"}]
    assert_base_rows_preserved(original, rebased, ignore_fields={"path"}) is None


def test_assert_base_rows_preserved_raises_on_non_path_change():
    original = [{"filename": "a.wav", "path": "old/a.wav", "label": "ALARM"}]
    rebased = [{"filename": "a.wav", "path": "new/a.wav", "label": "DIFFERENT"}]
    with pytest.raises(ManifestValidationError):
        assert_base_rows_preserved(original, rebased, ignore_fields={"path"})


# ---------------------------------------------------------------------------
# mix_vcm_noise / mix_ww_noise
# ---------------------------------------------------------------------------


def test_mix_vcm_noise_writes_a_new_noisy_row(tmp_path, vcm_wav_factory):
    from random import Random

    out_dir = tmp_path / "out"
    clean_path = vcm_wav_factory(out_dir / "clip.wav", duration_s=1.5)
    clean_row = {"filename": "clip.wav", "path": "clip.wav", "split": "train", "resampled": "True"}
    esc_root = _write_esc50(tmp_path)
    esc_rows = list(csv.DictReader((esc_root / "manifest.csv").open()))

    noisy_row = mix_vcm_noise(clean_row, esc_rows, esc_root, out_dir, Random(0))

    assert noisy_row["filename"] != clean_row["filename"]
    assert "_noisy" in noisy_row["filename"]
    assert (out_dir / noisy_row["path"]).is_file()
    assert float(noisy_row["duration"]) > 0


def test_mix_ww_noise_sets_noisy_source_dataset(tmp_path, vcm_wav_factory):
    from random import Random

    out_dir = tmp_path / "out"
    vcm_wav_factory(out_dir / "clip.wav", duration_s=3.0)
    clean_row = {"filename": "clip.wav", "path": "clip.wav", "split": "train", "source_dataset": "fil50_persona"}
    esc_root = _write_esc50(tmp_path)
    esc_rows = list(csv.DictReader((esc_root / "manifest.csv").open()))

    noisy_row = mix_ww_noise(clean_row, esc_rows, esc_root, out_dir, Random(0))

    assert noisy_row["source_dataset"] == "fil50_persona_noisy"
    assert noisy_row["noise_source_file"]
    assert (out_dir / noisy_row["path"]).is_file()


# ---------------------------------------------------------------------------
# end-to-end collate_vcm / collate_wakeword
# ---------------------------------------------------------------------------


@pytest.fixture
def vcm_base_fixture(tmp_path, vcm_wav_factory):
    base_dir = tmp_path / "base_vcm"
    audio_dir = base_dir / "audio"
    rows = []
    # train: 2 non-Filipino, 0 Filipino -> deficit 2
    for i in range(2):
        vcm_wav_factory(audio_dir / f"train_nf_{i}.wav", duration_s=1.5)
        rows.append({
            "filename": f"train_nf_{i}.wav", "path": f"audio/train_nf_{i}.wav", "bucket": "target_commands",
            "label": "ALARM", "duration": "1.500000", "sample_rate": "16000", "resampled": "False",
            "source_dataset": "optionb", "source_relpath": f"train_nf_{i}.wav", "group_id": NONFIL_ID,
            "split": "train", "transcript": "Alarm 6 AM", "original_dataset": "",
        })
    # test: 1 non-Filipino -> deficit 1
    vcm_wav_factory(audio_dir / "test_nf_0.wav", duration_s=1.5)
    rows.append({
        "filename": "test_nf_0.wav", "path": "audio/test_nf_0.wav", "bucket": "target_commands",
        "label": "STOP", "duration": "1.500000", "sample_rate": "16000", "resampled": "False",
        "source_dataset": "vcm_balanced", "source_relpath": "test_nf_0.wav", "group_id": "real_spk1",
        "split": "test", "transcript": "Stop", "original_dataset": "vcm_balanced",
    })
    manifest = _write_csv(base_dir / "manifest.csv", VCM_FIELDS, rows)
    return manifest


def _write_gen_and_qa(tmp_path, jobs_rows):
    from me2_voicegen.accent_balance.generate import GEN_FIELDS

    gen_manifest = _write_csv(tmp_path / "gen_manifest.csv", GEN_FIELDS, jobs_rows)
    qa_rows = [{"job_id": r["job_id"], "passed": "True", "flag_reason": ""} for r in jobs_rows]
    qa_pass = _write_csv(tmp_path / "qa_pass.csv", ["job_id", "passed", "flag_reason"], qa_rows)
    return gen_manifest, qa_pass


def test_collate_vcm_end_to_end_reaches_fifty_fifty(tmp_path, vcm_wav_factory, vcm_base_fixture):
    synth_dir = tmp_path / "synth"
    job_rows = []
    for i in range(2):
        p = vcm_wav_factory(synth_dir / f"vcm_train_{i}.wav", duration_s=1.5)
        job_rows.append({
            "job_id": f"vcm_train_{i:03d}", "model": "vcm", "split": "train", "voice_id": "fsc_1",
            "text": "Alarm 6 AM", "label": "ALARM", "source_row_ref": "train_nf_0.wav", "noisy_target": "0",
            "path": str(p), "duration": "1.500000", "seconds_elapsed": "0.1", "status": "ok",
            "speech_start_s": "", "speech_end_s": "",
        })
    p = vcm_wav_factory(synth_dir / "vcm_test_0.wav", duration_s=1.5)
    job_rows.append({
        "job_id": "vcm_test_000", "model": "vcm", "split": "test", "voice_id": "ref_tagalog1",
        "text": "Stop", "label": "STOP", "source_row_ref": "test_nf_0.wav", "noisy_target": "0",
        "path": str(p), "duration": "1.500000", "seconds_elapsed": "0.1", "status": "ok",
        "speech_start_s": "", "speech_end_s": "",
    })
    gen_manifest, qa_pass = _write_gen_and_qa(tmp_path, job_rows)

    from me2_voicegen.accent_balance.collate import load_passing_jobs

    passing_jobs = load_passing_jobs(gen_manifest, qa_pass)
    esc_root = _write_esc50(tmp_path)
    out_manifest = tmp_path / "out_vcm" / "manifest.csv"

    dest = collate_vcm(passing_jobs=passing_jobs, base_manifest_path=vcm_base_fixture, noise_root=esc_root, out_manifest_path=out_manifest, seed=0)

    rows = list(csv.DictReader(dest.open()))
    train_rows = [r for r in rows if r["split"] == "train" and r["bucket"] == "target_commands"]
    test_rows = [r for r in rows if r["split"] == "test" and r["bucket"] == "target_commands"]
    assert len(train_rows) == 4  # 2 base + 2 new
    assert len(test_rows) == 2  # 1 base + 1 new
    assert sum(1 for r in train_rows if r["source_dataset"] == "fil50_persona") == 2
    for r in rows:
        assert (dest.parent / r["path"]).is_file()


def test_collate_vcm_raises_when_not_enough_candidates(tmp_path, vcm_base_fixture):
    gen_manifest, qa_pass = _write_gen_and_qa(tmp_path, [])
    from me2_voicegen.accent_balance.collate import load_passing_jobs

    passing_jobs = load_passing_jobs(gen_manifest, qa_pass)
    esc_root = _write_esc50(tmp_path)
    with pytest.raises(ManifestValidationError):
        collate_vcm(passing_jobs=passing_jobs, base_manifest_path=vcm_base_fixture, noise_root=esc_root, out_manifest_path=tmp_path / "out_vcm2" / "manifest.csv", seed=0)


@pytest.fixture
def ww_base_fixture(tmp_path, vcm_wav_factory):
    base_dir = tmp_path / "base_ww"
    audio_dir = base_dir / "audio"
    rows = []
    for i in range(2):
        vcm_wav_factory(audio_dir / f"train_nf_{i}.wav", duration_s=3.0)
        rows.append({
            "filename": f"train_nf_{i}.wav", "path": f"audio/train_nf_{i}.wav", "label": "_wakeword_",
            "duration": "3.000000", "sample_rate": "16000", "resampled": "False", "source_dataset": "picovoice",
            "source_relpath": f"train_nf_{i}.wav", "group_id": f"train_nf_{i}", "split": "train", "ref_voice": "",
            "noise_source_file": "", "snr_db": "", "speech_start_s": "1.0", "speech_end_s": "2.0",
        })
    manifest = _write_csv(base_dir / "manifest.csv", WW_FIELDS, rows)
    return manifest


def test_collate_wakeword_end_to_end_reaches_fifty_fifty(tmp_path, vcm_wav_factory, ww_base_fixture):
    synth_dir = tmp_path / "synth_ww"
    job_rows = []
    for i in range(2):
        p = vcm_wav_factory(synth_dir / f"ww_train_{i}.wav", duration_s=3.0)
        job_rows.append({
            "job_id": f"ww_train_{i:03d}", "model": "wakeword", "split": "train", "voice_id": "fsc_1",
            "text": "Computer.", "label": "_wakeword_", "source_row_ref": "", "noisy_target": "0",
            "path": str(p), "duration": "3.000000", "seconds_elapsed": "0.1", "status": "ok",
            "speech_start_s": "1.0", "speech_end_s": "2.0",
        })
    gen_manifest, qa_pass = _write_gen_and_qa(tmp_path, job_rows)
    from me2_voicegen.accent_balance.collate import load_passing_jobs

    passing_jobs = load_passing_jobs(gen_manifest, qa_pass)
    esc_root = _write_esc50(tmp_path)
    out_manifest = tmp_path / "out_ww" / "manifest.csv"

    dest = collate_wakeword(passing_jobs=passing_jobs, base_manifest_path=ww_base_fixture, noise_root=esc_root, out_manifest_path=out_manifest, seed=0)

    rows = list(csv.DictReader(dest.open()))
    train_rows = [r for r in rows if r["split"] == "train" and r["label"] == "_wakeword_"]
    assert len(train_rows) == 4
    assert sum(1 for r in train_rows if r["source_dataset"] == "fil50_persona") == 2
    for r in rows:
        assert (dest.parent / r["path"]).is_file()
