import csv

import pytest

from me2_voicegen.wakeword import mix_background_noise as mbn
from me2_voicegen.wakeword.fetch_positives import ManifestValidationError, PathTraversalError


def _noise_manifest(tmp_path, vcm_wav_factory, n=5):
    root = tmp_path / "fake_background_noise"
    rows = []
    for i in range(n):
        filename = f"noise{i:02d}.wav"
        vcm_wav_factory(root / "audio" / filename, duration_s=0.3)
        rows.append(
            {
                "filename": filename,
                "category": "vacuum_cleaner" if i % 2 == 0 else "rain",
                "esc_group": "interior_domestic" if i % 2 == 0 else "weather_indoor_audible",
                "fold": "1",
                "duration": "0.3",
                "rms": "0.1",
                "source_file": f"src{i}.wav",
            }
        )
    manifest_path = root / "manifest.csv"
    with manifest_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return root


def _adversary_specs(n=6):
    return [
        {
            "source_dataset": "adversaries_tts",
            "label": "_unknown_",
            "filename": f"commuter__ref{i}.wav",
            "group_id": "commuter",
            "duration_s": 0.4,
        }
        for i in range(n)
    ]


def test_plan_jobs_determinism(tmp_path, vcm_wav_factory, wakeword_fake_manifest_factory):
    noise_root = _noise_manifest(tmp_path, vcm_wav_factory)
    noise_rows = mbn.load_noise_manifest(noise_root)
    manifest_path = wakeword_fake_manifest_factory(_adversary_specs(), subset_name="adversaries")
    source_rows = mbn.load_source_manifest(manifest_path.parent)

    jobs_a = mbn.plan_jobs(source_rows, noise_rows, noise_prob=0.5, snr_min_db=5.0, snr_max_db=25.0, seed=42)
    jobs_b = mbn.plan_jobs(source_rows, noise_rows, noise_prob=0.5, snr_min_db=5.0, snr_max_db=25.0, seed=42)

    assert [(j["source_row"]["filename"], j["noise_row"]["filename"], j["snr_db"]) for j in jobs_a] == [
        (j["source_row"]["filename"], j["noise_row"]["filename"], j["snr_db"]) for j in jobs_b
    ]


def test_plan_jobs_probability_boundaries(tmp_path, vcm_wav_factory, wakeword_fake_manifest_factory):
    noise_root = _noise_manifest(tmp_path, vcm_wav_factory)
    noise_rows = mbn.load_noise_manifest(noise_root)
    manifest_path = wakeword_fake_manifest_factory(_adversary_specs(20), subset_name="adversaries")
    source_rows = mbn.load_source_manifest(manifest_path.parent)

    none_selected = mbn.plan_jobs(source_rows, noise_rows, noise_prob=0.0, snr_min_db=5.0, snr_max_db=25.0, seed=1)
    all_selected = mbn.plan_jobs(source_rows, noise_rows, noise_prob=1.0, snr_min_db=5.0, snr_max_db=25.0, seed=1)

    assert none_selected == []
    assert len(all_selected) == len(source_rows)


def test_plan_jobs_snr_within_bounds(tmp_path, vcm_wav_factory, wakeword_fake_manifest_factory):
    noise_root = _noise_manifest(tmp_path, vcm_wav_factory)
    noise_rows = mbn.load_noise_manifest(noise_root)
    manifest_path = wakeword_fake_manifest_factory(_adversary_specs(30), subset_name="adversaries")
    source_rows = mbn.load_source_manifest(manifest_path.parent)

    jobs = mbn.plan_jobs(source_rows, noise_rows, noise_prob=1.0, snr_min_db=7.0, snr_max_db=13.0, seed=3)

    assert jobs
    assert all(7.0 <= j["snr_db"] <= 13.0 for j in jobs)


def test_dest_filename_unique_across_group_ids():
    a = {"group_id": "g1", "filename": "ref1.wav"}
    b = {"group_id": "g2", "filename": "ref1.wav"}
    noise = {"filename": "noise00.wav"}
    assert mbn.dest_filename(a, noise) != mbn.dest_filename(b, noise)


def test_mix_one_inherits_group_id_and_conforms_format(tmp_path, vcm_wav_factory, wakeword_fake_manifest_factory):
    noise_root = _noise_manifest(tmp_path, vcm_wav_factory)
    noise_rows = mbn.load_noise_manifest(noise_root)
    manifest_path = wakeword_fake_manifest_factory(_adversary_specs(1), subset_name="adversaries")
    subset_root = manifest_path.parent
    source_row = mbn.load_source_manifest(subset_root)[0]

    staging_root = tmp_path / "staging"
    staging_root.mkdir()
    row = mbn.mix_one(
        source_row=source_row,
        noise_row=noise_rows[0],
        snr_db=10.0,
        subset_root=subset_root,
        noise_root=noise_root,
        staging_root=staging_root,
    )

    assert row["group_id"] == source_row["group_id"]
    assert row["label"] == source_row["label"]
    assert row["source_dataset"] == f"{source_row['source_dataset']}_noisy"
    assert row["noise_source_file"] == noise_rows[0]["filename"]
    assert row["snr_db"] == "10.00"
    assert row["sample_rate"] == "16000"
    assert (staging_root / row["path"]).is_file()


def test_load_source_manifest_missing_raises(tmp_path):
    with pytest.raises(ManifestValidationError):
        mbn.load_source_manifest(tmp_path / "does_not_exist")


def test_load_noise_manifest_missing_raises(tmp_path):
    with pytest.raises(ManifestValidationError):
        mbn.load_noise_manifest(tmp_path / "does_not_exist")


def test_mix_one_rejects_path_traversal(tmp_path, vcm_wav_factory, wakeword_fake_manifest_factory):
    noise_root = _noise_manifest(tmp_path, vcm_wav_factory)
    noise_rows = mbn.load_noise_manifest(noise_root)
    manifest_path = wakeword_fake_manifest_factory(_adversary_specs(1), subset_name="adversaries")
    subset_root = manifest_path.parent
    source_row = dict(mbn.load_source_manifest(subset_root)[0])
    source_row["path"] = "../../../../etc/passwd"

    staging_root = tmp_path / "staging2"
    staging_root.mkdir()
    with pytest.raises(PathTraversalError):
        mbn.mix_one(
            source_row=source_row,
            noise_row=noise_rows[0],
            snr_db=10.0,
            subset_root=subset_root,
            noise_root=noise_root,
            staging_root=staging_root,
        )


def test_run_subset_dry_run_writes_nothing(tmp_path, vcm_wav_factory, wakeword_fake_manifest_factory):
    noise_root = _noise_manifest(tmp_path, vcm_wav_factory)
    noise_rows = mbn.load_noise_manifest(noise_root)
    manifest_path = wakeword_fake_manifest_factory(_adversary_specs(4), subset_name="adversaries")
    wakeword_root = manifest_path.parent.parent
    subset_dir_name = manifest_path.parent.name

    args = mbn.parse_args(["--seed", "1", "--noise-prob", "1.0", "--dry-run"])
    exit_code = mbn.run_subset(
        subset_dir_name,
        wakeword_root=wakeword_root,
        noise_root=noise_root,
        noise_rows=noise_rows,
        args=args,
    )

    assert exit_code == 0
    assert not (wakeword_root / f"{subset_dir_name}_noisy").exists()


def test_run_subset_end_to_end(tmp_path, vcm_wav_factory, wakeword_fake_manifest_factory):
    noise_root = _noise_manifest(tmp_path, vcm_wav_factory)
    noise_rows = mbn.load_noise_manifest(noise_root)
    manifest_path = wakeword_fake_manifest_factory(_adversary_specs(6), subset_name="adversaries")
    wakeword_root = manifest_path.parent.parent
    subset_dir_name = manifest_path.parent.name

    args = mbn.parse_args(["--seed", "7", "--noise-prob", "1.0"])
    exit_code = mbn.run_subset(
        subset_dir_name,
        wakeword_root=wakeword_root,
        noise_root=noise_root,
        noise_rows=noise_rows,
        args=args,
    )

    assert exit_code == 0
    out_manifest = wakeword_root / f"{subset_dir_name}_noisy" / "manifest.csv"
    assert out_manifest.is_file()
    with out_manifest.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 6
    for row in rows:
        assert (wakeword_root / f"{subset_dir_name}_noisy" / row["path"]).is_file()
        assert row["label"] == "_unknown_"
        assert row["source_dataset"] == "adversaries_tts_noisy"
