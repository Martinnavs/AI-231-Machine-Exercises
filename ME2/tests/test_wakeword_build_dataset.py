import csv

import pytest

from me2_voicegen.wakeword import build_dataset as bd
from me2_voicegen.wakeword.fetch_positives import MANIFEST_FIELDS as BASE_FIELDS, ManifestValidationError


def _write_subset(wakeword_root, vcm_wav_factory, subset: str, rows: list[dict]):
    subset_root = wakeword_root / subset
    fieldnames = BASE_FIELDS
    for row in rows:
        vcm_wav_factory(subset_root / row["path"], duration_s=0.2)
    manifest_path = subset_root / "manifest.csv"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with manifest_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows({field: row.get(field, "") for field in fieldnames} for row in rows)


def _row(filename, label, source_dataset, group_id):
    return {
        "filename": filename,
        "path": f"audio/{filename}",
        "label": label,
        "duration": "0.200000",
        "sample_rate": "16000",
        "resampled": "False",
        "source_dataset": source_dataset,
        "source_relpath": f"src/{filename}",
        "group_id": group_id,
        "split": "",
    }


def _build_minimal_dataset(wakeword_root, vcm_wav_factory, *, n_groups=20, rows_per_group=5):
    wakeword_rows = [
        _row(f"w{g}_{i}.wav", "_wakeword_", "picovoice", f"wgroup{g}")
        for g in range(n_groups)
        for i in range(rows_per_group)
    ]
    unknown_rows = [
        _row(f"u{g}_{i}.wav", "_unknown_", "common_voice_negative", f"ugroup{g}")
        for g in range(n_groups)
        for i in range(rows_per_group)
    ]
    silence_rows = [
        _row(f"s{g}_{i}.wav", "_silence_", "synthetic_noise", f"sgroup{g}")
        for g in range(n_groups)
        for i in range(rows_per_group)
    ]
    _write_subset(wakeword_root, vcm_wav_factory, "positives_real", wakeword_rows)
    _write_subset(wakeword_root, vcm_wav_factory, "common_voice_negative_sample", unknown_rows)
    _write_subset(wakeword_root, vcm_wav_factory, "silence_synthetic", silence_rows)
    return wakeword_rows, unknown_rows, silence_rows


def test_load_label_rows_rejects_label_mismatch(tmp_path, vcm_wav_factory):
    wakeword_root = tmp_path / "wakeword"
    bad_row = _row("p1.wav", "_unknown_", "picovoice", "p1")  # wrong label for positives_real
    _write_subset(wakeword_root, vcm_wav_factory, "positives_real", [bad_row])

    with pytest.raises(ManifestValidationError):
        bd.load_label_rows(wakeword_root, "_wakeword_")


def test_assign_group_disjoint_splits_approximates_target_ratio(tmp_path, vcm_wav_factory):
    wakeword_root = tmp_path / "wakeword"
    wakeword_rows, unknown_rows, silence_rows = _build_minimal_dataset(wakeword_root, vcm_wav_factory)

    rows_by_label = {
        "_wakeword_": bd.load_label_rows(wakeword_root, "_wakeword_"),
        "_unknown_": bd.load_label_rows(wakeword_root, "_unknown_"),
        "_silence_": bd.load_label_rows(wakeword_root, "_silence_"),
    }
    stats = bd.assign_group_disjoint_splits(rows_by_label, seed=0)

    for label in bd.LABELS:
        s = stats[label]
        assert s["train"] + s["val"] + s["test"] == s["total"]
        assert abs(s["train"] / s["total"] - 0.70) < 0.05
        assert abs(s["val"] / s["total"] - 0.20) < 0.05
        assert abs(s["test"] / s["total"] - 0.10) < 0.05

    all_rows = [r for rows in rows_by_label.values() for r in rows]
    bd.assert_group_disjointness(all_rows)  # must not raise


def test_assign_group_disjoint_splits_is_deterministic(tmp_path, vcm_wav_factory):
    wakeword_root = tmp_path / "wakeword"
    _build_minimal_dataset(wakeword_root, vcm_wav_factory)

    def run():
        rows_by_label = {label: bd.load_label_rows(wakeword_root, label) for label in bd.LABELS}
        bd.assign_group_disjoint_splits(rows_by_label, seed=7)
        return {row["group_id"]: row["split"] for rows in rows_by_label.values() for row in rows}

    assert run() == run()


def test_assign_group_disjoint_splits_reuses_cross_label_collision(tmp_path, vcm_wav_factory):
    wakeword_root = tmp_path / "wakeword"
    # Same group_id ("shared") appears under both _wakeword_ and _unknown_ --
    # a coincidental collision the module must handle by reuse, not conflict.
    wakeword_rows = [_row(f"w{i}.wav", "_wakeword_", "picovoice", "shared") for i in range(5)]
    unknown_rows = [_row(f"u{i}.wav", "_unknown_", "common_voice_negative", "shared") for i in range(5)]
    silence_rows = [_row(f"s{i}.wav", "_silence_", "synthetic_noise", "sgroup") for i in range(5)]
    _write_subset(wakeword_root, vcm_wav_factory, "positives_real", wakeword_rows)
    _write_subset(wakeword_root, vcm_wav_factory, "common_voice_negative_sample", unknown_rows)
    _write_subset(wakeword_root, vcm_wav_factory, "silence_synthetic", silence_rows)

    rows_by_label = {label: bd.load_label_rows(wakeword_root, label) for label in bd.LABELS}
    bd.assign_group_disjoint_splits(rows_by_label, seed=0)
    all_rows = [r for rows in rows_by_label.values() for r in rows]

    splits_for_shared = {row["split"] for row in all_rows if row["group_id"] == "shared"}
    assert len(splits_for_shared) == 1
    bd.assert_group_disjointness(all_rows)  # must not raise


def test_assert_group_disjointness_raises_on_violation():
    rows = [
        {"group_id": "g1", "split": "train"},
        {"group_id": "g1", "split": "test"},
    ]
    with pytest.raises(ManifestValidationError):
        bd.assert_group_disjointness(rows)


def test_verify_paths_exist_raises_on_missing(tmp_path):
    with pytest.raises(ManifestValidationError):
        bd.verify_paths_exist(tmp_path, [{"path": "does/not/exist.wav"}])


def test_main_end_to_end(tmp_path, vcm_wav_factory):
    wakeword_root = tmp_path / "wakeword"
    _build_minimal_dataset(wakeword_root, vcm_wav_factory)
    # build_dataset.py also needs adversaries/adversaries_noisy/positives_converted/
    # positives_converted_noisy present (even if empty) per SUBSETS_BY_LABEL --
    # missing subset dirs are tolerated (load_subset_rows returns []), only an
    # entirely-empty LABEL is fatal.

    exit_code = bd.main(["--wakeword-root", str(wakeword_root), "--seed", "3"])

    assert exit_code == 0
    manifest_path = wakeword_root / "manifest.csv"
    assert manifest_path.is_file()
    with manifest_path.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 20 * 5 * 3
    assert all(row["split"] in ("train", "val", "test") for row in rows)
    assert (wakeword_root / "summary.md").is_file()


def test_main_fails_loudly_when_a_label_is_empty(tmp_path, vcm_wav_factory):
    wakeword_root = tmp_path / "wakeword"
    wakeword_rows = [_row(f"w{i}.wav", "_wakeword_", "picovoice", f"g{i}") for i in range(5)]
    _write_subset(wakeword_root, vcm_wav_factory, "positives_real", wakeword_rows)
    # _unknown_/_silence_ subsets never created.

    exit_code = bd.main(["--wakeword-root", str(wakeword_root), "--seed", "0"])

    assert exit_code == 1
    assert not (wakeword_root / "manifest.csv").exists()
