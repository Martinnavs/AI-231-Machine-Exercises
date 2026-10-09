import csv

import pytest

from me2_voicegen.wakeword import consolidate_manifest as cm
from me2_voicegen.wakeword.fetch_positives import MANIFEST_FIELDS as BASE_FIELDS, ManifestValidationError


def _write_subset(wakeword_root, vcm_wav_factory, subset: str, rows: list[dict], extra_fields: list[str] | None = None):
    subset_root = wakeword_root / subset
    fieldnames = BASE_FIELDS + (extra_fields or [])
    full_rows = []
    for row in rows:
        vcm_wav_factory(subset_root / row["path"], duration_s=0.2)
        full_row = {field: "" for field in fieldnames}
        full_row.update(row)
        full_rows.append(full_row)

    manifest_path = subset_root / "manifest.csv"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with manifest_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(full_rows)


def _base_row(filename: str, label: str, source_dataset: str, group_id: str) -> dict:
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


def test_consolidate_merges_and_rewrites_paths(tmp_path, vcm_wav_factory):
    wakeword_root = tmp_path / "wakeword"
    _write_subset(
        wakeword_root,
        vcm_wav_factory,
        "positives_real",
        [_base_row("p1.wav", "_wakeword_", "picovoice", "p1")],
    )
    _write_subset(
        wakeword_root,
        vcm_wav_factory,
        "adversaries",
        [_base_row("a1.wav", "_unknown_", "adversaries_tts", "commuter")],
    )

    rows, counts = cm.consolidate(wakeword_root, list(cm.KNOWN_SUBSETS))

    assert counts["positives_real"] == 1
    assert counts["adversaries"] == 1
    assert counts["adversaries_noisy"] == 0
    paths = {row["path"] for row in rows}
    assert "positives_real/audio/p1.wav" in paths
    assert "adversaries/audio/a1.wav" in paths
    for row in rows:
        assert (wakeword_root / row["path"]).is_file()
        assert row["subset"] in ("positives_real", "adversaries")


def test_consolidate_tolerates_missing_subsets(tmp_path, vcm_wav_factory):
    wakeword_root = tmp_path / "wakeword"
    _write_subset(
        wakeword_root,
        vcm_wav_factory,
        "positives_real",
        [_base_row("p1.wav", "_wakeword_", "picovoice", "p1")],
    )

    rows, counts = cm.consolidate(wakeword_root, list(cm.KNOWN_SUBSETS))

    assert len(rows) == 1
    assert all(counts[s] == 0 for s in cm.KNOWN_SUBSETS if s != "positives_real")


def test_consolidate_preserves_extension_columns(tmp_path, vcm_wav_factory):
    wakeword_root = tmp_path / "wakeword"
    row = _base_row("c1.wav", "_wakeword_", "cosyvoice_conversion", "g1")
    row["ref_voice"] = "ilonggo1"
    _write_subset(
        wakeword_root, vcm_wav_factory, "positives_converted", [row], extra_fields=["ref_voice"]
    )

    noisy_row = _base_row("c1__noise-n1.wav", "_wakeword_", "cosyvoice_conversion_noisy", "g1")
    noisy_row["ref_voice"] = "ilonggo1"
    noisy_row["noise_source_file"] = "n1.wav"
    noisy_row["snr_db"] = "12.34"
    _write_subset(
        wakeword_root,
        vcm_wav_factory,
        "positives_converted_noisy",
        [noisy_row],
        extra_fields=["ref_voice", "noise_source_file", "snr_db"],
    )

    rows, _counts = cm.consolidate(wakeword_root, list(cm.KNOWN_SUBSETS))
    by_subset = {row["subset"]: row for row in rows}

    assert by_subset["positives_converted"]["ref_voice"] == "ilonggo1"
    assert by_subset["positives_converted"]["noise_source_file"] == ""
    assert by_subset["positives_converted_noisy"]["noise_source_file"] == "n1.wav"
    assert by_subset["positives_converted_noisy"]["snr_db"] == "12.34"


def test_consolidate_fails_loudly_on_duplicate_path(tmp_path, vcm_wav_factory):
    wakeword_root = tmp_path / "wakeword"
    row = _base_row("p1.wav", "_wakeword_", "picovoice", "p1")
    # Two manifest rows pointing at the same on-disk path within one
    # subset -- a real bug in whatever produced this manifest, and exactly
    # what `consolidate()`'s duplicate-path check must catch.
    _write_subset(wakeword_root, vcm_wav_factory, "positives_real", [row])
    with (wakeword_root / "positives_real" / "manifest.csv").open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=BASE_FIELDS)
        writer.writerow(row)

    with pytest.raises(ManifestValidationError):
        cm.consolidate(wakeword_root, list(cm.KNOWN_SUBSETS))


def test_consolidate_fails_loudly_on_missing_file(tmp_path, vcm_wav_factory):
    wakeword_root = tmp_path / "wakeword"
    _write_subset(
        wakeword_root,
        vcm_wav_factory,
        "positives_real",
        [_base_row("p1.wav", "_wakeword_", "picovoice", "p1")],
    )
    (wakeword_root / "positives_real" / "audio" / "p1.wav").unlink()

    with pytest.raises(ManifestValidationError):
        cm.consolidate(wakeword_root, list(cm.KNOWN_SUBSETS))


def test_write_manifest_and_summary_roundtrip(tmp_path, vcm_wav_factory):
    wakeword_root = tmp_path / "wakeword"
    _write_subset(
        wakeword_root,
        vcm_wav_factory,
        "positives_real",
        [_base_row("p1.wav", "_wakeword_", "picovoice", "p1")],
    )
    rows, counts = cm.consolidate(wakeword_root, list(cm.KNOWN_SUBSETS))

    manifest_path = cm.write_manifest(wakeword_root / "generated_manifest.csv", rows)
    cm.write_summary(wakeword_root / "generated_summary.md", subsets=list(cm.KNOWN_SUBSETS), counts=counts, rows=rows)

    with manifest_path.open(newline="", encoding="utf-8") as f:
        read_rows = list(csv.DictReader(f))
    assert read_rows == rows
    summary_text = (wakeword_root / "generated_summary.md").read_text(encoding="utf-8")
    assert "positives_real" in summary_text
    assert "No `*_noisy` subset present" in summary_text
