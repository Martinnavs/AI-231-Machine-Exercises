"""Tests for me2_voicegen.vcm.optionb.fetch_dataset.

No test here performs a real network clone -- `clone_optionb`'s subprocess
calls are either exercised against a fake pre-populated "clone" directory
(the reuse/no-re-download path) or bypassed entirely via monkeypatching, so
the suite runs offline and fast. The conversion logic (manifest mapping,
path-traversal rejection, copy+measure, idempotency) is exercised against
synthetic fixtures shaped like the real upstream manifest.
"""

from __future__ import annotations

import csv
import struct
import subprocess
import wave
from pathlib import Path

import pytest

from me2_voicegen.vcm.optionb import fetch_dataset as fd

UPSTREAM_FIELDS = [
    "path",
    "label",
    "intent",
    "speaker",
    "split",
    "phrase_id",
    "variant_id",
    "transcript",
    "slot",
    "slot_value",
    "duration_sec",
]


def write_wav(path: Path, *, sample_rate=16000, channels=1, sampwidth=2, n_frames=1600) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as w:
        w.setnchannels(channels)
        w.setsampwidth(sampwidth)
        w.setframerate(sample_rate)
        w.writeframes(struct.pack("<" + "h" * (n_frames * channels), *([0] * (n_frames * channels))))


def write_upstream_manifest(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=UPSTREAM_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def upstream_row(**overrides) -> dict:
    row = {
        "path": "ALARM_6_00AM/ALARM_6_00AM_s100_v1_clean.wav",
        "label": "ALARM_6_00AM",
        "intent": "ALARM",
        "speaker": "s100",
        "split": "train",
        "phrase_id": "v1",
        "variant_id": "clean",
        "transcript": "Alarm 6 AM",
        "slot": "time",
        "slot_value": "6 AM",
        "duration_sec": "1.5",
    }
    row.update(overrides)
    return row


@pytest.fixture
def optionb_dir(tmp_path):
    d = tmp_path / "clone" / "MEX2" / "OptionB"
    write_wav(d / "ALARM_6_00AM" / "ALARM_6_00AM_s100_v1_clean.wav")
    write_wav(d / "CALL" / "CALL_s101_v1_clean.wav")
    write_wav(d / "FLAGGED" / "junk.wav")  # must never be counted/copied
    write_upstream_manifest(
        d / "manifest.csv",
        [
            upstream_row(),
            upstream_row(
                path="CALL/CALL_s101_v1_clean.wav",
                label="CALL",
                intent="CALL",
                speaker="s101",
                split="val",
                transcript="Call mom",
                slot="",
                slot_value="",
            ),
        ],
    )
    return d


# ---------------------------------------------------------------------------
# Path-traversal / sanitization
# ---------------------------------------------------------------------------


def test_resolve_under_accepts_path_inside_root(tmp_path):
    root = tmp_path / "root"
    root.mkdir()
    (root / "sub").mkdir()
    assert fd.resolve_under(root, root / "sub" / "f.wav") == (root / "sub" / "f.wav").resolve()


def test_resolve_under_rejects_traversal(tmp_path):
    root = tmp_path / "root"
    root.mkdir()
    with pytest.raises(fd.PathTraversalError):
        fd.resolve_under(root, root / ".." / "escaped.wav")


@pytest.mark.parametrize("bad", ["../evil", "a/b", "..", ".", "", "bad;rm"])
def test_sanitize_component_rejects_unsafe(bad):
    with pytest.raises(fd.PathTraversalError):
        fd.sanitize_component(bad, "test")


def test_sanitize_component_accepts_safe():
    assert fd.sanitize_component("ALARM_6_00AM_v1.wav", "test") == "ALARM_6_00AM_v1.wav"


# ---------------------------------------------------------------------------
# Upstream manifest validation
# ---------------------------------------------------------------------------


def test_load_upstream_manifest_rejects_missing_column(tmp_path):
    path = tmp_path / "manifest.csv"
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=[c for c in UPSTREAM_FIELDS if c != "intent"])
        writer.writeheader()
        writer.writerow({k: "" for k in writer.fieldnames})
    with pytest.raises(fd.ManifestValidationError, match="intent"):
        fd.load_upstream_manifest(path)


def test_load_upstream_manifest_rejects_row_count_out_of_bounds(tmp_path):
    path = tmp_path / "manifest.csv"
    write_upstream_manifest(path, [upstream_row()])
    with pytest.raises(fd.ManifestValidationError, match="expected sanity range"):
        fd.load_upstream_manifest(path)


def test_load_upstream_manifest_accepts_valid(tmp_path):
    path = tmp_path / "manifest.csv"
    rows = [upstream_row(path=f"CALL/x{i}.wav") for i in range(fd.MIN_EXPECTED_ROWS)]
    write_upstream_manifest(path, rows)
    loaded = fd.load_upstream_manifest(path)
    assert len(loaded) == fd.MIN_EXPECTED_ROWS


# ---------------------------------------------------------------------------
# Target-row mapping (D4/D5/D12)
# ---------------------------------------------------------------------------


def test_build_target_manifest_rows_maps_columns(optionb_dir, tmp_path):
    out_root = tmp_path / "out"
    upstream_rows = read_rows_directly(optionb_dir / "manifest.csv")
    manifest_rows, copy_jobs = fd.build_target_manifest_rows(upstream_rows, optionb_dir, out_root)

    assert len(manifest_rows) == len(copy_jobs) == 2
    alarm = next(r for r in manifest_rows if r["filename"] == "ALARM_6_00AM_s100_v1_clean.wav")
    assert alarm["label"] == "ALARM"  # intent, not upstream's own 'label' column (D4)
    assert alarm["bucket"] == "target_commands"
    assert alarm["source_dataset"] == "optionb"
    assert alarm["source_relpath"] == "ALARM_6_00AM/ALARM_6_00AM_s100_v1_clean.wav"
    assert alarm["group_id"] == "s100"
    assert alarm["split"] == "train"
    assert alarm["transcript"] == "Alarm 6 AM"
    assert alarm["resampled"] == "False"
    assert alarm["path"] == "audio/ALARM/ALARM_6_00AM_s100_v1_clean.wav"


def test_build_target_manifest_rows_rejects_path_traversal(optionb_dir, tmp_path):
    out_root = tmp_path / "out"
    upstream_rows = [upstream_row(path="../../../etc/passwd")]
    with pytest.raises(fd.PathTraversalError):
        fd.build_target_manifest_rows(upstream_rows, optionb_dir, out_root)


def test_build_target_manifest_rows_rejects_unsafe_intent(optionb_dir, tmp_path):
    out_root = tmp_path / "out"
    upstream_rows = [upstream_row(intent="../escape")]
    with pytest.raises(fd.PathTraversalError):
        fd.build_target_manifest_rows(upstream_rows, optionb_dir, out_root)


def test_build_target_manifest_rows_rejects_duplicate_destination(optionb_dir, tmp_path):
    out_root = tmp_path / "out"
    upstream_rows = [upstream_row(), upstream_row(speaker="s999")]
    with pytest.raises(fd.ManifestValidationError, match="duplicate destination"):
        fd.build_target_manifest_rows(upstream_rows, optionb_dir, out_root)


@pytest.mark.parametrize(
    "bad_split",
    ["+HYPERLINK(\"http://evil\")", "TRAIN", "production", "", "../train", "train,val"],
)
def test_build_target_manifest_rows_rejects_invalid_split(optionb_dir, tmp_path, bad_split):
    out_root = tmp_path / "out"
    upstream_rows = [upstream_row(split=bad_split)]
    with pytest.raises(fd.ManifestValidationError, match="split"):
        fd.build_target_manifest_rows(upstream_rows, optionb_dir, out_root)


@pytest.mark.parametrize("good_split", sorted(fd.VALID_SPLITS))
def test_build_target_manifest_rows_accepts_valid_splits(optionb_dir, tmp_path, good_split):
    out_root = tmp_path / "out"
    upstream_rows = [upstream_row(split=good_split)]
    manifest_rows, _ = fd.build_target_manifest_rows(upstream_rows, optionb_dir, out_root)
    assert manifest_rows[0]["split"] == good_split


# ---------------------------------------------------------------------------
# Probe rows from test_set (D6c)
# ---------------------------------------------------------------------------


def read_rows_directly(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def make_test_set_manifest(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "filename",
        "path",
        "bucket",
        "label",
        "duration",
        "sample_rate",
        "resampled",
        "source_dataset",
        "source_relpath",
        "group_id",
        "split",
    ]
    rows = [
        {
            "filename": "b1.wav",
            "path": "audio/babble/b1.wav",
            "bucket": "babble",
            "label": "unknown",
            "duration": "1.0",
            "sample_rate": "16000",
            "resampled": "False",
            "source_dataset": "filipino_speech_corpus",
            "source_relpath": "audio/b1.wav",
            "group_id": "38",
            "split": "train",
        },
        {
            "filename": "s1.wav",
            "path": "audio/silence/s1.wav",
            "bucket": "silence",
            "label": "silence",
            "duration": "2.0",
            "sample_rate": "16000",
            "resampled": "False",
            "source_dataset": "background_noise",
            "source_relpath": "audio/s1.wav",
            "group_id": "esc1",
            "split": "test",
        },
        {
            "filename": "t1.wav",
            "path": "audio/target_commands/ALARM/t1.wav",
            "bucket": "target_commands",
            "label": "ALARM",
            "duration": "1.0",
            "sample_rate": "16000",
            "resampled": "False",
            "source_dataset": "sanitized_clean",
            "source_relpath": "clean/ALARM/t1.wav",
            "group_id": "s1",
            "split": "train",
        },
    ]
    for row in rows:
        write_wav(path.parent / row["path"])

    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def test_build_probe_manifest_rows_filters_and_rewrites_path(tmp_path):
    test_set_manifest = tmp_path / "test_set" / "manifest.csv"
    make_test_set_manifest(test_set_manifest)

    probe_rows = fd.build_probe_manifest_rows(test_set_manifest)

    assert {r["bucket"] for r in probe_rows} == {"babble", "silence"}
    assert len(probe_rows) == 2
    babble = next(r for r in probe_rows if r["bucket"] == "babble")
    assert babble["path"] == "../test_set/audio/babble/b1.wav"
    assert babble["transcript"] == ""
    assert babble["source_dataset"] == "filipino_speech_corpus"
    assert babble["group_id"] == "38"


# ---------------------------------------------------------------------------
# copy_and_measure
# ---------------------------------------------------------------------------


def test_copy_and_measure_fills_duration_and_sample_rate(optionb_dir, tmp_path):
    out_root = tmp_path / "out"
    upstream_rows = read_rows_directly(optionb_dir / "manifest.csv")
    manifest_rows, copy_jobs = fd.build_target_manifest_rows(upstream_rows, optionb_dir, out_root)

    fd.copy_and_measure(copy_jobs, manifest_rows)

    for row, (_src, dest) in zip(manifest_rows, copy_jobs):
        assert dest.is_file()
        assert row["sample_rate"] == "16000"
        assert float(row["duration"]) == pytest.approx(0.1, abs=1e-3)


def test_copy_and_measure_resamples_non_16k_file(tmp_path):
    optionb = tmp_path / "clone" / "MEX2" / "OptionB"
    write_wav(optionb / "VOLUME_DOWN" / "v2.wav", sample_rate=12000, n_frames=12000)
    src = optionb / "VOLUME_DOWN" / "v2.wav"
    dest = tmp_path / "out" / "audio" / "VOLUME_DOWN" / "v2.wav"
    row = {"duration": None, "sample_rate": None, "resampled": "False"}

    fd.copy_and_measure([(src, dest)], [row])

    assert row["sample_rate"] == "16000"
    assert row["resampled"] == "True"
    assert float(row["duration"]) == pytest.approx(1.0, abs=1e-2)

    with wave.open(str(dest), "rb") as w:
        assert w.getframerate() == 16000
        assert w.getnchannels() == 1
        assert w.getsampwidth() == 2


def test_copy_and_measure_leaves_native_16k_file_unresampled(tmp_path):
    optionb = tmp_path / "clone" / "MEX2" / "OptionB"
    write_wav(optionb / "ALARM" / "a.wav", sample_rate=16000)
    src = optionb / "ALARM" / "a.wav"
    dest = tmp_path / "out" / "audio" / "ALARM" / "a.wav"
    row = {"duration": None, "sample_rate": None, "resampled": "False"}

    fd.copy_and_measure([(src, dest)], [row])

    assert row["sample_rate"] == "16000"
    assert row["resampled"] == "False"


def test_converted_manifest_never_records_a_non_16k_sample_rate(optionb_dir, tmp_path):
    upstream_rows = read_rows_directly(optionb_dir / "manifest.csv")
    volume_down_path = optionb_dir / "VOLUME_DOWN_5" / "VOLUME_DOWN_5_s100_v2_clean.wav"
    write_wav(volume_down_path, sample_rate=12000, n_frames=12000)
    upstream_rows.append(
        upstream_row(
            path="VOLUME_DOWN_5/VOLUME_DOWN_5_s100_v2_clean.wav",
            label="VOLUME_DOWN_5",
            intent="VOLUME_DOWN",
            speaker="s100",
            transcript="volume down",
        )
    )
    out_root = tmp_path / "out"
    manifest_rows, copy_jobs = fd.build_target_manifest_rows(upstream_rows, optionb_dir, out_root)

    fd.copy_and_measure(copy_jobs, manifest_rows)

    bad = [r for r in manifest_rows if r["sample_rate"] != "16000"]
    assert bad == []
    volume_down_row = next(r for r in manifest_rows if r["source_relpath"].startswith("VOLUME_DOWN_5/"))
    assert volume_down_row["resampled"] == "True"


def test_copy_and_measure_rejects_wrong_format(tmp_path):
    optionb = tmp_path / "clone" / "MEX2" / "OptionB"
    write_wav(optionb / "CALL" / "bad.wav", channels=2)  # stereo, not mono
    src = optionb / "CALL" / "bad.wav"
    dest = tmp_path / "out" / "audio" / "CALL" / "bad.wav"
    row = {"duration": None, "sample_rate": None}

    with pytest.raises(RuntimeError, match="required"):
        fd.copy_and_measure([(src, dest)], [row])


def test_copy_and_measure_enforces_per_file_cap(monkeypatch, tmp_path):
    optionb = tmp_path / "clone" / "MEX2" / "OptionB"
    write_wav(optionb / "CALL" / "big.wav")
    src = optionb / "CALL" / "big.wav"
    dest = tmp_path / "out" / "audio" / "CALL" / "big.wav"
    monkeypatch.setattr(fd, "MAX_SINGLE_FILE_BYTES", 1)

    with pytest.raises(fd.ManifestValidationError, match="per-file cap"):
        fd.copy_and_measure([(src, dest)], [{"duration": None, "sample_rate": None}])


# ---------------------------------------------------------------------------
# verify_manifest_files_exist
# ---------------------------------------------------------------------------


def test_verify_manifest_files_exist_detects_missing(tmp_path):
    out_root = tmp_path / "out"
    out_root.mkdir()
    rows = [{"filename": "a.wav", "path": "audio/A/a.wav", "bucket": "target_commands"}]
    with pytest.raises(fd.ManifestValidationError, match="does not exist"):
        fd.verify_manifest_files_exist(rows, out_root)


def test_verify_manifest_files_exist_detects_duplicate_filename(tmp_path):
    out_root = tmp_path / "out"
    write_wav(out_root / "audio" / "A" / "a.wav")
    rows = [
        {"filename": "a.wav", "path": "audio/A/a.wav", "bucket": "target_commands"},
        {"filename": "a.wav", "path": "audio/A/a.wav", "bucket": "target_commands"},
    ]
    with pytest.raises(fd.ManifestValidationError, match="more than one"):
        fd.verify_manifest_files_exist(rows, out_root)


# ---------------------------------------------------------------------------
# count_active_wavs
# ---------------------------------------------------------------------------


def test_count_active_wavs_excludes_flagged(optionb_dir):
    assert fd.count_active_wavs(optionb_dir) == 2


# ---------------------------------------------------------------------------
# clone_optionb: reuse path (no re-download / no re-clone)
# ---------------------------------------------------------------------------


def test_clone_optionb_reuses_existing_valid_clone(tmp_path, monkeypatch):
    clone_dir = tmp_path / "clone"
    (clone_dir / ".git").mkdir(parents=True)
    optionb_dir = clone_dir / "MEX2" / "OptionB"
    write_upstream_manifest(optionb_dir / "manifest.csv", [upstream_row()])

    def fail_if_clone_invoked(cmd, **kwargs):
        if "clone" in cmd:
            raise AssertionError("clone_optionb re-cloned an already-valid clone dir")
        raise AssertionError(f"unexpected subprocess call: {cmd}")

    monkeypatch.setattr(subprocess, "run", fail_if_clone_invoked)
    monkeypatch.setattr(fd, "git_commit_sha", lambda repo_dir: "deadbeef")

    resolved_dir, sha = fd.clone_optionb(clone_dir)
    assert resolved_dir == optionb_dir
    assert sha == "deadbeef"


def test_clone_optionb_disables_git_lfs_smudge(tmp_path, monkeypatch):
    clone_dir = tmp_path / "clone"
    calls = []

    def fake_run(cmd, **kwargs):
        calls.append(cmd)
        if cmd[:2] == ["git", "-c"] and "clone" in cmd:
            (clone_dir / ".git" / "info").mkdir(parents=True)
            (clone_dir / "MEX2" / "OptionB").mkdir(parents=True)
            write_upstream_manifest(clone_dir / "MEX2" / "OptionB" / "manifest.csv", [upstream_row()])
        return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")

    monkeypatch.setattr(subprocess, "run", fake_run)
    monkeypatch.setattr(fd, "git_commit_sha", lambda repo_dir: "deadbeef")

    fd.clone_optionb(clone_dir)

    clone_cmd = next(c for c in calls if "clone" in c)
    assert "filter.lfs.smudge=" in clone_cmd
    assert "filter.lfs.process=" in clone_cmd
    assert "filter.lfs.required=false" in clone_cmd
    assert "core.hooksPath=/dev/null" in clone_cmd


# ---------------------------------------------------------------------------
# End-to-end: dry-run, real run, idempotency, test_set untouched
# ---------------------------------------------------------------------------


@pytest.fixture
def big_optionb_dir(tmp_path):
    """A manifest sized to clear MIN_EXPECTED_ROWS, with real backing wavs
    for every row plus one FLAGGED file that must never be touched."""
    d = tmp_path / "clone" / "MEX2" / "OptionB"
    rows = []
    intents = ["ALARM", "CALL", "STOP"]
    for i in range(fd.MIN_EXPECTED_ROWS):
        intent = intents[i % len(intents)]
        speaker = f"s{i % 5}"
        relpath = f"{intent}/{intent}_{i}.wav"
        write_wav(d / relpath)
        rows.append(
            upstream_row(
                path=relpath,
                label=f"{intent}_LBL",
                intent=intent,
                speaker=speaker,
                split=["train", "val", "test"][i % 3],
                transcript=f"transcript {i}",
            )
        )
    write_wav(d / "FLAGGED" / "junk.wav")
    write_upstream_manifest(d / "manifest.csv", rows)
    return d


def test_dry_run_reports_without_writing(big_optionb_dir, tmp_path, monkeypatch):
    out_root = tmp_path / "out" / "optionb"
    test_set_manifest = tmp_path / "out" / "test_set" / "manifest.csv"
    make_test_set_manifest(test_set_manifest)

    monkeypatch.setattr(fd, "clone_optionb", lambda clone_dir, repo_url, force=False: (big_optionb_dir, "abc123"))

    rc = fd.main(
        [
            "--out-root",
            str(out_root),
            "--test-set-manifest",
            str(test_set_manifest),
            "--dry-run",
        ]
    )
    assert rc == 0
    assert not out_root.exists()


def test_full_run_writes_manifest_and_is_idempotent(big_optionb_dir, tmp_path, monkeypatch):
    out_root = tmp_path / "out" / "optionb"
    test_set_dir = tmp_path / "out" / "test_set"
    test_set_manifest = test_set_dir / "manifest.csv"
    make_test_set_manifest(test_set_manifest)
    before_checksum = test_set_manifest.read_bytes()

    monkeypatch.setattr(fd, "clone_optionb", lambda clone_dir, repo_url, force=False: (big_optionb_dir, "abc123"))

    argv = ["--out-root", str(out_root), "--test-set-manifest", str(test_set_manifest)]
    rc = fd.main(argv)
    assert rc == 0

    with (out_root / "manifest.csv").open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == fd.MIN_EXPECTED_ROWS + 2  # + babble/silence probe rows
    target_rows = [r for r in rows if r["bucket"] == "target_commands" and r["source_dataset"] == "optionb"]
    assert len(target_rows) == fd.MIN_EXPECTED_ROWS
    assert all((out_root / r["path"]).is_file() for r in rows)
    assert (out_root / "summary.md").is_file()
    assert test_set_manifest.read_bytes() == before_checksum

    rc2 = fd.main(argv)
    assert rc2 == 0
    with (out_root / "manifest.csv").open(newline="", encoding="utf-8") as f:
        rows2 = list(csv.DictReader(f))
    assert len(rows2) == len(rows)
    assert test_set_manifest.read_bytes() == before_checksum


def test_main_rejects_oversized_clone_before_conversion(big_optionb_dir, tmp_path, monkeypatch):
    out_root = tmp_path / "out" / "optionb"
    test_set_manifest = tmp_path / "out" / "test_set" / "manifest.csv"
    make_test_set_manifest(test_set_manifest)

    monkeypatch.setattr(fd, "clone_optionb", lambda clone_dir, repo_url, force=False: (big_optionb_dir, "abc123"))
    monkeypatch.setattr(fd, "MAX_TOTAL_BYTES", 1)

    called = {"build": False}
    real_build = fd.build_target_manifest_rows

    def spy_build(*args, **kwargs):
        called["build"] = True
        return real_build(*args, **kwargs)

    monkeypatch.setattr(fd, "build_target_manifest_rows", spy_build)

    rc = fd.main(["--out-root", str(out_root), "--test-set-manifest", str(test_set_manifest)])

    assert rc == 1
    assert called["build"] is False  # rejected before any conversion work started
    assert not out_root.exists()


def test_main_preserves_prior_good_output_on_mid_run_failure(big_optionb_dir, tmp_path, monkeypatch):
    out_root = tmp_path / "out" / "optionb"
    test_set_manifest = tmp_path / "out" / "test_set" / "manifest.csv"
    make_test_set_manifest(test_set_manifest)

    monkeypatch.setattr(fd, "clone_optionb", lambda clone_dir, repo_url, force=False: (big_optionb_dir, "abc123"))

    argv = ["--out-root", str(out_root), "--test-set-manifest", str(test_set_manifest)]
    rc = fd.main(argv)
    assert rc == 0
    good_manifest = (out_root / "manifest.csv").read_bytes()
    good_marker = out_root / "audio"
    assert good_marker.is_dir()

    def broken_copy_and_measure(copy_jobs, manifest_rows):
        raise RuntimeError("simulated mid-run failure")

    monkeypatch.setattr(fd, "copy_and_measure", broken_copy_and_measure)
    rc2 = fd.main(argv)
    assert rc2 == 1

    assert out_root.is_dir()
    assert (out_root / "manifest.csv").read_bytes() == good_manifest
    assert good_marker.is_dir()
    staging_root = out_root.parent / f".{out_root.name}.staging"
    assert not staging_root.exists()
