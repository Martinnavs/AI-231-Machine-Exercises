"""Tests for me2_voicegen.wakeword.fetch_positives.

No test here performs a real network clone -- `clone_sparse`'s subprocess
calls are either exercised against a fake pre-populated "clone" directory
(the reuse/no-re-download path) or bypassed entirely via monkeypatching
`clone_sparse` itself (mirroring `tests/test_optionb_dataset.py`'s
`clone_optionb` monkeypatch pattern), so the suite runs offline and fast.
The conversion logic (manifest mapping, path-traversal rejection, copy+
measure, waiver-coverage verification, idempotency) is exercised against
synthetic fixtures shaped like the real upstream trees.
"""

from __future__ import annotations

import csv
import struct
import subprocess
import wave
from pathlib import Path

import pytest

from me2_voicegen.wakeword import fetch_positives as fp


def write_wav(path: Path, *, sample_rate=16000, channels=1, sampwidth=2, n_frames=1600) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as w:
        w.setnchannels(channels)
        w.setsampwidth(sampwidth)
        w.setframerate(sample_rate)
        w.writeframes(struct.pack("<" + "h" * (n_frames * channels), *([0] * (n_frames * channels))))


def make_picovoice_clone(tmp_path: Path, n: int, *, name_prefix: str = "pv") -> tuple[Path, list[str]]:
    clone_dir = tmp_path / "clone_picovoice"
    audio_dir = clone_dir / fp.PICOVOICE_SPARSE_PATH
    names = []
    for i in range(n):
        name = f"{name_prefix}-{i:04d}-0000-0000-0000-000000000000.wav"
        write_wav(audio_dir / name)
        names.append(name)
    (clone_dir / ".git").mkdir(parents=True)
    return clone_dir, names


def make_mycroft_clone(
    tmp_path: Path,
    n: int,
    *,
    name_prefix: str = "computer-en",
    uncovered_indices: frozenset[int] = frozenset(),
    header: str = "Files released into public domain:",
    include_readme: bool = True,
) -> tuple[Path, list[str]]:
    clone_dir = tmp_path / "clone_mycroft"
    audio_dir = clone_dir / fp.MYCROFT_SPARSE_PATH
    names = []
    for i in range(n):
        name = f"{name_prefix}-{i:04d}-0000-0000-0000-000000000000.wav"
        write_wav(audio_dir / name)
        names.append(name)
    if include_readme:
        (audio_dir / "README.md").write_text("# Computer\n\nPhonemes\n", encoding="utf-8")

    licenses_dir = clone_dir / fp.MYCROFT_LICENSES_PATH
    licenses_dir.mkdir(parents=True)
    covered = [n for i, n in enumerate(names) if i not in uncovered_indices]
    waiver_text = (
        "I, tester, hereby agree to waive all claim of copyright...\n\n"
        f"{header}\n" + "\n".join(f"computer/en/{name}" for name in covered) + "\n"
    )
    (licenses_dir / "license-20200101-tester.txt").write_text(waiver_text, encoding="utf-8")

    (clone_dir / ".git").mkdir(parents=True)
    return clone_dir, names


# ---------------------------------------------------------------------------
# Path-traversal / sanitization
# ---------------------------------------------------------------------------


def test_resolve_under_accepts_path_inside_root(tmp_path):
    root = tmp_path / "root"
    root.mkdir()
    (root / "sub").mkdir()
    assert fp.resolve_under(root, root / "sub" / "f.wav") == (root / "sub" / "f.wav").resolve()


def test_resolve_under_rejects_traversal_via_dotdot(tmp_path):
    root = tmp_path / "root"
    root.mkdir()
    with pytest.raises(fp.PathTraversalError):
        fp.resolve_under(root, root / ".." / "escaped.wav")


def test_resolve_under_rejects_traversal_via_symlink(tmp_path):
    root = tmp_path / "root"
    root.mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "secret.wav").write_bytes(b"x")
    link = root / "link.wav"
    link.symlink_to(outside / "secret.wav")
    with pytest.raises(fp.PathTraversalError):
        fp.resolve_under(root, link)


@pytest.mark.parametrize(
    "bad",
    ["../evil", "a/b", "..", ".", "", "bad;rm", "-rf", "+HYPERLINK(1)", "=cmd", "@sum(1)"],
)
def test_sanitize_component_rejects_unsafe(bad):
    with pytest.raises(fp.PathTraversalError):
        fp.sanitize_component(bad, "test")


def test_sanitize_component_accepts_safe():
    assert fp.sanitize_component("0386da81-9db7-499c.wav", "test") == "0386da81-9db7-499c.wav"


# ---------------------------------------------------------------------------
# assert_safe_rmtree_root (security-auditor R2-1 regression coverage)
#
# `--out-root ""` normalizes to `Path(".")`, and `main()`'s success path
# does an unconditional `shutil.rmtree(out_root)` -- unguarded, this
# deletes the current working directory's entire contents (reproduced live
# by security-auditor against a real sandbox CWD, ticket 01's Execution
# Log). These tests both unit-test the guard directly and adversarially
# re-verify the exact CWD-destruction scenario end-to-end via `main()`,
# the same way the auditor did.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("bad", ["", ".", "  "])
def test_assert_safe_rmtree_root_rejects_empty_and_dot(tmp_path, bad):
    with pytest.raises(fp.ManifestValidationError, match="empty or refers to the current directory"):
        fp.assert_safe_rmtree_root(Path(bad), "test")


def test_assert_safe_rmtree_root_rejects_home_directory(monkeypatch, tmp_path):
    fake_home = tmp_path / "home" / "someone"
    fake_home.mkdir(parents=True)
    monkeypatch.setattr(Path, "home", lambda: fake_home)
    with pytest.raises(fp.ManifestValidationError, match="home directory"):
        fp.assert_safe_rmtree_root(fake_home, "test")


def test_assert_safe_rmtree_root_rejects_filesystem_root(tmp_path):
    with pytest.raises(fp.ManifestValidationError, match="filesystem root"):
        fp.assert_safe_rmtree_root(Path(tmp_path.anchor), "test")


def test_assert_safe_rmtree_root_accepts_normal_path(tmp_path):
    fp.assert_safe_rmtree_root(tmp_path / "out" / "positives_real", "test")  # must not raise


def test_main_rejects_empty_out_root_without_deleting_cwd(tmp_path, monkeypatch):
    """Adversarial re-verification of the exact scenario security-auditor
    reproduced: run inside a throwaway CWD seeded with marker files/dirs
    that a real `shutil.rmtree(Path("."))` would destroy, and confirm
    `main()` now fails loudly *before* any deletion happens."""
    sandbox = tmp_path / "sandbox_cwd"
    sandbox.mkdir()
    (sandbox / ".git").mkdir()
    (sandbox / ".git" / "HEAD").write_text("ref: refs/heads/main\n", encoding="utf-8")
    (sandbox / "main.py").write_text("print('hello')\n", encoding="utf-8")

    monkeypatch.chdir(sandbox)

    rc = fp.main(["--out-root", "", "--dry-run"])

    assert rc == 1
    assert (sandbox / ".git" / "HEAD").is_file()
    assert (sandbox / "main.py").is_file()


def test_main_rejects_dot_out_root_non_dry_run_without_deleting_cwd(tmp_path, monkeypatch):
    sandbox = tmp_path / "sandbox_cwd"
    sandbox.mkdir()
    (sandbox / ".git").mkdir()
    (sandbox / ".git" / "HEAD").write_text("ref: refs/heads/main\n", encoding="utf-8")
    (sandbox / "main.py").write_text("print('hello')\n", encoding="utf-8")

    monkeypatch.chdir(sandbox)

    rc = fp.main(["--out-root", "."])

    assert rc == 1
    assert (sandbox / ".git" / "HEAD").is_file()
    assert (sandbox / "main.py").is_file()


def test_main_rejects_empty_clone_dir(tmp_path, monkeypatch):
    sandbox = tmp_path / "sandbox_cwd"
    sandbox.mkdir()
    (sandbox / "marker.txt").write_text("x", encoding="utf-8")
    monkeypatch.chdir(sandbox)

    rc = fp.main(["--clone-dir", "", "--out-root", "out/positives_real", "--dry-run"])

    assert rc == 1
    assert (sandbox / "marker.txt").is_file()


# ---------------------------------------------------------------------------
# clone_sparse
# ---------------------------------------------------------------------------


def test_clone_sparse_rejects_non_https_url(tmp_path):
    with pytest.raises(fp.PathTraversalError, match="non-https"):
        fp.clone_sparse(tmp_path / "clone", "git://evil.example/repo", ["x"], marker_relpath="x")


def test_clone_sparse_reuses_existing_valid_clone(tmp_path, monkeypatch):
    clone_dir, _ = make_picovoice_clone(tmp_path, 3)

    def fail_if_invoked(cmd, **kwargs):
        raise AssertionError(f"clone_sparse re-cloned an already-valid clone dir: {cmd}")

    monkeypatch.setattr(subprocess, "run", fail_if_invoked)
    monkeypatch.setattr(fp, "git_commit_sha", lambda repo_dir: "deadbeef")

    resolved_dir, sha = fp.clone_sparse(
        clone_dir, fp.PICOVOICE_REPO_URL, [fp.PICOVOICE_SPARSE_PATH], marker_relpath=fp.PICOVOICE_SPARSE_PATH
    )
    assert resolved_dir == clone_dir
    assert sha == "deadbeef"


def test_clone_sparse_disables_hooks_and_lfs_and_uses_dashdash(tmp_path, monkeypatch):
    clone_dir = tmp_path / "clone"
    calls = []

    def fake_run(cmd, **kwargs):
        calls.append(cmd)
        if "clone" in cmd:
            (clone_dir / ".git" / "info").mkdir(parents=True)
            write_wav(clone_dir / fp.PICOVOICE_SPARSE_PATH / "a.wav")
        return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")

    monkeypatch.setattr(subprocess, "run", fake_run)
    monkeypatch.setattr(fp, "git_commit_sha", lambda repo_dir: "deadbeef")

    fp.clone_sparse(clone_dir, fp.PICOVOICE_REPO_URL, [fp.PICOVOICE_SPARSE_PATH], marker_relpath=fp.PICOVOICE_SPARSE_PATH)

    clone_cmd = next(c for c in calls if "clone" in c)
    assert "filter.lfs.smudge=" in clone_cmd
    assert "filter.lfs.process=" in clone_cmd
    assert "filter.lfs.required=false" in clone_cmd
    assert "core.hooksPath=/dev/null" in clone_cmd
    assert "--" in clone_cmd
    dashdash_idx = clone_cmd.index("--")
    assert clone_cmd[dashdash_idx + 1] == fp.PICOVOICE_REPO_URL


def test_clone_sparse_force_reclones(tmp_path, monkeypatch):
    clone_dir, _ = make_picovoice_clone(tmp_path, 3)
    calls = []

    def fake_run(cmd, **kwargs):
        calls.append(cmd)
        if "clone" in cmd:
            (clone_dir / ".git" / "info").mkdir(parents=True)
            write_wav(clone_dir / fp.PICOVOICE_SPARSE_PATH / "a.wav")
        return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")

    monkeypatch.setattr(subprocess, "run", fake_run)
    monkeypatch.setattr(fp, "git_commit_sha", lambda repo_dir: "deadbeef")

    fp.clone_sparse(
        clone_dir,
        fp.PICOVOICE_REPO_URL,
        [fp.PICOVOICE_SPARSE_PATH],
        marker_relpath=fp.PICOVOICE_SPARSE_PATH,
        force=True,
    )
    assert any("clone" in c for c in calls)


# ---------------------------------------------------------------------------
# discover_wav_files
# ---------------------------------------------------------------------------


def test_discover_wav_files_excludes_non_wav_and_sorts(tmp_path):
    clone_dir, names = make_mycroft_clone(tmp_path, 5)
    audio_dir = clone_dir / fp.MYCROFT_SPARSE_PATH
    found = fp.discover_wav_files(audio_dir, clone_dir)
    assert [f.name for f in found] == sorted(names)
    assert all(f.name != "README.md" for f in found)


def test_discover_wav_files_rejects_symlinked_source(tmp_path):
    clone_dir, _ = make_picovoice_clone(tmp_path, 2)
    audio_dir = clone_dir / fp.PICOVOICE_SPARSE_PATH
    outside = tmp_path / "outside.wav"
    write_wav(outside)
    (audio_dir / "zz-link.wav").symlink_to(outside)

    with pytest.raises(fp.ManifestValidationError, match="non-regular-file/symlinked"):
        fp.discover_wav_files(audio_dir, clone_dir)


def test_discover_wav_files_missing_directory_raises(tmp_path):
    with pytest.raises(fp.ManifestValidationError, match="does not exist"):
        fp.discover_wav_files(tmp_path / "nope", tmp_path)


# ---------------------------------------------------------------------------
# Waiver-coverage verification (Mycroft)
# ---------------------------------------------------------------------------


def test_verify_mycroft_waiver_coverage_all_covered(tmp_path):
    clone_dir, _ = make_mycroft_clone(tmp_path, 10)
    audio_dir = clone_dir / fp.MYCROFT_SPARSE_PATH
    wavs = fp.discover_wav_files(audio_dir, clone_dir)
    assert fp.verify_mycroft_waiver_coverage(clone_dir, wavs) == 10


def test_verify_mycroft_waiver_coverage_alternate_header(tmp_path):
    clone_dir, _ = make_mycroft_clone(tmp_path, 4, header="Files covered by this update:")
    audio_dir = clone_dir / fp.MYCROFT_SPARSE_PATH
    wavs = fp.discover_wav_files(audio_dir, clone_dir)
    assert fp.verify_mycroft_waiver_coverage(clone_dir, wavs) == 4


def test_verify_mycroft_waiver_coverage_bare_filename_does_not_match(tmp_path):
    """Waivers enumerate repo-root-relative paths (`computer/en/<name>`),
    never bare filenames -- a waiver file written with bare names must be
    treated as not covering anything, per docs/WAKEWORD-DATASET-CONTRACT.md
    section 7."""
    clone_dir, names = make_mycroft_clone(tmp_path, 3)
    licenses_dir = clone_dir / fp.MYCROFT_LICENSES_PATH
    (licenses_dir / "license-20200101-tester.txt").write_text(
        "I, tester, ...\n\nFiles released into public domain:\n" + "\n".join(names) + "\n",
        encoding="utf-8",
    )
    audio_dir = clone_dir / fp.MYCROFT_SPARSE_PATH
    wavs = fp.discover_wav_files(audio_dir, clone_dir)
    with pytest.raises(fp.ManifestValidationError, match="not covered by any waiver"):
        fp.verify_mycroft_waiver_coverage(clone_dir, wavs)


def test_verify_mycroft_waiver_coverage_detects_uncovered_file(tmp_path):
    clone_dir, _ = make_mycroft_clone(tmp_path, 5, uncovered_indices=frozenset({2}))
    audio_dir = clone_dir / fp.MYCROFT_SPARSE_PATH
    wavs = fp.discover_wav_files(audio_dir, clone_dir)
    with pytest.raises(fp.ManifestValidationError, match="1 of 5"):
        fp.verify_mycroft_waiver_coverage(clone_dir, wavs)


def test_verify_mycroft_waiver_coverage_missing_licenses_dir(tmp_path):
    clone_dir, _ = make_mycroft_clone(tmp_path, 2)
    import shutil

    shutil.rmtree(clone_dir / fp.MYCROFT_LICENSES_PATH)
    audio_dir = clone_dir / fp.MYCROFT_SPARSE_PATH
    wavs = fp.discover_wav_files(audio_dir, clone_dir)
    with pytest.raises(fp.ManifestValidationError, match="missing"):
        fp.verify_mycroft_waiver_coverage(clone_dir, wavs)


# ---------------------------------------------------------------------------
# build_manifest_rows
# ---------------------------------------------------------------------------


def test_build_manifest_rows_maps_columns(tmp_path):
    clone_dir, names = make_picovoice_clone(tmp_path, 2)
    out_root = tmp_path / "out"
    audio_dir = clone_dir / fp.PICOVOICE_SPARSE_PATH
    wavs = fp.discover_wav_files(audio_dir, clone_dir)

    rows, jobs = fp.build_manifest_rows(wavs, clone_dir, out_root, fp.SOURCE_PICOVOICE)

    assert len(rows) == len(jobs) == 2
    row = rows[0]
    assert row["label"] == fp.LABEL_WAKEWORD
    assert row["source_dataset"] == fp.SOURCE_PICOVOICE
    assert row["group_id"] == Path(row["filename"]).stem
    assert row["source_relpath"] == f"{fp.PICOVOICE_SPARSE_PATH}/{row['filename']}"
    assert row["path"] == f"audio/{fp.SOURCE_PICOVOICE}/{row['filename']}"
    assert row["split"] == ""
    assert row["resampled"] == "False"


def test_build_manifest_rows_rejects_path_traversal(tmp_path):
    clone_dir, _ = make_picovoice_clone(tmp_path, 1)
    out_root = tmp_path / "out"
    outside_wav = tmp_path / "outside.wav"
    write_wav(outside_wav)

    with pytest.raises(fp.PathTraversalError):
        fp.build_manifest_rows([outside_wav], clone_dir, out_root, fp.SOURCE_PICOVOICE)


def test_build_manifest_rows_rejects_duplicate_destination(tmp_path):
    clone_dir, _ = make_picovoice_clone(tmp_path, 1)
    out_root = tmp_path / "out"
    audio_dir = clone_dir / fp.PICOVOICE_SPARSE_PATH
    wavs = fp.discover_wav_files(audio_dir, clone_dir)

    with pytest.raises(fp.ManifestValidationError, match="duplicate destination"):
        fp.build_manifest_rows(wavs + wavs, clone_dir, out_root, fp.SOURCE_PICOVOICE)


# ---------------------------------------------------------------------------
# copy_and_measure
# ---------------------------------------------------------------------------


def test_copy_and_measure_fills_duration_and_sample_rate(tmp_path):
    clone_dir, _ = make_picovoice_clone(tmp_path, 2)
    out_root = tmp_path / "out"
    audio_dir = clone_dir / fp.PICOVOICE_SPARSE_PATH
    wavs = fp.discover_wav_files(audio_dir, clone_dir)
    rows, jobs = fp.build_manifest_rows(wavs, clone_dir, out_root, fp.SOURCE_PICOVOICE)

    fp.copy_and_measure(jobs, rows)

    for row, (_src, dest) in zip(rows, jobs):
        assert dest.is_file()
        assert row["sample_rate"] == "16000"
        assert float(row["duration"]) == pytest.approx(0.1, abs=1e-3)


def test_copy_and_measure_resamples_non_16k_file(tmp_path):
    src = tmp_path / "clone" / "audio" / "computer" / "a.wav"
    write_wav(src, sample_rate=12000, n_frames=12000)
    dest = tmp_path / "out" / "audio" / "picovoice" / "a.wav"
    row = {"duration": None, "sample_rate": None, "resampled": "False"}

    fp.copy_and_measure([(src, dest)], [row])

    assert row["sample_rate"] == "16000"
    assert row["resampled"] == "True"
    with wave.open(str(dest), "rb") as w:
        assert w.getframerate() == 16000
        assert w.getnchannels() == 1
        assert w.getsampwidth() == 2


def test_copy_and_measure_leaves_native_16k_file_unresampled(tmp_path):
    src = tmp_path / "clone" / "audio" / "computer" / "a.wav"
    write_wav(src, sample_rate=16000)
    dest = tmp_path / "out" / "audio" / "picovoice" / "a.wav"
    row = {"duration": None, "sample_rate": None, "resampled": "False"}

    fp.copy_and_measure([(src, dest)], [row])

    assert row["sample_rate"] == "16000"
    assert row["resampled"] == "False"


def test_copy_and_measure_rejects_wrong_format(tmp_path):
    src = tmp_path / "clone" / "audio" / "computer" / "bad.wav"
    write_wav(src, channels=2)
    dest = tmp_path / "out" / "audio" / "picovoice" / "bad.wav"
    row = {"duration": None, "sample_rate": None}

    with pytest.raises(RuntimeError, match="required"):
        fp.copy_and_measure([(src, dest)], [row])


def test_copy_and_measure_enforces_per_file_cap(monkeypatch, tmp_path):
    src = tmp_path / "clone" / "audio" / "computer" / "big.wav"
    write_wav(src)
    dest = tmp_path / "out" / "audio" / "picovoice" / "big.wav"
    monkeypatch.setattr(fp, "MAX_SINGLE_FILE_BYTES", 1)

    with pytest.raises(fp.ManifestValidationError, match="per-file cap"):
        fp.copy_and_measure([(src, dest)], [{"duration": None, "sample_rate": None}])


def test_copy_and_measure_enforces_cumulative_cap(monkeypatch, tmp_path):
    src1 = tmp_path / "clone" / "audio" / "computer" / "a.wav"
    src2 = tmp_path / "clone" / "audio" / "computer" / "b.wav"
    write_wav(src1)
    write_wav(src2)
    dest1 = tmp_path / "out" / "audio" / "picovoice" / "a.wav"
    dest2 = tmp_path / "out" / "audio" / "picovoice" / "b.wav"
    size = src1.stat().st_size
    monkeypatch.setattr(fp, "MAX_TOTAL_BYTES", size)  # first file alone fits, second doesn't

    with pytest.raises(fp.ManifestValidationError, match="cumulative"):
        fp.copy_and_measure(
            [(src1, dest1), (src2, dest2)],
            [{"duration": None, "sample_rate": None}, {"duration": None, "sample_rate": None}],
        )


def test_copy_and_measure_rejects_symlinked_source(tmp_path):
    real = tmp_path / "real.wav"
    write_wav(real)
    src = tmp_path / "clone" / "audio" / "computer" / "link.wav"
    src.parent.mkdir(parents=True)
    src.symlink_to(real)
    dest = tmp_path / "out" / "audio" / "picovoice" / "link.wav"

    with pytest.raises(fp.ManifestValidationError, match="non-regular-file/symlinked"):
        fp.copy_and_measure([(src, dest)], [{"duration": None, "sample_rate": None}])


# ---------------------------------------------------------------------------
# verify_manifest_files_exist
# ---------------------------------------------------------------------------


def test_verify_manifest_files_exist_detects_missing(tmp_path):
    out_root = tmp_path / "out"
    out_root.mkdir()
    rows = [{"filename": "a.wav", "path": "audio/picovoice/a.wav", "source_dataset": "picovoice"}]
    with pytest.raises(fp.ManifestValidationError, match="does not exist"):
        fp.verify_manifest_files_exist(rows, out_root)


def test_verify_manifest_files_exist_detects_duplicate_filename_within_source(tmp_path):
    out_root = tmp_path / "out"
    write_wav(out_root / "audio" / "picovoice" / "a.wav")
    rows = [
        {"filename": "a.wav", "path": "audio/picovoice/a.wav", "source_dataset": "picovoice"},
        {"filename": "a.wav", "path": "audio/picovoice/a.wav", "source_dataset": "picovoice"},
    ]
    with pytest.raises(fp.ManifestValidationError, match="more than one"):
        fp.verify_manifest_files_exist(rows, out_root)


def test_verify_manifest_files_exist_allows_same_filename_across_sources(tmp_path):
    out_root = tmp_path / "out"
    write_wav(out_root / "audio" / "picovoice" / "a.wav")
    write_wav(out_root / "audio" / "mycroft_precise" / "a.wav")
    rows = [
        {"filename": "a.wav", "path": "audio/picovoice/a.wav", "source_dataset": "picovoice"},
        {"filename": "a.wav", "path": "audio/mycroft_precise/a.wav", "source_dataset": "mycroft_precise"},
    ]
    fp.verify_manifest_files_exist(rows, out_root)  # must not raise


# ---------------------------------------------------------------------------
# End-to-end: dry-run, real run, count-sanity, idempotency, mid-run failure
# ---------------------------------------------------------------------------


def _patch_clone_sparse(monkeypatch, picovoice_dir, picovoice_sha, mycroft_dir, mycroft_sha):
    def fake_clone_sparse(clone_dir, repo_url, sparse_paths, marker_relpath, force=False):
        if marker_relpath == fp.PICOVOICE_SPARSE_PATH:
            return picovoice_dir, picovoice_sha
        return mycroft_dir, mycroft_sha

    monkeypatch.setattr(fp, "clone_sparse", fake_clone_sparse)


def test_dry_run_reports_without_writing(tmp_path, monkeypatch):
    picovoice_dir, _ = make_picovoice_clone(tmp_path, fp.EXPECTED_PICOVOICE_COUNT)
    mycroft_dir, _ = make_mycroft_clone(tmp_path, fp.EXPECTED_MYCROFT_COUNT)
    _patch_clone_sparse(monkeypatch, picovoice_dir, "pvsha", mycroft_dir, "mycsha")

    out_root = tmp_path / "out" / "positives_real"
    rc = fp.main(["--out-root", str(out_root), "--dry-run"])

    assert rc == 0
    assert not out_root.exists()


def test_full_run_writes_manifest_and_is_idempotent(tmp_path, monkeypatch):
    picovoice_dir, _ = make_picovoice_clone(tmp_path, fp.EXPECTED_PICOVOICE_COUNT)
    mycroft_dir, _ = make_mycroft_clone(tmp_path, fp.EXPECTED_MYCROFT_COUNT)
    _patch_clone_sparse(monkeypatch, picovoice_dir, "pvsha", mycroft_dir, "mycsha")

    out_root = tmp_path / "out" / "positives_real"
    argv = ["--out-root", str(out_root)]
    rc = fp.main(argv)
    assert rc == 0

    with (out_root / "manifest.csv").open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == fp.EXPECTED_TOTAL_ROWS
    assert all(r["label"] == fp.LABEL_WAKEWORD for r in rows)
    assert {r["source_dataset"] for r in rows} == {fp.SOURCE_PICOVOICE, fp.SOURCE_MYCROFT}
    assert all((out_root / r["path"]).is_file() for r in rows)
    assert (out_root / "summary.md").is_file()
    summary = (out_root / "summary.md").read_text(encoding="utf-8")
    assert "pvsha" in summary
    assert "mycsha" in summary
    assert "heycomputer" in summary  # scope-exclusion note

    rc2 = fp.main(argv)
    assert rc2 == 0
    with (out_root / "manifest.csv").open(newline="", encoding="utf-8") as f:
        rows2 = list(csv.DictReader(f))
    assert len(rows2) == len(rows)


def test_main_rejects_picovoice_count_mismatch(tmp_path, monkeypatch):
    picovoice_dir, _ = make_picovoice_clone(tmp_path, fp.EXPECTED_PICOVOICE_COUNT - 1)
    mycroft_dir, _ = make_mycroft_clone(tmp_path, fp.EXPECTED_MYCROFT_COUNT)
    _patch_clone_sparse(monkeypatch, picovoice_dir, "pvsha", mycroft_dir, "mycsha")

    out_root = tmp_path / "out" / "positives_real"
    rc = fp.main(["--out-root", str(out_root), "--dry-run"])

    assert rc == 1
    assert not out_root.exists()


def test_main_rejects_mycroft_count_mismatch(tmp_path, monkeypatch):
    picovoice_dir, _ = make_picovoice_clone(tmp_path, fp.EXPECTED_PICOVOICE_COUNT)
    mycroft_dir, _ = make_mycroft_clone(tmp_path, fp.EXPECTED_MYCROFT_COUNT + 1)
    _patch_clone_sparse(monkeypatch, picovoice_dir, "pvsha", mycroft_dir, "mycsha")

    out_root = tmp_path / "out" / "positives_real"
    rc = fp.main(["--out-root", str(out_root), "--dry-run"])

    assert rc == 1
    assert not out_root.exists()


def test_main_rejects_uncovered_mycroft_waiver(tmp_path, monkeypatch):
    picovoice_dir, _ = make_picovoice_clone(tmp_path, fp.EXPECTED_PICOVOICE_COUNT)
    mycroft_dir, _ = make_mycroft_clone(tmp_path, fp.EXPECTED_MYCROFT_COUNT, uncovered_indices=frozenset({0}))
    _patch_clone_sparse(monkeypatch, picovoice_dir, "pvsha", mycroft_dir, "mycsha")

    out_root = tmp_path / "out" / "positives_real"
    rc = fp.main(["--out-root", str(out_root), "--dry-run"])

    assert rc == 1
    assert not out_root.exists()


def test_main_rejects_oversized_clone_before_conversion(tmp_path, monkeypatch):
    picovoice_dir, _ = make_picovoice_clone(tmp_path, fp.EXPECTED_PICOVOICE_COUNT)
    mycroft_dir, _ = make_mycroft_clone(tmp_path, fp.EXPECTED_MYCROFT_COUNT)
    _patch_clone_sparse(monkeypatch, picovoice_dir, "pvsha", mycroft_dir, "mycsha")
    monkeypatch.setattr(fp, "MAX_TOTAL_BYTES", 1)

    called = {"build": False}
    real_build = fp.build_manifest_rows

    def spy_build(*args, **kwargs):
        called["build"] = True
        return real_build(*args, **kwargs)

    monkeypatch.setattr(fp, "build_manifest_rows", spy_build)

    out_root = tmp_path / "out" / "positives_real"
    rc = fp.main(["--out-root", str(out_root)])

    assert rc == 1
    assert called["build"] is False
    assert not out_root.exists()


def test_main_preserves_prior_good_output_on_mid_run_failure(tmp_path, monkeypatch):
    picovoice_dir, _ = make_picovoice_clone(tmp_path, fp.EXPECTED_PICOVOICE_COUNT)
    mycroft_dir, _ = make_mycroft_clone(tmp_path, fp.EXPECTED_MYCROFT_COUNT)
    _patch_clone_sparse(monkeypatch, picovoice_dir, "pvsha", mycroft_dir, "mycsha")

    out_root = tmp_path / "out" / "positives_real"
    argv = ["--out-root", str(out_root)]
    rc = fp.main(argv)
    assert rc == 0
    good_manifest = (out_root / "manifest.csv").read_bytes()
    good_marker = out_root / "audio"
    assert good_marker.is_dir()

    def broken_copy_and_measure(copy_jobs, manifest_rows):
        raise RuntimeError("simulated mid-run failure")

    monkeypatch.setattr(fp, "copy_and_measure", broken_copy_and_measure)
    rc2 = fp.main(argv)
    assert rc2 == 1

    assert out_root.is_dir()
    assert (out_root / "manifest.csv").read_bytes() == good_manifest
    assert good_marker.is_dir()
    staging_root = out_root.parent / f".{out_root.name}.staging"
    assert not staging_root.exists()
