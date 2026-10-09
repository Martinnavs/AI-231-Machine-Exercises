"""Tests for me2_voicegen.wakeword.convert_positives.

`convert_voice` is stubbed everywhere except the single @pytest.mark.slow
test at the bottom (real CosyVoice2 backend, real reference clip, real
model on disk) -- mirrors this feature's ticket 01 test suite's
network-vs-offline split, but for "real model" vs "stubbed model" instead
of "real clone" vs "stubbed clone".
"""

from __future__ import annotations

import csv
import wave
from pathlib import Path

import numpy as np
import pytest

from me2_voicegen.synthesis.base import SynthesisResult
from me2_voicegen.wakeword import convert_positives as cp
from me2_voicegen.wakeword import fetch_positives as fp


def make_ref_pool(tmp_path: Path, vcm_wav_factory, spec: dict[str, int]) -> Path:
    """spec: family name -> count, e.g. {"tagalog": 3, "korean": 1}."""
    refs_dir = tmp_path / "refs"
    for family, count in spec.items():
        for i in range(1, count + 1):
            vcm_wav_factory(refs_dir / f"{family}{i}.wav", duration_s=1.0)
    return refs_dir


def probe(path: Path) -> tuple[int, int, int]:
    with wave.open(str(path), "rb") as w:
        return w.getframerate(), w.getnchannels(), w.getsampwidth()


class StubSynthesizer:
    """convert_voice stub: returns a fixed-shape 24kHz mono result (matching
    the real CosyVoice2 backend's actual output rate, per this session's
    real pilot run), unless `source_wav_path` is in `fail_on`."""

    def __init__(self, sample_rate: int = 24000, fail_on: frozenset[str] = frozenset()):
        self.sample_rate = sample_rate
        self.fail_on = fail_on
        self.calls: list[tuple[str, str]] = []

    def convert_voice(self, source_wav_path: str, prompt_wav_path: str) -> SynthesisResult:
        self.calls.append((source_wav_path, prompt_wav_path))
        if prompt_wav_path in self.fail_on:
            raise RuntimeError("stubbed conversion failure")
        t = np.arange(int(self.sample_rate * 0.5), dtype=np.float32) / self.sample_rate
        audio = (0.05 * np.sin(2 * np.pi * 220 * t)).astype(np.float32)[np.newaxis, :]
        return SynthesisResult(audio=audio, sample_rate=self.sample_rate)


# ---------------------------------------------------------------------------
# stratified_sample_refs
# ---------------------------------------------------------------------------


def test_stratified_sample_guarantees_one_per_family():
    import random

    families = {
        "tagalog": [Path(f"tagalog{i}.mp3") for i in range(1, 17)],
        "indonesian": [Path(f"indonesian{i}.mp3") for i in range(1, 9)],
        "mongolian": [Path(f"mongolian{i}.mp3") for i in range(1, 10)],
        "ilonggo": [Path("ilonggo1.mp3")],
        "korean": [Path("korean31.mp3")],
    }
    picked = cp.stratified_sample_refs(random.Random(0), families, k=8)
    picked_families = {p.stem.rstrip("0123456789") for p in picked}
    assert len(picked) == 8
    assert len(set(picked)) == 8  # no duplicates
    assert picked_families == set(families)  # every family represented
    assert Path("ilonggo1.mp3") in picked  # the family's only clip, guaranteed
    assert Path("korean31.mp3") in picked


def test_stratified_sample_matches_hand_derived_real_breakdown():
    # Real reference-voice pool breakdown this ticket re-verified live
    # (tagalog=16, indonesian=8, mongolian=9, ilonggo=1, korean=1) and its
    # hand-derived K=8 apportionment (see module docstring / ticket
    # Execution Log): ilonggo=1, korean=1, indonesian=2, mongolian=2,
    # tagalog=2.
    import random

    families = {
        "tagalog": [Path(f"tagalog{i}.mp3") for i in range(1, 17)],
        "indonesian": [Path(f"indonesian{i}.mp3") for i in range(1, 9)],
        "mongolian": [Path(f"mongolian{i}.mp3") for i in range(1, 10)],
        "ilonggo": [Path("ilonggo1.mp3")],
        "korean": [Path("korean31.mp3")],
    }
    picked = cp.stratified_sample_refs(random.Random(123), families, k=8)
    counts: dict[str, int] = {}
    for p in picked:
        fam = p.stem.rstrip("0123456789")
        counts[fam] = counts.get(fam, 0) + 1
    assert counts == {"tagalog": 2, "indonesian": 2, "mongolian": 2, "ilonggo": 1, "korean": 1}


def test_stratified_sample_deterministic_given_same_rng_state():
    import random

    families = {"a": [Path("a1"), Path("a2"), Path("a3")], "b": [Path("b1"), Path("b2")]}
    r1 = cp.stratified_sample_refs(random.Random(5), families, k=3)
    r2 = cp.stratified_sample_refs(random.Random(5), families, k=3)
    assert r1 == r2


def test_stratified_sample_caps_at_total_available():
    import random

    families = {"a": [Path("a1")], "b": [Path("b1")]}
    picked = cp.stratified_sample_refs(random.Random(0), families, k=8)
    assert len(picked) == 2


# ---------------------------------------------------------------------------
# discover_reference_families
# ---------------------------------------------------------------------------


def test_discover_reference_families_groups_by_alpha_prefix(tmp_path, vcm_wav_factory):
    refs_dir = make_ref_pool(tmp_path, vcm_wav_factory, {"tagalog": 2, "korean": 1})
    families = cp.discover_reference_families(refs_dir)
    assert set(families) == {"tagalog", "korean"}
    assert len(families["tagalog"]) == 2
    assert len(families["korean"]) == 1


def test_discover_reference_families_rejects_non_alpha_prefix(tmp_path, vcm_wav_factory):
    vcm_wav_factory(tmp_path / "refs" / "007.wav", duration_s=0.5)
    with pytest.raises(ValueError, match="language family"):
        cp.discover_reference_families(tmp_path / "refs")


# ---------------------------------------------------------------------------
# build_conversion_jobs / group_id fidelity / determinism
# ---------------------------------------------------------------------------


def test_build_conversion_jobs_preserves_group_id_and_source_relpath(
    tmp_path, wakeword_fake_manifest_factory, vcm_wav_factory
):
    manifest_path = wakeword_fake_manifest_factory(
        [
            {"filename": "aaa.wav", "group_id": "aaa"},
            {"filename": "bbb.wav", "group_id": "bbb"},
        ]
    )
    positive_rows = cp.load_positives_manifest(manifest_path)
    refs_dir = make_ref_pool(tmp_path, vcm_wav_factory, {"tagalog": 3, "korean": 1})
    families = cp.discover_reference_families(refs_dir)

    jobs = cp.build_conversion_jobs(positive_rows, manifest_path.parent, families, refs_dir, k=4, seed=1)

    assert len(jobs) == 2 * 4
    for row, group_id in zip(positive_rows, ("aaa", "bbb")):
        matching = [j for j in jobs if j["group_id"] == group_id]
        assert len(matching) == 4
        assert all(j["source_relpath"] == row["path"] for j in matching)


def test_build_conversion_jobs_same_seed_same_pairing_and_order(
    tmp_path, wakeword_fake_manifest_factory, vcm_wav_factory
):
    manifest_path = wakeword_fake_manifest_factory([{"filename": f"{i}.wav", "group_id": str(i)} for i in range(5)])
    positive_rows = cp.load_positives_manifest(manifest_path)
    refs_dir = make_ref_pool(tmp_path, vcm_wav_factory, {"tagalog": 5, "indonesian": 3, "korean": 1})
    families = cp.discover_reference_families(refs_dir)

    jobs_a = cp.build_conversion_jobs(positive_rows, manifest_path.parent, families, refs_dir, k=4, seed=99)
    jobs_b = cp.build_conversion_jobs(positive_rows, manifest_path.parent, families, refs_dir, k=4, seed=99)
    pairing_a = [(j["group_id"], j["ref_voice"]) for j in jobs_a]
    pairing_b = [(j["group_id"], j["ref_voice"]) for j in jobs_b]
    assert pairing_a == pairing_b


def test_build_conversion_jobs_different_seed_can_differ(
    tmp_path, wakeword_fake_manifest_factory, vcm_wav_factory
):
    manifest_path = wakeword_fake_manifest_factory([{"filename": f"{i}.wav", "group_id": str(i)} for i in range(8)])
    positive_rows = cp.load_positives_manifest(manifest_path)
    refs_dir = make_ref_pool(tmp_path, vcm_wav_factory, {"tagalog": 8, "indonesian": 5, "korean": 1})
    families = cp.discover_reference_families(refs_dir)

    jobs_a = cp.build_conversion_jobs(positive_rows, manifest_path.parent, families, refs_dir, k=3, seed=1)
    jobs_b = cp.build_conversion_jobs(positive_rows, manifest_path.parent, families, refs_dir, k=3, seed=2)
    pairing_a = [(j["group_id"], j["ref_voice"]) for j in jobs_a]
    pairing_b = [(j["group_id"], j["ref_voice"]) for j in jobs_b]
    assert pairing_a != pairing_b


def test_build_conversion_jobs_rejects_unsafe_group_id(tmp_path, wakeword_fake_manifest_factory, vcm_wav_factory):
    manifest_path = wakeword_fake_manifest_factory([{"filename": "x.wav", "group_id": "../escape"}])
    positive_rows = cp.load_positives_manifest(manifest_path)
    refs_dir = make_ref_pool(tmp_path, vcm_wav_factory, {"tagalog": 1})
    families = cp.discover_reference_families(refs_dir)
    with pytest.raises(fp.PathTraversalError):
        cp.build_conversion_jobs(positive_rows, manifest_path.parent, families, refs_dir, k=1, seed=0)


def test_build_conversion_jobs_missing_source_file_raises(tmp_path, wakeword_fake_manifest_factory, vcm_wav_factory):
    manifest_path = wakeword_fake_manifest_factory([{"filename": "x.wav", "group_id": "x"}])
    (manifest_path.parent / "audio" / "picovoice" / "x.wav").unlink()
    positive_rows = cp.load_positives_manifest(manifest_path)
    refs_dir = make_ref_pool(tmp_path, vcm_wav_factory, {"tagalog": 1})
    families = cp.discover_reference_families(refs_dir)
    with pytest.raises(fp.ManifestValidationError):
        cp.build_conversion_jobs(positive_rows, manifest_path.parent, families, refs_dir, k=1, seed=0)


# ---------------------------------------------------------------------------
# load_positives_manifest
# ---------------------------------------------------------------------------


def test_load_positives_manifest_missing_file_raises(tmp_path):
    with pytest.raises(fp.ManifestValidationError):
        cp.load_positives_manifest(tmp_path / "nope.csv")


def test_load_positives_manifest_missing_columns_raises(tmp_path):
    bad = tmp_path / "manifest.csv"
    with bad.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["filename", "label"])
        writer.writeheader()
        writer.writerow({"filename": "a.wav", "label": "_wakeword_"})
    with pytest.raises(fp.ManifestValidationError):
        cp.load_positives_manifest(bad)


# ---------------------------------------------------------------------------
# main(): --dry-run never loads a model, reports full planned count
# ---------------------------------------------------------------------------


def test_dry_run_reports_full_planned_count_and_never_creates_synthesizer(
    tmp_path, wakeword_fake_manifest_factory, vcm_wav_factory, monkeypatch, capsys
):
    manifest_path = wakeword_fake_manifest_factory([{"filename": f"{i}.wav", "group_id": str(i)} for i in range(3)])
    refs_dir = make_ref_pool(tmp_path, vcm_wav_factory, {"tagalog": 3, "korean": 1})

    def boom(*a, **kw):
        raise AssertionError("create_synthesizer must not be called during --dry-run")

    monkeypatch.setattr(cp, "create_synthesizer", boom)

    rc = cp.main(
        [
            "--positives-manifest",
            str(manifest_path),
            "--refs-dir",
            str(refs_dir),
            "--out-root",
            str(tmp_path / "out"),
            "--seed",
            "0",
            "--k",
            "4",
            "--dry-run",
        ]
    )
    assert rc == 0
    out = capsys.readouterr().out
    assert "12 planned pairs" in out  # 3 clips x k=4
    assert not (tmp_path / "out").exists()


# ---------------------------------------------------------------------------
# main(): real conversion path (stubbed backend)
# ---------------------------------------------------------------------------


def _run_convert(
    tmp_path,
    wakeword_fake_manifest_factory,
    vcm_wav_factory,
    monkeypatch,
    *,
    n_clips: int = 2,
    k: int = 3,
    seed: int = 0,
    fail_on: frozenset[str] = frozenset(),
    max_conversions=None,
    out_root=None,
):
    manifest_path = wakeword_fake_manifest_factory([{"filename": f"{i}.wav", "group_id": str(i)} for i in range(n_clips)])
    refs_dir = make_ref_pool(tmp_path, vcm_wav_factory, {"tagalog": 3, "indonesian": 2, "korean": 1})
    out_root = out_root or (tmp_path / "converted")

    stub = StubSynthesizer(fail_on=fail_on)
    monkeypatch.setattr(cp, "create_synthesizer", lambda *a, **kw: stub)

    argv = [
        "--positives-manifest",
        str(manifest_path),
        "--refs-dir",
        str(refs_dir),
        "--out-root",
        str(out_root),
        "--seed",
        str(seed),
        "--k",
        str(k),
    ]
    if max_conversions is not None:
        argv += ["--max-conversions", str(max_conversions)]
    rc = cp.main(argv)
    return rc, out_root, stub


def test_real_run_writes_contract_plus_ref_voice_manifest(
    tmp_path, wakeword_fake_manifest_factory, vcm_wav_factory, monkeypatch
):
    rc, out_root, stub = _run_convert(tmp_path, wakeword_fake_manifest_factory, vcm_wav_factory, monkeypatch)
    assert rc == 0

    manifest_path = out_root / "manifest.csv"
    with manifest_path.open(newline="") as f:
        rows = list(csv.DictReader(f))
    assert set(rows[0]) == set(cp.MANIFEST_FIELDS)
    assert set(fp.MANIFEST_FIELDS) < set(cp.MANIFEST_FIELDS)  # contract's 10 cols + ref_voice
    assert len(rows) == 2 * 3
    for row in rows:
        assert row["label"] == "_wakeword_"
        assert row["source_dataset"] == "cosyvoice_conversion"
        assert row["ref_voice"]
        assert row["split"] == ""
        wav_path = out_root / row["path"]
        assert wav_path.is_file()
        sr, ch, sw = probe(wav_path)
        assert (sr, ch, sw) == (fp.REQUIRED_SR, fp.REQUIRED_CHANNELS, fp.REQUIRED_SAMPWIDTH)
        assert row["resampled"] == "True"  # stub emits 24kHz, always needs coercion


def test_real_run_group_id_shared_across_all_k_conversions_of_one_clip(
    tmp_path, wakeword_fake_manifest_factory, vcm_wav_factory, monkeypatch
):
    rc, out_root, _ = _run_convert(tmp_path, wakeword_fake_manifest_factory, vcm_wav_factory, monkeypatch, n_clips=1, k=3)
    assert rc == 0
    with (out_root / "manifest.csv").open(newline="") as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 3
    assert len({row["group_id"] for row in rows}) == 1
    assert rows[0]["group_id"] == "0"


def test_real_run_partial_failure_semantics(tmp_path, wakeword_fake_manifest_factory, vcm_wav_factory, monkeypatch):
    manifest_path = wakeword_fake_manifest_factory([{"filename": "0.wav", "group_id": "0"}])
    refs_dir = make_ref_pool(tmp_path, vcm_wav_factory, {"tagalog": 3, "korean": 1})
    positive_rows = cp.load_positives_manifest(manifest_path)
    families = cp.discover_reference_families(refs_dir)
    jobs = cp.build_conversion_jobs(positive_rows, manifest_path.parent, families, refs_dir, k=4, seed=0)
    fail_ref_path = str(jobs[0]["ref_path"])

    stub = StubSynthesizer(fail_on=frozenset({fail_ref_path}))
    monkeypatch.setattr(cp, "create_synthesizer", lambda *a, **kw: stub)

    rc = cp.main(
        [
            "--positives-manifest",
            str(manifest_path),
            "--refs-dir",
            str(refs_dir),
            "--out-root",
            str(tmp_path / "converted"),
            "--seed",
            "0",
            "--k",
            "4",
        ]
    )
    assert rc == 1  # non-zero exit on any failure, matching generate_conversions.py's convention
    with (tmp_path / "converted" / "manifest.csv").open(newline="") as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 3  # 4 pairs planned, all 4 attempted, 1 failed -> 3 written


def test_real_run_summary_records_ref_voice_counts(tmp_path, wakeword_fake_manifest_factory, vcm_wav_factory, monkeypatch):
    rc, out_root, _ = _run_convert(tmp_path, wakeword_fake_manifest_factory, vcm_wav_factory, monkeypatch, n_clips=2, k=3)
    assert rc == 0
    summary = (out_root / "summary.md").read_text()
    assert "Realized per-reference-voice counts" in summary
    assert "seed=0" in summary
    assert "k=3" in summary


def test_max_conversions_caps_real_work_but_not_dry_run_projection(
    tmp_path, wakeword_fake_manifest_factory, vcm_wav_factory, monkeypatch
):
    rc, out_root, stub = _run_convert(
        tmp_path, wakeword_fake_manifest_factory, vcm_wav_factory, monkeypatch, n_clips=3, k=4, max_conversions=2
    )
    assert rc == 0
    assert len(stub.calls) == 2
    with (out_root / "manifest.csv").open(newline="") as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 2


def test_convert_voice_missing_on_backend_fails_loudly(
    tmp_path, wakeword_fake_manifest_factory, vcm_wav_factory, monkeypatch
):
    manifest_path = wakeword_fake_manifest_factory([{"filename": "0.wav", "group_id": "0"}])
    refs_dir = make_ref_pool(tmp_path, vcm_wav_factory, {"tagalog": 1})

    class NoConvert:
        pass

    monkeypatch.setattr(cp, "create_synthesizer", lambda *a, **kw: NoConvert())

    rc = cp.main(
        [
            "--positives-manifest",
            str(manifest_path),
            "--refs-dir",
            str(refs_dir),
            "--out-root",
            str(tmp_path / "converted"),
            "--seed",
            "0",
            "--k",
            "1",
        ]
    )
    assert rc == 1
    assert not (tmp_path / "converted").exists()


def test_no_resynthesis_branch(tmp_path):
    """Non-goal: this module never re-types content through Whisper/TTS.
    Checked structurally against the actual code surface (imports, CLI
    flags), not the module's own docstring, which legitimately mentions
    "resynthesis" while explaining the exclusion."""
    source = Path(cp.__file__).read_text()
    assert "transcribe_cached" not in source
    assert "--resynth-prob" not in source
    assert "--whisper-model" not in source
    assert not hasattr(cp, "transcribe_cached")


# ---------------------------------------------------------------------------
# Real-model coverage (slow)
# ---------------------------------------------------------------------------

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
_REAL_POSITIVES_MANIFEST = _PROJECT_ROOT / "out" / "conversions" / "v2" / "wakeword" / "positives_real" / "manifest.csv"
_REAL_REFS_DIR = Path.home() / "cosy-voice-data" / "References"
_REAL_MODEL_DIR = _PROJECT_ROOT / "models" / "CosyVoice2-0.5B"


@pytest.mark.slow
@pytest.mark.skipif(
    not (_REAL_POSITIVES_MANIFEST.is_file() and _REAL_REFS_DIR.is_dir() and _REAL_MODEL_DIR.is_dir()),
    reason="requires ticket 01's real positives manifest, the real reference-voice pool, and the real CosyVoice2 model on disk",
)
def test_real_backend_converts_one_pair_end_to_end(tmp_path):
    out_root = tmp_path / "converted"
    rc = cp.main(
        [
            "--positives-manifest",
            str(_REAL_POSITIVES_MANIFEST),
            "--refs-dir",
            str(_REAL_REFS_DIR),
            "--out-root",
            str(out_root),
            "--seed",
            "0",
            "--k",
            "1",
            "--max-conversions",
            "1",
        ]
    )
    assert rc == 0
    with (out_root / "manifest.csv").open(newline="") as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 1
    row = rows[0]
    wav_path = out_root / row["path"]
    sr, ch, sw = probe(wav_path)
    assert (sr, ch, sw) == (fp.REQUIRED_SR, fp.REQUIRED_CHANNELS, fp.REQUIRED_SAMPWIDTH)
