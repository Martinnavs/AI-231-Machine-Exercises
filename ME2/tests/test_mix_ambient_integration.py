"""Integration tests for dataset_tools.mix_ambient_noise (SPEC.md Proof #4):
the full mix -> gate -> finalize path over fixture manifests with real (tiny)
waveforms and mock transcriber reports in the exact on-disk format
(`build_sanitized_dataset.parse_report` parses). Verifies the derived-manifest
invariants: split inheritance, provenance columns, 0/1/2-sibling cases,
ineligibility rules, and the wakeword trigger gate.
"""

from __future__ import annotations

import csv
import math
from pathlib import Path

import pytest
import torch
import torchaudio

from me2_voicegen.common.ambient_mix import AmbientNoiseError
from me2_voicegen.dataset_tools.mix_ambient_noise import (
    decodes_as_wakeword,
    load_qa_plan,
    negative_passed,
    stage_finalize,
    stage_mix,
    v1_pair_passed,
)

VCM_FIELDS = [
    "filename", "path", "bucket", "label", "duration", "sample_rate", "resampled",
    "source_dataset", "source_relpath", "group_id", "split", "transcript", "original_dataset",
]
SESAME_FIELDS = [
    "filename", "path", "label", "duration", "sample_rate", "resampled", "source_dataset",
    "source_relpath", "group_id", "split", "ref_voice", "noise_source_file", "snr_db",
    "speech_start_s", "speech_end_s",
]


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _write_wav(path: Path, sr: int = 16000, n: int = 16000, seed: int = 0, amplitude: float = 0.1) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    g = torch.Generator().manual_seed(seed)
    wave = (torch.randn(1, n, generator=g) * amplitude).clamp(-1.0, 1.0)
    torchaudio.save(str(path), wave, sr, bits_per_sample=16)


def _sine_wav(path: Path, duration_s: float = 1.0) -> None:
    n = int(duration_s * 16000)
    t = torch.arange(n, dtype=torch.float32) / 16000
    wave = (0.3 * torch.sin(2 * math.pi * 440.0 * t)).unsqueeze(0)
    path.parent.mkdir(parents=True, exist_ok=True)
    torchaudio.save(str(path), wave, 16000, bits_per_sample=16)


def _make_corpus(root: Path) -> Path:
    root = Path(root)
    (root / "audio").mkdir(parents=True)
    rows = []
    i = 0
    for split in ("train", "val", "test"):
        for _ in range(2):
            i += 1
            name = f"chunk_{i:03d}.wav"
            _write_wav(root / "audio" / name, n=1600, seed=i)
            rows.append(
                {
                    "filename": name,
                    "source_file": f"src_{i % 2}.wav",
                    "split": split,
                    "start_s": "0.0",
                    "duration": "0.1",
                    "rms": "0.1",
                }
            )
    with (root / "manifest.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["filename", "source_file", "split", "start_s", "duration", "rms"])
        writer.writeheader()
        writer.writerows(rows)
    return root


def _write_manifest(path: Path, fields: list[str], rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields, restval="")
        writer.writeheader()
        writer.writerows(rows)


def _vcm_row(i: int, split: str = "train", **overrides) -> dict:
    row = {
        "filename": f"clip_{i:04d}.wav",
        "path": f"audio/clip_{i:04d}.wav",
        "bucket": "target_commands",
        "label": "PLAY_MUSIC",
        "duration": "1.000000",
        "sample_rate": "16000",
        "resampled": "False",
        "source_dataset": "optionb",
        "source_relpath": f"audio/clip_{i:04d}.wav",
        "group_id": f"g{i // 5:03d}",
        "split": split,
        "transcript": f"play music {i}",
        "original_dataset": "",
    }
    row.update(overrides)
    return row


def _build_vcm_root(root: Path, n_eligible: int = 16) -> tuple[Path, list[dict]]:
    rows: list[dict] = [_vcm_row(i) for i in range(n_eligible)]
    # ineligible: _noisy source_dataset (wakeword-style), _noisy filename stem
    # (Option B's own convention -- source_dataset stays "optionb"), empty
    # transcript, babble bucket
    rows.append(_vcm_row(90, source_dataset="optionb_noisy", filename="clip_90.wav", path="audio/clip_90.wav", source_relpath="audio/clip_90.wav"))
    rows.append(_vcm_row(93, filename="clip_93_noisy.wav", path="audio/clip_93_noisy.wav", source_relpath="audio/clip_93_noisy.wav"))
    rows.append(_vcm_row(91, transcript=""))
    rows.append(_vcm_row(92, bucket="babble", transcript="", label="unknown"))
    for row in rows:
        _sine_wav(root / row["path"])
    manifest = root / "manifest.csv"
    _write_manifest(manifest, VCM_FIELDS, rows)
    return manifest, rows


def _write_v1_pair_report(reports_dir: Path, unit: str, source_dir: Path, flagged: list[str]) -> None:
    reports_dir.mkdir(parents=True, exist_ok=True)
    lines = [
        f"# QA Report: {unit}",
        "",
        f"- **Source**: {source_dir}",
        "- **Files**: 1",
        "- **Model**: small",
        "- **Threshold**: 0.8",
        "",
        "---",
        "",
        "## Flagged",
        "",
        "| file | expected | transcribed | score | exact? |",
        "|---|---|---|---|---|",
    ]
    for name in flagged:
        lines.append(f"| {name} | x | y | 0.5 | no |")
    (reports_dir / f"report-{unit}.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _shim_name(transcript: str, group_id: str, filename: str, attempt: int) -> str:
    return f"{transcript} - {group_id}__{Path(filename).stem}-a{attempt}.wav"


# ---------------------------------------------------------------------------
# vcm end-to-end
# ---------------------------------------------------------------------------


def test_vcm_mix_finalize_end_to_end(tmp_path):
    corpus = _make_corpus(tmp_path / "corpus")
    root = tmp_path / "vcmx"
    manifest, base_rows = _build_vcm_root(root, n_eligible=16)
    out_root = root / "out"

    qa_plan = stage_mix(
        model="vcm",
        manifest_path=manifest,
        corpus_root=corpus,
        out_root=out_root,
        p_mix=1.0,
        seed=0,
    )
    assert qa_plan is not None
    staging = out_root / ".ambient.staging"
    plan_rows = load_qa_plan(staging)
    assert len(plan_rows) == 16 * 2  # 16 eligible rows x 2 attempts (the 3 ineligible rows are skipped)
    # every staged wav exists and is 16k mono
    for r in plan_rows:
        wave, sr = torchaudio.load(str(staging / r["wav"]))
        assert sr == 16000 and wave.shape[0] == 1

    # flag exactly row0/attempt1's shim name in the report -> row0 ships 1 sibling (a2), all others 2
    flagged_name = _shim_name("play music 0", "g000", "clip_0000.wav", 1)
    reports_dir = tmp_path / "reports"
    _write_v1_pair_report(reports_dir, "vcm", staging / "qa" / "shim" / "vcm", [flagged_name])

    manifest_out = out_root / "final" / "manifest.csv"
    audio_dir = out_root / "final" / "audio"
    summary_out = out_root / "final" / "summary.md"
    stage_finalize(
        model="vcm",
        manifest_path=manifest,
        staging=staging,
        reports_dir=reports_dir,
        negatives_decisions=None,
        audio_dir=audio_dir,
        manifest_out=manifest_out,
        summary_out=summary_out,
        seed=0,
        p_mix=1.0,
        snr_min_db=0.0,
        snr_max_db=30.0,
    )

    with manifest_out.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        fields = list(reader.fieldnames)
        final_rows = list(reader)

    # 20 base rows (16 eligible + 4 ineligible) + 15*2 + 1 = 31 ambient rows
    assert len(final_rows) == 20 + 31
    assert fields[-3:] == ["noise_source_file", "noise_chunk_id", "snr_db"]
    ambient = [r for r in final_rows if r["source_dataset"].endswith("_ambient")]
    assert len(ambient) == 31

    prefixes = [(f"{b['group_id']}__{Path(b['filename']).stem}__amb", b) for b in base_rows]
    for r in ambient:
        base = next(b for prefix, b in prefixes if r["filename"].startswith(prefix))
        assert r["split"] == base["split"]
        assert r["transcript"] == base["transcript"]
        assert r["label"] == base["label"]
        assert r["bucket"] == "target_commands"
        assert r["source_dataset"] == base["source_dataset"] + "_ambient"
        assert r["path"] == f"audio/{r['filename']}"
        assert (audio_dir / r["filename"]).is_file()
        assert r["noise_source_file"] in ("src_0.wav", "src_1.wav")
        assert r["noise_chunk_id"].startswith("chunk_")
        assert 0.0 <= float(r["snr_db"]) <= 30.0

    # row0: exactly one ambient row, and it is the attempt-2 (re-draw) sibling
    row0_ambient = [r for r in ambient if "__clip_0000__amb" in r["filename"]]
    assert len(row0_ambient) == 1
    assert "__amb2-" in row0_ambient[0]["filename"]
    # a well-behaved row: two siblings, distinct chunks
    row1_ambient = [r for r in ambient if "__clip_0001__amb" in r["filename"]]
    assert len(row1_ambient) == 2
    assert {r["filename"].split("__amb")[1].split("-")[0] for r in row1_ambient} == {"1", "2"}
    chunks = {r["noise_chunk_id"] for r in row1_ambient}
    assert len(chunks) == 2

    # ineligible rows never got ambient siblings
    for bad in ("clip_90.wav", "clip_93_noisy.wav", "clip_91.wav", "clip_92.wav"):
        assert not [r for r in ambient if f"__{bad}__amb" in r["filename"]]

    # identifiers unique across the whole derived manifest. (Raw `filename`
    # alone is NOT unique in real base manifests -- the same source clip can
    # appear in several subsets; the loader keys on `path` and on the
    # (group_id, filename) pair, and the ambient rows must be collision-free
    # with the base names they sit next to.)
    pairs = [(r["group_id"], r["filename"]) for r in final_rows]
    assert len(pairs) == len(set(pairs))
    paths = [r["path"] for r in final_rows]
    assert len(paths) == len(set(paths))
    base_names = {r["filename"] for r in final_rows if not r["source_dataset"].endswith("_ambient")}
    amb_names = [r["filename"] for r in ambient]
    assert len(amb_names) == len(set(amb_names))
    assert not (set(amb_names) & base_names)

    summary = summary_out.read_text(encoding="utf-8")
    assert "Realized SNR histogram" in summary
    assert "pass rate" in summary
    assert "License" in summary


def test_vcm_finalize_rebases_base_paths_and_fails_on_missing(tmp_path):
    # Base rows whose paths are local to the base dir (vcm_balanced's
    # audio_noisy/ rows in the fil50 base) must be re-expressed relative to
    # the derived manifest's dir, and any row that still does not resolve
    # must fail finalize loudly instead of crashing the DataLoader mid-epoch.
    corpus = _make_corpus(tmp_path / "corpus")
    root = tmp_path / "vcmx"
    rows = [
        _vcm_row(0),  # local path: audio/clip_0000.wav
        _vcm_row(1, path="../upstream/clip_0001.wav", source_relpath="../upstream/clip_0001.wav"),
        _vcm_row(2, transcript=""),  # ineligible -> never mixed, no wav written
    ]
    _sine_wav(root / "audio/clip_0000.wav")
    _sine_wav(tmp_path / "upstream/clip_0001.wav")
    manifest = root / "manifest.csv"
    _write_manifest(manifest, VCM_FIELDS, rows)

    out_root = root / "out"
    stage_mix(model="vcm", manifest_path=manifest, corpus_root=corpus, out_root=out_root, p_mix=1.0, seed=0)
    staging = out_root / ".ambient.staging"
    reports = tmp_path / "reports"
    _write_v1_pair_report(reports, "vcm", staging / "qa" / "shim" / "vcm", [])

    kwargs = dict(
        model="vcm",
        manifest_path=manifest,
        staging=staging,
        reports_dir=reports,
        negatives_decisions=None,
        audio_dir=out_root / "final" / "audio",
        manifest_out=out_root / "final" / "manifest.csv",
        summary_out=out_root / "final" / "summary.md",
        seed=0,
        p_mix=1.0,
        snr_min_db=0.0,
        snr_max_db=30.0,
    )
    # row2's base file is missing -> finalize fails loudly, writes nothing
    with pytest.raises(AmbientNoiseError, match="do not resolve"):
        stage_finalize(**kwargs)
    assert not kwargs["manifest_out"].exists()

    _sine_wav(root / "audio/clip_0002.wav")  # now everything resolves
    stage_finalize(**kwargs)

    with kwargs["manifest_out"].open(newline="", encoding="utf-8") as f:
        by_name = {r["filename"]: r["path"] for r in csv.DictReader(f)}
    # local base path rebased relative to the derived manifest's dir
    assert by_name["clip_0000.wav"] == "../../audio/clip_0000.wav"
    # path that already escaped the base dir rebased consistently
    assert by_name["clip_0001.wav"] == "../../../upstream/clip_0001.wav"
    # ambient rows stay local to the derived dir
    amb_paths = [p for name, p in by_name.items() if "__amb" in name]
    assert amb_paths and all(p.startswith("audio/") for p in amb_paths)


# ---------------------------------------------------------------------------
# wakeword: v1-pair positives + free-decode trigger gate
# ---------------------------------------------------------------------------


def _sesame_row(i: int, label: str, source_dataset: str, split: str = "train") -> dict:
    return {
        "filename": f"row_{i:04d}.wav",
        "path": f"audio/row_{i:04d}.wav",
        "label": label,
        "duration": "1.000000",
        "sample_rate": "16000",
        "resampled": "False",
        "source_dataset": source_dataset,
        "source_relpath": f"audio/row_{i:04d}.wav",
        "group_id": f"s{i:03d}",
        "split": split,
        "ref_voice": "ref_v" if label == "_wakeword_" else "",
        "noise_source_file": "",
        "snr_db": "",
        "speech_start_s": "",
        "speech_end_s": "",
    }


def test_wakeword_mix_finalize_trigger_gate(tmp_path):
    corpus = _make_corpus(tmp_path / "corpus")
    root = tmp_path / "sesame"
    rows = (
        [_sesame_row(i, "_wakeword_", "cosyvoice_conversion") for i in range(6)]
        + [_sesame_row(10, "_unknown_", "common_voice_negative"), _sesame_row(11, "_unknown_", "common_voice_negative")]
        + [_sesame_row(20, "_silence_", "synthetic_noise")]
        + [_sesame_row(30, "_unknown_", "common_voice_negative_noisy")]  # ineligible: existing _noisy
    )
    for row in rows:
        _sine_wav(root / row["path"])
    manifest = root / "manifest.csv"
    _write_manifest(manifest, SESAME_FIELDS, rows)

    stage_mix(model="wakeword", manifest_path=manifest, corpus_root=corpus, out_root=root, p_mix=1.0, seed=0)
    staging = root / ".ambient.staging"
    plan_rows = load_qa_plan(staging)
    # eligible: 6 positives + 2 negatives = 8 rows x 2 attempts
    assert len(plan_rows) == 16
    n_v1 = sum(1 for r in plan_rows if r["gate_mode"] == "v1_pair")
    n_free = sum(1 for r in plan_rows if r["gate_mode"] == "free_decode")
    assert (n_v1, n_free) == (12, 4)

    # all positives' shim names carry the expected text "Sesame"
    shim_dir = staging / "qa" / "shim" / "wakeword_sesame"
    assert {p.name.split(" - ")[0] for p in shim_dir.iterdir()} == {"Sesame"}

    # positives: report flags nothing -> all 12 pass
    reports_dir = tmp_path / "reports"
    _write_v1_pair_report(reports_dir, "wakeword_sesame", shim_dir, [])

    # negatives: row10 decodes fine (pass), row11 decodes as the wake word (fail, both attempts)
    decisions = tmp_path / "negatives.csv"
    with decisions.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["filename", "decoded_text"])
        writer.writeheader()
        writer.writerow({"filename": "s010__row_0010-a1.wav", "decoded_text": "turn on the light"})
        writer.writerow({"filename": "s010__row_0010-a2.wav", "decoded_text": "turn on the light"})
        writer.writerow({"filename": "s011__row_0011-a1.wav", "decoded_text": "Sesame."})
        writer.writerow({"filename": "s011__row_0011-a2.wav", "decoded_text": "can you sesame the lights?"})

    manifest_out = root / "manifest.ambient.csv"
    audio_dir = root / "ambient" / "audio"
    summary_out = root / "summary.md"
    stage_finalize(
        model="wakeword",
        manifest_path=manifest,
        staging=staging,
        reports_dir=reports_dir,
        negatives_decisions=decisions,
        audio_dir=audio_dir,
        manifest_out=manifest_out,
        summary_out=summary_out,
        seed=0,
        p_mix=1.0,
        snr_min_db=0.0,
        snr_max_db=30.0,
    )

    with manifest_out.open(newline="", encoding="utf-8") as f:
        final_rows = list(csv.DictReader(f))
    ambient = [r for r in final_rows if r["source_dataset"].endswith("_ambient")]
    # 12 positives + 2 (row10) + 0 (row11) = 14
    assert len(ambient) == 14
    assert len([r for r in ambient if "__row_0010__amb" in r["filename"]]) == 2
    assert not [r for r in ambient if "__row_0011__amb" in r["filename"]]
    assert not [r for r in ambient if "__row_0020__amb" in r["filename"]]  # _silence_ never mixed
    assert not [r for r in ambient if "__row_0030__amb" in r["filename"]]  # existing _noisy never re-mixed
    pos = [r for r in ambient if r["label"] == "_wakeword_"]
    assert all(r["source_dataset"] == "cosyvoice_conversion_ambient" for r in pos)
    neg = [r for r in ambient if r["label"] == "_unknown_"]
    assert all(r["source_dataset"] == "common_voice_negative_ambient" for r in neg)
    assert all(r["path"].startswith("ambient/audio/") for r in ambient)
    # base rows gained the new noise_chunk_id column, empty for them
    base_ambient = [r for r in final_rows if not r["source_dataset"].endswith("_ambient")]
    assert all(r["noise_chunk_id"] == "" for r in base_ambient)


def test_wakeword_missing_negative_decode_never_passes(tmp_path):
    corpus = _make_corpus(tmp_path / "corpus")
    root = tmp_path / "sesame"
    rows = [_sesame_row(10, "_unknown_", "common_voice_negative")]
    _sine_wav(root / rows[0]["path"])
    manifest = root / "manifest.csv"
    _write_manifest(manifest, SESAME_FIELDS, rows)
    stage_mix(model="wakeword", manifest_path=manifest, corpus_root=corpus, out_root=root, p_mix=1.0, seed=0)
    staging = root / ".ambient.staging"
    plan_rows = load_qa_plan(staging)

    # empty decisions file -> no decode for either attempt -> both fail
    decisions = tmp_path / "negatives.csv"
    decisions.write_text("filename,decoded_text\n", encoding="utf-8")
    passed = negative_passed(decisions, plan_rows)
    assert all(v is False for v in passed.values())


# ---------------------------------------------------------------------------
# gate parsers
# ---------------------------------------------------------------------------


def test_v1_pair_passed_missing_report_fails(tmp_path):
    corpus = _make_corpus(tmp_path / "corpus")
    root = tmp_path / "vcmx"
    manifest, _ = _build_vcm_root(root, n_eligible=2)
    stage_mix(model="vcm", manifest_path=manifest, corpus_root=corpus, out_root=root, p_mix=1.0, seed=0)
    plan_rows = load_qa_plan(root / ".ambient.staging")
    # reports dir exists but holds no report for unit "vcm"
    reports = tmp_path / "reports"
    reports.mkdir()
    passed = v1_pair_passed(reports, plan_rows)
    assert len(passed) == 4
    assert all(v is False for v in passed.values())


def test_decodes_as_wakeword_variants():
    assert decodes_as_wakeword("Sesame.")
    assert decodes_as_wakeword("can you sesame the lights?")
    assert decodes_as_wakeword("SESAME")
    assert decodes_as_wakeword("sesame's")
    assert not decodes_as_wakeword("turn on the light")
    assert not decodes_as_wakeword("")


# ---------------------------------------------------------------------------
# plan-stage failures
# ---------------------------------------------------------------------------


def test_vcm_mix_fails_loudly_on_unsafe_transcript(tmp_path):
    corpus = _make_corpus(tmp_path / "corpus")
    root = tmp_path / "vcmx"
    manifest, _ = _build_vcm_root(root, n_eligible=2)
    # rewrite one transcript to be v1-pair-unsafe
    with manifest.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    rows[0]["transcript"] = "stop - the light"
    _write_manifest(manifest, VCM_FIELDS, rows)
    with pytest.raises(AmbientNoiseError, match="stop - the light"):
        stage_mix(model="vcm", manifest_path=manifest, corpus_root=corpus, out_root=root, p_mix=1.0, seed=0)
    assert not (root / ".ambient.staging").exists()


def test_stage_mix_dry_run_writes_nothing(tmp_path):
    corpus = _make_corpus(tmp_path / "corpus")
    root = tmp_path / "vcmx"
    manifest, _ = _build_vcm_root(root, n_eligible=4)
    result = stage_mix(
        model="vcm",
        manifest_path=manifest,
        corpus_root=corpus,
        out_root=root,
        p_mix=1.0,
        seed=0,
        dry_run=True,
    )
    assert result is None
    assert not (root / ".ambient.staging").exists()
