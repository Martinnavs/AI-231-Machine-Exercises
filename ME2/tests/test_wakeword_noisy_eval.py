"""Fast, CPU-only, checkpoint-free tests for `wakeword.noisy_eval` (the
fixed-seed noisy/reverb eval gate pass;
.scratch/ambient-reverb-cooccurrence/tickets/00-RECAP.md T4). Mirrors
`me2-iteration3/ME2/tests/test_vcm_noisy_eval.py`'s structure, adapted to
the DS-CNN's argmax scoring path: covers the determinism contract (same
manifest+seed => identical perturbation specs and identical scored
results, across calls and across full `main()` runs), the per-row draw
mechanics (ranges, SNR bounds, noise-pool requirement),
`apply_perturbation`'s length/finite invariants, and `main()`'s report
wiring (seed + pool size + per-split blocks in the JSON, gate header in
the markdown). No real checkpoint or GPU is ever touched.
"""

from __future__ import annotations

import csv
import json

import pytest
import torch
import torchaudio

from me2_voicegen.common.augment import build_rir_pool
from me2_voicegen.wakeword.model import DSCNN, DSCNNConfig, LABELS
from me2_voicegen.wakeword.noisy_eval import (
    _seed_for,
    apply_perturbation,
    build_perturbations,
    build_rir_pool_for_seed,
    score_split_noisy,
)
from me2_voicegen.wakeword.dataset import WakewordDataset

FIELDS = [
    "filename",
    "path",
    "label",
    "duration",
    "sample_rate",
    "resampled",
    "source_dataset",
    "source_relpath",
    "group_id",
    "split",
    "ref_voice",
    "speech_start_s",
    "speech_end_s",
]


# ---------------------------------------------------------------------------
# fixtures: tiny manifest (precomputed spans => no live VAD) + noise root
# ---------------------------------------------------------------------------


def _write_wav(path, duration_s: float = 1.5, freq_hz: float = 440.0, silence: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    n = int(duration_s * 16000)
    if silence:
        wave = torch.zeros(1, n)
    else:
        t = torch.arange(n, dtype=torch.float32) / 16000
        wave = (0.1 * torch.sin(2 * 3.14159 * freq_hz * t)).unsqueeze(0)
    torchaudio.save(str(path), wave, 16000)


def _manifest_with_speech(root, split: str, n: int = 2) -> list[dict]:
    """`n` _wakeword_ + one _unknown_ + one _silence_ row in `split`, all
    with precomputed spans (the dataset never falls back to live VAD)."""
    rows = []
    for i in range(n):
        rel = f"positives_real/audio/wk_{split}_{i}.wav"
        _write_wav(root / rel)
        rows.append({
            "filename": f"wk_{split}_{i}.wav", "path": rel, "label": "_wakeword_",
            "duration": "1.500000", "sample_rate": "16000", "resampled": "False",
            "source_dataset": "positives_real", "source_relpath": f"wk_{split}_{i}.wav",
            "group_id": f"g{split}{i}", "split": split, "ref_voice": "",
            "speech_start_s": "0.200000", "speech_end_s": "1.300000",
        })
    rel_u = f"adversaries/audio/adv_{split}.wav"
    _write_wav(root / rel_u, freq_hz=880.0)
    rows.append({
        "filename": f"adv_{split}.wav", "path": rel_u, "label": "_unknown_",
        "duration": "1.500000", "sample_rate": "16000", "resampled": "False",
        "source_dataset": "adversaries", "source_relpath": f"adv_{split}.wav",
        "group_id": f"gu{split}", "split": split, "ref_voice": "",
        "speech_start_s": "0.200000", "speech_end_s": "1.300000",
    })
    rel_s = f"silence_synthetic/audio/sil_{split}.wav"
    _write_wav(root / rel_s, silence=True)
    rows.append({
        "filename": f"sil_{split}.wav", "path": rel_s, "label": "_silence_",
        "duration": "1.500000", "sample_rate": "16000", "resampled": "False",
        "source_dataset": "silence_synthetic", "source_relpath": f"sil_{split}.wav",
        "group_id": f"gs{split}", "split": split, "ref_voice": "",
        "speech_start_s": "", "speech_end_s": "",
    })
    return rows


def _build_manifest(tmp_path) -> tuple:
    root = tmp_path / "wakeword"
    rows = _manifest_with_speech(root, "val") + _manifest_with_speech(root, "test")
    manifest = root / "manifest.csv"
    with manifest.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)

    noise_root = tmp_path / "noise"
    for i in range(2):
        _write_wav(noise_root / f"noise_{i}.wav", duration_s=1.0, freq_hz=120.0)
    return manifest, noise_root, root


def _tiny_model() -> DSCNN:
    torch.manual_seed(0)
    return DSCNN(DSCNNConfig(n_blocks=1, channels=8, kernel_sizes=[5], prologue_channels=8))


# ---------------------------------------------------------------------------
# seed derivation
# ---------------------------------------------------------------------------


def test_seed_for_is_deterministic_and_distinct():
    assert _seed_for(0, "val") == _seed_for(0, "val")
    assert _seed_for(0, "val") != _seed_for(0, "test")
    assert _seed_for(0, "val") != _seed_for(1, "val")
    assert _seed_for(0, "rir_pool") == _seed_for(0, "rir_pool")
    assert 0 <= _seed_for(12345, "val") < 2**32


def test_build_rir_pool_for_seed_is_deterministic():
    pool_a = build_rir_pool_for_seed(7, pool_size=4)
    pool_b = build_rir_pool_for_seed(7, pool_size=4)
    assert len(pool_a) == 4
    for rir_a, rir_b in zip(pool_a, pool_b):
        assert torch.equal(rir_a, rir_b)
    pool_c = build_rir_pool_for_seed(8, pool_size=4)
    assert any(not torch.equal(a, c) for a, c in zip(pool_a, pool_c))


# ---------------------------------------------------------------------------
# build_perturbations: the determinism contract
# ---------------------------------------------------------------------------


def test_build_perturbations_is_deterministic_per_seed(tmp_path):
    manifest, noise_root, _ = _build_manifest(tmp_path)
    dataset = WakewordDataset(manifest, split="val", noise_root=noise_root)
    pool = build_rir_pool(3, sample_rate=16000, seed=1)

    first = build_perturbations(dataset, pool, seed=0)
    second = build_perturbations(dataset, pool, seed=0)
    assert first == second
    assert len(first) == len(dataset)

    for rir_idx, noise_idx, snr_db in first:
        assert 0 <= rir_idx < len(pool)
        assert 0 <= noise_idx < 2  # two noise wavs in the root
        assert 5.0 <= snr_db <= 25.0

    other_seed = build_perturbations(dataset, pool, seed=1)
    assert other_seed != first


def test_build_perturbations_raises_without_noise_pool(tmp_path):
    manifest, noise_root, _ = _build_manifest(tmp_path)
    dataset = WakewordDataset(manifest, split="test", noise_root=None)
    pool = build_rir_pool(2, sample_rate=16000, seed=1)
    with pytest.raises(ValueError, match="needs a noise pool"):
        build_perturbations(dataset, pool, seed=0)


def test_apply_perturbation_preserves_length_and_changes_signal():
    torch.manual_seed(0)
    sr = 16000
    t = torch.arange(sr, dtype=torch.float32) / sr
    waveform = (0.1 * torch.sin(2 * 3.14159 * 440.0 * t)).unsqueeze(0)
    noise = torch.randn(1, sr)
    rir = build_rir_pool(1, sample_rate=sr, seed=3)[0]

    out = apply_perturbation(waveform[0], rir, noise[0], snr_db=10.0)
    assert out.shape == waveform[0].shape
    assert torch.isfinite(out).all()
    assert not torch.equal(out, waveform[0])


# ---------------------------------------------------------------------------
# score_split_noisy: end-to-end determinism with a tiny real DSCNN
# ---------------------------------------------------------------------------


def test_score_split_noisy_is_deterministic_and_scores_all_classes(tmp_path):
    from me2_voicegen.common.features import LogMelFeatureExtractor

    manifest, noise_root, _ = _build_manifest(tmp_path)
    dataset = WakewordDataset(manifest, split="val", noise_root=noise_root)
    model = _tiny_model()
    pool = build_rir_pool(2, sample_rate=16000, seed=1)

    first = score_split_noisy(model, LogMelFeatureExtractor(), dataset, pool, seed=0, device="cpu")
    second = score_split_noisy(model, LogMelFeatureExtractor(), dataset, pool, seed=0, device="cpu")
    assert first == second
    assert set(first["per_class"].keys()) == set(LABELS)
    assert first["n_rows"] == len(dataset)
    assert set(first["accent_recall"].keys()) == {"filipino", "non_filipino"}


# ---------------------------------------------------------------------------
# main(): report wiring + cross-run determinism
# ---------------------------------------------------------------------------


def _run_main(argv, tmp_path, manifest, noise_root, monkeypatch):
    import me2_voicegen.wakeword.noisy_eval as noisy_mod

    monkeypatch.setattr(
        noisy_mod, "load_checkpoint", lambda path, device: (_tiny_model(), {"preset": "default", "epoch": 1})
    )
    out_dir = tmp_path / f"out_{len(list(tmp_path.glob('out_*')))}"
    noisy_mod.main(
        ["--manifest", str(manifest), "--checkpoint", "unused.pt", "--out-dir", str(out_dir),
         "--device", "cpu", "--noise-root", str(noise_root),
         "--noisy-eval-rir-pool-size", "2", *argv]
    )
    report = json.loads((out_dir / "metadata" / "noisy_eval_report.json").read_text())
    return report, out_dir


def test_main_reports_gate_block_and_is_deterministic_across_runs(tmp_path, monkeypatch):
    manifest, noise_root, _ = _build_manifest(tmp_path)

    report_a, out_dir = _run_main(["--noisy-eval-seed", "42"], tmp_path, manifest, noise_root, monkeypatch)
    report_b, _ = _run_main(["--noisy-eval-seed", "42"], tmp_path, manifest, noise_root, monkeypatch)
    # Cross-"checkpoint" comparability at the report level: same manifest +
    # same seed => identical noisy blocks.
    for key in ("seed", "val_split", "test_split", "rir_pool_size", "snr_range_db", "perturbation", "threshold_rule"):
        assert report_a[key] == report_b[key], key
    assert report_a["seed"] == 42
    assert report_a["rir_pool_size"] == 2
    assert report_a["snr_range_db"] == [5.0, 25.0]
    assert "rt60_range_s" in report_a
    for split in ("val_split", "test_split"):
        block = report_a[split]
        assert set(block["per_class"].keys()) == set(LABELS)
        assert block["n_rows"] == 4  # 2 _wakeword_ + 1 _unknown_ + 1 _silence_
        assert set(block["accent_recall"].keys()) == {"filipino", "non_filipino"}

    report_c, _ = _run_main(["--noisy-eval-seed", "43"], tmp_path, manifest, noise_root, monkeypatch)
    assert report_c["seed"] == 43
    # (Seed-dependence of the perturbation itself is proven at the draw
    # level in test_build_perturbations_is_deterministic_per_seed; with a
    # fixed tiny model the per-class blocks may legitimately match.)

    md = (out_dir / "metadata" / "noisy_eval_report.md").read_text()
    assert "Noisy/reverb eval gate (fixed seed 42)" in md
    assert "val_split" in md and "test_split" in md
    assert "Threshold rule" in md
