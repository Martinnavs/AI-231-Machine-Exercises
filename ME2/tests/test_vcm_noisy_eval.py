"""Fast, CPU-only, checkpoint-free tests for `vcm.noisy_eval` (the
fixed-seed noisy/reverb eval gate pass).

Covers the determinism contract (same manifest+seed => identical
perturbation specs and identical decoded results, across calls and across
full `main()` runs), the per-row draw mechanics (ranges, SNR bounds,
noise-pool requirement), `apply_perturbation`'s length/finite invariants,
and `main()`'s report wiring (flag present => `noisy_eval` block with the
clean-chosen threshold; flag absent => key absent, report unchanged).

Uses the shared `vcm_stub_model_factory`/`vcm_fake_manifest_factory`
fixtures throughout; no real checkpoint or GPU is ever touched.
"""

from __future__ import annotations

import dataclasses
import json

import pytest
import torch

from me2_voicegen.common.augment import build_rir_pool
from me2_voicegen.vcm import alphabet
from me2_voicegen.vcm.dataset import VCMDataset
from me2_voicegen.vcm.evaluate import main
from me2_voicegen.vcm.noisy_eval import (
    _seed_for,
    apply_perturbation,
    build_perturbations,
    build_rir_pool_for_seed,
    decode_split_noisy,
)
from me2_voicegen.vcm.optiona.grammar import TOY_GRAMMAR

CALL_IDS = alphabet.encode("call")


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


def _manifest_with_noise(vcm_fake_manifest_factory, split="val"):
    """One target + one babble + two background_noise rows in `split` --
    enough for a draw sequence and a non-empty noise pool."""
    return vcm_fake_manifest_factory(
        [
            {"bucket": "target_commands", "source_dataset": "optionb", "label": "CALL",
             "split": split, "transcript": "call me"},
            {"bucket": "babble", "source_dataset": "common_voice_negative", "label": "unknown",
             "split": split, "transcript": "some other speech"},
            {"bucket": "silence", "source_dataset": "background_noise", "label": "unknown",
             "split": split},
            {"bucket": "silence", "source_dataset": "background_noise", "label": "unknown",
             "split": split},
        ]
    )


def test_build_perturbations_is_deterministic_per_seed(vcm_fake_manifest_factory):
    manifest_path = _manifest_with_noise(vcm_fake_manifest_factory)
    dataset = VCMDataset(manifest_path, split="val")
    pool = build_rir_pool(3, sample_rate=16000, seed=1)

    first = build_perturbations(dataset, pool, seed=0)
    second = build_perturbations(dataset, pool, seed=0)
    assert first == second
    assert len(first) == len(dataset)

    for rir_idx, noise_idx, snr_db in first:
        assert 0 <= rir_idx < len(pool)
        assert 0 <= noise_idx < 2  # two background_noise rows in the split
        assert 5.0 <= snr_db <= 25.0

    other_seed = build_perturbations(dataset, pool, seed=1)
    assert other_seed != first


def test_build_perturbations_raises_without_noise_pool(vcm_fake_manifest_factory):
    manifest_path = vcm_fake_manifest_factory(
        [
            {"bucket": "target_commands", "source_dataset": "optionb", "label": "CALL",
             "split": "test", "transcript": "call me"},
        ]
    )
    dataset = VCMDataset(manifest_path, split="test")
    pool = build_rir_pool(2, sample_rate=16000, seed=1)
    with pytest.raises(ValueError, match="no `background_noise` rows"):
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
# decode_split_noisy: end-to-end determinism with the stub model
# ---------------------------------------------------------------------------


def _decode_args(vcm_stub_model_factory, feature_extractor):
    model = vcm_stub_model_factory(forced_ids=CALL_IDS)
    return model, feature_extractor


@pytest.fixture
def feature_extractor():
    from me2_voicegen.common.features import LogMelFeatureExtractor

    return LogMelFeatureExtractor()


def test_decode_split_noisy_is_deterministic(vcm_fake_manifest_factory, vcm_stub_model_factory, feature_extractor):
    manifest_path = _manifest_with_noise(vcm_fake_manifest_factory)
    dataset = VCMDataset(manifest_path, split="val")
    model, fx = _decode_args(vcm_stub_model_factory, feature_extractor)
    pool = build_rir_pool(2, sample_rate=16000, seed=1)

    first = decode_split_noisy(model, fx, dataset, TOY_GRAMMAR, 5, "cpu", pool, seed=0)
    second = decode_split_noisy(model, fx, dataset, TOY_GRAMMAR, 5, "cpu", pool, seed=0)
    assert [dataclasses.asdict(r) for r in first] == [dataclasses.asdict(r) for r in second]
    # Cross-SEED difference is asserted at the draw level
    # (test_build_perturbations_is_deterministic_per_seed) rather than here:
    # the shared stub model's forward is input-INdependent (it emits the
    # same forced ids regardless of the waveform), so with this stand-in a
    # different seed changes the perturbation but not the decode. The real
    # model is input-dependent, and test_apply_perturbation proves the
    # perturbation really does change the signal, so the two compose.


# ---------------------------------------------------------------------------
# main(): --noisy-eval-seed report wiring
# ---------------------------------------------------------------------------


def _build_main_manifest(vcm_fake_manifest_factory):
    specs = []
    for split in ("val", "test"):
        specs.append(
            {
                "bucket": "target_commands",
                "source_dataset": "optionb",
                "label": "CALL",
                "split": split,
                "transcript": "call me",
            }
        )
        specs.append(
            {
                "bucket": "babble",
                "source_dataset": "common_voice_negative",
                "label": "unknown",
                "split": split,
                "transcript": "some other speech",
            }
        )
        specs.append(
            {
                "bucket": "silence",
                "source_dataset": "background_noise",
                "label": "unknown",
                "split": split,
            }
        )
    return vcm_fake_manifest_factory(specs)


def _patch_load_checkpoint(monkeypatch, vcm_stub_model_factory):
    import me2_voicegen.vcm.evaluate as evaluate_mod

    model = vcm_stub_model_factory(forced_ids=CALL_IDS)
    checkpoint_meta = {"preset": "default", "epoch": 1, "val_loss": 1.0, "license": "CC-BY-NC-SA-4.0"}
    monkeypatch.setattr(evaluate_mod, "load_checkpoint", lambda path, device: (model, checkpoint_meta))


def _run_main(argv, tmp_path, manifest_path):
    out_dir = tmp_path / f"eval_out_{len(list(tmp_path.iterdir()))}"
    main(
        ["--manifest", str(manifest_path), "--checkpoint", "unused.pt",
         "--out-dir", str(out_dir),
         "--slot-eval-manifest", str(tmp_path / "does_not_exist" / "manifest.csv"),
         "--noisy-eval-rir-pool-size", "2",
         *argv]
    )
    report = json.loads((out_dir / "metadata" / "eval_report.json").read_text())
    return report, out_dir


def test_main_without_noisy_flag_has_no_noisy_eval_key(
    tmp_path, monkeypatch, vcm_fake_manifest_factory, vcm_stub_model_factory
):
    manifest_path = _build_main_manifest(vcm_fake_manifest_factory)
    _patch_load_checkpoint(monkeypatch, vcm_stub_model_factory)

    report, _out_dir = _run_main([], tmp_path, manifest_path)
    assert "noisy_eval" not in report


def test_main_with_noisy_seed_reports_section_at_clean_chosen_threshold(
    tmp_path, monkeypatch, vcm_fake_manifest_factory, vcm_stub_model_factory
):
    manifest_path = _build_main_manifest(vcm_fake_manifest_factory)
    _patch_load_checkpoint(monkeypatch, vcm_stub_model_factory)

    report, out_dir = _run_main(["--noisy-eval-seed", "0"], tmp_path, manifest_path)

    noisy = report["noisy_eval"]
    assert noisy["seed"] == 0
    assert noisy["rir_pool_size"] == 2
    assert noisy["snr_range_db"] == [5.0, 25.0]
    assert "rt60_range_s" in noisy and "perturbation" in noisy and "threshold_rule" in noisy

    assert len(noisy["sections"]) == len(report["grammar_sections"])
    for clean_section, noisy_section in zip(report["grammar_sections"], noisy["sections"]):
        assert noisy_section["grammar"] == clean_section["grammar"]
        # The gate scores at the CLEAN-val chosen threshold, never a new one.
        assert noisy_section["threshold"] == clean_section["chosen_operating_threshold"]
        for block_key in ("val_split", "test_split"):
            block = noisy_section[block_key]
            assert block["n_target_commands"] == 1
            assert "accept_rate" in block and "exact_accuracy" in block
            assert "false_accept_rate_babble" in block
            assert "false_accept_rate_silence" in block

    md = (out_dir / "metadata" / "eval_report.md").read_text()
    assert "Noisy/reverb eval gate (fixed seed 0)" in md
    assert "Clean vs. noisy (test split, same threshold)" in md


def test_main_noisy_eval_is_deterministic_across_runs(
    tmp_path, monkeypatch, vcm_fake_manifest_factory, vcm_stub_model_factory
):
    manifest_path = _build_main_manifest(vcm_fake_manifest_factory)
    _patch_load_checkpoint(monkeypatch, vcm_stub_model_factory)

    report_a, _ = _run_main(["--noisy-eval-seed", "42"], tmp_path, manifest_path)
    report_b, _ = _run_main(["--noisy-eval-seed", "42"], tmp_path, manifest_path)
    # Cross-"checkpoint" comparability at the report level: same manifest +
    # same seed => bit-identical noisy_eval block.
    assert report_a["noisy_eval"] == report_b["noisy_eval"]

    report_c, _ = _run_main(["--noisy-eval-seed", "43"], tmp_path, manifest_path)
    assert report_c["noisy_eval"]["seed"] == 43
    # (With the input-independent stub model the per-grammar metric blocks
    # cannot differ across seeds here; the seed-dependence of the
    # perturbation itself is proven at the draw level in
    # test_build_perturbations_is_deterministic_per_seed.)


def test_build_report_without_noisy_eval_has_no_noisy_key():
    """Clean-only runs (no --noisy-eval-seed) must serialize exactly as before."""
    from pathlib import Path

    from me2_voicegen.vcm.evaluate import build_report

    args = (Path("c.pt"), {"a": 1}, Path("m.csv"), "cpu", 50, [], None, "skipped")
    assert "noisy_eval" not in build_report(*args)
    assert build_report(*args, noisy_eval={"seed": 0})["noisy_eval"] == {"seed": 0}
