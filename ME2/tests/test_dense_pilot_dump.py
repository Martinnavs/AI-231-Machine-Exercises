"""T1 tests for scripts/dense_pilot_dump.py (dense-phonetic-scoring pilot)."""

import importlib.util
import json
import sys
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pytest
import torch

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "dense_pilot_dump.py"
spec = importlib.util.spec_from_file_location("dense_pilot_dump", SCRIPT)
dump = importlib.util.module_from_spec(spec)
sys.modules["dense_pilot_dump"] = dump
spec.loader.exec_module(dump)

FIXTURE_DIR = Path(__file__).resolve().parent / "fixtures" / "dense_pilot"


@pytest.fixture
def tiny_checkpoint(tmp_path):
    from me2_voicegen.vcm.model import MatchboxNetCTC, MatchboxNetConfig

    torch.manual_seed(0)
    cfg = MatchboxNetConfig(n_blocks=1, channels=16, kernel_sizes=[5], prologue_channels=16,
                            epilogue_channels=16)
    model = MatchboxNetCTC(cfg)
    path = tmp_path / "ckpt.pt"
    torch.save({"config": asdict(cfg), "model_state_dict": model.state_dict()}, path)
    return path


@pytest.fixture
def tiny_manifest(vcm_fake_manifest_factory):
    specs = [
        {"bucket": "target_commands", "source_dataset": "optionb", "label": "STOP",
         "split": "val", "transcript": "stop", "duration_s": 1.0},
        {"bucket": "babble", "source_dataset": "common_voice_negative", "label": "unknown",
         "split": "val", "duration_s": 1.0},
        {"bucket": "silence", "source_dataset": "background_noise", "split": "val",
         "duration_s": 1.0},
        {"bucket": "background_noise", "source_dataset": "background_noise", "split": "val",
         "duration_s": 1.0},
    ]
    return vcm_fake_manifest_factory(specs)


def test_dump_roundtrip_schema_and_logp_matches_pipeline(tiny_checkpoint, tiny_manifest, tmp_path):
    from me2_voicegen.common.features import LogMelFeatureExtractor
    from me2_voicegen.vcm.dataset import VCMDataset
    from me2_voicegen.vcm.pipeline import load_checkpoint, logp_for_waveform

    out = tmp_path / "dump"
    meta = dump.dump_split(tiny_checkpoint, tiny_manifest, "val", "clean", out, workers=2)
    assert meta["n_rows"] == 3  # background_noise row excluded
    logps, rows = dump.load_dump(out, "val", "clean")
    assert [r["bucket"] for r in rows] == ["target_commands", "babble", "silence"]
    assert list(rows[0]) == dump.ROWS_FIELDS
    assert all(lp.dtype == np.float32 and lp.shape[1] == 29 for lp in logps)
    assert json.loads((out / "logits" / "val_clean.meta.json").read_text())["condition"] == "clean"

    model, _ = load_checkpoint(tiny_checkpoint, device="cpu")
    ds = VCMDataset(tiny_manifest, split="val", augmenter=None)
    expect = logp_for_waveform(model, LogMelFeatureExtractor(), ds[0].waveform, device="cpu")
    # Not bitwise: the worker runs 1 torch thread, this process many (~1e-7 drift).
    np.testing.assert_allclose(logps[0], expect, atol=1e-6, rtol=0)


def test_unknown_condition_rejected(tiny_checkpoint, tiny_manifest, tmp_path):
    with pytest.raises(ValueError):
        dump.dump_split(tiny_checkpoint, tiny_manifest, "val", "bogus", tmp_path, workers=1)


def test_noisy_condition_without_noisy_eval_fails_with_hint(
    tiny_checkpoint, tiny_manifest, tmp_path, monkeypatch
):
    def boom():
        raise SystemExit(dump.NOISY_ENV_HINT)

    monkeypatch.setattr(dump, "_import_noisy_eval", boom)
    with pytest.raises(SystemExit, match="PYTHONPATH"):
        dump.dump_split(tiny_checkpoint, tiny_manifest, "val", "noisy_s0", tmp_path, workers=1)


def test_select_fixture_rows_picks_required_coverage():
    rows = (
        [{"bucket": "target_commands", "label": "STOP"}] * 3
        + [{"bucket": "target_commands", "label": "TIME"}] * 3
        + [{"bucket": "target_commands", "label": "ALARM"}] * 6
        + [{"bucket": "babble", "label": "unknown"}] * 6
        + [{"bucket": "silence", "label": "unknown"}] * 6
    )
    fa = {15, 21}  # one babble, one silence
    picks = dump.select_fixture_rows(rows, fa)
    assert picks == sorted(picks) and len(set(picks)) == len(picks)
    assert fa <= set(picks)
    labels = [rows[p]["label"] for p in picks if rows[p]["bucket"] == "target_commands"]
    assert labels.count("STOP") == 2 and labels.count("TIME") == 2
    assert sum(rows[p]["bucket"] == "babble" for p in picks) >= 4
    assert sum(rows[p]["bucket"] == "silence" for p in picks) >= 4


def test_committed_fixture_is_small_and_has_falsely_accepted_rows():
    from me2_voicegen.vcm.decoder import decode_utterance
    from me2_voicegen.vcm.optionb.grammar import OPTIONB_GRAMMAR

    size = sum(p.stat().st_size for p in FIXTURE_DIR.iterdir())
    assert size <= 1.5 * 1024 * 1024
    for cond in ("clean", "noisy_s0"):
        z = np.load(FIXTURE_DIR / f"val_{cond}.npz")
        n = len(z["offsets"]) - 1
        assert z["logp"].dtype == np.float32 and z["logp"].shape[1] == 29
        rows = list(__import__("csv").DictReader((FIXTURE_DIR / f"val_{cond}.rows.csv").open()))
        assert len(rows) == n
        fa = 0
        for k, r in enumerate(rows):
            if r["bucket"] == "target_commands":
                continue
            lp = z["logp"][z["offsets"][k]:z["offsets"][k + 1]]
            fa += not decode_utterance(lp, OPTIONB_GRAMMAR, -0.1).no_match
        assert fa >= 2
