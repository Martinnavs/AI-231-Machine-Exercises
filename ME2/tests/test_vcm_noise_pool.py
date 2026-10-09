"""Dataset-free training noise: synthetic pool, quiet-babble path, and the dataset's `noise_source='dataset'` mode."""

from __future__ import annotations

import json

import pytest
import torch

from me2_voicegen.common.augment import BABBLE_SNR_MAX_DB, BABBLE_SNR_MIN_DB, Augmenter
from me2_voicegen.vcm import train as train_mod
from me2_voicegen.vcm.dataset import VCMDataset
from me2_voicegen.vcm.noise_pool import synthetic_noise_pool


def test_synthetic_pool_is_deterministic_finite_and_seed_dependent():
    a = synthetic_noise_pool(n_clips=12, seconds=0.5, seed=0)
    b = synthetic_noise_pool(n_clips=12, seconds=0.5, seed=0)
    c = synthetic_noise_pool(n_clips=12, seconds=0.5, seed=1)
    assert all(torch.equal(x, y) for x, y in zip(a, b))
    assert not torch.equal(a[0], c[0])
    assert all(x.dtype == torch.float32 and x.shape == (8000,) and torch.isfinite(x).all() for x in a)


def test_synthetic_pool_colours_and_modulation():
    pool = synthetic_noise_pool(n_clips=6, seconds=2.0, seed=0)  # white, pink, brown, then the same three modulated

    def low_frac(x):  # share of spectral energy below 200 Hz
        p = torch.fft.rfft(x).abs() ** 2
        return float(p[: int(200 * len(x) / 16000)].sum() / p.sum())

    assert low_frac(pool[0]) < low_frac(pool[1]) < low_frac(pool[2])  # white < pink < brown
    env = lambda x: x.reshape(20, -1).abs().mean(1)  # 100 ms envelope
    assert env(pool[3]).std() > 1.5 * env(pool[0]).std()  # the modulated white clip really fluctuates


def _clean_and_babble():
    g = torch.Generator().manual_seed(3)
    return torch.randn(16000, generator=g), [torch.randn(12000, generator=g) for _ in range(3)]


def test_babble_is_off_by_default_and_changes_nothing():
    clean, babble = _clean_and_babble()
    a = Augmenter(seed=0).augment_waveform(clean.clone(), babble_pool=babble)
    b = Augmenter(seed=0).augment_waveform(clean.clone())
    assert torch.equal(a, b) and torch.equal(a, clean)


def test_babble_mixes_at_its_own_quiet_snr_range():
    clean, babble = _clean_and_babble()
    aug = Augmenter(p_babble=1.0, seed=0)
    for _ in range(25):
        out = aug.augment_waveform(clean.clone(), babble_pool=babble)
        assert out.shape == clean.shape
        snr = 10 * torch.log10(clean.pow(2).mean() / (out - clean).pow(2).mean())
        assert BABBLE_SNR_MIN_DB - 0.2 <= float(snr) <= BABBLE_SNR_MAX_DB + 0.2


@pytest.fixture
def noise_manifest(vcm_fake_manifest_factory):
    def spec(bucket, label, source="optionb", split="train", transcript="stop"):
        return {"bucket": bucket, "source_dataset": source, "label": label, "split": split, "transcript": transcript, "duration_s": 1.0}
    rows = [spec("target_commands", "STOP"), spec("target_commands", "STOP", split="val"),
            spec("babble", "unknown", transcript="hello there"), spec("babble", "unknown", transcript="how are you"),
            spec("silence", "silence", source="background_noise", transcript=""),
            spec("babble", "unknown", split="val", transcript="val babble")]
    return vcm_fake_manifest_factory(rows)


def test_dataset_noise_source_builds_pools_from_its_own_split_only(noise_manifest):
    ds = VCMDataset(noise_manifest, split="train", noise_source="dataset")
    assert len(ds._babble_pool()) == 2                      # train babble rows only (the val one is not leaked in)
    assert len(ds._noise_pool()) == 1 + 240                 # its noise-only row + the synthetic pool
    default = VCMDataset(noise_manifest, split="train")
    assert len(default._noise_pool()) == 1 and default._babble_pool() == []
    with pytest.raises(ValueError):
        VCMDataset(noise_manifest, split="train", noise_source="elsewhere")


def test_dataset_noise_source_augments_train_only_and_is_reproducible(noise_manifest):
    def item(split, seed):
        aug = Augmenter(p_noise=1.0, p_babble=1.0, seed=seed)
        return VCMDataset(noise_manifest, split=split, augmenter=aug, noise_source="dataset")[0].waveform
    plain = VCMDataset(noise_manifest, split="train")[0].waveform
    assert not torch.equal(item("train", 0), plain)         # noise + babble applied
    assert torch.equal(item("train", 0), item("train", 0))  # same seed, same signal
    assert torch.equal(item("val", 0), VCMDataset(noise_manifest, split="val")[0].waveform)  # never on val


def test_train_cli_noise_flags(noise_manifest, tmp_path):
    def run(out, *extra):
        train_mod.main(["--manifest", str(noise_manifest), "--out-dir", str(tmp_path / out), "--preset", "quartznet5x3",
                        "--max-epochs", "1", "--device", "cpu", "--num-workers", "0", "--batch-size", "2",
                        "--max-minutes", "5", *extra])
        return json.loads((tmp_path / out / "metadata" / "loss_history.json").read_text())
    hist = run("a", "--noise-source", "dataset", "--p-babble", "0.5", "--p-noise", "1.0")
    assert hist["noise_source"] == "dataset" and hist["p_babble"] == 0.5
    assert "noise_source" not in run("b")                   # default runs keep their loss_history unchanged
