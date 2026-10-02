"""Reproducible, class-stratified perturbation recipes (`vcm.perturbation_plan`) and their use by the dataset/trainer."""

from __future__ import annotations

import gzip
import json
import math
import random

import pytest
import torch

from me2_voicegen.common.ambient_mix import AMBIENT_SUFFIX
from me2_voicegen.common.augment import Recipe, apply_recipe, apply_timestretch
from me2_voicegen.vcm import train as train_mod
from me2_voicegen.vcm.dataset import VCMDataset
from me2_voicegen.vcm.perturbation_plan import PlanConfig, build_epoch_plan, class_key, plan_summary

CFG = PlanConfig(p_stretch=0.25, p_rir=0.7, p_noise=0.5, p_babble=0.15, n_rir=200, n_noise=251, n_babble=180)


def make_rows(sizes: dict[str, int]) -> list[dict]:
    rows = []
    for key, n in sizes.items():
        label, _, slot = key.partition("|")
        rows += [{"label": label, "slot_value": slot, "filename": f"{key}-{i}.wav"} for i in range(n)]
    return rows


SIZES = {"STOP|": 111, "ALARM|6 AM": 41, "ALARM|8 AM": 40, "COLOR|Red": 7, "unknown|": 180}


def counts(plan, rows):
    return plan_summary(plan, rows)


def test_plan_is_deterministic_and_independent_of_row_order():
    rows = make_rows(SIZES)
    idx = list(range(len(rows)))
    a = build_epoch_plan(rows, idx, CFG, seed=0, epoch=1)
    shuffled = idx[:]
    random.Random(1).shuffle(shuffled)
    assert a == build_epoch_plan(rows, shuffled, CFG, seed=0, epoch=1)
    assert a != build_epoch_plan(rows, idx, CFG, seed=0, epoch=2)
    assert a != build_epoch_plan(rows, idx, CFG, seed=1, epoch=1)


def test_every_class_gets_its_exact_quota():
    rows = make_rows(SIZES)
    plan = build_epoch_plan(rows, list(range(len(rows))), CFG, seed=0, epoch=1)
    for key, c in counts(plan, rows).items():
        for name, p in (("stretch", 0.25), ("rir", 0.7), ("noise", 0.5), ("babble", 0.15)):
            assert math.floor(p * c["n"]) <= c[name] <= math.ceil(p * c["n"]), (key, name, c)


def test_quota_expectation_is_p_even_for_tiny_classes():
    rows = make_rows({"COLOR|Red": 7})  # 7 * 0.15 = 1.05 babble clips per epoch
    total = sum(counts(build_epoch_plan(rows, list(range(7)), CFG, seed=0, epoch=e), rows)["COLOR|red"]["babble"] for e in range(1, 401))
    assert total / 400 == pytest.approx(1.05, abs=0.06)


def test_continuous_parameters_cover_their_range_evenly():
    rows = make_rows({"STOP|": 200})
    plan = build_epoch_plan(rows, list(range(200)), CFG, seed=0, epoch=1)
    for attr, lo, hi in (("stretch", 0.85, 1.15), ("noise_snr_db", 5.0, 25.0), ("babble_snr_db", 12.0, 25.0)):
        values = [getattr(r, attr) for r in plan.values() if getattr(r, attr) is not None]
        assert all(lo <= v <= hi for v in values)
        k = len(values)
        assert sorted(int((v - lo) / (hi - lo) * k) for v in values) == list(range(k))  # one value per 1/k stratum


def test_pool_entries_are_used_evenly():
    rows = make_rows({"STOP|": 100})
    cfg = PlanConfig(p_noise=0.5, p_rir=0.9, n_noise=5, n_rir=200)   # 50 noise draws from a 5-clip pool
    plan = build_epoch_plan(rows, list(range(100)), cfg, seed=0, epoch=1)
    noise_use = [sum(r.noise == j for r in plan.values()) for j in range(5)]
    assert noise_use == [10] * 5
    rir = [r.rir for r in plan.values() if r.rir is not None]
    assert len(rir) == len(set(rir)) == 90                              # fewer draws than pool entries: no repeats


def test_config_rejects_empty_pools_and_class_key_is_case_insensitive():
    with pytest.raises(ValueError, match="noise pool is empty"):
        PlanConfig(p_noise=0.5, n_noise=0)
    assert class_key({"label": "COLOR", "slot_value": "Red"}) == class_key({"label": "COLOR", "slot_value": "red"}) == "COLOR|red"
    assert class_key({"label": "STOP"}) == "STOP|"


def test_apply_recipe_matches_manual_composition_and_empty_recipe_is_identity():
    g = torch.Generator().manual_seed(0)
    w = torch.randn(16000, generator=g)
    rirs, noises, babbles = [torch.randn(800, generator=g).abs() for _ in range(3)], [torch.randn(9000, generator=g)], [torch.randn(7000, generator=g)]
    assert torch.equal(apply_recipe(w, Recipe(), rirs, noises, babbles), w)
    r = Recipe(stretch=1.1, rir=1, noise=0, noise_snr_db=10.0, babble=0, babble_snr_db=15.0)
    out = apply_recipe(w, r, rirs, noises, babbles)
    assert torch.equal(out, apply_recipe(w, r, rirs, noises, babbles))
    assert out.shape[-1] == round(16000 * 1.1) and torch.isfinite(out).all()
    assert not torch.equal(out, apply_timestretch(w, 1.1))


@pytest.fixture
def plan_dataset(vcm_fake_manifest_factory):
    def spec(bucket, label, source="optionb", split="train", transcript="stop"):
        return {"bucket": bucket, "source_dataset": source, "label": label, "split": split, "transcript": transcript, "duration_s": 1.0}
    rows = [spec("target_commands", "STOP") for _ in range(6)] + [spec("babble", "unknown", transcript="hello there") for _ in range(4)]
    rows += [spec("silence", "silence", source="background_noise", transcript=""), spec("target_commands", "STOP", split="val")]
    return vcm_fake_manifest_factory(rows)


def make_ds(manifest, split="train", **p):
    ds = VCMDataset(manifest, split=split, noise_source="dataset")
    ds.enable_perturbation_plan(ds.plan_config(**{"p_stretch": 0.5, "p_rir": 1.0, "p_noise": 1.0, "p_babble": 1.0, **p}), seed=0)
    return ds


def test_dataset_output_depends_only_on_epoch_not_on_call_order(plan_dataset):
    ds = make_ds(plan_dataset)
    ds.set_epoch(1)
    first = [ds[i].waveform for i in range(len(ds))]
    again = [ds[i].waveform for i in reversed(range(len(ds)))][::-1]     # a different access order
    assert all(torch.equal(a, b) for a, b in zip(first, again))
    other = make_ds(plan_dataset)                                         # a brand-new dataset object (e.g. a worker)
    other.set_epoch(1)
    assert all(torch.equal(a, other[i].waveform) for i, a in enumerate(first))
    ds.set_epoch(2)
    assert any(not torch.equal(a, ds[i].waveform) for i, a in enumerate(first))   # a new epoch is a new draw
    assert not torch.equal(first[0], VCMDataset(plan_dataset, split="train")[0].waveform)  # and it did perturb


def test_plan_never_touches_val_and_skips_noise_on_ambient_rows(plan_dataset):
    val = make_ds(plan_dataset, split="val", p_babble=0.0)  # val has no babble rows; asking for babble would raise
    val.set_epoch(1)
    assert torch.equal(val[0].waveform, VCMDataset(plan_dataset, split="val")[0].waveform)
    ds = make_ds(plan_dataset, p_stretch=0.0, p_rir=0.0, p_babble=0.0)    # noise only
    ds.set_epoch(1)
    clean = VCMDataset(plan_dataset, split="train")[0].waveform
    assert not torch.equal(ds[0].waveform, clean)
    ds.rows[0]["source_dataset"] = "optionb" + AMBIENT_SUFFIX
    assert torch.equal(ds[0].waveform, clean)


def test_set_epoch_requires_enabling_and_empty_pool_fails_loudly(plan_dataset):
    with pytest.raises(RuntimeError):
        VCMDataset(plan_dataset, split="train").set_epoch(1)
    with pytest.raises(ValueError):                                       # manifest source here has no background_noise rows... for babble
        VCMDataset(plan_dataset, split="train").plan_config(0.0, 0.0, 0.0, 0.5)


def test_train_cli_table_mode_dumps_a_reproducible_plan(plan_dataset, tmp_path):
    def run(out, workers):
        train_mod.main(["--manifest", str(plan_dataset), "--out-dir", str(tmp_path / out), "--preset", "quartznet5x3",
                        "--max-epochs", "2", "--device", "cpu", "--num-workers", str(workers), "--batch-size", "2",
                        "--max-minutes", "5", "--noise-source", "dataset", "--p-babble", "0.5", "--p-noise", "1.0",
                        "--perturbation-plan", "table", "--dump-plan", "--patience", "5"])
        d = tmp_path / out / "metadata" / "perturbation_plan"
        return json.loads((tmp_path / out / "metadata" / "loss_history.json").read_text()), d
    hist, d1 = run("a", 0)
    _, d2 = run("b", 2)
    assert hist["perturbation_plan"] == "table"
    assert (d1 / "summary_epoch_001.json").exists()
    for name in ("epoch_001.csv.gz", "epoch_002.csv.gz"):
        assert gzip.open(d1 / name, "rt").read() == gzip.open(d2 / name, "rt").read()   # same seed, any worker count
    assert gzip.open(d1 / "epoch_001.csv.gz", "rt").read() != gzip.open(d1 / "epoch_002.csv.gz", "rt").read().replace("2,", "1,")
