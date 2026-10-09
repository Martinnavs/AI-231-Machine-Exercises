"""QuartzNet model, length arithmetic, and the model factory."""

from __future__ import annotations

import dataclasses

import pytest
import torch

from me2_voicegen.vcm.model import (
    PRESETS,
    build_model,
    build_model_from_config,
    model_type_for_config,
    param_count,
)
from me2_voicegen.vcm.quartznet import QuartzNetConfig, QuartzNetCTC

OPTIOND_PARAMS = 1_009_725


def test_preset_param_count_in_budget():
    n = param_count(build_model("quartznet5x3"))
    assert 900_000 <= n <= OPTIOND_PARAMS  # width capped by the INT8-size gate, see quartznet.py
    assert n > param_count(build_model("optionc"))


def test_optiond_param_count_unchanged():
    assert param_count(build_model("optiond")) == OPTIOND_PARAMS


def test_forward_shape_stride2():
    model = build_model("quartznet5x3").eval()
    assert model(torch.randn(2, 40, 151)).shape == (2, 76, 29)


@pytest.mark.parametrize("stride", [1, 2, 4])
def test_output_lengths_match_forward(stride):
    model = QuartzNetCTC(QuartzNetConfig(channels=16, time_stride=stride, epilogue_channels=16)).eval()
    for length in range(1, 301):
        out = model(torch.randn(1, 40, length))
        assert model.output_lengths(torch.tensor([length])).item() == out.shape[1], length
        assert out.shape[1] == -(-length // stride)


def test_output_lengths_padded_batch():
    model = QuartzNetCTC(QuartzNetConfig(channels=16, epilogue_channels=16)).eval()
    lengths = torch.tensor([151, 100, 37])
    assert model(torch.randn(3, 40, 151)).shape[1] == model.output_lengths(lengths).max().item()
    assert model.output_lengths(lengths).dtype == torch.long


def test_matchbox_output_lengths_identity():
    lengths = torch.tensor([151, 37])
    assert torch.equal(build_model("default").output_lengths(lengths), lengths)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"time_stride": 3},
        {"time_stride": 0},
        {"kernel_sizes": [7, 9, 10, 13, 15]},
        {"kernel_sizes": [7, 9]},
        {"dropout": 1.0},
        {"dropout": -0.1},
    ],
)
def test_config_validation(kwargs):
    with pytest.raises(ValueError):
        QuartzNetConfig(**kwargs)


def test_factory_types_and_legacy():
    q = dataclasses.asdict(PRESETS["quartznet5x3"])
    assert isinstance(build_model_from_config(q, "quartznet"), QuartzNetCTC)
    d = dataclasses.asdict(PRESETS["default"])
    assert type(build_model_from_config(d, None)).__name__ == "MatchboxNetCTC"
    assert type(build_model_from_config(d, "matchboxnet")).__name__ == "MatchboxNetCTC"
    with pytest.raises(ValueError):
        build_model_from_config(d, "resnet")
    assert model_type_for_config(PRESETS["optiond"]) == "matchboxnet"
    assert model_type_for_config(PRESETS["quartznet5x3"]) == "quartznet"


def test_eval_deterministic_and_weights_only_config(tmp_path):
    model = QuartzNetCTC(QuartzNetConfig(channels=16, epilogue_channels=16, dropout=0.3)).eval()
    x = torch.randn(1, 40, 60)
    assert torch.equal(model(x), model(x))
    path = tmp_path / "c.pt"
    torch.save({"config": dataclasses.asdict(model.config), "model_type": "quartznet"}, path)
    assert torch.load(path, weights_only=True)["model_type"] == "quartznet"


def test_ablation_presets():
    assert build_model("quartznet5x3-s1").total_stride == 1
    assert build_model("quartznet5x3-s1")(torch.randn(1, 40, 151)).shape == (1, 151, 29)
    assert param_count(build_model("quartznet5x3-s1")) == param_count(build_model("quartznet5x3"))
    wide = build_model("optiond-wide")
    assert wide.config.kernel_sizes == [61] * 5
    assert wide(torch.randn(1, 40, 151)).shape == (1, 151, 29)
