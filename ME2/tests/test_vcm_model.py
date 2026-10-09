import torch

from me2_voicegen.vcm.alphabet import ALPHABET_SIZE
from me2_voicegen.vcm.model import (
    DEFAULT_CONFIG,
    SPEC_SCALE_CONFIG,
    MatchboxNetCTC,
    build_model,
    estimated_int8_bytes,
    param_count,
)


def test_default_preset_instantiates_and_reports_size():
    model = build_model("default")
    n_params = param_count(model)
    assert 250_000 <= n_params <= 400_000, n_params
    n_bytes = estimated_int8_bytes(model)
    assert n_bytes > n_params  # includes per-tensor scale overhead


def test_spec_scale_preset_instantiates_and_reports_size():
    model = build_model("spec-scale")
    n_params = param_count(model)
    assert 1_200_000 <= n_params <= 2_500_000, n_params
    n_bytes = estimated_int8_bytes(model)
    assert n_bytes > n_params


def test_optionc_preset_instantiates_under_1m_and_bigger_than_default():
    model = build_model("optionc")
    n_params = param_count(model)
    assert 400_000 <= n_params < 1_000_000, n_params
    assert n_params > param_count(MatchboxNetCTC(DEFAULT_CONFIG))
    n_bytes = estimated_int8_bytes(model)
    assert n_bytes > n_params


def test_optiond_preset_instantiates_at_normal_scale_and_bigger_than_optionc():
    model = build_model("optiond")
    n_params = param_count(model)
    assert 950_000 <= n_params <= 1_050_000, n_params
    assert n_params > param_count(build_model("optionc"))
    n_bytes = estimated_int8_bytes(model)
    assert n_bytes > n_params


def test_unknown_preset_raises():
    import pytest

    with pytest.raises(ValueError):
        build_model("nonexistent")


def test_forward_shape_matches_alphabet_and_preserves_time():
    model = MatchboxNetCTC(DEFAULT_CONFIG)
    batch, n_mels, t = 2, 40, 37
    features = torch.randn(batch, n_mels, t)
    logits = model(features)
    assert logits.shape == (batch, t, ALPHABET_SIZE)


def test_forward_shape_spec_scale():
    model = MatchboxNetCTC(SPEC_SCALE_CONFIG)
    features = torch.randn(1, 40, 21)
    logits = model(features)
    assert logits.shape == (1, 21, ALPHABET_SIZE)


def test_forward_shape_optiond():
    model = build_model("optiond")
    features = torch.randn(1, 40, 21)
    logits = model(features)
    assert logits.shape == (1, 21, ALPHABET_SIZE)
