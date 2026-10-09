import torch

from me2_voicegen.wakeword.model import (
    DEFAULT_CONFIG,
    LABEL_TO_ID,
    LABELS,
    DSCNN,
    build_model,
    estimated_int8_bytes,
    param_count,
)


def test_labels_and_label_to_id_are_consistent():
    assert LABELS == ("_wakeword_", "_unknown_", "_silence_")
    assert LABEL_TO_ID == {"_wakeword_": 0, "_unknown_": 1, "_silence_": 2}


def test_default_preset_instantiates_and_reports_size():
    model = build_model("default")
    n_params = param_count(model)
    assert n_params > 0
    n_bytes = estimated_int8_bytes(model)
    assert n_bytes > n_params  # includes per-tensor scale overhead


def test_unknown_preset_raises():
    import pytest

    with pytest.raises(ValueError):
        build_model("nonexistent")


def test_forward_shape_is_batch_by_n_classes():
    model = DSCNN(DEFAULT_CONFIG)
    batch, n_mels, t = 4, 40, 151
    features = torch.randn(batch, n_mels, t)
    logits = model(features)
    assert logits.shape == (batch, len(LABELS))


def test_forward_handles_varying_time_length():
    """Global average pool over time means any T works, unlike vcm's
    per-frame CTC model -- this is a genuine behavioral difference worth
    a dedicated test, not an oversight if it looks surprising next to
    vcm.model's frame-preserving contract."""
    model = build_model("default")
    for t in (10, 50, 151, 300):
        features = torch.randn(1, 40, t)
        logits = model(features)
        assert logits.shape == (1, len(LABELS))


def test_kernel_sizes_mismatch_raises():
    import pytest

    from me2_voicegen.wakeword.model import DSCNNConfig

    with pytest.raises(ValueError):
        DSCNNConfig(n_blocks=3, kernel_sizes=[10, 10])


def test_kernel_sizes_single_value_broadcasts():
    from me2_voicegen.wakeword.model import DSCNNConfig

    config = DSCNNConfig(n_blocks=3, kernel_sizes=[10])
    assert config.kernel_sizes == [10, 10, 10]
