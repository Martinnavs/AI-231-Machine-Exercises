"""Tests for `vcm.streaming.backends`. Fast tests use synthetic
checkpoints/fake ORT sessions under `tmp_path` and never touch a real
checkpoint or `.onnx` file; the one real-artifact parity check
(`OnnxBackend` vs. `TorchBackend` on the checked-in optionc run dir) is
`@pytest.mark.slow`.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

from me2_voicegen.vcm.model import MatchboxNetConfig, MatchboxNetCTC
from me2_voicegen.vcm.streaming.backends import OnnxBackend, TorchBackend, _log_softmax

PROJECT_ROOT = Path(__file__).resolve().parents[1]
OPTIONC_RUN_DIR = PROJECT_ROOT / "out" / "vcm" / "optionb-optionc"


def _tiny_config() -> MatchboxNetConfig:
    return MatchboxNetConfig(
        n_mels=40, n_blocks=1, channels=8, kernel_sizes=[5], prologue_channels=8, epilogue_channels=8
    )


def _write_synthetic_checkpoint(tmp_path, seed=0) -> Path:
    torch.manual_seed(seed)
    config = _tiny_config()
    model = MatchboxNetCTC(config)
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "preset": "test-fixture",
        "config": {
            "n_mels": config.n_mels,
            "n_blocks": config.n_blocks,
            "channels": config.channels,
            "kernel_sizes": config.kernel_sizes,
            "prologue_channels": config.prologue_channels,
            "epilogue_channels": config.epilogue_channels,
            "alphabet_size": config.alphabet_size,
        },
        "alphabet_size": config.alphabet_size,
        "seed": seed,
        "epoch": 1,
        "val_loss": 0.5,
        "license": "test-fixture-license",
    }
    path = tmp_path / "checkpoint.pt"
    torch.save(checkpoint, path)
    return path


# ---------------------------------------------------------------------------
# _log_softmax
# ---------------------------------------------------------------------------


def test_log_softmax_matches_torch_reference():
    rng = np.random.default_rng(0)
    logits = rng.normal(size=(7, 29)).astype(np.float32)
    expected = torch.from_numpy(logits).log_softmax(dim=-1).numpy()
    actual = _log_softmax(logits, axis=-1)
    assert np.allclose(actual, expected, atol=1e-5)


def test_log_softmax_rows_sum_to_one_in_probability_space():
    rng = np.random.default_rng(1)
    logits = rng.normal(size=(4, 29)).astype(np.float32) * 10
    log_probs = _log_softmax(logits, axis=-1)
    probs = np.exp(log_probs)
    assert np.allclose(probs.sum(axis=-1), 1.0, atol=1e-5)


# ---------------------------------------------------------------------------
# TorchBackend (synthetic checkpoint, no real artifacts)
# ---------------------------------------------------------------------------


def test_torch_backend_matches_pipeline_logp_for_waveform(tmp_path):
    from me2_voicegen.common.features import LogMelFeatureExtractor
    from me2_voicegen.vcm.pipeline import load_checkpoint, logp_for_waveform

    checkpoint_path = _write_synthetic_checkpoint(tmp_path)
    backend = TorchBackend(checkpoint_path, device="cpu")

    waveform = np.zeros(16000, dtype=np.float32)
    actual = backend.logp_for_waveform(waveform)

    model, _ = load_checkpoint(checkpoint_path, device="cpu")
    expected = logp_for_waveform(
        model, LogMelFeatureExtractor(), torch.from_numpy(waveform), device="cpu"
    )
    assert actual.shape == expected.shape
    assert np.allclose(actual, expected, atol=1e-5)


def test_torch_backend_loads_synthetic_checkpoint_with_weights_only(tmp_path):
    checkpoint_path = _write_synthetic_checkpoint(tmp_path)
    backend = TorchBackend(checkpoint_path, device="cpu")
    assert backend.checkpoint["preset"] == "test-fixture"


def test_torch_backend_does_not_silently_fall_back_on_weights_only_failure(monkeypatch, tmp_path):
    """SEC-2: TorchBackend must call load_checkpoint with
    allow_unsafe_load=False -- a checkpoint that fails weights_only=True
    is a hard error here, never a silent unsafe retry."""
    import me2_voicegen.vcm.pipeline as pipeline_module

    checkpoint_path = _write_synthetic_checkpoint(tmp_path)
    real_torch_load = pipeline_module.torch.load
    unsafe_load_calls = []

    def fake_torch_load(path, map_location=None, weights_only=False):
        if weights_only:
            raise RuntimeError("simulated weights_only allow-list rejection")
        unsafe_load_calls.append(1)
        return real_torch_load(path, map_location=map_location)

    monkeypatch.setattr(pipeline_module.torch, "load", fake_torch_load)
    with pytest.raises(RuntimeError, match="allow_unsafe_load"):
        TorchBackend(checkpoint_path, device="cpu")
    assert unsafe_load_calls == []


# ---------------------------------------------------------------------------
# OnnxBackend (fake ORT session, no real .onnx file)
# ---------------------------------------------------------------------------


class _FakeSession:
    def __init__(self, model_path, sess_options=None, providers=None):
        self.model_path = model_path
        self.sess_options = sess_options
        self.providers = providers

    def run(self, output_names, feed_dict):
        features = feed_dict["features"]
        batch, n_mels, t = features.shape
        rng = np.random.default_rng(0)
        logits = rng.normal(size=(batch, t, 29)).astype(np.float32)
        return [logits]


def test_onnx_backend_applies_log_softmax_to_session_output(monkeypatch, tmp_path):
    import onnxruntime

    monkeypatch.setattr(onnxruntime, "InferenceSession", _FakeSession)

    onnx_path = tmp_path / "fake_model.onnx"
    onnx_path.write_bytes(b"not-a-real-onnx-file")
    backend = OnnxBackend(onnx_path, ort_threads=1)

    waveform = np.zeros(16000, dtype=np.float32)
    logp = backend.logp_for_waveform(waveform)

    assert logp.ndim == 2
    assert logp.shape[1] == 29
    probs = np.exp(logp)
    assert np.allclose(probs.sum(axis=-1), 1.0, atol=1e-4)


def test_onnx_backend_passes_ort_threads_through_session_options(monkeypatch, tmp_path):
    captured = {}

    class _CapturingFakeSession(_FakeSession):
        def __init__(self, model_path, sess_options=None, providers=None):
            captured["intra_op"] = sess_options.intra_op_num_threads
            captured["inter_op"] = sess_options.inter_op_num_threads
            captured["providers"] = providers
            super().__init__(model_path, sess_options, providers)

    import onnxruntime

    monkeypatch.setattr(onnxruntime, "InferenceSession", _CapturingFakeSession)
    onnx_path = tmp_path / "fake_model.onnx"
    onnx_path.write_bytes(b"not-a-real-onnx-file")
    OnnxBackend(onnx_path, ort_threads=4)

    assert captured["intra_op"] == 4
    assert captured["inter_op"] == 1
    assert captured["providers"] == ["CPUExecutionProvider"]


# ---------------------------------------------------------------------------
# Slow: real artifacts, real numeric parity.
# ---------------------------------------------------------------------------


@pytest.mark.slow
@pytest.mark.skipif(
    not (OPTIONC_RUN_DIR / "checkpoints" / "checkpoint.pt").exists()
    or not (OPTIONC_RUN_DIR / "export" / "vcm_model.fp32.onnx").exists(),
    reason="no checked-in optionb-optionc checkpoint/onnx export on this machine",
)
def test_onnx_backend_matches_torch_backend_on_real_optionc_artifacts():
    torch_backend = TorchBackend(OPTIONC_RUN_DIR / "checkpoints" / "checkpoint.pt", device="cpu")
    onnx_backend = OnnxBackend(OPTIONC_RUN_DIR / "export" / "vcm_model.fp32.onnx", ort_threads=1)

    torch.manual_seed(0)
    waveform = (torch.randn(40000) * 0.01).numpy().astype(np.float32)  # 2.5s @ 16kHz

    torch_logp = torch_backend.logp_for_waveform(waveform)
    onnx_logp = onnx_backend.logp_for_waveform(waveform)

    assert torch_logp.shape == onnx_logp.shape
    max_abs_diff = np.abs(torch_logp - onnx_logp).max()
    assert max_abs_diff < 1e-2, max_abs_diff
