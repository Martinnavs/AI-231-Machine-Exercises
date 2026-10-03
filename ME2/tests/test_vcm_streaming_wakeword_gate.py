"""Tests for `vcm.streaming.wakeword_gate` (feature `wakeword-gate`,
`feature-engineering/wakeword-gate/SPEC.md`). Fast tests use synthetic
checkpoints/fake ORT sessions/scripted fake backends under `tmp_path` and
never touch a real checkpoint or `.onnx` file; the real-artifact checks
(backend parity, gate opens/stays-closed on real audio, latency) are
`@pytest.mark.slow` and skipped when `out/wakeword/{checkpoints,export}`
don't exist on this machine.
"""

from __future__ import annotations

import csv
import time
from pathlib import Path

import numpy as np
import pytest
import torch

# Importing config triggers its module-level side effect
# (GATE_REGISTRY["wakeword"] = WakeWordGate) -- see config.py's own comment
# at that line for why registration lives there and not in gate.py.
from me2_voicegen.vcm.streaming import config as _config  # noqa: F401
from me2_voicegen.vcm.streaming.gate import GateState, resolve_gate
from me2_voicegen.vcm.streaming.wakeword_gate import (
    WAKEWORD_WINDOW_SAMPLES,
    WakeWordGate,
    WakewordOnnxBackend,
    WakewordTorchBackend,
    _softmax_wakeword_prob,
    _trailing_wakeword_window,
)
from me2_voicegen.common.features import SAMPLE_RATE
from me2_voicegen.wakeword.model import DSCNN, DSCNNConfig

PROJECT_ROOT = Path(__file__).resolve().parents[1]
WAKEWORD_RUN_DIR = PROJECT_ROOT / "out" / "wakeword"
WAKEWORD_MANIFEST = PROJECT_ROOT / "out" / "conversions" / "v2" / "wakeword" / "manifest.csv"


def _tiny_config() -> DSCNNConfig:
    return DSCNNConfig(
        n_mels=40, n_classes=3, n_blocks=1, channels=8, kernel_sizes=[5], prologue_channels=8
    )


def _write_synthetic_checkpoint(tmp_path, seed=0) -> Path:
    torch.manual_seed(seed)
    config = _tiny_config()
    model = DSCNN(config)
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "preset": "test-fixture",
        "config": {
            "n_mels": config.n_mels,
            "n_classes": config.n_classes,
            "n_blocks": config.n_blocks,
            "channels": config.channels,
            "kernel_sizes": config.kernel_sizes,
            "prologue_channels": config.prologue_channels,
        },
        "labels": ["_wakeword_", "_unknown_", "_silence_"],
        "seed": seed,
        "epoch": 1,
        "val_loss": 0.5,
        "val_acc": 0.9,
        "window_seconds": 1.5,
        "license": "test-fixture-license",
    }
    path = tmp_path / "checkpoint.pt"
    torch.save(checkpoint, path)
    return path


# ---------------------------------------------------------------------------
# _softmax_wakeword_prob
# ---------------------------------------------------------------------------


def test_softmax_wakeword_prob_matches_torch_reference():
    rng = np.random.default_rng(0)
    logits = rng.normal(size=(3,)).astype(np.float32)
    expected = torch.from_numpy(logits).softmax(dim=-1)[0].item()  # index 0 = "_wakeword_"
    actual = _softmax_wakeword_prob(logits)
    assert actual == pytest.approx(expected, abs=1e-6)


def test_softmax_wakeword_prob_is_a_valid_probability():
    rng = np.random.default_rng(1)
    for _ in range(10):
        logits = (rng.normal(size=(3,)) * 10).astype(np.float32)
        prob = _softmax_wakeword_prob(logits)
        assert 0.0 <= prob <= 1.0


# ---------------------------------------------------------------------------
# _trailing_wakeword_window -- the one non-obvious correctness point
# ---------------------------------------------------------------------------


def test_trailing_window_crop_takes_most_recent_slice_not_centered():
    """The ring-buffer snapshot can be several seconds long; the classifier
    must only ever see the TRAILING 1.5s, never a centered slice of the
    whole buffer. Construct a buffer where a marker value sits only in the
    true trailing region and assert the crop captures exactly it -- a
    centered crop of this same buffer would land entirely on the leading
    silence instead and miss the marker completely."""
    silence_lead = np.zeros(int(3.0 * SAMPLE_RATE), dtype=np.float32)
    marker = np.full(int(0.5 * SAMPLE_RATE), 7.0, dtype=np.float32)
    silence_trail = np.zeros(int(1.0 * SAMPLE_RATE), dtype=np.float32)
    window = np.concatenate([silence_lead, marker, silence_trail])

    cropped = _trailing_wakeword_window(window)

    assert cropped.shape[-1] == WAKEWORD_WINDOW_SAMPLES
    marker_samples = marker.shape[-1]
    assert np.all(cropped[:marker_samples] == 7.0)
    assert np.all(cropped[marker_samples:] == 0.0)

    # Sanity: a naive centered crop of the SAME buffer would miss the
    # marker entirely (it falls in the pre-marker silence) -- pins the
    # "trailing, not centered" requirement against the alternative a
    # careless implementation could produce.
    n = window.shape[-1]
    center_start = (n - WAKEWORD_WINDOW_SAMPLES) // 2
    centered = window[center_start : center_start + WAKEWORD_WINDOW_SAMPLES]
    assert not np.any(centered == 7.0)


def test_trailing_window_crop_pads_when_buffer_shorter_than_window():
    """Buffer not yet filled to 1.5s -- `center_window`'s zero-pad branch,
    reused as a no-op-crop/pad-only step, not a centering step (the
    content itself is too short to be "centered" meaningfully differently
    from padded)."""
    short = np.full(4000, 3.0, dtype=np.float32)  # 0.25s
    cropped = _trailing_wakeword_window(short)
    assert cropped.shape[-1] == WAKEWORD_WINDOW_SAMPLES
    nonzero = np.flatnonzero(cropped)
    assert len(nonzero) == 4000
    pad_left = (WAKEWORD_WINDOW_SAMPLES - 4000) // 2
    assert nonzero[0] == pad_left
    assert nonzero[-1] == pad_left + 4000 - 1


def test_trailing_window_crop_is_a_noop_when_already_exact_length():
    exact = np.arange(WAKEWORD_WINDOW_SAMPLES, dtype=np.float32)
    cropped = _trailing_wakeword_window(exact)
    assert np.array_equal(cropped, exact)


# ---------------------------------------------------------------------------
# WakeWordGate state machine (scripted fake backend, no real model)
# ---------------------------------------------------------------------------


class _ScriptedBackend:
    """Returns one probability per `wakeword_prob` call, in order."""

    def __init__(self, probs) -> None:
        self._probs = list(probs)
        self.calls = 0

    def wakeword_prob(self, waveform: np.ndarray) -> float:
        self.calls += 1
        return self._probs.pop(0)


_DUMMY_WINDOW = np.zeros(WAKEWORD_WINDOW_SAMPLES, dtype=np.float32)


def test_low_probability_keeps_gate_closed():
    g = WakeWordGate(_ScriptedBackend([0.1]), threshold=0.9, period_s=5.0)
    assert g.poll(0, window=_DUMMY_WINDOW) == GateState(is_open=False, open_at_samples=None)


def test_probability_at_or_above_threshold_opens_at_current_samples_seen():
    g = WakeWordGate(_ScriptedBackend([0.9]), threshold=0.9, period_s=5.0)
    assert g.poll(1000, window=_DUMMY_WINDOW) == GateState(is_open=True, open_at_samples=1000)


def test_still_open_without_renewed_detection_keeps_original_open_at():
    g = WakeWordGate(_ScriptedBackend([0.95, 0.1]), threshold=0.9, period_s=5.0)
    assert g.poll(1000, window=_DUMMY_WINDOW) == GateState(is_open=True, open_at_samples=1000)
    assert g.poll(1250, window=_DUMMY_WINDOW) == GateState(is_open=True, open_at_samples=1000)


def test_renewed_detection_while_open_restarts_period_discard_and_restart():
    g = WakeWordGate(_ScriptedBackend([0.95, 0.95]), threshold=0.9, period_s=5.0)
    assert g.poll(1000, window=_DUMMY_WINDOW) == GateState(is_open=True, open_at_samples=1000)
    assert g.poll(2000, window=_DUMMY_WINDOW) == GateState(is_open=True, open_at_samples=2000)


def test_period_auto_closes_exactly_at_open_at_plus_period():
    g = WakeWordGate(_ScriptedBackend([0.95] + [0.0] * 3), threshold=0.9, period_s=5.0)
    g.poll(1000, window=_DUMMY_WINDOW)  # open at 1000
    close_at = 1000 + int(5.0 * SAMPLE_RATE)
    assert g.poll(close_at - 1, window=_DUMMY_WINDOW).is_open is True
    assert g.poll(close_at, window=_DUMMY_WINDOW) == GateState(is_open=False, open_at_samples=None)


def test_poll_with_no_window_stays_closed_and_never_calls_backend():
    backend = _ScriptedBackend([1.0])
    g = WakeWordGate(backend, threshold=0.9, period_s=5.0)
    assert g.poll(0, window=None) == GateState(is_open=False, open_at_samples=None)
    assert backend.calls == 0


def test_poll_with_empty_window_stays_closed_and_never_calls_backend():
    backend = _ScriptedBackend([1.0])
    g = WakeWordGate(backend, threshold=0.9, period_s=5.0)
    assert g.poll(0, window=np.zeros(0, dtype=np.float32)) == GateState(is_open=False, open_at_samples=None)
    assert backend.calls == 0


def test_close_is_idempotent_and_poll_after_close_returns_closed():
    backend = _ScriptedBackend([0.95, 1.0])
    g = WakeWordGate(backend, threshold=0.9, period_s=5.0)
    g.poll(1000, window=_DUMMY_WINDOW)
    g.close()
    g.close()
    assert g.poll(2000, window=_DUMMY_WINDOW) == GateState(is_open=False, open_at_samples=None)
    assert backend.calls == 1  # poll after close never reaches the backend


# ---------------------------------------------------------------------------
# resolve_gate("wakeword", ...) wiring
# ---------------------------------------------------------------------------


def test_resolve_gate_wakeword_returns_configured_gate():
    backend = _ScriptedBackend([0.95])
    g = resolve_gate("wakeword", period_s=5.0, wakeword_backend=backend, wakeword_threshold=0.8)
    assert isinstance(g, WakeWordGate)
    assert g.poll(500, window=_DUMMY_WINDOW) == GateState(is_open=True, open_at_samples=500)


def test_resolve_gate_wakeword_without_backend_is_actionable_system_exit():
    with pytest.raises(SystemExit) as excinfo:
        resolve_gate("wakeword", period_s=5.0)
    assert "wakeword_backend" in str(excinfo.value)


# ---------------------------------------------------------------------------
# __main__.main() cross-validation, generalized from `cfg.gate == "spacebar"`
# to `cfg.gate != "none"` -- both directions happen before any model load or
# mic open, so a minimal argv reaches the SystemExit without real artifacts.
# ---------------------------------------------------------------------------


def test_main_gate_wakeword_with_threshold_policy_is_actionable_system_exit():
    from me2_voicegen.vcm.streaming import __main__ as main_mod

    with pytest.raises(SystemExit) as excinfo:
        main_mod.main(["--policy", "threshold", "--gate", "wakeword"])
    message = str(excinfo.value)
    assert "wakeword" in message
    assert "mode_period" in message and "single_period" in message


def test_main_policy_mode_period_with_gate_none_is_actionable_system_exit():
    from me2_voicegen.vcm.streaming import __main__ as main_mod

    with pytest.raises(SystemExit) as excinfo:
        main_mod.main(["--policy", "mode_period", "--gate", "none"])
    assert "listening gate" in str(excinfo.value)


# ---------------------------------------------------------------------------
# WakewordTorchBackend (synthetic checkpoint, no real artifacts)
# ---------------------------------------------------------------------------


def test_wakeword_torch_backend_loads_synthetic_checkpoint_with_weights_only(tmp_path):
    checkpoint_path = _write_synthetic_checkpoint(tmp_path)
    backend = WakewordTorchBackend(checkpoint_path, device="cpu")
    waveform = np.zeros(WAKEWORD_WINDOW_SAMPLES, dtype=np.float32)
    prob = backend.wakeword_prob(waveform)
    assert 0.0 <= prob <= 1.0


def test_wakeword_torch_backend_does_not_silently_fall_back_on_weights_only_failure(monkeypatch, tmp_path):
    """Mirrors `backends.TorchBackend`'s equivalent test: a checkpoint
    that fails `weights_only=True` must raise, never silently retry with
    `weights_only=False`."""
    import me2_voicegen.vcm.streaming.wakeword_gate as wakeword_gate_module

    checkpoint_path = _write_synthetic_checkpoint(tmp_path)
    real_load = wakeword_gate_module.torch.load
    calls = []

    def fake_load(path, map_location=None, weights_only=False):
        calls.append(weights_only)
        if weights_only:
            raise RuntimeError("simulated weights_only allow-list rejection")
        return real_load(path, map_location=map_location)

    monkeypatch.setattr(wakeword_gate_module.torch, "load", fake_load)
    with pytest.raises(RuntimeError, match="simulated weights_only"):
        WakewordTorchBackend(checkpoint_path, device="cpu")
    assert calls == [True]  # never retried with weights_only=False


# ---------------------------------------------------------------------------
# WakewordOnnxBackend (fake ORT session, no real .onnx file)
# ---------------------------------------------------------------------------


class _FakeSession:
    def __init__(self, model_path, sess_options=None, providers=None):
        self.model_path = model_path
        self.sess_options = sess_options
        self.providers = providers

    def run(self, output_names, feed_dict):
        features = feed_dict["features"]
        batch = features.shape[0]
        rng = np.random.default_rng(0)
        logits = rng.normal(size=(batch, 3)).astype(np.float32)
        return [logits]


def test_wakeword_onnx_backend_applies_softmax_to_session_output(monkeypatch, tmp_path):
    import onnxruntime

    monkeypatch.setattr(onnxruntime, "InferenceSession", _FakeSession)
    onnx_path = tmp_path / "fake_model.onnx"
    onnx_path.write_bytes(b"not-a-real-onnx-file")
    backend = WakewordOnnxBackend(onnx_path, ort_threads=1)

    prob = backend.wakeword_prob(np.zeros(WAKEWORD_WINDOW_SAMPLES, dtype=np.float32))
    assert 0.0 <= prob <= 1.0


def test_wakeword_onnx_backend_passes_ort_threads_through_session_options(monkeypatch, tmp_path):
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
    WakewordOnnxBackend(onnx_path, ort_threads=4)

    assert captured["intra_op"] == 4
    assert captured["inter_op"] == 1
    assert captured["providers"] == ["CPUExecutionProvider"]


# ---------------------------------------------------------------------------
# Slow: real artifacts, real numeric behavior against real audio.
# ---------------------------------------------------------------------------

_REAL_ARTIFACTS_MISSING = (
    not (WAKEWORD_RUN_DIR / "checkpoints" / "checkpoint.pt").exists()
    or not (WAKEWORD_RUN_DIR / "export" / "wakeword_model.fp32.onnx").exists()
    or not WAKEWORD_MANIFEST.exists()
)


def _real_val_clip(label: str) -> tuple[np.ndarray, dict]:
    import soundfile as sf

    root = WAKEWORD_MANIFEST.parent
    with open(WAKEWORD_MANIFEST, newline="") as f:
        rows = list(csv.DictReader(f))
    row = next(r for r in rows if r.get("split") == "val" and r["label"] == label)
    wav, sr = sf.read(root / row["path"], dtype="float32")
    assert sr == SAMPLE_RATE
    return wav, row


@pytest.mark.slow
@pytest.mark.skipif(_REAL_ARTIFACTS_MISSING, reason="no checked-in wakeword checkpoint/onnx export on this machine")
def test_onnx_backend_matches_torch_backend_on_real_checkpoint():
    torch_backend = WakewordTorchBackend(WAKEWORD_RUN_DIR / "checkpoints" / "checkpoint.pt", device="cpu")
    onnx_backend = WakewordOnnxBackend(WAKEWORD_RUN_DIR / "export" / "wakeword_model.fp32.onnx", ort_threads=1)

    waveform, _ = _real_val_clip("_wakeword_")
    cropped = _trailing_wakeword_window(waveform.astype(np.float32))

    torch_prob = torch_backend.wakeword_prob(cropped)
    onnx_prob = onnx_backend.wakeword_prob(cropped)
    assert torch_prob == pytest.approx(onnx_prob, abs=1e-3)


@pytest.mark.slow
@pytest.mark.skipif(_REAL_ARTIFACTS_MISSING, reason="no checked-in wakeword checkpoint/onnx export on this machine")
def test_gate_opens_on_real_wakeword_clip_aligned_to_window_end():
    """Simulates the live-streaming case: the ring buffer's trailing edge
    tracks the end of a just-spoken wakeword utterance (the clip's own
    `speech_end_s`, from the dataset's precomputed VAD span, plus a small
    margin) -- confirmed by direct measurement (see this feature's SPEC.md
    Proof table) to score near 1.0 for every real val-split wakeword clip
    tried, well above the default threshold."""
    from me2_voicegen.vcm.streaming.gate import DEFAULT_WAKEWORD_THRESHOLD

    waveform, row = _real_val_clip("_wakeword_")
    speech_end_samples = int(float(row["speech_end_s"]) * SAMPLE_RATE) + int(0.1 * SAMPLE_RATE)
    window = waveform[: min(speech_end_samples, len(waveform))]

    backend = WakewordTorchBackend(WAKEWORD_RUN_DIR / "checkpoints" / "checkpoint.pt", device="cpu")
    gate = WakeWordGate(backend, threshold=DEFAULT_WAKEWORD_THRESHOLD, period_s=5.0)
    state = gate.poll(samples_seen=len(window), window=window.astype(np.float32))
    assert state.is_open is True


@pytest.mark.slow
@pytest.mark.skipif(_REAL_ARTIFACTS_MISSING, reason="no checked-in wakeword checkpoint/onnx export on this machine")
def test_gate_stays_closed_on_real_non_wakeword_clip():
    from me2_voicegen.vcm.streaming.gate import DEFAULT_WAKEWORD_THRESHOLD

    waveform, _ = _real_val_clip("_unknown_")
    backend = WakewordTorchBackend(WAKEWORD_RUN_DIR / "checkpoints" / "checkpoint.pt", device="cpu")
    gate = WakeWordGate(backend, threshold=DEFAULT_WAKEWORD_THRESHOLD, period_s=5.0)
    state = gate.poll(samples_seen=len(waveform), window=waveform.astype(np.float32))
    assert state.is_open is False


@pytest.mark.slow
@pytest.mark.skipif(_REAL_ARTIFACTS_MISSING, reason="no checked-in wakeword checkpoint/onnx export on this machine")
def test_wakeword_prob_latency_stays_under_default_stride_budget():
    """The runner's realtime loop evaluates every `stride_s` (default
    0.25s); a real forward pass must stay comfortably under that so the
    wakeword gate doesn't threaten the loop's cadence. Bound set generously
    above the ~55-60ms measured on this machine (see SPEC.md Proof)."""
    backend = WakewordTorchBackend(WAKEWORD_RUN_DIR / "checkpoints" / "checkpoint.pt", device="cpu")
    waveform = np.zeros(WAKEWORD_WINDOW_SAMPLES, dtype=np.float32)

    backend.wakeword_prob(waveform)  # warm-up (lazy CUDA/threadpool init, if any)
    n = 10
    start = time.perf_counter()
    for _ in range(n):
        backend.wakeword_prob(waveform)
    elapsed_ms = (time.perf_counter() - start) / n * 1000
    assert elapsed_ms < 150.0, elapsed_ms


# ---------------------------------------------------------------------------
# finer wake-word polling (`poll_step_s`)
# ---------------------------------------------------------------------------


class _EndBackend:
    """Fires only for the trailing window that ends exactly at `fire_end` (checked via the audio's last sample marker)."""

    def __init__(self, fire_marker: float) -> None:
        self.fire_marker = fire_marker
        self.calls = 0

    def wakeword_prob(self, waveform: np.ndarray) -> float:
        self.calls += 1
        return 1.0 if waveform[-1] == self.fire_marker else 0.0


def test_finer_polling_catches_a_wake_word_between_strides_and_opens_at_that_end():
    sr = 16000
    stride, step = sr // 4, sr // 20  # 0.25 s stride, 0.05 s steps
    audio = np.arange(1, 6 * sr + 1, dtype=np.float32)  # sample i holds the value i, so the last sample identifies the window end
    fire_end = 2 * sr + 3 * step  # between the 2.0 s and 2.25 s polls
    coarse = WakeWordGate(_EndBackend(float(fire_end)), threshold=0.9, period_s=5.0)
    fine = WakeWordGate(_EndBackend(float(fire_end)), threshold=0.9, period_s=5.0, poll_step_s=0.05)
    states = {}
    for name, g in (("coarse", coarse), ("fine", fine)):
        for seen in range(stride, 4 * sr, stride):
            st = g.poll(seen, audio[:seen])
            if st.is_open:
                states.setdefault(name, st)
    assert "coarse" not in states  # the strided polls never end on fire_end
    assert states["fine"].open_at_samples == fire_end
    assert fine._backend.calls > coarse._backend.calls


def test_default_polling_is_unchanged():
    g = WakeWordGate(_ScriptedBackend([0.95]), threshold=0.9, period_s=5.0)
    assert g.poll(4000, _DUMMY_WINDOW).open_at_samples == 4000 and g._backend.calls == 1
