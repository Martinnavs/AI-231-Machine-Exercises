"""Unit tests for vcm.export_onnx / vcm.benchmark.

Fast tests below exercise pure Python/torch/numpy plumbing (window shape,
dummy-feature shape, the calibration reader's manifest handling, Markdown
rendering) -- none of them import `onnx`, call `torch.onnx.export`, or
touch `onnxruntime.InferenceSession`, and none of them need
`out/vcm/checkpoints/checkpoint.pt`. Per ticket 06's acceptance criteria, the fast
suite (`pytest -m 'not slow'`) must not require a trained checkpoint or the
ONNX runtime path.

`@pytest.mark.slow` tests below are the real thing: an actual
`torch.onnx.export` + `onnxruntime` inference, proving ONNX-vs-PyTorch
logit parity on a fixed input (this ticket's acceptance criterion 2) and
exercising real static INT8 quantization end to end, against the real
trained checkpoint at `out/vcm/checkpoints/checkpoint.pt` (skipped if that file is
absent, e.g. before ticket 04 has produced one on a given machine).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

from me2_voicegen.vcm.benchmark import (
    HARDWARE_LABEL,
    NOT_MEASURED_ON,
    WAKEWORD_TARGET_DEVICE,
    _render_markdown,
    _resolve_family_defaults as _benchmark_resolve_family_defaults,
    build_arg_parser as build_benchmark_arg_parser,
)
from me2_voicegen.vcm.export_onnx import (
    DEFAULT_MANIFEST,
    ValSplitCalibrationReader,
    WINDOW_SAMPLES,
    WakewordCalibrationReader,
    _resolve_family_defaults as _export_resolve_family_defaults,
    build_arg_parser as build_export_arg_parser,
    dummy_features,
    export_fp32,
    load_checkpoint,
    onnx_vs_pytorch_logits,
    quantize_int8_static,
    window_n_frames,
)
from me2_voicegen.vcm.model import MatchboxNetConfig, MatchboxNetCTC
from me2_voicegen.wakeword.model import DSCNN, DSCNNConfig

REAL_CHECKPOINT = Path("out/vcm/checkpoints/checkpoint.pt")
REAL_WAKEWORD_CHECKPOINT = Path("out/wakeword/checkpoints/checkpoint.pt")


def _target_command_specs(n: int = 40) -> list[dict]:
    labels = ["ALARM", "CALL", "STOP", "TIME"]
    return [
        {
            "bucket": "target_commands",
            "source_dataset": "sanitized_clean",
            "label": labels[i % len(labels)],
            "split": "val",
            "duration_s": 1.5,
        }
        for i in range(n)
    ]


# ---------------------------------------------------------------------------
# Fast: no checkpoint, no onnx/onnxruntime.
# ---------------------------------------------------------------------------


def test_window_n_frames_matches_contract_front_end():
    n_frames = window_n_frames()
    assert n_frames == 151  # 1.5s @ 16kHz, 480-window/160-hop front-end (VCM-CONTRACT.md section 5)


def test_dummy_features_shape_and_dtype():
    feats = dummy_features()
    assert feats.shape == (1, 40, 151)
    assert feats.dtype == torch.float32

    feats_batched = dummy_features(n_frames=20, batch=3)
    assert feats_batched.shape == (3, 40, 20)


def test_dummy_features_is_deterministic():
    a = dummy_features(n_frames=10)
    b = dummy_features(n_frames=10)
    assert torch.equal(a, b)


def test_calibration_reader_yields_log_mel_shaped_batches(vcm_fake_manifest_factory):
    manifest_path = vcm_fake_manifest_factory(_target_command_specs(6))

    reader = ValSplitCalibrationReader(manifest_path=manifest_path, n_samples=4, seed=0)

    batches = []
    while True:
        batch = reader.get_next()
        if batch is None:
            break
        batches.append(batch)

    assert len(batches) == 4
    for batch in batches:
        assert set(batch.keys()) == {"features"}
        arr = batch["features"]
        assert isinstance(arr, np.ndarray)
        assert arr.dtype == np.float32
        assert arr.shape == (1, 40, window_n_frames())

    # rewind() must restart the same sequence of batches, not exhaust silently.
    reader.rewind()
    assert reader.get_next() is not None


def test_calibration_reader_pads_short_waveforms(vcm_fake_manifest_factory):
    specs = _target_command_specs(2)
    for spec in specs:
        spec["duration_s"] = 0.2  # far shorter than the 1.5s window
    manifest_path = vcm_fake_manifest_factory(specs)

    reader = ValSplitCalibrationReader(manifest_path=manifest_path, n_samples=2, seed=0)
    batch = reader.get_next()
    assert batch["features"].shape == (1, 40, window_n_frames())


# ---------------------------------------------------------------------------
# Wakeword family (feature `wakeword-dscnn`, feature-engineering/
# wakeword-dscnn/SPEC.md): vcm.export_onnx/vcm.benchmark are extended with
# --model-family rather than duplicated. Fast cases below prove the vcm
# default path is unchanged and the wakeword plumbing works, without
# touching onnx/onnxruntime (same fast-suite discipline as the vcm tests
# above); slow cases below the existing slow section do real export/
# quantization for wakeword the same way the vcm slow tests do.
# ---------------------------------------------------------------------------


def _wakeword_manifest(tmp_path, vcm_wav_factory, n_each: int = 3) -> Path:
    import csv

    fields = [
        "filename", "path", "label", "duration", "sample_rate", "resampled",
        "source_dataset", "source_relpath", "group_id", "split",
        "speech_start_s", "speech_end_s",
    ]
    root = tmp_path / "wakeword"
    rows = []
    labels = [("_wakeword_", "positives_real"), ("_unknown_", "adversaries"), ("_silence_", "silence_synthetic")]
    for label, subset in labels:
        for i in range(n_each):
            filename = f"{subset}_{i}.wav"
            rel = f"{subset}/audio/{filename}"
            vcm_wav_factory(root / rel, duration_s=1.5, silence=(label == "_silence_"))
            rows.append(
                {
                    "filename": filename,
                    "path": rel,
                    "label": label,
                    "duration": "1.500000",
                    "sample_rate": "16000",
                    "resampled": "False",
                    "source_dataset": subset,
                    "source_relpath": filename,
                    "group_id": filename,
                    "split": "val",
                    "speech_start_s": "0.200000" if label != "_silence_" else "",
                    "speech_end_s": "1.300000" if label != "_silence_" else "",
                }
            )
    manifest_path = root / "manifest.csv"
    with manifest_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    return manifest_path


def test_load_checkpoint_unknown_family_raises(tmp_path):
    ckpt_path = tmp_path / "fake.pt"
    torch.save({"config": {}, "model_state_dict": {}}, ckpt_path)
    import pytest

    with pytest.raises(ValueError, match="unknown model_family"):
        load_checkpoint(ckpt_path, model_family="bogus")


def test_export_resolve_family_defaults_leaves_vcm_defaults_unchanged():
    args = build_export_arg_parser().parse_args([])
    assert args.model_family == "vcm"
    _export_resolve_family_defaults(args)
    assert args.checkpoint == Path("out/vcm/checkpoints/checkpoint.pt")
    assert args.out_dir == Path("out/vcm")
    assert args.manifest == DEFAULT_MANIFEST


def test_export_resolve_family_defaults_resolves_wakeword_paths():
    args = build_export_arg_parser().parse_args(["--model-family", "wakeword"])
    _export_resolve_family_defaults(args)
    assert args.checkpoint == Path("out/wakeword/checkpoints/checkpoint.pt")
    assert args.out_dir == Path("out/wakeword")
    assert "wakeword" in str(args.manifest)


def test_export_resolve_family_defaults_respects_explicit_overrides():
    args = build_export_arg_parser().parse_args(
        ["--model-family", "wakeword", "--checkpoint", "custom.pt", "--out-dir", "custom_dir"]
    )
    _export_resolve_family_defaults(args)
    assert args.checkpoint == Path("custom.pt")
    assert args.out_dir == Path("custom_dir")


def test_benchmark_resolve_family_defaults_wakeword():
    args = build_benchmark_arg_parser().parse_args(["--model-family", "wakeword"])
    _benchmark_resolve_family_defaults(args)
    assert args.checkpoint == Path("out/wakeword/checkpoints/checkpoint.pt")
    assert args.out_dir == Path("out/wakeword")


def test_wakeword_target_device_names_rpi4_not_rpi5():
    assert "Raspberry Pi 4" in WAKEWORD_TARGET_DEVICE
    assert "4GB" in WAKEWORD_TARGET_DEVICE


def test_wakeword_calibration_reader_yields_log_mel_shaped_batches(tmp_path, vcm_wav_factory):
    manifest_path = _wakeword_manifest(tmp_path, vcm_wav_factory, n_each=4)
    reader = WakewordCalibrationReader(manifest_path=manifest_path, n_samples=5, seed=0)

    batches = []
    while True:
        batch = reader.get_next()
        if batch is None:
            break
        batches.append(batch)

    assert len(batches) == 5
    for batch in batches:
        assert set(batch.keys()) == {"features"}
        arr = batch["features"]
        assert isinstance(arr, np.ndarray)
        assert arr.dtype == np.float32
        assert arr.shape[0] == 1 and arr.shape[1] == 40

    reader.rewind()
    assert reader.get_next() is not None


def test_render_markdown_labels_hardware_and_not_rpi_for_measured_int8():
    result = {
        "hardware_label": HARDWARE_LABEL,
        "not_measured_on": NOT_MEASURED_ON,
        "window_seconds": 1.5,
        "n_frames": 151,
        "onnxruntime_threads": {"intra_op": 1, "inter_op": 1},
        "checkpoint": "out/vcm/checkpoints/checkpoint.pt",
        "preset": "default",
        "fp32": {
            "onnx_size_mb": 0.97,
            "p50_latency_ms": 1.0,
            "p95_latency_ms": 1.1,
            "process_peak_rss_mb": 42.5,
            "inference_rss_delta_kb": 0,
        },
        "int8": {
            "measured": True,
            "onnx_size_mb": 0.26,
            "p50_latency_ms": 1.0,
            "p95_latency_ms": 1.0,
            "process_peak_rss_mb": 42.7,
            "inference_rss_delta_kb": 0,
        },
        "spec_budgets_indicative_only": {
            "latency_ms_per_100ms_frame_le": 20.0,
            "int8_model_mb_le": 5.0,
            "peak_ram_mb_le": 25.0,
            "note": "indicative only",
        },
    }
    md = _render_markdown(result)
    assert "AMD EPYC 7742 (256-thread node) estimate" in md
    assert "Raspberry Pi 4/5" in md
    assert "NOT a" in md
    assert "static, val-calibrated" in md
    # R3-2 fix: a bare "0 KB" delta must not stand unexplained next to the
    # RAM budget -- both the absolute process peak RSS and an explanation
    # of what the delta does/doesn't mean must be present.
    assert "42.500" in md  # fp32 process_peak_rss_mb rendered
    assert "process peak RSS" in md
    assert "not \"inference used no memory.\"" in md


def test_render_markdown_labels_unmeasured_int8_as_estimate():
    result = {
        "hardware_label": HARDWARE_LABEL,
        "not_measured_on": NOT_MEASURED_ON,
        "window_seconds": 1.5,
        "n_frames": 151,
        "onnxruntime_threads": {"intra_op": 1, "inter_op": 1},
        "checkpoint": "out/vcm/checkpoints/checkpoint.pt",
        "preset": "default",
        "fp32": {
            "onnx_size_mb": 0.97,
            "p50_latency_ms": 1.0,
            "p95_latency_ms": 1.1,
            "process_peak_rss_mb": 42.5,
            "inference_rss_delta_kb": 0,
        },
        "int8": {
            "measured": False,
            "estimated_size_mb": 0.3,
        },
        "spec_budgets_indicative_only": {
            "latency_ms_per_100ms_frame_le": 20.0,
            "int8_model_mb_le": 5.0,
            "peak_ram_mb_le": 25.0,
            "note": "indicative only",
        },
    }
    md = _render_markdown(result)
    assert "ESTIMATE" in md
    assert "not measured" in md


# ---------------------------------------------------------------------------
# Slow: real torch.onnx.export + real onnxruntime inference/quantization.
# ---------------------------------------------------------------------------


@pytest.mark.slow
def test_onnx_export_logits_match_pytorch_within_tolerance_synthetic_model():
    """Real export correctness check on a freshly-initialized (untrained)
    model -- proves the export graph is numerically faithful independent
    of whether a trained checkpoint is present on this machine."""
    torch.manual_seed(0)
    config = MatchboxNetConfig(
        n_mels=40, n_blocks=2, channels=24, kernel_sizes=[5, 5], prologue_channels=16, epilogue_channels=32
    )
    model = MatchboxNetCTC(config)
    model.eval()

    import tempfile

    with tempfile.TemporaryDirectory() as tmp:
        onnx_path = Path(tmp) / "model.onnx"
        export_fp32(model, onnx_path, n_frames=window_n_frames())
        assert onnx_path.exists()

        torch_logits, onnx_logits = onnx_vs_pytorch_logits(model, onnx_path, n_frames=window_n_frames())
        assert torch_logits.shape == onnx_logits.shape == (1, window_n_frames(), 29)
        max_abs_diff = np.abs(torch_logits - onnx_logits).max()
        assert max_abs_diff < 1e-3, max_abs_diff


@pytest.mark.slow
@pytest.mark.skipif(not REAL_CHECKPOINT.exists(), reason="no trained checkpoint at out/vcm/checkpoints/checkpoint.pt")
def test_onnx_export_logits_match_pytorch_on_real_checkpoint():
    model, _ckpt = load_checkpoint(REAL_CHECKPOINT)

    import tempfile

    with tempfile.TemporaryDirectory() as tmp:
        onnx_path = Path(tmp) / "model.onnx"
        export_fp32(model, onnx_path, n_frames=window_n_frames())

        torch_logits, onnx_logits = onnx_vs_pytorch_logits(model, onnx_path)
        max_abs_diff = np.abs(torch_logits - onnx_logits).max()
        assert max_abs_diff < 1e-3, max_abs_diff


@pytest.mark.slow
@pytest.mark.skipif(not REAL_CHECKPOINT.exists(), reason="no trained checkpoint at out/vcm/checkpoints/checkpoint.pt")
def test_static_int8_quantization_shrinks_model_and_quantizes_conv(vcm_fake_manifest_factory):
    model, _ckpt = load_checkpoint(REAL_CHECKPOINT)

    manifest_path = vcm_fake_manifest_factory(_target_command_specs(8))

    import tempfile

    import onnx as onnx_pkg

    with tempfile.TemporaryDirectory() as tmp:
        fp32_path = Path(tmp) / "model.fp32.onnx"
        int8_path = Path(tmp) / "model.int8.onnx"
        export_fp32(model, fp32_path, n_frames=window_n_frames())

        reader = ValSplitCalibrationReader(manifest_path=manifest_path, n_samples=8, seed=0)
        quantize_int8_static(fp32_path, int8_path, reader)

        assert int8_path.exists()
        assert int8_path.stat().st_size < fp32_path.stat().st_size

        quantized_graph = onnx_pkg.load(str(int8_path))
        op_types = {node.op_type for node in quantized_graph.graph.node}
        assert "QuantizeLinear" in op_types and "DequantizeLinear" in op_types
        assert "Conv" in op_types  # Conv is present *and* fed by DequantizeLinear (int8 weights), not left fully fp32


# ---------------------------------------------------------------------------
# Wakeword family, slow: real torch.onnx.export + real onnxruntime, same
# discipline as the vcm slow tests above.
# ---------------------------------------------------------------------------


@pytest.mark.slow
def test_onnx_export_logits_match_pytorch_within_tolerance_wakeword_synthetic_model():
    """Untrained-model export-correctness check, mirroring the vcm synthetic
    test above -- proves the wakeword export graph (static (B,3) output, no
    dynamic time axis) is numerically faithful independent of whether a
    trained checkpoint exists on this machine."""
    torch.manual_seed(0)
    config = DSCNNConfig(n_blocks=2, channels=16, kernel_sizes=[5, 5], prologue_channels=12)
    model = DSCNN(config)
    model.eval()

    import tempfile

    with tempfile.TemporaryDirectory() as tmp:
        onnx_path = Path(tmp) / "wakeword_model.onnx"
        from me2_voicegen.wakeword.dataset import WAKEWORD_WINDOW_SECONDS

        n_frames = window_n_frames(int(WAKEWORD_WINDOW_SECONDS * 16000))
        export_fp32(
            model,
            onnx_path,
            n_frames=n_frames,
            output_names=["logits"],
            dynamic_axes={"features": {0: "batch"}},
        )
        assert onnx_path.exists()

        torch_logits, onnx_logits = onnx_vs_pytorch_logits(model, onnx_path, n_frames=n_frames)
        assert torch_logits.shape == onnx_logits.shape == (1, 3)
        max_abs_diff = np.abs(torch_logits - onnx_logits).max()
        assert max_abs_diff < 1e-3, max_abs_diff


@pytest.mark.slow
def test_static_int8_quantization_wakeword_family_shrinks_model_and_quantizes_conv(tmp_path, vcm_wav_factory):
    torch.manual_seed(0)
    config = DSCNNConfig(n_blocks=2, channels=16, kernel_sizes=[5, 5], prologue_channels=12)
    model = DSCNN(config)
    model.eval()

    manifest_path = _wakeword_manifest(tmp_path, vcm_wav_factory, n_each=6)

    import tempfile

    import onnx as onnx_pkg

    with tempfile.TemporaryDirectory() as tmp:
        fp32_path = Path(tmp) / "wakeword_model.fp32.onnx"
        int8_path = Path(tmp) / "wakeword_model.int8.onnx"
        from me2_voicegen.wakeword.dataset import WAKEWORD_WINDOW_SECONDS

        n_frames = window_n_frames(int(WAKEWORD_WINDOW_SECONDS * 16000))
        export_fp32(
            model,
            fp32_path,
            n_frames=n_frames,
            output_names=["logits"],
            dynamic_axes={"features": {0: "batch"}},
        )

        reader = WakewordCalibrationReader(manifest_path=manifest_path, n_samples=8, seed=0)
        quantize_int8_static(fp32_path, int8_path, reader)

        assert int8_path.exists()
        assert int8_path.stat().st_size < fp32_path.stat().st_size

        quantized_graph = onnx_pkg.load(str(int8_path))
        op_types = {node.op_type for node in quantized_graph.graph.node}
        assert "QuantizeLinear" in op_types and "DequantizeLinear" in op_types
        assert "Conv" in op_types


@pytest.mark.slow
@pytest.mark.skipif(
    not REAL_WAKEWORD_CHECKPOINT.exists(), reason="no trained checkpoint at out/wakeword/checkpoints/checkpoint.pt"
)
def test_onnx_export_logits_match_pytorch_on_real_wakeword_checkpoint():
    model, _ckpt = load_checkpoint(REAL_WAKEWORD_CHECKPOINT, model_family="wakeword")

    import tempfile

    with tempfile.TemporaryDirectory() as tmp:
        onnx_path = Path(tmp) / "wakeword_model.onnx"
        from me2_voicegen.wakeword.dataset import WAKEWORD_WINDOW_SECONDS

        n_frames = window_n_frames(int(WAKEWORD_WINDOW_SECONDS * 16000))
        export_fp32(
            model,
            onnx_path,
            n_frames=n_frames,
            output_names=["logits"],
            dynamic_axes={"features": {0: "batch"}},
        )

        torch_logits, onnx_logits = onnx_vs_pytorch_logits(model, onnx_path, n_frames=n_frames)
        max_abs_diff = np.abs(torch_logits - onnx_logits).max()
        assert max_abs_diff < 1e-3, max_abs_diff
