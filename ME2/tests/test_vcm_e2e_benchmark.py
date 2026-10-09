"""Loader/export/e2e-bench tests for the QuartzNet path (all use the real
checkpoint from `vcm_quartznet_checkpoint`, produced by `vcm.train.main`).
ONNX-touching tests are `slow`, matching test_vcm_export.py's convention."""

from __future__ import annotations

import dataclasses
import json
import os

import numpy as np
import pytest
import torch

from me2_voicegen.vcm.export_onnx import DEFAULT_DYNAMIC_AXES, export_fp32
from me2_voicegen.vcm.export_onnx import load_checkpoint as export_load_checkpoint
from me2_voicegen.vcm.model import PRESETS, build_model
from me2_voicegen.vcm.pipeline import load_checkpoint
from me2_voicegen.vcm.quartznet import QuartzNetCTC


@pytest.mark.parametrize("weights_only", [False, True])
def test_pipeline_loader_builds_quartznet(vcm_quartznet_checkpoint, weights_only):
    path, _, _ = vcm_quartznet_checkpoint
    model, ckpt = load_checkpoint(path, weights_only=weights_only)
    assert isinstance(model, QuartzNetCTC) and ckpt["model_type"] == "quartznet"
    assert not model.training


def test_export_loader_builds_quartznet(vcm_quartznet_checkpoint):
    model, _ = export_load_checkpoint(vcm_quartznet_checkpoint[0])
    assert isinstance(model, QuartzNetCTC)


def _legacy_ckpt(tmp_path, with_key=None):
    model = build_model("default")
    ckpt = {"model_state_dict": model.state_dict(), "config": dataclasses.asdict(PRESETS["default"])}
    if with_key is not None:
        ckpt["model_type"] = with_key
    path = tmp_path / "legacy.pt"
    torch.save(ckpt, path)
    return path


@pytest.mark.parametrize("weights_only", [False, True])
def test_legacy_checkpoint_without_key_loads_as_matchbox(tmp_path, weights_only):
    path = _legacy_ckpt(tmp_path)
    assert type(load_checkpoint(path, weights_only=weights_only)[0]).__name__ == "MatchboxNetCTC"
    assert type(export_load_checkpoint(path)[0]).__name__ == "MatchboxNetCTC"


def test_unknown_model_type_rejected(tmp_path):
    path = _legacy_ckpt(tmp_path, with_key="resnet")
    with pytest.raises(ValueError, match="unknown model_type"):
        load_checkpoint(path)
    with pytest.raises(ValueError, match="unknown model_type"):
        export_load_checkpoint(path)


@pytest.mark.slow
def test_fp32_export_parity_and_time_out_axis(vcm_quartznet_checkpoint, tmp_path):
    import onnx
    import onnxruntime as ort

    model, _ = load_checkpoint(vcm_quartznet_checkpoint[0])
    onnx_path = export_fp32(model, tmp_path / "q.onnx")
    graph = onnx.load(str(onnx_path)).graph
    assert graph.output[0].type.tensor_type.shape.dim[1].dim_param == "time_out"
    ops = {n.op_type for n in graph.node}
    assert "Dropout" not in ops
    session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    for t in (37, 151, 251):
        x = torch.randn(1, 40, t)
        with torch.no_grad():
            expected = model(x).numpy()
        (got,) = session.run(None, {"features": x.numpy()})
        assert got.shape == expected.shape and got.shape[1] == -(-t // 2)
        assert np.abs(got - expected).max() <= 1e-4


@pytest.mark.slow
def test_matchbox_export_keeps_default_axes(tmp_path):
    import onnx

    p = export_fp32(build_model("default").eval(), tmp_path / "m.onnx")
    assert DEFAULT_DYNAMIC_AXES["logits"] == {0: "batch", 1: "time"}
    dim = onnx.load(str(p)).graph.output[0].type.tensor_type.shape.dim[1]
    assert dim.dim_param == "time"


@pytest.mark.slow
def test_benchmark_export_then_e2e_bench_is_read_only(vcm_quartznet_checkpoint, tmp_path):
    from me2_voicegen.vcm import benchmark, e2e_benchmark
    from me2_voicegen.vcm.streaming.backends import OnnxBackend

    ckpt, manifest, run_dir = vcm_quartznet_checkpoint
    benchmark.main(
        ["--checkpoint", str(ckpt), "--out-dir", str(run_dir), "--manifest", str(manifest),
         "--calibration-samples", "4", "--n-iters", "3"]
    )
    int8 = run_dir / "export" / "vcm_model.int8.onnx"
    assert int8.stat().st_size <= 1_048_576

    backend = OnnxBackend(int8)
    assert backend.logp_for_waveform(np.zeros(int(2.5 * 16000), dtype=np.float32)).shape == (126, 29)

    def snapshot():
        return {str(p): p.stat().st_mtime_ns for p in run_dir.rglob("*") if p.is_file()}

    before = snapshot()
    out = tmp_path / "e2e.json"
    e2e_benchmark.main(
        ["--run-dir", str(run_dir), "--manifest", str(manifest), "--split", "val",
         "--n-clips", "3", "--window-s", "1.0", "--grammar", "optionb", "--out", str(out)]
    )
    assert snapshot() == before
    data = json.loads(out.read_text())
    assert data["int8"]["t_out"] == 51 and data["fp32"]["t_out"] == 51
    assert 0.0 <= data["int8_fp32_intent_agreement"] <= 1.0 and data["n_clips"] == 3
