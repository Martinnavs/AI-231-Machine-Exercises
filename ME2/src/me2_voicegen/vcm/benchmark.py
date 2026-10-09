"""CPU size/latency benchmark for the exported ONNX `vcm.model`.

**Every number this module reports is measured on this development node's
CPU (AMD EPYC 7742, 2x64 cores / 256 threads) and is explicitly labeled
"AMD EPYC 7742 (256-thread node) estimate". None of these numbers are a
Raspberry Pi 4/5 measurement -- no such hardware exists on this node.** The
original spec's RPi-targeted budgets (<=20ms/100ms-frame latency, <=5MB
INT8 model, <=25MB peak RAM) are reported alongside purely as indicative
context, never as a pass/fail claim about real RPi behavior.

`onnxruntime` is pinned to 1 intra-op thread and 1 inter-op thread for the
latency benchmark, per this task's ticket, so the number reflects
single-core throughput rather than this node's 256-way parallelism.

`--model-family {vcm,wakeword}` (feature `wakeword-dscnn`,
feature-engineering/wakeword-dscnn/SPEC.md): `benchmark_session`/
`_make_session`/`_render_markdown`/RSS-latency measurement are reused
unchanged -- they only ever touch an onnxruntime session + a frame count,
never a model class. `NOT_MEASURED_ON` already read "Raspberry Pi 4/5 (no
such hardware exists on this node)" before this feature existed, which
happens to already name wakeword's real target device (RPi 4, 2018, 4GB
RAM, confirmed by the user) -- unchanged; the wakeword result additionally
records that confirmed target under its own `target_device` key so the
distinction between "the label this module always used" and "the actual
confirmed target for this specific model" is explicit, not implied.
"""

from __future__ import annotations

import argparse
import json
import resource
import statistics
import time
from pathlib import Path

from me2_voicegen.vcm.export_onnx import (
    DEFAULT_MANIFEST,
    SAMPLE_RATE,
    WINDOW_SECONDS,
    ValSplitCalibrationReader,
    WakewordCalibrationReader,
    dummy_features,
    export_fp32,
    load_checkpoint,
    quantize_int8_static,
    window_n_frames,
)
from me2_voicegen.vcm.model import estimated_int8_bytes as vcm_estimated_int8_bytes

HARDWARE_LABEL = "AMD EPYC 7742 (256-thread node) estimate"
NOT_MEASURED_ON = "Raspberry Pi 4/5 (no such hardware exists on this node)"
WAKEWORD_TARGET_DEVICE = "Raspberry Pi 4 (2018), 4GB RAM (Broadcom BCM2711, quad-core Cortex-A72) -- confirmed target, not yet measured on this node"

SPEC_BUDGETS_INDICATIVE_ONLY = {
    "latency_ms_per_100ms_frame_le": 20.0,
    "int8_model_mb_le": 5.0,
    "peak_ram_mb_le": 25.0,
    "note": (
        "Original spec budgets for real RPi 4/5 hardware -- shown here as "
        "indicative context only, NOT as a pass/fail claim about this "
        f"node's numbers (which are all {HARDWARE_LABEL})."
    ),
}


def _make_session(model_path: str | Path, intra_op_threads: int = 1, inter_op_threads: int = 1):
    import onnxruntime as ort

    so = ort.SessionOptions()
    so.intra_op_num_threads = intra_op_threads
    so.inter_op_num_threads = inter_op_threads
    return ort.InferenceSession(str(model_path), sess_options=so, providers=["CPUExecutionProvider"])


def benchmark_session(session, n_frames: int, n_warmup: int = 10, n_iters: int = 100) -> dict:
    """p50/p95 per-window latency (ms) and RSS figures for `n_iters`
    single-window inferences, after `n_warmup` untimed warmup calls.

    `resource.getrusage(...).ru_maxrss` is a process-lifetime high-water
    mark (Linux, KB units), sampled *after* warmup -- i.e. after the model
    and the ORT session are already resident. Two figures are reported,
    deliberately not just one, per tech-lead review finding R3-2 (ticket
    06's Execution Log): a lone "delta" of 0 reads as "zero memory used,"
    which it is not -- it means inference itself never exceeded the
    already-large load-time peak, not that inference is free.
      - `process_peak_rss_kb`: the absolute high-water mark at the end of
        the timed loop -- the real total-footprint figure comparable to a
        peak-RAM budget.
      - `inference_rss_delta_kb`: growth in that high-water mark caused
        only by the timed inference loop itself (0 is common and expected
        here, since model/session load-time RSS usually dominates).
    """
    feats = dummy_features(n_frames).numpy()

    for _ in range(n_warmup):
        session.run(None, {"features": feats})

    rss_before_kb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    latencies_ms = []
    for _ in range(n_iters):
        t0 = time.perf_counter()
        session.run(None, {"features": feats})
        t1 = time.perf_counter()
        latencies_ms.append((t1 - t0) * 1000.0)
    rss_after_kb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss

    latencies_ms.sort()
    p50 = statistics.median(latencies_ms)
    p95_index = min(len(latencies_ms) - 1, round(0.95 * (len(latencies_ms) - 1)))
    p95 = latencies_ms[p95_index]

    return {
        "hardware_label": HARDWARE_LABEL,
        "not_measured_on": NOT_MEASURED_ON,
        "n_warmup": n_warmup,
        "n_iters": n_iters,
        "p50_latency_ms": p50,
        "p95_latency_ms": p95,
        "process_peak_rss_kb": rss_after_kb,
        "process_peak_rss_mb": rss_after_kb / 1024,
        "inference_rss_delta_kb": max(0, rss_after_kb - rss_before_kb),
    }


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-family", choices=["vcm", "wakeword"], default="vcm")
    parser.add_argument("--checkpoint", type=Path, default=None, help="default: out/<family>/checkpoints/checkpoint.pt")
    parser.add_argument("--out-dir", type=Path, default=None, help="default: out/<family>")
    parser.add_argument("--manifest", type=Path, default=None, help="default: that family's own DEFAULT_MANIFEST")
    parser.add_argument("--calibration-samples", type=int, default=32)
    parser.add_argument("--n-warmup", type=int, default=10)
    parser.add_argument("--n-iters", type=int, default=100)
    parser.add_argument(
        "--skip-quantization",
        action="store_true",
        help=(
            "fallback path (this ticket's Unknowns/fallback): report real "
            "fp32 ONNX size + measured fp32 latency, plus a param-count-"
            "derived INT8 SIZE ESTIMATE only (no INT8 latency claim)."
        ),
    )
    return parser


def _resolve_family_defaults(args: argparse.Namespace) -> None:
    """Mirrors `export_onnx._resolve_family_defaults` -- `model_family="vcm"`
    (the default) resolves to exactly today's static defaults."""
    if args.model_family == "wakeword":
        if args.checkpoint is None:
            args.checkpoint = Path("out/wakeword/checkpoints/checkpoint.pt")
        if args.out_dir is None:
            args.out_dir = Path("out/wakeword")
        if args.manifest is None:
            from me2_voicegen.wakeword.train import DEFAULT_MANIFEST as WAKEWORD_DEFAULT_MANIFEST

            args.manifest = WAKEWORD_DEFAULT_MANIFEST
    else:
        if args.checkpoint is None:
            args.checkpoint = Path("out/vcm/checkpoints/checkpoint.pt")
        if args.out_dir is None:
            args.out_dir = Path("out/vcm")
        if args.manifest is None:
            args.manifest = DEFAULT_MANIFEST


def main(argv: list[str] | None = None) -> None:
    args = build_arg_parser().parse_args(argv)
    _resolve_family_defaults(args)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    export_dir = args.out_dir / "export"
    metadata_dir = args.out_dir / "metadata"
    export_dir.mkdir(parents=True, exist_ok=True)
    metadata_dir.mkdir(parents=True, exist_ok=True)

    model, ckpt = load_checkpoint(args.checkpoint, model_family=args.model_family)

    if args.model_family == "wakeword":
        from me2_voicegen.wakeword.dataset import WAKEWORD_WINDOW_SECONDS
        from me2_voicegen.wakeword.model import estimated_int8_bytes as wakeword_estimated_int8_bytes

        window_seconds = WAKEWORD_WINDOW_SECONDS
        n_frames = window_n_frames(int(window_seconds * SAMPLE_RATE))
        model_prefix = "wakeword_model"
        export_kwargs: dict = {"output_names": ["logits"], "dynamic_axes": {"features": {0: "batch"}}}
        reader = WakewordCalibrationReader(manifest_path=args.manifest, n_samples=args.calibration_samples)
        estimate_fn = wakeword_estimated_int8_bytes
        extra_result_fields = {"target_device": WAKEWORD_TARGET_DEVICE}
    else:
        window_seconds = WINDOW_SECONDS
        n_frames = window_n_frames()
        model_prefix = "vcm_model"
        export_kwargs = {}
        reader = ValSplitCalibrationReader(manifest_path=args.manifest, n_samples=args.calibration_samples)
        estimate_fn = vcm_estimated_int8_bytes
        extra_result_fields = {}

    fp32_path = export_dir / f"{model_prefix}.fp32.onnx"
    export_fp32(model, fp32_path, n_frames=n_frames, **export_kwargs)
    fp32_size_bytes = fp32_path.stat().st_size

    fp32_session = _make_session(fp32_path)
    fp32_bench = benchmark_session(fp32_session, n_frames, args.n_warmup, args.n_iters)

    result: dict = {
        "hardware_label": HARDWARE_LABEL,
        "not_measured_on": NOT_MEASURED_ON,
        "window_seconds": window_seconds,
        "n_frames": n_frames,
        "onnxruntime_threads": {"intra_op": 1, "inter_op": 1},
        "checkpoint": str(args.checkpoint),
        "preset": ckpt.get("preset"),
        "fp32": {
            "onnx_size_bytes": fp32_size_bytes,
            "onnx_size_mb": fp32_size_bytes / (1024 * 1024),
            **fp32_bench,
        },
        "spec_budgets_indicative_only": SPEC_BUDGETS_INDICATIVE_ONLY,
        **extra_result_fields,
    }

    if args.skip_quantization:
        est_bytes = estimate_fn(model)
        result["int8"] = {
            "measured": False,
            "estimate_basis": (
                "param_count * 1 byte/param + 4 bytes/tensor scale -- SIZE "
                "ESTIMATE ONLY, not a measured quantized artifact; no "
                "latency estimate is reported for this fallback path."
            ),
            "estimated_size_bytes": est_bytes,
            "estimated_size_mb": est_bytes / (1024 * 1024),
        }
    else:
        int8_path = export_dir / f"{model_prefix}.int8.onnx"
        quantize_int8_static(fp32_path, int8_path, reader)
        int8_size_bytes = int8_path.stat().st_size

        int8_session = _make_session(int8_path)
        int8_bench = benchmark_session(int8_session, n_frames, args.n_warmup, args.n_iters)

        result["int8"] = {
            "measured": True,
            "onnx_size_bytes": int8_size_bytes,
            "onnx_size_mb": int8_size_bytes / (1024 * 1024),
            **int8_bench,
        }

    json_path = metadata_dir / f"{model_prefix.replace('_model', '')}_benchmark.json"
    with json_path.open("w") as f:
        json.dump(result, f, indent=2)

    md_path = metadata_dir / f"{model_prefix.replace('_model', '')}_benchmark.md"
    md_path.write_text(_render_markdown(result))

    print(f"wrote {json_path}")
    print(f"wrote {md_path}")
    print(json.dumps(result, indent=2))


def _render_markdown(result: dict) -> str:
    title = "Wakeword DS-CNN" if result.get("target_device") else "VCM"
    lines = [
        f"# {title} ONNX export/quantization benchmark",
        "",
        f"**Hardware: {result['hardware_label']}. NOT a {result['not_measured_on']} measurement.**",
        "",
        f"Window: {result['window_seconds']}s ({result['n_frames']} log-mel frames). "
        f"onnxruntime threads: intra_op={result['onnxruntime_threads']['intra_op']}, "
        f"inter_op={result['onnxruntime_threads']['inter_op']}. "
        f"Checkpoint: `{result['checkpoint']}` (preset={result['preset']}).",
    ]
    if result.get("target_device"):
        lines.append(f"**Confirmed target device: {result['target_device']}.**")
    lines += [
        "",
        "| variant | size (MB) | p50 latency (ms) | p95 latency (ms) | process peak RSS (MB) | inference-only RSS delta (KB) | note |",
        "|---|---|---|---|---|---|---|",
    ]
    fp32 = result["fp32"]
    lines.append(
        f"| fp32 ONNX | {fp32['onnx_size_mb']:.3f} | {fp32['p50_latency_ms']:.3f} | "
        f"{fp32['p95_latency_ms']:.3f} | {fp32['process_peak_rss_mb']:.3f} | "
        f"{fp32['inference_rss_delta_kb']} | {result['hardware_label']} |"
    )
    int8 = result["int8"]
    if int8.get("measured"):
        lines.append(
            f"| INT8 ONNX (static, val-calibrated) | {int8['onnx_size_mb']:.3f} | "
            f"{int8['p50_latency_ms']:.3f} | {int8['p95_latency_ms']:.3f} | "
            f"{int8['process_peak_rss_mb']:.3f} | {int8['inference_rss_delta_kb']} | "
            f"{result['hardware_label']} |"
        )
    else:
        lines.append(
            f"| INT8 (size ESTIMATE, no measured artifact) | {int8['estimated_size_mb']:.3f} | "
            "n/a | n/a | n/a | n/a | param-count-derived estimate, not measured |"
        )

    lines += [
        "",
        "**Reading the RSS columns (tech-lead review R3-2):** \"process peak RSS\" is "
        "the absolute `ru_maxrss` high-water mark at the end of the timed loop -- the "
        "real total-footprint figure to compare against the peak-RAM budget below. "
        "\"inference-only RSS delta\" is only the *growth* in that high-water mark "
        "caused by the timed inference loop itself, sampled after the model/session "
        "were already loaded; it is commonly 0 because load-time RSS (model weights, "
        "ORT session, allocator arenas) usually already exceeds anything a single "
        "window's inference allocates. A `0` there means \"inference didn't push the "
        "high-water mark higher,\" not \"inference used no memory.\"",
    ]

    budgets = result["spec_budgets_indicative_only"]
    lines += [
        "",
        "## Original spec budgets (indicative only, not a pass/fail claim about real RPi behavior)",
        "",
        f"- latency <= {budgets['latency_ms_per_100ms_frame_le']} ms per 100ms frame",
        f"- INT8 model size <= {budgets['int8_model_mb_le']} MB",
        f"- peak RAM <= {budgets['peak_ram_mb_le']} MB (compare against the \"process peak RSS\" column above, not the RSS delta)",
        "",
        budgets["note"],
        "",
    ]
    return "\n".join(lines)


if __name__ == "__main__":
    main()
