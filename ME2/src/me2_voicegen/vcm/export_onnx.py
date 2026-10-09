"""Export a trained `vcm.model` checkpoint (Task 04) to ONNX, with a
dynamic time axis, and quantize it to static INT8.

Per docs/VCM-CONTRACT.md section 5, this module's dummy/calibration inputs
are always `(B, 40, T)` log-mel features built from `common.features` -- the
same front-end training/eval use -- so no train/infer skew is introduced
here.

Static (not dynamic) quantization is used deliberately: `onnxruntime`'s
`quantize_dynamic` is known to often leave `Conv` layers in fp32, which
would defeat the purpose for this 1D-CNN model (see ticket
.scratch/vcm-toy/tickets/06-onnx-export-benchmark.md's "Unknowns"). Static
quantization needs a calibration data reader, built here from a small
sample of the real `val` split.

`--model-family {vcm,wakeword}` (feature `wakeword-dscnn`,
feature-engineering/wakeword-dscnn/SPEC.md): this module is deliberately
shared across both models rather than duplicated into a `wakeword/`
copy -- export/quantize mechanics (`export_fp32`, `quantize_int8_static`,
`onnx_vs_pytorch_logits`) are model-architecture-agnostic and are reused
unchanged; only checkpoint-loading and the export graph's output shape
differ (`vcm`'s CTC output has a dynamic time axis, `wakeword`'s 3-way
classifier output is static `(B, 3)`). `family="wakeword"` branches import
`me2_voicegen.wakeword.*` lazily (inside functions, not at module import
time) so this module gains no hard top-level dependency on `wakeword`, and
every `family="vcm"` (the default) code path is unchanged from before this
feature -- see SPEC.md's Assumptions for the test backing that claim.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from me2_voicegen.vcm.dataset import VCMDataset
from me2_voicegen.common.features import SAMPLE_RATE, LogMelFeatureExtractor
from me2_voicegen.vcm.model import MODEL_TYPE_KEY, build_model_from_config

WINDOW_SECONDS = 1.5
WINDOW_SAMPLES = int(WINDOW_SECONDS * SAMPLE_RATE)

DEFAULT_MANIFEST = (
    Path(__file__).resolve().parents[3] / "out" / "conversions" / "v2" / "test_set" / "manifest.csv"
)


def load_checkpoint(
    checkpoint_path: str | Path, model_family: str = "vcm"
) -> tuple[torch.nn.Module, dict]:
    ckpt = torch.load(checkpoint_path, map_location="cpu")
    if model_family == "vcm":
        model = build_model_from_config(ckpt["config"], ckpt.get(MODEL_TYPE_KEY))
    elif model_family == "wakeword":
        from me2_voicegen.wakeword.model import DSCNN, DSCNNConfig

        config = DSCNNConfig(**ckpt["config"])
        model = DSCNN(config)
    else:
        raise ValueError(f"unknown model_family {model_family!r}, expected 'vcm' or 'wakeword'")
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model, ckpt


def window_n_frames(window_samples: int = WINDOW_SAMPLES) -> int:
    """Number of log-mel frames `common.features` produces for a
    `window_samples`-sample waveform (151 for the default 1.5s/24000-sample
    window at the real 16kHz/480/160 front-end parameters)."""
    extractor = LogMelFeatureExtractor()
    return int(extractor(torch.zeros(window_samples)).shape[-1])


def dummy_features(n_frames: int | None = None, batch: int = 1) -> torch.Tensor:
    """`(batch, 40, n_frames)` log-mel-shaped tensor for tracing/benchmarking.

    Content is arbitrary (fixed seed for reproducibility) -- only the shape
    and dtype need to match a real `(B, 40, T)` log-mel batch (docs/
    VCM-CONTRACT.md section 5) for export/benchmark purposes.
    """
    if n_frames is None:
        n_frames = window_n_frames()
    generator = torch.Generator().manual_seed(0)
    return torch.randn(batch, 40, n_frames, generator=generator)


DEFAULT_DYNAMIC_AXES = {
    "features": {0: "batch", 2: "time"},
    "logits": {0: "batch", 1: "time"},
}
"""vcm's CTC output shape: `(B, T, alphabet_size)`, time axis dynamic same
as the input. `export_fp32`'s default when `dynamic_axes` is omitted --
preserved unchanged so every pre-existing `export_fp32(model, path, ...)`
call site keeps today's exact export graph. `wakeword`'s call site passes
its own dict instead (`{"features": {0: "batch"}}` -- fixed window length,
static `(B, 3)` classifier output, no time axis anywhere)."""

STRIDED_DYNAMIC_AXES = {
    "features": {0: "batch", 2: "time"},
    "logits": {0: "batch", 1: "time_out"},
}
"""Axes for a temporally-subsampled model: the output time axis gets its own
symbol so ONNX shape inference does not assume it equals the input's."""


def export_fp32(
    model: torch.nn.Module,
    out_path: str | Path,
    n_frames: int | None = None,
    output_names: list[str] | None = None,
    dynamic_axes: dict | None = None,
) -> Path:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    model.eval()
    dummy = dummy_features(n_frames)
    torch.onnx.export(
        model,
        dummy,
        str(out_path),
        input_names=["features"],
        output_names=output_names if output_names is not None else ["logits"],
        dynamic_axes=dynamic_axes
        if dynamic_axes is not None
        else (STRIDED_DYNAMIC_AXES if getattr(model, "total_stride", 1) != 1 else DEFAULT_DYNAMIC_AXES),
        opset_version=17,
        do_constant_folding=True,
    )
    return out_path


class HeadsOnly(torch.nn.Module):
    """Encoder + attention pooling + intent/slot heads, without the CTC output layer (the classifier half of the hybrid).
    Outputs `intent_logits` (B, 21) and one `slot_<INTENT>` (B, n_values) per slotted intent, in `SLOT_INTENTS` order.
    Pools over every frame (no lengths), the same call `vcm.hybrid` and `vcm.semantic_eval` make."""

    def __init__(self, model: torch.nn.Module) -> None:
        super().__init__()
        from me2_voicegen.vcm.semantic_labels import SLOT_INTENTS, slot_head_name

        self.model = model
        self.slot_keys = [slot_head_name(i) for i in SLOT_INTENTS]
        self.output_names = ["intent_logits"] + [f"slot_{i}" for i in SLOT_INTENTS]

    def forward(self, features: torch.Tensor):
        enc = self.model._encode(features)
        pooled, _ = self.model.pool(enc, None)
        return (self.model.intent_head(pooled), *(self.model.slot_heads[k](pooled) for k in self.slot_keys))


def export_heads_fp32(model: torch.nn.Module, out_path: str | Path, n_frames: int | None = None) -> Path:
    wrapper = HeadsOnly(model).eval()
    axes = {"features": {0: "batch", 2: "time"}, **{n: {0: "batch"} for n in wrapper.output_names}}
    return export_fp32(wrapper, out_path, n_frames=n_frames, output_names=wrapper.output_names, dynamic_axes=axes)


class ValSplitCalibrationReader:
    """`onnxruntime.quantization.CalibrationDataReader`-compatible reader
    sourced from the real `val` split (per this ticket's Unknowns section:
    static quantization needs real calibration data, not synthetic noise).

    Duck-typed rather than subclassing `CalibrationDataReader` directly --
    it only needs `get_next()`/`rewind()`, and duck-typing keeps this
    importable/testable without onnxruntime.quantization on the fast-test
    path.
    """

    def __init__(
        self,
        manifest_path: str | Path = DEFAULT_MANIFEST,
        audio_root: str | Path | None = None,
        n_samples: int = 32,
        window_samples: int = WINDOW_SAMPLES,
        seed: int = 0,
    ) -> None:
        dataset = VCMDataset(manifest_path, audio_root=audio_root, split="val", augmenter=None)
        extractor = LogMelFeatureExtractor()

        rng = np.random.default_rng(seed)
        n = len(dataset)
        indices = rng.permutation(n)[: min(n_samples, n)].tolist()

        self._batches: list[dict[str, np.ndarray]] = []
        for idx in indices:
            waveform = dataset[idx].waveform
            if waveform.shape[-1] < window_samples:
                waveform = torch.nn.functional.pad(waveform, (0, window_samples - waveform.shape[-1]))
            else:
                waveform = waveform[:window_samples]
            feats = extractor(waveform).unsqueeze(0).numpy().astype(np.float32)
            self._batches.append({"features": feats})
        self._iter = iter(self._batches)

    def get_next(self) -> dict[str, np.ndarray] | None:
        return next(self._iter, None)

    def rewind(self) -> None:
        self._iter = iter(self._batches)


class WakewordCalibrationReader:
    """`ValSplitCalibrationReader`'s sibling for the wakeword DS-CNN,
    sourced from `wakeword.dataset.WakewordDataset` instead of `VCMDataset`
    -- reuses that dataset's own windowing (VAD-anchored crop, deterministic
    `center_window` since `split="val"`/`shift=False`) rather than a plain
    pad/truncate, so calibration data matches real eval-time preprocessing
    exactly. Duck-typed the same way `ValSplitCalibrationReader` is (only
    `get_next()`/`rewind()`), for the same reason: importable/testable
    without `onnxruntime.quantization` on the fast-test path."""

    def __init__(
        self,
        manifest_path: str | Path | None = None,
        n_samples: int = 32,
        seed: int = 0,
    ) -> None:
        from me2_voicegen.wakeword.dataset import WakewordDataset
        from me2_voicegen.wakeword.train import DEFAULT_MANIFEST as WAKEWORD_DEFAULT_MANIFEST

        resolved_manifest = manifest_path if manifest_path is not None else WAKEWORD_DEFAULT_MANIFEST
        dataset = WakewordDataset(resolved_manifest, split="val", augmenter=None, shift=False)
        extractor = LogMelFeatureExtractor()

        rng = np.random.default_rng(seed)
        n = len(dataset)
        indices = rng.permutation(n)[: min(n_samples, n)].tolist()

        self._batches: list[dict[str, np.ndarray]] = []
        for idx in indices:
            waveform = dataset[idx].waveform
            feats = extractor(waveform).unsqueeze(0).numpy().astype(np.float32)
            self._batches.append({"features": feats})
        self._iter = iter(self._batches)

    def get_next(self) -> dict[str, np.ndarray] | None:
        return next(self._iter, None)

    def rewind(self) -> None:
        self._iter = iter(self._batches)


def quantize_int8_static(
    fp32_path: str | Path,
    int8_path: str | Path,
    calibration_reader: ValSplitCalibrationReader,
) -> Path:
    from onnxruntime.quantization import QuantFormat, QuantType, quantize_static
    from onnxruntime.quantization.shape_inference import quant_pre_process

    fp32_path = Path(fp32_path)
    int8_path = Path(int8_path)
    preprocessed_path = fp32_path.with_suffix(".preprocessed.onnx")
    quant_pre_process(str(fp32_path), str(preprocessed_path))

    quantize_static(
        model_input=str(preprocessed_path),
        model_output=str(int8_path),
        calibration_data_reader=calibration_reader,
        quant_format=QuantFormat.QDQ,
        weight_type=QuantType.QInt8,
        activation_type=QuantType.QInt8,
        per_channel=False,
    )
    return int8_path


def onnx_vs_pytorch_logits(model: torch.nn.Module, onnx_path: str | Path, n_frames: int | None = None):
    """Run the same fixed input through the PyTorch model and an ONNX
    Runtime session over `onnx_path`, return `(torch_logits, onnx_logits)`
    as numpy arrays -- for export-correctness comparison."""
    import onnxruntime as ort

    model.eval()
    feats = dummy_features(n_frames)
    with torch.no_grad():
        torch_logits = model(feats).numpy()

    session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    (onnx_logits,) = session.run(None, {"features": feats.numpy()})
    return torch_logits, onnx_logits


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-family", choices=["vcm", "wakeword"], default="vcm")
    parser.add_argument("--checkpoint", type=Path, default=None, help="default: out/<family>/checkpoints/checkpoint.pt")
    parser.add_argument("--out-dir", type=Path, default=None, help="default: out/<family>")
    parser.add_argument("--manifest", type=Path, default=None, help="default: that family's own DEFAULT_MANIFEST")
    parser.add_argument("--calibration-samples", type=int, default=32)
    parser.add_argument("--heads-only", action="store_true",
                        help="vcm heads checkpoints: export encoder + intent/slot heads (no CTC output) as vcm_heads.{fp32,int8}.onnx")
    parser.add_argument(
        "--skip-quantization",
        action="store_true",
        help="export fp32 only; skip static INT8 quantization (fallback path)",
    )
    return parser


def _resolve_family_defaults(args: argparse.Namespace) -> None:
    """Fill in `--checkpoint`/`--out-dir`/`--manifest` per `--model-family`
    when the caller left them unset. `model_family="vcm"` (the default)
    resolves to exactly today's static defaults -- SPEC.md's Assumption
    that this refactor doesn't change the vcm path relies on this."""
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
    export_dir.mkdir(parents=True, exist_ok=True)

    model, ckpt = load_checkpoint(args.checkpoint, model_family=args.model_family)

    if args.model_family == "wakeword":
        from me2_voicegen.wakeword.dataset import WAKEWORD_WINDOW_SECONDS

        n_frames = window_n_frames(int(WAKEWORD_WINDOW_SECONDS * SAMPLE_RATE))
        model_prefix = "wakeword_model"
        export_kwargs: dict = {"output_names": ["logits"], "dynamic_axes": {"features": {0: "batch"}}}
        reader_cls = WakewordCalibrationReader
        reader_kwargs = {"manifest_path": args.manifest, "n_samples": args.calibration_samples}
    else:
        n_frames = window_n_frames()
        model_prefix = "vcm_model"
        export_kwargs = {}
        reader_cls = ValSplitCalibrationReader
        reader_kwargs = {"manifest_path": args.manifest, "n_samples": args.calibration_samples}

    if getattr(args, "heads_only", False):
        model_prefix = "vcm_heads"
    fp32_path = export_dir / f"{model_prefix}.fp32.onnx"
    if getattr(args, "heads_only", False):
        export_heads_fp32(model, fp32_path, n_frames=n_frames)
    else:
        export_fp32(model, fp32_path, n_frames=n_frames, **export_kwargs)
    print(f"exported fp32 ONNX -> {fp32_path} ({fp32_path.stat().st_size:,} bytes)")

    if args.skip_quantization:
        return

    int8_path = export_dir / f"{model_prefix}.int8.onnx"
    reader = reader_cls(**reader_kwargs)
    quantize_int8_static(fp32_path, int8_path, reader)
    print(f"exported static INT8 ONNX -> {int8_path} ({int8_path.stat().st_size:,} bytes)")


if __name__ == "__main__":
    main()
