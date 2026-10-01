"""`python -m me2_voicegen.vcm.streaming [flags]`: the spoken-command
streaming CLI. Wires `StreamingConfig` (registry/threshold resolution) and
`StreamingRunner` (loop modes) together; see
`ME2/docs/STREAMING-CONTRACT.md` for the shapes this consumes.

No argument opens the live microphone by default. The microphone is only
ever opened inside `main()`, immediately before the runner loop starts --
importing this module, building a parser, or resolving a `StreamingConfig`
never touches a device.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import sys
from pathlib import Path
from typing import Optional

from me2_voicegen.common.features import SAMPLE_RATE
from me2_voicegen.vcm.streaming.backends import InferenceBackend, OnnxBackend, TorchBackend
from me2_voicegen.vcm.streaming.config import (
    StreamingConfig,
    _candidate_run_dir,
    resolve_grammar,
    resolve_model,
    PERIOD_POLICIES,
    resolve_policy,
    resolve_threshold,
    resolve_wakeword_model,
)
from me2_voicegen.vcm.streaming.gate import (
    GateState,
    GateUnavailableError,
    ListeningGate,
    resolve_gate,
)
from me2_voicegen.vcm.streaming.policy import PolicyDecision
from me2_voicegen.vcm.streaming.runner import StreamingRunner
from me2_voicegen.vcm.streaming.sources import (
    DEFAULT_MIC_COMMAND,
    MicrophoneUnavailableError,
    WavFileSource,
    open_microphone_source,
)
from me2_voicegen.vcm.streaming.wakeword_gate import (
    WakewordInferenceBackend,
    WakewordOnnxBackend,
    WakewordTorchBackend,
)

LICENSE_NOTE = (
    "Checkpoint trained on out/conversions/v2/test_set/, which includes "
    "background_noise (ESC-50, CC-BY-NC-SA-4.0) in the silence bucket. Per "
    "docs/VCM-CONTRACT.md section 8, any checkpoint trained on this data "
    "inherits CC-BY-NC-SA-4.0: non-commercial use only, share-alike on "
    "redistribution."
)

_CONFIG_FIELD_NAMES = frozenset(f.name for f in dataclasses.fields(StreamingConfig))


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=None, help="StreamingConfig JSON file")
    parser.add_argument("--model", type=str, default=None)
    parser.add_argument("--backend", type=str, default=None, choices=["onnx", "torch"])
    parser.add_argument(
        "--onnx-variant", dest="onnx_variant", type=str, default=None, choices=["fp32", "int8"]
    )
    parser.add_argument("--ort-threads", dest="ort_threads", type=int, default=None)
    parser.add_argument("--policy", type=str, default=None)
    parser.add_argument("--grammar", type=str, default=None)
    parser.add_argument("--threshold", type=float, default=None)
    parser.add_argument(
        "--required-command-margin",
        dest="required_command_margin",
        type=float,
        default=None,
        help=(
            "incomplete-prefix rejection gate margin (raw unnormalized beam "
            "log-mass units; docs/INCOMPLETE-GRAMMAR-REJECTION.md). Omit to "
            "leave the gate disabled; --threshold above stays an independent "
            "gate"
        ),
    )
    parser.add_argument(
        "--score-mode",
        dest="score_mode",
        choices=("mean_frame", "per_char"),
        default=None,
        help=(
            "what the confidence threshold measures: 'mean_frame' (default; "
            "beam log mass / window frames) or 'per_char' (beam log mass / "
            "phrase length, duration-invariant). The two are on different "
            "scales: pick --threshold for the mode you choose (per_char "
            "thresholds are around -1, not -0.1)"
        ),
    )
    parser.add_argument("--window-s", dest="window_s", type=float, default=None)
    parser.add_argument("--stride-s", dest="stride_s", type=float, default=None)
    parser.add_argument("--refractory-s", dest="refractory_s", type=float, default=None)
    parser.add_argument("--beam-width", dest="beam_width", type=int, default=None)
    parser.add_argument("--block-s", dest="block_s", type=float, default=None)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument(
        "--source",
        type=str,
        default=None,
        help="'mic' (default) for the live microphone, or a path to a wav file to replay",
    )
    parser.add_argument("--mic-command", dest="mic_command", type=str, default=None)
    parser.add_argument(
        "--listen-for",
        dest="listen_for",
        type=float,
        default=None,
        help="stop after this many seconds of captured audio (realtime mode only)",
    )
    parser.add_argument(
        "--log-all-windows",
        dest="log_all_windows",
        action="store_true",
        default=None,
        help="emit a JSONL 'window' record for every evaluated window, not just triggers",
    )
    parser.add_argument(
        "--log-periods",
        dest="log_periods",
        action="store_true",
        default=None,
        help="print a per-period digest to stderr: gate open/close lines plus "
        "the period's consolidated result (rejected periods included) -- the "
        "JSONL on stdout is unchanged; use --log-all-windows for every window",
    )
    parser.add_argument("--gate", type=str, default=None, choices=["none", "spacebar", "wakeword"])
    parser.add_argument("--gate-period", dest="gate_period_s", type=float, default=None)
    # --policy endpointed: decode every stride inside the period, fire on the
    # first confident, stable, ended command; --gate-period is the time-out.
    parser.add_argument("--min-audio", dest="min_audio_s", type=float, default=None)
    parser.add_argument("--stable-strides", dest="stable_strides", type=int, default=None)
    parser.add_argument("--hold-ms", dest="hold_ms", type=float, default=None)
    parser.add_argument("--blank-floor", dest="blank_floor", type=float, default=None)
    parser.add_argument(
        "--wakeword-model",
        dest="wakeword_model",
        type=str,
        default=None,
        help="registry name / run dir / file for --gate wakeword's own model "
        "(independent of --model, which is the VCM decode model)",
    )
    parser.add_argument(
        "--wakeword-backend",
        dest="wakeword_backend",
        type=str,
        default=None,
        choices=["onnx", "torch"],
    )
    parser.add_argument(
        "--wakeword-onnx-variant",
        dest="wakeword_onnx_variant",
        type=str,
        default=None,
        choices=["fp32", "int8"],
    )
    parser.add_argument(
        "--wakeword-threshold",
        dest="wakeword_threshold",
        type=float,
        default=None,
        help="softmax probability of the _wakeword_ class --gate wakeword opens a period at",
    )
    return parser


def _cli_overrides(args: argparse.Namespace) -> dict:
    return {name: getattr(args, name) for name in _CONFIG_FIELD_NAMES if hasattr(args, name)}


def _print_period_event(
    event: str,
    samples_seen: Optional[int],
    decision: Optional[PolicyDecision],
    out: object = sys.stderr,
) -> None:
    """One `--log-periods` digest line (stderr only -- the JSONL on stdout
    is untouched): the gate lifecycle plus the period's consolidated
    result. `event` is "open"/"reopened" (samples_seen set, decision
    None), "closed" (the flush -- decision set), or "run_ended" (the run
    stopped with a period still open: the in-flight period is discarded,
    never flushed)."""
    if event == "run_ended":
        print("gate: closed    (run ended)", file=out)
        return
    t = f"t={samples_seen / SAMPLE_RATE:.2f}s"
    if event == "closed":
        verdict = "ACCEPT" if decision.accept else "REJECT"
        reason = decision.reason.removeprefix("mode_period: ")
        print(f"gate: closed    {t}  period: {verdict}  {reason}", file=out)
    else:  # "open" | "reopened"
        print(f"gate: {event:<9} {t}", file=out)


class _RunEndGateLogger:
    """`ListeningGate` pass-through that balances the digest when the run
    ends with a period still open (Ctrl-C/EOF): the in-flight period is
    discarded and never flushed, so without this line the digest would
    end on an unbalanced `gate: open`. A poll that never observed an open
    period prints nothing, matching the digest (no `open` was printed)."""

    def __init__(self, inner: ListeningGate, out: object = sys.stderr) -> None:
        self._inner = inner
        self._out = out
        self._was_open = False

    def poll(self, samples_seen: int, window=None) -> GateState:
        state = self._inner.poll(samples_seen, window)
        self._was_open = state.is_open
        return state

    def close(self) -> None:
        if self._was_open:
            _print_period_event("run_ended", None, None, out=self._out)
        self._inner.close()


def _load_checkpoint_meta(run_dir: Optional[Path]) -> dict:
    if run_dir is None:
        return {}
    report_path = Path(run_dir) / "metadata" / "eval_report.json"
    try:
        report = json.loads(report_path.read_text())
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return {}
    return report.get("checkpoint_meta") or {}


def _print_banner(
    cfg: StreamingConfig,
    model_path: Path,
    run_dir: Optional[Path],
    threshold: float,
    grammar_label: str,
    out: object = sys.stderr,
) -> None:
    meta = _load_checkpoint_meta(run_dir)
    preset = meta.get("preset", cfg.model)
    license_note = meta.get("license", LICENSE_NOTE)
    if cfg.gate == "none":
        gate_line = "listening gate: none"
    elif cfg.gate == "wakeword":
        gate_line = (
            f"listening gate: wakeword (backend={cfg.wakeword_backend} "
            f"threshold={cfg.wakeword_threshold}, opens a {cfg.gate_period_s} s "
            "listening period on detection)"
        )
    else:
        gate_line = (
            "listening gate: spacebar "
            f"(press SPACE to open a {cfg.gate_period_s} s listening period)"
        )
    lines = [
        "me2_voicegen streaming spoken-command runner",
        f"  checkpoint/run dir: {run_dir if run_dir is not None else model_path}",
        f"  preset: {preset}",
        f"  backend: {cfg.backend} (onnx variant: {cfg.onnx_variant})",
        f"  grammar: {cfg.grammar} ({grammar_label})",
        f"  window_s={cfg.window_s} stride_s={cfg.stride_s} "
        f"refractory_s={cfg.refractory_s} beam_width={cfg.beam_width}",
        f"  resolved threshold: {threshold}",
        f"  required_command_margin: {cfg.required_command_margin}",
        f"  score_mode: {cfg.score_mode}",
        f"  license: {license_note}",
        f"  {gate_line}",
    ]
    if cfg.log_periods:
        lines.append("  logging: per-period digest on stderr (--log-periods)")
    for line in lines:
        print(line, file=out)


def main(argv: Optional[list[str]] = None) -> None:
    args = build_arg_parser().parse_args(argv)
    cfg = StreamingConfig.merge(json_path=args.config, cli_overrides=_cli_overrides(args))

    # Hard --gate/--policy cross-validation: both must fire before any side
    # effect (no model load, no TTY raw mode, no microphone).
    if cfg.policy in PERIOD_POLICIES and cfg.gate == "none":
        raise SystemExit(
            f"--policy {cfg.policy} requires a listening gate: pass --gate "
            "spacebar (with --gate-period for the period length) -- --gate "
            "none is not a valid combination with --policy mode_period"
        )
    if cfg.gate != "none" and cfg.policy not in PERIOD_POLICIES:
        raise SystemExit(
            f"--gate {cfg.gate} opens a bounded listening period that only "
            "--policy mode_period, single_period or endpointed consumes: pass one, or "
            "use --gate none with --policy threshold"
        )
    if cfg.policy == "single_period" and cfg.gate_period_s != 3.0:
        raise SystemExit("--policy single_period requires --gate-period 3 (the evaluated full-inference duration)")

    gate: Optional[ListeningGate] = None
    if cfg.gate != "none":
        wakeword_backend: Optional[WakewordInferenceBackend] = None
        if cfg.gate == "wakeword":
            # Fail fast on a missing wakeword checkpoint/export before the
            # microphone is opened -- same slot/rationale as the spacebar
            # gate's GateUnavailableError handling below.
            wakeword_model_path = resolve_wakeword_model(
                cfg.wakeword_model, cfg.wakeword_backend, cfg.wakeword_onnx_variant
            )
            if cfg.wakeword_backend == "onnx":
                wakeword_backend = WakewordOnnxBackend(wakeword_model_path, ort_threads=cfg.ort_threads)
            else:
                wakeword_backend = WakewordTorchBackend(wakeword_model_path, device=cfg.device)
        # Fail fast on a non-interactive stdin (piped/CI) before the
        # microphone is opened -- mirrors MicrophoneUnavailableError.
        try:
            gate = resolve_gate(
                cfg.gate,
                period_s=cfg.gate_period_s,
                wakeword_backend=wakeword_backend,
                wakeword_threshold=cfg.wakeword_threshold,
            )
        except GateUnavailableError as exc:
            raise SystemExit(str(exc)) from None
        if cfg.log_periods:
            gate = _RunEndGateLogger(gate)

    model_path = resolve_model(cfg.model, cfg.backend, cfg.onnx_variant)
    run_dir = _candidate_run_dir(cfg.model)
    grammar, grammar_label = resolve_grammar(cfg.grammar)
    threshold = resolve_threshold(run_dir, override=cfg.threshold, grammar_label=grammar_label)
    policy = resolve_policy(
        cfg.policy,
        threshold,
        gate=gate,
        period_s=cfg.gate_period_s,
        on_period_event=_print_period_event if cfg.log_periods else None,
        endpoint_options={
            "window_s": cfg.window_s,
            "min_audio_s": cfg.min_audio_s,
            "stable_strides": cfg.stable_strides,
            "hold_ms": cfg.hold_ms,
            "blank_floor": cfg.blank_floor,
        },
    )

    backend: InferenceBackend
    if cfg.backend == "onnx":
        backend = OnnxBackend(model_path, ort_threads=cfg.ort_threads)
    else:
        backend = TorchBackend(model_path, device=cfg.device)

    _print_banner(cfg, model_path, run_dir, threshold, grammar_label)

    block_samples = max(1, int(round(cfg.block_s * SAMPLE_RATE)))

    if cfg.source == "mic":
        try:
            source = open_microphone_source(cfg.mic_command or DEFAULT_MIC_COMMAND, block_samples)
        except MicrophoneUnavailableError as exc:
            raise SystemExit(str(exc)) from None
    else:
        wav_path = Path(cfg.source)
        if not wav_path.exists():
            raise SystemExit(
                f"--source {cfg.source!r} is not 'mic' and does not exist as a wav "
                f"file. Pass --source <wav-path> for file replay, or 'mic' (with an "
                f"optional --mic-command) for the live microphone."
            )
        source = WavFileSource(wav_path, block_samples, realtime=False)

    runner = StreamingRunner(
        source=source,
        backend=backend,
        grammar=grammar,
        policy=policy,
        window_s=cfg.window_s,
        stride_s=cfg.stride_s,
        refractory_s=cfg.refractory_s,
        beam_width=cfg.beam_width,
        listen_for_s=cfg.listen_for,
        log_all_windows=cfg.log_all_windows,
        required_command_margin=cfg.required_command_margin,
        score_mode=cfg.score_mode,
    )
    try:
        runner.run()
    finally:
        # atexit inside SpacebarGate is only a backstop for failure paths
        # between construction and this try.
        if gate is not None:
            gate.close()


if __name__ == "__main__":
    main()
