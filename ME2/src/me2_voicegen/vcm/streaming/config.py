"""Streaming configuration: registry-based model/policy resolution, JSON+CLI
precedence merge, and operating-threshold resolution from a run dir's
`eval_report.json`. See `ME2/docs/STREAMING-CONTRACT.md` section 2 for the
`StreamingConfig` field list this module is the source of truth for.

SECURITY -- trust boundary, read before wiring a CLI to this module:
`--model` (and `resolve_model`'s return value) is later handed to one of two
code-execution-capable parsers: `torch.load` (pickle deserialization --
`vcm.pipeline.load_checkpoint`, see its `weights_only` kwarg) for
`--backend torch`, or ONNX Runtime's own model parser
(`backends.OnnxBackend`) for `--backend onnx`. An `.onnx` file is just as
much untrusted input to ORT as a `.pt` file is to `torch.load` -- neither
format is safe to point at an artifact you didn't produce or don't trust.

This is deliberately NOT run through `vcm.optionb.fetch_dataset.resolve_under`
-style path-containment: `--model` is the operator's own deliberate CLI
argument (unlike that module's untrusted-upstream-manifest paths), and
confining it under `out/` would break the "point this at any
checkpoint/export you trust" requirement this registry exists to support.
The real control is at the parser boundary (see `backends.py`), not at the
path-resolution layer here -- only ever point `--model` at a checkpoint or
`.onnx` file you produced yourself or otherwise trust.

A `--config` JSON file is NOT a passive data file in the "safe to share/
copy from a README" sense -- `StreamingConfig.from_json` can set `model`
(the trust boundary above) and `mic_command` (an argv `MicrophoneSource`
spawns via `subprocess.Popen`, `shell=False` -- see
`vcm.streaming.sources`'s own trust-boundary note) exactly as freely as
the equivalent CLI flags can. A JSON config file must be trusted at the
same level as the command line itself, not treated as inert configuration;
only load a `--config` file you wrote yourself or otherwise trust.
"""

from __future__ import annotations

import dataclasses
import json
import sys
from pathlib import Path
from typing import Any, Mapping, Optional

from me2_voicegen.vcm.evaluate import GRAMMAR_REGISTRY
from me2_voicegen.vcm.streaming import wakeword_gate
from me2_voicegen.vcm.streaming.gate import DEFAULT_WAKEWORD_THRESHOLD, GATE_REGISTRY, ListeningGate
from me2_voicegen.vcm.streaming.policy import (
    AcceptancePolicy,
    EndpointedPeriodPolicy,
    ModePeriodPolicy,
    SinglePeriodPolicy,
    PeriodEventCallback,
    ThresholdPolicy,
)

# "wakeword" is registered here rather than in gate.py itself: wakeword_gate.py
# imports ListeningGate/GateState/DEFAULT_WAKEWORD_THRESHOLD from gate.py, so
# gate.py importing wakeword_gate.py back would be a cycle. This module is
# where the other registries (MODEL_REGISTRY, POLICY_REGISTRY) already live,
# so it's where this registration lives too.
GATE_REGISTRY["wakeword"] = wakeword_gate.WakeWordGate

PROJECT_ROOT = Path(__file__).resolve().parents[4]

MODEL_REGISTRY: dict[str, Path] = {
    "optionc": PROJECT_ROOT / "out" / "vcm" / "optionb-optionc",
    "default": PROJECT_ROOT / "out" / "vcm" / "optionb",
}

WAKEWORD_MODEL_REGISTRY: dict[str, Path] = {
    "default": PROJECT_ROOT / "out" / "wakeword",
}

# Policies that consume a listening gate's bounded period.
PERIOD_POLICIES = ("mode_period", "single_period", "endpointed")

POLICY_REGISTRY: dict[str, type[AcceptancePolicy]] = {
    "threshold": ThresholdPolicy,
    "mode_period": ModePeriodPolicy,
    "single_period": SinglePeriodPolicy,
    "endpointed": EndpointedPeriodPolicy,
}

DEFAULT_GRAMMAR_LABEL = "OPTIONB_GRAMMAR"

# Used only when `<run_dir>/metadata/eval_report.json` cannot be read at all
# (missing file, malformed JSON, or no matching grammar section) -- never
# the primary path. Deliberately NOT the real checked-in operating point
# (-0.1): a mean-log-prob-per-frame threshold of 0.0 makes `ThresholdPolicy`
# accept almost nothing, which is the safe direction to fail in when the
# real, audited operating point couldn't be resolved. Pass `--threshold`
# explicitly to override.
FALLBACK_THRESHOLD = 0.0

# Mirrors the type/choice constraints `__main__.build_arg_parser`'s argparse
# flags already enforce on the CLI -- JSON-supplied config (`from_json`)
# bypasses argparse entirely, so this is applied by hand at the same
# dataclass boundary rather than letting an unvalidated JSON value (e.g. a
# string where `ort_threads` needs an int, or an `onnx_variant` outside the
# CLI's `["fp32", "int8"]` choices) reach ORT/torch downstream.
_FIELD_TYPES: dict[str, type] = {
    "model": str,
    "backend": str,
    "onnx_variant": str,
    "ort_threads": int,
    "policy": str,
    "grammar": str,
    "threshold": float,
    "window_s": float,
    "stride_s": float,
    "refractory_s": float,
    "beam_width": int,
    "block_s": float,
    "device": str,
    "source": str,
    "realtime": bool,
    "mic_command": str,
    "listen_for": float,
    "log_all_windows": bool,
    "gate": str,
    "gate_period_s": float,
    "log_periods": bool,
    "required_command_margin": float,
    "score_mode": str,
    "wakeword_model": str,
    "wakeword_backend": str,
    "wakeword_onnx_variant": str,
    "wakeword_threshold": float,
    "min_audio_s": float,
    "stable_strides": int,
    "hold_ms": float,
    "blank_floor": float,
}

_FIELD_CHOICES: dict[str, tuple[str, ...]] = {
    "backend": ("onnx", "torch"),
    "onnx_variant": ("fp32", "int8"),
    "gate": ("none", "spacebar", "wakeword"),
    "wakeword_backend": ("onnx", "torch"),
    "wakeword_onnx_variant": ("fp32", "int8"),
    "score_mode": ("mean_frame", "per_char"),
}

_NULLABLE_FIELDS = frozenset(
    {"threshold", "mic_command", "listen_for", "required_command_margin"}
)


def _coerce_field_value(name: str, value: Any) -> Any:
    if value is None:
        if name in _NULLABLE_FIELDS:
            return None
        raise SystemExit(f"StreamingConfig field {name!r} must not be null")

    expected = _FIELD_TYPES[name]

    if expected is bool:
        if not isinstance(value, bool):
            raise SystemExit(
                f"StreamingConfig field {name!r} must be a bool, got {value!r}"
            )
    elif expected is int:
        # bool is an int subclass in Python -- reject it explicitly so
        # `{"ort_threads": true}` doesn't silently become 1.
        if isinstance(value, bool):
            raise SystemExit(f"StreamingConfig field {name!r} must be an int, got {value!r}")
        if isinstance(value, int):
            pass
        elif isinstance(value, str):
            try:
                value = int(value)
            except ValueError:
                raise SystemExit(
                    f"StreamingConfig field {name!r} must be an int, got {value!r}"
                ) from None
        else:
            raise SystemExit(f"StreamingConfig field {name!r} must be an int, got {value!r}")
    elif expected is float:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise SystemExit(
                f"StreamingConfig field {name!r} must be a number, got {value!r}"
            )
        value = float(value)
    elif expected is str:
        if not isinstance(value, str):
            raise SystemExit(f"StreamingConfig field {name!r} must be a string, got {value!r}")

    if name in _FIELD_CHOICES and value not in _FIELD_CHOICES[name]:
        raise SystemExit(
            f"StreamingConfig field {name!r}={value!r} must be one of "
            f"{_FIELD_CHOICES[name]}"
        )
    return value


@dataclasses.dataclass(frozen=True)
class StreamingConfig:
    model: str = "optionc"
    backend: str = "onnx"
    onnx_variant: str = "fp32"
    ort_threads: int = 1
    policy: str = "threshold"
    grammar: str = "optionb"
    threshold: Optional[float] = None
    window_s: float = 2.5
    stride_s: float = 0.25
    refractory_s: float = 1.5
    beam_width: int = 25
    block_s: float = 0.064
    device: str = "cpu"
    source: str = "mic"
    realtime: bool = True
    mic_command: Optional[str] = None
    listen_for: Optional[float] = None
    log_all_windows: bool = False
    gate: str = "none"
    gate_period_s: float = 5.0
    log_periods: bool = False
    required_command_margin: Optional[float] = None
    score_mode: str = "mean_frame"
    wakeword_model: str = "default"
    wakeword_backend: str = "torch"
    wakeword_onnx_variant: str = "fp32"
    wakeword_threshold: float = DEFAULT_WAKEWORD_THRESHOLD
    # --policy endpointed only (gate_period_s is its time-out)
    min_audio_s: float = 0.3
    stable_strides: int = 2
    hold_ms: float = 300.0
    blank_floor: float = 0.9

    @classmethod
    def from_json(cls, path: str | Path) -> "StreamingConfig":
        raw_path = Path(path)
        try:
            data: dict[str, Any] = json.loads(raw_path.read_text())
        except FileNotFoundError:
            raise SystemExit(f"--config {raw_path} does not exist") from None
        except json.JSONDecodeError as exc:
            raise SystemExit(f"--config {raw_path} is not valid JSON: {exc}") from None

        valid_fields = {f.name for f in dataclasses.fields(cls)}
        unknown = sorted(set(data) - valid_fields)
        if unknown:
            raise SystemExit(
                f"unknown StreamingConfig field(s) in {raw_path}: {unknown}; "
                f"valid fields are {sorted(valid_fields)}"
            )
        coerced = {name: _coerce_field_value(name, value) for name, value in data.items()}
        return cls(**coerced)

    @classmethod
    def merge(
        cls,
        json_path: str | Path | None = None,
        cli_overrides: Mapping[str, Any] | None = None,
    ) -> "StreamingConfig":
        """Precedence: dataclass defaults < JSON config < explicit CLI
        flags. `cli_overrides` must only contain keys the caller explicitly
        set on the command line (e.g. argparse defaults left at `None` so
        "not passed" is distinguishable from "passed and equal to the
        dataclass default") -- values of `None` in `cli_overrides` are
        treated as "not explicitly set" and skipped.
        """
        base = cls.from_json(json_path) if json_path is not None else cls()
        overrides = {k: v for k, v in (cli_overrides or {}).items() if v is not None}
        unknown = sorted(set(overrides) - {f.name for f in dataclasses.fields(cls)})
        if unknown:
            raise SystemExit(f"unknown StreamingConfig CLI override(s): {unknown}")
        return dataclasses.replace(base, **overrides)


def _candidate_run_dir(model: str, *, registry: dict[str, Path] = MODEL_REGISTRY) -> Optional[Path]:
    """Best-effort run-dir for threshold resolution. Registry name -> its
    run dir. Existing directory -> itself. A direct file living under
    `<run_dir>/checkpoints/` or `<run_dir>/export/` -> its run dir. Anything
    else (e.g. a bare file with no recognizable run-dir layout) -> `None`,
    which `resolve_threshold` treats as "no report to read". `registry`
    defaults to `MODEL_REGISTRY` (VCM); `resolve_model` passes
    `WAKEWORD_MODEL_REGISTRY` for the wakeword gate's own model."""
    if model in registry:
        return registry[model]
    candidate = Path(model)
    if candidate.is_dir():
        return candidate
    if candidate.is_file() and candidate.parent.name in ("checkpoints", "export"):
        return candidate.parent.parent
    return None


def resolve_model(
    model: str,
    backend: str,
    variant: str = "fp32",
    *,
    registry: dict[str, Path] = MODEL_REGISTRY,
    model_prefix: str = "vcm_model",
    checkpoint_name: str = "checkpoint.pt",
) -> Path:
    """Registry-name / run-dir / direct-file resolution for `--model`.
    `backend='onnx'` resolves to `<run_dir>/export/<model_prefix>.{variant}.onnx`;
    `backend='torch'` resolves to `<run_dir>/checkpoints/<checkpoint_name>`. A
    direct existing file is returned as-is regardless of `backend`/`variant`
    -- the caller named it explicitly.

    `registry`/`model_prefix`/`checkpoint_name` default to today's VCM
    values so every existing call site (`resolve_model(cfg.model,
    cfg.backend, cfg.onnx_variant)`) is unaffected; the wakeword gate calls
    this with `registry=WAKEWORD_MODEL_REGISTRY, model_prefix="wakeword_model"`
    instead of duplicating this function (per `vcm.export_onnx`'s own
    `wakeword_model.{variant}.onnx` / `checkpoints/checkpoint.pt` layout,
    which already mirrors the VCM layout one-for-one)."""
    if backend not in ("onnx", "torch"):
        raise SystemExit(f"unknown --backend {backend!r}; choices are ['onnx', 'torch']")

    if model not in registry:
        candidate = Path(model)
        if candidate.is_file():
            return candidate

    run_dir = _candidate_run_dir(model, registry=registry)
    if run_dir is None or not run_dir.is_dir():
        raise SystemExit(
            f"--model {model!r} is not a registry name ({sorted(registry)}), "
            f"an existing run directory, or an existing file."
        )

    if backend == "onnx":
        artifact = run_dir / "export" / f"{model_prefix}.{variant}.onnx"
        if not artifact.exists():
            raise SystemExit(
                f"{run_dir} has no export/{artifact.name} (no ONNX artifact for "
                f"--onnx-variant {variant!r}) -- re-run "
                f"`python -m me2_voicegen.vcm.export_onnx` against this checkpoint, "
                f"or pass --backend torch to use the .pt checkpoint directly."
            )
        return artifact

    artifact = run_dir / "checkpoints" / checkpoint_name
    if not artifact.exists():
        raise SystemExit(
            f"{run_dir} has no checkpoints/{checkpoint_name} -- re-run vcm.train against "
            f"this run dir, or point --model at a run dir/checkpoint file that has "
            f"one, or pass --backend onnx to use an ONNX export instead."
        )
    return artifact


def resolve_wakeword_model(model: str, backend: str, variant: str = "fp32") -> Path:
    """`resolve_model` specialized for the wakeword gate's own model
    selection (`WAKEWORD_MODEL_REGISTRY`, artifact prefix
    `wakeword_model`) -- independent of `--model`/`--backend`, which
    resolve the VCM decode model the gate sits beside."""
    return resolve_model(
        model,
        backend,
        variant,
        registry=WAKEWORD_MODEL_REGISTRY,
        model_prefix="wakeword_model",
    )


def resolve_threshold(
    run_dir: Optional[str | Path],
    override: Optional[float] = None,
    grammar_label: str = DEFAULT_GRAMMAR_LABEL,
) -> float:
    """Reads `<run_dir>/metadata/eval_report.json`'s `grammar_label` section
    for `chosen_operating_threshold`. `override` (an explicit `--threshold`)
    always wins. Warns to stderr and returns `FALLBACK_THRESHOLD` if
    `run_dir` is `None`, the report is missing/unreadable, or it has no
    section for `grammar_label`."""
    if override is not None:
        return override

    if run_dir is None:
        print(
            f"warning: no run dir to resolve an operating threshold from; "
            f"falling back to {FALLBACK_THRESHOLD} -- pass --threshold explicitly.",
            file=sys.stderr,
        )
        return FALLBACK_THRESHOLD

    report_path = Path(run_dir) / "metadata" / "eval_report.json"
    try:
        report = json.loads(report_path.read_text())
        for section in report["grammar_sections"]:
            if section.get("grammar") == grammar_label:
                return float(section["chosen_operating_threshold"])
        raise KeyError(f"no {grammar_label!r} section in {report_path}")
    except (FileNotFoundError, json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
        print(
            f"warning: could not resolve operating threshold from {report_path} "
            f"({exc!r}); falling back to {FALLBACK_THRESHOLD}, which will accept "
            f"almost nothing -- pass --threshold explicitly to override.",
            file=sys.stderr,
        )
        return FALLBACK_THRESHOLD


def resolve_policy(
    name: str,
    threshold: float,
    *,
    gate: Optional[ListeningGate] = None,
    period_s: float = 5.0,
    on_period_event: Optional[PeriodEventCallback] = None,
    endpoint_options: Optional[Mapping[str, Any]] = None,
) -> AcceptancePolicy:
    try:
        policy_cls = POLICY_REGISTRY[name]
    except KeyError:
        raise SystemExit(
            f"unknown --policy {name!r}; choices are {sorted(POLICY_REGISTRY)}"
        ) from None
    if name in PERIOD_POLICIES:
        if gate is None:
            raise SystemExit(
                f"--policy {name!r} requires a listening gate: pass --gate "
                f"spacebar (with --gate-period for the period length) -- "
                f"--gate none is not a valid combination with --policy {name!r}"
            )
        if name == "single_period" and period_s != 3.0:
            raise SystemExit("--policy single_period requires --gate-period 3 (the evaluated full-inference duration)")
        return policy_cls(
            threshold, gate=gate, period_s=period_s, on_period_event=on_period_event,
            **(dict(endpoint_options or {}) if name == "endpointed" else {}),
        )
    return policy_cls(threshold)


def resolve_grammar(name: str):
    """Reuses `vcm.evaluate.GRAMMAR_REGISTRY` rather than defining a second,
    driftable grammar registry for streaming."""
    try:
        grammar, label = GRAMMAR_REGISTRY[name]
    except KeyError:
        raise SystemExit(
            f"unknown --grammar {name!r}; choices are {sorted(GRAMMAR_REGISTRY)}"
        ) from None
    return grammar, label
