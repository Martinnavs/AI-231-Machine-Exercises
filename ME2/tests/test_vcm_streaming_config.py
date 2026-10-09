"""Fast, CPU-only tests for `vcm.streaming.config`. Config resolution logic
uses fixture JSON/dirs under `tmp_path`; the one real-artifact check (ONNX
vs. torch parity) lives in `test_vcm_streaming_backends.py` and is
`@pytest.mark.slow`.
"""

from __future__ import annotations

import json

import pytest

from me2_voicegen.common.features import SAMPLE_RATE
from me2_voicegen.vcm.streaming.config import (
    FALLBACK_THRESHOLD,
    MODEL_REGISTRY,
    POLICY_REGISTRY,
    WAKEWORD_MODEL_REGISTRY,
    StreamingConfig,
    resolve_model,
    resolve_policy,
    resolve_threshold,
    resolve_wakeword_model,
)
from me2_voicegen.vcm.streaming.policy import ModePeriodPolicy, ThresholdPolicy


def _make_run_dir(tmp_path, name="run", with_export=True, with_checkpoint=True, variants=("fp32",)):
    run_dir = tmp_path / name
    if with_export:
        export_dir = run_dir / "export"
        export_dir.mkdir(parents=True)
        for variant in variants:
            (export_dir / f"vcm_model.{variant}.onnx").write_bytes(b"fake-onnx")
    if with_checkpoint:
        ckpt_dir = run_dir / "checkpoints"
        ckpt_dir.mkdir(parents=True)
        (ckpt_dir / "checkpoint.pt").write_bytes(b"fake-checkpoint")
    return run_dir


def _write_eval_report(run_dir, threshold=-0.1, grammar_label="OPTIONB_GRAMMAR"):
    metadata_dir = run_dir / "metadata"
    metadata_dir.mkdir(parents=True, exist_ok=True)
    report = {
        "grammar_sections": [
            {"grammar": grammar_label, "chosen_operating_threshold": threshold}
        ]
    }
    (metadata_dir / "eval_report.json").write_text(json.dumps(report))
    return metadata_dir / "eval_report.json"


# ---------------------------------------------------------------------------
# resolve_model
# ---------------------------------------------------------------------------


def test_resolve_model_registry_name_onnx(monkeypatch, tmp_path):
    run_dir = _make_run_dir(tmp_path)
    monkeypatch.setitem(MODEL_REGISTRY, "fake-registry-entry", run_dir)
    path = resolve_model("fake-registry-entry", backend="onnx", variant="fp32")
    assert path == run_dir / "export" / "vcm_model.fp32.onnx"


def test_resolve_model_registry_name_torch(monkeypatch, tmp_path):
    run_dir = _make_run_dir(tmp_path)
    monkeypatch.setitem(MODEL_REGISTRY, "fake-registry-entry", run_dir)
    path = resolve_model("fake-registry-entry", backend="torch")
    assert path == run_dir / "checkpoints" / "checkpoint.pt"


def test_resolve_model_run_dir_form(tmp_path):
    run_dir = _make_run_dir(tmp_path)
    assert resolve_model(str(run_dir), backend="onnx") == run_dir / "export" / "vcm_model.fp32.onnx"
    assert resolve_model(str(run_dir), backend="torch") == run_dir / "checkpoints" / "checkpoint.pt"


def test_resolve_model_direct_file_form(tmp_path):
    onnx_file = tmp_path / "somewhere" / "my_model.onnx"
    onnx_file.parent.mkdir(parents=True)
    onnx_file.write_bytes(b"fake")
    assert resolve_model(str(onnx_file), backend="onnx") == onnx_file


def test_resolve_model_onnx_variant_int8_selects_int8_artifact(tmp_path):
    run_dir = _make_run_dir(tmp_path, variants=("fp32", "int8"))
    path = resolve_model(str(run_dir), backend="onnx", variant="int8")
    assert path.name == "vcm_model.int8.onnx"


def test_resolve_model_nonexistent_is_actionable_system_exit(tmp_path):
    with pytest.raises(SystemExit) as excinfo:
        resolve_model(str(tmp_path / "does-not-exist"), backend="onnx")
    message = str(excinfo.value)
    assert "does-not-exist" in message
    assert "registry" in message.lower()


def test_resolve_model_missing_export_names_torch_alternative(tmp_path):
    run_dir = _make_run_dir(tmp_path, with_export=False, with_checkpoint=True)
    with pytest.raises(SystemExit) as excinfo:
        resolve_model(str(run_dir), backend="onnx")
    message = str(excinfo.value)
    assert "export" in message
    assert "--backend torch" in message


def test_resolve_model_missing_checkpoint_names_export_onnx_alternative(tmp_path):
    run_dir = _make_run_dir(tmp_path, with_export=True, with_checkpoint=False)
    with pytest.raises(SystemExit) as excinfo:
        resolve_model(str(run_dir), backend="torch")
    message = str(excinfo.value)
    assert "checkpoint.pt" in message
    assert "export_onnx" in message or "--backend onnx" in message


def test_resolve_model_unknown_backend_is_system_exit(tmp_path):
    run_dir = _make_run_dir(tmp_path)
    with pytest.raises(SystemExit):
        resolve_model(str(run_dir), backend="tensorflow")


# ---------------------------------------------------------------------------
# resolve_model generalization (registry/model_prefix/checkpoint_name) --
# feature `wakeword-gate`. The wakeword gate's own model selection reuses
# resolve_model rather than duplicating it; these tests pin that today's
# VCM-prefix behavior above is unaffected by the new keyword-only params,
# and that the wakeword-prefix path resolves correctly through the same
# function.
# ---------------------------------------------------------------------------


def _make_wakeword_run_dir(tmp_path, name="wakeword-run", with_export=True, with_checkpoint=True, variants=("fp32",)):
    run_dir = tmp_path / name
    if with_export:
        export_dir = run_dir / "export"
        export_dir.mkdir(parents=True)
        for variant in variants:
            (export_dir / f"wakeword_model.{variant}.onnx").write_bytes(b"fake-onnx")
    if with_checkpoint:
        ckpt_dir = run_dir / "checkpoints"
        ckpt_dir.mkdir(parents=True)
        (ckpt_dir / "checkpoint.pt").write_bytes(b"fake-checkpoint")
    return run_dir


def test_resolve_wakeword_model_onnx_uses_wakeword_prefix(tmp_path):
    run_dir = _make_wakeword_run_dir(tmp_path)
    path = resolve_wakeword_model(str(run_dir), backend="onnx", variant="fp32")
    assert path == run_dir / "export" / "wakeword_model.fp32.onnx"


def test_resolve_wakeword_model_torch(tmp_path):
    run_dir = _make_wakeword_run_dir(tmp_path)
    path = resolve_wakeword_model(str(run_dir), backend="torch")
    assert path == run_dir / "checkpoints" / "checkpoint.pt"


def test_resolve_wakeword_model_registry_name(monkeypatch, tmp_path):
    run_dir = _make_wakeword_run_dir(tmp_path)
    monkeypatch.setitem(WAKEWORD_MODEL_REGISTRY, "fake-wakeword-entry", run_dir)
    path = resolve_wakeword_model("fake-wakeword-entry", backend="onnx")
    assert path == run_dir / "export" / "wakeword_model.fp32.onnx"


def test_resolve_wakeword_model_missing_export_error_names_wakeword_prefix(tmp_path):
    run_dir = _make_wakeword_run_dir(tmp_path, with_export=False, with_checkpoint=True)
    with pytest.raises(SystemExit) as excinfo:
        resolve_wakeword_model(str(run_dir), backend="onnx")
    assert "wakeword_model.fp32.onnx" in str(excinfo.value)


def test_resolve_model_default_prefix_and_registry_unchanged_by_generalization(tmp_path):
    """The new registry/model_prefix/checkpoint_name kwargs are
    keyword-only with defaults equal to today's VCM values -- every
    existing `resolve_model(model, backend, variant)` call site is
    unaffected. A wakeword-prefix-only run dir must NOT resolve through
    the default (VCM-prefixed) call."""
    run_dir = _make_wakeword_run_dir(tmp_path, with_checkpoint=False)  # only wakeword_model.fp32.onnx
    with pytest.raises(SystemExit):
        resolve_model(str(run_dir), backend="onnx")  # looks for vcm_model.fp32.onnx, not present


# ---------------------------------------------------------------------------
# resolve_threshold
# ---------------------------------------------------------------------------


def test_resolve_threshold_reads_real_report_value(tmp_path):
    run_dir = tmp_path / "run"
    _write_eval_report(run_dir, threshold=-0.1)
    assert resolve_threshold(run_dir, override=None) == -0.1


def test_resolve_threshold_override_wins(tmp_path):
    run_dir = tmp_path / "run"
    _write_eval_report(run_dir, threshold=-0.1)
    assert resolve_threshold(run_dir, override=-2.5) == -2.5


def test_resolve_threshold_warns_and_falls_back_when_report_missing(tmp_path, capsys):
    run_dir = tmp_path / "run-with-no-report"
    run_dir.mkdir()
    value = resolve_threshold(run_dir, override=None)
    assert value == FALLBACK_THRESHOLD
    captured = capsys.readouterr()
    assert "warning" in captured.err.lower()


def test_resolve_threshold_warns_and_falls_back_when_section_absent(tmp_path, capsys):
    run_dir = tmp_path / "run"
    _write_eval_report(run_dir, threshold=-0.1, grammar_label="SOME_OTHER_GRAMMAR")
    value = resolve_threshold(run_dir, override=None, grammar_label="OPTIONB_GRAMMAR")
    assert value == FALLBACK_THRESHOLD
    captured = capsys.readouterr()
    assert "warning" in captured.err.lower()


def test_resolve_threshold_none_run_dir_warns_and_falls_back(capsys):
    value = resolve_threshold(None, override=None)
    assert value == FALLBACK_THRESHOLD
    captured = capsys.readouterr()
    assert "warning" in captured.err.lower()


def test_fallback_threshold_is_not_the_real_checked_in_operating_point():
    assert FALLBACK_THRESHOLD != -0.1


# ---------------------------------------------------------------------------
# resolve_policy
# ---------------------------------------------------------------------------


def test_resolve_policy_known_name_returns_configured_instance():
    policy = resolve_policy("threshold", threshold=-0.3)
    assert isinstance(policy, ThresholdPolicy)
    assert policy.threshold == -0.3


def test_resolve_policy_unknown_name_lists_valid_choices():
    with pytest.raises(SystemExit) as excinfo:
        resolve_policy("nonexistent-policy", threshold=0.0)
    message = str(excinfo.value)
    assert "nonexistent-policy" in message
    for choice in POLICY_REGISTRY:
        assert choice in message


def test_resolve_policy_mode_period_without_gate_is_actionable_system_exit():
    with pytest.raises(SystemExit) as excinfo:
        resolve_policy("mode_period", threshold=-0.1)
    message = str(excinfo.value)
    assert "--gate" in message
    assert "spacebar" in message
    assert "none" in message


class _FakeGate:
    def poll(self, samples_seen, window=None):
        raise NotImplementedError

    def close(self) -> None:
        raise NotImplementedError


def test_resolve_policy_mode_period_wires_gate_and_period():
    gate = _FakeGate()
    policy = resolve_policy("mode_period", threshold=-0.1, gate=gate, period_s=2.5)
    assert isinstance(policy, ModePeriodPolicy)
    assert policy.threshold == -0.1
    assert policy._gate is gate
    assert policy._period_samples == int(2.5 * SAMPLE_RATE)


def test_resolve_policy_mode_period_defaults_period_to_5_s():
    gate = _FakeGate()
    policy = resolve_policy("mode_period", threshold=-0.1, gate=gate)
    assert isinstance(policy, ModePeriodPolicy)
    assert policy._period_samples == int(5.0 * SAMPLE_RATE)


# ---------------------------------------------------------------------------
# StreamingConfig.from_json / precedence merge
# ---------------------------------------------------------------------------


def test_from_json_overrides_defaults(tmp_path):
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps({"window_s": 3.0, "backend": "torch"}))
    config = StreamingConfig.from_json(config_path)
    assert config.window_s == 3.0
    assert config.backend == "torch"
    # Untouched fields keep dataclass defaults.
    assert config.stride_s == StreamingConfig().stride_s


def test_from_json_unknown_field_is_actionable_system_exit(tmp_path):
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps({"not_a_real_field": 1}))
    with pytest.raises(SystemExit) as excinfo:
        StreamingConfig.from_json(config_path)
    assert "not_a_real_field" in str(excinfo.value)


def test_from_json_missing_file_is_system_exit(tmp_path):
    with pytest.raises(SystemExit):
        StreamingConfig.from_json(tmp_path / "missing.json")


def test_from_json_invalid_json_is_system_exit(tmp_path):
    config_path = tmp_path / "config.json"
    config_path.write_text("{not valid json")
    with pytest.raises(SystemExit):
        StreamingConfig.from_json(config_path)


def test_merge_precedence_defaults_lt_json_lt_cli(tmp_path):
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps({"window_s": 3.0, "stride_s": 0.5, "backend": "torch"}))

    # No JSON, no CLI -> pure defaults.
    defaults_only = StreamingConfig.merge()
    assert defaults_only.window_s == StreamingConfig().window_s

    # JSON overrides defaults.
    json_only = StreamingConfig.merge(json_path=config_path)
    assert json_only.window_s == 3.0
    assert json_only.stride_s == 0.5
    assert json_only.backend == "torch"
    # Field JSON didn't touch keeps the dataclass default.
    assert json_only.refractory_s == StreamingConfig().refractory_s

    # CLI overrides JSON (only for keys explicitly set, i.e. not None).
    merged = StreamingConfig.merge(
        json_path=config_path,
        cli_overrides={"window_s": 4.0, "backend": None, "device": "cuda"},
    )
    assert merged.window_s == 4.0  # CLI wins over JSON
    assert merged.stride_s == 0.5  # JSON wins over default (CLI didn't set it)
    assert merged.backend == "torch"  # CLI value was None -> not explicitly set, JSON wins
    assert merged.device == "cuda"  # CLI wins over default (JSON didn't set it)


def test_merge_unknown_cli_override_is_system_exit():
    with pytest.raises(SystemExit):
        StreamingConfig.merge(cli_overrides={"not_a_real_field": 1})


def test_from_json_rejects_onnx_variant_outside_cli_choices(tmp_path):
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps({"onnx_variant": "not-a-real-variant"}))
    with pytest.raises(SystemExit) as excinfo:
        StreamingConfig.from_json(config_path)
    assert "onnx_variant" in str(excinfo.value)


def test_from_json_rejects_backend_outside_cli_choices(tmp_path):
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps({"backend": "tensorflow"}))
    with pytest.raises(SystemExit) as excinfo:
        StreamingConfig.from_json(config_path)
    assert "backend" in str(excinfo.value)


def test_streaming_config_gate_field_defaults():
    config = StreamingConfig()
    assert config.gate == "none"
    assert config.gate_period_s == 5.0


def test_streaming_config_wakeword_field_defaults():
    config = StreamingConfig()
    assert config.wakeword_model == "default"
    assert config.wakeword_backend == "torch"
    assert config.wakeword_onnx_variant == "fp32"
    assert config.wakeword_threshold == pytest.approx(0.9)


def test_from_json_accepts_gate_wakeword(tmp_path):
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps({"gate": "wakeword"}))
    config = StreamingConfig.from_json(config_path)
    assert config.gate == "wakeword"


def test_policy_registry_contains_mode_period():
    assert POLICY_REGISTRY["mode_period"] is ModePeriodPolicy


def test_from_json_accepts_gate_and_gate_period(tmp_path):
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps({"gate": "spacebar", "gate_period_s": 2.5}))
    config = StreamingConfig.from_json(config_path)
    assert config.gate == "spacebar"
    assert config.gate_period_s == 2.5


def test_from_json_rejects_gate_outside_cli_choices(tmp_path):
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps({"gate": "bogus"}))
    with pytest.raises(SystemExit) as excinfo:
        StreamingConfig.from_json(config_path)
    message = str(excinfo.value)
    assert "gate" in message
    assert "none" in message
    assert "spacebar" in message


def test_from_json_rejects_non_string_gate(tmp_path):
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps({"gate": 123}))
    with pytest.raises(SystemExit) as excinfo:
        StreamingConfig.from_json(config_path)
    assert "gate" in str(excinfo.value)


def test_from_json_rejects_non_numeric_gate_period_s(tmp_path):
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps({"gate_period_s": "not-a-number"}))
    with pytest.raises(SystemExit) as excinfo:
        StreamingConfig.from_json(config_path)
    assert "gate_period_s" in str(excinfo.value)


def test_from_json_rejects_non_int_ort_threads_that_does_not_coerce(tmp_path):
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps({"ort_threads": "not-a-number"}))
    with pytest.raises(SystemExit) as excinfo:
        StreamingConfig.from_json(config_path)
    assert "ort_threads" in str(excinfo.value)


def test_from_json_coerces_numeric_string_ort_threads_to_int(tmp_path):
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps({"ort_threads": "4"}))
    config = StreamingConfig.from_json(config_path)
    assert config.ort_threads == 4
    assert isinstance(config.ort_threads, int)


def test_from_json_rejects_bool_for_int_field(tmp_path):
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps({"ort_threads": True}))
    with pytest.raises(SystemExit) as excinfo:
        StreamingConfig.from_json(config_path)
    assert "ort_threads" in str(excinfo.value)


def test_from_json_rejects_wrong_type_for_string_field(tmp_path):
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps({"device": 123}))
    with pytest.raises(SystemExit) as excinfo:
        StreamingConfig.from_json(config_path)
    assert "device" in str(excinfo.value)


def test_from_json_rejects_wrong_type_for_bool_field(tmp_path):
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps({"realtime": "true"}))
    with pytest.raises(SystemExit) as excinfo:
        StreamingConfig.from_json(config_path)
    assert "realtime" in str(excinfo.value)


def test_from_json_accepts_int_for_float_field(tmp_path):
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps({"window_s": 3}))
    config = StreamingConfig.from_json(config_path)
    assert config.window_s == 3.0
    assert isinstance(config.window_s, float)


def test_from_json_allows_null_for_nullable_fields(tmp_path):
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps({"threshold": None, "mic_command": None, "listen_for": None}))
    config = StreamingConfig.from_json(config_path)
    assert config.threshold is None
    assert config.mic_command is None
    assert config.listen_for is None


def test_from_json_rejects_null_for_non_nullable_field(tmp_path):
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps({"model": None}))
    with pytest.raises(SystemExit) as excinfo:
        StreamingConfig.from_json(config_path)
    assert "model" in str(excinfo.value)


def test_streaming_config_is_frozen():
    config = StreamingConfig()
    with pytest.raises(dataclasses_frozen_error_types()):
        config.window_s = 99.0


def dataclasses_frozen_error_types():
    import dataclasses

    return (dataclasses.FrozenInstanceError,)


# ---------------------------------------------------------------------------
# log_periods field + resolve_policy's period-event sink (--log-periods)
# ---------------------------------------------------------------------------


def test_streaming_config_log_periods_defaults_false():
    assert StreamingConfig().log_periods is False
    assert StreamingConfig(log_periods=True).log_periods is True


def test_from_json_accepts_log_periods_bool(tmp_path):
    path = tmp_path / "config.json"
    path.write_text('{"log_periods": true}')
    assert StreamingConfig.from_json(path).log_periods is True


def test_from_json_rejects_non_bool_log_periods(tmp_path):
    path = tmp_path / "config.json"
    path.write_text('{"log_periods": "yes"}')
    with pytest.raises(SystemExit) as excinfo:
        StreamingConfig.from_json(path)
    assert "log_periods" in str(excinfo.value)


def test_resolve_policy_forwards_period_event_sink_to_mode_period_only():
    class _StandinGate:
        """Duck-typed `ListeningGate`; resolve_policy only stores it."""

    events: list = []

    def _sink(event, samples_seen, decision):
        events.append(event)

    policy = resolve_policy(
        "mode_period", -0.1, gate=_StandinGate(), period_s=5.0, on_period_event=_sink
    )
    assert policy._on_period_event is _sink

    # default stays None; the threshold policy never receives the kwarg
    assert resolve_policy("mode_period", -0.1, gate=_StandinGate())._on_period_event is None
    threshold_policy = resolve_policy("threshold", -0.1)
    assert not hasattr(threshold_policy, "_on_period_event")


# ---------------------------------------------------------------------------
# required_command_margin (incomplete-prefix rejection gate; ticket 03 of
# .scratch/incomplete-grammar-rejection/tickets, docs/
# INCOMPLETE-GRAMMAR-REJECTION.md Step 3). Same validation rules as the
# existing nullable float `threshold` field: null allowed, int coerced to
# float, strings/bools rejected.
# ---------------------------------------------------------------------------


def test_streaming_config_required_command_margin_defaults_none():
    config = StreamingConfig()
    assert config.required_command_margin is None


@pytest.mark.parametrize(
    ("json_value", "expected"),
    [(None, None), (0, 0.0), (-0.5, -0.5), (1.5, 1.5)],
)
def test_from_json_accepts_nullable_required_command_margin(tmp_path, json_value, expected):
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps({"required_command_margin": json_value}))
    config = StreamingConfig.from_json(config_path)
    assert config.required_command_margin == expected


@pytest.mark.parametrize("json_value", [True, "0.5"])
def test_from_json_rejects_bad_required_command_margin(tmp_path, json_value):
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps({"required_command_margin": json_value}))
    with pytest.raises(SystemExit) as excinfo:
        StreamingConfig.from_json(config_path)
    assert "required_command_margin" in str(excinfo.value)


def test_streaming_config_score_mode_defaults_mean_frame():
    assert StreamingConfig().score_mode == "mean_frame"


@pytest.mark.parametrize("value", ["mean_frame", "per_char"])
def test_from_json_accepts_score_mode(tmp_path, value):
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps({"score_mode": value}))
    assert StreamingConfig.from_json(config_path).score_mode == value


@pytest.mark.parametrize("value", ["dense", 1, None, True])
def test_from_json_rejects_bad_score_mode(tmp_path, value):
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps({"score_mode": value}))
    with pytest.raises(SystemExit) as excinfo:
        StreamingConfig.from_json(config_path)
    assert "score_mode" in str(excinfo.value)
