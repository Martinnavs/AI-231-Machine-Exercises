from __future__ import annotations

import inspect
from pathlib import Path

import numpy as np
import pytest

from me2_voicegen.generation import generate_sample
from me2_voicegen.synthesis.base import SynthesisResult, Synthesizer, VoicePrompt
from me2_voicegen.synthesis.factory import _BACKENDS


class FakeSynthesizer(Synthesizer):
    """Deliberately unrelated constructor signature to CosyVoice2Synthesizer's
    (model_dir/device/fp16) - this is what makes criterion 6 (no backend-specific
    identifier in generate_sample.py) actually provable, not merely asserted."""

    def __init__(self, voice_id: str = "default-voice", speed: float = 1.0) -> None:
        self.voice_id = voice_id
        self.speed = speed
        self.calls: list[tuple] = []

    def synthesize(self, text, prompt=None) -> SynthesisResult:
        self.calls.append((text, prompt))
        samples = np.arange(8, dtype=np.float32) / 8.0
        return SynthesisResult(audio=samples[np.newaxis, :], sample_rate=16000)


@pytest.fixture(autouse=True)
def registered_fake_backend(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setitem(_BACKENDS, "fake", FakeSynthesizer)


def test_no_backend_specific_identifiers_in_generate_sample_source() -> None:
    """Criterion 6 bans backend *dispatch/API* identifiers - the literal registry
    key "cosyvoice2" (outside DEFAULT_BACKEND), the vendor entry point AutoModel,
    and the vendor method inference_zero_shot. It does NOT ban this file's
    documented reliance (see this ticket's Shared Context) on
    me2_voicegen.generation.cosyvoice_env - a Ticket 01 shared-infra module (vendor/models/out
    path resolution) whose name happens to contain "cosyvoice" as a substring but
    carries no backend-specific dispatch logic of its own; it's imported here only
    for OUT_DIR. What's actually banned there is importing the *vendor package*
    itself (`import cosyvoice` / `from cosyvoice...`), which this file never does.
    """
    source = Path(generate_sample.__file__).read_text()
    lines = source.splitlines()

    def _occurrences(token: str) -> list[str]:
        return [
            line
            for line in lines
            if token.lower() in line.lower() and "DEFAULT_BACKEND" not in line
        ]

    assert _occurrences("cosyvoice2") == []
    assert _occurrences("AutoModel") == []
    assert _occurrences("inference_zero_shot") == []
    vendor_package_imports = [
        line
        for line in lines
        if ("import cosyvoice" in line or "from cosyvoice." in line or "from cosyvoice import" in line)
        and "cosyvoice_env" not in line
    ]
    assert vendor_package_imports == []


def test_fake_backend_runs_end_to_end_through_cli(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(generate_sample, "OUT_DIR", tmp_path)

    exit_code = generate_sample.main(
        [
            "--backend",
            "fake",
            "--text",
            "hello from a fake backend",
            "--opt",
            "voice_id=narrator",
            "--opt",
            "speed=1.5",
        ]
    )

    assert exit_code == 0
    wavs = list(tmp_path.glob("*.wav"))
    assert len(wavs) == 1


def test_build_config_forwards_device_when_backend_accepts_it() -> None:
    class _AcceptsDevice(Synthesizer):
        def __init__(self, device: str = "auto") -> None:
            self.device = device

        def synthesize(self, text, prompt=None):  # pragma: no cover
            raise NotImplementedError

    args = generate_sample.parse_args(["--device", "cpu"])
    config = generate_sample._build_config(_AcceptsDevice, args)

    assert config == {"device": "cpu"}


def test_build_config_silently_drops_device_for_unrelated_constructor() -> None:
    args = generate_sample.parse_args(["--device", "cuda"])
    config = generate_sample._build_config(FakeSynthesizer, args)

    assert config == {}


def test_build_config_merges_opt_overrides() -> None:
    args = generate_sample.parse_args(["--opt", "voice_id=narrator", "--opt", "speed=2.0"])
    config = generate_sample._build_config(FakeSynthesizer, args)

    assert config == {"voice_id": "narrator", "speed": 2.0}


def test_parse_opts_coerces_types() -> None:
    opts = generate_sample._parse_opts(
        ["fp16=true", "beam_size=5", "temperature=0.2", "language=none", "name=en"]
    )

    assert opts == {
        "fp16": True,
        "beam_size": 5,
        "temperature": 0.2,
        "language": None,
        "name": "en",
    }


def test_parse_opts_rejects_malformed_entry() -> None:
    with pytest.raises(ValueError, match="KEY=VALUE"):
        generate_sample._parse_opts(["not-a-key-value-pair"])


def test_main_unknown_opt_key_is_a_loud_error(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(generate_sample, "OUT_DIR", tmp_path)

    exit_code = generate_sample.main(
        ["--backend", "fake", "--opt", "not_a_real_kwarg=1"]
    )

    assert exit_code == 1
    assert list(tmp_path.glob("*.wav")) == []


def test_main_unknown_backend_exits_nonzero() -> None:
    with pytest.raises(SystemExit):
        generate_sample.parse_args(["--backend", "not-registered"])


def test_default_text_used_when_text_is_empty_string(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Mirrors the Makefile's `generate` recipe, which always passes --text
    (default TEXT ?= , i.e. an empty string) rather than omitting the flag."""
    monkeypatch.setattr(generate_sample, "OUT_DIR", tmp_path)
    recorded: dict[str, object] = {}

    class _RecordingBackend(FakeSynthesizer):
        def synthesize(self, text, prompt=None):
            recorded["text"] = text
            return super().synthesize(text, prompt=prompt)

    from me2_voicegen.synthesis.factory import _BACKENDS

    monkeypatch.setitem(_BACKENDS, "fake", _RecordingBackend)

    exit_code = generate_sample.main(["--backend", "fake", "--text", ""])

    assert exit_code == 0
    assert recorded["text"] == generate_sample.DEFAULT_TEXT


def test_default_prompt_used_when_prompt_wav_not_given(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(generate_sample, "OUT_DIR", tmp_path)
    recorded: dict[str, object] = {}

    class _DefaultPromptBackend(FakeSynthesizer):
        DEFAULT_PROMPT_WAV = Path("/vendored/sample.wav")
        DEFAULT_PROMPT_TEXT = "vendored transcript"

        def synthesize(self, text, prompt=None):
            recorded["prompt"] = prompt
            return super().synthesize(text, prompt=prompt)

    from me2_voicegen.synthesis.factory import _BACKENDS

    monkeypatch.setitem(_BACKENDS, "fake", _DefaultPromptBackend)

    generate_sample.main(["--backend", "fake"])

    prompt: VoicePrompt = recorded["prompt"]
    assert prompt.text == "vendored transcript"
    assert prompt.wav_path == Path("/vendored/sample.wav")


def test_backend_with_no_default_prompt_gets_none_when_none_given() -> None:
    args = generate_sample.parse_args(["--backend", "fake"])
    prompt = generate_sample._build_prompt(FakeSynthesizer, args)

    assert prompt is None


def test_custom_prompt_wav_without_prompt_text_leaves_text_none() -> None:
    args = generate_sample.parse_args(["--prompt-wav", "/some/custom.wav"])
    prompt = generate_sample._build_prompt(FakeSynthesizer, args)

    assert prompt.wav_path == Path("/some/custom.wav")
    assert prompt.text is None


def test_cosyvoice2_backend_class_declares_real_default_prompt() -> None:
    from me2_voicegen.synthesis.cosyvoice2_backend import CosyVoice2Synthesizer

    args = generate_sample.parse_args(["--backend", "cosyvoice2"])
    prompt = generate_sample._build_prompt(CosyVoice2Synthesizer, args)

    assert prompt.wav_path == CosyVoice2Synthesizer.DEFAULT_PROMPT_WAV
    assert prompt.text == CosyVoice2Synthesizer.DEFAULT_PROMPT_TEXT
