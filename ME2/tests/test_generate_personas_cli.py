from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from me2_voicegen.generation import generate_personas
from me2_voicegen.synthesis.base import SynthesisResult, Synthesizer
from me2_voicegen.synthesis.factory import _BACKENDS


class FakeSynthesizer(Synthesizer):
    """Deliberately unrelated constructor signature to CosyVoice2Synthesizer's,
    same rationale as tests/test_generate_sample_cli.py's FakeSynthesizer."""

    construction_count = 0

    def __init__(self, voice_id: str = "default-voice", speed: float = 1.0) -> None:
        type(self).construction_count += 1
        self.voice_id = voice_id
        self.speed = speed
        self.calls: list[tuple] = []

    def synthesize(self, text, prompt=None) -> SynthesisResult:
        self.calls.append((text, prompt))
        samples = np.arange(8, dtype=np.float32) / 8.0
        return SynthesisResult(audio=samples[np.newaxis, :], sample_rate=16000)


class RaisesOnConstructionSynthesizer(Synthesizer):
    def __init__(self) -> None:
        raise AssertionError("must never be constructed when manifest is invalid")

    def synthesize(self, text, prompt=None):  # pragma: no cover
        raise NotImplementedError


@pytest.fixture(autouse=True)
def reset_fake_construction_count():
    FakeSynthesizer.construction_count = 0
    yield


@pytest.fixture(autouse=True)
def registered_fake_backend(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setitem(_BACKENDS, "fake", FakeSynthesizer)


def _touch_wav(directory: Path, name: str) -> None:
    (directory / name).write_bytes(b"RIFF....WAVEfmt ")


def _write_manifest(directory: Path, entries: list[dict], filename: str = "personas.json") -> Path:
    manifest_path = directory / filename
    manifest_path.write_text(json.dumps({"personas": entries}))
    return manifest_path


def test_no_backend_specific_identifiers_in_generate_personas_source() -> None:
    source = Path(generate_personas.__file__).read_text()
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


def test_writes_one_wav_per_persona_sharing_run_timestamp(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _touch_wav(tmp_path, "a.wav")
    _touch_wav(tmp_path, "b.wav")
    manifest_path = _write_manifest(
        tmp_path,
        [
            {"name": "english_woman", "wav_path": "a.wav", "text": "hi"},
            {"name": "indian_man", "wav_path": "b.wav", "text": "hello"},
        ],
    )
    out_dir = tmp_path / "out"

    exit_code = generate_personas.main(
        [
            "--manifest",
            str(manifest_path),
            "--backend",
            "fake",
            "--text",
            "hello personas",
            "--out-dir",
            str(out_dir),
        ]
    )

    assert exit_code == 0
    wavs = sorted(p.name for p in out_dir.glob("*.wav"))
    assert len(wavs) == 2
    assert any(name.startswith("sample_english_woman_") for name in wavs)
    assert any(name.startswith("sample_indian_man_") for name in wavs)

    timestamps = {name.rsplit("_", 1)[-1] for name in wavs}
    assert len(timestamps) == 1


def test_backend_constructed_exactly_once_regardless_of_persona_count(tmp_path: Path) -> None:
    _touch_wav(tmp_path, "a.wav")
    _touch_wav(tmp_path, "b.wav")
    _touch_wav(tmp_path, "c.wav")
    manifest_path = _write_manifest(
        tmp_path,
        [
            {"name": "p1", "wav_path": "a.wav", "text": "hi"},
            {"name": "p2", "wav_path": "b.wav", "text": "hi"},
            {"name": "p3", "wav_path": "c.wav", "text": "hi"},
        ],
    )

    exit_code = generate_personas.main(
        [
            "--manifest",
            str(manifest_path),
            "--backend",
            "fake",
            "--out-dir",
            str(tmp_path / "out"),
        ]
    )

    assert exit_code == 0
    assert FakeSynthesizer.construction_count == 1


def test_invalid_manifest_exits_1_with_zero_backend_constructions(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(_BACKENDS, "fake", RaisesOnConstructionSynthesizer)
    manifest_path = _write_manifest(tmp_path, [{"name": "p1", "text": "hi"}])  # missing wav_path

    exit_code = generate_personas.main(
        [
            "--manifest",
            str(manifest_path),
            "--backend",
            "fake",
            "--out-dir",
            str(tmp_path / "out"),
        ]
    )

    assert exit_code == 1
    assert list((tmp_path / "out").glob("*.wav")) == []


def test_one_persona_failure_does_not_abort_batch(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _touch_wav(tmp_path, "a.wav")
    _touch_wav(tmp_path, "b.wav")
    manifest_path = _write_manifest(
        tmp_path,
        [
            {"name": "good", "wav_path": "a.wav", "text": "hi"},
            {"name": "bad", "wav_path": "b.wav", "text": "hi"},
        ],
    )

    class _FlakySynthesizer(FakeSynthesizer):
        def synthesize(self, text, prompt=None):
            if prompt is not None and str(prompt.wav_path).endswith("b.wav"):
                raise RuntimeError("boom")
            return super().synthesize(text, prompt=prompt)

    monkeypatch.setitem(_BACKENDS, "fake", _FlakySynthesizer)
    out_dir = tmp_path / "out"

    exit_code = generate_personas.main(
        [
            "--manifest",
            str(manifest_path),
            "--backend",
            "fake",
            "--out-dir",
            str(out_dir),
        ]
    )

    assert exit_code == 1
    wavs = list(out_dir.glob("sample_good_*.wav"))
    assert len(wavs) == 1
    assert list(out_dir.glob("sample_bad_*.wav")) == []


def test_relative_wav_path_resolves_against_manifest_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    manifest_dir = tmp_path / "manifest_dir"
    manifest_dir.mkdir()
    _touch_wav(manifest_dir, "voice.wav")
    manifest_path = _write_manifest(
        manifest_dir, [{"name": "voiceA", "wav_path": "voice.wav", "text": "hi"}]
    )

    other_cwd = tmp_path / "elsewhere"
    other_cwd.mkdir()
    monkeypatch.chdir(other_cwd)

    recorded: dict[str, object] = {}

    class _RecordingBackend(FakeSynthesizer):
        def synthesize(self, text, prompt=None):
            recorded["prompt"] = prompt
            return super().synthesize(text, prompt=prompt)

    monkeypatch.setitem(_BACKENDS, "fake", _RecordingBackend)

    exit_code = generate_personas.main(
        [
            "--manifest",
            str(manifest_path),
            "--backend",
            "fake",
            "--out-dir",
            str(tmp_path / "out"),
        ]
    )

    assert exit_code == 0
    assert recorded["prompt"].wav_path == manifest_dir / "voice.wav"


def test_default_backend_choices_come_from_list_backends() -> None:
    with pytest.raises(SystemExit):
        generate_personas.parse_args(["--manifest", "x.json", "--backend", "not-registered"])
