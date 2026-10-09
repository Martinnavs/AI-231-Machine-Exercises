from __future__ import annotations

import json
from pathlib import Path

import pytest

from me2_voicegen.generation.personas import Persona, load_personas


def _write_manifest(tmp_path: Path, entries: list[dict], filename: str = "personas.json") -> Path:
    manifest_path = tmp_path / filename
    manifest_path.write_text(json.dumps({"personas": entries}))
    return manifest_path


def _touch_wav(tmp_path: Path, name: str) -> None:
    (tmp_path / name).write_bytes(b"RIFF....WAVEfmt ")


def test_load_personas_happy_path(tmp_path: Path) -> None:
    _touch_wav(tmp_path, "a.wav")
    _touch_wav(tmp_path, "b.wav")
    manifest_path = _write_manifest(
        tmp_path,
        [
            {"name": "english_woman", "wav_path": "a.wav", "text": "hello"},
            {"name": "indian_man", "wav_path": "b.wav", "text": "hi there"},
        ],
    )

    personas = load_personas(manifest_path)

    assert personas == [
        Persona(name="english_woman", prompt=personas[0].prompt),
        Persona(name="indian_man", prompt=personas[1].prompt),
    ]
    assert personas[0].prompt.wav_path == tmp_path / "a.wav"
    assert personas[0].prompt.text == "hello"


def test_relative_wav_path_resolves_against_manifest_dir_not_cwd(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    sub_dir = tmp_path / "manifest_dir"
    sub_dir.mkdir()
    _touch_wav(sub_dir, "voice.wav")
    manifest_path = _write_manifest(
        sub_dir, [{"name": "voiceA", "wav_path": "voice.wav", "text": "hello"}]
    )

    other_cwd = tmp_path / "elsewhere"
    other_cwd.mkdir()
    monkeypatch.chdir(other_cwd)

    personas = load_personas(manifest_path)

    assert personas[0].prompt.wav_path == sub_dir / "voice.wav"


def test_missing_name_raises(tmp_path: Path) -> None:
    _touch_wav(tmp_path, "a.wav")
    manifest_path = _write_manifest(tmp_path, [{"wav_path": "a.wav", "text": "hi"}])

    with pytest.raises(ValueError, match="name"):
        load_personas(manifest_path)


def test_missing_wav_path_raises(tmp_path: Path) -> None:
    manifest_path = _write_manifest(tmp_path, [{"name": "p1", "text": "hi"}])

    with pytest.raises(ValueError, match="p1"):
        load_personas(manifest_path)


def test_missing_text_raises(tmp_path: Path) -> None:
    _touch_wav(tmp_path, "a.wav")
    manifest_path = _write_manifest(tmp_path, [{"name": "p1", "wav_path": "a.wav"}])

    with pytest.raises(ValueError, match="p1"):
        load_personas(manifest_path)


def test_duplicate_name_raises(tmp_path: Path) -> None:
    _touch_wav(tmp_path, "a.wav")
    _touch_wav(tmp_path, "b.wav")
    manifest_path = _write_manifest(
        tmp_path,
        [
            {"name": "dupe", "wav_path": "a.wav", "text": "hi"},
            {"name": "dupe", "wav_path": "b.wav", "text": "hello"},
        ],
    )

    with pytest.raises(ValueError, match="dupe"):
        load_personas(manifest_path)


def test_invalid_name_pattern_raises(tmp_path: Path) -> None:
    _touch_wav(tmp_path, "a.wav")
    manifest_path = _write_manifest(
        tmp_path, [{"name": "bad name!", "wav_path": "a.wav", "text": "hi"}]
    )

    with pytest.raises(ValueError, match="bad name!"):
        load_personas(manifest_path)


def test_nonexistent_wav_raises(tmp_path: Path) -> None:
    manifest_path = _write_manifest(
        tmp_path, [{"name": "p1", "wav_path": "missing.wav", "text": "hi"}]
    )

    with pytest.raises(ValueError, match="p1"):
        load_personas(manifest_path)


def test_empty_personas_list_raises(tmp_path: Path) -> None:
    manifest_path = _write_manifest(tmp_path, [])

    with pytest.raises(ValueError, match="personas"):
        load_personas(manifest_path)


def test_non_dict_entry_raises(tmp_path: Path) -> None:
    manifest_path = tmp_path / "personas.json"
    manifest_path.write_text(json.dumps({"personas": ["not-an-object"]}))

    with pytest.raises(ValueError, match="#0"):
        load_personas(manifest_path)


def test_invalid_json_raises(tmp_path: Path) -> None:
    manifest_path = tmp_path / "personas.json"
    manifest_path.write_text("{not valid json")

    with pytest.raises(ValueError):
        load_personas(manifest_path)


def test_missing_manifest_file_raises_oserror(tmp_path: Path) -> None:
    with pytest.raises(OSError):
        load_personas(tmp_path / "does_not_exist.json")


def test_personas_key_not_a_list_raises(tmp_path: Path) -> None:
    manifest_path = tmp_path / "personas.json"
    manifest_path.write_text(json.dumps({"personas": "english_woman"}))

    with pytest.raises(ValueError, match="personas"):
        load_personas(manifest_path)
