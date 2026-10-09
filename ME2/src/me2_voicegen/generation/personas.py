"""Persona manifest loading for batch generation (generate_personas.py)."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path

from me2_voicegen.synthesis.base import VoicePrompt

NAME_RE = re.compile(r"^[A-Za-z0-9_-]{1,64}$")


@dataclass(frozen=True)
class Persona:
    name: str
    prompt: VoicePrompt


def load_personas(manifest_path: Path) -> list[Persona]:
    manifest_path = Path(manifest_path)
    raw = json.loads(manifest_path.read_text())

    entries = raw.get("personas") if isinstance(raw, dict) else None
    if not isinstance(entries, list) or not entries:
        raise ValueError(
            f"manifest {manifest_path}: expected a non-empty 'personas' list"
        )

    base_dir = manifest_path.resolve().parent
    personas: list[Persona] = []
    seen_names: set[str] = set()

    for index, entry in enumerate(entries):
        if not isinstance(entry, dict):
            raise ValueError(
                f"manifest entry #{index}: expected an object, got {entry!r}"
            )

        name = entry.get("name")
        if not name or not isinstance(name, str):
            raise ValueError(f"manifest entry #{index}: missing required 'name'")
        if not NAME_RE.match(name):
            raise ValueError(
                f"manifest entry {name!r}: name must match {NAME_RE.pattern}"
            )
        if name in seen_names:
            raise ValueError(f"manifest entry {name!r}: duplicate name")

        wav_path_raw = entry.get("wav_path")
        if not wav_path_raw or not isinstance(wav_path_raw, str):
            raise ValueError(f"manifest entry {name!r}: missing required 'wav_path'")

        text = entry.get("text")
        if not text or not isinstance(text, str):
            raise ValueError(f"manifest entry {name!r}: missing required 'text'")

        wav_path = Path(wav_path_raw)
        if not wav_path.is_absolute():
            wav_path = base_dir / wav_path
        if not wav_path.is_file():
            raise ValueError(
                f"manifest entry {name!r}: wav_path does not exist: {wav_path}"
            )

        seen_names.add(name)
        personas.append(Persona(name=name, prompt=VoicePrompt(wav_path=wav_path, text=text)))

    return personas
