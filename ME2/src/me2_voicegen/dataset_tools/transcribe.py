"""Whisper-based transcription with a sidecar-file cache.

Kept import-lazy (whisper/torch only imported inside transcribe_cached, not at
module level) so importing this module - and therefore generate_conversions.py
- doesn't pull in whisper's model-loading machinery for callers that never hit
the resynthesis branch (e.g. --help, or a run with --resynth-prob 0).
"""

from __future__ import annotations

import logging
from pathlib import Path

logger = logging.getLogger(__name__)

_MODEL_CACHE: dict[str, object] = {}


def _load_whisper_model(model_size: str):
    if model_size not in _MODEL_CACHE:
        import whisper

        logger.info("loading whisper model %r", model_size)
        _MODEL_CACHE[model_size] = whisper.load_model(model_size)
    return _MODEL_CACHE[model_size]


def transcript_path(audio_path: Path) -> Path:
    return audio_path.with_suffix(".txt")


def transcribe_cached(audio_path: Path, model_size: str = "base") -> str:
    """Returns audio_path's transcript, reading a cached `<stem>.txt` sidecar
    next to it if present, else transcribing with Whisper and writing that
    sidecar for future runs. Raises if audio_path doesn't exist."""
    if not audio_path.is_file():
        raise FileNotFoundError(f"audio file not found: {audio_path}")

    cache_path = transcript_path(audio_path)
    if cache_path.is_file():
        text = cache_path.read_text(encoding="utf-8").strip()
        if text:
            return text
        logger.warning("cached transcript %s is empty; re-transcribing", cache_path)

    model = _load_whisper_model(model_size)
    result = model.transcribe(str(audio_path))
    text = result["text"].strip()
    if not text:
        raise RuntimeError(f"whisper produced an empty transcript for {audio_path}")

    cache_path.write_text(text + "\n", encoding="utf-8")
    return text
