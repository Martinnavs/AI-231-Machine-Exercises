from __future__ import annotations

import pytest

from me2_voicegen.synthesis.base import SynthesisResult, Synthesizer
from me2_voicegen.synthesis.factory import _BACKENDS, create_synthesizer, get_backend_class, list_backends


class _DummySynthesizer(Synthesizer):
    def __init__(self, **config) -> None:
        self.config = config

    def synthesize(self, text, prompt=None):  # pragma: no cover - not exercised here
        import numpy as np

        return SynthesisResult(audio=np.zeros((1, 1), dtype="float32"), sample_rate=16000)


def test_list_backends_includes_cosyvoice2() -> None:
    assert "cosyvoice2" in list_backends()


def test_list_backends_is_sorted() -> None:
    assert list_backends() == sorted(list_backends())


def test_get_backend_class_unknown_backend_raises() -> None:
    with pytest.raises(ValueError, match="unknown synthesis backend"):
        get_backend_class("not-a-real-backend")


def test_get_backend_class_error_names_available_backends() -> None:
    with pytest.raises(ValueError, match="cosyvoice2"):
        get_backend_class("not-a-real-backend")


def test_create_synthesizer_unknown_backend_raises() -> None:
    with pytest.raises(ValueError, match="unknown synthesis backend"):
        create_synthesizer("not-a-real-backend")


def test_create_synthesizer_passes_config_through(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(_BACKENDS, "dummy", _DummySynthesizer)

    synthesizer = create_synthesizer("dummy", voice_id="narrator", speed=1.2)

    assert isinstance(synthesizer, _DummySynthesizer)
    assert synthesizer.config == {"voice_id": "narrator", "speed": 1.2}
