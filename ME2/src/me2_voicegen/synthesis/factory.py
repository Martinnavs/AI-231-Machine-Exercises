from __future__ import annotations

from .base import Synthesizer
from .cosyvoice2_backend import CosyVoice2Synthesizer

_BACKENDS: dict[str, type[Synthesizer]] = {
    "cosyvoice2": CosyVoice2Synthesizer,
}


def list_backends() -> list[str]:
    return sorted(_BACKENDS)


def get_backend_class(name: str) -> type[Synthesizer]:
    try:
        return _BACKENDS[name]
    except KeyError:
        available = ", ".join(list_backends()) or "none"
        raise ValueError(
            f"unknown synthesis backend {name!r}; available: {available}"
        ) from None


def create_synthesizer(name: str, **config) -> Synthesizer:
    return get_backend_class(name)(**config)
