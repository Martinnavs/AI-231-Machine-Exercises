"""Backend-agnostic CLI helpers shared by generate_sample.py and
generate_personas.py. No backend-specific identifier may appear here except
the DEFAULT_BACKEND constant's value - see tests/test_generate_sample_cli.py's
test_no_backend_specific_identifiers_in_generate_sample_source, which this
module keeps green by holding the one place that literal is allowed to live.
"""

from __future__ import annotations

import inspect
import logging

logger = logging.getLogger(__name__)

DEFAULT_BACKEND = "cosyvoice2"

# Long enough to dodge the len(tts_text) < 0.5 * len(prompt_text) warning a
# cloning backend's default prompt transcript might trigger, and contains a
# natural KWS wake word.
DEFAULT_TEXT = "Hey computer, could you please turn on the lights in the living room?"


def coerce_opt_value(raw: str) -> object:
    lowered = raw.lower()
    if lowered in ("true", "false"):
        return lowered == "true"
    if lowered == "none":
        return None
    for cast in (int, float):
        try:
            return cast(raw)
        except ValueError:
            continue
    return raw


def parse_opts(items: list[str]) -> dict[str, object]:
    opts: dict[str, object] = {}
    for item in items:
        if "=" not in item:
            raise ValueError(f"--opt must be KEY=VALUE, got {item!r}")
        key, _, raw_value = item.partition("=")
        opts[key] = coerce_opt_value(raw_value)
    return opts


def accepted_param_names(backend_cls: type) -> set[str] | None:
    """The backend's __init__ param names, or None if it takes **kwargs (accepts anything)."""
    parameters = inspect.signature(backend_cls.__init__).parameters
    if any(p.kind is inspect.Parameter.VAR_KEYWORD for p in parameters.values()):
        return None
    return set(parameters) - {"self"}


def build_config(
    backend_cls: type,
    backend_name: str,
    device: str,
    opt_items: list[str],
) -> dict[str, object]:
    """Common CLI flags map to conventional constructor kwarg names, but only get
    forwarded if the chosen backend's constructor actually accepts them. --opt
    values are never filtered: an unknown key there is a real user-facing error."""
    common = {"device": device}
    accepted = accepted_param_names(backend_cls)
    if accepted is not None:
        for key in list(common):
            if key not in accepted:
                logger.debug(
                    "backend %r has no %r parameter; ignoring --%s",
                    backend_name,
                    key,
                    key.replace("_", "-"),
                )
                del common[key]

    config = {**common, **parse_opts(opt_items)}
    return config
