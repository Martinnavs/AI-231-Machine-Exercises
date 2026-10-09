"""The listening-gate seam: whether the runner is currently inside a
bounded listening period. `SpacebarGate` is the manual stand-in: the
operator presses SPACE to open a `period_s`-long period, a re-press
discards and restarts it, and it auto-closes at the period end. The
trained wake-word model satisfies the same `ListeningGate` protocol with
zero policy/runner changes -- see `WakeWordGate` in `wakeword_gate.py`
(feature `wakeword-gate`, `feature-engineering/wakeword-gate/SPEC.md`).
See `ME2/docs/STREAMING-CONTRACT.md` for the pipeline position (the gate
sits between the observation stream and the acceptance policy).

SECURITY: the only input is the trusted local operator's own TTY
keystrokes, read one byte at a time -- no parsing, no shell, no
subprocess. The raw-mode termios change is confined to the operator's
own stdin and is always restored: `close()` plus an `atexit` backstop.
"""

from __future__ import annotations

import atexit
import os
import select
import sys
import termios
from dataclasses import dataclass
from typing import Any, Optional, Protocol

import numpy as np

from ...common.features import SAMPLE_RATE

# Default confidence threshold (softmax probability of the `_wakeword_`
# class) `WakeWordGate` opens a period at. Single source of truth: defined
# here (not in `wakeword_gate.py`) so `resolve_gate` can default to it
# without importing that module -- see the `GATE_REGISTRY` note below for
# why that import direction is avoided.
DEFAULT_WAKEWORD_THRESHOLD: float = 0.9


@dataclass(frozen=True)
class GateState:
    is_open: bool
    open_at_samples: Optional[int]  # None when closed


class ListeningGate(Protocol):
    """The seam the future wake-word gate will satisfy. `window` is the
    current ring-buffer snapshot (the same audio the model decoded); a
    WakeWordGate consumes it, SpacebarGate ignores it."""

    def poll(self, samples_seen: int, window: Optional[np.ndarray] = None) -> GateState: ...
    def close(self) -> None: ...


class GateUnavailableError(Exception):
    """Raised at construction when stdin is not a usable TTY (piped/
    redirected input, or no file descriptor at all). Callers should run
    in an interactive shell, pass `--gate none`, or replay a file with
    `--source <wav>` -- mirroring `MicrophoneUnavailableError`'s
    fail-fast handling (actionable message, never a traceback)."""


_GATE_UNAVAILABLE_MESSAGE = (
    "stdin is not a usable TTY, so the spacebar listening gate cannot "
    "read keypresses. Run in an interactive shell, pass --gate none, or "
    "replay a file with --source <wav>."
)


class SpacebarGate:
    """`ListeningGate` whose open signal is the operator pressing SPACE
    on their own terminal. Enters minimal raw mode on `stdin` (clears
    `ICANON|ECHO`, keeps `ISIG`) and, per `poll`, drains pending
    keypresses with `select(timeout=0)` + `os.read(fd, 1)`: a press of
    `key` opens a period at the current `samples_seen` (a press while
    already open restarts it there), every other byte is read and
    discarded, and the period auto-closes at
    `open_at + int(period_s * SAMPLE_RATE)`. `close()` is idempotent,
    restores the saved termios attributes, and is registered with
    `atexit` as a backstop."""

    def __init__(self, stdin, *, key: bytes = b" ", period_s: float = 5.0) -> None:
        """Minimal raw TTY (ICANON|ECHO off, ISIG on). Raises
        GateUnavailableError if stdin is not a usable TTY."""
        try:
            fd = stdin.fileno()
        except (AttributeError, OSError) as exc:
            raise GateUnavailableError(_GATE_UNAVAILABLE_MESSAGE) from exc
        try:
            original = termios.tcgetattr(fd)
        except (termios.error, OSError) as exc:
            # termios.error is a plain Exception, not an OSError subclass.
            raise GateUnavailableError(_GATE_UNAVAILABLE_MESSAGE) from exc

        raw = list(original)
        # lflag: clear ICANON|ECHO but keep ISIG, so Ctrl-C still
        # delivers SIGINT instead of arriving as a gate key.
        raw[3] = raw[3] & ~(termios.ICANON | termios.ECHO)
        termios.tcsetattr(fd, termios.TCSANOW, raw)

        self._fd = fd
        self._key = key
        self._period_s = period_s
        self._original_attrs = original
        self._open_at_samples: Optional[int] = None
        self._closed = False
        atexit.register(self.close)

    def poll(self, samples_seen: int, window=None) -> GateState:
        """select(timeout=0) + os.read(fd, 1). Space -> open at
        samples_seen (discard-and-restart). Auto-close when
        samples_seen >= open_at + int(period_s * SAMPLE_RATE)."""
        if self._closed:
            return GateState(is_open=False, open_at_samples=None)

        if self._open_at_samples is not None and samples_seen >= (
            self._open_at_samples + int(self._period_s * SAMPLE_RATE)
        ):
            self._open_at_samples = None

        while True:
            readable, _, _ = select.select([self._fd], [], [], 0.0)
            if not readable:
                break
            byte = os.read(self._fd, 1)
            if not byte:
                break  # EOF: stop, or a closed stdin would spin the drain loop
            if byte == self._key:
                self._open_at_samples = samples_seen
            # any other byte: read-and-discard, no state change

        return GateState(
            is_open=self._open_at_samples is not None,
            open_at_samples=self._open_at_samples,
        )

    def close(self) -> None:
        """Idempotent; restores the saved termios attributes."""
        if self._closed:
            return
        termios.tcsetattr(self._fd, termios.TCSANOW, self._original_attrs)
        self._open_at_samples = None
        self._closed = True


class AlwaysOpenGate:
    """`ListeningGate` with no wake word: the period is always open, re-pressed at every poll, so it never times out and
    `EndpointedPolicy` reopens it right after each accept (`continuous`). Every window is decoded -- the always-on
    baseline for the wake-word gate (soak test without the gate, docs/SOAK-TEST.md)."""

    continuous = True

    def __init__(self, stdin=None, *, period_s: float = 5.0) -> None:
        self.period_s = period_s

    def poll(self, samples_seen: int, window: Optional[np.ndarray] = None) -> GateState:
        return GateState(is_open=True, open_at_samples=samples_seen)

    def close(self) -> None:
        pass


# Holds "spacebar" and "always" here. "wakeword" is registered as a module-level
# side effect by `config.py` (`GATE_REGISTRY["wakeword"] = WakeWordGate`),
# not imported here directly: `wakeword_gate.py` imports `ListeningGate`/
# `GateState`/`DEFAULT_WAKEWORD_THRESHOLD` from this module, so importing
# `wakeword_gate` from here would be a cycle. `config.py` is where the
# other registries (`MODEL_REGISTRY`, `POLICY_REGISTRY`) already live, so
# that's where this registration lives too.
GATE_REGISTRY: dict[str, type[ListeningGate]] = {"spacebar": SpacebarGate, "always": AlwaysOpenGate}


def resolve_gate(
    name: str,
    *,
    period_s: float,
    stdin=sys.stdin,
    wakeword_backend: Optional[Any] = None,
    wakeword_threshold: float = DEFAULT_WAKEWORD_THRESHOLD,
    wakeword_poll_s: Optional[float] = None,
) -> ListeningGate:
    """Unknown name -> SystemExit naming sorted(GATE_REGISTRY). `name ==
    "wakeword"` takes a different constructor shape (a `WakewordInference
    Backend` instead of `stdin`) -- `wakeword_backend` must be supplied by
    the caller in that case (an internal-wiring error otherwise, not a
    user-facing one: `__main__.py` always builds the backend before
    calling this for `--gate wakeword`)."""
    try:
        gate_cls = GATE_REGISTRY[name]
    except KeyError:
        raise SystemExit(
            f"unknown --gate {name!r}; choices are {sorted(GATE_REGISTRY)}"
        ) from None
    if name == "wakeword":
        if wakeword_backend is None:
            raise SystemExit(
                "internal error: --gate wakeword requires a wakeword_backend "
                "(resolve_gate was called without constructing one first)"
            )
        return gate_cls(wakeword_backend, threshold=wakeword_threshold, period_s=period_s, poll_step_s=wakeword_poll_s)
    return gate_cls(stdin, period_s=period_s)
