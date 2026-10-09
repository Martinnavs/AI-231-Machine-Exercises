"""Fast, CPU-only tests for `vcm.streaming.gate` -- `GateState`,
`SpacebarGate` (state machine via a fake key reader, non-TTY fail-fast,
termios restore, atexit backstop), and `resolve_gate`. No real TTY:
`termios`/`select`/`os.read` are monkeypatched (SPEC Assumption 2 --
live single-byte TTY delivery is verified only in the manual handoff
step).
"""

from __future__ import annotations

import dataclasses
import io
import os
import termios

import pytest

from me2_voicegen.common.features import SAMPLE_RATE
from me2_voicegen.vcm.streaming import gate
from me2_voicegen.vcm.streaming.gate import (
    GATE_REGISTRY,
    GateState,
    GateUnavailableError,
    SpacebarGate,
    resolve_gate,
)


class _FakeStdin:
    """File-like stdin stand-in with a `fileno()` but no TTY -- these
    tests always monkeypatch `termios`, so the fd number never touches
    a real descriptor."""

    def __init__(self, fd: int = 3) -> None:
        self._fd = fd

    def fileno(self) -> int:
        return self._fd


# A plausible cooked-mode lflag: ICANON|ECHO|ISIG all set. Iflag/oflag/
# cflag stay zero -- the gate must not touch them.
_ORIGINAL_ATTRS = [0, 0, 0, termios.ICANON | termios.ECHO | termios.ISIG]


def _install_fake_termios(monkeypatch) -> list:
    """Monkeypatches `termios.tcgetattr`/`tcsetattr`; returns the list of
    `tcsetattr` calls as (fd, when, attrs)."""
    setattr_calls = []
    monkeypatch.setattr(gate.termios, "tcgetattr", lambda fd: list(_ORIGINAL_ATTRS))
    monkeypatch.setattr(
        gate.termios,
        "tcsetattr",
        lambda fd, when, attrs: setattr_calls.append((fd, when, list(attrs))),
    )
    return setattr_calls


def _install_fake_reader(monkeypatch, pressed: bytes = b""):
    """Monkeypatches `select.select` + `os.read` to serve one byte per
    read from a queue seeded with `pressed`; appends to the returned
    queue inject later keypresses. Returns (queue, read_calls)."""
    # single-byte slices -- `list(pressed)` would yield ints, not bytes
    queue = [pressed[i : i + 1] for i in range(len(pressed))]
    read_calls = []

    def fake_select(rlist, wlist, xlist, timeout=None):
        return ([rlist[0]] if queue else [], [], [])

    def fake_read(fd, size):
        read_calls.append((fd, size))
        return queue.pop(0) if queue else b""

    monkeypatch.setattr(gate.select, "select", fake_select)
    monkeypatch.setattr(gate.os, "read", fake_read)
    return queue, read_calls


def _build_spacebar_gate(monkeypatch, pressed: bytes = b"", period_s: float = 5.0):
    setattr_calls = _install_fake_termios(monkeypatch)
    registered = []
    monkeypatch.setattr(gate.atexit, "register", registered.append)
    queue, read_calls = _install_fake_reader(monkeypatch, pressed)
    g = SpacebarGate(_FakeStdin(), period_s=period_s)
    return g, setattr_calls, registered, queue, read_calls


def test_gate_state_fields_and_frozen():
    open_state = GateState(is_open=True, open_at_samples=123)
    assert open_state.is_open is True
    assert open_state.open_at_samples == 123
    closed = GateState(is_open=False, open_at_samples=None)
    assert closed.open_at_samples is None
    with pytest.raises(dataclasses.FrozenInstanceError):
        open_state.open_at_samples = 456


def test_starts_closed_and_reads_nothing_when_idle(monkeypatch):
    g, _, _, _, read_calls = _build_spacebar_gate(monkeypatch)
    assert g.poll(0) == GateState(is_open=False, open_at_samples=None)
    assert read_calls == []  # select says nothing pending -> no os.read at all


def test_space_press_opens_period_at_current_samples_seen(monkeypatch):
    g, _, _, _, _ = _build_spacebar_gate(monkeypatch, pressed=b" ")
    assert g.poll(1000) == GateState(is_open=True, open_at_samples=1000)
    # still open at the same open_at on later polls (no new keypress)
    assert g.poll(1250) == GateState(is_open=True, open_at_samples=1000)


def test_non_space_bytes_are_read_and_discarded_without_state_change(monkeypatch):
    g, _, _, _, read_calls = _build_spacebar_gate(monkeypatch, pressed=b"xy")
    assert g.poll(0) == GateState(is_open=False, open_at_samples=None)
    assert g.poll(100) == GateState(is_open=False, open_at_samples=None)
    assert read_calls == [(3, 1), (3, 1)]  # both bytes consumed as 1-byte reads


def test_repress_while_open_restarts_period_at_new_samples_seen(monkeypatch):
    g, _, _, queue, _ = _build_spacebar_gate(monkeypatch, pressed=b" ")
    assert g.poll(1000) == GateState(is_open=True, open_at_samples=1000)
    queue.append(b" ")
    # discard & restart: open_at moves to the new press position
    assert g.poll(2000) == GateState(is_open=True, open_at_samples=2000)


def test_double_space_press_in_one_poll_opens_one_period_at_poll_position(monkeypatch):
    # Pins ticket 01's logged deviation: poll() drains ALL keypresses
    # pending at a poll boundary, so a double press within one stride
    # opens a single period at THIS poll's position. A strict
    # one-read-per-poll reading would leave the second byte to re-open
    # (restart) at the next poll instead.
    g, _, _, _, read_calls = _build_spacebar_gate(monkeypatch, pressed=b"  ")
    assert g.poll(1000) == GateState(is_open=True, open_at_samples=1000)
    assert len(read_calls) == 2  # both bytes drained at this poll
    assert g.poll(1500) == GateState(is_open=True, open_at_samples=1000)


def test_period_auto_closes_exactly_at_open_at_plus_period(monkeypatch):
    g, _, _, _, _ = _build_spacebar_gate(monkeypatch, pressed=b" ", period_s=5.0)
    g.poll(1000)  # open at 1000
    close_at = 1000 + int(5.0 * SAMPLE_RATE)
    assert g.poll(close_at - 1) == GateState(is_open=True, open_at_samples=1000)
    assert g.poll(close_at) == GateState(is_open=False, open_at_samples=None)


def test_expiry_and_press_in_same_poll_reopen_at_that_position(monkeypatch):
    g, _, _, queue, _ = _build_spacebar_gate(monkeypatch, period_s=0.25)
    queue.append(b" ")
    g.poll(0)  # open at 0
    close_at = int(0.25 * SAMPLE_RATE)
    assert g.poll(close_at - 1).is_open is True
    queue.append(b" ")
    # the settled flush/reopen edge: period expires AND a press is
    # pending on the same poll -> open again at this very position
    assert g.poll(close_at) == GateState(is_open=True, open_at_samples=close_at)


def test_poll_after_close_returns_closed_and_never_reads(monkeypatch):
    g, _, _, _, read_calls = _build_spacebar_gate(monkeypatch, pressed=b" ")
    assert g.poll(1000).is_open is True
    g.close()
    n_reads = len(read_calls)
    assert g.poll(2000) == GateState(is_open=False, open_at_samples=None)
    assert len(read_calls) == n_reads  # no read on the restored (cooked) TTY


def test_close_restores_saved_termios_exactly_and_is_idempotent(monkeypatch):
    g, setattr_calls, _, _, _ = _build_spacebar_gate(monkeypatch, pressed=b" ")
    g.poll(1000)
    n_after_init = len(setattr_calls)
    assert n_after_init == 1
    g.close()
    assert len(setattr_calls) == n_after_init + 1
    fd, when, attrs = setattr_calls[-1]
    assert fd == 3
    assert when == termios.TCSANOW
    assert attrs == _ORIGINAL_ATTRS  # the exact saved attributes, bit for bit
    g.close()  # idempotent: no second restore
    assert len(setattr_calls) == n_after_init + 1


def test_constructor_enters_minimal_raw_mode_keeps_isig(monkeypatch):
    g, setattr_calls, _, _, _ = _build_spacebar_gate(monkeypatch)
    fd, when, attrs = setattr_calls[0]
    assert when == termios.TCSANOW
    assert attrs[0:3] == _ORIGINAL_ATTRS[0:3]  # iflag/oflag/cflag untouched
    assert attrs[3] == (_ORIGINAL_ATTRS[3] & ~(termios.ICANON | termios.ECHO))
    assert attrs[3] & termios.ISIG  # ISIG kept -> Ctrl-C still delivers SIGINT


def test_constructor_registers_close_with_atexit(monkeypatch):
    g, _, registered, _, _ = _build_spacebar_gate(monkeypatch)
    assert registered == [g.close]


def test_non_tty_stdin_raises_gate_unavailable_with_actionable_message(monkeypatch):
    setattr_calls = _install_fake_termios(monkeypatch)
    registered = []
    monkeypatch.setattr(gate.atexit, "register", registered.append)

    with pytest.raises(GateUnavailableError) as excinfo:
        SpacebarGate(io.BytesIO())  # fileno() raises OSError -> no usable TTY

    message = str(excinfo.value)
    assert "TTY" in message
    assert "interactive" in message
    assert "--gate none" in message
    assert "--source" in message  # the file-replay alternative
    assert setattr_calls == []  # never entered raw mode
    assert registered == []  # no atexit backstop for a gate that never opened


def test_stdin_without_fileno_raises_gate_unavailable(monkeypatch):
    _install_fake_termios(monkeypatch)
    with pytest.raises(GateUnavailableError):
        SpacebarGate(object())


def test_piped_fd_raises_gate_unavailable():
    """The realistic piped-stdin case of SPEC edge 8: `fileno()` works
    but the fd is not a TTY, so the real `termios.tcgetattr` fails
    (no monkeypatch here -- this is the actual failure path)."""
    read_fd, write_fd = os.pipe()
    try:

        class _PipeStdin:
            def fileno(self) -> int:
                return read_fd

        with pytest.raises(GateUnavailableError):
            SpacebarGate(_PipeStdin())
    finally:
        os.close(read_fd)
        os.close(write_fd)


def test_resolve_gate_unknown_name_exits_naming_sorted_registry(monkeypatch):
    _install_fake_termios(monkeypatch)

    class _BoobyTrapStdin:
        def fileno(self) -> int:
            raise AssertionError("no gate may be constructed for an unknown name")

    with pytest.raises(SystemExit) as excinfo:
        resolve_gate("bogus", period_s=5.0, stdin=_BoobyTrapStdin())

    message = str(excinfo.value)
    assert "bogus" in message
    for name in sorted(GATE_REGISTRY):
        assert name in message


def test_resolve_gate_spacebar_returns_configured_gate(monkeypatch):
    setattr_calls = _install_fake_termios(monkeypatch)
    queue, _ = _install_fake_reader(monkeypatch)
    monkeypatch.setattr(gate.atexit, "register", lambda fn: None)

    g = resolve_gate("spacebar", period_s=0.25, stdin=_FakeStdin())
    assert isinstance(g, SpacebarGate)
    assert len(setattr_calls) == 1

    queue.append(b" ")
    assert g.poll(7) == GateState(is_open=True, open_at_samples=7)
    g.close()


def test_custom_key_is_honoured(monkeypatch):
    _install_fake_termios(monkeypatch)
    monkeypatch.setattr(gate.atexit, "register", lambda fn: None)
    queue, _ = _install_fake_reader(monkeypatch, b" ")

    g = SpacebarGate(_FakeStdin(), key=b"q")
    # with key=b"q", a space is just another ignored byte
    assert g.poll(0) == GateState(is_open=False, open_at_samples=None)
    queue.append(b"q")
    assert g.poll(100) == GateState(is_open=True, open_at_samples=100)
    g.close()
