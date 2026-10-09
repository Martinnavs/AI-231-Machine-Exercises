"""Audio sources for the streaming feature: a file-replay source for
tests/eval and a live-microphone source for the real runner. See
`ME2/docs/STREAMING-CONTRACT.md` section 1 for the `AudioSource` shape
and how `is_realtime` selects the runner-loop's backpressure mode.

SECURITY: `MicrophoneSource`/`open_microphone_source` execute an
operator-supplied capture command via `subprocess.Popen` with a parsed
argv list and `shell=False` ALWAYS -- never a shell string. The operator
who supplies `--mic-command` (or `DEFAULT_MIC_COMMAND`) is the trust
boundary: this module never interprets that string through a shell, but
it also does not sanitize or restrict what the resulting argv can be, so
whoever controls that command string controls what process gets spawned.
"""

from __future__ import annotations

import shlex
import subprocess
import sys
import time
from typing import Iterator, Protocol, runtime_checkable

import numpy as np
import torchaudio

from ...common.features import SAMPLE_RATE

DEFAULT_MIC_COMMAND = "arecord -f S16_LE -r 16000 -c 1 -t raw -"

_S16_SCALE = 32768.0


class MicrophoneUnavailableError(RuntimeError):
    """Raised when the microphone-capture subprocess is missing or exits
    on its own before/without producing audio. Callers should fall back
    to `--source <wav>` or supply a different `--mic-command`."""


@runtime_checkable
class AudioSource(Protocol):
    is_realtime: bool

    def blocks(self) -> Iterator[np.ndarray]:
        """Yields float32 mono blocks at `SAMPLE_RATE`. Blocks are
        exactly `block_samples` long except possibly the final block."""
        ...

    def close(self) -> None: ...

    def __enter__(self) -> "AudioSource": ...

    def __exit__(self, *exc_info) -> None: ...


class RawPcmStreamSource:
    """Wraps a raw S16_LE mono 16kHz byte stream (any object with a
    `.read(n)` method returning `bytes`, e.g. a subprocess's stdout pipe
    or an in-memory `io.BytesIO`) as an `AudioSource` of float32 blocks.

    Tolerates short/partial reads from `stream.read()` (accumulates until
    a full block is available) and an odd trailing byte at EOF (that
    final unpaired byte is dropped, not raised on).
    """

    is_realtime = False

    def __init__(self, stream, block_samples: int) -> None:
        if block_samples <= 0:
            raise ValueError(f"block_samples must be > 0, got {block_samples}")
        self._stream = stream
        self._block_samples = block_samples
        self._block_bytes = block_samples * 2
        self._closed = False

    def _read_exact_or_less(self, n: int) -> bytes:
        """Reads up to `n` bytes, issuing repeated `.read()` calls to
        tolerate short reads, stopping early only at EOF (an empty
        `.read()` result)."""
        chunks = []
        remaining = n
        while remaining > 0:
            chunk = self._stream.read(remaining)
            if not chunk:
                break
            chunks.append(chunk)
            remaining -= len(chunk)
        return b"".join(chunks)

    def blocks(self) -> Iterator[np.ndarray]:
        while True:
            raw = self._read_exact_or_less(self._block_bytes)
            if not raw:
                return
            usable_len = len(raw) - (len(raw) % 2)
            if usable_len == 0:
                return
            raw = raw[:usable_len]
            samples_i16 = np.frombuffer(raw, dtype="<i2")
            block = (samples_i16.astype(np.float32)) / _S16_SCALE
            yield block
            if usable_len < self._block_bytes:
                return

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        closer = getattr(self._stream, "close", None)
        if closer is not None:
            closer()

    def __enter__(self) -> "RawPcmStreamSource":
        return self

    def __exit__(self, *exc_info) -> None:
        self.close()


class WavFileSource:
    """Loads a wav file via `torchaudio`, downmixes to mono, resamples to
    `SAMPLE_RATE` when needed (warns on stderr when it does), and yields
    it in `block_samples`-sized float32 blocks. Follows the same
    load/downmix/resample idiom as
    `vcm.evaluate.evaluate_slot_eval_set`.

    `realtime=True` self-paces block delivery to (approximately)
    wall-clock playback speed, for demoing/testing the runner's
    realtime-vs-file loop behavior against a file; `is_realtime` itself
    stays `False` regardless, because this source still guarantees every
    sample is delivered exactly once (no skip-and-count drops), which is
    the file-replay contract per `STREAMING-CONTRACT.md` section 1.
    """

    is_realtime = False

    def __init__(self, path, block_samples: int, realtime: bool = False) -> None:
        if block_samples <= 0:
            raise ValueError(f"block_samples must be > 0, got {block_samples}")
        self._block_samples = block_samples
        self._realtime = realtime
        self._closed = False

        waveform, sample_rate = torchaudio.load(str(path))
        if waveform.dim() == 2:
            waveform = waveform.mean(dim=0)
        if sample_rate != SAMPLE_RATE:
            print(
                f"WavFileSource: resampling {path} from {sample_rate}Hz to "
                f"{SAMPLE_RATE}Hz",
                file=sys.stderr,
            )
            waveform = torchaudio.functional.resample(waveform, sample_rate, SAMPLE_RATE)

        self._samples = waveform.numpy().astype(np.float32)

    def blocks(self) -> Iterator[np.ndarray]:
        start_time = time.monotonic()
        samples_emitted = 0
        n = self._samples.shape[0]
        pos = 0
        while pos < n:
            end = min(pos + self._block_samples, n)
            block = self._samples[pos:end]
            if self._realtime:
                samples_emitted += block.shape[0]
                target_elapsed = samples_emitted / SAMPLE_RATE
                actual_elapsed = time.monotonic() - start_time
                sleep_s = target_elapsed - actual_elapsed
                if sleep_s > 0:
                    time.sleep(sleep_s)
            yield block
            pos = end

    def close(self) -> None:
        self._closed = True

    def __enter__(self) -> "WavFileSource":
        return self

    def __exit__(self, *exc_info) -> None:
        self.close()


class MicrophoneSource:
    """Live-microphone `AudioSource`: spawns `command` (an already-parsed
    argv list) as a subprocess producing raw S16_LE mono 16kHz PCM on
    stdout, and wraps that stdout pipe in a `RawPcmStreamSource`.

    The process is always spawned with `shell=False`; `command` must
    therefore already be a list of argv tokens (see
    `open_microphone_source`, which does the `shlex.split` for a
    string-form `--mic-command`). `is_realtime=True`: this source never
    yields `None`/ends on its own under normal operation, it just blocks
    until the driver has more audio.
    """

    is_realtime = True

    # A few seconds is generous for a well-behaved capture process to react
    # to SIGTERM, while still bounding close()'s worst case for a process
    # that ignores it -- the shutdown path must never hang indefinitely.
    _TERMINATE_TIMEOUT_S = 3.0

    def __init__(self, command: list[str], block_samples: int) -> None:
        try:
            process = subprocess.Popen(
                command,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                shell=False,
            )
        except OSError as exc:
            raise MicrophoneUnavailableError(
                f"failed to launch microphone command {command!r}: {exc}. "
                "Use --source <wav> to replay a file instead, or pass a "
                "working --mic-command."
            ) from exc

        try:
            # Give the process a brief moment to fail fast (missing device,
            # bad args) before we commit to treating it as a live source.
            time.sleep(0.05)
            early_returncode = process.poll()
            if early_returncode is not None and early_returncode != 0:
                stderr = b""
                if process.stderr is not None:
                    stderr = process.stderr.read() or b""
                raise MicrophoneUnavailableError(
                    f"microphone command {command!r} exited immediately with "
                    f"code {early_returncode} ({stderr.decode(errors='replace').strip()}). "
                    "Use --source <wav> to replay a file instead, or pass a "
                    "working --mic-command."
                )

            self._process = process
            self._stream = RawPcmStreamSource(process.stdout, block_samples)
        except BaseException:
            # We're about to raise instead of returning a wrapped
            # MicrophoneSource -- no object will exist afterward for
            # close() to reach, so close the pipes here or they leak.
            if process.stdout is not None:
                process.stdout.close()
            if process.stderr is not None:
                process.stderr.close()
            process.wait()
            raise

    def blocks(self) -> Iterator[np.ndarray]:
        yield from self._stream.blocks()

    def close(self) -> None:
        if self._process.poll() is None:
            self._process.terminate()
            try:
                self._process.wait(timeout=self._TERMINATE_TIMEOUT_S)
            except subprocess.TimeoutExpired:
                # Child ignored SIGTERM -- escalate rather than hang the
                # runner's shutdown path forever.
                self._process.kill()
        self._stream.close()
        self._process.wait()
        if self._process.stderr is not None:
            self._process.stderr.close()

    def __enter__(self) -> "MicrophoneSource":
        return self

    def __exit__(self, *exc_info) -> None:
        self.close()


def open_microphone_source(
    command: str | list[str] = DEFAULT_MIC_COMMAND,
    block_samples: int = 1024,
) -> MicrophoneSource:
    """Factory for `MicrophoneSource`. `command` may be a pre-parsed argv
    list, or a string that gets `shlex.split` (never a shell string --
    `shell=False` is always used for the resulting `subprocess.Popen`)."""
    argv = shlex.split(command) if isinstance(command, str) else list(command)
    return MicrophoneSource(argv, block_samples)
