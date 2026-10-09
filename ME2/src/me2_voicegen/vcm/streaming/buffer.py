"""Preallocated, thread-safe ring buffer of the last `window_samples`
float32 audio samples fed to it. See `ME2/docs/STREAMING-CONTRACT.md`.
"""

from __future__ import annotations

import threading

import numpy as np


class RingBuffer:
    """Fixed-capacity circular buffer over the most recently written
    `window_samples` samples. `write` and `snapshot` are both guarded by
    the same lock, so a `snapshot()` racing a concurrent `write()` never
    sees a torn/interleaved buffer -- it sees either the pre- or
    post-write state, never a partial mix of the two.
    """

    def __init__(self, window_samples: int) -> None:
        if window_samples <= 0:
            raise ValueError(f"window_samples must be > 0, got {window_samples}")
        self.window_samples = window_samples
        self._data = np.zeros(window_samples, dtype=np.float32)
        self._write_pos = 0
        self._filled = False
        self.samples_written = 0
        self._lock = threading.Lock()

    def reset(self) -> None:
        with self._lock:
            self._data[:] = 0.0
            self._write_pos = 0
            self._filled = False
            self.samples_written = 0

    def write(self, block: np.ndarray) -> None:
        """Append `block` (any 1-D-reshapeable array of samples, any
        length) to the buffer. A block longer than `window_samples`
        contributes only its last `window_samples` samples -- everything
        earlier in it would be immediately overwritten anyway."""
        block = np.asarray(block, dtype=np.float32).reshape(-1)
        n = block.size
        if n == 0:
            return

        with self._lock:
            if n >= self.window_samples:
                self._data[:] = block[-self.window_samples :]
                self._write_pos = 0
                self._filled = True
            else:
                end = self._write_pos + n
                if end <= self.window_samples:
                    self._data[self._write_pos : end] = block
                else:
                    first_len = self.window_samples - self._write_pos
                    self._data[self._write_pos :] = block[:first_len]
                    self._data[: end - self.window_samples] = block[first_len:]
                if not self._filled and end >= self.window_samples:
                    self._filled = True
                self._write_pos = end % self.window_samples
            self.samples_written += n

    def snapshot(self) -> np.ndarray:
        """Last `min(samples_written, window_samples)` samples, oldest
        first. Before the buffer has filled once, returns only the valid
        prefix actually written so far -- never the preallocated zeros
        past it, so a caller (e.g. `LogMelFeatureExtractor`, which accepts
        any input length) never mistakes stale/never-written zeros for
        real silence."""
        with self._lock:
            if not self._filled:
                return self._data[: self._write_pos].copy()
            pos = self._write_pos
            if pos == 0:
                return self._data.copy()
            return np.concatenate([self._data[pos:], self._data[:pos]])

    def snapshot_range(self, start_samples: int, end_samples: int) -> np.ndarray:
        """Return an exact absolute sample interval still retained in the ring."""
        with self._lock:
            oldest = max(0, self.samples_written - self.window_samples)
            if start_samples < oldest or end_samples > self.samples_written or start_samples >= end_samples:
                raise ValueError(
                    f"requested [{start_samples}, {end_samples}) is not retained "
                    f"in [{oldest}, {self.samples_written})"
                )
            if not self._filled:
                return self._data[start_samples:end_samples].copy()
            ordered = self._data.copy() if self._write_pos == 0 else np.concatenate([self._data[self._write_pos:], self._data[:self._write_pos]])
            return ordered[start_samples - oldest : end_samples - oldest].copy()
