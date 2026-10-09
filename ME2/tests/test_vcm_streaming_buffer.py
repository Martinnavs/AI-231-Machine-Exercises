"""Fast, CPU-only tests for `vcm.streaming.buffer.RingBuffer`."""

from __future__ import annotations

import threading

import numpy as np
import pytest

from me2_voicegen.vcm.streaming.buffer import RingBuffer


def test_ring_buffer_snapshot_before_fill_returns_only_valid_prefix():
    buf = RingBuffer(window_samples=10)
    buf.write(np.array([1.0, 2.0, 3.0], dtype=np.float32))
    snap = buf.snapshot()
    assert snap.tolist() == [1.0, 2.0, 3.0]
    assert buf.samples_written == 3


def test_ring_buffer_never_exceeds_capacity():
    buf = RingBuffer(window_samples=5)
    for _ in range(20):
        buf.write(np.array([1.0, 2.0], dtype=np.float32))
        assert buf.snapshot().size <= 5


def test_ring_buffer_chronological_order_across_wraparound():
    buf = RingBuffer(window_samples=5)
    # feed 1..12 in chunks of 2; after filling, the last 5 samples fed so
    # far must be returned oldest-first.
    values = list(range(1, 13))
    for i in range(0, len(values), 2):
        buf.write(np.array(values[i : i + 2], dtype=np.float32))
    snap = buf.snapshot()
    assert snap.tolist() == [8.0, 9.0, 10.0, 11.0, 12.0]
    assert buf.samples_written == 12


def test_ring_buffer_write_larger_than_capacity_keeps_tail():
    buf = RingBuffer(window_samples=4)
    buf.write(np.arange(10, dtype=np.float32))
    snap = buf.snapshot()
    assert snap.tolist() == [6.0, 7.0, 8.0, 9.0]


def test_ring_buffer_write_exactly_capacity_fills_in_one_shot():
    buf = RingBuffer(window_samples=4)
    buf.write(np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32))
    assert buf.snapshot().tolist() == [1.0, 2.0, 3.0, 4.0]
    buf.write(np.array([5.0], dtype=np.float32))
    assert buf.snapshot().tolist() == [2.0, 3.0, 4.0, 5.0]


def test_ring_buffer_write_empty_block_is_noop():
    buf = RingBuffer(window_samples=4)
    buf.write(np.array([], dtype=np.float32))
    assert buf.samples_written == 0
    assert buf.snapshot().size == 0


def test_ring_buffer_reset_clears_state():
    buf = RingBuffer(window_samples=4)
    buf.write(np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=np.float32))
    assert buf.samples_written == 5

    buf.reset()
    assert buf.samples_written == 0
    assert buf.snapshot().size == 0
    # buffer behaves exactly like a fresh one after reset.
    buf.write(np.array([9.0, 9.0], dtype=np.float32))
    assert buf.snapshot().tolist() == [9.0, 9.0]


def test_ring_buffer_rejects_non_positive_capacity():
    with pytest.raises(ValueError):
        RingBuffer(window_samples=0)


def test_ring_buffer_concurrent_write_and_snapshot_never_tears():
    """Basic thread-safety probe: a writer thread continuously writes
    monotonically increasing values while the main thread repeatedly
    snapshots; every snapshot must be internally consistent -- a
    contiguous run of consecutive integers -- never an interleaved mix of
    old and new writes half-copied."""
    capacity = 64
    buf = RingBuffer(window_samples=capacity)
    stop = threading.Event()
    errors: list[str] = []

    def writer():
        counter = 0
        block = 8
        while not stop.is_set():
            buf.write(np.arange(counter, counter + block, dtype=np.float32))
            counter += block

    t = threading.Thread(target=writer, daemon=True)
    t.start()
    try:
        for _ in range(500):
            snap = buf.snapshot()
            if snap.size > 1:
                diffs = np.diff(snap)
                if not np.all(diffs == 1.0):
                    errors.append(f"non-contiguous snapshot: {snap.tolist()}")
                    break
    finally:
        stop.set()
        t.join(timeout=2.0)

    assert errors == []
