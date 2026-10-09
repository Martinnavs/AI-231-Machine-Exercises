"""Streaming-inference building blocks for `vcm` (Task 01: ring buffer,
debounce, acceptance-policy seam). See `ME2/docs/STREAMING-CONTRACT.md` for
the full pipeline order and shared shapes; downstream tasks (audio source,
runner loop, CLI) import from here rather than from `vcm.pipeline` directly.
"""

from __future__ import annotations

from me2_voicegen.vcm.streaming.buffer import RingBuffer
from me2_voicegen.vcm.streaming.debounce import Debouncer
from me2_voicegen.vcm.streaming.policy import (
    AcceptancePolicy,
    PolicyDecision,
    ThresholdPolicy,
    WindowObservation,
)

__all__ = [
    "RingBuffer",
    "Debouncer",
    "AcceptancePolicy",
    "PolicyDecision",
    "ThresholdPolicy",
    "WindowObservation",
]
