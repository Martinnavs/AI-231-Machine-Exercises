"""Refractory/debounce state machine, extracted from
`vcm.pipeline.SlidingWindowPipeline` with its exact existing semantics
(sample-counted, decrement-by-stride before inference, `<=0` means armed).
See `ME2/docs/STREAMING-CONTRACT.md` for how this composes with
`vcm.streaming.policy.AcceptancePolicy` in the fixed pipeline order.
"""

from __future__ import annotations


class Debouncer:
    """`refractory_samples == 0` means the cooldown is always already
    elapsed (`armed` is always True), i.e. debounce is effectively
    disabled -- every matched stride emits."""

    def __init__(self, refractory_samples: int) -> None:
        self.refractory_samples = refractory_samples
        self._cooldown_samples_remaining = 0
        self.suppressed_count = 0

    def reset(self) -> None:
        self._cooldown_samples_remaining = 0
        self.suppressed_count = 0

    @property
    def cooldown_samples_remaining(self) -> int:
        return self._cooldown_samples_remaining

    @property
    def armed(self) -> bool:
        return self._cooldown_samples_remaining <= 0

    def tick(self, stride_samples: int) -> None:
        """Decrement the cooldown by one stride's worth of samples,
        floored at 0. Call exactly once per stride, before `gate()`."""
        if self._cooldown_samples_remaining > 0:
            self._cooldown_samples_remaining = max(
                0, self._cooldown_samples_remaining - stride_samples
            )

    def gate(self, matched: bool) -> bool:
        """Call once per stride, after `tick()`. `matched` is whether this
        stride's evidence (e.g. an accepted `PolicyDecision`) is a
        trigger candidate. Returns True iff it should actually be emitted
        now -- which also rearms the cooldown to `refractory_samples`.
        Returns False without side effect if `matched` is False;
        increments `suppressed_count` if `matched` is True but the
        cooldown had not yet fully elapsed."""
        if not matched:
            return False
        if self.armed:
            self._cooldown_samples_remaining = self.refractory_samples
            return True
        self.suppressed_count += 1
        return False
