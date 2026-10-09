"""CTC forced alignment and segment-local confidence metrics.

This module is deliberately decoder-adjacent rather than streaming code.  It
scores one grammar candidate against log-posteriors that have already been
computed for a complete listening period; it never invokes a model.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from . import alphabet


@dataclass(frozen=True)
class ForcedAlignment:
    """Viterbi CTC alignment for one fixed transcript."""

    token_ids: tuple[int, ...]
    state_path: tuple[int, ...]
    frame_token_ids: tuple[int, ...]
    log_probability: float
    start_frame: int
    end_frame: int

    @property
    def full_path_mean(self) -> float:
        return self.log_probability / len(self.frame_token_ids)


@dataclass(frozen=True)
class SegmentScores:
    """Position-invariant candidate evidence derived from one alignment."""

    full_path_mean: float
    segment_path_mean: float
    token_blank_margin: float
    start_frame: int
    end_frame: int


def _extended_ctc_labels(token_ids: tuple[int, ...]) -> tuple[int, ...]:
    labels = [alphabet.BLANK_ID]
    for token_id in token_ids:
        labels.extend((token_id, alphabet.BLANK_ID))
    return tuple(labels)


def force_align(logp: np.ndarray, text: str) -> ForcedAlignment:
    """Return the best standard-CTC Viterbi path for ``text``.

    The transition set is stay, advance one state, and skip a blank only when
    the destination label is non-blank and differs from the source label.  In
    particular, repeated characters require an intervening blank.
    """
    logp = np.asarray(logp, dtype=np.float64)
    if logp.ndim != 2:
        raise ValueError(f"expected (T, C) logp, got {logp.shape}")
    if logp.shape[0] == 0:
        raise ValueError("cannot align an empty logp sequence")

    token_ids = tuple(alphabet.encode(text))
    if not token_ids:
        raise ValueError("cannot align an empty transcript")
    labels = _extended_ctc_labels(token_ids)
    frames, states = logp.shape[0], len(labels)
    scores = np.full((frames, states), -np.inf, dtype=np.float64)
    back = np.full((frames, states), -1, dtype=np.int32)
    scores[0, 0] = logp[0, alphabet.BLANK_ID]
    scores[0, 1] = logp[0, labels[1]]

    for frame in range(1, frames):
        for state, label in enumerate(labels):
            sources = [state]
            if state >= 1:
                sources.append(state - 1)
            if state >= 2 and label != alphabet.BLANK_ID and label != labels[state - 2]:
                sources.append(state - 2)
            source = max(sources, key=lambda candidate: scores[frame - 1, candidate])
            if np.isfinite(scores[frame - 1, source]):
                scores[frame, state] = scores[frame - 1, source] + logp[frame, label]
                back[frame, state] = source

    terminal_states = [states - 1, states - 2]
    final_state = max(terminal_states, key=lambda state: scores[-1, state])
    final_score = float(scores[-1, final_state])
    if not np.isfinite(final_score):
        raise ValueError(f"impossible CTC alignment for {text!r} in {frames} frames")

    state_path = [final_state]
    for frame in range(frames - 1, 0, -1):
        previous = int(back[frame, state_path[-1]])
        if previous < 0:
            raise AssertionError("finite CTC path has no predecessor")
        state_path.append(previous)
    state_path.reverse()
    frame_token_ids = tuple(labels[state] for state in state_path)
    nonblank = [i for i, token_id in enumerate(frame_token_ids) if token_id != alphabet.BLANK_ID]
    if not nonblank:
        raise AssertionError("non-empty transcript aligned only to blanks")
    return ForcedAlignment(
        token_ids=token_ids,
        state_path=tuple(state_path),
        frame_token_ids=frame_token_ids,
        log_probability=final_score,
        start_frame=nonblank[0],
        end_frame=nonblank[-1],
    )


def segment_scores(logp: np.ndarray, alignment: ForcedAlignment) -> SegmentScores:
    """Score aligned speech locally, retaining only CTC's outer blank context.

    ``segment_path_mean`` averages the Viterbi emissions between the first and
    last aligned non-blank.  ``token_blank_margin`` compares every aligned
    character frame with the blank posterior at that same frame.  Both remove
    the incidental amount of leading/trailing silence from the decision.
    """
    logp = np.asarray(logp, dtype=np.float64)
    frame_ids = np.asarray(alignment.frame_token_ids, dtype=np.int64)
    frame_indices = np.arange(len(frame_ids))
    emissions = logp[frame_indices, frame_ids]
    start, end = alignment.start_frame, alignment.end_frame
    nonblank = frame_ids != alphabet.BLANK_ID
    return SegmentScores(
        full_path_mean=float(emissions.mean()),
        segment_path_mean=float(emissions[start : end + 1].mean()),
        token_blank_margin=float((emissions[nonblank] - logp[nonblank, alphabet.BLANK_ID]).mean()),
        start_frame=start,
        end_frame=end,
    )
