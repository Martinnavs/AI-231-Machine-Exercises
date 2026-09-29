"""CTC prefix beam search constrained to a `common.grammar_core` character
trie (as compiled by a per-experiment grammar, e.g. `vcm.optiona.grammar` or
`vcm.optionb.grammar`).

Per decision (C) (see ticket .scratch/vcm-toy/tickets/03-grammar-decoder.md):
plain Python only, no `kaldifst` (present in the venv but only transitive
via `wetext`, not a declared dependency) and no `sherpa-onnx` (not
installed). `kaldifst`/a real WFST is a plausible future scale-up path if
the grammar ever grows past a few thousand phrases -- not needed here.

At each search step, a beam can only extend to characters that are children
of its current grammar-trie node (see `common.grammar_core.TrieNode`), so the
search space is pruned to grammar-valid paths by construction rather than
searched unconstrained and filtered after the fact.

Input/output contract: docs/VCM-CONTRACT.md section 7.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from me2_voicegen.common.grammar_core import Grammar, TrieNode

from . import alphabet as vcm_alphabet

NEG_INF = float("-inf")

SCORE_MODES = ("mean_frame", "per_char")


def _logsumexp(a: float, b: float) -> float:
    if a == NEG_INF:
        return b
    if b == NEG_INF:
        return a
    m = a if a > b else b
    return m + math.log(math.exp(a - m) + math.exp(b - m))


@dataclass
class BeamEntry:
    node: TrieNode
    pb: float = NEG_INF   # log-prob mass of alignments ending in blank
    pnb: float = NEG_INF  # log-prob mass of alignments ending in non-blank

    def total(self) -> float:
        return _logsumexp(self.pb, self.pnb)


@dataclass
class DecodeResult:
    intent: str | None
    slots: dict
    text: str
    confidence: float
    no_match: bool
    out_of_grammar_gap: float
    # Diagnostics + rejection metadata (docs/INCOMPLETE-GRAMMAR-REJECTION.md,
    # Step 2): all defaulted so existing construction/read sites stay valid.
    # `grammar_text` is the selected terminal phrase, which may differ from
    # the unconstrained greedy `text`. The raw scores are UNNORMALIZED beam
    # log masses (not /T), so `incomplete_gap` is invariant to trailing
    # blank padding. `rejection_reason` is only ever set when the gate is
    # enabled (`required_command_margin` is not None).
    grammar_text: str = ""
    rejection_reason: str | None = None
    incomplete_prefix: str | None = None
    incomplete_gap: float | None = None
    command_raw_score: float | None = None
    incomplete_raw_score: float | None = None


def _greedy_unconstrained(logp: np.ndarray) -> tuple[str, float]:
    """Unconstrained per-frame argmax path: CTC-collapsed text (this is the
    decoder's `text` field, independent of grammar/threshold) plus its mean
    per-frame log-probability (the out-of-grammar-gap baseline)."""
    if logp.shape[0] == 0:
        return "", NEG_INF
    argmax_ids = logp.argmax(axis=-1)
    scores = logp[np.arange(len(argmax_ids)), argmax_ids]
    mean_score = float(scores.mean())
    collapsed = vcm_alphabet.collapse(argmax_ids.tolist())
    return vcm_alphabet.decode(collapsed), mean_score


def prefix_beam_search(
    logp: np.ndarray, root: TrieNode, beam_width: int = 50
) -> dict[str, BeamEntry]:
    """Grammar-constrained CTC prefix beam search over `logp` (T, 29),
    starting from grammar-trie node `root`. Returns the final beam:
    prefix -> BeamEntry. Exposed (not private) so tests can verify it
    against a brute-force reference on small synthetic cases."""
    T = logp.shape[0]
    beams: dict[str, BeamEntry] = {"": BeamEntry(node=root, pb=0.0, pnb=NEG_INF)}

    for t in range(T):
        next_beams: dict[str, BeamEntry] = {}

        def _add(prefix: str, node: TrieNode, d_pb: float = NEG_INF, d_pnb: float = NEG_INF) -> None:
            entry = next_beams.get(prefix)
            if entry is None:
                entry = BeamEntry(node=node)
                next_beams[prefix] = entry
            entry.pb = _logsumexp(entry.pb, d_pb)
            entry.pnb = _logsumexp(entry.pnb, d_pnb)

        logp_t = logp[t]
        for prefix, entry in beams.items():
            p_total = entry.total()
            last_char = prefix[-1] if prefix else None

            # blank: prefix unchanged.
            _add(prefix, entry.node, d_pb=p_total + float(logp_t[vcm_alphabet.BLANK_ID]))

            for char_id, ch in vcm_alphabet.ID_TO_CHAR.items():
                if char_id == vcm_alphabet.BLANK_ID:
                    continue
                logp_c = float(logp_t[char_id])
                if ch == last_char:
                    # Repeat of the trailing char: collapsing into the same
                    # prefix draws only from the previous non-blank mass
                    # (already-counted trailing char continuing); a genuine
                    # new occurrence of the same char (needs an intervening
                    # blank in the raw alignment) draws from the previous
                    # blank mass and extends the prefix.
                    _add(prefix, entry.node, d_pnb=entry.pnb + logp_c)
                    child = entry.node.children.get(ch)
                    if child is not None:
                        _add(prefix + ch, child, d_pnb=entry.pb + logp_c)
                else:
                    child = entry.node.children.get(ch)
                    if child is None:
                        continue  # not reachable in this grammar: pruned
                    _add(prefix + ch, child, d_pnb=p_total + logp_c)

        if len(next_beams) > beam_width:
            top = sorted(next_beams.items(), key=lambda kv: kv[1].total(), reverse=True)[:beam_width]
            next_beams = dict(top)
        beams = next_beams

    return beams


def decode_utterance(
    logp: np.ndarray,
    grammar: Grammar,
    threshold: float,
    beam_width: int = 50,
    required_command_margin: float | None = None,
    score_mode: str = "mean_frame",
) -> DecodeResult:
    """logp: (T, 29) log-probabilities/log-posteriors over the 29-token
    alphabet (docs/VCM-CONTRACT.md section 7). `threshold` is a mean
    per-frame log-probability cutoff the caller must supply -- never
    hardcoded here (Task 05 sweeps it).

    `required_command_margin` (docs/INCOMPLETE-GRAMMAR-REJECTION.md, Step 3):
    when None (default) the gate is disabled and acceptance is exactly the
    baseline. When set, the strongest designated incomplete-prefix beam
    (a beam whose exact prefix is in `grammar.incomplete_prefixes`) is
    compared against the best completed terminal on RAW unnormalized beam log
    mass -- not /T, which rewards unrelated trailing blank frames. If
    `incomplete_gap = best_command_raw - best_incomplete_raw` falls below the
    margin, the result is `no_match` with `rejection_reason =
    "incomplete_prefix"`. The existing confidence threshold stays an
    independent second gate. Decision order: (1) no completed terminal ->
    `no_match`, incomplete-prefix reason when applicable; (2) margin gate;
    (3) confidence threshold; (4) accept.

    `score_mode` picks what `confidence` (and therefore the threshold gate)
    measures. `"mean_frame"` (default, the historical behavior) is the winning
    terminal's beam log mass / T. `"per_char"` is that same raw mass divided
    by the length of the winning phrase's text: duration-invariant (trailing
    blank frames cannot dilute it) and on a different scale (roughly -1 for
    a weak accept), so thresholds are NOT interchangeable between modes.
    Winner selection is identical in both modes (T is constant within an
    utterance, so argmax raw mass == argmax mass/T), and `out_of_grammar_gap`
    keeps its mean-frame meaning in both. See
    `.scratch/dense-d2-loose-impl/PLAN.md`."""
    if score_mode not in SCORE_MODES:
        raise ValueError(f"score_mode must be one of {SCORE_MODES}, got {score_mode!r}")
    T = logp.shape[0]
    greedy_text, greedy_score = _greedy_unconstrained(logp)

    beams = prefix_beam_search(logp, grammar.root, beam_width=beam_width)

    best_intent: str | None = None
    best_slots: dict = {}
    best_score = NEG_INF
    best_command_prefix: str | None = None
    best_command_raw: float | None = None
    for prefix, entry in beams.items():
        if entry.node.terminal is None:
            continue
        score = entry.total() / T if T else entry.total()
        if score > best_score:
            intent, slots_ = entry.node.terminal[0]
            best_intent, best_slots, best_score = intent, slots_, score
            best_command_prefix, best_command_raw = prefix, entry.total()

    # `best_score` is the mean-frame score above; `gap` below always uses it
    # (greedy_score is mean-frame too). `accept_score` is what the threshold
    # gate and the reported `confidence` use, per `score_mode`.
    frame_score = best_score
    if score_mode == "per_char" and best_intent is not None and best_command_prefix:
        accept_score = best_command_raw / len(best_command_prefix)
    else:
        accept_score = best_score

    # Strongest designated incomplete-prefix competitor in the final beam, on
    # raw (unnormalized, duration-invariant) beam log mass. Grammars that did
    # not opt in carry the empty set -> no competitor -> gate is a no-op.
    best_incomplete_prefix: str | None = None
    best_incomplete_raw: float | None = None
    for prefix in grammar.incomplete_prefixes:
        entry = beams.get(prefix)
        if entry is None:
            continue
        raw = entry.total()
        if best_incomplete_raw is None or raw > best_incomplete_raw:
            best_incomplete_prefix, best_incomplete_raw = prefix, raw

    incomplete_gap: float | None = None
    if best_command_raw is not None and best_incomplete_raw is not None:
        incomplete_gap = best_command_raw - best_incomplete_raw

    if required_command_margin is None:
        # Baseline: unchanged acceptance rule, no rejection reason.
        no_match = best_intent is None or accept_score < threshold
        rejection_reason: str | None = None
    elif best_intent is None:
        no_match = True
        rejection_reason = (
            "incomplete_prefix" if best_incomplete_prefix is not None else None
        )
    elif incomplete_gap is not None and incomplete_gap < required_command_margin:
        no_match = True
        rejection_reason = "incomplete_prefix"
    else:
        no_match = accept_score < threshold
        rejection_reason = None

    gap = greedy_score - frame_score if best_intent is not None else float("inf")

    return DecodeResult(
        intent=None if no_match else best_intent,
        slots={} if no_match else best_slots,
        text=greedy_text,
        confidence=accept_score,
        no_match=no_match,
        out_of_grammar_gap=gap,
        grammar_text=best_command_prefix if best_command_prefix is not None else "",
        rejection_reason=rejection_reason,
        incomplete_prefix=best_incomplete_prefix,
        incomplete_gap=incomplete_gap,
        command_raw_score=best_command_raw,
        incomplete_raw_score=best_incomplete_raw,
    )


def decode(
    logp: np.ndarray,
    grammar: Grammar,
    threshold: float,
    beam_width: int = 50,
    required_command_margin: float | None = None,
    score_mode: str = "mean_frame",
) -> DecodeResult | list[DecodeResult]:
    """Accepts (T, 29) for a single utterance (-> one DecodeResult) or
    (B, T, 29) for a batch (-> list[DecodeResult]), per
    docs/VCM-CONTRACT.md section 7. `required_command_margin` (see
    `decode_utterance`) and `score_mode` apply to every row of a batch."""
    logp = np.asarray(logp)
    if logp.ndim == 2:
        return decode_utterance(
            logp,
            grammar,
            threshold,
            beam_width=beam_width,
            required_command_margin=required_command_margin,
            score_mode=score_mode,
        )
    if logp.ndim == 3:
        return [
            decode_utterance(
                logp[b],
                grammar,
                threshold,
                beam_width=beam_width,
                required_command_margin=required_command_margin,
                score_mode=score_mode,
            )
            for b in range(logp.shape[0])
        ]
    raise ValueError(f"expected logp of shape (T, 29) or (B, T, 29), got {logp.shape}")
