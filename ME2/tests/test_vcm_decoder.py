"""Synthetic-posterior tests for `vcm.decoder`'s grammar-constrained CTC
prefix beam search. No audio, no model -- posteriors are built directly
from known text strings, per ticket 03's acceptance criteria (this also
satisfies the feature's acceptance criterion 4: slot extraction verified
deterministically without slot-bearing audio)."""

from __future__ import annotations

import itertools

import numpy as np
import pytest

from me2_voicegen.vcm import alphabet as vcm_alphabet
from me2_voicegen.vcm import decoder as dec
from me2_voicegen.vcm.optiona import grammar as g
from me2_voicegen.vcm.optiona.phrases import INTENT_PHRASES

PEAK = 20.0


def make_posterior(text: str, peak: float = PEAK) -> np.ndarray:
    """Near-one-hot (T, 29) log-softmax posterior that CTC-collapses back to
    exactly `text`. Inserts a blank frame between two identical consecutive
    characters so the greedy collapse doesn't merge them into one."""
    frame_ids: list[int] = []
    prev: int | None = None
    for ch in text:
        cid = vcm_alphabet.CHAR_TO_ID[ch]
        if cid == prev:
            frame_ids.append(vcm_alphabet.BLANK_ID)
            prev = None
        frame_ids.append(cid)
        prev = cid

    T = len(frame_ids)
    alphabet_size = vcm_alphabet.ALPHABET_SIZE
    logits = np.full((T, alphabet_size), -peak, dtype=np.float64)
    for t, cid in enumerate(frame_ids):
        logits[t, cid] = peak
    m = logits.max(axis=-1, keepdims=True)
    logp = logits - (m + np.log(np.exp(logits - m).sum(axis=-1, keepdims=True)))
    return logp


def test_make_posterior_greedy_round_trips():
    for text in ("play music", "call mom", "billie eilish", "weeknd"):
        logp = make_posterior(text)
        decoded_text, score = dec._greedy_unconstrained(logp)
        assert decoded_text == text
        assert score > -1.0  # near-one-hot: mean per-frame log-prob near 0


# ---------------------------------------------------------------------------
# Every grammar phrase decodes to the correct intent + slots, for both
# grammars.
# ---------------------------------------------------------------------------

SPEC_PHRASES = list(g.SPEC_GRAMMAR.all_phrases())
TOY_PHRASES = list(g.TOY_GRAMMAR.all_phrases())

LOW_THRESHOLD = -1.0  # near-one-hot posteriors score close to 0/frame


@pytest.mark.parametrize("text,intent,slots", SPEC_PHRASES)
def test_spec_grammar_decodes_every_phrase(text, intent, slots):
    logp = make_posterior(text)
    result = dec.decode_utterance(logp, g.SPEC_GRAMMAR, threshold=LOW_THRESHOLD)
    assert not result.no_match, f"unexpected no_match for {text!r}"
    assert result.intent == intent
    assert result.slots == slots


@pytest.mark.parametrize("text,intent,slots", TOY_PHRASES)
def test_toy_grammar_decodes_every_phrase(text, intent, slots):
    logp = make_posterior(text)
    result = dec.decode_utterance(logp, g.TOY_GRAMMAR, threshold=LOW_THRESHOLD)
    assert not result.no_match, f"unexpected no_match for {text!r}"
    assert result.intent == intent
    assert result.slots == slots


def test_toy_grammar_decodes_all_20_intent_phrases():
    for intent, phrase in INTENT_PHRASES.items():
        logp = make_posterior(phrase)
        result = dec.decode_utterance(logp, g.TOY_GRAMMAR, threshold=LOW_THRESHOLD)
        assert not result.no_match, f"TOY_GRAMMAR failed to decode {intent}: {phrase!r}"
        assert result.intent == intent
        assert result.text == phrase


# ---------------------------------------------------------------------------
# Out-of-grammar battery -> no_match.
# ---------------------------------------------------------------------------

OUT_OF_GRAMMAR_TEXTS = [
    "gibberish text not in grammar",
    "the quick brown fox",
    "call grandma",
    "set timer",  # not in SPEC_GRAMMAR (missing $NUMBER $TIME_UNIT); IS a TOY_ALIASES phrase
    "lights",
    "play the music",
]

# "set timer" is a valid bare TOY_ALIASES phrase, so it's excluded from the
# TOY_GRAMMAR battery (still covered as a SPEC_GRAMMAR no_match case above).
TOY_OUT_OF_GRAMMAR_TEXTS = [t for t in OUT_OF_GRAMMAR_TEXTS if t != "set timer"]


@pytest.mark.parametrize("text", OUT_OF_GRAMMAR_TEXTS)
def test_spec_grammar_no_match_for_out_of_grammar_text(text):
    logp = make_posterior(text)
    result = dec.decode_utterance(logp, g.SPEC_GRAMMAR, threshold=LOW_THRESHOLD)
    assert result.no_match
    assert result.intent is None
    assert result.slots == {}


@pytest.mark.parametrize("text", TOY_OUT_OF_GRAMMAR_TEXTS)
def test_toy_grammar_no_match_for_out_of_grammar_text(text):
    logp = make_posterior(text)
    result = dec.decode_utterance(logp, g.TOY_GRAMMAR, threshold=LOW_THRESHOLD)
    assert result.no_match
    assert result.intent is None


def test_threshold_is_a_caller_parameter_not_hardcoded():
    logp = make_posterior("play music")
    low = dec.decode_utterance(logp, g.SPEC_GRAMMAR, threshold=-1.0)
    high = dec.decode_utterance(logp, g.SPEC_GRAMMAR, threshold=100.0)
    assert not low.no_match
    assert high.no_match  # same posterior, only the threshold changed


def test_out_of_grammar_gap_is_large_for_rejected_text_and_small_for_accepted():
    accepted = dec.decode_utterance(make_posterior("play music"), g.SPEC_GRAMMAR, threshold=LOW_THRESHOLD)

    # A truly impossible posterior: every id but 'z' has literal -inf
    # log-prob at every frame (not just "very low", per `make_posterior`'s
    # near-one-hot softmax, which still leaves every other char a finite --
    # if tiny -- probability a wide enough beam can occasionally exploit to
    # spell out some other short valid phrase at very low confidence).
    # 'z' starts no $CMD_* rule, so every grammar-constrained extension is
    # pruned at frame 0: no terminal beam is reachable at all.
    T = 10
    logp = np.full((T, vcm_alphabet.ALPHABET_SIZE), dec.NEG_INF)
    logp[:, vcm_alphabet.CHAR_TO_ID["z"]] = 0.0
    rejected = dec.decode_utterance(logp, g.SPEC_GRAMMAR, threshold=LOW_THRESHOLD)

    assert accepted.out_of_grammar_gap < 1.0
    assert rejected.out_of_grammar_gap == float("inf")
    assert rejected.no_match
    assert rejected.intent is None


def test_batch_decode_returns_list_of_results():
    a, b = make_posterior("play music"), make_posterior("stop")
    # (B, T, 29) requires a common T across the batch, same as any padded
    # CTC batch (docs/VCM-CONTRACT.md section 6) -- pad the shorter one
    # with extra high-confidence blank frames, which don't change its
    # greedy/grammar decode.
    T = max(a.shape[0], b.shape[0])

    def pad(x: np.ndarray) -> np.ndarray:
        if x.shape[0] == T:
            return x
        pad_frame = np.full((T - x.shape[0], x.shape[1]), -PEAK)
        pad_frame[:, vcm_alphabet.BLANK_ID] = PEAK
        m = pad_frame.max(axis=-1, keepdims=True)
        pad_frame = pad_frame - (m + np.log(np.exp(pad_frame - m).sum(axis=-1, keepdims=True)))
        return np.concatenate([x, pad_frame], axis=0)

    batch = np.stack([pad(a), pad(b)])
    results = dec.decode(batch, g.SPEC_GRAMMAR, threshold=LOW_THRESHOLD)
    assert isinstance(results, list)
    assert len(results) == 2
    assert results[0].intent == "PLAY_MUSIC"
    assert results[1].intent == "STOP"


def test_single_utterance_decode_returns_single_result():
    result = dec.decode(make_posterior("stop"), g.SPEC_GRAMMAR, threshold=LOW_THRESHOLD)
    assert result.intent == "STOP"


# ---------------------------------------------------------------------------
# Beam search vs exhaustive brute force on a small synthetic grammar.
# ---------------------------------------------------------------------------


def _tiny_grammar() -> g.Grammar:
    rules = {
        "$TINY": [
            (("h", "i"), {}, "A"),
            (("h", "e", "y"), {}, "B"),
        ]
    }
    return g.compile_grammar("TINY", rules)


def _brute_force_best(logp: np.ndarray, grammar: g.Grammar, active_ids: list[int]) -> tuple[str, float]:
    """Exhaustively enumerate every raw label sequence over `active_ids`
    (small support set), collapse each via CTC rules, and return the
    grammar-accepted collapsed text with the greatest total (summed, not
    per-frame-mean) log-probability mass across all alignments collapsing
    to it -- the same quantity `prefix_beam_search`'s BeamEntry.total()
    represents for a completed prefix."""
    T = logp.shape[0]
    best_text = None
    best_score = dec.NEG_INF
    totals: dict[str, float] = {}
    for seq in itertools.product(active_ids, repeat=T):
        score = sum(float(logp[t, cid]) for t, cid in enumerate(seq))
        collapsed = vcm_alphabet.collapse(list(seq))
        text = vcm_alphabet.decode(collapsed)
        if grammar.accepts(text) is None:
            continue
        totals[text] = dec._logsumexp(totals.get(text, dec.NEG_INF), score)
    for text, score in totals.items():
        if score > best_score:
            best_text, best_score = text, score
    return best_text, best_score


def test_beam_search_matches_brute_force_on_tiny_grammar():
    grammar = _tiny_grammar()
    active_chars = ["h", "i", "e", "y", " "]  # space: both tiny phrases are 2-3 space-joined words
    active_ids = [vcm_alphabet.BLANK_ID] + [vcm_alphabet.CHAR_TO_ID[c] for c in active_chars]

    rng = np.random.default_rng(0)
    logits = rng.normal(scale=2.0, size=(5, vcm_alphabet.ALPHABET_SIZE))
    # zero out probability mass on every id outside the tiny active set so
    # brute force only needs to enumerate `len(active_ids) ** T` sequences.
    mask = np.full(vcm_alphabet.ALPHABET_SIZE, -1e6)
    mask[active_ids] = 0.0
    logits = logits + mask
    m = logits.max(axis=-1, keepdims=True)
    logp = logits - (m + np.log(np.exp(logits - m).sum(axis=-1, keepdims=True)))

    brute_text, brute_score = _brute_force_best(logp, grammar, active_ids)
    assert brute_text is not None

    beams = dec.prefix_beam_search(logp, grammar.root, beam_width=1000)
    terminal_beams = {p: e for p, e in beams.items() if e.node.terminal is not None}
    assert terminal_beams, "beam search found no accepted path at all"
    beam_text, beam_entry = max(terminal_beams.items(), key=lambda kv: kv[1].total())

    assert beam_text == brute_text
    assert beam_entry.total() == pytest.approx(brute_score, abs=1e-6)
