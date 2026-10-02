"""Invariants for the experimental Option B incomplete-prefix grammar."""

from __future__ import annotations

from me2_voicegen.vcm.optiona.grammar import SPEC_GRAMMAR, TOY_GRAMMAR
from me2_voicegen.vcm.optionb.grammar import OPTIONB_GRAMMAR, OPTIONB_RULES
from me2_voicegen.vcm.optionb.incomplete_prefix_grammar import (
    OPTIONB_INCOMPLETE_PREFIX_GRAMMAR,
    derive_incomplete_prefixes,
)
from test_optionb_grammar import EXPECTED_STRICT_PREFIX_PAIRS


def test_slot_bearing_prefixes_are_rejection_competitors():
    rejection_grammar = OPTIONB_INCOMPLETE_PREFIX_GRAMMAR

    assert rejection_grammar.is_incomplete("change color to")
    assert rejection_grammar.is_incomplete("set color to")
    assert rejection_grammar.is_incomplete("brightness")
    assert rejection_grammar.is_incomplete("set an alarm for")


def test_only_complete_words_are_included():
    incomplete = OPTIONB_INCOMPLETE_PREFIX_GRAMMAR.incomplete_prefixes

    assert "colo" not in incomplete
    assert "lights o" not in incomplete
    assert "set color to " not in incomplete


def test_accepted_commands_are_never_marked_incomplete():
    accepted = {text for text, _, _ in OPTIONB_GRAMMAR.all_phrases()}
    incomplete = OPTIONB_INCOMPLETE_PREFIX_GRAMMAR.incomplete_prefixes

    assert accepted.isdisjoint(incomplete)
    assert "time" not in incomplete
    assert "pause" not in incomplete
    assert OPTIONB_GRAMMAR.accepts("time") is not None
    assert OPTIONB_GRAMMAR.accepts("pause") is not None


def test_rejection_grammar_does_not_accept_color():
    assert OPTIONB_INCOMPLETE_PREFIX_GRAMMAR.command_grammar is OPTIONB_GRAMMAR
    assert OPTIONB_INCOMPLETE_PREFIX_GRAMMAR.command_grammar.accepts("color") is None


def test_every_incomplete_prefix_has_a_longer_accepted_completion():
    rejection_grammar = OPTIONB_INCOMPLETE_PREFIX_GRAMMAR

    for prefix in rejection_grammar.incomplete_prefixes:
        completions = rejection_grammar.completions_for(prefix)
        assert completions
        assert all(command.startswith(f"{prefix} ") for command in completions)


def test_derivation_is_deterministic_and_validated():
    assert derive_incomplete_prefixes(OPTIONB_GRAMMAR) == (
        OPTIONB_INCOMPLETE_PREFIX_GRAMMAR.incomplete_prefixes
    )
    OPTIONB_INCOMPLETE_PREFIX_GRAMMAR.validate()


# ---------------------------------------------------------------------------
# Production-merge invariants (ticket 01): the derived set now also lives on
# OPTIONB_GRAMMAR itself, and accepted strict prefixes -- `time`, `pause`,
# and every other pair found programmatically -- are never rejection
# competitors.
# ---------------------------------------------------------------------------


def test_accepted_strict_prefixes_are_never_rejection_competitors():
    incomplete = OPTIONB_GRAMMAR.incomplete_prefixes
    accepted = {text for text, _, _ in OPTIONB_GRAMMAR.all_phrases()}

    # (a) Word-level: every accepted phrase that is a whole-word prefix of
    # another accepted phrase (today: `pause` x2) is never a competitor.
    for short in accepted:
        short_words = short.split()
        for long in accepted:
            if long.split()[: len(short_words)] == short_words and short != long:
                assert short not in incomplete, (
                    f"accepted strict prefix {short!r} (of {long!r}) marked incomplete"
                )

    # (b) The doc-named strict-prefix commands keep their intent semantics.
    assert "time" not in incomplete
    assert "pause" not in incomplete
    assert OPTIONB_GRAMMAR.accepts("time") is not None
    assert OPTIONB_GRAMMAR.accepts("pause") is not None

    # (c) Character-level: the exact 9-pair strict-prefix set of
    # test_optionb_grammar.py (D6) -- the accepted shorter phrase of each
    # pair is never a rejection competitor.
    for short, _long in EXPECTED_STRICT_PREFIX_PAIRS:
        assert short not in incomplete, f"accepted prefix {short!r} marked incomplete"


def test_optionb_grammar_carries_derived_incomplete_prefixes():
    incomplete = OPTIONB_GRAMMAR.incomplete_prefixes

    assert incomplete == derive_incomplete_prefixes(OPTIONB_GRAMMAR)
    assert "change color to" in incomplete
    assert "set color to" in incomplete
    # Slot-value prefixes are whole-word prefixes of accepted phrases, so the
    # spec's derivation includes them (e.g. of "alarm 6 am", "adjust
    # brightness to 100 percent").
    assert "alarm 6" in incomplete
    assert "adjust brightness to 100" in incomplete
    assert "colo" not in incomplete
    assert "time" not in incomplete
    assert "pause" not in incomplete
    assert "color red" not in incomplete
    assert len(incomplete) == 168

    # Cross-invariant: the experimental view and the production field agree.
    assert (
        OPTIONB_INCOMPLETE_PREFIX_GRAMMAR.incomplete_prefixes
        == OPTIONB_GRAMMAR.incomplete_prefixes
    )

    # Grammars that did not opt in keep the empty default.
    assert SPEC_GRAMMAR.incomplete_prefixes == frozenset()
    assert TOY_GRAMMAR.incomplete_prefixes == frozenset()


def test_optionb_grammar_phrase_surface_unchanged():
    from me2_voicegen.common.grammar_core import compile_grammar

    fresh = compile_grammar("OPTIONB_GRAMMAR", OPTIONB_RULES)

    assert len(list(OPTIONB_GRAMMAR.all_phrases())) == 129
    assert list(OPTIONB_GRAMMAR.all_phrases()) == list(fresh.all_phrases())
    assert set(OPTIONB_GRAMMAR.incomplete_prefixes).isdisjoint(
        {text for text, _, _ in OPTIONB_GRAMMAR.all_phrases()}
    )
    assert OPTIONB_GRAMMAR.accepts("color") is None
    assert OPTIONB_GRAMMAR.accepts("time") is not None
    assert OPTIONB_GRAMMAR.accepts("pause") is not None
    # The rules dict is the same object in both (compile_grammar takes it by
    # reference) -- the attachment changed nothing about the grammar's data.
    assert OPTIONB_GRAMMAR.rules is fresh.rules
