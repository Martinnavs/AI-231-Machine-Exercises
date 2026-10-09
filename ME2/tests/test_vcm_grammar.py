"""Table-driven accept/reject coverage for `vcm.grammar`'s SPEC_GRAMMAR and
TOY_GRAMMAR: every $CMD_* rule, every optional-token presence/absence, and
every value in every slot vocabulary (docs/VCM-CONTRACT.md section 7 /
ticket 03's acceptance criteria)."""

from __future__ import annotations

import pytest

from me2_voicegen.vcm.optiona import grammar as g
from me2_voicegen.vcm.optiona.phrases import INTENT_PHRASES

# ---------------------------------------------------------------------------
# 1. Table-driven accept tests: every $CMD_* rule, every optional-token
#    combination, every slot vocabulary value.
# ---------------------------------------------------------------------------

# (text, expected_intent, expected_slots)
ACCEPT_CASES: list[tuple[str, str, dict]] = [
    # $CMD_PLAY_MUSIC
    ("play music", "PLAY_MUSIC", {}),
    ("start music", "PLAY_MUSIC", {}),
    ("play music by taylor swift", "PLAY_MUSIC", {"ARTIST": "taylor swift"}),
    ("play music from the weeknd", "PLAY_MUSIC", {"ARTIST": "the weeknd"}),
    ("play music by bad bunny", "PLAY_MUSIC", {"ARTIST": "bad bunny"}),
    ("play music by drake", "PLAY_MUSIC", {"ARTIST": "drake"}),
    ("play music by billie eilish", "PLAY_MUSIC", {"ARTIST": "billie eilish"}),
    # $CMD_WEATHER
    ("what is weather", "WEATHER", {}),
    ("what is the weather", "WEATHER", {}),
    ("whats weather", "WEATHER", {}),
    ("whats the weather", "WEATHER", {}),
    ("check weather", "WEATHER", {}),
    ("check the weather", "WEATHER", {}),
    # $CMD_TIME
    ("what is time", "TIME", {}),
    ("what is the time", "TIME", {}),
    ("what is time is it", "TIME", {}),
    ("what is the time is it", "TIME", {}),
    ("whats time", "TIME", {}),
    ("whats the time is it", "TIME", {}),
    # $CMD_LIGHTS_CTRL
    ("turn lights on", "LIGHT_ON", {}),
    ("turn the lights on", "LIGHT_ON", {}),
    ("switch lights on", "LIGHT_ON", {}),
    ("switch the lights off", "LIGHT_OFF", {}),
    ("lights on", "LIGHT_ON", {}),
    ("lights off", "LIGHT_OFF", {}),
    # $CMD_LIGHTS_DIM
    ("dim lights", "DIM_DOWN", {}),
    ("dim the lights", "DIM_DOWN", {}),
    ("brighten lights", "DIM_UP", {}),
    ("brighten the lights", "DIM_UP", {}),
    ("dim the lights to twenty", "DIM_DOWN", {"NUMBER": "twenty"}),
    ("dim the lights to twenty percent", "DIM_DOWN", {"NUMBER": "twenty", "PERCENT": "percent"}),
    ("brighten the lights to fifty percent", "DIM_UP", {"NUMBER": "fifty", "PERCENT": "percent"}),
    # $CMD_SET_TIMER
    ("set timer for five minutes", "TIMER", {"NUMBER": "five", "TIME_UNIT": "minutes"}),
    ("set a timer for five minutes", "TIMER", {"NUMBER": "five", "TIME_UNIT": "minutes"}),
    ("start timer five minutes", "TIMER", {"NUMBER": "five", "TIME_UNIT": "minutes"}),
    ("start a timer for one hour", "TIMER", {"NUMBER": "one", "TIME_UNIT": "hour"}),
    # $CMD_SET_ALARM
    ("set alarm for five am", "ALARM", {"NUMBER": "five", "AMPM": "am"}),
    ("set an alarm for five am", "ALARM", {"NUMBER": "five", "AMPM": "am"}),
    ("start alarm seven pm", "ALARM", {"NUMBER": "seven", "AMPM": "pm"}),
    ("start an alarm for seven pm", "ALARM", {"NUMBER": "seven", "AMPM": "pm"}),
    # $CMD_TEMPERATURE (see grammar.py docstring: BNF has no direction word)
    ("set temperature to seventy", "$CMD_TEMPERATURE", {"NUMBER": "seventy"}),
    ("set the temperature to seventy degrees", "$CMD_TEMPERATURE", {"NUMBER": "seventy"}),
    ("adjust temperature to seventy", "$CMD_TEMPERATURE", {"NUMBER": "seventy"}),
    ("adjust the temperature to seventy degrees", "$CMD_TEMPERATURE", {"NUMBER": "seventy"}),
    # $CMD_MEDIA_CTRL
    ("pause", "PAUSE", {}),
    ("pause music", "PAUSE", {}),
    ("stop", "STOP", {}),
    ("stop music", "STOP", {}),
    ("next", "NEXT", {}),
    ("next track", "NEXT", {}),
    ("next song", "NEXT", {}),
    ("volume up", "VOLUME_UP", {}),
    ("volume down", "VOLUME_DOWN", {}),
    # $CMD_REMINDERS
    ("give me reminders", "LIST_REMINDERS", {}),
    ("give me my reminders", "LIST_REMINDERS", {}),
    ("what are reminders", "LIST_REMINDERS", {}),
    ("what are my reminders", "LIST_REMINDERS", {}),
    # $CMD_CALL
    ("call mom", "CALL", {"PERSONA": "mom"}),
    ("call dad", "CALL", {"PERSONA": "dad"}),
    ("phone mom", "CALL", {"PERSONA": "mom"}),
    ("phone dad", "CALL", {"PERSONA": "dad"}),
]

# Every $NUMBER value, in a minimal carrier phrase ($CMD_SET_TIMER).
NUMBER_CASES = [
    (f"set timer for {value} minutes", "TIMER", {"NUMBER": value, "TIME_UNIT": "minutes"})
    for _, value in g.NUMBER_VOCAB
]

# Every $TIME_UNIT value.
TIME_UNIT_CASES = [
    (f"set timer for five {value}", "TIMER", {"NUMBER": "five", "TIME_UNIT": value})
    for _, value in g.TIME_UNIT_VOCAB
]


@pytest.mark.parametrize("text,expected_intent,expected_slots", ACCEPT_CASES)
def test_spec_grammar_accepts(text, expected_intent, expected_slots):
    result = g.SPEC_GRAMMAR.accepts(text)
    assert result is not None, f"SPEC_GRAMMAR rejected {text!r}"
    intent, slots = result[0]
    assert intent == expected_intent
    assert slots == expected_slots


@pytest.mark.parametrize("text,expected_intent,expected_slots", NUMBER_CASES)
def test_spec_grammar_every_number_value(text, expected_intent, expected_slots):
    result = g.SPEC_GRAMMAR.accepts(text)
    assert result is not None, f"SPEC_GRAMMAR rejected {text!r}"
    intent, slots = result[0]
    assert intent == expected_intent
    assert slots == expected_slots


@pytest.mark.parametrize("text,expected_intent,expected_slots", TIME_UNIT_CASES)
def test_spec_grammar_every_time_unit_value(text, expected_intent, expected_slots):
    result = g.SPEC_GRAMMAR.accepts(text)
    assert result is not None, f"SPEC_GRAMMAR rejected {text!r}"
    intent, slots = result[0]
    assert intent == expected_intent
    assert slots == expected_slots


def test_number_vocab_has_24_values():
    assert len(g.NUMBER_VOCAB) == 24


def test_every_rule_alternative_round_trips_through_accepts():
    """Every alternative SPEC_GRAMMAR itself generated is, tautologically,
    accepted by SPEC_GRAMMAR -- a regression guard against a compiler bug
    silently dropping alternatives during trie insertion."""
    for text, intent, slots in g.SPEC_GRAMMAR.all_phrases():
        result = g.SPEC_GRAMMAR.accepts(text)
        assert result is not None, f"SPEC_GRAMMAR compiled but doesn't accept its own phrase {text!r}"
        assert (intent, slots) in result


# ---------------------------------------------------------------------------
# 2. Reject tests: malformed / out-of-grammar variants.
# ---------------------------------------------------------------------------

REJECT_CASES = [
    "play the music",  # extra "the" not in grammar
    "musics play",  # wrong order
    "lights",  # missing on/off
    "turn lights",  # missing on/off
    "dim lights to",  # dangling "to" with no $NUMBER
    "dim lights to twenty one",  # "twenty one" not a $NUMBER value
    "set timer",  # missing $NUMBER $TIME_UNIT
    "set timer for five",  # missing $TIME_UNIT
    "set alarm for five",  # missing am/pm
    "set alarm for five oclock",  # not am/pm
    "set temperature to seventy fahrenheit",  # extra trailing word
    "call",  # missing $PERSONA
    "call grandma",  # not a $PERSONA value
    "reminders",  # missing verb phrase
    "volume",  # missing up/down
    "next tracks",  # not "track"/"song"
    "",  # empty string
    "gibberish text not in grammar",
]


@pytest.mark.parametrize("text", REJECT_CASES)
def test_spec_grammar_rejects(text):
    assert g.SPEC_GRAMMAR.accepts(text) is None


# "set timer" and "call" are valid TOY_ALIASES phrases (bare, no slot), so
# they're excluded here -- SPEC_GRAMMAR still rejects them (covered above).
TOY_REJECT_CASES = [t for t in REJECT_CASES if t not in ("set timer", "call")]


@pytest.mark.parametrize("text", TOY_REJECT_CASES)
def test_toy_grammar_rejects_same_garbage(text):
    assert g.TOY_GRAMMAR.accepts(text) is None


# ---------------------------------------------------------------------------
# 3. SPEC_GRAMMAR accepts exactly the 8 established INTENT_PHRASES values;
#    TOY_GRAMMAR accepts all 20.
# ---------------------------------------------------------------------------

SPEC_PARSABLE_INTENTS = {
    "PLAY_MUSIC", "LIGHT_ON", "LIGHT_OFF", "PAUSE", "STOP", "NEXT", "VOLUME_UP", "VOLUME_DOWN",
}


def test_spec_grammar_accepts_exactly_the_8_established_phrases():
    accepted = {
        intent for intent, phrase in INTENT_PHRASES.items() if g.SPEC_GRAMMAR.accepts(phrase) is not None
    }
    assert accepted == SPEC_PARSABLE_INTENTS
    assert len(accepted) == 8


def test_spec_grammar_rejects_the_other_12_phrases():
    rejected = {
        intent for intent, phrase in INTENT_PHRASES.items() if g.SPEC_GRAMMAR.accepts(phrase) is None
    }
    assert rejected == set(INTENT_PHRASES) - SPEC_PARSABLE_INTENTS
    assert len(rejected) == 12


def test_toy_grammar_accepts_all_20_intent_phrases():
    for intent, phrase in INTENT_PHRASES.items():
        result = g.TOY_GRAMMAR.accepts(phrase)
        assert result is not None, f"TOY_GRAMMAR rejected INTENT_PHRASES[{intent!r}] = {phrase!r}"


def test_spec_grammar_and_toy_grammar_are_never_merged():
    # TOY_GRAMMAR is a strict superset of what SPEC_GRAMMAR accepts, but
    # they must remain distinct objects with independently owned tries.
    assert g.SPEC_GRAMMAR is not g.TOY_GRAMMAR
    assert g.SPEC_GRAMMAR.root is not g.TOY_GRAMMAR.root
    assert g.SPEC_GRAMMAR.accepts("dimmer") is None
    assert g.TOY_GRAMMAR.accepts("dimmer") is not None


def test_toy_aliases_cover_exactly_the_12_non_bnf_phrases():
    aliased_intents = {intent for _, _, intent in g.TOY_ALIASES}
    expected = set(INTENT_PHRASES) - SPEC_PARSABLE_INTENTS
    assert aliased_intents == expected
    assert len(g.TOY_ALIASES) == 12


def test_message_and_set_reminder_have_no_spec_grammar_rule():
    for intent in ("MESSAGE", "SET_REMINDER"):
        phrase = INTENT_PHRASES[intent]
        assert g.SPEC_GRAMMAR.accepts(phrase) is None
        assert g.TOY_GRAMMAR.accepts(phrase) is not None
