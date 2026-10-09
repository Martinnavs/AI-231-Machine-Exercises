"""SPEC_GRAMMAR (verbatim BNF, see docstring below) and TOY_GRAMMAR (the
same grammar plus toy-only bare-phrase aliases), compiled into a shared
character-level trie for `vcm.decoder`'s CTC prefix beam search.

Design (see ticket .scratch/vcm-toy/tickets/03-grammar-decoder.md, decision A):

- Each `$CMD_*` production is built with small combinators (`literal`,
  `seq`, `alt`, `opt`, `slot`) that expand every optional/alternation
  combination into a flat list of (word-sequence, slots) alternatives --
  small enough here (~1.6k total phrases across all 11 rules) to fully
  enumerate at import time rather than build a generalized NFA.
- `with_intent` stamps the final decoder-output intent (one of the 20
  `vcm.optiona.phrases.INTENT_PHRASES` keys) onto a finished alternative
  list. Intent
  is attached per *alternative*, not per rule, because some BNF rules
  (`$CMD_LIGHTS_CTRL`, `$CMD_LIGHTS_DIM`, `$CMD_MEDIA_CTRL`) cover more than
  one INTENT_PHRASES key depending on which literal branch matched (e.g.
  "lights on" vs "lights off").
- KNOWN GRAMMAR LIMITATION: `$CMD_TEMPERATURE`'s BNF text ("(set | adjust)
  (the)? temperature to $NUMBER (degrees)?") has no directional word, so a
  SPEC_GRAMMAR-parsed temperature phrase cannot be resolved to TEMP_UP vs
  TEMP_DOWN -- the BNF itself is ambiguous here, not an implementation
  gap. Those alternatives are stamped with the rule name itself
  ("$CMD_TEMPERATURE") as a placeholder intent. Only the TOY_ALIASES bare
  words "cooler"/"warmer" are unambiguous and get tagged TEMP_DOWN/TEMP_UP
  directly.
- All character-trie compilation (`compile_grammar`) inserts each
  alternative's full text (words joined by single spaces) char-by-char
  into a shared `TrieNode` trie, merging common prefixes naturally; each
  terminal node stores the (intent, slots) pair(s) reachable by ending a
  decode exactly there. `vcm.decoder`'s CTC prefix beam search walks this
  same trie so only characters reachable from the current grammar state
  are ever expanded -- per decision (C), no `kaldifst` (present in the
  venv only transitively via `wetext`, not declared) or `sherpa-onnx`
  (not installed); `kaldifst`/a real WFST library is a plausible future
  scale-up path if the grammar grows far past this size, not used here.

SPEC_GRAMMAR and TOY_GRAMMAR are compiled independently (each gets its own
fresh trie) and must never be silently merged into one object -- Task 05
needs both, separately, to compare SPEC-only vs SPEC+toy-alias decode
behavior.
"""

from __future__ import annotations

from me2_voicegen.common.grammar_core import (  # re-export
    EMPTY,
    Grammar,
    TrieNode,
    _insert,
    _vocab,
    alt,
    compile_grammar,
    literal,
    opt,
    seq,
    slot,
    with_intent,
)

# ---------------------------------------------------------------------------
# Slot vocabularies (BNF verbatim).
# ---------------------------------------------------------------------------

NUMBER_VOCAB: list[tuple[tuple[str, ...], str]] = _vocab(
    "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten",
    "fifteen", "twenty", "twenty five", "thirty", "thirty five",
    "forty", "forty five", "fifty", "fifty five", "sixty",
    "seventy", "eighty", "ninety", "one hundred",
)
PERSONA_VOCAB: list[tuple[tuple[str, ...], str]] = _vocab("mom", "dad")
ARTIST_VOCAB: list[tuple[tuple[str, ...], str]] = _vocab(
    "taylor swift", "the weeknd", "bad bunny", "drake", "billie eilish"
)
TIME_UNIT_VOCAB: list[tuple[tuple[str, ...], str]] = _vocab(
    "seconds", "second", "minutes", "minute", "hours", "hour"
)
PERCENT_VOCAB: list[tuple[tuple[str, ...], str]] = _vocab("percent")
# Not a named BNF vocabulary ($AM_PM isn't one of the 5 listed slot
# vocabularies) but $CMD_SET_ALARM needs it and it behaves identically to a
# slot (a value-bearing alternation), so it's extracted the same way. Named
# "AMPM" to match the existing convention in `vcm.slot_eval_set`.
AMPM_VOCAB: list[tuple[tuple[str, ...], str]] = _vocab("am", "pm")


# ---------------------------------------------------------------------------
# $CMD_* rules (BNF verbatim -- see the ticket for the exact text each of
# these encodes). SPEC_RULES groups alternatives by rule name (for
# rule-by-rule test coverage); trie compilation flattens across all rules.
# ---------------------------------------------------------------------------

CMD_PLAY_MUSIC = with_intent(
    "PLAY_MUSIC",
    seq(
        alt(literal("play"), literal("start")),
        literal("music"),
        opt(seq(alt(literal("by"), literal("from")), slot("ARTIST", ARTIST_VOCAB))),
    ),
)

CMD_WEATHER = with_intent(
    "WEATHER",
    seq(
        alt(literal("what", "is"), literal("whats"), literal("check")),
        opt(literal("the")),
        literal("weather"),
    ),
)

CMD_TIME = with_intent(
    "TIME",
    seq(
        alt(literal("what", "is"), literal("whats")),
        opt(literal("the")),
        literal("time"),
        opt(literal("is", "it")),
    ),
)

CMD_LIGHTS_CTRL = alt(
    with_intent(
        "LIGHT_ON",
        alt(
            seq(alt(literal("turn"), literal("switch")), opt(literal("the")), literal("lights"), literal("on")),
            seq(literal("lights"), literal("on")),
        ),
    ),
    with_intent(
        "LIGHT_OFF",
        alt(
            seq(alt(literal("turn"), literal("switch")), opt(literal("the")), literal("lights"), literal("off")),
            seq(literal("lights"), literal("off")),
        ),
    ),
)

_LIGHTS_DIM_TAIL = opt(seq(literal("to"), slot("NUMBER", NUMBER_VOCAB), opt(slot("PERCENT", PERCENT_VOCAB))))

CMD_LIGHTS_DIM = alt(
    with_intent("DIM_DOWN", seq(literal("dim"), opt(literal("the")), literal("lights"), _LIGHTS_DIM_TAIL)),
    with_intent("DIM_UP", seq(literal("brighten"), opt(literal("the")), literal("lights"), _LIGHTS_DIM_TAIL)),
)

CMD_SET_TIMER = with_intent(
    "TIMER",
    seq(
        alt(literal("set"), literal("start")),
        opt(literal("a")),
        literal("timer"),
        opt(literal("for")),
        slot("NUMBER", NUMBER_VOCAB),
        slot("TIME_UNIT", TIME_UNIT_VOCAB),
    ),
)

CMD_SET_ALARM = with_intent(
    "ALARM",
    seq(
        alt(literal("set"), literal("start")),
        opt(literal("an")),
        literal("alarm"),
        opt(literal("for")),
        slot("NUMBER", NUMBER_VOCAB),
        slot("AMPM", AMPM_VOCAB),
    ),
)

# See module docstring "KNOWN GRAMMAR LIMITATION": the BNF has no
# directional word, so this can't resolve to TEMP_UP/TEMP_DOWN on its own.
CMD_TEMPERATURE = with_intent(
    "$CMD_TEMPERATURE",
    seq(
        alt(literal("set"), literal("adjust")),
        opt(literal("the")),
        literal("temperature"),
        literal("to"),
        slot("NUMBER", NUMBER_VOCAB),
        opt(literal("degrees")),
    ),
)

CMD_MEDIA_CTRL = alt(
    with_intent("PAUSE", seq(literal("pause"), opt(literal("music")))),
    with_intent("STOP", seq(literal("stop"), opt(literal("music")))),
    with_intent("NEXT", seq(literal("next"), opt(alt(literal("track"), literal("song"))))),
    with_intent("VOLUME_UP", seq(literal("volume"), literal("up"))),
    with_intent("VOLUME_DOWN", seq(literal("volume"), literal("down"))),
)

CMD_REMINDERS = with_intent(
    "LIST_REMINDERS",
    seq(alt(literal("give", "me"), literal("what", "are")), opt(literal("my")), literal("reminders")),
)

CMD_CALL = with_intent("CALL", seq(alt(literal("call"), literal("phone")), slot("PERSONA", PERSONA_VOCAB)))


SPEC_RULES: dict[str, list[tuple[tuple[str, ...], dict, str]]] = {
    "$CMD_PLAY_MUSIC": CMD_PLAY_MUSIC,
    "$CMD_WEATHER": CMD_WEATHER,
    "$CMD_TIME": CMD_TIME,
    "$CMD_LIGHTS_CTRL": CMD_LIGHTS_CTRL,
    "$CMD_LIGHTS_DIM": CMD_LIGHTS_DIM,
    "$CMD_SET_TIMER": CMD_SET_TIMER,
    "$CMD_SET_ALARM": CMD_SET_ALARM,
    "$CMD_TEMPERATURE": CMD_TEMPERATURE,
    "$CMD_MEDIA_CTRL": CMD_MEDIA_CTRL,
    "$CMD_REMINDERS": CMD_REMINDERS,
    "$CMD_CALL": CMD_CALL,
}

# ---------------------------------------------------------------------------
# TOY_ALIASES: the 12 non-BNF-parsing INTENT_PHRASES values (per ticket
# decision A / docs/VCM-CONTRACT.md section 3), each a bare alias phrase
# with no slot. "message" and "set reminder" are toy-only: MESSAGE and
# SET_REMINDER are real INTENT_PHRASES keys but have NO SPEC_GRAMMAR rule
# at all (not part of the BNF spec, ticket-confirmed toy-only additions).
# ---------------------------------------------------------------------------

TOY_ALIASES: list[tuple[tuple[str, ...], dict, str]] = [
    (("set", "alarm"), {}, "ALARM"),
    (("set", "timer"), {}, "TIMER"),
    (("call",), {}, "CALL"),
    (("time",), {}, "TIME"),
    (("weather",), {}, "WEATHER"),
    (("list", "reminders"), {}, "LIST_REMINDERS"),
    (("dimmer",), {}, "DIM_DOWN"),
    (("brighter",), {}, "DIM_UP"),
    (("cooler",), {}, "TEMP_DOWN"),
    (("warmer",), {}, "TEMP_UP"),
    (("message",), {}, "MESSAGE"),  # toy-only, no BNF counterpart
    (("set", "reminder"), {}, "SET_REMINDER"),  # toy-only, no BNF counterpart
]


SPEC_GRAMMAR: Grammar = compile_grammar("SPEC_GRAMMAR", SPEC_RULES)

_TOY_RULES: dict[str, list[tuple[tuple[str, ...], dict, str]]] = dict(SPEC_RULES)
_TOY_RULES["$TOY_ALIASES"] = TOY_ALIASES
TOY_GRAMMAR: Grammar = compile_grammar("TOY_GRAMMAR", _TOY_RULES)
