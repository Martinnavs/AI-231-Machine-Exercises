"""Intent vocabulary + display labels for the UI site.

The 19-intent / 6-slot vocabulary is **derived from the production Option B
grammar** (`me2_voicegen.vcm.optionb.OPTIONB_GRAMMAR`) rather than re-listed
here, so the UI and the model can never drift: the grammar is the single source
of truth (feature `ui-site`, decision Q1-a). That import is pure Python
(`grammar_core`/`numbers`/`text` only -- no torch/onnx), verified to keep the UI
and the fast test suite dependency-light.

The model side has already grammar-decoded each command into `(intent, slots)`,
so the UI never runs the grammar over raw text -- it only needs the *set* of
valid intents (to reject unknown ones) and the slot names (to read arguments).

"Output" convention (feature `ui-site`, decision Q5): the indicator's
detected-word line and any UI logging show the **canonical intent label**
(one label covering all three phrasings) **plus the slot value if present** --
never the raw spoken variant.
"""

from __future__ import annotations

from me2_voicegen.vcm.optionb import OPTIONB_GRAMMAR

# The 19 valid intents, straight from the grammar (single source of truth).
INTENTS: frozenset[str] = frozenset(
    {intent for _, intent, _ in OPTIONB_GRAMMAR.all_phrases()}
)

# The 6 slot names used by the slotted intents.
SLOT_NAMES: frozenset[str] = frozenset(
    {name for _, _, slots in OPTIONB_GRAMMAR.all_phrases() for name in slots}
)

# Which (single) slot each slotted intent carries. Slotted intents in this
# grammar each have exactly one slot; unslotted intents are absent here.
SLOT_FOR_INTENT: dict[str, str] = {}
for _text, _intent, _slots in OPTIONB_GRAMMAR.all_phrases():
    if _slots:
        SLOT_FOR_INTENT[_intent] = next(iter(_slots))

# The reminder task vocabulary, grammar-derived (the UI's "Log" button is
# restricted to the same three choices as voice -- decision Q11).
REMINDER_TASKS: frozenset[str] = frozenset(
    {slots["TASK"] for _, intent, slots in OPTIONB_GRAMMAR.all_phrases()
     if intent == "CREATE_REMINDER"}
)

# Canonical display label per intent (title-case; collapses all three phrasings
# into one). Used by the indicator's detected-word line and UI logging.
INTENT_LABELS: dict[str, str] = {
    "PLAY_MUSIC": "Play Music",
    "PAUSE": "Pause",
    "STOP": "Stop",
    "NEXT": "Next",
    "VOLUME_UP": "Volume Up",
    "VOLUME_DOWN": "Volume Down",
    "LIGHT_ON": "Lights On",
    "LIGHT_OFF": "Lights Off",
    "BRIGHTNESS": "Brightness",
    "COLOR": "Color",
    "TEMPERATURE": "Temperature",
    "WEATHER": "Weather",
    "TIME": "Time",
    "TIMER": "Timer",
    "ALARM": "Alarm",
    "CALL": "Call",
    "MESSAGE": "Message",
    "CREATE_REMINDER": "Reminder",
    "LIST_REMINDERS": "Reminders",
}


def intent_label(intent: str) -> str:
    """Canonical display label for `intent` (falls back to the raw name)."""
    return INTENT_LABELS.get(intent, intent.title())


def format_detected(intent: str, slots: dict[str, str]) -> str:
    """The indicator's detected-word string: the canonical label, plus the
    slot's canonical value if the intent is slotted and the slot is present.
    e.g. ("BRIGHTNESS", {"PERCENT": "60 percent"}) -> "Brightness 60 percent"."""
    label = intent_label(intent)
    slot_name = SLOT_FOR_INTENT.get(intent)
    if slot_name and slot_name in slots:
        return f"{label} {slots[slot_name]}"
    return label
