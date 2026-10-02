"""Intent and slot label tables for the attention heads, derived from OPTIONB_GRAMMAR.

Never hardcoded: `all_phrases()` is the source of truth, and the derivation
raises if the grammar stops having the shape the heads assume (one slot per
slotted intent), so a grammar change cannot silently shift head sizes.
"""

from __future__ import annotations

from me2_voicegen.vcm.optionb.grammar import OPTIONB_GRAMMAR

# Manifest `label` of babble rows / silence rows; they are not grammar intents.
EXTRA_INTENT_CLASSES: tuple[str, ...] = ("unknown", "silence")


def _derive() -> tuple[tuple[str, ...], dict[str, tuple[str, tuple[str, ...]]]]:
    intents: set[str] = set()
    values: dict[str, dict[str, set[str]]] = {}
    for _text, intent, slots in OPTIONB_GRAMMAR.all_phrases():
        intents.add(intent)
        for name, value in slots.items():
            values.setdefault(intent, {}).setdefault(name, set()).add(value)
    for intent, by_name in values.items():
        if len(by_name) != 1:
            raise ValueError(f"{intent} has slots {sorted(by_name)}; the heads assume exactly one per intent")
    slots_table = {
        intent: (name, tuple(sorted(vals)))
        for intent, by_name in sorted(values.items())
        for name, vals in by_name.items()
    }
    return tuple(sorted(intents)) + EXTRA_INTENT_CLASSES, slots_table


INTENT_CLASSES, SLOTS = _derive()
"""`INTENT_CLASSES`: the 19 intents (sorted) + `unknown` + `silence`.
`SLOTS`: slotted intent -> (slot name, sorted canonical values)."""

N_SLOT_VALUES = 3
if any(len(vals) != N_SLOT_VALUES for _name, vals in SLOTS.values()):
    raise ValueError("the slot heads assume 3 values per slot")


def slot_head_name(intent: str) -> str:
    """Attribute/key name of an intent's slot head, e.g. `slot_ALARM_ALARM_TIME`."""
    return f"slot_{intent}_{SLOTS[intent][0]}"


SLOT_INTENTS: tuple[str, ...] = tuple(SLOTS)
"""Slotted intents in head order; column k of a batch's `slot_targets` belongs to SLOT_INTENTS[k]."""

IGNORE = -100
"""CrossEntropy ignore_index: no target for this row/head."""


def row_targets(label: str, text: str | None, slot_value: str = "") -> tuple[int, tuple[int, ...], str]:
    """Per-row head targets: `(intent_id, slot_targets, source)`.

    `label` is the manifest `label` (intent id comes from it); `text` is the
    row's normalized CTC transcript (`None` if it has none). The slot target
    comes from an exact grammar-phrase match of `text`; failing that, from
    the manifest's canonical `slot_value` (paraphrases such as "set the timer
    for one minute" carry one); failing that it is ignored. `source` is
    counted by the caller so nothing is dropped silently: `phrase`,
    `manifest_slot_value`, `no_slot_target`, `phrase_label_mismatch`
    (transcript is another intent's phrase), `nonslotted`, `nontarget`
    (babble/silence), `intent_unmapped` (label not in INTENT_CLASSES).
    """
    slots = [IGNORE] * len(SLOT_INTENTS)
    if label not in INTENT_CLASSES:
        return IGNORE, tuple(slots), "intent_unmapped"
    intent_id = INTENT_CLASSES.index(label)
    if label not in SLOTS:
        return intent_id, tuple(slots), "nontarget" if label in EXTRA_INTENT_CLASSES else "nonslotted"
    slot_name, values = SLOTS[label]
    k = SLOT_INTENTS.index(label)
    accepted = OPTIONB_GRAMMAR.accepts(text) if text else None
    if accepted:
        for intent, found in accepted:
            if intent == label and found.get(slot_name) in values:
                slots[k] = values.index(found[slot_name])
                return intent_id, tuple(slots), "phrase"
        return intent_id, tuple(slots), "phrase_label_mismatch"
    # manifest slot values are capitalised ("Red", "Drink water"); the grammar's are lowercase
    lowered = [v.lower() for v in values]
    if slot_value.lower() in lowered:
        slots[k] = lowered.index(slot_value.lower())
        return intent_id, tuple(slots), "manifest_slot_value"
    return intent_id, tuple(slots), "no_slot_target"
