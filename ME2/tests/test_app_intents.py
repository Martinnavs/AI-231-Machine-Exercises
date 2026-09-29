"""Vocabulary guards for the UI site: the intent/slot set the UI accepts
is derived from the production Option B grammar, and these pin that
derivation so UI and model can never drift (decision Q1-a)."""

from app.intents import (
    INTENTS,
    INTENT_LABELS,
    REMINDER_TASKS,
    SLOT_FOR_INTENT,
    SLOT_NAMES,
    format_detected,
)


def test_nineteen_intents():
    assert len(INTENTS) == 19
    assert "PLAY_MUSIC" in INTENTS
    assert "LIST_REMINDERS" in INTENTS


def test_six_slot_names():
    assert SLOT_NAMES == frozenset(
        {"ALARM_TIME", "COLOR", "DEGREES", "DURATION", "PERCENT", "TASK"}
    )


def test_exactly_the_six_slotted_intents_carry_slots():
    assert set(SLOT_FOR_INTENT) == {
        "ALARM",
        "BRIGHTNESS",
        "COLOR",
        "CREATE_REMINDER",
        "TEMPERATURE",
        "TIMER",
    }


def test_every_intent_has_a_canonical_label():
    assert set(INTENT_LABELS) == set(INTENTS)
    assert INTENT_LABELS["BRIGHTNESS"] == "Brightness"


def test_reminder_tasks_are_the_grammar_choices():
    assert REMINDER_TASKS == frozenset({"drink water", "exercise", "study"})


def test_format_detected_unslotted():
    assert format_detected("PLAY_MUSIC", {}) == "Play Music"


def test_format_detected_slotted_uses_canonical_value():
    assert format_detected("BRIGHTNESS", {"PERCENT": "60 percent"}) == (
        "Brightness 60 percent"
    )
    assert format_detected("ALARM", {"ALARM_TIME": "6 AM"}) == "Alarm 6 AM"
