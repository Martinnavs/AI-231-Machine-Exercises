from me2_voicegen.vcm.optionb.grammar import OPTIONB_GRAMMAR
from me2_voicegen.vcm.optionb.import_ai231 import convert_row, is_val_speaker
from me2_voicegen.vcm.text import normalize_text, resolve_transcript


def _row(**kw):
    base = {"command": "ALARM", "variation": "1", "slot_value": "6:00 AM", "transcript": "Alarm 6:00 AM",
            "variation_match": "exact"}
    return {**base, **kw}


def test_alarm_time_folds_to_grammar_value():
    out = convert_row(_row(), "train")
    assert out["slot_value"] == "6 AM" and out["label"] == "ALARM"
    spoken = normalize_text(resolve_transcript({"source_dataset": "optionb", **out}))
    assert OPTIONB_GRAMMAR.accepts(spoken) == [("ALARM", {"ALARM_TIME": "6 AM"})]


def test_off_schema_slot_skipped_and_oos_mapped():
    assert convert_row(_row(variation="", slot_value=""), "train") is None
    speech = convert_row(_row(command="OUT_OF_SCOPE", variation="", transcript="call 2023 now"), "test")
    assert (speech["bucket"], speech["label"]) == ("babble", "unknown") and "2023" not in speech["transcript"]
    assert convert_row(_row(command="OUT_OF_SCOPE", variation="", transcript=""), "test")["label"] == "silence"


def test_val_speakers_are_deterministic_subset():
    assert is_val_speaker("s10") == is_val_speaker("s10")
