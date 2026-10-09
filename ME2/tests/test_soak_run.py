"""`scripts/soak_run.py` scoring and `scripts/build_soak_continuous.py` bookkeeping, on synthetic records (no audio, no models)."""
from __future__ import annotations

import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _load(name: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _session(label, slot="", oos=False, gap=0.2, speech_end=3.0, unit_end=4.0, records=()):
    return {"label": label, "slot_value": slot, "is_oos": oos, "gap_s": gap, "speech_end_s": speech_end, "unit_end_s": unit_end,
            "returncode": 0, "records": list(records)}


def _trig(t, intent, slots=None, text="x"):
    return {"event": "trigger", "t_seconds": t, "intent": intent, "slots": slots or {}, "text": text, "gate_ms": 2.0, "decode_ms": 60.0}


def test_score_counts_correct_wrong_missed_early_and_oos():
    soak = _load("soak_run")
    results = [
        _session("STOP", records=[_trig(3.4, "STOP")]),                                              # correct, 0.4 s after speech end
        _session("TIMER", "10 seconds", records=[_trig(3.5, "TIMER", {"DURATION": "1 minute"}, "[classifier]")]),  # wrong slot
        _session("PAUSE", records=[]),                                                                 # missed, no period opened
        _session("NEXT", records=[_trig(2.5, "NEXT")]),                                              # correct but early
        _session("unknown", oos=True, records=[_trig(3.0, "WEATHER")]),                              # false accept
        _session("unknown", oos=True, records=[]),
        _session("STOP", records=[_trig(3.2, "STOP"), _trig(6.0, "STOP")]),                          # second trigger in the gap
    ]
    s = soak.score(results)
    assert (s["commands"], s["correct_first_trigger"], s["missed"], s["missed_no_period_opened"]) == (5, 3, 1, 1)
    assert s["wrong_intent_triggers"] == 0 and s["wrong_action_first_triggers"] == 1      # the wrong slot is a wrong action, not a wrong intent
    assert s["early_first_triggers"] == 1 and s["answered_by_classifier"] == 0
    assert (s["oos_sessions"], s["oos_false_accepts"]) == (2, 1)
    assert s["triggers_in_ambient_gaps"] == 1 and s["extra_triggers"] == 1
    assert s["objective"] == 3 - 2 * (1 + 1 + 1)
    assert abs(s["latency_after_speech_end_s"]["correct_p50"] - 0.4) < 0.31
    assert s["decode_ms"]["mean"] == 60.0


def test_slot_ok_is_case_insensitive_and_skips_unslotted():
    soak = _load("soak_run")
    assert soak.slot_ok({"slots": {"DURATION": "10 Seconds"}}, "10 seconds")
    assert not soak.slot_ok({"slots": {"DURATION": "1 minute"}}, "10 seconds")
    assert soak.slot_ok({"slots": {}}, "")
