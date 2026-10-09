"""`scripts/hybrid_metrics.py` metric definitions on synthetic rows (no models, no audio)."""
from __future__ import annotations

import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _load():
    spec = importlib.util.spec_from_file_location("hybrid_metrics", ROOT / "scripts" / "hybrid_metrics.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _row(bucket, label, variation="", slot="", source="optionb", rel="test/audio/test_0_group_synthetic_x.wav"):
    return {"bucket": bucket, "label": label, "variation": variation, "slot_value": slot, "source_dataset": source, "source_relpath": rel}


def test_metrics_definitions():
    hm = _load()
    rows = [
        _row("target_commands", "STOP", "Stop"),                                                    # right
        _row("target_commands", "STOP", "Stop", rel="test/audio/test_1_real_voice_STOP.wav"),     # rejected, human voice
        _row("target_commands", "TIMER", "Timer 10 seconds", "10 seconds"),                         # right command, wrong slot
        _row("babble", "unknown"),                                                                  # out of scope, rejected
        _row("babble", "unknown"),                                                                  # out of scope, accepted
        _row("babble", "unknown", source="negative_babble"),                                        # synthetic negative, accepted
    ]
    pred = {0: ("STOP", []), 1: (None, []), 2: ("TIMER", ["1 minute"]), 3: (None, []), 4: ("NEXT", []), 5: ("PAUSE", [])}
    m = hm.metrics(rows, pred)
    assert m["accuracy"] == (1, 3) and m["command_acc"] == (2, 3) and m["slot_acc"] == (0, 1)
    assert m["human"] == (0, 1) and m["synthetic"] == (1, 2)
    assert m["oos_fa"] == (1, 2) and m["false_reject"] == (1, 3) and m["synneg_misfire"] == (1, 1)
    # recall per variation: Stop 1/2, Timer 0/1, plus the out-of-scope group 1/2 -> mean 1/3
    assert abs(m["var_bal_acc"] - (0.5 + 0.0 + 0.5) / 3) < 1e-9 and m["n_variations"] == 2


def test_wilson_interval_is_sane():
    hm = _load()
    lo, hi = hm.wilson(50, 100)
    assert 0.40 < lo < 0.5 < hi < 0.60
