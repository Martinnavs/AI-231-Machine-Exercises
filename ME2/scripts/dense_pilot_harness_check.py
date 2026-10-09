#!/usr/bin/env python
"""Cross-check (dense-d2-loose-impl task 4a): the REAL `vcm.evaluate --score-mode per_char`
val threshold sweep vs. the offline D2 records at the same thresholds (margin off).
Every grid row must agree exactly on (target accept rate, reject-probe FA rate);
the report's test split at its chosen threshold must agree on accepted/exact/FA counts.
"""
import json, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
D = ROOT / "out/vcm/option-d-fil50-ambient-rir-135m/dense_pilot"
from me2_voicegen.vcm import dense_pilot as dp


def main():
    rep = json.loads((D / "eval_per_char/metadata/eval_report.json").read_text())
    sec = rep["grammar_sections"][0]
    assert sec["score_mode"] == "per_char", sec.get("score_mode")
    recs = dp._load_records(D / "records")
    val = recs[("val", "clean")]
    bad = []
    for row in sec["threshold_sweep_on_val"]:
        t = row["threshold"]
        m = dp.metrics(val, "d2", {"default": t}, None)
        want_t = m["accepted"] / m["n_targets"]
        want_fa = m["reject_fa"] / (m["n_babble"] + m["n_silence"])
        if abs(row["target_accept_rate"] - want_t) > 1e-12 or abs(row["false_accept_rate"] - want_fa) > 1e-12:
            bad.append((t, row["target_accept_rate"], want_t, row["false_accept_rate"], want_fa))
    t = sec["chosen_operating_threshold"]
    m = dp.metrics(recs[("test", "clean")], "d2", {"default": t}, None)
    ts = sec["test_split"]
    test_ok = (ts["n_accepted"], ts["n_exact_correct"], ts["false_accept_rate_babble"]["false_accepts"],
               ts["false_accept_rate_silence"]["false_accepts"]) == (m["accepted"], m["exact"], m["babble_fa"], m["silence_fa"])
    out = {"grid_rows_checked": len(sec["threshold_sweep_on_val"]), "grid_mismatches": bad,
           "chosen_threshold_youden": t, "test_counts_match_at_chosen": test_ok,
           "verdict": "MATCH" if (not bad and test_ok) else "MISMATCH"}
    (D / "metadata" / "dense_harness_check.json").write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
