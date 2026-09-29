"""T2 tests for vcm/dense_pilot.py, driven by the REAL val fixture written by
scripts/dense_pilot_dump.py (never hand-built logits for the decoder paths)."""

import csv
import json
import math
import shutil
import subprocess
import sys
from fractions import Fraction
from pathlib import Path

import numpy as np
import pytest

from me2_voicegen.vcm import dense_pilot as dp
from me2_voicegen.vcm.decoder import decode_utterance
from me2_voicegen.vcm.optionb.grammar import OPTIONB_GRAMMAR

FIXTURE_DIR = Path(__file__).resolve().parent / "fixtures" / "dense_pilot"
MV4_MAX_GAP = 15.0  # measured full val targets: p95 9.64 clean / 12.46 noisy, + margin


def load_fixture(cond):
    z = np.load(FIXTURE_DIR / f"val_{cond}.npz")
    with (FIXTURE_DIR / f"val_{cond}.rows.csv").open(newline="") as f:
        rows = list(csv.DictReader(f))
    lps = [z["logp"][z["offsets"][k]:z["offsets"][k + 1]] for k in range(len(rows))]
    return lps, rows


@pytest.fixture(scope="module")
def scored():
    out = {}
    for cond in dp.CONDITIONS:
        lps, rows = load_fixture(cond)
        out[cond] = (lps, rows, dp.score_rows(lps, rows, workers=1))
    return out


def brute_viterbi(logp, text):
    """Reference CTC Viterbi over the blank-extended target: (total, nonblank mean)."""
    from me2_voicegen.vcm.alphabet import encode

    tgt = encode(text)
    ext = [0]
    for t in tgt:
        ext += [t, 0]
    T, S = logp.shape[0], len(ext)
    NEG = -1e30
    dp_ = np.full((T, S), NEG)
    bp = np.zeros((T, S), dtype=int)
    dp_[0, 0] = logp[0, 0]
    if S > 1:
        dp_[0, 1] = logp[0, ext[1]]
    for t in range(1, T):
        for s in range(S):
            cands = [(dp_[t - 1, s], s)]
            if s >= 1:
                cands.append((dp_[t - 1, s - 1], s - 1))
            if s >= 2 and ext[s] != 0 and ext[s] != ext[s - 2]:
                cands.append((dp_[t - 1, s - 2], s - 2))
            best, arg = max(cands)
            dp_[t, s] = best + logp[t, ext[s]]
            bp[t, s] = arg
    s = S - 1 if dp_[T - 1, S - 1] >= dp_[T - 1, S - 2] else S - 2
    total = dp_[T - 1, s]
    path = []
    for t in range(T - 1, -1, -1):
        path.append(s)
        s = bp[t, s]
    path.reverse()
    scores = np.array([logp[t, ext[path[t]]] for t in range(T)])
    nb = np.array([ext[p] != 0 for p in path])
    return float(total), float(scores[nb].mean())


# (a) baseline reproduces the unchanged decoder exactly
def test_baseline_confidence_equals_decode_utterance(scored):
    for cond in dp.CONDITIONS:
        lps, _, recs = scored[cond]
        for lp, rec in zip(lps, recs):
            d = decode_utterance(lp, OPTIONB_GRAMMAR, -0.1, beam_width=50)
            if rec["intent"] is not None:
                assert rec["confidence"] == d.confidence
            assert dp.accepts(rec, "baseline", {"default": -0.1}, None) == (not d.no_match)


# (b) forced_align D1 == brute-force Viterbi
def test_d1_matches_bruteforce_viterbi(scored):
    checked = 0
    for cond in dp.CONDITIONS:
        lps, _, recs = scored[cond]
        for lp, rec in zip(lps, recs):
            if rec["intent"] is None or checked >= 6:
                continue
            total, mean_nb = brute_viterbi(lp, rec["grammar_text"])
            assert rec["viterbi_total"] == pytest.approx(total, abs=1e-4)
            assert rec["d1"] == pytest.approx(mean_nb, abs=1e-4)
            checked += 1
    assert checked >= 3


# (c) beam mass (logsumexp over alignments) >= Viterbi (max), plus (g) MV4 bound
def test_raw_mass_dominates_viterbi_and_mv4_bound(scored):
    seen = 0
    for cond in dp.CONDITIONS:
        for rec in scored[cond][2]:
            if rec["mass_minus_viterbi"] is None:
                continue
            seen += 1
            assert rec["command_raw_score"] >= rec["viterbi_total"] - 1e-5
            if rec["bucket"] == dp.TARGET:
                assert rec["mass_minus_viterbi"] <= MV4_MAX_GAP
            assert not rec["align_failed"]
    assert seen > 10


# (d) post-hoc margin gate == live gated decode
def test_posthoc_margin_matches_live_gate(scored):
    for cond in dp.CONDITIONS:
        lps, _, recs = scored[cond]
        for lp, rec in zip(lps, recs):
            live = decode_utterance(lp, OPTIONB_GRAMMAR, -0.1, beam_width=50,
                                    required_command_margin=dp.PRODUCTION_MARGIN)
            assert dp.accepts(rec, "baseline", {"default": -0.1}, dp.PRODUCTION_MARGIN) == (not live.no_match)


# (h) infeasible alignment raises instead of yielding a score
def test_infeasible_alignment_raises():
    lp = np.log(np.full((3, 29), 1 / 29, dtype=np.float32))
    with pytest.raises(dp.AlignmentFailed):
        dp.viterbi_nonblank(lp, "stop now")


# (f) iso-accept calibration on a hand-built case
def _rec(intent, conf, bucket=dp.TARGET, gap=None):
    return {"intent": intent, "confidence": conf, "d1": conf, "d2": conf, "bucket": bucket,
            "label": intent, "incomplete_gap": gap, "index": 0}


def test_iso_threshold_and_calibration():
    assert dp.iso_threshold([-0.5, -0.1, -0.3, -0.2], 2) == -0.2
    assert dp.iso_threshold([-0.5], 0) == float("inf")
    assert dp.iso_threshold([-0.5], 3) == float("-inf")
    val = [_rec("A", -0.05), _rec("A", -0.09), _rec("A", -0.5), _rec("B", -0.08),
           _rec("B", -0.2), _rec("B", -0.3), _rec("C", -0.6)]
    # baseline @-0.1 accepts A:2, B:1 -> 3 total
    g = dp.calibrate(val, "d1", per_intent=False, margin=None)
    assert g["default"] == -0.09  # 3rd best of all scores (-0.05, -0.08, -0.09)
    pi = dp.calibrate(val, "d1", per_intent=True, margin=None)
    assert pi["A"] == -0.09 and pi["B"] == -0.08 and "C" not in pi
    # margin gate removes a row from eligibility
    val2 = val + [_rec("A", -0.01, gap=1.0)]
    assert dp.calibrate(val2, "d1", False, 4.0)["default"] == -0.09


# (e) frozen rule boundaries
def _inp(**kw):
    base = dict(noisy_fa_B=10, noisy_fa_S=5, noisy_babble_fa_B=6, noisy_babble_fa_S=5,
                removed=8, added=0, clean_fa_B=4, clean_fa_S=4,
                clean_exact_B=1000, clean_exact_S=1000, clean_n=1000,
                noisy_exact_B=1000, noisy_exact_S=1000, noisy_n=1000,
                per_intent={"TIME": dict(clean_n=50, clean_exact_B=50, clean_exact_S=50,
                                         noisy_exact_B=50, noisy_exact_S=50)},
                noisy_ts_fa_B=5, noisy_ts_fa_S=5)
    base.update(kw)
    return base


def test_decide_go_and_parity_invalid():
    assert dp.decide(_inp(), True)["verdict"] == "GO"
    assert dp.decide(_inp(), False)["verdict"] == "INVALID"


def test_r1_boundary_half():
    assert dp.decide(_inp(noisy_fa_S=5), True)["criteria"]["R1"]["pass"]
    assert not dp.decide(_inp(noisy_fa_S=6), True)["criteria"]["R1"]["pass"]


def test_r2_strict():
    assert not dp.decide(_inp(noisy_babble_fa_S=6), True)["criteria"]["R2"]["pass"]


def test_r3_sign_test_alpha():
    assert dp.sign_test_p(8, 0) == pytest.approx(1 / 256)
    assert dp.sign_test_p(0, 0) == 1.0
    assert dp.decide(_inp(removed=5, added=0), True)["criteria"]["R3"]["pass"]  # p=1/32
    assert not dp.decide(_inp(removed=4, added=0), True)["criteria"]["R3"]["pass"]  # p=1/16


def test_r4_r7_non_strict():
    assert dp.decide(_inp(clean_fa_S=4), True)["criteria"]["R4"]["pass"]
    assert not dp.decide(_inp(clean_fa_S=5), True)["criteria"]["R4"]["pass"]
    assert dp.decide(_inp(noisy_ts_fa_S=5), True)["criteria"]["R7"]["pass"]
    assert not dp.decide(_inp(noisy_ts_fa_S=6), True)["criteria"]["R7"]["pass"]


def test_r5_half_point_budget_exact():
    assert dp.decide(_inp(clean_exact_S=995), True)["criteria"]["R5"]["pass"]
    assert not dp.decide(_inp(clean_exact_S=994), True)["criteria"]["R5"]["pass"]
    assert not dp.decide(_inp(noisy_exact_S=994), True)["criteria"]["R5"]["pass"]


def test_r6_three_point_budget_and_min_targets():
    pi = lambda n, s: {"TIME": dict(clean_n=n, clean_exact_B=n, clean_exact_S=s,
                                    noisy_exact_B=n, noisy_exact_S=n)}
    # n=100: dropping 3 is on the boundary (pass), 4 fails
    assert dp.decide(_inp(per_intent=pi(100, 97)), True)["criteria"]["R6"]["pass"]
    assert not dp.decide(_inp(per_intent=pi(100, 96)), True)["criteria"]["R6"]["pass"]
    # n=49 is below the 50-target floor: ignored
    assert dp.decide(_inp(per_intent=pi(49, 0)), True)["criteria"]["R6"]["pass"]


def test_baseline_far_absorbed_is_nogo():
    d = dp.decide(_inp(noisy_fa_B=3, noisy_fa_S=0), True)
    assert d["verdict"] == "NO-GO" and d["reason"] == "baseline_far_already_absorbed_by_margin"


# e2e through the real CLI entry points; fixture parity cannot hold -> INVALID
def test_end_to_end_cli_score_then_report(tmp_path):
    (tmp_path / "logits").mkdir()
    for split in ("val", "test"):
        for cond in dp.CONDITIONS:
            for ext in ("npz", "rows.csv"):
                shutil.copy(FIXTURE_DIR / f"val_{cond}.{ext}", tmp_path / "logits" / f"{split}_{cond}.{ext}")
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
    dp.main(["score", "--dump-dir", str(tmp_path), "--workers", "2"])
    dp.main(["report", "--dump-dir", str(tmp_path)])
    rep = json.loads((tmp_path / "metadata" / "dense_pilot_report.json").read_text())
    assert rep["verdict"]["verdict"] == "INVALID" and rep["parity"]["pass"] is False
    assert {"parity", "selected_config", "grid", "overlap_margin_x_dense_test", "verdict"} <= set(rep)
    expected_cells = len(dp.CONFIGS) * len(dp.MARGINS) * 2 * len(dp.CONDITIONS)
    assert len(rep["grid"]) == expected_cells
    assert (tmp_path / "metadata" / "dense_pilot_report.md").read_text().startswith("# Dense")


def test_no_production_module_imports_dense_pilot():
    root = Path(__file__).resolve().parents[1]
    out = subprocess.run(["grep", "-rln", "dense_pilot", str(root / "src")],
                         capture_output=True, text=True).stdout.split()
    assert all(p.endswith("vcm/dense_pilot.py") for p in out if "__pycache__" not in p)


# ---- hybrid pilot ----------------------------------------------------------

def test_relax_only_never_stricter_than_global():
    val = [_rec("A", -0.05), _rec("A", -0.09), _rec("A", -0.5), _rec("B", -0.08),
           _rec("B", -0.2), _rec("B", -0.3), _rec("C", -0.6)]
    tau = dp.relax_only_tau(val)
    assert all(v <= tau["default"] for v in tau.values())


def test_val_halves_deterministic_and_group_disjoint():
    ids = [f"s{i}" for i in range(200)]
    halves = {g: dp.half_of(g) for g in ids}
    assert halves == {g: dp.half_of(g) for g in ids}
    assert 60 < sum(halves.values()) < 140  # both halves populated


def _m(exact, fa, per_n=None, per_exact=None, n=1000):
    return {"n_targets": n, "exact": exact, "reject_fa": fa,
            "per_intent_n": per_n or {"X": 100}, "per_intent_exact": per_exact or {"X": 100}}


def _cond(**kw):
    return {"clean": _m(**kw), "noisy_s0": _m(**kw)}


def test_gates_boundaries():
    B = _cond(exact=1000, fa=4)
    C = _cond(exact=1000, fa=0)
    assert dp._all_pass(dp.gates(B, _cond(exact=995, fa=4, per_exact={"X": 97}), C)) is False  # G5: 8 > 0+2
    ok = dp.gates(B, {"clean": _m(995, 1, per_exact={"X": 97}), "noisy_s0": _m(995, 1, per_exact={"X": 97})}, C)
    assert dp._all_pass(ok)  # boundaries: 0.5pp, 3pp, total FA 2 <= 0+2
    bad = dp.gates(B, {"clean": _m(994, 1), "noisy_s0": _m(995, 1)}, C)
    assert not bad["G3"]
    bad4 = dp.gates(B, {"clean": _m(1000, 0, per_exact={"X": 96}), "noisy_s0": _m(1000, 0)}, C)
    assert bad4["G4_failing"] == ["X"]
    assert dp.gates(B, {"clean": _m(1000, 5), "noisy_s0": _m(1000, 0)}, C)["G2"] is False


def test_hybrid_verdict_paths():
    ok = {"G1": True, "G2": True, "G3": True, "G4": True, "G5": True, "G4_failing": []}
    g4 = {**ok, "G4": False, "G4_failing": ["PAUSE"]}
    assert dp.hybrid_verdict(ok, ok) == "GO"
    assert dp.hybrid_verdict(ok, g4) == "INCONCLUSIVE"
    assert dp.hybrid_verdict(ok, {**g4, "G4_failing": ["A", "B", "C"]}) == "NO-GO"
    assert dp.hybrid_verdict(g4, ok) == "NO-GO"
    assert dp.hybrid_verdict(ok, {**g4, "G1": False}) == "NO-GO"
