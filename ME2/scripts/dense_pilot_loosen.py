#!/usr/bin/env python
"""Looser-threshold probe for D2 (margin OFF). Experiment-only.

Idea: iso-accept tau (-0.907) leaves large FA headroom vs production, so a looser
GLOBAL tau might recover PAUSE/STOP accepts. Selection uses VAL ONLY:
  tau* = the loosest (lowest) tau whose val reject-probe FAs are
         <= production's val FAs per condition (B = baseline -0.1 + margin 4.0;
         B_val = clean 3, noisy 8, computed in-code, not hardcoded).
Pre-registered rule (frozen before the run) for S = D2 margin-off at tau*, vs B:
  L1 test reject FA: clean <= B and noisy <= B
  L2 ghost FA (141 clips, independent) <= B's ghost FA
  L3 exact-correct >= B on val AND test, clean AND noisy
  L4 per intent (>=50 targets), clean & noisy exact >= B - 3pp on val AND test
PASS iff L1-L4. Test was seen for d2 already (non-independent); ghosts and the
val-only selection are the independent parts. Caveat: tau* sits at a val FA cliff
edge, so it may be optimistic; the printed tau curve shows how fragile it is.
"""
import csv, importlib.util, json, sys
from concurrent.futures import ProcessPoolExecutor
from fractions import Fraction
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
D = ROOT / "out/vcm/option-d-fil50-ambient-rir-135m/dense_pilot"
spec = importlib.util.spec_from_file_location("g", ROOT / "scripts/dense_pilot_ghosts.py")
g = importlib.util.module_from_spec(spec); sys.modules["g"] = g; spec.loader.exec_module(g)
from me2_voicegen.vcm import dense_pilot as dp


def main():
    recs = dp._load_records(D / "records")
    rows = list(csv.DictReader(open(g.MANIFEST, newline="")))
    idx = list(enumerate(rows))
    with ProcessPoolExecutor(32) as pool:
        parts = list(pool.map(g._ghost_worker, [(str(g.CHECKPOINT), str(g.MANIFEST), idx[k::32]) for k in range(32)]))
    by = {i: lp for p in parts for i, lp in p}
    ghost = dp.score_rows([by[i] for i in range(len(rows))],
                          [{"index": i, "bucket": "ghost", "label": "x"} for i in range(len(rows))], workers=32)

    base = {"default": dp.BASELINE_THRESHOLD}
    B = {(s, c): dp.metrics(recs[(s, c)], "baseline", base, dp.PRODUCTION_MARGIN) for s in ("val", "test") for c in dp.CONDITIONS}
    b_ghost = sum(dp.accepts(r, "baseline", base, dp.PRODUCTION_MARGIN) for r in ghost)
    budget = {c: B[("val", c)]["reject_fa"] for c in dp.CONDITIONS}

    def val_fa(tau, cond):
        return sum(1 for r in recs[("val", cond)] if r["bucket"] in dp.REJECT and r["d2"] is not None
                   and r["intent"] is not None and r["d2"] >= tau)
    cands = sorted({r["d2"] for r in recs[("val", "clean")] if r["d2"] is not None and r["bucket"] == dp.TARGET})
    cs = np.array(cands)
    fa = {c: np.array([[r["d2"] for r in recs[("val", c)] if r["bucket"] in dp.REJECT and r["d2"] is not None]]) for c in dp.CONDITIONS}
    ok = np.ones(len(cs), bool)
    for c in dp.CONDITIONS:
        ok &= (fa[c][0][None, :] >= cs[:, None]).sum(1) <= budget[c]
    tau_star = float(cs[ok].min())

    def eval_tau(tau):
        t = {"default": tau}
        m = {(s, c): dp.metrics(recs[(s, c)], "d2", t, None) for s in ("val", "test") for c in dp.CONDITIONS}
        return m, sum(dp.accepts(r, "d2", t, None) for r in ghost)

    S, s_ghost = eval_tau(tau_star)
    L1 = all(S[("test", c)]["reject_fa"] <= B[("test", c)]["reject_fa"] for c in dp.CONDITIONS)
    L2 = s_ghost <= b_ghost
    L3 = all(S[(s, c)]["exact"] >= B[(s, c)]["exact"] for s in ("val", "test") for c in dp.CONDITIONS)
    failing = {}
    for s in ("val", "test"):
        for label, n in B[(s, "clean")]["per_intent_n"].items():
            if n < dp.R6_MIN_TARGETS:
                continue
            for c in dp.CONDITIONS:
                if Fraction(S[(s, c)]["per_intent_exact"].get(label, 0) - B[(s, c)]["per_intent_exact"].get(label, 0), n) < -dp.R6_ACC_BUDGET:
                    failing.setdefault(label, []).append(f"{s}/{c}")
    L4 = not failing
    out = {"tau_star": tau_star, "iso_tau": dp.calibrate(recs[("val", "clean")], "d2", False, None)["default"],
           "budget_val_fa": budget, "B_ghost_fa": b_ghost, "S_ghost_fa": s_ghost,
           "L1": L1, "L2": L2, "L3": L3, "L4": L4, "L4_failing": failing,
           "verdict": "PASS" if (L1 and L2 and L3 and L4) else "FAIL", "curve": []}
    for tau in np.linspace(out["iso_tau"], tau_star - 0.3, 8).tolist() + [tau_star]:
        m, gh = eval_tau(tau)
        out["curve"].append({"tau": round(tau, 3), "ghost_fa": gh,
                             **{f"{s}/{c}": [m[(s, c)]["exact"], m[(s, c)]["reject_fa"]] for s in ("val", "test") for c in dp.CONDITIONS},
                             "test_PAUSE_STOP_clean": [m[("test", "clean")]["per_intent_exact"].get(i, 0) for i in ("PAUSE", "STOP")],
                             "test_PAUSE_STOP_noisy": [m[("test", "noisy_s0")]["per_intent_exact"].get(i, 0) for i in ("PAUSE", "STOP")]})
    out["B"] = {f"{s}/{c}": [B[(s, c)]["exact"], B[(s, c)]["reject_fa"]] for s in ("val", "test") for c in dp.CONDITIONS}
    (D / "metadata" / "dense_loosen_probe.json").write_text(json.dumps(out, indent=1))
    print(json.dumps({k: v for k, v in out.items() if k != "curve"}, indent=1))
    for row in out["curve"]:
        print(row)


if __name__ == "__main__":
    main()
