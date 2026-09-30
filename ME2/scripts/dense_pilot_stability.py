#!/usr/bin/env python
"""Threshold-stability check for D2 per_char tau* (dense-d2-loose-impl task 4).

tau* rule (same as scripts/dense_pilot_loosen.py): the loosest global D2 threshold
whose val reject-probe FAs (per condition) are <= production's (baseline -0.1 +
margin 4.0) on the SAME rows. Here it is fit separately on each of two
speaker-disjoint val halves (crc32(group_id) & 1), then each half's tau*
is applied to the OTHER half.
Pre-registered (frozen before the run) stability rule:
  S1 |tau*_half0 - tau*_half1| <= 0.30
  S2 held-out reject FAs (clean AND noisy) <= production's on that held-out half, both directions
  S3 held-out exact-correct, pooled over both directions, >= production - 0.5pp, clean AND noisy
STABLE iff S1-S3. Otherwise the loose tau* is a val artifact and must not be shipped as-is.
"""
import csv, json, sys
from fractions import Fraction
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
D = ROOT / "out/vcm/option-d-fil50-ambient-rir-135m/dense_pilot"
from me2_voicegen.vcm import dense_pilot as dp

S1_MAX_TAU_GAP = 0.30


def tau_star(half, budget):
    cands = np.array(sorted({r["d2"] for r in half["clean"] if r["d2"] is not None and r["bucket"] == dp.TARGET}))
    ok = np.ones(len(cands), bool)
    for c in dp.CONDITIONS:
        fa = np.array([r["d2"] for r in half[c] if r["bucket"] in dp.REJECT and r["intent"] is not None and r["d2"] is not None])
        ok &= (fa[None, :] >= cands[:, None]).sum(1) <= budget[c]
    return float(cands[ok].min())


def main():
    recs = dp._load_records(D / "records")
    gid = {int(r["index"]): r["group_id"] for r in csv.DictReader(open(D / "logits/val_clean.rows.csv", newline=""))}
    halves = {h: {c: [r for r in recs[("val", c)] if dp.half_of(gid[r["index"]]) == h] for c in dp.CONDITIONS} for h in (0, 1)}
    base = {"default": dp.BASELINE_THRESHOLD}
    B = {h: {c: dp.metrics(halves[h][c], "baseline", base, dp.PRODUCTION_MARGIN) for c in dp.CONDITIONS} for h in (0, 1)}
    out = {"tau_star": {}, "held_out": {}}
    for h in (0, 1):
        budget = {c: B[h][c]["reject_fa"] for c in dp.CONDITIONS}
        out["tau_star"][str(h)] = {"tau": tau_star(halves[h], budget), "budget": budget}
    exact_S = {c: 0 for c in dp.CONDITIONS}; exact_B = dict(exact_S); n = dict(exact_S)
    s2 = True
    for h in (0, 1):
        o = 1 - h
        tau = {"default": out["tau_star"][str(h)]["tau"]}
        row = {}
        for c in dp.CONDITIONS:
            m = dp.metrics(halves[o][c], "d2", tau, None)
            row[c] = {"S_exact": m["exact"], "S_fa": m["reject_fa"], "B_exact": B[o][c]["exact"], "B_fa": B[o][c]["reject_fa"]}
            s2 &= m["reject_fa"] <= B[o][c]["reject_fa"]
            exact_S[c] += m["exact"]; exact_B[c] += B[o][c]["exact"]; n[c] += m["n_targets"]
        out["held_out"][f"fit_half{h}_apply_half{o}"] = row
    gap = abs(out["tau_star"]["0"]["tau"] - out["tau_star"]["1"]["tau"])
    s1 = gap <= S1_MAX_TAU_GAP
    s3 = all(Fraction(exact_S[c] - exact_B[c], n[c]) >= -dp.R5_ACC_BUDGET for c in dp.CONDITIONS)
    out.update(tau_gap=gap, S1=s1, S2=s2, S3=s3, exact_pooled={c: [exact_S[c], exact_B[c], n[c]] for c in dp.CONDITIONS},
               verdict="STABLE" if (s1 and s2 and s3) else "UNSTABLE")
    (D / "metadata" / "dense_tau_stability.json").write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
