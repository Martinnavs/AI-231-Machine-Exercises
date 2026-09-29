"""Dense-phonetic-scoring pilot: offline rescoring of dumped VCM logits.

EXPERIMENT-ONLY -- never imported by production code (decoder, pipeline,
evaluate, streaming). See `.scratch/dense-phonetic-scoring-pilot/PLAN.md`.

Stages (CLI: `python -m me2_voicegen.vcm.dense_pilot {score,report}`):

  score   decode every dumped row once with the UNCHANGED
          `decoder.decode_utterance` (threshold=-inf, margin off) and attach
          two dense scores for the winning phrase:
            d1 = mean log-prob over non-blank frames of the Viterbi
                 (forced) alignment of the winner (literal §3 formula)
            d2 = raw beam mass / len(winner text)  (cheap in-loop proxy)
  report  pure arithmetic over the score records: calibrate thresholds on
          clean val, select a variant on val, evaluate test once, apply the
          frozen GO/NO-GO rule, write JSON + Markdown.

The winner (intent) never changes: T is constant within an utterance, so
argmax total/T == argmax raw mass. Dense scoring only moves accept/reject.
"""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
from fractions import Fraction
from pathlib import Path

import numpy as np

BASELINE_THRESHOLD = -0.1
PRODUCTION_MARGIN = 4.0
MARGINS = (None, PRODUCTION_MARGIN)
CONDITIONS = ("clean", "noisy_s0")
BEAM_WIDTH = 50
TARGET = "target_commands"
REJECT = ("babble", "silence")

# ---------------------------------------------------------------------------
# Frozen GO/NO-GO constants. Committed BEFORE the single T3 run; pinned by
# tests at their boundaries. Do not change after test numbers are computed.
# ---------------------------------------------------------------------------
R1_FA_RATIO = Fraction(1, 2)            # S noisy reject FA <= 0.5 * B
R3_ALPHA = 0.05                         # one-sided sign test on disagreements
R5_ACC_BUDGET = Fraction(5, 1000)       # overall exact-accuracy drop <= 0.5pp
R6_ACC_BUDGET = Fraction(3, 100)        # per-intent exact-accuracy drop <= 3pp
R6_MIN_TARGETS = 50                     # intents with >= 50 test targets
MIN_BASELINE_NOISY_FA = 4               # below this the margin gate absorbed it
TIME_STOP = ("TIME", "STOP")

# Documented baselines (out/vcm/option-d-fil50-ambient-rir-135m/noisy_eval/
# metadata/eval_report.json), clean-val threshold -0.1, margin None, beam 50.
# None = not published in the report.
EXPECTED_PARITY = {
    ("val", "clean"): {"accepted": 3266, "reject_fa_total": 13},
    ("val", "noisy_s0"): {"accepted": 3085, "exact": 3062, "babble_fa": 30, "silence_fa": 4},
    ("test", "clean"): {"accepted": 3450, "exact": 3448, "babble_fa": 3, "silence_fa": 1},
    ("test", "noisy_s0"): {"accepted": 3359, "exact": 3345, "babble_fa": 10, "silence_fa": 6},
}
PARITY_NEAR_THRESHOLD = 1e-4

# Variant tie-break order (D3).
TIEBREAK = ("d2-global", "d1-global", "d2-per_intent", "d1-per_intent", "baseline-per_intent")
CONFIGS = ("baseline-global",) + TIEBREAK


class AlignmentFailed(RuntimeError):
    pass


# ---------------------------------------------------------------------------
# score stage
# ---------------------------------------------------------------------------


def viterbi_nonblank(logp: np.ndarray, text: str) -> tuple[float, float, int]:
    """(d1, viterbi_total, n_nonblank) for `text` against `logp` (T, 29) via
    torchaudio's CTC forced alignment. Raises AlignmentFailed if infeasible."""
    import torch
    import torchaudio.functional as taf

    from me2_voicegen.vcm.alphabet import BLANK_ID, encode

    targets = encode(text)
    if not targets:
        raise AlignmentFailed("empty target")
    lp = torch.from_numpy(np.ascontiguousarray(logp, dtype=np.float32)).unsqueeze(0)
    try:
        ali, scores = taf.forced_align(lp, torch.tensor([targets], dtype=torch.int32), blank=BLANK_ID)
    except RuntimeError as exc:
        raise AlignmentFailed(str(exc)) from exc
    ali, scores = ali[0].numpy(), scores[0].numpy().astype(np.float64)
    nb = ali != BLANK_ID
    n_nb = int(nb.sum())
    if n_nb == 0:
        raise AlignmentFailed("no non-blank frames")
    return float(scores[nb].mean()), float(scores.sum()), n_nb


def score_one(logp: np.ndarray) -> dict:
    """Decode one row with the unchanged decoder and attach dense scores."""
    from me2_voicegen.vcm.decoder import NEG_INF, decode_utterance
    from me2_voicegen.vcm.optionb.grammar import OPTIONB_GRAMMAR

    d = decode_utterance(logp, OPTIONB_GRAMMAR, NEG_INF, beam_width=BEAM_WIDTH)
    rec = {
        "intent": d.intent,
        "grammar_text": d.grammar_text,
        "confidence": None if d.intent is None else d.confidence,
        "command_raw_score": d.command_raw_score,
        "incomplete_gap": d.incomplete_gap,
        "T": int(logp.shape[0]),
        "d1": None, "d2": None, "viterbi_total": None, "n_nonblank": None,
        "mass_minus_viterbi": None, "align_failed": False,
    }
    if d.intent is not None and d.command_raw_score is not None and d.grammar_text:
        rec["d2"] = d.command_raw_score / len(d.grammar_text)
        try:
            d1, vt, n_nb = viterbi_nonblank(logp, d.grammar_text)
            rec.update(d1=d1, viterbi_total=vt, n_nonblank=n_nb,
                       mass_minus_viterbi=d.command_raw_score - vt)
        except AlignmentFailed:
            rec["align_failed"] = True
    return rec


def _score_chunk(chunk: list[np.ndarray]) -> list[dict]:
    import torch

    torch.set_num_threads(1)
    return [score_one(lp) for lp in chunk]


def score_rows(logps: list[np.ndarray], rows: list[dict], workers: int = 32) -> list[dict]:
    n = max(1, min(workers, len(logps)))
    chunks = [logps[k::n] for k in range(n)]
    if n == 1:
        parts = [_score_chunk(chunks[0])]
    else:
        with ProcessPoolExecutor(max_workers=n) as pool:
            parts = list(pool.map(_score_chunk, chunks))
    recs: list[dict | None] = [None] * len(logps)
    for k, part in enumerate(parts):
        for j, rec in enumerate(part):
            recs[k + j * n] = rec
    out = []
    for rec, row in zip(recs, rows):
        out.append({"index": int(row["index"]), "bucket": row["bucket"], "label": row["label"], **rec})
    return out


# ---------------------------------------------------------------------------
# report stage: acceptance, calibration
# ---------------------------------------------------------------------------


def row_score(rec: dict, scorer: str) -> float | None:
    if rec["intent"] is None:
        return None
    return {"baseline": rec["confidence"], "d1": rec["d1"], "d2": rec["d2"]}[scorer]


def margin_ok(rec: dict, margin: float | None) -> bool:
    """Post-hoc equivalent of decode_utterance(required_command_margin=m)."""
    if margin is None:
        return True
    gap = rec["incomplete_gap"]
    return gap is None or gap >= margin


def accepts(rec: dict, scorer: str, tau: dict, margin: float | None) -> bool:
    """`tau` = {"default": float, <intent>: float}."""
    s = row_score(rec, scorer)
    if s is None or not margin_ok(rec, margin):
        return False
    return s >= tau.get(rec["intent"], tau["default"])


def iso_threshold(scores: list[float], n_accept: int) -> float:
    """Highest tau accepting >= n_accept of `scores` (ties accept together)."""
    if n_accept <= 0:
        return float("inf")
    ranked = sorted(scores, reverse=True)
    if n_accept > len(ranked):
        return float("-inf")
    return ranked[n_accept - 1]


def calibrate(val_clean: list[dict], scorer: str, per_intent: bool, margin: float | None) -> dict:
    """Iso-accept on clean val (D2): accept as many val targets as the
    baseline (-0.1, same margin setting) does -- globally, or per predicted
    intent (falls back to the global tau when the baseline accepts none of
    that intent). No false-accept counts are used."""
    base_tau = {"default": BASELINE_THRESHOLD}
    targets = [r for r in val_clean if r["bucket"] == TARGET]
    base_acc = [r for r in targets if accepts(r, "baseline", base_tau, margin)]
    eligible = [r for r in targets if row_score(r, scorer) is not None and margin_ok(r, margin)]
    tau = {"default": iso_threshold([row_score(r, scorer) for r in eligible], len(base_acc))}
    if per_intent:
        base_by = Counter(r["intent"] for r in base_acc)
        for intent, n in base_by.items():
            scores = [row_score(r, scorer) for r in eligible if r["intent"] == intent]
            if scores and n > 0:
                tau[intent] = iso_threshold(scores, n)
    return tau


# ---------------------------------------------------------------------------
# metrics
# ---------------------------------------------------------------------------


def metrics(recs: list[dict], scorer: str, tau: dict, margin: float | None) -> dict:
    tgt = [r for r in recs if r["bucket"] == TARGET]
    acc = [r for r in tgt if accepts(r, scorer, tau, margin)]
    exact = [r for r in acc if r["intent"] == r["label"]]
    per_intent_n, per_intent_exact = Counter(), Counter()
    for r in tgt:
        per_intent_n[r["label"]] += 1
    for r in exact:
        per_intent_exact[r["label"]] += 1
    fa = {b: [r for r in recs if r["bucket"] == b and accepts(r, scorer, tau, margin)] for b in REJECT}
    fa_by_intent = Counter(r["intent"] for b in REJECT for r in fa[b])
    return {
        "n_targets": len(tgt),
        "accepted": len(acc),
        "exact": len(exact),
        "accept_rate": len(acc) / len(tgt) if tgt else None,
        "exact_accuracy": len(exact) / len(tgt) if tgt else None,
        "n_babble": sum(1 for r in recs if r["bucket"] == "babble"),
        "n_silence": sum(1 for r in recs if r["bucket"] == "silence"),
        "babble_fa": len(fa["babble"]),
        "silence_fa": len(fa["silence"]),
        "reject_fa": len(fa["babble"]) + len(fa["silence"]),
        "per_intent_n": dict(per_intent_n),
        "per_intent_exact": dict(per_intent_exact),
        "fa_by_predicted_intent": dict(fa_by_intent),
        "time_stop_fa": sum(fa_by_intent.get(i, 0) for i in TIME_STOP),
    }


def fa_positions(recs: list[dict], scorer: str, tau: dict, margin: float | None) -> set[int]:
    return {
        r["index"] for r in recs
        if r["bucket"] in REJECT and accepts(r, scorer, tau, margin)
    }


# ---------------------------------------------------------------------------
# frozen decision rule
# ---------------------------------------------------------------------------


def sign_test_p(removed: int, added: int) -> float:
    """Exact one-sided binomial p = P(X >= removed | n = removed + added, 0.5)."""
    n = removed + added
    if n == 0:
        return 1.0
    return sum(math.comb(n, k) for k in range(removed, n + 1)) / 2**n


def decide(inp: dict, parity_ok: bool) -> dict:
    """Pure GO/NO-GO function. `inp` (S = selected variant, B = baseline,
    all on TEST at the production margin):
      noisy_fa_B/S, noisy_babble_fa_B/S, removed, added (noisy reject rows
      only S / only B accepts), clean_fa_B/S, clean_exact_B/S, clean_n,
      noisy_exact_B/S, noisy_n, per_intent: {label: {clean_n, clean_exact_B,
      clean_exact_S, noisy_exact_B, noisy_exact_S}}, noisy_ts_fa_B/S."""
    if not parity_ok:
        return {"verdict": "INVALID", "reason": "parity_failed", "criteria": {}}
    if inp["noisy_fa_B"] < MIN_BASELINE_NOISY_FA:
        return {"verdict": "NO-GO", "reason": "baseline_far_already_absorbed_by_margin",
                "criteria": {}}
    p = sign_test_p(inp["removed"], inp["added"])
    r6 = {}
    for label, d in inp["per_intent"].items():
        if d["clean_n"] < R6_MIN_TARGETS:
            continue
        n = d["clean_n"]
        r6[label] = (
            Fraction(d["clean_exact_S"] - d["clean_exact_B"], n) >= -R6_ACC_BUDGET
            and Fraction(d["noisy_exact_S"] - d["noisy_exact_B"], n) >= -R6_ACC_BUDGET
        )
    crit = {
        "R1": {"value": [inp["noisy_fa_S"], inp["noisy_fa_B"]],
               "pass": Fraction(inp["noisy_fa_S"]) <= R1_FA_RATIO * inp["noisy_fa_B"]},
        "R2": {"value": [inp["noisy_babble_fa_S"], inp["noisy_babble_fa_B"]],
               "pass": inp["noisy_babble_fa_S"] < inp["noisy_babble_fa_B"]},
        "R3": {"value": p, "pass": p <= R3_ALPHA},
        "R4": {"value": [inp["clean_fa_S"], inp["clean_fa_B"]],
               "pass": inp["clean_fa_S"] <= inp["clean_fa_B"]},
        "R5": {"value": {"clean": [inp["clean_exact_S"], inp["clean_exact_B"], inp["clean_n"]],
                         "noisy": [inp["noisy_exact_S"], inp["noisy_exact_B"], inp["noisy_n"]]},
               "pass": Fraction(inp["clean_exact_S"] - inp["clean_exact_B"], inp["clean_n"]) >= -R5_ACC_BUDGET
               and Fraction(inp["noisy_exact_S"] - inp["noisy_exact_B"], inp["noisy_n"]) >= -R5_ACC_BUDGET},
        "R6": {"value": r6, "pass": all(r6.values())},
        "R7": {"value": [inp["noisy_ts_fa_S"], inp["noisy_ts_fa_B"]],
               "pass": inp["noisy_ts_fa_S"] <= inp["noisy_ts_fa_B"]},
    }
    go = all(c["pass"] for c in crit.values())
    return {"verdict": "GO" if go else "NO-GO",
            "reason": "all_criteria_pass" if go else "criteria_failed:" + ",".join(
                k for k, c in crit.items() if not c["pass"]),
            "criteria": crit}


# ---------------------------------------------------------------------------
# parity
# ---------------------------------------------------------------------------


def parity_block(records: dict) -> dict:
    """Baseline at -0.1, margin None vs. the documented counts. A +/-1
    difference is tolerated only if some row of that (split, condition) sits
    within PARITY_NEAR_THRESHOLD of the threshold (GPU vs CPU numerics)."""
    tau = {"default": BASELINE_THRESHOLD}
    out, ok = {}, True
    for key, exp in EXPECTED_PARITY.items():
        recs = records[key]
        m = metrics(recs, "baseline", tau, None)
        obs = {"accepted": m["accepted"], "exact": m["exact"], "babble_fa": m["babble_fa"],
               "silence_fa": m["silence_fa"], "reject_fa_total": m["reject_fa"]}
        near = sum(1 for r in recs if r["confidence"] is not None
                   and abs(r["confidence"] - BASELINE_THRESHOLD) < PARITY_NEAR_THRESHOLD)
        cells = {}
        for name, e in exp.items():
            diff = obs[name] - e
            tolerated = abs(diff) == 1 and near >= 1
            cells[name] = {"expected": e, "observed": obs[name],
                           "pass": diff == 0 or tolerated, "tolerated_numerics": tolerated and diff != 0}
            ok &= cells[name]["pass"]
        out[f"{key[0]}/{key[1]}"] = cells
    return {"pass": ok, "cells": out}


# ---------------------------------------------------------------------------
# report
# ---------------------------------------------------------------------------


def config_parts(cfg: str) -> tuple[str, bool]:
    scorer, mode = cfg.split("-")
    return scorer, mode == "per_intent"


def build_report(records: dict, mv4_scores: dict | None = None) -> dict:
    """`records[(split, condition)]` -> list of score records."""
    parity = parity_block(records)
    val_clean = records[("val", "clean")]
    taus = {}
    for margin in MARGINS:
        for cfg in CONFIGS:
            scorer, per = config_parts(cfg)
            taus[(cfg, margin)] = (
                {"default": BASELINE_THRESHOLD} if cfg == "baseline-global"
                else calibrate(val_clean, scorer, per, margin)
            )

    def m_for(cfg, margin, split, cond):
        scorer, _ = config_parts(cfg)
        return metrics(records[(split, cond)], scorer, taus[(cfg, margin)], margin)

    grid = {}
    for cfg in CONFIGS:
        for margin in MARGINS:
            for split in ("val", "test"):
                for cond in CONDITIONS:
                    grid[f"{cfg}|{margin}|{split}|{cond}"] = m_for(cfg, margin, split, cond)

    # D3: select on val reject-probe FAs (clean + noisy) at the production margin.
    val_fa = {
        cfg: sum(grid[f"{cfg}|{PRODUCTION_MARGIN}|val|{c}"]["reject_fa"] for c in CONDITIONS)
        for cfg in TIEBREAK
    }
    selected = min(TIEBREAK, key=lambda c: (val_fa[c], TIEBREAK.index(c)))

    m = PRODUCTION_MARGIN
    B, S = "baseline-global", selected
    sc_S, _ = config_parts(S)
    noisy_B = records[("test", "noisy_s0")]
    fa_B = fa_positions(noisy_B, "baseline", taus[(B, m)], m)
    fa_S = fa_positions(noisy_B, sc_S, taus[(S, m)], m)
    g = lambda cfg, split, cond: grid[f"{cfg}|{m}|{split}|{cond}"]
    per_intent = {}
    cB, cS, nB, nS = (g(B, "test", "clean"), g(S, "test", "clean"),
                      g(B, "test", "noisy_s0"), g(S, "test", "noisy_s0"))
    for label, n in cB["per_intent_n"].items():
        per_intent[label] = {
            "clean_n": n,
            "clean_exact_B": cB["per_intent_exact"].get(label, 0),
            "clean_exact_S": cS["per_intent_exact"].get(label, 0),
            "noisy_exact_B": nB["per_intent_exact"].get(label, 0),
            "noisy_exact_S": nS["per_intent_exact"].get(label, 0),
        }
    inp = {
        "noisy_fa_B": nB["reject_fa"], "noisy_fa_S": nS["reject_fa"],
        "noisy_babble_fa_B": nB["babble_fa"], "noisy_babble_fa_S": nS["babble_fa"],
        "removed": len(fa_B - fa_S), "added": len(fa_S - fa_B),
        "clean_fa_B": cB["reject_fa"], "clean_fa_S": cS["reject_fa"],
        "clean_exact_B": cB["exact"], "clean_exact_S": cS["exact"], "clean_n": cB["n_targets"],
        "noisy_exact_B": nB["exact"], "noisy_exact_S": nS["exact"], "noisy_n": nB["n_targets"],
        "per_intent": per_intent,
        "noisy_ts_fa_B": nB["time_stop_fa"], "noisy_ts_fa_S": nS["time_stop_fa"],
    }
    verdict = decide(inp, parity["pass"])
    if selected == "baseline-per_intent":
        verdict["note"] = "control (baseline + per-intent thresholds) won on val: dense scoring not credited"

    # margin x dense overlap on test (margin None baseline FAs)
    overlap = {}
    for cond in CONDITIONS:
        recs = records[("test", cond)]
        base = fa_positions(recs, "baseline", {"default": BASELINE_THRESHOLD}, None)
        by_margin = fa_positions(recs, "baseline", {"default": BASELINE_THRESHOLD}, m)
        by_dense = fa_positions(recs, sc_S, taus[(S, None)], None)
        removed_m, removed_d = base - by_margin, base - by_dense
        overlap[cond] = {
            "baseline_fa_margin_none": len(base),
            "removed_by_margin_only": len(removed_m - removed_d),
            "removed_by_dense_only": len(removed_d - removed_m),
            "removed_by_both": len(removed_m & removed_d),
            "removed_by_neither": len(base - removed_m - removed_d),
        }

    mv4 = {}
    vals = [r["mass_minus_viterbi"] for r in records[("val", "clean")]
            if r["bucket"] == TARGET and r["mass_minus_viterbi"] is not None]
    if vals:
        mv4 = {"n": len(vals), "median": float(np.median(vals)), "p95": float(np.percentile(vals, 95)),
               "max": float(max(vals))}
    failures = {f"{k[0]}/{k[1]}": sum(1 for r in recs if r["align_failed"]) for k, recs in records.items()}
    return {
        "parity": parity,
        "selected_config": selected,
        "val_reject_fa_at_production_margin": val_fa,
        "thresholds": {f"{c}|{mg}": {str(a): b for a, b in t.items()} for (c, mg), t in taus.items()},
        "decision_inputs": inp,
        "verdict": verdict,
        "overlap_margin_x_dense_test": overlap,
        "mv2_alignment_failures": failures,
        "mv4_mass_minus_viterbi_val_targets": mv4,
        "grid": grid,
    }


def render_markdown(rep: dict) -> str:
    v = rep["verdict"]
    lines = [
        "# Dense phonetic scoring pilot", "",
        f"**Verdict: {v['verdict']}** ({v['reason']})", "",
        f"Selected config (val): `{rep['selected_config']}`; parity: "
        f"{'PASS' if rep['parity']['pass'] else 'FAIL'}", "",
        "## Criteria (test, margin 4.0, selected vs baseline)", "",
        "| criterion | pass | value |", "|---|---|---|",
    ]
    for k, c in v["criteria"].items():
        lines.append(f"| {k} | {c['pass']} | `{json.dumps(c['value'])}` |")
    lines += ["", "## Headline (test)", "",
              "| config | margin | cond | exact | babble FA | silence FA | TIME+STOP FA |",
              "|---|---|---|---|---|---|---|"]
    for cfg in ("baseline-global", rep["selected_config"]):
        for margin in ("None", "4.0"):
            for cond in CONDITIONS:
                m = rep["grid"][f"{cfg}|{margin}|test|{cond}"]
                lines.append(
                    f"| {cfg} | {margin} | {cond} | {m['exact']}/{m['n_targets']} "
                    f"({m['exact_accuracy']:.4f}) | {m['babble_fa']}/{m['n_babble']} | "
                    f"{m['silence_fa']}/{m['n_silence']} | {m['time_stop_fa']} |")
    lines += ["", "## Margin x dense overlap (test, baseline FAs at margin None)", "",
              "```json", json.dumps(rep["overlap_margin_x_dense_test"], indent=2), "```", "",
              f"MV2 alignment failures: `{rep['mv2_alignment_failures']}`",
              f"MV4 mass-minus-viterbi (val targets): `{rep['mv4_mass_minus_viterbi_val_targets']}`", ""]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Follow-up (post-hoc hypothesis): can dense scoring REPLACE the margin gate?
# Pre-registered 2026-09-30 before any follow-up numbers were computed. NOTE the
# test split was already seen once for d2-global at margin None (main pilot),
# so test here is a non-independent confirmation; val is the selection split.
# Candidate (margin None, iso-accept thresholds from clean val) vs production
# baseline = baseline-global at margin 4.0.
#   F1 noisy reject FA: S <= B      F2 clean reject FA: S <= B   (val AND test)
#   F3 clean and noisy exact accuracy: S >= B - 0.5pp            (val AND test)
#   F4 per intent (>=50 targets), clean and noisy exact: S >= B - 3pp (val AND test)
# Selection among the 4 dense configs: lowest val clean+noisy reject FA, TIEBREAK order.
# ---------------------------------------------------------------------------
FOLLOWUP_CANDIDATES = ("d2-global", "d1-global", "d2-per_intent", "d1-per_intent")


def followup_report(records: dict) -> dict:
    base_tau = {"default": BASELINE_THRESHOLD}
    val_clean = records[("val", "clean")]
    cand_tau = {c: calibrate(val_clean, config_parts(c)[0], config_parts(c)[1], None)
                for c in FOLLOWUP_CANDIDATES}

    def m_B(split, cond):
        return metrics(records[(split, cond)], "baseline", base_tau, PRODUCTION_MARGIN)

    def m_S(c, split, cond):
        return metrics(records[(split, cond)], config_parts(c)[0], cand_tau[c], None)

    val_fa = {c: sum(m_S(c, "val", k)["reject_fa"] for k in CONDITIONS) for c in FOLLOWUP_CANDIDATES}
    sel = min(FOLLOWUP_CANDIDATES, key=lambda c: (val_fa[c], TIEBREAK.index(c)))
    out = {"selected": sel, "val_reject_fa": val_fa, "splits": {}}
    ok_all = True
    for split in ("val", "test"):
        cB = {k: m_B(split, k) for k in CONDITIONS}
        cS = {k: m_S(sel, split, k) for k in CONDITIONS}
        f1 = cS["noisy_s0"]["reject_fa"] <= cB["noisy_s0"]["reject_fa"]
        f2 = cS["clean"]["reject_fa"] <= cB["clean"]["reject_fa"]
        f3 = all(Fraction(cS[k]["exact"] - cB[k]["exact"], cB[k]["n_targets"]) >= -R5_ACC_BUDGET
                 for k in CONDITIONS)
        f4 = {}
        for label, n in cB["clean"]["per_intent_n"].items():
            if n < R6_MIN_TARGETS:
                continue
            f4[label] = all(
                Fraction(cS[k]["per_intent_exact"].get(label, 0) - cB[k]["per_intent_exact"].get(label, 0), n)
                >= -R6_ACC_BUDGET for k in CONDITIONS)
        res = {"F1": f1, "F2": f2, "F3": f3, "F4": all(f4.values()), "F4_failing": [k for k, v in f4.items() if not v],
               "B": {k: [cB[k]["exact"], cB[k]["reject_fa"]] for k in CONDITIONS},
               "S": {k: [cS[k]["exact"], cS[k]["reject_fa"]] for k in CONDITIONS}}
        ok_all &= f1 and f2 and f3 and all(f4.values())
        out["splits"][split] = res
    out["verdict"] = "FOLLOWUP-GO" if ok_all else "FOLLOWUP-NO-GO"
    return out


# ---------------------------------------------------------------------------
# Hybrid pilot: D2, margin OFF, global iso-accept tau_g with RELAX-ONLY
# per-intent overrides tau_i = min(tau_g, tau_i_iso). One pre-committed
# candidate H (no selection). Val is evaluated OUT-OF-FOLD: rows are split
# into two speaker-disjoint halves by crc32(group_id), each half is scored with
# thresholds fit on the other half, and the two are pooled. Test uses
# thresholds fit on full clean val and is a NON-independent confirmation
# (test was already seen for d2-global). Pre-registered rule vs B (baseline
# -0.1 + margin 4.0), on OOF val AND test:
#   G1 noisy reject FA H<=B   G2 clean reject FA H<=B
#   G3 clean & noisy exact H >= B - 0.5pp
#   G4 per intent (>=50 targets) clean & noisy exact H >= B - 3pp
#   G5 total reject FA (clean+noisy) of H <= control d2-global + 2
# GO iff G1-G5 hold on both. INCONCLUSIVE iff OOF val passes all, test fails
# only G4 on <=2 intents. Else NO-GO.
# ---------------------------------------------------------------------------
G5_SLACK = 2
G4_INCONCLUSIVE_MAX_INTENTS = 2


def relax_only_tau(records: list[dict]) -> dict:
    g = calibrate(records, "d2", False, None)["default"]
    pi = calibrate(records, "d2", True, None)
    tau = {"default": g}
    for intent, t in pi.items():
        if intent != "default":
            tau[intent] = min(g, t)
    return tau


def global_tau(records: list[dict]) -> dict:
    return calibrate(records, "d2", False, None)


def half_of(group_id: str) -> int:
    import zlib

    return zlib.crc32(group_id.encode("utf-8")) & 1


def add_metrics(a: dict, b: dict) -> dict:
    out = {}
    for k, v in a.items():
        if isinstance(v, dict):
            c = Counter(v)
            c.update(b[k])
            out[k] = dict(c)
        elif k in ("accept_rate", "exact_accuracy"):
            out[k] = None
        else:
            out[k] = v + b[k]
    out["accept_rate"] = out["accepted"] / out["n_targets"] if out["n_targets"] else None
    out["exact_accuracy"] = out["exact"] / out["n_targets"] if out["n_targets"] else None
    return out


def gates(cB: dict, cS: dict, cC: dict) -> dict:
    """cX = {condition: metrics} for B, hybrid S, control C."""
    g1 = cS["noisy_s0"]["reject_fa"] <= cB["noisy_s0"]["reject_fa"]
    g2 = cS["clean"]["reject_fa"] <= cB["clean"]["reject_fa"]
    g3 = all(Fraction(cS[k]["exact"] - cB[k]["exact"], cB[k]["n_targets"]) >= -R5_ACC_BUDGET
             for k in CONDITIONS)
    failing = []
    for label, n in cB["clean"]["per_intent_n"].items():
        if n < R6_MIN_TARGETS:
            continue
        if not all(Fraction(cS[k]["per_intent_exact"].get(label, 0) - cB[k]["per_intent_exact"].get(label, 0), n)
                   >= -R6_ACC_BUDGET for k in CONDITIONS):
            failing.append(label)
    tot = lambda c: sum(c[k]["reject_fa"] for k in CONDITIONS)
    g5 = tot(cS) <= tot(cC) + G5_SLACK
    return {"G1": g1, "G2": g2, "G3": g3, "G4": not failing, "G4_failing": failing, "G5": g5,
            "B": {k: [cB[k]["exact"], cB[k]["reject_fa"]] for k in CONDITIONS},
            "H": {k: [cS[k]["exact"], cS[k]["reject_fa"]] for k in CONDITIONS},
            "control_d2_global": {k: [cC[k]["exact"], cC[k]["reject_fa"]] for k in CONDITIONS}}


def _all_pass(g: dict) -> bool:
    return all(g[k] for k in ("G1", "G2", "G3", "G4", "G5"))


def hybrid_verdict(val_g: dict, test_g: dict) -> str:
    if _all_pass(val_g) and _all_pass(test_g):
        return "GO"
    only_g4 = all(test_g[k] for k in ("G1", "G2", "G3", "G5")) and not test_g["G4"]
    if _all_pass(val_g) and only_g4 and len(test_g["G4_failing"]) <= G4_INCONCLUSIVE_MAX_INTENTS:
        return "INCONCLUSIVE"
    return "NO-GO"


def hybrid_report(records: dict, group_ids: dict) -> dict:
    """`group_ids`: {dataset index: group_id} for val rows."""
    base = {"default": BASELINE_THRESHOLD}
    val_clean = records[("val", "clean")]
    in_half = lambda recs, h: [r for r in recs if half_of(group_ids[r["index"]]) == h]

    def oof(scorer_tau_fn, cond):
        total = None
        for h in (0, 1):
            tau = scorer_tau_fn(in_half(val_clean, 1 - h))
            m = metrics(in_half(records[("val", cond)], h), "d2", tau, None)
            total = m if total is None else add_metrics(total, m)
        return total

    cB_val = {k: metrics(records[("val", k)], "baseline", base, PRODUCTION_MARGIN) for k in CONDITIONS}
    cH_val = {k: oof(relax_only_tau, k) for k in CONDITIONS}
    cC_val = {k: oof(global_tau, k) for k in CONDITIONS}
    tH, tC = relax_only_tau(val_clean), global_tau(val_clean)
    cB_t = {k: metrics(records[("test", k)], "baseline", base, PRODUCTION_MARGIN) for k in CONDITIONS}
    cH_t = {k: metrics(records[("test", k)], "d2", tH, None) for k in CONDITIONS}
    cC_t = {k: metrics(records[("test", k)], "d2", tC, None) for k in CONDITIONS}
    val_g, test_g = gates(cB_val, cH_val, cC_val), gates(cB_t, cH_t, cC_t)
    return {"verdict": hybrid_verdict(val_g, test_g), "val_oof": val_g,
            "test_non_independent": test_g,
            "hybrid_thresholds_full_val": {str(k): v for k, v in tH.items()},
            "val_half_sizes": [len(in_half(val_clean, 0)), len(in_half(val_clean, 1))]}


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _load_records(rec_dir: Path) -> dict:
    out = {}
    for split in ("val", "test"):
        for cond in CONDITIONS:
            with (rec_dir / f"{split}_{cond}.jsonl").open(encoding="utf-8") as f:
                out[(split, cond)] = [json.loads(line) for line in f]
    return out


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = p.add_subparsers(dest="stage", required=True)
    s = sub.add_parser("score")
    s.add_argument("--dump-dir", type=Path, required=True)
    s.add_argument("--workers", type=int, default=32)
    s.add_argument("--conditions", default=",".join(CONDITIONS))
    s.add_argument("--splits", default="val,test")
    r = sub.add_parser("report")
    r.add_argument("--dump-dir", type=Path, required=True)
    f = sub.add_parser("followup")
    f.add_argument("--dump-dir", type=Path, required=True)
    h = sub.add_parser("hybrid")
    h.add_argument("--dump-dir", type=Path, required=True)
    args = p.parse_args(argv)

    rec_dir = args.dump_dir / "records"
    if args.stage == "score":
        import sys

        sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "scripts"))
        from dense_pilot_dump import load_dump

        rec_dir.mkdir(parents=True, exist_ok=True)
        for split in args.splits.split(","):
            for cond in args.conditions.split(","):
                logps, rows = load_dump(args.dump_dir, split, cond)
                recs = score_rows(logps, rows, args.workers)
                with (rec_dir / f"{split}_{cond}.jsonl").open("w", encoding="utf-8") as f:
                    for rec in recs:
                        f.write(json.dumps(rec) + "\n")
                print(f"{split}/{cond}: scored {len(recs)}", flush=True)
    elif args.stage == "hybrid":
        with (args.dump_dir / "logits" / "val_clean.rows.csv").open(newline="", encoding="utf-8") as fh:
            import csv as _csv

            gids = {int(r["index"]): r["group_id"] for r in _csv.DictReader(fh)}
        rep = hybrid_report(_load_records(rec_dir), gids)
        (args.dump_dir / "metadata").mkdir(parents=True, exist_ok=True)
        (args.dump_dir / "metadata" / "dense_hybrid_report.json").write_text(json.dumps(rep, indent=2))
        print(json.dumps(rep, indent=1))
    elif args.stage == "followup":
        rep = followup_report(_load_records(rec_dir))
        (args.dump_dir / "metadata").mkdir(parents=True, exist_ok=True)
        (args.dump_dir / "metadata" / "dense_followup_margin_off.json").write_text(json.dumps(rep, indent=2))
        print(json.dumps(rep, indent=1))
    else:
        rep = build_report(_load_records(rec_dir))
        meta = args.dump_dir / "metadata"
        meta.mkdir(parents=True, exist_ok=True)
        (meta / "dense_pilot_report.json").write_text(json.dumps(rep, indent=2, default=str))
        (meta / "dense_pilot_report.md").write_text(render_markdown(rep))
        print(f"verdict={rep['verdict']['verdict']} reason={rep['verdict']['reason']} "
              f"selected={rep['selected_config']} parity={rep['parity']['pass']}")


if __name__ == "__main__":
    main()
