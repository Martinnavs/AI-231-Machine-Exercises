# 03 — Evaluation: classifier metrics, agreement with CTC, decision

**Depends on:** 02 (both run directories). **Test pass:** required for new code.

## Do

1. Extend `vcm/evaluate.py` and `vcm/noisy_eval.py` additively (no change to existing keys or to output when no
   heads exist): when the checkpoint has heads, also compute per row `intent_probs` (softmax) and slot argmaxes,
   using the same clean/noisy pipeline and fixed seed 0 (`--noisy-eval-seed 0`).
2. Report, test split, clean and noisy: (a) classifier intent accuracy and exact (intent + slots) among target
   clips; (b) false-accept rate on babble and silence at a max-softmax threshold chosen on **clean val** to match
   the CTC path's clean-val false-accept budget at -0.1; the threshold is never re-swept on noisy data;
   (c) per-intent confusion with TIME/TIMER, PAUSE/STOP, LIGHT_ON/LIGHT_OFF called out; (d) agreement between
   classifier top-1 and the CTC decode's intent; (e) top-1/top-3 intent coverage of the true intent.
3. Gate A and the competitiveness table from `00-RECAP.md`, for run A and run B, next to the baseline numbers.
   Write `docs/CTC-ATTENTION.md`: results, the pre-registered rules and whether each passed, and a plain
   statement of what was and was not shown (one seed).
4. Commands go in the doc; outputs under `out/vcm/quartznet-ctc-compare/` or each run dir's `noisy_eval/`.

## Acceptance criteria

- [ ] Tests for the new metric code on a fake manifest (known predictions give known counts); existing
      `test_vcm_evaluate.py` / `test_vcm_noisy_eval.py` unchanged and green.
- [ ] `docs/CTC-ATTENTION.md` with the tables, the gate outcomes, and a recommendation. **Stop here and report
      if Gate A fails; do not start ticket 04.**

## Non-goals
Threshold tuning on noisy data, changing the CTC decode, serving.

## Execution Log
