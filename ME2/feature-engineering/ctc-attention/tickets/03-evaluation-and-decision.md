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

### 2026-10-02 -- note received from the human (`ME2/.scratch/ctc-attention-baseline-note.md`)
- Same margin and threshold for baseline and run A: **margin 4.0, threshold -0.1, beam 50**, whole-clip, ai231 test `exact` rows
  (3,601 clips); out-of-scope false accepts counted on the 47 babble rows. Margins are not retuned on this data.
- Reference numbers in the note (the OLD production checkpoint `quartznet5x3-s2-fil50-ambient-rir-135m`, trained on the old
  data, scored on the ai231 test): margin 20: 94.9/94.8%; **margin 4.0: intent 97.1% / intent+slot 97.0%**; gate off 97.3/97.2%
  (intent / intent+slot); OOS FA 0/47 at every margin; holdout (167 exact) 86.2/91.0/94.6%. The old 3468/3416 figures are void.
- That is a *cross-dataset* reference. The like-for-like Gate A baseline is the plain `quartznet5x3` trained on ai231 with the same
  recipe as run A (`out/vcm/quartznet5x3-baseline-ai231-rir-135m`, training started 2026-10-02). Report both.
- Script: `ME2/.scratch/ai231_eval.py <split> <margin> [checkpoint]` (3rd arg added, defaults to the old checkpoint; ~25 min for
  test on CPU).

### 2026-10-02 -- scored and written up (Claude; nothing committed)
- New `vcm/semantic_eval.py` (build-manifest / score / report) instead of editing `vcm/evaluate.py` and `vcm/noisy_eval.py`:
  the existing files are untouched, so their tests are unchanged; the new module imports their perturbation and decode code.
  The noisy gate needs a `background_noise` pool the ai231 manifest lacks, so the eval manifest adds the old manifest's ESC-50
  val/test rows. 9 tests in `tests/test_vcm_semantic_eval.py`.
- Headline = unperturbed test (3,601 `exact` + 47 babble); perturbed (seed 0) reported separately. Scorer reproduces the note's
  `old` numbers exactly (97.11 / 97.03 / 0 of 47).
- Unperturbed intent+slot: old 97.0, **base 95.0 (3422)**, A CTC 95.6 (3441), A classifier 97.7 (98.0 intent), B classifier 94.9.
  **Gate A pass** (clean 3441 >= 3403, perturbed 2930 >= 2758, perturbed babble FA 0 <= 1). **Gate B pass** (3443/3443).
  Classifier babble FA 8/47 vs CTC 0/47. Perturbed: ai231-trained CTC 77.5-81.6% vs old 95.8% (no ESC-50 noise in ai231 training).
- Bugs found and fixed: case-sensitive slot comparison in the scorer and in `row_targets` (A and B trained without slot labels on
  320 COLOR/CREATE_REMINDER paraphrase rows; not retrained). See `docs/CTC-ATTENTION.md`.
- Outputs: `out/vcm/quartznet-ctc-compare/ctc-attention/{report.md,report.json}`. Ticket 04 not started.
- **Transfer test on the old (internal) test set** (human request): ai231-trained base/A/B scored on `optionb-v3-vcmx-fil50` test with the
  operating points fixed from ai231 (no tuning). Intent+slot: old (in-distribution) 97.4%; base 89.5%; A CTC 89.8%; A classifier 94.9%;
  B 89.9%. Real recordings (129): base 89.9, A CTC 92.2, A classifier 94.6 (+/-5 pp). Weakest source: Filipino-accented synthetic personas
  (82.5-82.7% CTC). Classifier babble FA 46/255 vs CTC 0/255. Perturbed: 70-73% CTC, 84% classifier vs old 95.4%. Details: `docs/CTC-ATTENTION.md`;
  outputs `out/vcm/quartznet-ctc-compare/ctc-attention-oldtest/`.
