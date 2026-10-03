# 02 — Training: labels, joint loss, runs A and B

**Depends on:** 01 merged in the worktree. **Test pass:** required for code; the two runs are measurements.

## Do

1. **Labels.** In `vcm/dataset.py` (or a thin wrapper used only by `vcm.train`), attach per-row targets without
   changing existing outputs: `intent_id` from the manifest `label` via `INTENT_CLASSES`; slot targets from the
   resolved transcript by exact phrase lookup in `OPTIONB_GRAMMAR.all_phrases()` (normalize with
   `vcm.text.normalize_text`): the slot value index for the row's slotted intent, ignore (-100) for other slot
   heads and for babble/silence rows. A target row whose transcript is not an exact grammar phrase gets intent
   from `label` and slot -100, and is **counted and logged**, not dropped. `collate_fn` default output must be
   unchanged for existing callers (add the keys only when asked).
2. **Loss.** `train.py` gains `--heads-loss-weights intent=0.3,slot=0.1` and `--ctc-weight 1.0`; total =
   `ctc_weight * CTC + w_i * CE(intent) + w_s * mean over slot heads of CE(slot, ignore_index=-100)`. With
   `--preset quartznet5x3-heads` only. Every other preset's loop is byte-for-byte the same path (regression test
   with a spy on the loss call as in `test_train_uses_output_lengths_for_ctc`).
3. **Logging/checkpoint.** `loss_history.json` gains per-epoch `intent_acc_val`, `slot_acc_val` (labelled rows
   only), the three loss terms, and counts of non-grammar-phrase rows. Checkpoint config carries `heads: true`.
4. **Runs** (GPU, same recipe as D-2; seed 0; `--max-minutes 135`; at most 2 at once, never GPU 6):
   - **Run A:** `--preset quartznet5x3-heads` (CTC + heads), out dir
     `out/vcm/quartznet5x3-heads-fil50-ambient-rir-135m`.
   - **Run B:** same, `--ctc-weight 0` (classifier-only; the CTC head gets no gradient), out dir
     `out/vcm/quartznet5x3-heads-only-fil50-ambient-rir-135m`. Answers "is a classifier competitive?".
   - Smoke each for 5 minutes first (as in `docs/archive/QUARTZNET-STUDENT.md`); record seconds per epoch.

## Acceptance criteria

- [ ] Unit tests: label mapping on a small fake manifest (intent ids, slot targets, -100 handling, non-phrase
      rows counted); loss terms present and finite; `--ctc-weight 0` gives zero CTC gradient to the CTC linear.
- [ ] Both runs finish without a failed checkpoint; `train.log` and `loss_history.json` kept; epoch counts and
      validation intent accuracy reported in the Execution Log.
- [ ] `uv run pytest -q` green.

## Non-goals
Class re-weighting, prefix-crop augmentation, changing the data or recipe, a second seed.

## Amendments (human, 2026-10-02) -- these supersede the Do list where they differ

- **Data:** train/val on `out/conversions/v2/ai231-me2-voice-commands/manifest.csv` (not the fil50-ambient manifest);
  `close` paraphrases are trained on; val = the converted 10% of train speakers; gates use `exact` rows only.
  Run dirs are named for the data: `out/vcm/quartznet5x3-heads-ai231-rir-135m` (A) and
  `out/vcm/quartznet5x3-heads-only-ai231-rir-135m` (B).
- **Time-stretch:** both runs add `--p-timestretch 0.25` (factor U[0.85, 1.15], resample-based, applied to the waveform
  before log-mel, train split only, drawn before RIR/noise). `apply_timestretch` and the `Augmenter` hook are ported from
  the uncommitted work in the `me2-iteration3` worktree (patch of `common/augment.py` + its tests; `--p-timestretch`
  flag in `vcm/train.py`; default stays 0.0 so every other run is unchanged).
- **Slot targets for paraphrases:** the slot target comes from an exact grammar-phrase match; if the transcript is a
  paraphrase, from the manifest's canonical `slot_value` column; otherwise ignored (-100) and counted
  (`semantic_label_counts` in `loss_history.json`). The ticket text said -100 for all non-phrase rows; this uses the
  paraphrase rows' slot labels instead of discarding them.
- **Checkpoint selection:** CTC val loss as for every run, except classifier-only (`--ctc-weight 0`), whose CTC head is
  untrained, so it is selected on `0.3*intent CE + 0.1*slot CE` (val).
- **Known recipe gap:** the ai231 manifest has no `background_noise` rows, so the online additive-noise augmentation has an
  empty pool and is skipped (RIR, SpecAugment and time-stretch still apply). Re-measure the baseline on this data with
  the same recipe before ticket 03.

## Execution Log

### 2026-10-02 -- code done, runs pending (Claude, worktree `me2-ctc-attention`, nothing committed)
- Code: `vcm/semantic_labels.py` (`row_targets`, `SLOT_INTENTS`), `vcm/dataset.py` (`semantic_labels=`, `LabelledVCMExample`,
  `collate_fn(with_labels=)`), `vcm/train.py` (`--ctc-weight`, `--heads-loss-weights`, `--p-timestretch`, joint loss,
  `HeadsMeter`, per-epoch heads metrics in `loss_history.json`, `heads: true` in the checkpoint config).
- Tests: `tests/test_vcm_semantic_training.py` (13) + ported time-stretch tests.
- Smoke, 5 min each on GPU 1 (A) / GPU 2 (B), ai231 manifest, no time-stretch, 10 epochs, ~33 s/epoch, no NaN:
  A final val `intent_acc` 0.782, `slot_acc` 0.812, `exact_acc` 0.707 (CTC val loss 1.45); B `intent_acc` 0.799,
  `slot_acc` 0.873, `exact_acc` 0.752. Smoke dirs (`out/vcm/_smoke-heads-*`) are throwaway.
- **Bugs found in the ported `apply_timestretch` (fixed here, tests added in `tests/test_vcm_augment.py`):** (1) its edge-pad
  branch used `expand(-1, n)`, which fails on the 1-D waveforms the dataset passes (crash in the first stretch smoke);
  (2) it resampled to `round(16000*factor)` Hz, which is almost always coprime with 16000 and builds a 16000-phase filter:
  5.3 s per 2 s clip at factor 1.1507 (2 ms at 1.15, the only values the original tests used). It now resamples by the
  best fraction with denominator <= 200 (ratio error <= 2.5e-5) and still trims/pads to exactly `round(N*factor)`:
  worst call 10 ms over 800 random calls. The same fix should go back to the `me2-iteration3` worktree.
- Smoke with `--p-timestretch 0.25` (3 min, GPU 1): ~33 s/epoch (same as without), 0 CTC-infeasible items in train and val.
- Full suite on the final code: 6010 passed, 33 deselected.
- **Runs started** (main training, 135 min, seed 0, `--p-rir 0.7 --p-timestretch 0.25`, ai231 manifest):
  A `out/vcm/quartznet5x3-heads-ai231-rir-135m` on GPU 1; B `out/vcm/quartznet5x3-heads-only-ai231-rir-135m`
  (`--ctc-weight 0`) on GPU 2. Epoch counts and final val accuracies to be appended when they finish.
- **Run B finished (2026-10-02 11:43)** `out/vcm/quartznet5x3-heads-only-ai231-rir-135m`: early stop (patience 10, no heads-val-loss
  improvement) at epoch 41 after 1,742 s (29 min, ~42 s/epoch; not the 135 min budget); best epoch 31; no NaN, 0 CTC-infeasible
  train items; `train.log` and `metadata/loss_history.json` kept. Val at the best epoch (all labelled val rows incl. babble and
  paraphrases): `intent_acc` 0.887, `slot_acc` 0.978, `exact_acc` 0.892; `intent_ce` 0.467, `slot_ce` 0.047. The CTC val loss
  (15.0) is meaningless here, the CTC head gets no gradient. Train label sources: phrase 4691, manifest_slot_value 294,
  no_slot_target 320, nonslotted 3814, nontarget 191.
- **Run A finished** `out/vcm/quartznet5x3-heads-ai231-rir-135m` (CTC + heads, `--p-timestretch 0.25`): 76 epochs in 2994 s
  (deadline_hit=False), selected on CTC val loss, best epoch 66 (CTC val loss 0.4739); no NaN, 0 CTC-infeasible train items;
  `train.log` + `loss_history.json` kept. Val at best epoch: `intent_acc` 0.915, `slot_acc` 0.991, `exact_acc` 0.923.
  Both runs done: ticket 02 acceptance met (pytest 6010 passed). Baseline re-measure on ai231 is still needed before ticket 03.

### 2026-10-02 -- reproducible training noise (human request): code done, runs in progress
- `vcm/noise_pool.py` (deterministic synthetic coloured noise), `--noise-source dataset`, `--p-babble` (12-25 dB SNR babble from the
  train split's own babble rows), and `--perturbation-plan table` / `--dump-plan` (`vcm/perturbation_plan.py`: per-(clip, epoch)
  recipes, a pure function of seed/epoch/class, stratified by intent+slot class). Defaults unchanged. Tests:
  `test_vcm_noise_pool.py` (9), `test_vcm_perturbation_plan.py` (11); 165 passed across the touched suites.
- Retrains with `--noise-source dataset --p-babble 0.15` in the legacy online mode (noise pool 251, babble pool 180):
  `quartznet5x3-baseline-ai231-noise-rir-135m`, `quartznet5x3-heads-ai231-noise-rir-135m` (started 13:3x).
- ESC-50 pinning for evaluation deliberately left internal (human decision).
