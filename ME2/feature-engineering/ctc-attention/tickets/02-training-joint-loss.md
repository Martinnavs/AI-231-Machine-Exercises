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
   - Smoke each for 5 minutes first (as in `docs/QUARTZNET-STUDENT.md`); record seconds per epoch.

## Acceptance criteria

- [ ] Unit tests: label mapping on a small fake manifest (intent ids, slot targets, -100 handling, non-phrase
      rows counted); loss terms present and finite; `--ctc-weight 0` gives zero CTC gradient to the CTC linear.
- [ ] Both runs finish without a failed checkpoint; `train.log` and `loss_history.json` kept; epoch counts and
      validation intent accuracy reported in the Execution Log.
- [ ] `uv run pytest -q` green.

## Non-goals
Class re-weighting, prefix-crop augmentation, changing the data or recipe, a second seed.

## Execution Log
