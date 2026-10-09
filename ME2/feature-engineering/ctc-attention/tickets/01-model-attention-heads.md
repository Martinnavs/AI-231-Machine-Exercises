# 01 — Model: attention-pooled intent and slot heads

**Depends on:** worktree prerequisites in `00-RECAP.md`. **Test pass:** required. **Docs:** `docs/VCM-CONTRACT.md`
model-seam note.

## Do

1. `vcm/quartznet.py`: add `heads: bool = False` (and `head_dim: int = 128`, `pooling: "attention"|"mean" =
   "attention"`) to `QuartzNetConfig`. Defaults keep every existing checkpoint and preset identical (the config
   dict saved in old checkpoints has no new keys; `QuartzNetConfig(**old_config)` must still work).
2. When `heads` is true, `QuartzNetCTC` builds, on the encoder output `(B, 256, T')` (the tensor feeding the
   existing CTC linear), an `AttentivePool` (score = `Linear(256, head_dim) -> tanh -> Linear(head_dim, 1)`,
   softmax over valid frames only, padding masked by `output_lengths`) and heads: `intent` = `Linear(256, 21)`,
   one `slot_<INTENT>_<SLOT>` = `Linear(256, 3)` per slotted intent. Class and slot value lists come from a small
   `vcm/semantic_labels.py` derived from `OPTIONB_GRAMMAR.all_phrases()` (D-3); expose
   `INTENT_CLASSES: tuple[str, ...]`, `SLOTS: dict[intent, tuple[slot_name, tuple[values]]]`.
3. API (backward compatible):
   `forward(features) -> logits` unchanged; new `forward_heads(features, input_lengths=None) -> HeadsOutput`
   (`ctc_logits (B,T',29)`, `intent_logits (B,21)`, `slot_logits: dict[str, (B,3)]`, `attention (B,T')`).
   Pooling must be masked when `input_lengths` is given (use `output_lengths`); with `None`, all frames valid.
4. `model.py`: a new preset `quartznet5x3-heads` (same width/kernels, `heads=True`); `build_model`,
   `model_type_for_config` and checkpoint load must work unchanged (same `model_type: "quartznet"`).
5. Param budget: report the added parameters; they must be < 2% of the encoder (about 60k) and the INT8 export
   of the CTC graph is unaffected (heads are not exported in this ticket).

## Acceptance criteria

- [ ] Existing `tests/test_vcm_quartznet.py`, `test_vcm_model.py`, `test_vcm_train.py`, export tests pass
      unmodified; old checkpoints (with and without `model_type`) load and give bit-identical CTC logits.
- [ ] New tests: forward shapes; `forward` output identical with heads on/off for the same encoder weights;
      attention weights sum to 1 over valid frames and are exactly 0 on padded frames; a padded batch gives the
      same pooled vector as the unpadded clip; label tables match `OPTIONB_GRAMMAR` (19 intents + 2, 6 slots x 3
      values) and fail loudly if the grammar changes.
- [ ] `QuartzNetConfig(heads=True)` round-trips through `torch.save(..., weights_only=True)`.

## Non-goals
Training code, loss, ONNX export of heads, any streaming change.

## Execution Log

### 2026-10-02 — implemented (Claude, worktree `me2-ctc-attention`, detached at `d931906`; nothing committed)
- Added `vcm/semantic_labels.py`; `HeadsOutput`, `AttentivePool`, `heads`/`head_dim`/`pooling` config fields,
  `_encode`/`forward_heads` in `vcm/quartznet.py`; preset `quartznet5x3-heads` in `vcm/model.py`;
  `tests/test_vcm_quartznet_heads.py` (17 tests); model-seam note in `docs/VCM-CONTRACT.md`.
- Added params: 43,048 (pool 33,025 + intent 5,397 + 6 slot heads x 771) = 4.7% of the 911,189 model. The ticket's
  "< 2% (about 60k)" is self-inconsistent; 43k is under the 60k figure, not under 2%. Needs a human call.
- Whole-model "padded batch == unpadded clip" is not true for a conv encoder (receptive field sees padding near the
  clip end); tested instead at pool level (exact) and that garbage past `output_lengths` cannot reach the logits.
- Pytest: 247 passed (quartznet, model, train, export, e2e_benchmark, streaming_backends, optionb_grammar, heads).
  Old checkpoints (`quartznet5x3-s2-...-135m`, `optiond-seed1-...-135m`): CTC logits bit-identical, old code vs new.
- Added a streaming-shaped test (batch 1, `lengths=None`, windows 0.3-6 s): finite outputs, attention sums to 1, >=15 output frames at 0.3 s. Streaming never pads, so only training batches use the mask.
