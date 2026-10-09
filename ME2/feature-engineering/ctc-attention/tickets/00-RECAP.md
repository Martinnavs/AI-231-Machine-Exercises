# 00 — RECAP: Attention-pooled intent/slot heads on the QuartzNet CTC encoder

**Repo:** `AI-222-Machine-Exercises/ME2`. **Branch:** `optionb-ctc-attention` (from `optionb-grammar-v2` @ `492f73a`).
**Slug:** `ctc-attention`. **Executor:** opencode in its own worktree. **Security-sensitive:** no (checkpoint
loading already goes through the closed `ARCHITECTURES` table; this adds a config field, not a new loader path).
**Test pass:** required per ticket. **Docs:** each ticket names its doc.
**Proposed (not yet approved) — the human confirms scope before ticket 01 starts.**

## Why

Idea under assessment: a QuartzNet-based *semantic* model that returns `{intent, slots}` directly instead of
transcribing text and parsing it with a grammar. Assessment in short: possible and cheap to try, but as a
*replacement* it loses things this project depends on (slots, structural rejection of non-commands, rejection
of unfinished phrases, and the CTC blank signal that the new streaming mode uses to detect end of speech).
So the feature is a **hybrid**: keep the CTC head and add an attention-pooled intent head and slot heads on
the same encoder, trained jointly. CTC keeps endpointing and rejection; the classifier is evaluated as a
cross-check and as a way to cut the beam-search cost (the beam search is about 84 ms of the 135 ms p50 per
window with QuartzNet INT8; `docs/archive/QUARTZNET-STUDENT.md`).

The feature also answers one open question cheaply: *is a classifier competitive on this data at all?* (run B,
ticket 02).

## Confirmed decisions (do not re-litigate)

- **D-1 Hybrid, not replacement.** The CTC head, decoder, grammar, noisy gate and streaming policies stay.
- **D-2 Baseline** = the existing stride-2 QuartzNet, seed 0: `out/vcm/quartznet5x3-s2-fil50-ambient-rir-135m`.
  Same data (`out/conversions/v2/optionb-v3-vcmx-fil50-ambient/manifest.csv`), recipe (all `vcm.train` defaults,
  `--p-rir 0.7`, `--max-minutes 135`, `--seed 0`) and gates as `docs/archive/QUARTZNET-STUDENT.md`.
- **D-3 Label space (verified 2026-10-02 from `OPTIONB_GRAMMAR` and the manifest):**
  - intent head: 21 classes = the 19 Option B intents + `unknown` (manifest `label` of babble rows) + `silence`.
    Manifest `label` values already are these strings, so no relabeling.
  - slot heads: 6 slotted intents, one slot each, 3 values each (plus "not applicable"):
    `BRIGHTNESS.PERCENT` {100 percent, 20 percent, 60 percent}; `COLOR.COLOR` {blue, green, red};
    `TEMPERATURE.DEGREES` {18, 22, 26 degrees}; `TIMER.DURATION` {1 minute, 10 seconds, 30 seconds};
    `ALARM.ALARM_TIME` {6 AM, 8 AM, 9 PM}; `CREATE_REMINDER.TASK` {drink water, exercise, study}.
    Derive these from `OPTIONB_GRAMMAR.all_phrases()` (tuples of `(text, intent, slots)`), never hardcode.
- **D-4 Pooling:** attentive pooling over the encoder's output frames (`(B, 256, T')` before the linear head),
  padding-masked with `model.output_lengths`. Average pooling is the ablation control.

## Facts (verified; don't redo)

- `vcm/quartznet.py::QuartzNetCTC.forward` returns CTC logits `(B, T', 29)` only. `vcm/model.py` has
  `ARCHITECTURES` (closed table), `PRESETS`, `build_model_from_config`, and checkpoints carry `model_type`;
  an absent key means MatchboxNet. Old checkpoints must keep loading unchanged.
- `vcm/train.py` computes CTC loss at two sites (training loop and `run_eval`) with `model.output_lengths`.
  `vcm/dataset.py::collate_fn` returns `features, input_lengths, target_ids, target_len`; `VCMExample` carries
  no intent/slot label, but every manifest row has `label` and `bucket`.
- ONNX export names inputs/outputs `features` -> `logits` (`export_onnx.py`; strided models use output axis
  `time_out`). `streaming/backends.py::OnnxBackend` returns log-posteriors only.
- `vcm/evaluate.py` (`RowResult`, `decode_split`), `vcm/noisy_eval.py` (fixed-seed reverb+noise perturbation,
  scored at the clean-val threshold) and `vcm/streaming/replay_eval.py` (wake word + command sessions) are the
  scoring tools to extend.
- Baseline numbers to beat or match (QuartzNet stride 2 seed 0, margin 4.0 in streaming):
  whole-clip test 3,494 targets: clean exact 3468, noisy exact 3416, noisy babble FA 1/255, noisy silence FA 2/324
  (threshold -0.1); streaming replay test 821 sessions: 741 correct first trigger, 21 wrong-intent triggers,
  latency from end of speech p50/p95 0.36/0.51 s, 7.1 decodes per session.
- Prefix hazard: "time" is a character prefix of every "timer ..." phrase; "pause"/"stop" prefix longer
  accepted phrases. A classifier must be judged on these (per-intent confusion, not just totals).

## Tickets

| # | Ticket | Depends on |
|---|---|---|
| 01 | Model: attention-pooled intent and slot heads (off by default, back compatible) | none |
| 02 | Training: labels, joint loss, run A (CTC + heads) and run B (heads only) | 01 |
| 03 | Evaluation: classifier metrics on the whole-clip noisy gate, agreement with CTC, decision | 02 |
| 04 | Use: classifier-guided grammar decode (pre-selector) and its streaming replay | 03, only if its gate passes |

Order 01 → 02 → 03 → 04. 02 consumes 01's real model and 03 consumes 02's real checkpoints (producer/consumer:
do not parallelize).

## Prerequisites (do first)

1. `git worktree add ../me2-ctc-attention optionb-ctc-attention` (the branch exists on origin).
2. `out/`, `raw_datasets/` are untracked: symlink them into the worktree from the main checkout. `.scratch/` is
   gitignored (these tickets live in `feature-engineering/ctc-attention/tickets/`, which is tracked).
3. Resources: at most 2 concurrent jobs of ours; **GPU 6 is never ours**; run `nvidia-smi` first. A 135-minute
   training run is about 2.25 h wall clock when the node is quiet.
4. No commit/push/PR without the human.

## Pre-registered decision rules (set now, before any result)

- **Gate A (ticket 03, joint model must not hurt CTC):** run A's CTC path vs the baseline on the same
  whole-clip test: clean exact >= 3468 - 17 (0.5 pp) and noisy exact >= 3416 - 17, noisy babble FA <= 3/255.
- **Classifier competitiveness (reported, not gating):** classifier-only intent and exact (intent + slots)
  accuracy at a max-softmax threshold chosen on clean val under the same babble/silence false-accept budget as
  the CTC path at -0.1; clean and noisy; per-intent confusion for TIME/TIMER, PAUSE/STOP, LIGHT_ON/LIGHT_OFF.
- **Gate B (ticket 04 may start):** Gate A holds AND classifier top-3 intent contains the true intent on
  >= 99.5% of correct-by-CTC clean test targets (so restricting the decode to the top 3 cannot drop much).

## Non-goals (all tickets)

Replacing the CTC head or grammar; retraining the baseline; changing `common/features.py`, `vcm/decoder.py`
semantics, the wake word, or the streaming policies' accept rules; a different encoder; stride 4; promotion to
production; Pi measurement; committing or pushing without the human.

## Execution Log
(append dated entries; never rewrite earlier ones)
