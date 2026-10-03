# Current model: the hybrid (wide CTC + XL classifier heads)

Start here if you are picking up the VCM work. This page says what the current model is, how to run it in three commands, what it
scores and how far to trust that, how it was trained and evaluated, where the code lives, and what is still open. It links to the
long, dated records instead of repeating them.

- **Status (2026-10-03, branch `optionb-ctc-attention`):** best model so far for streaming tests, trained only on public data.
  It is **not promoted**: `make app-pipeline*` and the README's "production checkpoint" still name the older model
  (`option-d-fil50-ambient-rir-135m`, and `quartznet5x3-s2-fil50-ambient-rir-135m` for the live targets). Promotion is a separate decision.
- **Reproducibility:** the task is to be reproducible from public data. Read [`REPRODUCE-HYBRID.md`](REPRODUCE-HYBRID.md) before
  claiming a result: the training manifest rebuilds from the published datasets except 4 persona clips (28,854 of 28,858 rows); retraining on the rebuilt manifest was not re-run.
- **Archived checkpoints:** the other models' binaries were untracked on 2026-10-03; see [`ARCHIVED-CHECKPOINTS.md`](ARCHIVED-CHECKPOINTS.md).
- **Model card:** [`out/vcm/hybrid-ctcwide-clsxl/MODEL-CARD.md`](../out/vcm/hybrid-ctcwide-clsxl/MODEL-CARD.md).

## What it is

Two QuartzNet-5x3 CTC encoders (stride 2, 40-mel features, 10 ms hop) run on the same window of audio:

| | `ctc-wide` | `cls-xl` |
|---|---|---|
| preset | `quartznet5x3-wide-heads` (channels 416, epilogue 512, head 192) | `quartznet5x3-xl-heads` (channels 648, epilogue 800, head 256) |
| parameters | 4.18 M | 9.99 M |
| used for | the grammar-constrained CTC answer | the attention-pooled intent head (21-way) and six slot heads |
| INT8 ONNX | `ctc-wide/export/vcm_model.int8.onnx`, 4.16 MB (features -> CTC logits) | `cls-xl/export/vcm_heads.int8.onnx`, 10.10 MB (features -> intent + slot logits, no CTC layer) |
| PyTorch checkpoint | `ctc-wide/checkpoints/checkpoint.pt` (17 MB) | `cls-xl/checkpoints/checkpoint.pt` (39 MB) |

Decision rule (**policy A, "fallback"**, `vcm/hybrid.py`):

1. Decode the wide model's log-posteriors with the grammar-constrained CTC prefix beam search (`OPTIONB_GRAMMAR`: 19 intents, 93
   wordings; beam 50; threshold -0.1; incomplete-prefix margin 4.0, the repo's streaming operating point). If it accepts, that is
   the answer, slots included.
2. Otherwise, if the XL intent head's top class is a command (not `unknown`/`silence`) with max-softmax >= **0.8787**, the head's
   intent and slot are the answer. 0.8787 was fit on validation to the XL model's own CTC false-accept rate, never tuned on test.
3. Otherwise reject.

Why a hybrid: the CTC path almost never answers wrongly (3 wrong intents in 3,823 clean ai231 commands) but rejects real ones
(187); the classifier recovers most of those rejections (145 of 187) at the price of more false accepts on non-commands. The head
cannot be hosted without the XL encoder (it reads that encoder's features; the heads are 0.24 M of its 9.99 M parameters).
Policy B (`agree`: classifier gates, CTC answers only when intents agree) is implemented and scored but is worse; use A.

## Run it

```bash
# 1. decode wav files (checked-in INT8 ONNX; no GPU, no PyTorch checkpoint needed)
make hybrid-decode HYBRID_WAV="clip1.wav clip2.wav"       # JSON per clip: decision + what each component said + counters

# 2. streaming: wake word, then the command (endpointed policy, classifier fallback on, wake word polled every 0.05 s,
#    wake-word threshold 0.8 and a 0.6 slot-confidence gate on the classifier: both tuned on a validation soak)
make hybrid-stream                                         # live microphone
make hybrid-stream HYBRID_SOURCE=path/to/session.wav       # deterministic file replay

# 3. tests that cover it, and the soak test (wake word + gap + command with reverb and noise; docs/SOAK-TEST.md)
make hybrid-test
make soak-run SOAK_DIR=soak/holdout-wake-gap-v1 SOAK_NAME=run SOAK_ARGS="--backend onnx"
```

The plain commands behind these targets, the replay evaluation and the scoring scripts are in
[`AI231-FIL50.md`](AI231-FIL50.md), "Run the hybrid" and "Hybrid in streaming". Python: `HybridDecoder(ctc_model, cls_model,
feature_extractor, OPTIONB_GRAMMAR, cls_threshold=0.8787).decode(waveform, truth=None)` returns `(decision, trace)` and keeps
counters (`answered_by_ctc/cls/none`, and with a truth `both / only_ctc / only_cls / neither`). `OnnxCtc` / `OnnxHeads` /
`load_hybrid_part` accept `.onnx` paths; `LazyCascade` loads the heads model only on a CTC rejection.

## What it scores

Intent + slot, one seed, one training run per model. **Read the caveats in the next section.**

| set | hybrid | wide CTC alone | XL classifier alone |
|---|---|---|---|
| ai231 test, original rows (3,823), clean | **98.8** | 95.0 | 97.7 |
| same, perturbed (RIR, dataset noise, fixed seed 0) | **94.9** | 87.8 | 92.9 |
| ai231 persona rows, voice-disjoint (2,946) | 99.7 | 98.7 | 99.2 |
| 129 real recordings (old internal test) | **98.4** | 96.1 | 84.5 |
| leak-free internal held-out (638), clean / perturbed | 97.8 / 87.3 | 93.4 / 80.7 | 94.2 / 85.1 |
| ai231 holdout (186 commands, one real Filipino speaker) | 77.4 | 66.7 | 76.9 |
| streaming replay, wake word + command (784 sessions), first trigger correct | **90.1** | 88.9 | |
| soak: holdout behind a wake word, reverb + noise (186 commands, tuned settings), first trigger correct | **81.7** (CPU INT8: 81.2) | 74.7 | |

- The soak test (`docs/SOAK-TEST.md`): 0.38 s median latency after the end of speech, ~70 ms per decoded window on a server core (A100 is no faster: the beam search is CPU), 9 wrong actions for the hybrid vs 2-3 for CTC only; 0 triggers in the ambient gaps; 3-4 of 16 out-of-scope clips trigger.
- ONNX fp32 equals PyTorch on every set; INT8 costs at most one clip per set (`out/vcm/hybrid-ctcwide-clsxl/eval/hybrid-onnx-vs-pytorch.md`).
- False accepts on the 326 ai231 non-command clips: 3.4% clean / 6.1% perturbed for the hybrid, 0.9% / 2.5% for the CTC alone.
  The wake word sits in front in deployment, which lowers this in practice (not measured as a rate).
- Missed commands are the main error: on the holdout 19% of commands are still rejected, and 39 of 186 are wrong for both models.
  That is accent coverage in the training data, not decoding.
- The reference models trained on the internal data (`H-int`, old production) are better on real Filipino speech (holdout 84.9% for
  `H-int`) and are **not reproducible**; they are diagnostics, not targets.

## How far to trust the numbers

- **Leakage.** 2,632 of the 4,895 ai231 test/holdout rows are byte-identical to clips in the internal training manifest, and most
  test speakers appear in it. Anything trained on the internal data (old production, `H-int`) is inflated on the ai231 test.
  The models in this page were trained on public data only, so their ai231 numbers are clean. Use the leak-free sets
  (`scripts/build_internal_heldout.py`, `scripts/build_user_voice_eval.py`, `scripts/filter_internal_overlap.py`) when comparing
  against an internal-data model. Details: `AI231-FIL50.md`, "Leak finding".
- **Seen vs unseen.** Test and holdout are clip-disjoint and speaker-disjoint from training (README, "Seen vs unseen"), but the perturbed gate and the soak use the same seed-0 synthetic rooms as training, and 131 of the 202 soak sessions use noise clips from the training split; whether any persona or synthetic voice is the same person as a real test speaker is unverified.
- **One seed, small real-speaker sets.** The holdout is 186 clips from one speaker; the user-voice sets are 20 raw and 648 converted
  clips. The hybrid's cross-model edge over wide+wide (same-model hybrid) is small except on the holdout and the perturbed ai231 test.
- **Pi 4 not measured.** There is no Raspberry Pi on the development node. Latency numbers are server cores
  (ONNX Runtime, one thread): network 13.6 ms (wide) / 30.2 ms (XL) per 2.5 s window, beam search ~40-90 ms, streaming latency
  after end of speech 0.38 s median / 0.60 s p95 (algorithmic).
- **Pre-registered rules** (A-D in `AI231-FIL50.md`) were written against the old production numbers, which turned out to be inflated;
  every public-data model fails some of them. Report against the public-only baseline and the leak-free sets.

## Data

| Source | Role | Where |
|---|---|---|
| `airimonda/ai231-me2-voice-commands` v2 (Hugging Face, public) | train / val / test / holdout, `synthetic_negatives`, `supplemental_synth`, `variations.csv` | `raw_datasets/ai231-me2-voice-commands-v2/`; converted with `vcm.optionb.import_ai231` to `out/conversions/v2/ai231-v2/` |
| Filipino persona clips (17 cloned voices; `martinnavs/ai231-fil-supplemental-data`, public) | extra train rows; 10 voices train, 2 val, 5 test, capped per wording (50 / 12 / 35) | manifest `out/conversions/v2/ai231-fil50/` (+ `ai231-fil50-supp/` adds the ai231 `supplemental_synth` train-voice clips) |

The training manifest of both models is `ai231-fil50-supp/manifest.csv`: 28,858 rows, 16.5 h; train 18,300 rows (12,868 ai231,
4,535 persona, 897 negatives and noise), val 2,717, test 7,639, holdout 202. Noise and babble for augmentation come only from the
dataset's own `noise_only` and babble clips plus synthetic coloured noise (`--noise-source dataset`); **never use ESC-50**.

## Training recipe

```bash
uv run python -m me2_voicegen.vcm.train --device cuda:0 --manifest out/conversions/v2/ai231-fil50-supp/manifest.csv \
  --preset quartznet5x3-wide-heads \        # or quartznet5x3-xl-heads
  --seed 0 --p-rir 0.7 --p-timestretch 0.25 --p-noise 0.5 --p-babble 0.15 --noise-source dataset \
  --perturbation-plan table --dump-plan --skip-prenoised --noise-random-offset \
  --onecycle-epochs 60 --max-epochs 70 --patience 20 --max-minutes 135 --out-dir out/vcm/<run> \
  --license-note "<terms of the data you used>"
```

- AdamW, lr 1e-4, weight decay 0.01, OneCycle (10% warm-up), batch 16, AMP; loss = CTC 1.0 + intent 0.3 + slot 0.1.
- `--device` defaults to **cpu**: always pass `--device cuda:N`. Never use GPU 6 on the shared node; at most two jobs at once.
- The perturbation plan is a pure function of seed, epoch and a class key (`label | slot | variation | source_dataset`), so real
  and persona clips get the same share of each perturbation. Details: `CTC-ATTENTION.md`, "Grouping key changed".
- The default checkpoint licence text is the old ESC-50 note; pass `--license-note` for ai231-based runs.
- Wide: best epoch 38 (val loss 0.391, 94 min). XL: best epoch 23 (0.336, 43 min); it overfits early, so more capacity did not help the CTC.
- Export: `uv run python -m me2_voicegen.vcm.export_onnx --checkpoint <ckpt> --out-dir <dir> --manifest <manifest>` (CTC output),
  add `--heads-only` for the encoder + intent/slot heads file. INT8 uses static quantization with 32 validation clips.

## Evaluating

```bash
uv run python -m me2_voicegen.vcm.semantic_eval score ...      # per model, per condition, shardable (CTC rows + classifier probabilities)
uv run python scripts/hybrid_score.py score|summarize ...      # hybrid end to end from audio, policies A and B
uv run python scripts/hybrid_eval.py --out report.md           # offline merge of already-scored rows (its policy B is superseded)
uv run python scripts/ai231_fil50_rules.py ...                 # pre-registered rules and tables
uv run python -m me2_voicegen.vcm.streaming.session_replay --split test --vcm-manifest <eval manifest> --out-dir <sessions>
uv run python -m me2_voicegen.vcm.streaming.replay_eval --sessions <sessions> --model <run dir> ... [--cls-model ... --wakeword-poll-s 0.05]
```

Perturbed gate = `vcm.noisy_eval` (RIR + dataset noise at fixed seed 0, no babble). Held-out tables in
`out/vcm/hybrid-ctcwide-clsxl/eval/`. Full commands and every earlier run: `AI231-FIL50.md`, "Commands" and the round sections.

## Where things are

| Path | What |
|---|---|
| `src/me2_voicegen/vcm/hybrid.py` | `hybrid_decode_all`, `HybridDecoder`, `LazyCascade`, `OnnxCtc`, `OnnxHeads`, `classifier_result` |
| `src/me2_voicegen/vcm/streaming/` | `policy.py` (`EndpointedPeriodPolicy` + classifier fallback), `wakeword_gate.py` (`poll_step_s`), `replay_eval.py`, `session_replay.py` |
| `src/me2_voicegen/vcm/{model,quartznet}.py` | presets (`quartznet5x3[-heads\|-wide-heads\|-xl-heads]`), `HeadsOutput` |
| `src/me2_voicegen/vcm/{train,perturbation_plan,dataset}.py`, `common/augment.py` | training, the perturbation table, the zero-energy-noise guard |
| `src/me2_voicegen/vcm/export_onnx.py` | ONNX export, `--heads-only`, static INT8 |
| `src/me2_voicegen/vcm/semantic_eval.py`, `semantic_labels.py` | scoring rows, the 21-way intent classes and slot values |
| `src/me2_voicegen/accent_balance/build_ai231_fil50.py`, `plan_gap_jobs.py` | how the persona-padded manifest was built (needs internal inputs, see `REPRODUCE-HYBRID.md`) |
| `scripts/` | `soak_run.py`, `build_soak_audio.py`, `build_soak_continuous.py`, `hybrid_score.py`, `hybrid_eval.py`, `hybrid_decode_file.py`, `ai231_fil50_rules.py`, `build_ai231_fil50_supp.py`, `build_internal_heldout.py`, `build_user_voice_eval.py`, `filter_internal_overlap.py` |
| `tests/` | `test_soak_run.py`, `test_vcm_hybrid.py`, `test_vcm_streaming_endpointed.py`, `test_vcm_streaming_wakeword_gate.py`, `test_vcm_quartznet_heads.py`, `test_vcm_perturbation_plan.py` |
| `soak/holdout-wake-gap-v1/` | the soak test's truth, results and Pi instructions (the wav is not in git) |
| `out/vcm/hybrid-ctcwide-clsxl/` | the model: checkpoints, ONNX, logs, loss history, model card, eval tables (force-added; `out/` is otherwise git-ignored) |
| `out/wakeword-sesame-ambient-rir-45m/` | the wake-word DS-CNN the streaming gate uses |

## Open items and decisions

- **Decided:** policy A with both models resident; ticket 04 (classifier as a beam-search pre-selector) deferred: it is a speed change
  only, the XL top-3 recall is 99.8% on ai231 but 93.5% on the real-accent holdout, and Pi latency is unknown. It needs an explicit go-ahead.
- **Open:** Pi 4 latency and memory; a one-command reproduction (see `REPRODUCE-HYBRID.md`); second seeds for both models; a stricter
  fallback to cut non-command false accepts (sweep on validation only); more Filipino-accented speakers in training and a larger
  real-speaker holdout; wake-word model retraining (38 of 784 replay sessions never open a period); a DOI and licence for the persona
  data; promoting the hybrid in `make app-pipeline*`.
- **Open (soak):** false wakes from ordinary speech at wake-word threshold 0.8 are unmeasured (the soak gaps hold only noise); decide hybrid vs CTC only given the wrong-action cost (`--cls-slot-threshold` trades them); run the soak on a Raspberry Pi 4 and fill the comparison (`soak/holdout-wake-gap-v1/README.md`).
- **Gotchas:** `--device` default is CPU; the heads are not in the CTC ONNX files (use `vcm_heads.int8.onnx`); the Make targets use wake-word threshold 0.8 and slot gate 0.6, older docs and `make app-pipeline*` use 0.9; the streaming `--device cuda` path needs the wake-word fix in `wakeword_gate.py` (CPU feature extractor); some wake-word clips in the
  replay sessions carry ESC-50 noise (rebuild sessions without them for strict public-only claims); the checkpoint `license` field of
  older runs is the ESC-50 text.

## Reading order

1. This page, then the model card.
2. [`AI231-FIL50.md`](AI231-FIL50.md): the experiment log (rounds 1-4, leak finding, hybrid, streaming, ticket 04 re-evaluation, handoff).
3. [`CTC-ATTENTION.md`](CTC-ATTENTION.md): the heads architecture, joint loss, perturbation table, v2 results on the ai231 data.
4. [`SOAK-TEST.md`](SOAK-TEST.md), [`STREAMING-CONTRACT.md`](STREAMING-CONTRACT.md), [`WAKEWORD-SLIDING.md`](WAKEWORD-SLIDING.md): the soak test, the streaming policy, gate and replay.
5. [`OPTIONB-GRAMMAR-CONTRACT.md`](OPTIONB-GRAMMAR-CONTRACT.md), [`INCOMPLETE-GRAMMAR-REJECTION.md`](INCOMPLETE-GRAMMAR-REJECTION.md): the grammar and the margin gate.
6. [`PROCESS-OVERVIEW.md`](archive/PROCESS-OVERVIEW.md) and the other `PROCESS-*` docs: the older end-to-end narrative (describes the earlier ~1M-parameter model).
