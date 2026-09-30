# MLOps Projects — dataset iterations, yields, and decisions

Per `docs/20260925_suggestions.md` §1: "All dataset iterations and yields must be
tracked verbatim in the MLOps Projects file." Every entry below records raw numbers
as measured (no rounding beyond what the source artifact shows). Update the matching
iteration's **Results** section when a run completes; do not edit historical entries —
append corrections with a date.

Conventions:
- **Yield** = QA-passed clips / generated clips, per `(model, prompt_source)`.
- **QA gate config** = the exact gate used to produce a yield (backend, models,
  threshold). A yield is only comparable to another if its gate config is noted.
- Filipino definition for row counting: VCM `optionb` rows with
  `FILIPINO_VCM_SPEAKER_IDS` (s68–s80, s89, s90, s100); wakeword rows with
  `ref_voice` matching `^(tagalog|ilonggo)`; synthetic `fil50_persona` rows are
  Filipino by construction.

## Current production baselines (what every iteration is compared against)

### VCM — `out/vcm/option-d-dataset-v2` (treatment checkpoint, 2026-09-24)
- Model: MatchboxNetCTC preset `optiond`, ~1.01M params, 5 TCS blocks, 128 channels.
- Data: Option B + VCM Dataset B, seed 0, 60-min budget, refreshed grammar.
- Test exact accuracy **0.979** (control, Option-B-only: 0.953).
- Babble false-accept rate **0.012** (control: 0.086).
- Filipino-accent accuracy gap **0.050** (control: 0.162).
- Remaining false accepts concentrate on TIME/STOP (16 of 21); `required_command_margin ≈ 4.5`
  would catch 13 of 16 at zero cost (finding, not shipped).
- Exports: `vcm_model.fp32.onnx` 3.9M, `vcm_model.int8.onnx` 1,018K (1.0 MB).
- Serving: `make vcmx-serve` — threshold -0.1, margin 4.0, beam 50.
- Filipino-voiced share of positives: **15%** (the gap this program targets).

### Wakeword — `out/wakeword/checkpoints/checkpoint.pt` (2026-09-24)
- DS-CNN, epoch 20.
- `_wakeword_`: P 0.999 / R 0.995 / F1 0.997 (support 1097)
- `_unknown_`: P 0.995 / R 0.998 / F1 0.997 (support 1099)
- `_silence_`: P 0.996 / R 1.000 / F1 0.998 (support 270)
- Export: `wakeword_model.int8.onnx` 49,576 bytes (49 KB).
- Data includes ESC-50 (CC-BY-NC-SA-4.0) mixed additively — checkpoint inherits the license.
- Serving: `make vcmx-serve-wakeword` — gate 0.9, single_period 3s.
- Filipino-voiced share of positives: **35%**.

## Decision log

- **2026-09-25 — synthesis source (options from pilot finding):**
  - **Option 1 (References-only) — IMMEDIATE PATH.** 17 `references` voices (Filipino
    people already speaking the English prompt paragraph). Pilot QA pass 87.5% (VCM) /
    95.8% (wakeword). Clean English phonemes; unblocks deployment now.
  - **Option 2 (Voice conversion of sapinsapin) — FAST-FOLLOW.** `convert_voice`
    timbre-transfer onto existing correctly-worded English audio (same mechanism as
    `wakeword/convert_positives.py`). TCS convs read per-frame spectral formants, so
    timbre transfer regularizes the accent while target words are correct by
    construction. Needs its own pilot before the full run.
  - **Option 3 (Blend of references + as-is sapinsapin QA-passes) — DROPPED.**
    The 30–42% sapinsapin pass clips are content-corrupted (cross-lingual
    hallucination), not merely harshly scored; borderline clips would regress the
    1.01M CTC net's exact-intent accuracy.
- **2026-09-25 — ghost clips received:** 142 `ACOUSTIC_GHOST` clips
  (`raw_datasets/ghost-clips.zip` → `raw_datasets/ghost-clips/`), 5 podcasts
  (21/19/59/19/24), 2.5 s @ 16 kHz mono 16-bit, per-clip `intent` labels in
  `manifest.csv`. Intent distribution: STOP 34, TIME 30, PAUSE 21, CALL 16,
  LIGHT_ON 11, VOLUME_DOWN 8, WEATHER 6, PLAY_MUSIC 5, VOLUME_UP 5, MESSAGE 4,
  CREATE_REMINDER 1, LIGHT_OFF 1. 23 empty transcripts, 23 with Tagalog tokens.
  Podcasts are mixed English/Tagalog → QA validation of ghost-adjacent work must
  use the dual EN+TL gate, not English-only.
- **2026-09-26 — ghost 1.25 s outlier dropped** (user decision):
  `podcast_3_t00-00-01_CALL_acoustic_ghost.wav` excluded; 141 ghosts remain
  (all 2.5 s). Non-destructive copy under `.scratch/ghosts-filtered/`.
- **2026-09-26 — references-voice `_unknown_` content source** (user decision):
  voice-convert real Tagalog corpus clips into the 17 references voices via
  `convert_voice` (supersedes the earlier "raw Stella read-aloud" answer).

## Iteration 1 — fil50 references-only (COMPLETE — all pre-committed criteria met)

Goal: 50/50 Filipino/non-Filipino positives per split for both models, using only
`references` voices (Option 1), zero-shot persona TTS.

QA gate config (this iteration): English-only, faster-whisper `small`, threshold
0.80 — for comparability with the pilot's measured yields. Dual EN+TL diagnostic
run over the flagged subset (diagnostic only, not gating).

Planned deltas vs. the built pipeline: `plan_jobs --voice-sources references`
(cross-split references pool with scoped exception in
`vcmx_merge._check_group_id_disjoint` + `collate.assert_group_id_split_disjoint` —
timbre already spans all splits via converted wakeword positives);
`OVERGEN=1.3` (was 1.15, to absorb QA rejects at 17-voice round-robin);
`QA_BACKEND`/`QA_OPTS` env passthrough in `accent_balance_fil50.sh` stage 3.

Pre-committed success criteria (must hold before replacing production):
- Test exact accuracy ≥ 0.969 (baseline 0.979 minus 1 pt tolerance)
- Babble FAR ≤ 0.012
- Filipino gap ≤ 0.050 and Filipino share ≈ 50% ± 1% per split

### Inputs
- Deficits (measured 2026-09-25 against real base manifests):

  | model | split | non-Filipino | Filipino | deficit |
  |---|---|---|---|---|
  | VCM | train | 13,723 | 2,214 | 11,509 |
  | VCM | val | 1,665 | 348 | 1,317 |
  | VCM | test | 1,747 | 180 | 1,567 |
  | wakeword | train | 2,511 | 1,334 | 1,177 |
  | wakeword | val | 712 | 385 | 327 |
  | wakeword | test | 359 | 191 | 168 |

- Voice pool: 17 references voices (English "Please call Stella…" paragraph, 8–12 s
  cuts, freshly transcribed); split assignment N/A (train-only in pool builder,
  cross-split exception applied for this iteration).

### Pilot yield (reference point, measured 2026-09-25 — `out/conversions/v2/fil50-pilot60/`)
- Run: `GPUS="3" PILOT=60`, 120 clips (60 VCM + 60 wakeword), ~9 min wall-clock.
- Throughput: VCM 3.30 s/clip, wakeword 2.37 s/clip.
- QA gate: English-only, threshold 0.80.

  | unit | n | pass rate |
  |---|---|---|
  | `vcm_references` | 24 | 87.5% |
  | `vcm_sapinsapin` | 36 | 41.7% |
  | `wakeword_references` | 24 | 95.8% |
  | `wakeword_sapinsapin` | 36 | 30.6% |

- Root cause of sapinsapin failure (verbatim finding): content-fidelity, not
  accent severity — Tagalog-language prompts pull CosyVoice2 toward the prompt's
  language; transcriptions of English targets contain Tagalog sentences or
  garbage. References (prompt language = target language = English) passes cleanly.

### Respelling pilot — 2026-09-25 (mode evaluation, NOT part of the Phase 1 run)

Question: can the user's Filipino-English phonetics table (2 accent tiers) be
fed to CosyVoice2 as respelled text to add heavier-accent diversity? Mechanism
verified: CosyVoice2's inference has no G2P stage — `frontend.py:84`
`_extract_text_token` feeds raw text straight through the LLM's SentencePiece
tokenizer, so respellings reach the LLM verbatim.

Setup: 10 real grammar phrasings × 3 tiers (plain / mild / heavy) = 30 clips,
3 references voices (ref_tagalog1/5/9, same voice across a phrasing's tiers),
dual-gated (EN small + TL small-tagalog, 0.80) with the ORIGINAL phrase as
expected text. Artifacts: `.scratch/respelling-pilot/` (report.md, audio/,
listen/).

| tier | n | passed | pass rate |
|---|---|---|---|
| plain | 10 | 8 | 80.0% |
| mild | 10 | 6 | 60.0% |
| heavy | 10 | 2 | 20.0% |

Findings (verbatim transcripts from the QA reports):
- Working respellings: `nekst sahng`→"next song" 1.000, `stahrt myoo-sik` /
  `is-tart myoo-sik`→"Start my music" 0.880, `pley myoo-sik`→0.818–0.880,
  `vol-yoom ahp`→"volume HP." 0.889, `pawz`→"Pause." 1.000, `stahp`→"Stop" 1.000.
- Broken respellings (systematic, need dictionary revision): `weh-dhur`→"We
  door" 0.429 / `weh-der`→silence 0.000 (the `dh` token is unreliable),
  `meh-sehj`→"Ma say" 0.462 / `me-sej`→"Misha State, Brands State" 0.387,
  `bol-yoom`→"BoYuMap" 0.500 (`yoom` renders "You", drops the m),
  `kohl`→"Cool." 0.500 / `kol`→"Go!" 0.000, `layts ahn`→"it's uh" 0.533.
- Single-word phrases are fragile regardless of tier (plain also failed:
  `weather`→silence, `pause`→"P.O.S." 0.400) — short clips leave whisper
  little to lock onto.

Decision: respelling is NOT baked into the Phase 1 run (keeps the baseline
comparison single-variable; pre-committed criteria stay meaningful).
Follow-up iteration after Phase 1 validates: revise the ~5 broken tokens,
re-pilot the fixes, then add respelled clips as a top-up (not a replacement)
in the next dataset version. Value of the "accent simulation" vs. real
Filipino-English acoustics remains an open empirical question.

### Launch (2026-09-25 21:48)
- Tech-lead review (dev-flow): **APPROVE, no R1/R2**; 4 non-blocking findings
  (F1 thin QA margin, F2 train budget vs. 70% larger dataset, F3 prefix-only
  `ref_` exemption inert today, F4 cosmetic).
- Launch knobs (user decision, per F1/F2): `OVERGEN=1.2` (VCM train needs
  >=83.3% pass, vs. 86.95% at 1.15), `VCM_MINUTES=90` (old run was
  deadline-bound at epoch 90).
- Command: `GPUS="4 5" VOICE_SOURCES=references QA_BACKEND=faster-whisper-dual
  OVERGEN=1.2 VCM_MINUTES=90 scripts/accent_balance_fil50.sh` via
  `.scratch/accent-balance-fil50/launch-fullrun.sh` in tmux session `fil50`;
  log `.scratch/accent-balance-fil50/fullrun.log`. Golden rule: max 2 GPUs
  (verified per-stage by review: gen 2 shards, QA sequential, train
  VCM+ww on the 2 GPUs, eval sequential/CPU).
- Expected: ~19,278 jobs (17,272 VCM + 2,006 ww); ~12.5-13.5 h total.

### Results

All 7 stages complete 2026-09-26 (generate 21:50→07:59, QA/collate 07:59→10:30,
train 10:30→12:01, eval 12:01→13:12). **All three pre-committed success
criteria met** (thresholds from the "Pre-committed success criteria" list
above): test exact accuracy 0.979 ≥ 0.969 ✓; babble FAR 0.008 ≤ 0.012 ✓;
Filipino accuracy gap 0.008 ≤ 0.050 ✓ (Filipino share exactly 50% ± 0% per
split, per the collate check below).

- Generation: **19,281/19,281 clips, 0 failed** (17,273 VCM + 2,008 ww);
  21:50 → 07:59 (10.1 h, 2 GPUs 4+5).
- QA (dual EN+TL gate, 0.80): **17,140/19,281 = 88.9%** overall — clears the
  ≥83.3% fail-loud margin gate at OVERGEN=1.2.

  | unit | n | pass rate |
  |---|---|---|
  | vcm_references | 17,273 | 88.7% |
  | wakeword_references | 2,008 | 90.5% |

  Per-voice range: 73.9% (ref_tagalog16) – 94.2% (ref_tagalog12).
- Collate (stage 4, done 10:30): 50/50 check passed on all six (model, split)
  cells — VCM train 13,723/27,446, val 1,665/3,330, test 1,747/3,494; ww
  train 2,511/5,022, val 712/1,424, test 359/718 (Filipino = exactly 50% per
  split). Outputs: `out/conversions/v2/optionb-v3-vcmx-fil50/manifest.csv`,
  `out/conversions/v2/wakeword/manifest.fil50.csv`.
- Train (stage 5, 10:30→12:01, 90-min VCM budget on GPU 4 ∥ ww on GPU 5):
  VCM hit the wall-clock deadline at 48 epochs, `best_val_loss=0.3050`
  (`nan_or_inf_seen=True` — recurring non-finite-loss batches, auto-recovered
  per-batch by skipping + restoring pre-step BatchNorm stats, not a training
  failure). Wakeword finished within its 30-min budget at epoch 29.
- Eval (stage 6, 12:01→13:12) — new checkpoint vs. the pre-fil50 baseline,
  same new manifest for both (clean before/after, no confound from dataset
  differences):

  | VCM metric | old checkpoint (`option-d-dataset-v2`) | new (`option-d-fil50`) |
  |---|---|---|
  | test exact accuracy | 0.956 | **0.979** |
  | accept rate | 0.962 | 0.979 |
  | babble FAR | 0.012 | 0.008 |
  | silence FAR | 0.0154 | **0.000** |
  | Filipino vs. foreign accuracy gap | — (not measured on old ckpt this run) | 0.008 (filipino_reference 0.978 n=1747, foreign_reference 0.986 n=1618) |

  | Wakeword metric | old checkpoint (epoch 20) | new (fil50, epoch 29) |
  |---|---|---|
  | `_wakeword_` P/R/F1 | 0.999/0.929/0.963 | 0.999/0.995/0.997 |
  | `_unknown_` P/R/F1 | 0.916/0.998/0.955 | 0.994/0.998/0.996 |
  | `_silence_` P/R/F1 | 0.996/1.000/0.998 | 1.000/1.000/1.000 |
  | **Filipino-accent `_wakeword_` recall** | **0.864** (615/712) | **0.993** (707/712) |
  | non-Filipino `_wakeword_` recall | 0.994 (708/712) | 0.997 (710/712) |
  | **accent recall gap** | **13.0 pts** | **0.4 pts** |

  The wakeword accent-recall gap closing from 13.0 points to 0.4 points is
  the headline result — a much larger effect than the VCM-side numbers, and
  the direct payoff of the 35%→50% Filipino-share rebalance on this model.
  VCM slot accuracy: 1.000 (1053/1053 intent-correct clips, all slots right).
  Full per-intent confusion + val-split threshold sweep in the checkpoint's
  own `metadata/eval_report.md`.
- Export + benchmark (both checkpoints, AMD EPYC estimate — **not** measured
  on the Raspberry Pi 4 target hardware, indicative only): wakeword fp32 ONNX
  124,542 B / int8 49,576 B, p50 latency 0.252 ms / 0.144 ms; VCM fp32 ONNX
  3.85 MB / int8 0.99 MB, p50 latency 3.80 ms / 3.69 ms.

Reproduce/re-evaluate/serve this run's specific checkpoints via raw `uv run`
commands (no new Makefile recipes added for it): `ACCENT-BALANCE-FIL50-COMMANDS.md`.

**Not yet done:** promoting `option-d-fil50`/`wakeword-fil50` to the
"Current production baselines" section above and to the served configs
(`make vcmx-serve` / `make vcmx-serve-wakeword`) is a separate deployment
decision, not automatic on criteria-met — numbers above are recorded, but
the baseline section is left as-is pending that decision.

## Iteration 2 — sapinsapin via voice conversion (PLANNED, fast-follow)

Goal: add 124-speaker Filipino diversity via `convert_voice` (timbre transfer onto
same-split non-Filipino English recordings) after Iteration 1 lands.

QA gate config (this iteration): **dual EN+TL primary** —
`faster-whisper-dual`, `model_a=faster-whisper-small`,
`model_b=faster-whisper-small-tagalog` (`LWobole/whisper-small-tagalog`, CT2
int8_float16), per-segment fusion on higher `avg_logprob` (≥50% overlap),
threshold 0.80. Expected to be stricter than English-only (e.g. "kumputer" vs.
"Computer." ≈ 0.75 < 0.80) → re-size OVERGEN from a dual-gated pilot before the
full run. Dual = 2× QA cost; QA is not the bottleneck (generation is).

Prerequisites: sapinsapin HF license documented before any full run; small
one-pilot (stages 2–3) with dual gate.

### Results
(pending)

## Iteration 3 — ghosts, Tagalog negatives, timestretch (BUILT; pre-measurements)

Worktree `me2-iteration3/ME2`, branch `iteration3-negatives-timestretch`.
Fast suite: 5,637 tests green (2026-09-26).

- **Acoustic ghosts (141, not 142):** 1.25 s outlier dropped (decision log
  above). New `accent_balance/ghosts.py` builds schema-exact 13-column rows
  (`source_dataset=acoustic_ghost`, transcript `""`, per-clip `intent` in the
  sidecar `<manifest>.ghost_intents.csv`); `vcm/text.py` maps to `""`
  (loss-bearing, same mechanism as `background_noise`); `vcm/evaluate.py` adds
  the `acoustic_ghost` reject-probe bucket + per-intent breakdown. Grouping:
  one group per podcast; split assignment train = podcasts 1+2+4 (59),
  val = podcast 5 (24), test = podcast 3 (59) — keeps each recording session
  within one split. Variable-length input confirmed OK (`(B, 40, T)` +
  `input_lengths`); 2.5 s clips need no cropping.
- **Ghost "before" measurement (2026-09-26 10:04):** production checkpoint
  `out/vcm/option-d-dataset-v2`, serving config (beam 50, margin 4.0, optionb
  grammar), CPU. **Test-split ghost FAR = 0.0 (0/58 false accepts)** — the
  current model already rejects every held-out test ghost. Per-intent n:
  STOP 18, PAUSE 8, TIME 8, CALL 6, LIGHT_ON 5, PLAY_MUSIC 3, VOLUME_DOWN 3,
  VOLUME_UP 2, LIGHT_OFF 1, MESSAGE 1, WEATHER 3. Artifacts:
  `.scratch/ghosts-before-eval/metadata/eval_report.{json,md}`. (Bug found +
  fixed here: `render_markdown` crashed on None accept_rate/exact_accuracy for
  probes-only manifests.)
- **Tagalog `_unknown_` negatives:** new `accent_balance/ww_negatives.py`
  (plan/convert/collate). Plan (seed 0): **819 raw corpus rows** (90 speakers,
  cap 10, spontaneous→machine fallback, `read` speech_type excluded) +
  **170 voice-conversion jobs** (17 references voices × 10) = 989 rows; raw
  splits train/val/test = 615/119/85 (split assignment verified to match
  `build_refs.assign_splits` seed 0, so mined negatives share a split with each
  speaker's iteration-2 VC positives). 34 speakers are read-only (no eligible
  clips; read-only skip). Conversion: **170 ok / 0 failed, 528 s** (2026-09-26,
  GPU 7) → `out/ww_negatives/convert/`. Dual-transcription audit (keep iff fused
  transcript non-empty AND no `computer`/`kumputer` token in fused/EN/TL
  transcripts): **169/170 kept** (2026-09-26, 95 s, GPU 7) — 0 wake-word
  contamination, 1 dropped for empty transcript
  (`02323_c01.wav__ref_tagalog13.wav`). Artifacts:
  `me2-iteration3/ME2/.scratch/ww-audit/audit.{csv,summary.md}`.
- **Time-stretch:** `apply_timestretch` U[0.85, 1.15] in `common/augment.py`
  (single resample, sinc_interp_hann, trim/pad back to original length),
  train-split only, applied at load before mel (mels not cached, per-epoch).
  `vcm/train.py --p-timestretch` default **0.0 (off = byte-identical
  pipeline)**.
- **DS-CNN oversampling:** `wakeword/train.py --oversample-accent-unknown`
  (WeightedRandomSampler with replacement; default **1.0 = byte-identical**).

### Results

- **Collate (2026-09-26):** `manifest.fil50-wwneg.csv` (14,864 rows:
  13,876 fil50 base + 819 raw + 169 audited converted; the 1 audited-out
  clip `02323_c01.wav__ref_tagalog13.wav` excluded) — train 10,415 /
  val 2,942 / test 1,507.
- **Pilot (2026-09-26, 15 min, seed 0, `--oversample-accent-unknown 1.5`,
  GPU 5):** 20 epochs, best val_loss 0.0142 (epoch 16), no NaN/Inf.
  Same-test before/after on the NEW val (old fil50 ckpt vs pilot ckpt):
  wakeword recall 0.9951→0.9930, `_unknown_` recall 0.9984→0.9952 (2→6
  misfires), `_silence_` recall 1.0 both, accent fil/non-fil recall
  0.9930/0.9972 → 0.9916/0.9944. Read: pipeline green end-to-end; the
  small regressions are the half-budget under-annealed signature (20 vs
  29 epochs, OneCycle shaped for 150), not a feature-harm signal — and
  the old ckpt already handles the 149 new val negatives well (2
  misfires), so val isn't where the accent-shortcut risk shows. The real
  probe is the test split's 105 new rows. Full-scale recommendation:
  30-min budget (like stage 5), oversample 1.5 (run 1.0/2.0 if a sweep
  is wanted), optional `--onecycle-epochs` = expected epoch count,
  before/after comparison on the new TEST split. Artifacts:
  `me2-iteration3/ME2/out/wakeword-wwneg-pilot/`.
- **Ghost FAR post-training: PENDING** — VCM-with-ghosts was excluded
  from the pilot (pre-measurement already shows 0/58 test ghosts
  false-accepted; a 15-min VCM run on the 39.6k-row manifest is a few
  epochs, uninformative). `ghosts.py` merge is built; one command from a
  ghosts-inclusive manifest when the 60–90 min VCM budget is OK'd.
- **Noisy/reverb VCM eval gate (2026-09-26, separate human task, same
  session):** new `vcm/noisy_eval.py` + `--noisy-eval-seed` on
  `vcm.evaluate` — fixed-seed RIR+noise pass over val/test, scored at the
  clean-val threshold. Production checkpoint (fil50 VCM, seed 0):
  accept 0.979→0.905, exact 0.979→0.902, babble FAR 0.008→0.024, silence
  FAR 0.000→0.015; loss is under-acceptance (REJECTED +34/+34/+32/+23
  on NEXT/CREATE_REMINDER/TEMPERATURE/TIMER), not mis-recognition.
  Artifacts: `me2-iteration3/ME2/out/vcm/noisy-eval-fil50-seed0/`, plan
  `.scratch/vcm-noisy-eval-gate/00-PLAN.md` (worktree).

## Iteration 4 — decoding/deployment levers (PLANNED)

- Dense score in `_greedy_unconstrained` (`vcm/decoder.py:75`); additive-first vs.
  replacement decision pending final model.
- Per-intent thresholds (JSON, calibrated on final model).
- Greedy intrusion gate using existing `out_of_grammar_gap` (`decoder.py:222`);
  `required_command_margin ≈ 4.5` finding (TIME/STOP, 13/16 free catch).
- Gate-first dormant serving mode (decode only when DS-CNN opens); update
  `STREAMING-CONTRACT` §4.
- INT8 re-export; correct stale size figures (VCM INT8 is ~1.0 MB, not 0.26 MB).

### Results

**Dense phonetic scoring offline pilot (2026-09-30) — NO-GO as pre-committed.** Feature
`dense-phonetic-scoring-pilot` (plan/design: `.scratch/dense-phonetic-scoring-pilot/`). Rescored the
production checkpoint `out/vcm/option-d-fil50-ambient-rir-135m` from CPU logits (no retraining, no GPU,
production decoder untouched; code: `scripts/dense_pilot_dump.py`, `src/me2_voicegen/vcm/dense_pilot.py`).
Noisy = fixed-seed-0 `noisy_eval` (dumped cross-tree from `me2-iteration3`). Full report:
`out/vcm/option-d-fil50-ambient-rir-135m/dense_pilot/metadata/dense_pilot_report.{json,md}`.

- **Parity:** baseline recomputed from the CPU logits reproduced every documented integer count exactly
  (0 rows tolerated as numerics), so the verdict is valid. Baseline for this checkpoint is clean babble
  FAR 3/255, silence 1/324 (not the older checkpoint's ~0.008/0.000).
- **Variants:** D1 = mean log-prob over non-blank frames of the Viterbi alignment of the winner; D2 = raw beam
  mass / len(winner text). Thresholds calibrated on clean val only (iso-accept vs. baseline −0.1).
  Val-selected variant: **`d2-global`** (val reject-probe FAs, clean+noisy at margin 4.0: d2-global 2,
  d2-per_intent 4, baseline-per_intent 7, d1-global 9, d1-per_intent 36).
- **Verdict at margin 4.0 (test, selected vs. baseline): NO-GO — R3, R5, R6 failed.**

  | | baseline (−0.1) | d2-global |
  |---|---|---|
  | clean exact | 3434/3494 | 3425/3494 |
  | noisy exact | 3282/3494 | 3251/3494 (−0.89pp, R5 budget 0.5pp) |
  | clean reject FA | 2 | 0 |
  | noisy reject FA (babble+silence) | 4 (2+2) | 0 |
  | noisy TIME+STOP FA | 1 | 0 |

  R1, R2, R4, R7 pass. **R6 fails on exactly the short intents dense scoring targets:** noisy exact
  PAUSE 89→84/100, STOP 133→126/134, TIME 95→91/107 (clean PAUSE 99→94/100).
- **R3 caveat (a flaw in the pre-committed rule, not changed after the fact):** at margin 4.0 the baseline has
  only 4 noisy-test reject FAs (the production margin gate already removes 12 of the 16 seen at margin
  None). The minimum achievable sign-test p with 4 removals and 0 additions is 1/16 = 0.0625 > 0.05, so R3
  was unpassable at that baseline count; the `MIN_BASELINE_NOISY_FA = 4` floor should have been 5.
  The verdict stands regardless (R5 and R6 fail independently).
- **Exploratory, post-hoc, not a criterion (test looked at once, n tiny):** `d2-global` with the margin gate
  OFF has 0 reject FAs clean and noisy, at exact 3437 clean / 3297 noisy — better than the production
  baseline + margin 4.0 on both accuracy (3434 / 3282) and FAs (2 / 4). That hints dense scoring might
  substitute for the margin gate rather than stack on it, but the margin-none configuration was not the
  pre-registered decision setting; treat as a hypothesis for a follow-up, not a result.
- **Follow-up, margin-gate replacement (pre-registered rule in `dense_pilot.followup_report`, hash in
  `metadata/frozen_before_followup.txt`; result `metadata/dense_followup_margin_off.json`) — FOLLOWUP-NO-GO,
  narrowly.** `d2-global`, margin OFF, vs. production baseline + margin 4.0. Passes on val AND test: noisy
  FA (val 1 vs 8, test 0 vs 4), clean FA (val 1 vs 3, test 0 vs 2), overall exact accuracy (val clean
  3264 vs 3242 / noisy 3021 vs 2985; test clean 3437 vs 3434 / noisy 3297 vs 3282). Fails only the
  per-intent budget: val CALL, MESSAGE, PAUSE, PLAY_MUSIC; test PAUSE, STOP (>3pp drop in clean or noisy
  exact). Test was already seen once for this config, so it is a non-independent confirmation; val is the
  independent evidence. Not implemented in the decoder; per-intent variants were worse on val FAs
  (d2-per_intent 10, d1-per_intent 243, vs. d2-global 2).
- **Hybrid pilot (D2, margin OFF, relax-only per-intent thresholds tau_i = min(tau_g, tau_i_iso); single
  pre-committed candidate; val evaluated out-of-fold over two speaker-disjoint halves; rule + hash in
  `metadata/frozen_before_hybrid.txt`, result `metadata/dense_hybrid_report.json`) — NO-GO.** G4 (per-intent
  budget) now passes on both OOF val and test, and accuracy rises (test exact clean 3449 / noisy 3337 vs.
  production 3434 / 3282; OOF val 3279 / 3103 vs. 3242 / 2985). But it gives the FA win back: OOF val clean
  reject FA 5 vs. B 3 (**G2 fails**), and total reject FA exceeds the `d2-global` control + 2 slack on both
  splits (**G5 fails**: OOF val 9 vs. 6, test 4 vs. 0). Test H vs. B: clean FA 2 vs. 2, noisy FA 2 vs. 4
  (non-independent). The relaxed thresholds (PAUSE −3.10, MESSAGE −2.30, STOP −1.93, NEXT −1.60,
  TIME −1.52 vs. global −0.91) are what let FAs back in. Caveats: FA counts are tiny (B: 3/8 val, 2/4 test),
  test was already seen for `d2-global`, and the `hybrid` CLI stage has unit tests but no end-to-end test.
- **Ghost-clip probe (2026-09-30; independent data: 141 `acoustic_ghost` clips, never in this checkpoint's
  training manifest nor used for any threshold; script `scripts/dense_pilot_ghosts.py`, rule frozen in
  `metadata/frozen_before_ghost_probe.txt`, result `metadata/dense_ghost_probe.json`).** Every ghost must be
  rejected, so any accept is a false accept. Rule: candidate PASSES iff ghost FA <= production's. Results
  (FA of 141): **production (−0.1 + margin 4.0) 27 (19.1%, Wilson upper 26.4%)**; baseline margin off 57;
  **`d2-global` margin off 3 (2.1%, upper 6.1%) — PASS**; **hybrid (relax-only per-intent) 11 (7.8%) — PASS**.
  Hybrid's FAs cluster on exactly the relaxed intents (STOP 5, PAUSE 4); `d2-global`'s 3 are TIME/STOP/NEXT.
  Caveat: ghost ground truth is "not a command" by construction (harvested where a detector fired), and
  the podcast audio is out-of-distribution for babble/silence-based calibration. **Unexplained discrepancy:**
  this section's earlier note reports 0/58 test-split ghost FAs for the older `option-d-dataset-v2`
  checkpoint, but the current production checkpoint accepts 13/58 on the same podcast (podcast_3) at margin 4.0;
  not investigated (different checkpoint/threshold, old figure not re-run).
- **Margin gate vs. D2, head to head (2026-09-30; `metadata/dense_margin_vs_d2.json`).** Exact-correct targets /
  reject-probe FAs (babble+silence); val = 3330 targets, test = 3494; ghost FA of 141:

  | config | tau | val clean | val noisy | test clean | test noisy | ghost FA |
  |---|---|---|---|---|---|---|
  | baseline, margin off | −0.100 | 3263/13 | 3062/34 | 3448/4 | 3345/16 | 57 |
  | baseline, margin 2 | −0.100 | 3251/5 | 3002/8 | 3436/2 | 3298/4 | 36 |
  | **baseline, margin 4 (production)** | −0.100 | 3242/3 | 2985/8 | 3434/2 | 3282/4 | 27 |
  | baseline, margin 6 | −0.100 | 3230/3 | 2962/8 | 3425/2 | 3265/3 | 23 |
  | **d2-global, margin off** | −0.907 | 3264/1 | 3021/1 | 3437/0 | 3297/0 | 3 |
  | d2-global + margin 4 | −0.880 | 3243/1 | 2971/1 | 3425/0 | 3251/0 | 3 |

  The margin gate buys FA reduction at a steep accuracy price (margin 4 vs. off: −77 noisy val, −63 noisy test
  exact) and saturates on babble/silence beyond 4. D2 without the gate has fewer FAs AND more exact-correct
  than production on every split. **Adding the margin gate on top of D2 removes no further FAs (val/test/ghost
  unchanged) and only costs accuracy — the gate is redundant under D2, so the ghost win is not an artifact of
  dropping the gate.** Cost of D2 vs. production persists per intent (test noisy PAUSE 89→86, STOP 133→126;
  clean PAUSE 99→94). D2's iso-accept tau was set to the margin-off baseline's accept count; its FA headroom
  (test 0 vs. 4, ghost 3 vs. 27) suggests a looser tau could recover PAUSE/STOP accepts — untested hypothesis.
- **Looser D2 threshold (2026-09-30; `scripts/dense_pilot_loosen.py`, rule frozen in
  `metadata/frozen_before_loosen.txt`, result `metadata/dense_loosen_probe.json`) — FAIL by the pre-registered
  rule, by one intent-cell.** Margin OFF; global tau chosen on VAL ONLY as the loosest tau keeping val reject FAs
  within production's val counts (clean 3, noisy 8): **tau* = −1.204** (iso-accept was −0.907). Rule: L1 test FA
  <= B, L2 ghost FA <= B, L3 exact >= B (val+test, clean+noisy), L4 per-intent (>=50) <= 3pp drop.

  | | production (m=4) | D2 iso (−0.907) | D2 loose (−1.204) |
  |---|---|---|---|
  | val exact/FA clean | 3242/3 | 3264/1 | 3277/2 |
  | val exact/FA noisy | 2985/8 | 3021/1 | 3094/4 |
  | test exact/FA clean | 3434/2 | 3437/0 | 3446/0 |
  | test exact/FA noisy | 3282/4 | 3297/0 | 3344/0 |
  | ghost FA of 141 | 27 | 3 | 5 |
  | test PAUSE / STOP exact, clean | 99 / 132 | 94 / 132 | 97 / 133 |
  | test PAUSE / STOP exact, noisy | 89 / 133 | 86 / 126 | 87 / 129 |

  L1, L2, L3 pass; **L4 fails on one cell: MESSAGE val/clean 95→92 of 97 (−3.09pp vs. the 3.00pp budget)** — i.e.
  one more correct row would have passed. PAUSE/STOP are recovered to within budget on val and test. The loosening
  curve (`curve` in the JSON) shows where it breaks: ghost FA rises steadily as tau loosens (3 at −0.907 → 9 at
  −1.504), and test noisy FA first appears at ~−1.42; tau* sits at a val FA cliff edge (−1.248 already has val
  clean FA 4 > 3), so the val selection may be optimistic. Test was already seen for D2; ghosts and the val-only
  selection are the independent parts.
- **Implementation status + calibration checks for shipping D2 (2026-09-30; feature `dense-d2-loose-impl`,
  plan `.scratch/dense-d2-loose-impl/PLAN.md`).** Tasks 1-3 landed as opt-in code, default behavior unchanged
  (`score_mode` in `decode_utterance`/`decode`/pipeline/streaming/config/CLI/Makefile; `evaluate.py
  --score-mode` with `PER_CHAR_THRESHOLD_GRID`). Two checks on the production checkpoint:
  1. **Real-harness cross-check (`scripts/dense_pilot_harness_check.py`, `metadata/dense_harness_check.json`):
     MATCH.** `python -m me2_voicegen.vcm.evaluate --score-mode per_char` (CPU, fil50 manifest, beam 50; report in
     `dense_pilot/eval_per_char/metadata/`) agrees exactly with the offline D2 records on all 20 val sweep rows
     (target accept rate and reject FA rate) and on the test counts at its own chosen threshold (Youden -> −1.1:
     test accept 0.986, exact 0.985, babble FA 0/255, silence FA 0/324). So the offline pilot numbers describe the
     real decoder path.
  2. **tau* stability (`scripts/dense_pilot_stability.py`, rule frozen in `metadata/frozen_before_stability.txt`,
     `metadata/dense_tau_stability.json`): UNSTABLE by one false accept.** tau* fit separately on two
     speaker-disjoint val halves: −1.344 (half 0) and −1.074 (half 1), gap 0.27 (S1 <= 0.30 passes).
     Applied to the other half: fit-half0→apply-half1 clean FA 3 vs. production 2 (S2 fails; noisy FA 2 vs. 5), the
     other direction 0 vs. 1 clean and 1 vs. 3 noisy. Pooled held-out exact: clean 3273 vs. 3242, noisy 3078
     vs. 2985 (S3 passes). Interpretation: the loose end of the range (~−1.34) leaks FAs; the val-full tau*
     (−1.204) sits inside a −1.07..−1.34 band that a two-way split can't pin down. A more conservative tau is a
     hypothesis, not a result: choosing it after seeing this would need a new pre-registered rule.
  Test/val were both seen during pilots, so these are consistency checks, not fresh-data evidence.
- **Task-5 validation gates, run sequentially for production B, conservative C1 (per_char −1.1, val-Youden from
  the real harness) and current C2 (per_char −1.204); no margin gate on C1/C2 (`scripts/dense_pilot_gates.py`,
  rule frozen in `metadata/frozen_before_gates.txt`, summary `metadata/dense_gates_summary.json`). Verdict per the
  pre-registered strict per-cell rule: FAIL for BOTH candidates — on P1 only.** C1 was fixed from val-only
  information, but the earlier loosening curve had printed test numbers near −1.1, so it is not test-blind.

  | gate | B (production) | C1 (−1.1) | C2 (−1.204) |
  |---|---|---|---|
  | P1 offline, strict per-cell (val+test, clean+noisy) | — | **FAIL** (MESSAGE val/clean, val/noisy) | **FAIL** (MESSAGE val/clean) |
  | ...FA <= B and exact >= B in every cell | — | pass | pass |
  | P2 ghost FA of 141 (B = 27) | 27 | 4 pass | 5 pass |
  | P3 real wakeword->VCM cascade soak, ACCEPT count | 0 | 0 | 0 |
  | P3b VCM-only ambient soak triggers (4 recordings, ~3.5 h) | 303 | 14 pass | 17 pass |
  | P4 streaming recall, INT8 CLI, gate none, 821 test clips: correct first trigger | 720 | 746 pass | 737 pass |
  | ...wrong-intent triggers | 111 | 64 | 74 |
  | ...PAUSE / STOP / TIME | 87 / 129 / 86 | 96 / 131 / 92 | 95 / 131 / 92 |

  Caveats stated plainly. **P3 is vacuous:** on all 4 ambient recordings the wakeword never opened a listening
  period (0 periods for every config), so the VCM never ran inside the real cascade; the earlier soak's 10
  false triggers came from podcast speech that is not available here. **P3b was added after B's P3 result showed
  that (amendment logged in `frozen_before_gates.txt`, before any candidate number existed)** — it runs the VCM
  alone over the same audio, so it upper-bounds compound false actions (no wakeword filter). **A cascade
  "sesame"+command composition test was tried and discarded:** the wakeword gate re-fires through the command
  speech, restarting the 3 s period so it never closes on the command — it would have measured the wakeword, not
  the scoring change. P4 is therefore a streaming-VCM test, not a full cascade recall. Composed clips are
  mostly voice-cloned test audio (see Iteration 3 notes on synthetic share). **Clip-level vs. streaming:** on
  whole clips D2 loses a few PAUSE/STOP accepts vs. production (see loosen probe), but in streaming windows
  (2.5 s, mostly silence around the command) it wins on all three focus intents — the per-frame score is diluted
  by non-speech frames there, which is what per_char removes. **The strict per-cell rule fails on a single
  intent (MESSAGE, n=97 on val):** C1 is worse than C2 there (stricter threshold rejects more MESSAGE), so
  "more conservative" does not fix it; MESSAGE also appeared in the val-half stability analysis. Not fixed;
  candidates for follow-up are MESSAGE-specific (long/low-per-char phrasing) analysis or a per-intent floor
  registered before looking at fresh data. All of val/test have now been seen many times; only ghosts, the
  streaming clips and the ambient soak are independent of threshold selection.
- MV2: 0 alignment failures on all 4 dumps. MV4: beam mass exceeds the Viterbi score by median 3.9 nats
  (p95 9.6, max 21.1) on clean val targets, so D1 is a loose proxy for beam belief.
- Decision constants were frozen before test scoring (`metadata/frozen_before_test_scoring.txt`, sha256 of
  `dense_pilot.py`); the test split was scored and reported in a single run. Nothing was committed to git.

## wakeword-sesame — parallel "sesame" wakeword phrase-instance (COMPLETE, comparison-only)

Separate initiative from the accent-balance-fil50 iterations above — motivated by
"computer"'s pronunciation variance as a wakeword, not by accent balance. Ticket:
`.scratch/wakeword-sesame/tickets/00-RECAP.md`. Built on
`iteration3-negatives-timestretch` (commit `b50541a`); model artifacts committed on
`optionb-grammar-v2` (commit `612f72b` — see that commit's message for why the split).
**Not a production cutover**: the shipped "computer" checkpoint (`out/wakeword/`,
`out/wakeword-fil50/`) is untouched throughout; this is a parallel, comparison-only
dataset + checkpoint (`out/conversions/v2/wakeword-sesame/`, `out/wakeword-sesame/`).

**Key finding:** today's "computer" positives (`positives_real` → `positives_converted`,
~4,212 of 5,492 `_wakeword_` rows) are fundamentally computer-only — voice conversion is
content-preserving timbre transfer and cannot change the spoken word, and neither
upstream corpus (Picovoice `wake-word-benchmark`, Mycroft `Precise-Community-Data` — both
checked live against the real repos via `gh api`) has a "sesame" directory. Sesame
positives required zero-shot TTS instead of the real-recording pipeline.

**Two-approach QA journey for the positive class** (the interesting part):
1. Bare `"Sesame."` zero-shot resynthesis across the 35-voice reference pool
   (`convert_positives.py --mode resynthesize`, new): 3,744 planned pairs → **8.8% QA
   pass (329 clips)**. Root cause confirmed from CosyVoice2's own generation-time
   warnings: target text far shorter than the ~8-12s reference prompts.
2. Voice-converting those 329 verified seeds into new voices (`convert_positives.py
   --mode convert`, the *existing, unmodified* mechanism — structurally immune to the
   text-length failure since conversion never resynthesizes text): 2,632 planned pairs
   → **31.0% QA pass (816 clips)**. Combined final positive class: **1,145 clips**.

**Final dataset** (`out/conversions/v2/wakeword-sesame/manifest.csv`, gitignored generated
data like every other wakeword subset): **3,056 rows** — `_wakeword_` 1,555
(1,089/311/155 train/val/test), `_unknown_` 1,247 (873/249/125, incl. 315 new
phonetically-justified sesame adversary phrases), `_silence_` 254 (170/56/28). Zero
group-disjointness violations; zero "sesame" token leakage in `common_voice_negative`
(checked corpus-wide: 0/28,186).

**Checkpoint eval** (epoch 16 of 26, early-stopped): `_wakeword_` P0.984/R0.994/F1 0.989,
`_unknown_` P0.992/R0.980/F1 0.986, `_silence_` 1.000/1.000/1.000. Accent recall filipino
0.993 (150/151) vs non-filipino 0.994 (159/160) — a **0.1-point gap**, essentially at
parity without any dedicated rebalancing pass (unlike "computer", which needed the full
accent-balance-fil50 program above to close a 13-point gap). Export: fp32 0.119 MB, int8
0.047 MB (identical size to the computer model); p50 latency 0.248 ms / 0.143 ms (EPYC
estimate, not real RPi hardware).

**Code changes** (both backward-compatible, each proven by a regression test):
`generate_adversaries.py` gains a `--phrase-set {computer,sesame}` registry;
`build_unknown_external.py` gains `--scrub-word` (was hardcoded to "computer");
`convert_positives.py` gains `--mode {convert,resynthesize}`; `wakeword/train.py`'s
`LICENSE_NOTE` (previously hardcoded to the computer dataset root) is now
`license_note(manifest_path)`; `accent_balance/plan_jobs.py`'s wakeword text is now a
`--wakeword-text` flag (default unchanged) instead of a bare `"Computer."` literal.

**Open items, not addressed here (recorded, not fixed):** promoting this checkpoint to
production; recalibrating the streaming gate's wakeword threshold for sesame (still the
unchanged 0.9 "computer" default). Full numbers, per-voice QA tables, and the complete
task-by-task history: the ticket's Execution Log.

## ambient-reverb-cooccurrence — RIR+babble joint training experiment (COMPLETE, PROMOTED TO PRODUCTION)

Follow-up to the `ambient-noise-overlay` feature (which materialized offline babble noise
into the VCM fil50 and sesame wakeword manifests). Ticket:
`.scratch/ambient-reverb-cooccurrence/tickets/00-RECAP.md`. Original question: is the
current babble-only training regime enough for production, or does reverb need to be
layered on top, and is doing so even acoustically realistic given babble is already
additively mixed into the audio (verdict: yes, realistic — convolution is linear, so
convolving the already-mixed signal with one shared RIR is equivalent to reverberating
both sources separately in the same room).

**Verified asymmetry that motivated the work:** VCM's `--p-rir` already defaulted to 0.3
(so `option-d-fil50-ambient` already had joint RIR+babble exposure); wakeword's
`train.py` had no `--p-rir` flag at all and silently ran at 0.0 — zero reverb exposure
despite having babble.

**First pass (90/30-min budgets) — a null-to-negative result that turned out to be
confounded, not real:** raising VCM's `p_rir` 0.3→0.7 and adding `p_rir 0.3` to wakeword
produced flat-to-worse results (VCM noisy exact accuracy unchanged to 3 decimals with a
10× noisy-babble-FAR increase; wakeword test recall fell 0.899→0.856). Root causes,
found by a deliberate follow-up rather than accepted at face value: (a) the `-rir` runs
were undertrained relative to their baselines under the same wall-clock budget (RIR
convolution is expensive per batch); (b) `WakewordDataset._noise_pool()`'s
`noise_root.glob("*.wav")` was non-recursive while the actual noise corpus's wavs live
under `.../background_noise/audio/` — the online ESC-50 noise pool had been **silently
empty for every wakeword training run to date** (production `wakeword`, `wakeword-fil50`,
and every sesame variant). Fixed (`.rglob`), with a regression test proving the real bug
shape (`tests/test_wakeword_dataset.py::test_noise_pool_finds_wavs_nested_under_an_audio_subdirectory`).

**Second pass (1.5× time, fixed noise pool) — confound resolved, real gains on both
models:**

| VCM (test split, cross-tree noisy_eval, seed 0) | clean exact | noisy exact | noisy-clean gap | noisy babble FAR |
|---|---|---|---|---|
| `option-d-fil50-ambient` (p_rir 0.3, 56ep, 90min) | 0.9797 | 0.9256 | −5.4pt | 0.004 |
| control: same config, 135min (p_rir 0.3, 78ep) | 0.9834 | 0.9359 | −4.6pt | 0.012 |
| **`option-d-fil50-ambient-rir-135m`** (p_rir 0.7, 67ep, 135min) | **0.9868** | **0.9574** | **−2.9pt** | **0.039** |

The p_rir-0.3-at-135min control isolates the two effects: more training time alone
recovers ~28% of the gain at ~1/3 the FAR cost; the added reverb exposure (0.3→0.7 at
equal time) buys the rest at proportionally more FAR cost. Both effects are real, not one
masking the other — this was a deliberate confound-settling experiment, not an assumption.

| Wakeword (test split, wakeword noisy_eval gate, seed 0) | test recall | test F1 | test silence recall |
|---|---|---|---|
| `wakeword-sesame-ambient` (broken pool, p_rir 0, 30min) | 0.899 | 0.944 | 0.821 |
| `wakeword-sesame-ambient-rir` (broken pool, p_rir 0.3, 30min) | 0.856 | 0.913 | 0.679 |
| **`wakeword-sesame-ambient-rir-45m`** (fixed pool, p_rir 0.3, 45min) | **0.952** | **0.964** | **1.000** |

Fixing the noise-pool bug and extending training time together reverse the silence-recall
collapse entirely and produce the best-performing sesame checkpoint measured to date.

**Promotion (2026-09-28/29, user decision) — production checkpoints as of this entry:**
- VCM: **`out/vcm/option-d-fil50-ambient-rir-135m`** (commit `7c94abb`), superseding
  `option-d-fil50-ambient` / `option-d-fil50`. Accent gap 0.0074 vs. 0.0153 (roughly
  halved); 18/19 intents improved. Traded cost, quantified not hidden: noisy babble FAR
  0.039 vs. 0.004 (the checkpoint's own clean-val-chosen threshold moved looser,
  −0.075→−0.1).
- Wakeword: **`out/wakeword-sesame-ambient-rir-45m`** (commit `d5cd0e8`), superseding the
  "computer" wakeword and plain `wakeword-sesame`. No regression found anywhere; accent
  gap 0.5pt (filipino 0.990/202, non-filipino 0.995/206).
- Both exports/benchmarks pass their latency budgets with large margin (VCM p50
  3.7-3.8ms vs. no stated ceiling beyond the RPi-indicative 20ms/100ms-frame figure;
  wakeword p50 0.14-0.31ms).

**Cascade soak test (2026-09-29, real hardware, ~8 cumulative hours,
`docs/CASCADE-SOAK-TEST.md`):** the promotion above rested on each checkpoint's own
*isolated* noisy-gate FAR; multiplying VCM's 0.039 by wakeword's 0.040 (`_unknown_` miss
rate under noise) gives an estimated ~0.16% compound false-action rate assuming
independence — never actually measured end to end. This test runs the real
`me2_voicegen.vcm.streaming` cascade (`ListeningGate`/`wakeword_gate.py` → VCM, not a
hand-reconstructed call graph) against real podcast/ambience recordings
(`raw_datasets/ambient-noise/`) containing zero genuine "sesame" utterances. Result: **10
wakeword false triggers, 0 resulted in a VCM accept** — all 10 on loud/emphasized,
close-mic natural speech outside the training distribution, a failure mode the synthetic
noisy_eval gate never exercises. Caveat stated plainly: n=10 is a small sample (95% CI
upper bound on the true compound rate is ~25-30%, not near-zero) — this is "no compound
false actions observed under real, hard conditions," not a confidently near-zero claim.
Secondary, non-blocking finding: 1.25 wakeword false-triggers/hour on this content is a
real standby-cost concern independent of the (so-far zero) compound-accept risk.

**Code changes** (all backward-compatible): `wakeword/train.py` gains `--p-rir` (default
`0.0`); `wakeword/dataset.py`'s `_noise_pool()` now `rglob`s instead of `glob`;
`wakeword/noisy_eval.py` is new (a wakeword port of `vcm/noisy_eval.py`'s fixed-seed
determinism contract).

**Open items, not addressed here:** the standby-cost false-trigger rate (1.25/hr on
emphatic close-mic speech) is not fixed, only measured and flagged; a p_rir=0.3-at-135min
wakeword control (mirroring the VCM confound-settling control) was not run, since the
wakeword result already combined the pool fix with the time extension by design. Full
task-by-task history: the ticket's Execution Log.
