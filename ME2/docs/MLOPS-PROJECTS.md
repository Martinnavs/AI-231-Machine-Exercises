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
(pending — post-training ghost FAR + mined-negatives `_unknown_` F1, after the
audited negatives are collated into the ww manifest)

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
(pending)
