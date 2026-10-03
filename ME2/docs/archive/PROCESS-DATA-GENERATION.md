# Process: Data Generation

How raw audio training data gets made, from zero-shot TTS / voice conversion through the final
Option B VCM dataset and its in-flight rebalancing experiment. See `PROCESS-OVERVIEW.md` for how
this fits into the whole pipeline.

## Pipeline stages

| Stage | Command | What it does |
|---|---|---|
| 0. Setup | `make sync && make vendor && make download-model` | Vendors upstream CosyVoice2 at a pinned SHA, downloads the CosyVoice2-0.5B checkpoint (~4.9GB). Prerequisite for everything below. |
| 1. Single-sample synthesis | `make generate` → `me2_voicegen.generation.generate_sample` | One zero-shot TTS call (`--backend cosyvoice2 --text ...`); proves the backend works end to end. |
| 2. Cartesian voice-conversion batch | `make generate-conversions` → `me2_voicegen.generation.generate_conversions` | Every clip in `--raw-dir` × every reference voice in `--refs-dir` run through `CosyVoice2Synthesizer.convert_voice` (timbre transfer, content/prosody preserved exactly, no transcript needed). With probability `--resynth-prob` (default 0.3) per pair, also produces a zero-shot resynthesis from a Whisper-transcribed (cached `.txt` sidecar) text. Built `out/conversions/v2/test_set/`, the Option A (toy VCM) dataset. |
| 3. Multi-persona batch synthesis | `make generate-personas` → `me2_voicegen.generation.generate_personas` | One text, once per persona in a JSON manifest (name/wav_path/text); builds the backend once, reuses across the batch. Fail-fast manifest validation before model load; a bad persona doesn't abort the batch. |
| 4. Option B dataset ingestion | `.scratch/optionb-dataset/` tooling | Fetches a real upstream labeled command corpus, reconciles it into the VCM manifest schema, wires it into `evaluate.py --grammar optionb`. This is the real-audio foundation everything downstream (VCMX, accent-balance) builds on. |
| 5. VCM Dataset B merge / VCMX build | `make optionb-refresh && make vcmx-build` | Merges Option B (refreshed to live upstream grammar) with a second balanced dataset into speaker-disjoint treatment/control manifests. Full numbers in `PROCESS-VCM-MODEL.md` / `PROCESS-STREAMING-SERVING.md`. |
| 6. `accent-balance-fil50` (complete) | `scripts/accent_balance_fil50.sh` | 7 stages: `build_refs → plan_jobs → generate → qa → pilot_report → collate → train/eval`. Full run complete, all pre-committed criteria met — see "Current status" below and `MLOPS-PROJECTS.md`. |
| 7. Wakeword dataset build | `src/me2_voicegen/wakeword/` | Reuses the same voice-conversion mechanism as stage 2 for positive augmentation. Full detail in `PROCESS-WAKEWORD.md`. |

## Key design decisions and why

- **Voice conversion vs. zero-shot TTS — chosen per use case, not universally.** `convert_voice`
  preserves exact target text/prosody with no transcript needed, so it's used wherever
  content-correctness matters (the `test_set/` build, wakeword positive augmentation).
  `synthesize` (zero-shot) lets the model re-type content in a new voice, so it's used where
  genuinely novel utterances are wanted (persona batches, the `accent-balance-fil50` references).
  The `accent-balance-fil50` pilot (below) is the direct, quantified confirmation of why this
  distinction matters: zero-shot from a foreign-language prompt voice is unreliable at
  content-fidelity; conversion is not.
- **`--resynth-prob` blends both mechanisms per raw/reference pair** in `generate_conversions.py`
  rather than picking one exclusively, to get both a guaranteed-correct converted variant and a
  genuinely novel resynthesized variant from the same cartesian pair.
- **Reference-clip auto-trim (30s → 25s)** rather than failing pairs outright, so one reference
  voice going over CosyVoice2's hard prompt-length limit doesn't lose every pair that uses it.
- **Whisper transcription caching** (`.txt` sidecars) so repeat batch runs don't re-transcribe
  already-processed clips.
- **50/50 Filipino/non-Filipino rebalance rationale.** The product targets a mostly-Filipino
  audience but real data skews far off that: VCM is 15% Filipino, wakeword is 35% Filipino
  (measured — see table below). Design choices logged 2026-09-25: persona (zero-shot) TTS for
  both models, all 141 available Filipino voices (124 `sapinsapin` + 17 `references`), no
  dialect-family balancing, pilot-first-then-stop-for-review.
- **Fail-loud row-budget math in `collate.py`.** A synthesis job can yield 1 or 2 manifest rows
  (clean + noise-mixed sibling), so hitting an exact 50/50 split requires greedy selection against
  a *row* budget, not a *job-count* budget — an overshoot-by-one is resolved by dropping that
  job's noisy pairing rather than approximating the ratio.

## Experiments run, with actual numbers

### Option B dataset ingestion (2026-09-19)

Real upstream manifest reconciled: 17,656 target-command rows + 786 borrowed babble/silence probe
rows = **18,442 total** rows. Two real bugs found and fixed during ingestion, not just noted:

- 196 files (98 speakers, `VOLUME_DOWN` clean+noisy) were actually at 12kHz despite a claimed
  16kHz — captured per-row instead of hardcoded, resampled defensively at load time.
- Grammar drift: `CMD_VOLUME_DOWN`'s literal was corrected from "decrease the volume" (spec text)
  to "lower the volume" (actual manifest, verified 196/196 rows via full set-difference, not
  eyeballing).

Result: 93/93 real manifest transcripts grammar-accept end-to-end; full test suite passed
(4,556–4,568 tests across these tickets).

### VCM Dataset Compatibility worked example (Option B, `optiond` preset, epoch 71)

| metric | value |
|---|---|
| target commands accepted (test split) | 1,717 / 1,766 (0.972) |
| exact-intent-correct | 1,715 / 1,766 (0.971) |
| false-accept rate, babble | 1 / 56 (0.018) |
| false-accept rate, silence | 0 / 22 (0.000) |
| slot values correct (of intent-correct, slot-bearing) | 1,022 / 1,023 |

### `accent-balance-fil50` build status (2026-09-25)

Build itself is solid: T1–T7 stages built, 173 feature tests + 5,610 full-suite fast tests green,
no regressions.

**Measured Filipino-speaker deficit** against current manifests (this is *why* the feature exists):

| model | split | non-Filipino | Filipino | deficit |
|---|---|---|---|---|
| VCM | train | 13,723 | 2,214 | 11,509 |
| VCM | val | 1,665 | 348 | 1,317 |
| VCM | test | 1,747 | 180 | 1,567 |
| wakeword | train | 2,511 | 1,334 | 1,177 |
| wakeword | val | 712 | 385 | 327 |
| wakeword | test | 359 | 191 | 168 |

**Pilot run** (2026-09-25, `PILOT=60`, GPU 3, ~9 min wall-clock, 120 real clips): throughput VCM
3.30s/clip, wakeword 2.37s/clip → full-run projection **16.4 GPU-hours** (16,554 VCM jobs + 1,925
wakeword jobs), matching the original 15–20 GPU-hour estimate.

**Pilot QA pass rate — the key finding that's blocking the full run:**

| unit | n | pass rate |
|---|---|---|
| `vcm_references` | 24 | 87.5% |
| `vcm_sapinsapin` | 36 | **41.7%** |
| `wakeword_references` | 24 | 95.8% |
| `wakeword_sapinsapin` | 36 | **30.6%** |

Root cause, verified against actual transcripts (not just QA scores): zero-shot synthesis from a
Tagalog-language voice *prompt* frequently produces Tagalog-*sounding content* instead of the
English target text — a cross-lingual content-fidelity failure, not a QA-threshold miscalibration.
The `references` voices (Filipino people already speaking English) have no such mismatch and pass
at baseline rate, matching the design rationale for why the original spike (see
`PROCESS-OVERVIEW.md`) picked references-via-TTS over sapinsapin-via-TTS in the first place.

Two shell-script bugs (both `set -e` interactions, not Python bugs) were also found and fixed
during the pilot: a command-substitution's failing last-iteration test silently killing the whole
script with zero output, and a pilot-report stage silently never running because its stage-id
token was missing from the filtered stage list.

## Current status / open decision points

**Decided (2026-09-25) and complete (2026-09-26): Option 1, references-only.** Of the three
options below, references-only was chosen — ship now with the 17 Filipino `references` voices,
deferring `sapinsapin` diversity to a fast-follow iteration (see Iteration 2 in
`MLOPS-PROJECTS.md`). The full run (collate/train/eval) completed 2026-09-26 with all
pre-committed success criteria met; full numbers in `MLOPS-PROJECTS.md`'s Iteration 1 Results.

1. **References-only — CHOSEN.** Drop `sapinsapin` entirely, ship now with only 17 Filipino
   voices. Zero new engineering; caps Filipino-speaker diversity at 17 vs. the 124-speaker goal.
2. **Fix `sapinsapin` via voice conversion instead of zero-shot — fast-follow (Iteration 2).**
   Swap `synthesize()` for `convert_voice()` (the same mechanism as stage 2 / wakeword positives),
   using a same-split non-Filipino real recording as the source clip. Keeps target words correct
   by construction; carries the caveat that timbre transfer alone isn't a full accent signal.
3. **Blend — not pursued.** References for guaranteed quality + whatever fraction of `sapinsapin`
   passes QA as-is (~30–42% yield, i.e. ~60 usable speakers' worth instead of 124).

## Known limitations/gaps

- `accent-balance-fil50`: `collate.py` is unit-tested only, never run against real data;
  `docs/ACCENT-BALANCE.md` runbook not started; tech-lead review not done; pilot output
  (`out/conversions/v2/fil50-pilot60/`) sits on disk uncommitted/unreviewed (gitignored, so safe,
  just not cleaned up).
- Reference-clip consent/licensing is manual/human-verified only — no automated check that a
  reference wav's usage rights permit voice cloning. Real manifests and audio are deliberately
  kept out of git for this reason.
- No RIR (reverb) augmentation at dataset-*build* time anywhere in this pipeline — still
  train-time-only via `common/augment.py`, an explicit non-goal of the original CosyVoice2 spike
  that has never been revisited.
- Option B ingestion: no upstream `LICENSE` file was found in the source repo — treated as
  all-rights-reserved if that ever matters; low risk currently because `out/` is gitignored and
  nothing built from it is redistributed.
- `VCM-DATASET-COMPATIBILITY.md` flags a structural hazard any new dataset (including
  `accent-balance-fil50`'s eventual output) must watch for: a manifest with no babble/silence
  reject probes produces a silently degenerate (always-0.0) false-accept rate instead of an error.
