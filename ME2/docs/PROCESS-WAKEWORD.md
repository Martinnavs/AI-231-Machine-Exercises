# Process: Wakeword DS-CNN

The "computer" keyword-spotting pipeline end to end — dataset build, model, training, benchmark,
and its integration into the streaming `ListeningGate` seam. See `PROCESS-OVERVIEW.md` for how
this fits into the whole pipeline, and `WAKEWORD-DATASET-CONTRACT.md` for the exact schema/
licensing contract this doc doesn't repeat.

A parallel, comparison-only **"sesame" phrase instance** (same pipeline modules, its own
dataset root and checkpoint) is documented in the section
["Parallel "sesame" phrase instance (comparison-only)"](#parallel-sesame-phrase-instance-comparison-only)
below.

## Dataset build pipeline

Ordered Makefile stages, run from `ME2/` (`WAKEWORD_ROOT = out/conversions/v2/wakeword/`):

| # | Command | Produces |
|---|---|---|
| 1 | `make wakeword-fetch-positives` | `positives_real/` — 468 clips (411 Picovoice `wake-word-benchmark`, Apache-2.0; 57 Mycroft `Precise-Community-Data`, per-file public-domain waivers, verified 57/57) |
| 2 | `make wakeword-generate-adversaries` | `adversaries/` — 245/245 TTS renders (7 near-miss phrases × 35 reference voices), `_unknown_` class |
| 3 | `make wakeword-convert-positives` | `positives_converted/` — 3,744/3,744 (468 real × 8 reference voices via `convert_voice`; full run ~5.3h single-GPU) |
| 4 | `make wakeword-generate-silence WAKEWORD_SILENCE_TOTAL_COUNT=1220` | `silence_synthetic/` — 1,220 Gaussian/pink/brown colored-noise clips, `_silence_` class |
| 5 | `make wakeword-mix-noise` | `adversaries_noisy/` (75) + `positives_converted_noisy/` (1,280) — additive ESC-50 mix at SNR 5–25dB; originals untouched |
| 6 | `make wakeword-build-unknown-external WAKEWORD_UNKNOWN_TARGET_COUNT=5172` | `common_voice_negative_sample/` — 5,172 rows sampled from 28,186 Common Voice rows (CC0-1.0), scrubbed for any "computer" token, resampled 48kHz→16kHz |
| 7 | `make wakeword-build-dataset` | `manifest.csv`/`summary.md` — merges everything into `_wakeword_`/`_unknown_`/`_silence_`, group-disjoint per-label-stratified 70/20/10 split |
| 8 | `make wakeword-derive-spans` | Precomputes `speech_start_s`/`speech_end_s` via `torchaudio.functional.vad`, one-time offline pass (training-time input, not part of dataset-build proper) |

**Final assembled dataset: 12,204 rows total** — `_wakeword_` 5,492 (3,845/1,097/550
train/val/test), `_unknown_` 5,492 (3,844/1,099/549), `_silence_` 1,220 (815/270/135, lumpier
split — only 9 groups). Every stage is seeded and reproducible byte-identical from the same seed.

`group_id` discipline prevents leakage: a real speaker's converted variants, and a clip's
noise-augmented twin, always share `group_id` (or `(group_id, ref_voice)`) so they can't land on
opposite sides of train/val/test.

**QA sanity check** (`simple-audio-transcriber`) against `_wakeword_` subsets: `positives_real`
98.7% pass (468 files, 6 flagged), `positives_converted` 90.1% (3,744, 370 flagged),
`positives_converted_noisy` 90.2% (1,280, 125 flagged) — noise mixing did not measurably hurt
intelligibility; most flags are the QA tool's own ratio-scoring quirk, not real degradation.

**Empirically measured facts baked into design decisions:** voice conversion preserves timing
(11ms mean / 64ms max drift across 3,744 rows); additive noise mixing preserves timing exactly
(0.0s drift across 1,280 rows); VAD empty-span rate is `positives_real` 5%, `adversaries` 54.3%,
`adversaries_noisy` 45.3%, `common_voice_negative_sample` ~53–55% — this asymmetry is why span
precomputation is applied only to the `_wakeword_` chain + `common_voice_negative_sample`, while
`adversaries`/`adversaries_noisy` stay on a live-VAD-with-fallback path.

## Parallel "sesame" phrase instance (comparison-only when built; PROMOTED TO PRODUCTION 2026-09-28/29 — see "Confound resolution, promotion, and cascade soak test" below)

Built 2026-09-27 (ticket: `ME2/.scratch/wakeword-sesame/tickets/00-RECAP.md`
in the main tree). Motivation: "computer" has too much pronunciation variance
as a wakeword. It is a **parallel, comparison-only** dataset
(`out/conversions/v2/wakeword-sesame/`) + DS-CNN checkpoint
(`out/wakeword-sesame/`), reusing this doc's pipeline modules unchanged
(`generate_adversaries.py` gained a `--phrase-set` switch in T1,
`build_unknown_external.py` a `--scrub-word` switch in T2 — both
backward-compatible, default behavior byte-identical). **Not a production
cutover at build time**: the "computer" checkpoints (`out/wakeword/`,
`out/wakeword-fil50/`) and the original `out/conversions/v2/wakeword/`
dataset were left untouched, and promotion of "sesame" was deferred as a
separate later human decision (mirrors the fil50-promotion precedent).
**That promotion has since happened** — see "Confound resolution,
promotion, and cascade soak test" below: `wakeword-sesame-ambient-rir-45m`
(a descendant of this dataset, with the ambient-reverb-cooccurrence
ticket's later changes layered on) is now the production wakeword
checkpoint, superseding "computer".

**Why the positive class had to be built differently (the part that isn't
just re-running stages):** `positives_real`/`positives_converted` are
"computer"-only by construction — voice conversion is content-preserving
timbre transfer, and neither upstream real corpus (Picovoice, Mycroft) has
a "sesame" directory (verified live against both repos). So the sesame
positives come from in-repo TTS, in two QA-gated batches (dual-transcriber
QA, faster-whisper EN+TL small, threshold 0.80, same tool/threshold as
fil50 stage 3):

| batch | content | QA result |
|---|---|---|
| `positives_seed_resynth/` (renamed from `positives_converted/` at the combine) | 329 QA-passed zero-shot resyntheses of "Sesame." across reference voices (`convert_positives.py --mode resynthesize`, T3) | 329/3,744 (8.8%) — CosyVoice's "synthesis text too short than prompt" warning line up with the failure mode |
| `positives_converted_v2/` | 816 QA-passed voice-converted copies of the 329 verified seeds (unmodified `--mode convert`, k=8, per-seed-unique `group_id` remap to avoid output-path collisions) | 816/2,632 (31.0%), per-voice 12.2%–52.0% |

**Combined positive class W = 1,145** (329 + 816, both used together).
Because the unchanged `build_dataset.py` only sweeps in its fixed subset
names, the combined class lives under the canonical `positives_converted/`
name (physical copies of both batches' audio + merged manifest; the two
original batch dirs stay intact as provenance), and `positives_real/` is an
intentionally empty, documented placeholder (no upstream real corpus exists
for "sesame"). Consequence: all sesame `_wakeword_` rows have no
precomputed speech spans (nothing to join from) and train on the
documented live-VAD-with-fallback path — the same supported state
`adversaries`/`adversaries_noisy` always use.

**Stage numbers (realized, all seeds 0, same conventions as the table
above):** 9 sesame adversary phrases × 35 reference voices = **315/315**
TTS renders (`adversaries/`); `unknown_target = max(1145 − 315, 100) =
830` → **830 rows** in `common_voice_negative_sample/` (scrubbed for
"sesame" — 0 of 28,186 corpus rows matched, so the full pool survived);
`silence_target = round(1145 × 1220/5492) = 254` → **254 rows** in
`silence_synthetic/`; noise mixing (p=0.35) → **102** `adversaries_noisy/`
+ **410** `positives_converted_noisy/` (CC-BY-NC-SA-4.0 note on both
subsets' `summary.md`, same as the computer subsets).

**Final assembled dataset (`build_dataset.py --seed 0`): 3,056 rows** —
`_wakeword_` 1,555 (1,089/311/155 train/val/test), `_unknown_` 1,247
(873/249/125), `_silence_` 254 (170/56/28); group-disjoint split verified
(0 violations).

**Training + eval (30-min budget, the computer instance's wall-clock
convention; preset=default, seed 0, GPU):** 26 epochs (early stop,
patience 10), best val_loss 0.0337 at epoch 16, 242 s wall-clock, no
NaN/Inf. Val-split eval (`out/wakeword-sesame/metadata/eval_report.md`):

| label | precision | recall | f1 | support |
|---|---|---|---|---|
| `_wakeword_` | 0.984 | 0.994 | 0.989 | 311 |
| `_unknown_` | 0.992 | 0.980 | 0.986 | 249 |
| `_silence_` | 1.000 | 1.000 | 1.000 | 56 |

`_wakeword_` recall by voice accent: filipino 0.993 (150/151),
non-filipino 0.994 (159/160) — a 0.1-pt gap, i.e. the accent-shortcut
risk that motivated `accent-balance-fil50` for "computer" does not show up
in the sesame comparison instance at this scale.

**Export/benchmark** (same `--model-family wakeword` reuse of the VCM
export/benchmark modules): fp32 ONNX 0.119 MB, p50 0.248 ms; INT8 (static,
val-calibrated) 0.047 MB (49,576 bytes — same size as the computer model,
same architecture), p50 0.143 ms. AMD EPYC 7742 estimates, not RPi
hardware (`out/wakeword-sesame/metadata/wakeword_benchmark.md`).

**Licensing:** pure in-repo TTS introduces no new license entry (contract
§7); the `*_noisy` rows make the whole sesame dataset
CC-BY-NC-SA-4.0-encumbered exactly like the computer dataset.

### Ambient babble + reverb co-occurrence experiment (2026-09-28, SUPERSEDED — see "Confound resolution, promotion, and cascade soak test" below)

Layered on the `ambient-noise-overlay` feature (its SPEC +
`out/conversions/v2/wakeword-sesame/summary.md` are the canonical
records): a derived `wakeword-sesame-ambient` dataset (3,056 base rows +
1,620 ASR-gated offline babble `_ambient` rows, SNR U[0,30] dB; 1,620 of
2,236 planned mixes passed the transcriber gate) and a **fixed-seed
noisy/reverb eval gate for this model family**
(`src/me2_voicegen/wakeword/noisy_eval.py`, new in the
`ambient-reverb-cooccurrence` ticket — a wakeword port of
iteration3's `vcm/noisy_eval.py`: every val/test row gets one
deterministic RIR (RT60 U[0.1,0.5] s, 200-entry pool) + one ESC-50 noise
clip at SNR U[5,25] dB, scored through the clean eval's argmax path;
seed 0, reports under `out/wakeword-sesame*/noisy_eval/metadata/`). A
`--p-rir` training flag was added to `wakeword/train.py` (default `0.0` —
every existing invocation byte-unchanged; this model family previously
had **no** RIR exposure at all) and drove one comparison checkpoint
`out/wakeword-sesame-ambient-rir` (p_rir 0.3; every other flag identical
to the ambient run).

`_wakeword_` recall under the fixed noisy/reverb gate (seed 0):

| checkpoint (training-time exposure) | val recall | test recall | test F1 |
|---|---|---|---|
| `out/wakeword-sesame` (clean + offline `_noisy` subsets) | 0.961 | 0.897 | 0.939 |
| `out/wakeword-sesame-ambient` (offline babble rows; p_rir 0) | 0.931 | 0.899 | 0.944 |
| `out/wakeword-sesame-ambient-rir` (offline babble + RIR p=0.3) | 0.936 | 0.856 | 0.913 |

**Verdict (stated plainly): elevating RIR did not improve the
noisy/reverb gate** — test-split `_wakeword_` recall 0.899 → 0.856 (F1
0.944 → 0.913); val roughly flat (0.931 → 0.936). Confounds, in order of
likely size: (a) under the same 30-min wall-clock budget the -rir run fit
29 epochs vs 42 (RIR-convolution cost) and its clean val loss was worse
(0.0267 vs 0.0203), so "more RIR" and "fewer epochs" moved together; (b)
the gate applies RIR+noise to *all* rows of the ambient manifest,
including rows that already carry baked-in babble — a double-condition
stress, not a matched-condition measurement; (c) the plain
`wakeword-sesame` row uses the base manifest (311/155 val/test
`_wakeword_` rows) vs the ambient manifests (408/208), so the
cross-manifest comparison is population-mixed. Comparison-only; no
promotion (per the ticket's non-goals, mirroring this feature's
"comparison, not cutover" framing).

**Latent finding (flagged, not fixed — it affects all wakeword training,
not just this experiment):** every wakeword run to date used
`--noise-root out/conversions/v2/background_noise`, but
`WakewordDataset._noise_pool()` globs `noise_root/*.wav`
**non-recursively** and the wavs live under
`.../background_noise/audio/` — so the online ESC-50 noise pool has been
silently **empty** in every run (`Augmenter.augment_waveform` skips the
noise step on an empty pool). The handoff doc's online noise probability
never actually fired; wakeword noise exposure came only from the offline
`*_noisy` subset rows. Baseline and -rir runs are equally affected, so
the comparison above is valid within that frame; fixing the root (or the
glob) is a separate decision with its own retraining cost.

### Confound resolution, promotion, and cascade soak test (2026-09-28/29 — PROMOTED TO PRODUCTION)

The "elevating RIR did not improve" verdict above turned out to be an artifact of two
confounds, not a real finding: the `-rir` runs were undertrained relative to their
baselines (43/29 epochs vs 56/42 in the same wall-clock budget), and the noise-root glob
bug (above) meant wakeword's online noise pool was silently empty in every run compared.
Both were controlled for:

1. **The noise-root glob was fixed** (`WakewordDataset._noise_pool()` now `rglob`s;
   `out/conversions/v2/background_noise` resolves to 2,153 wavs, not 0).
2. **Training time was extended 1.5×** (30min → 45min) with the fixed pool, `p_rir 0.3`:
   `out/wakeword-sesame-ambient-rir-45m` (26 epochs, best val loss 0.0244).

`_wakeword_` recall under the same fixed noisy/reverb gate (seed 0), full comparison:

| checkpoint | val recall | test recall | test F1 | test `_silence_` recall |
|---|---|---|---|---|
| `wakeword-sesame` (base manifest, no ambient) | 0.961 | 0.897 | 0.939 | 0.929 |
| `wakeword-sesame-ambient` (broken pool, p_rir 0) | 0.931 | 0.899 | 0.944 | 0.821 |
| `wakeword-sesame-ambient-rir` (broken pool, p_rir 0.3) | 0.936 | 0.856 | 0.913 | 0.679 |
| **`wakeword-sesame-ambient-rir-45m`** (fixed pool, p_rir 0.3, 1.5× time) | **0.975** | **0.952** | **0.964** | **1.000** |

The silence-recall collapse across the broken-pool runs (0.929 → 0.821 → 0.679) fully
reverses once real online noise diversity is restored — best target-word recall and
perfect silence recall of any sesame checkpoint tested. This is "working noise pool +
adequate training time", not "RIR probability" in isolation — the three changes moved
together, so this does not re-litigate the RIR-specific null result above, it supersedes
the broken baseline it was measured against.

**Promotion (2026-09-28/29, user decision):** `wakeword-sesame-ambient-rir-45m` checked
in as the production wakeword checkpoint (commit `d5cd0e8` on `optionb-grammar-v2`),
superseding both the "computer" wakeword and the plain (non-ambient) `wakeword-sesame`
checkpoint. Clean val `_wakeword_` recall 0.9926 — on par with `wakeword-sesame-ambient`
(0.9951), no regression. Accent recall gap 0.5pt (filipino 0.990/202, non-filipino
0.995/206). Export/benchmark: fp32 124.5 KB / INT8 49.6 KB, p50 0.31/0.14 ms — well
inside the ≤20 ms/100 ms-frame budget.

**Cascade soak test (2026-09-29, real hardware, ~8 cumulative hours):**
`docs/CASCADE-SOAK-TEST.md` runs the actual `me2_voicegen.vcm.streaming` cascade
(`ListeningGate`/`wakeword_gate.py` → VCM, the two newly-promoted checkpoints) against
real podcast/ambience recordings (`raw_datasets/ambient-noise/`) containing zero genuine
"sesame" utterances — this measures the real end-to-end false-action rate, not the two
stages' isolated FARs multiplied together (an independence assumption that was never
verified). Result: **10 wakeword-gate false triggers, 0 resulted in a VCM accept** (all
10 correctly REJECTed) — every trigger was loud/emphasized, close-mic natural speech
(e.g. "necesseties", "tapos tapos", "kisser he kisses"), an acoustic register outside the
training distribution and a failure mode the synthetic noisy_eval gate doesn't cover.
Caveat: n=10 is a small sample (95% CI upper bound on the true compound-accept rate is
~25-30%, not near-zero) — "no compound false actions observed under real, hard
conditions," not a confident near-zero claim. Secondary, non-blocking finding: 1.25
false wakeword triggers/hour on this content is a real standby-cost concern independent
of the (so-far zero) compound-accept risk; adding this speech register to wakeword's
`_unknown_` training data is a candidate future fix, not done here.

Full task-by-task history, the confound-control numbers, and the soak test's caveats in
full: `.scratch/ambient-reverb-cooccurrence/tickets/00-RECAP.md`'s Execution Log.

## Model architecture

`DSCNN` in `src/me2_voicegen/wakeword/model.py` — depthwise-separable CNN, 3-way classifier
(`_wakeword_`/`_unknown_`/`_silence_`), input `(B, 40, T)` log-mel (40 mel bins, 10ms hop, 30ms
window, 16kHz — same `LogMelFeatureExtractor` as the VCM model), output `(B, 3)` logits. Config is
a `DSCNNConfig` dataclass (`n_blocks`, `channels`, `kernel_sizes`) with a `PRESETS` registry. Sized
in the small-DS-CNN family (~24K–305K param range per the design doc); the actual trained model's
fp32 ONNX export is 124,542 bytes / int8 49,576 bytes, i.e. ≈50K params. Topology is informed by
the "Honk" DS-CNN reference; no code/weights vendored from it.

Export/quantize/benchmark deliberately reuse `vcm/export_onnx.py` and `vcm/benchmark.py` (extended
with a `--model-family {vcm,wakeword}` switch) rather than duplicating that code under `wakeword/`
— `wakeword/` owns only training-time code (model, dataset, augment, train).

## Training + integration

**Training** (`make wakeword-train`, checkpoint checked in at commit `99c529b`): windows audio to
1.5s (`WAKEWORD_WINDOW_SECONDS`), crop anchored to the precomputed `speech_start_s`/`speech_end_s`
span (+0.15s margin) when present, else live-VAD → center-crop fallback. Trained checkpoint:
preset=default, 20 epochs, `out/wakeword/checkpoints/checkpoint.pt`.

**Wiring into `ListeningGate`** (commit `3612feb`, 2026-09-24): adds `WakeWordGate` in
`src/me2_voicegen/vcm/streaming/wakeword_gate.py`, implementing the same `ListeningGate` protocol
`SpacebarGate` already satisfies, with zero changes to `AcceptancePolicy`/`StreamingRunner`.
Selected via `--gate wakeword`; polls a **trailing** (not centered) 1.5s slice of the ring buffer
each stride, runs it through `WakewordTorchBackend`/`WakewordOnnxBackend` (full torch+onnx parity,
mirroring the VCM backend split), and opens/closes the listening period at a fixed default
threshold `DEFAULT_WAKEWORD_THRESHOLD = 0.9` (no FAR/FRR calibration pipeline — explicitly
deferred). Sustained detection while already open restarts the period (same semantics as a
`SpacebarGate` re-press).

**`vcmx-serve-wakeword`** (commit `99c529b`, 2026-09-25): the trained VCM (Option B grammar model)
can now be served either always-listening (`vcmx-serve`, gate=none) or fronted by the trained
wakeword DS-CNN (`vcmx-serve-wakeword` = `--gate wakeword --policy single_period --gate-period 3
--wakeword-backend onnx`). Both inherit the same calibrated threshold/margin/beam-width from the
shared `VCMX_SERVE_*` config block — this is the actual "sequential hardware gate" deployment
shape (wakeword gates the VCM, not both always running; see `PROCESS-OVERVIEW.md`'s diagram).

## Experiments run, with actual numbers

**Wakeword classifier eval** (`out/wakeword/metadata/eval_report.md`, val split, checkpoint
epoch=20):

| label | precision | recall | f1 | support |
|---|---|---|---|---|
| `_wakeword_` | 0.999 | 0.995 | 0.997 | 1,097 |
| `_unknown_` | 0.995 | 0.998 | 0.997 | 1,099 |
| `_silence_` | 0.996 | 1.000 | 0.998 | 270 |

**Benchmark** (`out/wakeword/metadata/wakeword_benchmark.md`, measured on an AMD EPYC 7742 node —
**not** the RPi4/5 target, which doesn't physically exist on this dev node):

| variant | size | p50 latency | p95 latency | process peak RSS |
|---|---|---|---|---|
| fp32 ONNX | 0.119 MB | 0.256 ms | 0.267 ms | 442.7 MB |
| INT8 ONNX (static, val-calibrated) | 0.047 MB (~48.4 KiB) | 0.148 ms | 0.181 ms | 478.7 MB |

INT8 size (~48.4 KiB) falls inside the handoff doc's own cited MLPerf Tiny reference range
(38.6–52.5 KB). `WakewordTorchBackend.wakeword_prob()` latency measured 54–58ms on both backends
— well under the 150ms stride budget.

**Real end-to-end CLI smoke test:** a positive clip opens the gate at t=2.00s (matching that
clip's own VAD-derived `speech_end_s=2.12s`), stays reopened through t=3.00s; a negative clip
produces zero gate-open events for the whole run, on both onnx and torch backends. A later 8-clip
sweep near each clip's `speech_end_s` found peak detection probability ≥0.95 (mostly 1.0) for all
8, confirming the trailing-window design depends on continuous polling, not a lucky single
snapshot (an earlier single-static-snapshot check had shown a false near-miss at 0.89 before this
was understood).

**Licensing consequence of the noise-mixing experiment:** because `background_noise` (ESC-50,
CC-BY-NC-SA-4.0, via Kaggle `mmoreaux/environmental-sound-classification-50`) is additively mixed
into `*_noisy` subsets, **the whole wakeword dataset — and any model trained on it, including the
checked-in checkpoint — is CC-BY-NC-SA-4.0-encumbered**: non-commercial use only, share-alike on
redistribution. This was a deliberate, explicitly-surfaced tradeoff (2026-09-23) against the
alternative of staying Apache-2.0/public-domain-clean by excluding ESC-50 entirely; every
checkpoint/eval report carries a `LICENSE_NOTE` stating this.

## Known limitations/gaps

- No FAR/FRR calibration pipeline for the wakeword gate — fixed threshold (0.9), tunable via
  `--wakeword-threshold` but not data-driven the way the VCM's `chosen_operating_threshold` is
  (see `PROCESS-VCM-MODEL.md`'s margin-calibration history).
- `filipino_speech_corpus`/`youtube_institutional` were considered as extra `_unknown_` sources
  but deliberately excluded (license-unresolved) in favor of CC0-only `common_voice_negative`;
  broader `_unknown_` diversity remains a real, un-taken option.
- `adversaries`/`adversaries_noisy` never get precomputed speech spans — left on a live-VAD-with-
  fallback path because their VAD-empty rate (~45–54%) undercuts the value of precomputing, and a
  misfire there is lower-risk than on `_wakeword_` content.
- The clean/flagged physical QA split (`build_sanitized_dataset.py` adapter) was never wired up
  for the wakeword output layout, even though the QA transcription itself was run and results are
  known good.
- Two deferred low-risk findings from the fetch-positives ticket (unvalidated reuse of a `/tmp`
  scratch clone dir; a few exception types can escape `fetch_positives.py`'s `main()` handler on
  malformed upstream content) — both explicitly deferred by user decision, not fixed.
- Test pass/tech-lead review for the dataset-build feature as a whole never fully ran end-to-end
  per the project's usual dev-flow convention — only the fetch-positives ticket was individually
  security-audited; the test-engineer pass is listed "not started" in the build handoff doc. (The
  later `wakeword-dscnn` and `wakeword-gate` features, by contrast, show full proof-table
  test/regression runs — 5,020 and 5,487 passing respectively.)
- All latency/RSS/size benchmark numbers are estimates from an AMD EPYC dev node, not real RPi 4/5
  hardware — none of this project's benchmark claims (wakeword or VCM) have been measured on the
  actual target device, which doesn't physically exist on the dev node.
