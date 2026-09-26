# Process: Wakeword DS-CNN

The "computer" keyword-spotting pipeline end to end — dataset build, model, training, benchmark,
and its integration into the streaming `ListeningGate` seam. See `PROCESS-OVERVIEW.md` for how
this fits into the whole pipeline, and `WAKEWORD-DATASET-CONTRACT.md` for the exact schema/
licensing contract this doc doesn't repeat.

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
