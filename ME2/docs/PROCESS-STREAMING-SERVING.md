# Process: Streaming & Serving

The live/runtime story — the streaming inference loop, listening-gate/acceptance-policy design,
combined VCM+wakeword export and serving (VCMX), and the Raspberry Pi deployment target. See
`PROCESS-OVERVIEW.md` for how this fits into the whole pipeline, and `STREAMING-CONTRACT.md` for
the exact technical contract this doc doesn't repeat.

## Streaming runtime architecture

The live inference stack lives in `me2_voicegen.vcm.streaming`, built as a chain of independently
swappable seams, added incrementally across four commits (`90f2086` → `a249291` → `c4e5587` →
`3612feb`):

- **`AudioSource`** (`sources.py`): `RawPcmStreamSource`, `WavFileSource` (lockstep, no drops —
  for tests/deterministic replay), `MicrophoneSource` (spawns `arecord` via subprocess, threaded
  capture that skips-and-counts when inference falls behind, to avoid ALSA buffer underruns).
- **`RingBuffer`** — holds recent audio; snapshots of it are what both the model and any gate see.
- **`Debouncer`** — pure time-based refractory gate, independent of *why* the previous emission
  fired.
- **`InferenceBackend`** (`backends.py`): `OnnxBackend` (default; applies log-softmax in numpy
  since ONNX emits raw logits) and `TorchBackend` — verified within 1e-2 max-abs-diff of each other
  on real `optionc` artifacts.
- **`AcceptancePolicy`** (`policy.py`) — decides if a decode is "real evidence."
- **`ListeningGate`** (`gate.py`) — bounds a listening period.

**Fixed pipeline order** (deliberately never reordered): `logp → decode(threshold=−inf) →
AcceptancePolicy.observe → Debouncer → TriggerEvent`. Every window decodes at the most permissive
threshold first; the policy and debouncer apply constraints afterward, so diagnostics
(`--log-all-windows`) can see what the decoder actually found.

## Acceptance policies

- **`ThresholdPolicy`** — original, per-window gate: accept iff `intent is not None and confidence
  >= threshold`.
- **`ModePeriodPolicy`** (shipped 2026-09-21, feature `mode-period-gate`) — collects every window
  across a bounded listening period, then flushes one consolidated event: the mode `(intent,
  slots)` class by count, tie-broken by confidence-sum then first-seen order, accepted iff that
  class's mean confidence clears the threshold.

It fixes two failure modes observed directly on real hardware:

1. **Lingering duplicate triggers** — a spoken phrase stays in the 2.5s sliding window across ~10
   consecutive strides, so it decoded correctly enough times to fire twice, exactly `refractory_s`
   (1.5s) apart, from one utterance. Fixed because a period yields at most one accept.
2. **Ambient false positives** (`TIME` triggered at confidence −0.085/−0.09, just inside the −0.1
   threshold, with nobody speaking) — fixed because gate-closed windows are rejected before any
   threshold arithmetic, and noise must additionally win the period's mode *and* clear the
   threshold on its mean.

`SinglePeriodPolicy` is a cheaper alternative: exactly one 3.0s inference per gate-open, using the
normal threshold/debounce — the config used by `stream-single-period`/`stream-wakeword`/
`vcmx-serve-wakeword`.

## Gates

The `ListeningGate` protocol (`poll(samples_seen, window) -> GateState`, `close()`) is the swap
point. Two implementations:

- **`SpacebarGate`** — raw-TTY keypress, manual, the original stand-in.
- **`WakeWordGate`** (shipped `3612feb`) — runs the trained DS-CNN (see `PROCESS-WAKEWORD.md`)
  against the trailing 1.5s slice of the ring-buffer window through the shared
  `LogMelFeatureExtractor`; crossing `DEFAULT_WAKEWORD_THRESHOLD = 0.9` (fixed, not calibrated)
  opens/restarts a period, identical discard-and-restart semantics to a `SpacebarGate` re-press.

Both satisfy `ListeningGate` with **zero changes to `AcceptancePolicy` or `StreamingRunner`** —
proof the seam design worked as intended. `--gate none|spacebar|wakeword`;
`--wakeword-backend {onnx,torch}` (default torch) mirrors the VCM backend split but is
classifier-shaped (`wakeword_prob(waveform) -> float`).

## Serving/export (VCMX)

`vcmx_merge.py` (feature `optionb-v3-vcmx`, commit `3b36a17`) merges Option B (refreshed to live
upstream grammar) with a second balanced dataset ("VCM Dataset B") into speaker-disjoint
**treatment** (25,231 rows) vs. **control** (21,055 rows) manifests with byte-identical val/test,
for a fair before/after training comparison. See `PROCESS-VCM-MODEL.md` for the training/model
side of this same merge.

Makefile pipeline: `optionb-refresh` → `vcmx-build` → (`optionb-train`/`optionb-eval` with
overridden manifest paths) → `vcmx-export` (fp32 + INT8 ONNX) → `vcmx-serve` / `vcmx-serve-wakeword`.

`vcmx-serve` runs `vcm.streaming` against the INT8 export with the calibrated threshold,
incomplete-prefix margin, and beam-width (locked to 50, matching what the margin was calibrated
against) baked in as Makefile defaults. `vcmx-serve-wakeword` is the same but gated by the
wakeword DS-CNN instead of an always-listening mic (`--policy single_period --gate wakeword
--gate-period 3`) — see `PROCESS-WAKEWORD.md` for that integration's own numbers. This realizes
the "sequential hardware gate" design from `20260925_suggestions.md` §4 exactly: `WakeWordGate`
sits upstream of the VCM decode path in the same runner loop, not a separate process.

**fp32 is the streaming default, not INT8**, because the shipped −0.1 operating threshold was
tuned on torch-fp32 logits; INT8 would need its own re-tuning pass against its quantized
confidence distribution.

## Experiments/decisions with real numbers

- **RTF/latency** (this-node CPU estimates, no real Raspberry Pi hardware available): beam search
  over a real 2.5s window decodes in ~83.6ms; beam search is ~97% of per-window compute, the ONNX
  forward pass itself is only ~2.8ms.
- **Backend parity:** ONNX vs. torch logits verified within 1e-2 max-abs-diff on checked-in
  `optionc` artifacts; same intent/slots on the same wav.
- **Padding-sensitivity measurement** (holding utterance constant, varying blank-confidence margin
  during non-speech padding of a 2.5s window):

  | blank-confidence margin | confidence | accepted at −0.1? |
  |---|---|---|
  | 12.0 | −0.0002 | yes |
  | 6.0 | −0.0436 | yes |
  | 3.0 | −0.5670 | **no** |
  | 1.5 | −1.2863 | **no** |

- **VCMX dataset merge real counts:** B in-grammar commands kept 1,933 → 1,891 after
  augmented/non-train-speaker drops; B unknown 2,497 kept; B silence 632 (all forced to train);
  ESC-50 silence 1,633 chunks rebuilt by fold (2 Freesound IDs spanned folds, fixed by majority-fold
  resolution) plus a fix that resolved 44 pre-existing `group_id`-spans-split violations (38
  `filipino_speech_corpus`, 6 `youtube_institutional`) via a val>test>train tie-break. Tech-lead
  review: **approved, no R1/R2**; one R3 finding flagged for a human decision (`optionb-v3`
  val/test babble/silence probes get trained on in the vcmx split, contaminating both arms
  equally, so a secondary eval choice is still needed).
- **Streaming test coverage:** `tests/test_vcm_streaming_policy.py` (both `ModePeriodPolicy`
  backlog acceptance criteria as named tests), `tests/test_vcm_streaming_integration.py`,
  `tests/test_vcm_streaming_backends.py::test_onnx_backend_matches_torch_backend_on_real_optionc_artifacts`.

## Known limitations/gaps

- **No real Raspberry Pi hardware exists on this node** — all latency numbers are AMD EPYC CPU
  estimates, not RPi numbers. This applies to every latency claim in this doc set.
- **Confidence is sensitive to window padding** (table above) — a correctly-decoded phrase can be
  rejected purely due to how much silence surrounds it in the fixed window, because confidence is
  mean per-frame log-prob over the *entire* window.
- **Security:** `--model`/`--config`/`--mic-command` are operator-supplied trust boundaries, not
  sandboxed (`torch.load` pickle deserialization / ONNX parser risk); the pinned `torch==2.3.1` is
  within CVE-2025-32434's affected range (a `weights_only=True` bypass, fixed in 2.6.0) — a
  deliberately separate, backlogged upgrade.
- **Wakeword gate has no FAR/FRR calibration pipeline** (unlike VCM's per-grammar
  `chosen_operating_threshold`) — fixed default threshold 0.9, tunable by hand, explicitly
  deferred.
- **Cold start is non-trivial for either backend** (torch/torchaudio import, ORT session
  construction, 129-phrase trie build, ~0.2s `arecord` open) — the spawn-per-trigger vs.
  signal-a-running-process product decision is explicitly not yet decided.
- Live-mic path verified end-to-end only on this specific dev node; mic/device availability will
  vary elsewhere.
