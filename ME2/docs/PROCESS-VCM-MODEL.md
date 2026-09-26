# Process: VCM Model — Architecture, Training, Grammar & Decode

The Voice Command Model (VCM): the CTC acoustic model, the grammar it's constrained to, and every
rejection-gating experiment run against it. See `PROCESS-OVERVIEW.md` for how this fits into the
whole pipeline, and `VCM-CONTRACT.md` / `OPTIONB-GRAMMAR-CONTRACT.md` for the exact technical
contract this doc doesn't repeat.

## Architecture

`MatchboxNet`-style 1D time-channel-separable-conv CTC network (`src/me2_voicegen/vcm/model.py`),
outputting per-frame logits over a 29-token character alphabet (blank + a–z + space + apostrophe;
the original spec text said "32 tokens" — its own enumeration lists 29, a documented discrepancy,
not a padding bug).

| preset | params | status |
|---|---|---|
| `default` | ~254K | toy-scale; the one actually trained for Option A |
| `spec-scale` | ~2.1M | instantiated/sized only, never trained |
| `optionc` | 192ch | intermediate preset |
| `optiond` | ~1.01M (5 TCS blocks, 128 channels, 64ch prologue, 224ch epilogue) | **the one actually shipped/trained for Option B and everything downstream** (VCMX, calibration, INT8 serving) |

The same model code is reused for both Option A (toy, 20-intent) and Option B (real grammar,
19-intent) — only the preset/manifest/grammar differ, not the architecture code.

The decoder (`vcm/decoder.py`, `common/grammar_core.py`,
`vcm/optiona/grammar.py`/`vcm/optionb/grammar.py`) is a hand-written Python grammar/trie plus a CTC
**prefix beam search** — deliberately not an FST and not greedy-only (`kaldifst` was available but
unused). Two independent grammar families exist and are never interchangeable: `vcm.optiona`
(20 intents) and `vcm.optionb` (19 intents, different taxonomy — e.g. no `DIM_UP`/`DOWN` split, has
`COLOR`, `CREATE_REMINDER` not `SET_REMINDER`). Both share `grammar_core.py`, factored out
specifically to make this reuse possible.

## Training recipe

Confirmed against `src/me2_voicegen/vcm/train.py` (lines ~175–320), not just docs:

| hyperparameter | value |
|---|---|
| optimizer | AdamW, lr=1e-4, weight_decay=1e-2 |
| grad clipping | `clip_grad_norm_(max_norm=5.0)` |
| loss | `nn.CTCLoss(blank=0, zero_infinity=True)` |
| schedule | OneCycleLR |
| batch size | 16 |
| precision | mixed (`torch.cuda.amp`) |
| `optiond` config | 5 residual (TCS) blocks, 128 channels |

Also applied: on-the-fly RIR convolution, additive noise, SpecAugment (same `train.py` code path
for both options).

This recipe is a **complete, already-implemented match** to `20260925_suggestions.md` §2's
"Hyperparameter Lock" — that section of the proposal doc is documenting an existing, already-used
recipe, not a future change.

## Grammar → trie → decode pipeline, and rejection gates

- **CFG → trie:** a hand-written combinator grammar (`literal`/`seq`/`alt`/`slot`/`with_intent`
  building word-sequence tuples) compiled by `compile_grammar` into a character trie
  (`Grammar.accepts`). Word-level rules, character-level acceptance — deliberately separated so a
  future phoneme-level lexicon could replace only the compile step (architecturally reserved, not
  built — see `OPTIONB-GRAMMAR-CONTRACT.md` §7).
- **Decode:** grammar-constrained CTC prefix beam search keeps multiple grammar-reachable
  prefixes. A separate unconstrained greedy pass (`_greedy_unconstrained`) exists today only as a
  diagnostic field (`DecodeResult.text`) — **not** promoted to an active rejection gate.
- **Current acceptance rule:** largest `BeamEntry.total()/T` among terminal beams, then a single
  global confidence threshold.

### What's implemented vs. still just proposed (`20260925_suggestions.md` §3)

| proposal | status |
|---|---|
| Incomplete-prefix margin gate | **Implemented and shipped** — see below |
| Unconstrained greedy gating (Δ lexical-intrusion defense) | **Not implemented** — diagnostic only |
| Dense phonetic scoring (score over non-blank frames only) | **Not implemented** |
| Intent-length-dependent thresholding (per-intent thresholds) | **Not implemented** — single global threshold still in use |

Confirmed by grep: `grep -rln "dense_phonetic\|unconstrained_greedy_gate\|intent.length.dependent\|per_intent_threshold" src/` → zero hits.

### Incomplete-prefix margin gate (implemented)

`Grammar.incomplete_prefixes: frozenset[str]` (whole-word, non-accepted proper prefixes of every
accepted phrase) is compared against the best completed terminal on **raw unnormalized beam log
mass** — deliberately not `/T`, because `/T` normalization lets trailing high-confidence blank
frames dilute/mask weak character activations (the exact problem §3's not-yet-built "dense
phonetic scoring" proposal is also trying to solve, via a different mechanism).
`required_command_margin: float | None` threads through `decode` → `decode_utterance` →
`pipeline.infer_waveform` → `SlidingWindowPipeline` → `StreamingRunner` → `evaluate.py`
(`VCM_EVAL_REQUIRED_COMMAND_MARGIN`) → Makefile (`VCMX_SERVE_MARGIN`). `None` everywhere by default
= gate disabled = baseline behavior.

## Experiments, in chronological order, with real numbers

### 1. Option A toy VCM training/eval (`default` preset, ~254K params)

- Val CTC loss: 105.34 → **1.04** (best, epoch 165/225, ~18 min).
- Greedy decode: 290/292 (99.3%) non-empty, 194/292 (66.4%) exact match.
- Grammar-constrained eval: **TOY_GRAMMAR 91.1%** exact-match (133/146); **SPEC_GRAMMAR 30.8%**
  (45/146 — a grammar-coverage artifact: 10 of 12 "failing" intents have a real rule that needs a
  different slot/phrasing than the dataset's bare phrase, not a model failure).
- 0% false-accept on babble/silence at chosen thresholds; TOY_GRAMMAR's FAR climbs to 42.7% past
  threshold ≈ −0.4.
- ONNX/INT8 benchmark (AMD EPYC 7742, **not** RPi hardware): fp32 1,020,103 bytes / 1.065ms p50;
  INT8 277,219 bytes / 1.039ms p50.

### 2. Failure discovery — incomplete-prefix false accepts (`docs/INCOMPLETE-GRAMMAR-REJECTION.md`)

Motivating example: saying only "color" could decode as `TIME`. Root cause, diagnosed as two
independent bugs:

- Strong nonterminal beams (e.g. "color" at raw mass ≈ −1.953) get discarded because only terminal
  nodes are eligible, letting a much weaker terminal ("time" at ≈ −6.953) win.
- `/T` normalization lets trailing blank frames make a fixed weak raw score look progressively more
  confident as window length grows (demonstrated across T=44→251 frames flipping accept/reject at
  threshold −0.1 with *constant* raw "time" mass).

Beam width (25/50/1000) does not affect this — beam pruning was ruled out as the cause.

### 3. BatchNorm-poisoning fix during Option B dataset training (2026-09-19)

Non-finite training batches were poisoning running BatchNorm stats even though the
backward/optimizer step was skipped — every downstream val batch could go NaN. Fixed by
snapshotting BN running stats before each forward pass and restoring them if the loss is
non-finite (`snapshot_batchnorm_stats`/`restore_batchnorm_stats` in `train.py`).

### 4. Option B refresh to live upstream grammar + VCM Dataset B merge (commit `3b36a17`, 2026-09-24)

Re-fetched Option B at upstream `b9d86ea` (7 of 19 intent rules changed wording), rebuilt
`OPTIONB_GRAMMAR` (canonical-93/accepted-129 structure unchanged, 9 strict-prefix pairs
re-derived), merged a new balanced dataset ("VCM Dataset B") into one pool.

Real yields: 1,933/12,136 B command rows in-grammar (135 speakers; 956 of them "next song"), 2,500
B unknown/babble rows, 632 B silence rows, 1,633 ESC-50 silence chunks (fold-based; a cross-fold
Freesound-ID leak was found and fixed, affecting 2/1,633 chunks). Final treatment manifest: 25,231
rows; control (no B train): 21,055 rows; val/test byte-identical across both.

A pre-existing 44-row `group_id` split-leakage bug in inherited `test_set/` probe rows was also
found and fixed during this pass, not scoped around.

Tech-lead review: **approved, no R1/R2**, 6 non-blocking findings (R3–R5) logged.

### 5. Incomplete-prefix margin calibration — three iterations

- **First real val run (2026-09-23):** chosen margin **5.0**, but flagged as not the true optimum
  (FAR still strictly decreasing at the grid's edge, accuracy headroom remaining) — reported to
  the user rather than shipped; grid/regression-budget made configurable; re-run.
- **Commit `33530af` (2026-09-25):** margin **6.0** chosen against the pooled `optionb-v3-vcmx`
  manifest (~90% voice-cloned/synthetic). Val: FAR 10.80% → 3.73% (excl. digital-zero padding),
  ~0.7pp accuracy cost. Held-out test: FAR 14.65% → 6.46%, target exact accuracy 97.92% → 97.04%
  (0.88pp drop), gate-caused FRR 0.95%.
- **Commit `7885c07` (same day, recalibration):** discovered the pooled-manifest check was
  dominated by synthetic audio — splitting by source showed real (`vcm_balanced`) commands take a
  much larger accuracy hit than synthetic (2.33pp real vs. 0.78pp synthetic at margin=6.0). Rerun
  scored against real recordings only: margin=6.0 costs 1.54pp real accuracy — already over the
  1pp budget, so 6.0 was never actually safe. **Real-only selection picks margin=4.0**: FAR
  11.06% → 6.73%, target exact accuracy 88.37% → 86.82% (1.55pp), gate-caused FRR 1.75%.
  `VCMX_SERVE_MARGIN` updated 6.0 → 4.0.

  Open caveat, still documented in the Makefile: the FAR side of this number is still measured
  against synthetic-only incomplete-prefix probes (Dataset B doesn't have enough real coverage of
  the designated prefixes) — only the accuracy/regression-budget side is now real-audio-validated.

  A subtlety surfaced during this work, worth keeping in mind for any future re-calibration:
  digital-zero silence padding (used when no quiet lead-in/tail exists — ~35% of probes)
  measurably biases the false-accept rate low relative to room-tone padding (flipped 19% of accept
  decisions in a reviewer reproduction) — the selection metric was changed to exclude
  digital-zero-padded probes specifically because of this.

## Known limitations/gaps

- Option A toy VCM: slot extraction never tested on real audio (ready-to-run, needs real persona
  reference wavs, not abandoned).
- No Raspberry Pi hardware exists anywhere in this project — every latency/memory number here is a
  same-architecture-class CPU estimate, never a real on-device measurement.
- `$CMD_TEMPERATURE` has no up/down direction in the original BNF spec — a spec gap, not a decoder
  bug.
- Option A training data is ~45 min / 20 fixed phrases — explicitly memorization-scale, not
  generalization evidence.
- Incomplete-prefix rejection Step 6 (segment-local scoring as a secondary defense) remains
  unbuilt/deferred, explicitly gated behind "only if residual failures remain after the targeted
  fix is evaluated."
- The incomplete-prefix probe set is Option-B-clip-only (cropped/padded from existing recordings
  via forced alignment, not fully organic incomplete utterances) — a permanent measurement
  caveat, not a bug to fix.
- Both Option A/B checkpoints, and anything trained on `test_set/`/ESC-50-derived silence, inherit
  CC-BY-NC-SA-4.0 (non-commercial, share-alike) from ESC-50.

## Open items — proposed but not yet implemented

From `20260925_suggestions.md` §3, still open:

- Unconstrained greedy gating (the Δ lexical-intrusion active rejection gate) — currently a
  diagnostic only.
- Dense phonetic scoring (non-blank-frame-only confidence) — the incomplete-prefix gate's raw-mass
  comparison is a different, already-shipped mitigation for a related but distinct problem
  (nonterminal-vs-terminal comparison, not confidence normalization).
- Intent-length-dependent thresholding — still one global threshold.

Section 1's specific 50/50 Filipino/non-Filipino dataset-balancing plan is the `accent-balance-fil50`
feature — see `PROCESS-DATA-GENERATION.md` for its status (complete, all pre-committed criteria
met) and `MLOPS-PROJECTS.md` for full results.
