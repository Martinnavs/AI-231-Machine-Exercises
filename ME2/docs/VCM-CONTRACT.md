# VCM Contract

Shared contract for the toy CTC voice-command model (`vcm`) feature. Every
task in this feature (02 dataset/features, 03 grammar decoder, 04/05
training/eval, 07 integration) reads this doc instead of re-deriving any of
the facts below. Source of truth for each section is the module/file named;
if this doc and that module ever disagree, the module wins and this doc is
stale and needs fixing.

## 1. Alphabet (source of truth: `src/me2_voicegen/vcm/alphabet.py`)

29-token CTC alphabet:

| id | token |
|---|---|
| 0 | CTC blank |
| 1-26 | `a`-`z` |
| 27 | ` ` (space) |
| 28 | `'` (apostrophe) |

**Discrepancy, documented, not "fixed":** the original spec text this
project derives from says the alphabet has "32 tokens", but the spec's own
enumeration (blank + 26 letters + space + apostrophe) only lists 29 distinct
symbols. `vcm.alphabet` implements the 29 tokens the spec's enumeration
actually describes. It does **not** pad to 32 with unused/reserved ids —
any later task that needs a 32-wide output layer for some other reason must
say so explicitly and explain why, rather than assuming 32 is this
contract's alphabet size.

`vcm.alphabet` provides `encode(text) -> list[int]`, `decode(ids) -> str`,
and `collapse(ids) -> list[int]` (CTC repeat-collapse then blank-drop, for
raw per-frame argmax sequences — not for `encode`'s output, which never
contains blanks or repeats to collapse).

## 2. Text normalization (source of truth: `src/me2_voicegen/vcm/text.py`)

`normalize_text(text) -> str` guarantees:
- lowercased
- curly apostrophes (`'` U+2018/U+2019, `ʼ` U+02BC) folded to straight `'`
- every other character not in the 29-token alphabet (digits, punctuation,
  non-ASCII letters, symbols) is dropped and replaced by a word boundary
  (space), so adjacent words are never glued together
- whitespace collapsed to single spaces, leading/trailing whitespace
  stripped

Output of `normalize_text` is always encodable by `vcm.alphabet.encode`
unchanged.

## 3. Canonical intent phrases (`INTENT_PHRASES`, in `vcm/optiona/phrases.py`)

20 entries, `<INTENT label>` (as it appears in the `target_commands` bucket
of `test_set/manifest.csv`'s `label` column) -> canonical normalized phrase:

| intent | phrase |
|---|---|
| ALARM | `set alarm` |
| CALL | `call` |
| DIM_DOWN | `dimmer` |
| DIM_UP | `brighter` |
| LIGHT_OFF | `lights off` |
| LIGHT_ON | `lights on` |
| LIST_REMINDERS | `list reminders` |
| MESSAGE | `message` |
| NEXT | `next` |
| PAUSE | `pause` |
| PLAY_MUSIC | `play music` |
| SET_REMINDER | `set reminder` |
| STOP | `stop` |
| TEMP_DOWN | `cooler` |
| TEMP_UP | `warmer` |
| TIME | `time` |
| TIMER | `set timer` |
| VOLUME_DOWN | `volume down` |
| VOLUME_UP | `volume up` |
| WEATHER | `weather` |

Each entry was verified against its own QA report's "expected" column
(`out/conversions/v2/reports/tmp-qa-<INTENT with `_`->`-`>-*.md`) this
session. `tests/test_vcm_text.py` has a `@pytest.mark.slow` drift-guard test
that re-derives this table from those report files on disk and asserts an
exact match against the hardcoded table above — run it whenever the QA
reports change.

## 4. Transcript resolution (source of truth: `vcm.text.resolve_transcript`)

`resolve_transcript(manifest_row) -> str | None` takes one row (dict) of
`out/conversions/v2/test_set/manifest.csv` and joins it back to its own
source dataset's manifest, keyed on `basename(source_relpath)` ==
that manifest's own `filename` column (verified zero misses across all four
sources with real manifests, this session, over the full 2246-row
`test_set/manifest.csv`):

| `source_dataset` | rule |
|---|---|
| `sanitized_clean` (1460 rows) | No manifest exists for this source. `vcm.optiona.phrases.INTENT_PHRASES[label]`. |
| `common_voice_negative` (187 rows) | Has a `transcript` column (per-chunk Whisper transcript). Use `transcript`, **not** `sentence` (that's the whole-clip prompt text, unreliable per-chunk). 10 of 187 rows have an empty transcript — resolves to `""`, not `None`. |
| `youtube_institutional` (218 rows) | Has a `transcript` column. 34 of 218 rows are blank — correctly, these are `ambient`-bucket non-speech rows; resolves to `""`, not `None`. |
| `background_noise` (194 rows) | No transcript column, no speech present. Always resolves to `""`. |
| `filipino_speech_corpus` (187 rows) | Has a `sentence` column, but per decision (B): **all** 187 rows resolve to `None` unconditionally, regardless of whether the row is a whole-clip or `_cNN.wav` chunked row. Excluded from CTC loss entirely; stays in the eval set only as a rejection/false-accept probe. |
| `optionb` (loaded from its own `out/conversions/v2/optionb/manifest.csv`, not `test_set/manifest.csv`) | Has its own `transcript` column already, read **directly off the row passed in — no source-manifest join** (unlike `common_voice_negative`/`youtube_institutional`), then passed through `vcm.optionb.transcript.prepare_ctc_transcript` before returning. `bucket` is one of `target_commands` (93 distinct real command transcripts, `label` = the intent, e.g. `ALARM`/`BRIGHTNESS`/...), `babble` (`label` = `unknown`, `transcript` = `""`), or `silence` (`label` = `silence`, `transcript` = `""`) — all three resolve via the same branch and all resolve to a string, never `None`. |
| `vcm_balanced` (VCM Dataset B, merged in by `vcm.vcmx_merge` into `out/conversions/v2/optionb-v3-vcmx{,-control}/manifest.csv`; not present in `optionb`/`optionb-v3` alone) | Shares the `optionb` branch above verbatim (`source_dataset in ("optionb", "vcm_balanced")`) — its own `transcript` column is already alias-normalized, digit-kept text (`vcm.vcmx_merge.alias_normalize`/`normalize_unknown_text`), so the same `prepare_ctc_transcript` digit-spelling step applies unmodified. `bucket` is one of `target_commands` (B rows whose A-joined transcript is exactly in-grammar for `OPTIONB_GRAMMAR`, kept under the label-mapped intent), `babble` (B `UNKNOWN` rows, `label` = `unknown`), or `silence` (B `SILENCE` rows, `label` = `silence`, `transcript` = `""`) — see `docs/OPTIONB-GRAMMAR-CONTRACT.md` and the `optionb-v3-vcmx` ticket for the full selection/labeling rules. |

Callers must distinguish `""` (real empty transcript / silence — include in
CTC loss as an empty-target sequence) from `None` (excluded from CTC loss
entirely) — they are not interchangeable.

### `optionb` digit handling and where it lives relative to `vcm`

`normalize_text` (section 2) drops digits entirely — they are outside the
29-token CTC alphabet. Option B's real transcripts carry digit-bearing slot
values (`"Alarm 6 AM"`, `"Brightness 100 percent"`), so feeding them through
`normalize_text` unchanged would silently corrupt the target (`"alarm am"`,
`"brightness percent"`). `vcm.optionb.transcript.prepare_ctc_transcript`
spells digit runs out as words first (via `vcm.optionb.numbers.spell_integer`,
which raises `ValueError` above 100 rather than silently mis-spelling), so no
numeric content is lost by the time `normalize_text` runs.

`vcm.optiona` and `vcm.optionb` are sub-packages of `vcm` (the per-dataset-
experiment content for the toy/original dataset and the AI231 MEX2 Option B
dataset respectively, per docs/OPTIONB-GRAMMAR-CONTRACT.md) — `vcm.text` ->
`vcm.optionb.transcript` -> `vcm.optionb.numbers` is therefore an ordinary
parent-package-imports-its-own-child relationship, not a cross-package
dependency edge. The invariant that matters is the reverse: neither
`vcm.optiona` nor `vcm.optionb` may import from `vcm`'s own generic CTC
machinery (`vcm.train`, `vcm.evaluate`, `vcm.model`, `vcm.dataset`, etc.) or
from each other — each experiment package only ever imports from
`common.grammar_core` (dataset/methodology-agnostic) and its own sibling
modules. Duplicating `spell_integer` inside `vcm.text` instead would give two
number-spelling implementations that can drift, so this one-directional
dependency (registry -> its own experiment children) is intentional.

## 5. Log-mel feature contract (Task 02 implements this; not yet built)

- 40 mel bins
- 480-sample analysis window (30ms at the manifest's 16kHz `sample_rate`)
- 160-sample hop (10ms)
- log magnitude with an epsilon floor (no `log(0)`)
- output layout: `(batch, n_mels, frames)`, i.e. `(B, 40, T)`

## 6. Padded-batch convention (Task 02's `collate_fn`; not yet built)

A batch is: `(B, 40, T)` log-mel float tensor (per §5) + `input_lengths`
(per-example valid frame count before padding) + `target_ids` (concatenated
or padded `vcm.alphabet.encode` output per example) + `target_len`
(per-example target token count) — the standard PyTorch CTC-loss batch
shape (`nn.CTCLoss` expects exactly this: padded input + input_lengths +
target + target_lengths).

### Model seam: output time may be shorter than input time

Acoustic models (`vcm.model.CTCAcousticModel`) map `(B, 40, T)` features to
`(B, T', 29)` logits. `T' == T` for MatchboxNet; the strided
`vcm.quartznet.QuartzNetCTC` gives `T' = ceil(T / total_stride)`. CTC input
lengths must come from `model.output_lengths(input_lengths)`, never from the
feature lengths. Checkpoints carry a top-level `model_type` key
(`"matchboxnet"` | `"quartznet"`); an absent key means `matchboxnet`. Anything
that maps a posterior frame index to seconds must multiply by
`10 ms * total_stride` -- use `vcm.pipeline.frame_hop_samples(model)` (waveform
samples per posterior frame = `HOP_LENGTH * total_stride`, MatchboxNet treated
as stride 1) rather than open-coding the factor. The ONNX I/O names stay
`features` -> `logits`; a strided model's output time axis is named `time_out`.

**Optional semantic heads (feature `ctc-attention`).** `QuartzNetConfig(heads=True)`
(preset `quartznet5x3-heads`; default off, so every existing checkpoint and
config dict is unchanged) adds an attention-pooled intent head (21 classes: the
19 grammar intents + `unknown` + `silence`) and one 3-way slot head per slotted
intent on the same encoder output. `forward(features)` still returns only the
CTC logits; `forward_heads(features, input_lengths=None)` returns `HeadsOutput`
(`ctc_logits`, `intent_logits`, `slot_logits`, `attention`), pooling over the
frames valid per `output_lengths`. Label tables are derived from
`OPTIONB_GRAMMAR` in `vcm/semantic_labels.py`. The heads are not part of the
ONNX export yet, and a plain-CTC checkpoint loads into a heads model only with
`strict=False` (the head weights are then freshly initialised).

## 7. Decoder input/output contract (source of truth: `vcm/decoder.py`)

- **Input:** a CTC posterior/log-prob array over the 29-token alphabet
  (§1), shape `(T, 29)` for a single utterance or `(B, T, 29)` for a batch.
- **Output:** a `DecodeResult` — `{intent, slots: dict, text, confidence,
  no_match: bool, out_of_grammar_gap: float}` plus these defaulted
  diagnostics/gate-metadata fields (docs/INCOMPLETE-GRAMMAR-REJECTION.md,
  Step 2/3): `grammar_text` (the selected accepted terminal phrase; may
  differ from the unconstrained greedy `text`; `""` when no terminal),
  `rejection_reason` (`"incomplete_prefix"`, and only when the margin gate
  below is enabled and rejected; otherwise `None`), `incomplete_prefix`
  (the winning designated incomplete prefix, or `None`), `incomplete_gap`
  (`command_raw_score - incomplete_raw_score`, or `None`), and
  `command_raw_score` / `incomplete_raw_score` — raw UNNORMALIZED beam log
  masses (not /T), hence invariant to trailing blank padding. `intent` is
  one of the 20 `INTENT_PHRASES` keys or `None`, `slots` is
  whatever the grammar for that intent extracted (empty dict if none),
  `text` is the raw decoded/collapsed transcript, `confidence` is the
  decoder's own score, `no_match` is `True` when nothing in the grammar
  accepted the decode (`intent` is then `None`).
- **Margin gate (`required_command_margin`, docs/INCOMPLETE-GRAMMAR-
  REJECTION.md Step 3):** `decode`/`decode_utterance` take
  `required_command_margin: float | None = None`, threaded through this
  contract's caller chain (`pipeline.infer_waveform`,
  `SlidingWindowPipeline`, `StreamingRunner`, `vcm.evaluate`'s
  `--required-command-margin` / `decode_split`, and the Makefile variable
  `VCM_EVAL_REQUIRED_COMMAND_MARGIN`). `None` — the default at every
  surface — disables the gate and acceptance is exactly the baseline.
  When set, the strongest designated incomplete-prefix beam (a beam whose
  exact prefix is in `grammar.incomplete_prefixes`) is compared against
  the best completed terminal on raw unnormalized beam log mass: if
  `incomplete_gap < required_command_margin` (strict `<` — a tie passes)
  the result is `no_match` with `rejection_reason = "incomplete_prefix"`.
  The existing confidence threshold stays an independent second gate.
  Grammars with an empty `incomplete_prefixes` (e.g. `SPEC_GRAMMAR`,
  `TOY_GRAMMAR`) are unaffected by the gate.

## 8. Grammar selection in `vcm.evaluate` (source of truth: `vcm/evaluate.py`'s `GRAMMAR_REGISTRY`/`--grammar`)

`evaluate.py --grammar` picks which grammar(s) each manifest's rows are
decoded against, from `{spec, toy, optionb}`:

| key | grammar | label set used in the per-intent table |
|---|---|---|
| `spec` | `vcm.optiona.grammar.SPEC_GRAMMAR` | `vcm.optiona.phrases.INTENT_PHRASES` (all 20, including the 12 SPEC_GRAMMAR has no accepting rule for at all -- see §7's coverage note; unchanged from before this flag existed) |
| `toy` | `vcm.optiona.grammar.TOY_GRAMMAR` | `INTENT_PHRASES` (same as `spec`) |
| `optionb` | `vcm.optionb.grammar.OPTIONB_GRAMMAR` | derived from the grammar itself (`{intent for _, intent, _ in OPTIONB_GRAMMAR.all_phrases()}`, 19 labels) -- **not** `INTENT_PHRASES`, whose label vocabulary is disjoint from Option B's (e.g. `BRIGHTNESS`/`COLOR`/`CREATE_REMINDER`/`TEMPERATURE` vs. `DIM_UP`/`DIM_DOWN`/`SET_REMINDER`/`TEMP_UP`/`TEMP_DOWN`) |

Default is `spec,toy` -- this reproduces the original hardcoded
`[(SPEC_GRAMMAR, "SPEC_GRAMMAR"), (TOY_GRAMMAR, "TOY_GRAMMAR")]` behavior
bit-for-bit; passing `--grammar` at all is opt-in.

**Degenerate-sweep hazard (D5/D11, `.scratch/optionb-dataset/tickets/
04-pipeline-wiring.md`).** `sweep_thresholds`'s false-accept-rate side is
computed over `REJECT_PROBE_BUCKETS = ("babble", "silence")` rows in
*whatever manifest was passed via `--manifest`* -- it is not aware of which
grammar was chosen. `--grammar optionb` only produces a meaningful,
non-degenerate sweep against a manifest that itself carries `babble`/
`silence` reject-probe rows alongside its Option B `target_commands` rows,
which is exactly why `out/conversions/v2/optionb/manifest.csv` embeds the
existing 786 `test_set/` probe rows by reference (D6c, §4's `optionb` row
above) rather than shipping Option-B-only. Running `--grammar optionb`
against a manifest with zero `babble`/`silence` rows makes `reject_idxs`
empty, `false_accept_rate` identically `0.0` at every threshold, and
`choose_operating_threshold` picks purely on `target_accept_rate` -- an
apparently-clean sweep table that is actually uninformative, not a sign the
model has no false accepts. Likewise, running `--grammar spec` or `--grammar
toy` against `out/conversions/v2/optionb/manifest.csv` is not meaningful in
the other direction: Option B's `label` values are never equal to a VCM
`INTENT_PHRASES` key, so `n_exact_correct`/`exact_accuracy` would be ~0 by
construction, the same D11 problem this flag exists to fix, just triggered
by picking the wrong grammar for the manifest instead of having no choice
at all.

`Makefile`'s `optionb-train`/`optionb-eval` targets exist so the correct
manifest+grammar pairing (`OPTIONB_MANIFEST` = `out/conversions/v2/optionb/
manifest.csv`, `--grammar optionb`) is one command rather than a
`VCM_MANIFEST=...` override a caller could accidentally pair with the wrong
`--grammar` (or the default `spec,toy`, silently reproducing the exact D11
failure mode above).

## 9. License provenance (CC-BY-NC-SA-4.0)

`out/conversions/v2/background_noise/` is sourced from ESC-50 and is
licensed CC-BY-NC-SA-4.0 (non-commercial, share-alike). It feeds the
`silence` bucket of `test_set/`. Because it is assembled into the same
`test_set/` used to train/eval any VCM checkpoint, **CC-BY-NC-SA-4.0
governs any checkpoint trained on this data** — non-commercial use only,
and any redistribution of the checkpoint or derived work must be shared
under the same license. Every later report or doc (Tasks 04/05/07) that
references a trained VCM checkpoint must carry this note; do not drop it
just because a given experiment's own training subset happened not to
sample any `background_noise` rows.
