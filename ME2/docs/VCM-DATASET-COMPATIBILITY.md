# VCM Dataset Compatibility Spec

A checklist for deciding whether a candidate dataset can be pointed at the
existing VCM CTC training/eval pipeline (`me2_voicegen.vcm.{dataset,train,
evaluate,export_onnx,benchmark}`) as a **hot-swap** — i.e. produce a
manifest, set two Makefile variables, and get a trainable/evaluable run
with no code changes. This doc is derived from the pipeline's actual
source (`vcm/dataset.py`, `vcm/text.py`, `vcm/model.py`,
`common/features.py`, `vcm/evaluate.py`, `Makefile`) and from the one
dataset that has actually been run through it end-to-end,
`out/conversions/v2/optionb/manifest.csv` -> `out/vcm/optionb-optiond`
(preset `optiond`, epoch 71 — see that run's
`out/vcm/optionb-optiond/metadata/eval_report.md`). Source of truth for
each requirement is the module named; if this doc and that module ever
disagree, the module wins.

This doc complements, and does not repeat, `docs/VCM-CONTRACT.md` (the
alphabet/feature/decoder contract every dataset shares) and
`docs/OPTIONB-GRAMMAR-CONTRACT.md` (one dataset's own grammar). Read this
doc first to decide *if* a new dataset fits; read those two for *what it
must produce* once it does.

## 0. Two tiers of compatibility

The pipeline has a **hard tier** (§1–§4: manifest schema, audio format,
alphabet, CTC length feasibility) — miss any of these and training either
crashes or silently produces a degenerate model. It also has a **soft
tier** (§5–§6: reject probes, a matching grammar) that isn't required to
train a CTC acoustic model at all, but is required for `vcm.evaluate` to
produce a *meaningful* result rather than a technically-successful but
uninformative one (see VCM-CONTRACT.md §8's "degenerate-sweep hazard").
A dataset can clear the hard tier and still need real engineering work to
clear the soft tier — that work (a grammar module, a `resolve_transcript`
branch) is exactly what `optionb`'s own integration did, and this doc's
§7 explains where those extension points are.

## 1. Manifest schema (hard requirement)

`vcm.dataset.VCMDataset` and `vcm.evaluate` both load a `manifest.csv` via
`csv.DictReader` and index columns by name (order doesn't matter). Every
row must carry these 11 columns — this is the exact column set of
`out/conversions/v2/test_set/manifest.csv` and
`out/conversions/v2/optionb/manifest.csv`, both already in production use:

| column | required by | meaning | constraint |
|---|---|---|---|
| `filename` | (provenance only — no code path reads it) | bare filename on disk | — |
| `path` | `VCMDataset._load_waveform` | audio path, **relative to `audio_root`** (default: `manifest_path.parent`) | must resolve to a readable audio file |
| `bucket` | `vcm.evaluate` (`RowResult.bucket`, `REJECT_PROBE_BUCKETS`, `TARGET_BUCKET`) | eval-time row category | see §5 for the vocabulary this actually needs |
| `label` | `vcm.evaluate` (`confusion_counts`), and `resolve_transcript` for `sanitized_clean` rows | ground-truth intent/class name | free text; only load-bearing where a specific `source_dataset` branch reads it (§2) |
| `duration` | provenance / QA only | seconds, as measured from the on-disk file | float-parseable |
| `sample_rate` | `VCMDataset._load_waveform` | **the rate `torchaudio.functional.resample` targets** | must be `16000` — see §3, this is not "the file's real rate," it's a resample target |
| `resampled` | provenance only | whether the file was resampled to hit the sample_rate above | `"True"`/`"False"` string |
| `source_dataset` | `resolve_transcript` (dispatch key), `VCMDataset` (background-noise pool selection: `row["source_dataset"] == "background_noise"`) | which transcript-resolution rule applies (§2) | must be one of the 6 known values, or a new `resolve_transcript` branch must be added (§7) |
| `source_relpath` | `resolve_transcript` for `common_voice_negative`/`youtube_institutional` (joins on `Path(source_relpath).name`) | provenance path into the upstream corpus | only load-bearing for those two source types |
| `group_id` | `vcm.evaluate.classify_speaker_group` (optional) | speaker/session grouping | optional — absence just means no speaker-group breakdown, not a failure |
| `split` | `VCMDataset.__init__` filter | `train` / `val` / `test` | any other value silently excludes the row from every split-filtered load — not an error, a silent no-op |

Datasets with digit-bearing or otherwise structured transcripts (like
Option B's slot values) add one more column, read directly by
`resolve_transcript`'s dispatch:

| column | required by | meaning |
|---|---|---|
| `transcript` | `resolve_transcript`'s `optionb` branch | ground-truth text for this row, already resolved (no join to an upstream manifest) |

**Failure mode if a column is missing:** `KeyError` at load time
(`row["sample_rate"]`, `row["source_dataset"]`, etc.) — this is a hard
crash, not a graceful skip, for every column above except `group_id`.

## 2. Transcript resolution (hard requirement — this is the actual integration point)

`vcm.dataset.VCMDataset` never reads ground-truth text directly off a row;
every row goes through `vcm.text.resolve_transcript(row) -> str | None`,
keyed on `source_dataset`. This function is a **closed dispatch table**,
not an open one — a `source_dataset` value it doesn't recognize raises
`ValueError: unknown source_dataset`. This is the one hard blocker a
"just add rows to the manifest" hot-swap cannot get around by manifest
content alone:

- A dataset whose rows can honestly reuse an **existing** branch's
  resolution rule (e.g. it already has a `transcript` column with
  per-clip ground truth, like `common_voice_negative`/
  `youtube_institutional`/`optionb`) is a true hot-swap: point
  `source_dataset` at the matching value (or extend the existing branch)
  and it loads with zero code changes.
- A dataset that needs its **own** resolution rule (a new label
  vocabulary, a new digit/number-spelling convention, a new join key) is
  a one-function, additive integration — a new `if source_dataset ==
  "<new>":` branch in `resolve_transcript`, following `optionb`'s own
  pattern (`vcm.optionb.transcript.prepare_ctc_transcript`, §7) — not a
  redesign of `vcm.dataset`/`vcm.train`/`vcm.model`/`vcm.evaluate`
  themselves, none of which know or care what `source_dataset` values
  exist.

Whatever text comes back from `resolve_transcript` (when not `None`) must
survive **the alphabet**, not necessarily unmodified:

- Blank/no-speech rows (silence, negative/babble probes) resolve to
  `""` (empty string) — this is a valid, loss-bearing empty-target CTC
  example, not an exclusion.
- Rows that must be excluded from CTC loss entirely (no reliable
  transcript exists, e.g. `filipino_speech_corpus`) resolve to `None`.
  `None` rows stay loadable by index (for eval-only rejection probes,
  `VCMDataset.__getitem__` returns an empty target for them) but are
  dropped from `VCMDataset.loss_bearing_indices`, which every training
  loop and val-loss computation must iterate instead of `range(len(ds))`.
- `""` and `None` are **not interchangeable** — mixing them up either
  silently drops real training data or corrupts the loss with fake empty
  targets.

## 3. Audio format (hard requirement)

Source of truth: `vcm.dataset.VCMDataset._load_waveform` +
`common.features.LogMelFeatureExtractor`.

- **16 kHz.** `LogMelFeatureExtractor`'s `MelSpectrogram` is constructed
  with a hardcoded `sample_rate=16000` (`common/features.py`) — it never
  reads the manifest's `sample_rate` column and never resamples anything
  itself. The *only* resampling in the whole pipeline happens in
  `VCMDataset._load_waveform`, which resamples the loaded file to
  `int(row["sample_rate"])` — i.e. **the manifest's `sample_rate` column
  must itself be `16000`** for the two to agree; setting it to anything
  else produces features computed at the wrong assumed rate with no
  error raised.
- **Mono.** Multi-channel audio is averaged to mono (`waveform.mean(dim=0)`)
  in both `_load_waveform` and `LogMelFeatureExtractor.forward` — stereo
  input is accepted but silently downmixed, never rejected.
- **Any container/codec `torchaudio.load` supports** (WAV is what every
  existing source uses; nothing in the pipeline assumes WAV specifically).
- **No hard minimum/maximum duration is enforced by the loader** — but see
  §4 for the real constraint duration interacts with.

## 4. CTC length feasibility (hard requirement, dataset-shape-dependent)

Source of truth: `common/features.py` (frame math) + `vcm/train.py`
(`nn.CTCLoss(blank=0, zero_infinity=True)`).

Each clip's frame count is approximately `duration_s * 16000 / 160`
(160-sample hop, §5 of VCM-CONTRACT.md) — e.g. a 1.5s clip yields ~150
frames. `MatchboxNetCTC` never changes the time dimension (stride 1
throughout, per `vcm/model.py`), so this is also the model's output
length. Standard CTC requires **output length ≥ target length**, and
strictly more when the target has adjacent repeated characters (a blank
must separate them) — e.g. `"call"` has one adjacent repeat (`ll`), so it
needs `len("call") + 1 = 5` frames, not 4.

`zero_infinity=True` means a row that violates this constraint does not
crash training — its loss is silently zeroed and it contributes no
gradient. **This degrades silently, not loudly**: a dataset with many
long transcripts paired with very short clips will train "successfully"
while quietly starving on a large fraction of its own rows. There is no
guard rail against this in the pipeline today; a candidate dataset should
be checked (transcript char length vs. frame count, per row) before
assuming a poor-training-curve result is a modeling problem rather than a
data-shape problem.

## 5. Reject-probe buckets (soft requirement — needed for a meaningful eval, not for training)

`vcm.evaluate`'s threshold sweep (`sweep_thresholds`,
`REJECT_PROBE_BUCKETS = ("babble", "silence")`, VCM-CONTRACT.md §8) treats
any row whose `bucket` column is `"babble"` or `"silence"` as a
false-accept probe, and any row whose `bucket` is `"target_commands"` (via
`TARGET_BUCKET`) as a real command. A dataset with:

- **no `babble`/`silence` rows at all** — the sweep still runs, but
  `false_accept_rate` is identically `0.0` at every threshold and
  `choose_operating_threshold` picks purely on `target_accept_rate`. This
  is a **silent degenerate result**, not an error — the resulting
  operating threshold looks clean but has never actually been tested
  against a false trigger. This is exactly why
  `out/conversions/v2/optionb/manifest.csv` embeds `test_set/`'s existing
  786 `babble`/`silence` probe rows by reference rather than shipping
  Option-B-only rows (VCM-CONTRACT.md §8's "degenerate-sweep hazard").
- **`bucket` values outside `{target_commands, babble, silence}`** — those
  rows are simply invisible to every eval computation keyed on `bucket`
  (confusion counts, false-accept stats, speaker-group breakdown); they
  still load and can still contribute to CTC training loss as long as
  `resolve_transcript` resolves them.

A dataset that lacks its own babble/silence rows is still trainable
end-to-end; it just needs to borrow reject probes from an existing source
(as Option B does) or accept an uninformative sweep. This is a design
decision to make explicit in any new run, not a blocking defect.

## 6. Grammar / taxonomy match (soft requirement, closed-world decoding only)

`vcm.evaluate --grammar {spec,toy,optionb}` (`GRAMMAR_REGISTRY`,
VCM-CONTRACT.md §8) decodes each row against a specific `Grammar` object
and reports per-intent accuracy against **that grammar's own label
vocabulary**, not the manifest's `label` column directly. A candidate
dataset's `label`/intent taxonomy is only meaningfully scored if:

- it matches one of the two existing taxonomies verbatim
  (`vcm.optiona.phrases.INTENT_PHRASES`'s 20 intents for `spec`/`toy`, or
  `OPTIONB_GRAMMAR`'s 19 intents for `optionb`) — these two taxonomies are
  **disjoint and never interchangeable** (VCM-CONTRACT.md §8,
  OPTIONB-GRAMMAR-CONTRACT.md §5): running `--grammar spec` against a
  dataset built for the `optionb` taxonomy (or vice versa) makes
  `exact_accuracy` ~0 by construction, not a real accuracy measurement; or
- a brand-new closed-world `Grammar` is authored for the new taxonomy
  (mirroring `vcm/optionb/grammar.py`'s pattern: `alt()`/`seq()`/`slot()`
  building a compiled trie via `common.grammar_core`) and registered in
  `GRAMMAR_REGISTRY`.

A dataset can still train an acoustic model, and even be greedy-decoded
character-by-character (`vcm.train.greedy_decode`), with **no grammar at
all** — the grammar layer is an `evaluate.py`/`decoder.py`/`pipeline.py`
concern, entirely downstream of and decoupled from the CTC model and
training loop. Skipping this section only means skipping closed-world
intent/slot scoring, not skipping training.

## 7. Extension points, concretely (where "one function, additive" actually lives)

If a candidate dataset needs new code rather than pure hot-swap, this is
the complete list of places, based on Option B's own integration:

| what's new | where it goes | what it must NOT touch |
|---|---|---|
| A new `source_dataset` transcript-resolution rule | a new branch in `vcm.text.resolve_transcript`, delegating to a new `vcm.<experiment>.transcript` module if normalization needs dataset-specific logic (digit spelling, slot templating — see `vcm.optionb.transcript.prepare_ctc_transcript`) | must not import from `vcm.train`/`vcm.model`/`vcm.dataset`/`vcm.evaluate`, and must not be imported by another experiment package (VCM-CONTRACT.md §4's one-directional-dependency rule) |
| A dataset-specific normalizer (e.g. one that must preserve digits) | its own `vcm.<experiment>.text.normalize_text`, **not** a change to `vcm.text.normalize_text` (which every other dataset already depends on dropping digits — see OPTIONB-GRAMMAR-CONTRACT.md §3's D2) | the shared 29-token alphabet (`vcm/alphabet.py`) itself — a new normalizer must still emit only `a-z`, space, apostrophe |
| A new closed-world grammar/taxonomy | a new `vcm/<experiment>/grammar.py` + a new key in `evaluate.py`'s `GRAMMAR_REGISTRY` | `vcm.decoder`'s beam-search machinery, which is grammar-agnostic |
| A Makefile hot-swap invocation | new `<EXPERIMENT>_MANIFEST ?=` / `<EXPERIMENT>_OUT_DIR ?=` variables plus `<experiment>-train`/`<experiment>-eval` targets that pin `--grammar <experiment>`, mirroring `optionb-train`/`optionb-eval` (Makefile lines ~110-121) | `vcm-train`/`vcm-eval`'s own defaults — the point of a dedicated target pair is to make the correct manifest+grammar pairing one command, not to change what the generic targets do (VCM-CONTRACT.md §8's rationale for why these targets exist at all) |

Everything else — `vcm/dataset.py`, `vcm/model.py`, `vcm/train.py`,
`common/features.py`, `common/augment.py`, `vcm/decoder.py`,
`vcm/pipeline.py`, `vcm/export_onnx.py`, `vcm/benchmark.py` — is genuinely
dataset-agnostic: none of them contain a `source_dataset`/taxonomy branch,
and none needed to change for Option B's integration.

## 8. Licensing/provenance (hard requirement for any checkpoint distribution)

Any manifest that includes ESC-50-sourced rows (`source_dataset ==
"background_noise"`, CC-BY-NC-SA-4.0) makes **every checkpoint trained on
it** inherit that license — non-commercial use only, share-alike on
redistribution (VCM-CONTRACT.md §9). This is not conditional on a given
training run actually sampling one of those rows; it's a property of the
manifest the run drew from. `vcm.train`'s `LICENSE_NOTE` constant is
hardcoded to name `out/conversions/v2/test_set/`, so a new manifest that
inherits the same license obligation through a different path (e.g. by
embedding `test_set/`'s rows by reference, as `optionb/manifest.csv`
does) needs this note re-verified, not silently reused verbatim by a
copy-paste of the training CLI.

## 9. Quick pass/fail checklist

Given a candidate dataset, in order:

1. **Manifest.** Can it be expressed as CSV rows with the 11 (or 12, with
   `transcript`) columns in §1, with `sample_rate` = 16000 for every row
   after resampling and `path` resolvable from a single `audio_root`?
2. **Transcript resolution.** Does every row's `source_dataset` value
   either already exist in `resolve_transcript`, or come with (or need) a
   new branch per §2/§7? Is `None` vs `""` assigned correctly per row?
3. **CTC feasibility.** For the shortest clips paired with their longest
   transcripts, does frame count clear target length (§4)? Spot-check,
   don't assume.
4. **Eval usefulness (optional but recommended).** Does the dataset carry
   or borrow `babble`/`silence` reject probes (§5)? Does its intent
   taxonomy match an existing `GRAMMAR_REGISTRY` entry, or is a new
   grammar module in scope (§6)?
5. **License.** Does the manifest pull in any `background_noise` (or
   other non-commercial-licensed) rows, directly or by reference (§8)?

A dataset that clears 1–3 is trainable today with `make vcm-train
VCM_MANIFEST=<path>` and zero code changes. Clearing 4 as well (as
`optionb/manifest.csv` does, by embedding `test_set/`'s existing probes
and using its own dedicated `OPTIONB_GRAMMAR`) is what makes `vcm-eval`'s
output a real, non-degenerate accuracy/false-accept measurement rather
than a technically-complete but uninformative report.

## 10. Worked example: Option B against this checklist

`out/conversions/v2/optionb/manifest.csv` (18,443 rows: 17,656
`target_commands`, 561 `babble`, 225 `silence`) against `preset=optiond`
(`out/vcm/optionb-optiond`, epoch 71, `make optionb-train`/`optionb-eval`
equivalent with `OPTIONB_OUT_DIR=out/vcm/optionb-optiond`):

- **§1/§3:** all rows 16 kHz mono WAV, `sample_rate` column = 16000.
- **§2:** `source_dataset == "optionb"` branch added to
  `resolve_transcript`, delegating to
  `vcm.optionb.transcript.prepare_ctc_transcript` for digit-spelling.
- **§4:** not hit in practice — clips are 1.0-2.0s (~100-200 frames),
  transcripts are short phrases; no systematic zero-loss starvation seen.
- **§5:** babble/silence rows are `test_set/`'s existing 786 probes,
  embedded by reference rather than newly recorded, specifically to avoid
  the degenerate zero-false-accept-rate sweep.
- **§6:** `vcm/optionb/grammar.py`'s `OPTIONB_GRAMMAR` (19 intents) +
  `GRAMMAR_REGISTRY["optionb"]`.
- **Result** (test split, chosen operating threshold -0.075):
  1717/1766 `target_commands` accepted (0.972), 1715 exact-intent-correct
  (0.971); false-accept rate 1/56 (0.018) on babble, 0/22 (0.000) on
  silence; 1022/1023 intent-correct slot-bearing clips also had every slot
  value correct. Full breakdown, per-intent confusion, and the val-split
  threshold sweep are in
  `out/vcm/optionb-optiond/metadata/eval_report.md`.

This is the concrete existence proof that a dataset clearing this
checklist's hard tier, plus a modest (~2 file) soft-tier integration
(a `resolve_transcript` branch + a grammar module), reaches production
quality on this pipeline without touching `vcm/dataset.py`,
`vcm/model.py`, `vcm/train.py`, or `common/features.py` at all.
