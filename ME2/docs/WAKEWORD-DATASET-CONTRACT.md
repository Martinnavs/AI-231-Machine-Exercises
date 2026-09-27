# Wakeword Dataset Contract

Shared contract for the wakeword-detector dataset-build feature
(`src/me2_voicegen/wakeword/`). Every task in this feature (this ticket
01, plus 02/03/04/05, plus the parallel "sesame" phrase instance added
2026-09-27) reads this doc instead of re-deriving the schema/taxonomy
below. Source of truth for each section is the module named; if this doc
and that module ever disagree, the module wins and this doc is stale and
needs fixing.

## 1. Scope: one single spoken word per phrase instance

This feature's `_wakeword_` class covers, in any one dataset, **only the
single spoken word of that dataset's phrase instance**. It explicitly does
**not** cover multi-word wake phrases ("hey computer", "hey sesame", ...) —
that is a deliberate scope exclusion (see Established/Non-Goals in ticket
01), not an oversight, and a possible separate follow-up feature later.

The repo now supports **multiple phrase instances** of `_wakeword_`, each
its own parallel dataset under its own root, built by the same unchanged
pipeline modules:

- **`computer`** — `out/conversions/v2/wakeword/`, the original and
  **shipped/production** instance (its checkpoint at `out/wakeword/`,
  `out/wakeword-fil50/`).
- **`sesame`** — `out/conversions/v2/wakeword-sesame/`, a **parallel,
  comparison-only** instance built 2026-09-27 because "computer" has too
  much pronunciation variance as a wakeword. Not a production cutover:
  promoting "sesame" (or any instance) to production is a separate, later
  human decision, same shape as the fil50-checkpoint-promotion precedent.

No later ticket in this feature may add a new phrase instance (or
"hey <word>" or any other multi-word phrase) into any `_wakeword_` class
without this section being updated first as a deliberate decision, not a
silent scope creep. Each new instance gets its own root directory, its own
checkpoint out-dir, and its own row in this table.

## 2. Manifest schema

Every subset produced by this feature (`positives_real/` from this ticket;
later `_unknown_`/converted-positive/`_silence_` subsets from tickets
02/03/05) writes a `manifest.csv` with exactly this column set, in this
order:

| column | meaning |
|---|---|
| `filename` | bare filename of the audio file on disk (e.g. `0386da81-....wav`) |
| `path` | path to the audio file, relative to that subset's own `out_root` (e.g. `audio/picovoice/0386da81-....wav`) |
| `label` | one of `_wakeword_`, `_unknown_`, `_silence_` (section 3) |
| `duration` | seconds, measured from the *final* on-disk file (never assumed/copied from upstream metadata) |
| `sample_rate` | Hz, measured from the *final* on-disk file after any resampling (section 4) |
| `resampled` | `"True"`/`"False"` -- whether this file was resampled from its original rate to hit the section-4 invariant |
| `source_dataset` | which upstream corpus this row came from (e.g. `picovoice`, `mycroft_precise`; tickets 02/04/05 add their own values) |
| `source_relpath` | the file's path relative to its upstream clone root, for provenance |
| `group_id` | see section 5 |
| `split` | `train`/`val`/`test`, or empty string if not yet assigned. **Ticket 04's job**, not this ticket's -- every subset produced before ticket 04 runs leaves this column empty for every row. |

A subset may append additional provenance columns after these 10, when its
own generation process produces information the base schema has no column
for. Two are recognized so far:

- `ref_voice` -- the reference voice used for a voice-converted clip
  (`positives_converted`).
- `noise_source_file` and `snr_db` -- the `background_noise` chunk filename
  mixed in and the SNR (dB) used (any `*_noisy` subset, section 7).

A subset without a reason to populate a given extension column simply
doesn't include it. A ticket-04-assembled final manifest, or any other
merge across subsets (e.g. `consolidate_manifest.py`'s
`generated_manifest.csv`), carries the union of extension columns actually
present across its inputs, `""` where a given row's own subset doesn't
populate one -- it does not invent new columns beyond what some subset's
own manifest already established.

## 3. Class taxonomy

Exactly three label values are valid anywhere in this feature's manifests:

- **`_wakeword_`** -- a real or converted recording of the single word
  "computer" (section 1). This ticket (01) is the first to populate this
  label, from real recorded audio only (no synthetic/converted positives
  yet -- that's ticket 03).
- **`_unknown_`** -- non-wakeword speech: phonetic adversaries (ticket 02)
  and general negative speech drawn from this repo's existing corpora
  (ticket 04; see ticket 01's Established fact 6 for which corpora and
  why).
- **`_silence_`** -- non-speech audio. Sourced only from synthetic
  generation (ticket 05), never from `out/conversions/v2/background_noise`
  (ESC-50-derived) directly as a `_silence_` clip in its own right. Note,
  however, that as of the section 7 addendum below, `background_noise` IS
  used elsewhere in this dataset (mixed additively into `_unknown_`/
  `_wakeword_` clips) -- so the framing that follows a few lines below
  ("this dataset's otherwise-Apache-2.0/public-domain... license basis")
  is now historical, not current: read section 7 before citing this
  dataset's license anywhere.

The leading/trailing underscores on all three label values are intentional
and load-bearing (distinguishes them at a glance from `source_dataset`
values and from the unrelated `label` vocabulary used by the `vcm`/Option B
features in this same repo, which uses bare intent names like `ALARM`).
Never write a bare `wakeword`/`unknown`/`silence` -- always the
underscore-wrapped form.

## 4. Audio format invariant

Every audio file this feature ever writes to `out/conversions/v2/wakeword/`
is 16 kHz, mono, 16-bit PCM WAV, with no exceptions. This is enforced, not
assumed: every file is probed after being written (via Python's `wave`
module) and, if the measured format doesn't match, resampled in place and
re-probed to confirm the resample actually landed the target format before
the manifest row is written. `sample_rate`/`resampled` in the manifest
always describe the file as it actually exists on disk at the moment the
manifest is written, never an upstream-claimed or assumed value.

## 5. `group_id` semantics

`group_id` is the unit that must never be split across `train`/`val`/`test`
by ticket 04's split assignment -- it exists so a single real speaker (or a
converted family of clips derived from one real speaker) can't leak across
partitions.

- **Real positive clips (this ticket, 01):** neither upstream source
  (Picovoice `wake-word-benchmark`, Mycroft `Precise-Community-Data`)
  publishes a speaker-identity mapping (Picovoice's README claims
  "more than 50 distinct speakers" but filenames are bare UUIDs with no
  speaker key; Mycroft's per-contributor waiver files group by *submitter*,
  not by *speaker*, and one submitter's waiver can cover clips from many
  different underlying speakers). Absent a true speaker key, this feature
  treats **each real positive clip as its own group**: `group_id` = that
  clip's own filename stem (the UUID, minus `.wav`). This is a
  known-conservative choice -- it cannot under-group (merge two different
  speakers into one group) but it also can't exploit shared-speaker
  structure that may exist but isn't recoverable from either upstream's
  public metadata.
- **Converted positives (ticket 03):** every synthetic voice-converted
  variant of a given real positive clip inherits that seed clip's
  `group_id` unchanged -- all K converted variants of one real recording
  are one group, so a train/val/test split can never put one speaker's
  converted variants on both sides of a partition boundary.
- **Zero-shot-TTS-only positives (added 2026-09-27 for the "sesame"
  phrase instance; any phrase instance with no upstream real-recording
  corpus):** there is no real source recording to inherit a `group_id`
  from, so the only speaker-like identity axis that exists is the
  **reference voice** whose timbre the TTS voice is cloned from. Rule:
  group by the reference voice used (`group_id` = that reference voice's
  name) -- distinct from `positives_converted`'s "inherit the source
  clip's `group_id`" rule above. In the sesame instance the 329
  QA-passed zero-shot-resynthesis seeds are grouped this way (25 groups),
  while the 816 voice-converted copies of those seeds carry the
  per-seed-unique `group_id` remap that prevented output-path collisions
  in their generation run (see the ticket's Execution Log); both groupings
  are leak-safe under the disjointness guard because the guard keys on
  `group_id`, and the reference voice's timbre legitimately appears
  across splits by this pipeline's existing, accepted design (the same
  35 references convert/synthesize clips in every split in the computer
  instance too).
- **`_unknown_`/`_silence_` rows (tickets 02/04/05):** each sourcing path
  defines its own `group_id` convention appropriate to its own upstream
  corpus's available metadata; this section only binds the `_wakeword_`
  rows this ticket and ticket 03 own.
- **Noise-augmented rows (`adversaries_noisy`/`positives_converted_noisy`,
  section 7):** every noisy copy inherits its source row's `group_id`
  unchanged -- same rationale as converted positives above: a clip and its
  noise-augmented twin must never land on opposite sides of a split.

## 6. Output layout

```
out/conversions/v2/wakeword/                  (gitignored, per repo .gitignore's `out/` rule)
  positives_real/                             (this ticket, 01)
    audio/
      picovoice/<uuid>.wav
      mycroft_precise/<uuid>.wav
    manifest.csv
    summary.md
  adversaries/                                 (ticket 02)
  adversaries_noisy/                           (noise augmentation, section 7)
  positives_converted/                         (ticket 03)
  positives_converted_noisy/                   (noise augmentation, section 7)
  silence_synthetic/                           (ticket 05)
  common_voice_negative_sample/                (build_unknown_external.py --
    scrubbed/sampled real speech, CC0-1.0, broadens `_unknown_` beyond
    `adversaries`' phonetic near-misses)
  generated_manifest.csv, generated_summary.md (consolidate_manifest.py --
    a flat, manifest-only merge of every subset above; NOT the final
    ticket-04-assembled dataset -- see that file's own module docstring)
  manifest.csv, summary.md                     (build_dataset.py -- ticket 04's
    real final assembled dataset: merges positives_real/positives_converted/
    positives_converted_noisy -> `_wakeword_`, adversaries/adversaries_noisy/
    common_voice_negative_sample -> `_unknown_`, silence_synthetic ->
    `_silence_`, then assigns a group-disjoint, per-label-stratified
    70/20/10 train/val/test split)
```

Each subset is self-contained (its own `audio/`, `manifest.csv`,
`summary.md`); `manifest.csv`'s own `path` column is always relative to
that subset's own directory, never to a sibling subset or to
`out/conversions/v2/wakeword/` itself. Ticket 04 is responsible for merging
subsets into one final assembled manifest+split; it is not this ticket's
job to pre-merge anything. `generated_manifest.csv` above is a lighter-
weight, ticket-04-independent convenience merge (path column rewritten
relative to `out/conversions/v2/wakeword/` itself, prefixed by subset name
-- the one exception to the "always relative to that subset's own
directory" rule, since a merged manifest has no single subset directory of
its own).

## 7. Licensing notes (established by this ticket, binding for later tickets referencing these two sources)

- **Picovoice `wake-word-benchmark`:** whole-repo Apache-2.0 (root
  `LICENSE` file).
- **Mycroft `Precise-Community-Data`:** **no repo-root `LICENSE` file**
  (GitHub reports `license: None`). Licensing is per-contributor, via
  `licenses/license-*.txt` waiver files, each a public-domain waiver. Never
  describe this source as simply "public domain" unqualified -- always
  "per-file public-domain waivers, verified N/N, no repo-root LICENSE",
  and record the verified N/N count in the owning ticket's `summary.md`.
  Waiver files enumerate covered files as **repo-root-relative paths**
  (e.g. `computer/en/<uuid>.wav`), under one of two header lines this
  repo's contributors have used interchangeably: `Files released into
  public domain:` or `Files covered by this update:`. A bare-filename
  match (without the `computer/en/` path prefix) will silently
  under-report coverage -- confirmed by this ticket's own execution log,
  which found a false "0/57 covered" result before switching to
  path-prefixed matching, which correctly finds all 57 files covered.
- **`out/conversions/v2/background_noise` (ESC-50, mixed into
  `adversaries_noisy`/`positives_converted_noisy`):** licensed
  **CC-BY-NC-SA-4.0** (NonCommercial + ShareAlike) -- Kaggle
  `mmoreaux/environmental-sound-classification-50`, a 16kHz re-upload of
  academic ESC-50; verified against Kaggle's own `dataset-metadata.json` by
  `background_noise/fetch_kaggle.py`'s hard-coded `EXPECTED_LICENSE` check.
  This section originally excluded this corpus from the wakeword dataset
  entirely for exactly this reason. **As of 2026-09-23, that exclusion was
  deliberately narrowed by explicit user decision**: `background_noise` is
  now additively mixed into a probabilistic sample of `adversaries`/
  `positives_converted` clips (via `mix_background_noise.py`), padding the
  dataset with new `*_noisy` sibling subsets -- originals are never
  modified, and `_silence_` itself is still synthetic-only (section 3), but
  the wakeword dataset **as a whole is no longer cleanly
  Apache-2.0/public-domain**. Consequence, stated the same way
  `vcm/train.py`'s `LICENSE_NOTE` already states it for VCM checkpoints
  trained on this same corpus: **any use, redistribution, or model trained
  on this dataset once `*_noisy` rows are included is CC-BY-NC-SA-4.0-
  encumbered -- non-commercial use only, and any redistribution (of the
  dataset or of anything trained on it) must be shared under the same
  license.** This is recorded on every `*_noisy` subset's own `summary.md`
  and in `generated_summary.md` whenever a `*_noisy` subset is present in
  that consolidation, not just here.
- **`out/conversions/v2/common_voice_negative` (sampled into
  `common_voice_negative_sample`, `_unknown_` class):** licensed
  **CC0-1.0** ("Creative Commons 0, No Rights Reserved"), verified against
  both Kaggle's dataset metadata and the archive's own `LICENSE.txt`
  (`out/conversions/v2/README.md` section 4). No encumbrance of its own --
  it broadens `_unknown_`'s diversity beyond `adversaries`' phonetic
  near-misses without adding any new license obligation. Every row is
  re-scrubbed by `build_unknown_external.py` for its instance's wakeword
  token (case-insensitive, word-boundary, catches the bare word plus its
  plural/possessive forms -- "computer"/"computers"/"computer's" for the
  computer instance, "sesame"/"sesames"/"sesame's" for the sesame
  instance, via `--scrub-word`) before sampling -- this corpus had
  previously only ever been scrubbed against the 20 VCM command phrases,
  never against either wakeword token itself. (Recorded 2026-09-27: the
  sesame scrub found zero matching rows in the 28,186-row corpus.)
- **Pure in-repo TTS / voice-conversion output (stated explicitly
  2026-09-27 for the "sesame" phrase instance, whose positives and
  adversaries are 100% in-repo synthetic):** synthesis with this repo's
  own `cosyvoice2` backend (zero-shot `synthesize` for the 329 sesame
  resynthesis seeds and 315 sesame adversaries, `convert_voice` for the
  816 sesame converted copies) redistributes nothing from any upstream
  corpus -- the audio is generated locally, so **it introduces no new
  license entry and no new encumbrance of its own**: same posture as the
  pre-existing `adversaries_tts`/`fil50_persona` synthetic content in the
  computer instance. The only licensing consequence for a phrase instance
  remains the `*_noisy` rows' CC-BY-NC-SA-4.0 (ESC-50) mix-in described
  above, which governs the whole dataset once those rows are included.

## 8. `speech_start_s`/`speech_end_s` extension columns (feature `wakeword-dscnn`)

Written once, offline, by `wakeword/derive_speech_spans.py`
(feature-engineering/wakeword-dscnn/SPEC.md) -- not by any dataset-build
ticket above, but recorded here per this doc's own extension-column
mechanism (section 2) since it's dataset-level provenance, not
training-feature trivia. Present on `positives_real`,
`common_voice_negative_sample` (VAD run directly, each is its own
ancestor) and propagated onto `positives_converted`/
`positives_converted_noisy` via `group_id`/`(group_id, ref_voice)` joins.
**Absent everywhere else**, including `adversaries`/`adversaries_noisy` --
intentionally, not an oversight (see below). `build_dataset.py`'s
`EXTENSION_FIELDS` must keep these two names whitelisted or the final
assembled `manifest.csv` silently drops them on the next rebuild.

Two facts this section records so later features don't have to re-derive
them:

- **Voice conversion (`synthesizer.convert_voice`) preserves clip timing
  closely**: measured across all 3,744 `positives_converted` rows joined
  to their `positives_real` source via `group_id` -- duration drift mean
  11ms, max 64ms, zero rows over 100ms.
- **Additive noise mixing (`apply_noise`, used identically by
  `mix_background_noise.py` for both `positives_converted_noisy` and
  `adversaries_noisy`) does not shift clip timing/content at all**:
  measured across all 1,280 `positives_converted_noisy` rows joined to
  their immediate `positives_converted` parent via `(group_id, ref_voice)`
  -- **exactly 0.0s duration drift for every row**. A span detected before
  noise mixing remains exactly valid after it.
- **`adversaries`/`adversaries_noisy` are excluded from this precompute
  pipeline by design**, not oversight: measured VAD-empty (fallback-to-
  center-crop) rates on the real dataset are `positives_real` 5%,
  `adversaries` 54.3%, `adversaries_noisy` 45.3%, `common_voice_negative_sample`
  53% -- noise is not the dominant driver of VAD misses on the
  `adversaries` pair (its clean parent already misses ~54% of the time),
  so precomputing would not have bought materially cleaner anchoring
  there. Those two subsets are left on `WakewordDataset`'s live-VAD-with-
  fallback path at train time instead.
