# Incomplete Grammar Rejection: Design and Agent Handoff

Status: Steps 1–4 implemented locally — grammar metadata merged into the
production `Grammar`, decoder diagnostics plus the parameterized
`required_command_margin` gate (disabled by default; `None` = baseline),
and evaluate/streaming/Makefile plumbing, all behind a deterministic no-WAV
regression suite. Step 5 (cluster margin calibration + held-out reporting)
remains, on the cluster.

This document explains the observed false acceptance of incomplete Option B
commands, records a deterministic reproduction, and gives the next agent an
implementation and evaluation plan. The motivating example is a speaker saying
only `color`, after which the decoder can return the valid but acoustically
unsupported `TIME` intent.

The proposed fix is inference-only. It does **not** require CTC retraining,
changes to training transcripts, a new checkpoint, or ONNX re-export. The
Option B training data is needed only to build a representative held-out
evaluation/calibration set on the cluster.

## Next agent: start here

Steps 1–4 are implemented (ticket logs: `.scratch/
incomplete-grammar-rejection/tickets/`): the production `Grammar` carries
`incomplete_prefixes`, and `decode`/`decode_utterance` (plus every caller
in VCM-CONTRACT.md section 7's chain) take `required_command_margin`,
default `None` = gate disabled = baseline acceptance unchanged. Continue at
**Step 5** (cluster margin calibration + held-out reporting); do not rebuild
Steps 1–4.

1. Read `incomplete_prefix_grammar.py`, `decoder.py`, and the Candidate
   comparison section below.
2. Run the focused grammar tests to establish a green baseline:

   ```bash
   .venv/bin/pytest tests/test_optionb_incomplete_prefix_grammar.py tests/test_optionb_grammar.py
   ```

3. Continue at **Step 5**: move evaluation to the cluster only after local
   unit tests pass; discover the real dataset through `out/` and the
   resolved `optionb-train` command as described in Step 5.

The handoff is successful when a fresh agent can identify Step 5 as the first
unfinished action, preserve the accepted grammar unchanged, and avoid assuming
that local audio files exist.

## Outcome sought

When speech strongly supports a whole-word, incomplete prefix of a command,
the decoder should return `no_match` instead of choosing a weaker completed
command. For example:

| Speech evidence | Desired result |
|---|---|
| `color` | `no_match`, reason `incomplete_prefix` |
| `color red` | `COLOR`, `{"COLOR": "red"}` |
| `time` | `TIME` |
| `pause` | `PAUSE` |

`color` must remain absent from the accepted grammar. It is a rejection
competitor, not a new command or training label.

Completion means all of the following are true:

1. Incomplete word-boundary prefixes are derived from the active grammar and
   are never returned as intents.
2. The decoder compares the strongest completed command with the strongest
   designated incomplete-prefix hypothesis using unnormalized beam log mass.
3. The deterministic `color`-versus-`time` regression rejects consistently
   across different trailing-blank durations and practical beam widths.
4. Existing accepted commands, especially commands that are also prefixes of
   longer commands such as `time` and `pause`, still decode normally.
5. The new margin is calibrated on validation data and checked once on a
   separate held-out set from the cluster's actual Option B data.
6. No training or model-export artifact changes as part of this fix.

## Source-of-truth map

Read these files before changing code. Function and type names are more stable
than line numbers; use `rg` to locate them in the current checkout.

| Concern | Source of truth |
|---|---|
| Grammar trie and `Grammar` type | `src/me2_voicegen/common/grammar_core.py` |
| Option B accepted phrases | `src/me2_voicegen/vcm/optionb/grammar.py` |
| Experimental incomplete-prefix metadata | `src/me2_voicegen/vcm/optionb/incomplete_prefix_grammar.py` |
| Prefix beam search and final selection | `src/me2_voicegen/vcm/decoder.py` |
| Forced alignment and segment-local scores | `src/me2_voicegen/vcm/segment_scorer.py` |
| Runtime/streaming decoder configuration | `src/me2_voicegen/vcm/stream.py` and its callers |
| Evaluation and threshold sweep | `src/me2_voicegen/vcm/evaluate.py` |
| Grammar invariants and reject examples | `tests/test_optionb_grammar.py` |
| Shared model/data/decoder contracts | `docs/VCM-CONTRACT.md` |
| Option B grammar contract | `docs/OPTIONB-GRAMMAR-CONTRACT.md` |
| Actual training invocation | `Makefile`, target `optionb-train` |

Do not treat copied defaults in this document as authoritative. Inspect the
current code, Makefile, command expansion, and cluster artifacts first.

## What the decoder currently does

The production path is a grammar-constrained CTC **prefix beam search**. It is
not greedy decoding:

- `prefix_beam_search` keeps multiple grammar-reachable prefixes and their CTC
  blank/non-blank masses.
- `_greedy_unconstrained` only supplies the diagnostic `DecodeResult.text` and
  an out-of-grammar baseline.
- `decode_utterance` examines the final beam but skips every entry whose trie
  node has no terminal intent.
- Among terminal entries, it chooses the largest `BeamEntry.total() / T`, then
  applies the configured confidence threshold.

The observed failure therefore is not caused by a greedy decoder and was not
fixed by increasing beam width. It comes from final candidate selection and
score normalization.

### Failure mechanism 1: strong incomplete prefixes are discarded

For speech supporting `color`, the beam may correctly rank the `color` prefix
first. That trie node is nonterminal because every valid color command needs a
slot value, so final selection ignores it. A much weaker terminal path such as
`time` can then win merely because it is complete.

In a deterministic synthetic reproduction, the final beam at 151 frames had:

| Prefix | Kind | Raw beam log mass |
|---|---|---:|
| `color` | nonterminal | about `-1.953` |
| `time` | terminal (`TIME`) | about `-6.953` |

The acoustics favored `color` by roughly 5 log units, but only `time` was
eligible for output.

### Failure mechanism 2: `/ T` rewards unrelated trailing blank frames

The current command confidence is terminal beam log mass divided by the full
number of posterior frames. In the synthetic case, `time` had nearly constant
raw mass while high-confidence trailing blank frames made its normalized score
look progressively better:

| Frames (`T`) | Selected result at threshold `-0.1` | Confidence | Raw `time` mass |
|---:|---|---:|---:|
| 44 | rejected | about `-0.158` | about `-6.946` |
| 68 | rejected | about `-0.102` | about `-6.948` |
| 69 | rejected | about `-0.101` | about `-6.948` |
| 70 | `TIME` | about `-0.099` | about `-6.948` |
| 80 | `TIME` | about `-0.087` | about `-6.948` |
| 151 | `TIME` | about `-0.046` | about `-6.953` |
| 251 | `TIME` | about `-0.028` | about `-6.959` |

Beam widths 25, 50, and 1000 gave the same behavior. This rules out practical
beam pruning as the cause in this reproduction.

`segment_scorer.py` already contains duration-insensitive diagnostics. Its
`segment_path_mean` remained approximately `-1.408` for the synthetic broken
candidate at 70, 151, and 251 frames. However, earlier small-sample calibration
showed lower positive-command recall for a segment-score threshold than for the
existing decode confidence. Segment score should therefore be evaluated as a
secondary safeguard, not substituted blindly for the current threshold.

## Recommended fix: designated incomplete-prefix competitors

Derive every proper, whole-word prefix of every accepted phrase, excluding any
prefix that is itself an accepted phrase. Examples include:

- `color`
- `change the lights to`
- `set the lights to`
- `brightness`
- `set the brightness to`
- `alarm`
- `set an alarm for`

Only whole-word prefixes belong in this set. Character fragments such as
`colo` or `lights o` do not. Restricting the set is important: a naïve rule
that rejects whenever *any* nonterminal beam beats the best terminal rejected
many valid commands in a small probe.

Accepted commands must also be excluded even when they prefix longer commands.
Option B has strict-prefix relationships; examples include `time` and `pause`.
Those must retain their existing intent semantics. Preventing an early
streaming emission while a user continues into a longer phrase is an
endpointing/stability problem and is outside this fix.

### Prefix derivation

The derivation should be deterministic and grammar-generic. Equivalent logic:

```python
accepted = {text for text, _, _ in grammar.all_phrases()}
incomplete = set()
for text in accepted:
    words = text.split()
    for count in range(1, len(words)):
        prefix = " ".join(words[:count])
        if prefix not in accepted:
            incomplete.add(prefix)
```

The first experiment is isolated in
`vcm.optionb.incomplete_prefix_grammar.IncompletePrefixRejectionGrammar`. It
derives immutable rejection metadata from `Grammar.all_phrases()` and validates
that no accepted command is marked incomplete. It deliberately does not change
the shared `Grammar` type or `OPTIONB_GRAMMAR`, and it must not be exported as a
normal accepted grammar.

After model and decoder validation, the preferred merge is immutable metadata
such as `Grammar.incomplete_prefixes: frozenset[str]`, with a default that
preserves existing constructors. Do not attach a fake intent to these trie
nodes and do not make `Grammar.accepts("color")` succeed.

### Candidate comparison

After prefix beam search, retain both:

- the strongest completed command terminal; and
- the strongest beam whose exact prefix is in `incomplete_prefixes`.

Compare `BeamEntry.total()` values, not values divided by `T`. Both candidates
come from the same posterior window, so common high-confidence blank padding
largely cancels in the raw-score difference.

Define a diagnostic gap with an unambiguous sign:

```text
incomplete_gap = best_command_raw - best_incomplete_raw
```

Positive values favor the completed command. Reject when the gap does not meet
the calibrated command advantage:

```python
if best_incomplete is not None:
    incomplete_gap = best_command.total() - best_incomplete.total()
    if incomplete_gap < required_command_margin:
        return no_match(reason="incomplete_prefix")
```

A zero margin is a sensible experiment baseline: the synthetic example has a
gap near `-5`, so it rejects decisively. Do not ship a margin based only on this
one reproduction. Sweep it on cluster validation data.

The recommended decision order is:

1. If there is no completed terminal, return `no_match` as today; record an
   incomplete-prefix reason when applicable.
2. If a designated incomplete prefix beats the completed command under the
   calibrated margin, return `no_match` with reason `incomplete_prefix`.
3. Otherwise apply the existing completed-command confidence threshold.
4. Return the accepted intent and slots only after both gates pass.

### Result diagnostics

Expose enough data to evaluate this without rerunning beam search. Candidate
fields, all optional/defaulted to minimize call-site breakage, are:

```text
grammar_text
rejection_reason
incomplete_prefix
incomplete_gap
command_raw_score
incomplete_raw_score
```

`grammar_text` is the selected terminal phrase, which is not necessarily the
unconstrained greedy `text`. The current segment-scoring spike reruns beam
search because `DecodeResult` does not expose this phrase; avoid that duplicate
work in production.

Thread the required margin through the same explicit configuration path used
by evaluation and streaming. Avoid a decoder-only magic constant. During the
experiment, a disabled state such as `None` can preserve baseline behavior,
while an explicit numeric value enables the gate.

## Approaches not recommended as the primary fix

### Increasing beam width

The reproduction behaves the same at widths 25, 50, and 1000. More beams add
latency and memory without changing the terminal-only final selection rule.

### Rejecting whenever any nonterminal outranks the terminal

This is too broad. Normal CTC beams often contain strong character fragments
of valid commands. In a small real-audio probe, only 6 of 16 finite valid
examples had the terminal above every nonterminal, so a universal veto would
damage command recall.

### Hard-gating on greedy-text edit similarity

Valid paused or noisy commands can have poor unconstrained greedy text. Earlier
examples produced similarities around `0.286` and `0.571`. The greedy string is
useful diagnostics but is not a reliable acceptance oracle.

### Replacing confidence with `segment_path_mean` without recalibration

Segment-local scoring solves the blank-padding invariance problem, but an
earlier limited scorer comparison found roughly 74.1% positive recall at its
zero-false-accept threshold versus roughly 87.0% for decode confidence. These
numbers are historical observations, not a benchmark: reproduce them against
the cluster data before making policy decisions.

### Retraining for this grammar behavior

The model predicts CTC characters; the grammar and acceptance policy run after
those predictions. Adding incomplete-prefix rejection therefore does not
change CTC targets or model parameters. Retraining is optional future work only
if broader evaluation shows that the acoustic model itself cannot distinguish
the relevant speech. If that happens, partial/OOG speech should carry truthful
transcripts rather than being mislabeled as blank silence.

## Implementation plan

### Step 1: lock in grammar semantics (implemented experimentally)

Add tests around prefix derivation before editing decoder behavior:

- `color` is a designated incomplete prefix.
- `set the lights to` is a designated incomplete prefix.
- `colo` is not included because it is not a word-boundary prefix.
- every derived prefix is absent from the accepted phrase set.
- `time` and `pause` are not incomplete because they are accepted commands.
- `OPTIONB_GRAMMAR.accepts("color")` remains `None`.

Completion criterion: the prefix set is generated from the active grammar,
contains only proper word-boundary prefixes, and does not change the 129
accepted Option B alternatives.

### Step 2: add decoder diagnostics without changing acceptance

Modify final beam inspection to retain the best terminal phrase/raw score and
the best designated incomplete prefix/raw score. Add defaulted result fields
and populate them. Leave the new gate disabled for this step.

Completion criterion: existing decoder tests pass unchanged except for
intentional assertions on the new diagnostic fields.

### Step 3: add the parameterized rejection gate

Add `required_command_margin` (final name may follow repository conventions)
and thread it through batch decode, evaluation, and streaming configuration.
Record the rejection reason. Keep the existing confidence threshold as an
independent second gate.

Completion criterion: enabling the gate rejects the deterministic broken case;
disabling it reproduces baseline behavior.

### Step 4: add deterministic posterior regressions

Construct log-posteriors directly; do not depend on a WAV file. The fixture
should make unconstrained greedy collapse to `color`, provide weaker alternate
character support for `time`, then append high-confidence blanks. Parameterize:

- frame counts around 44, 70, 151, and 251;
- beam widths 25, 50, and 1000;
- enabled and disabled incomplete-prefix gate.

Assertions when enabled:

- result is `no_match` for every trailing-blank duration;
- `text == "color"`;
- `rejection_reason == "incomplete_prefix"`;
- `incomplete_prefix == "color"`;
- the raw-score gap is invariant within a small numerical tolerance;
- no model checkpoint or audio fixture is loaded.

Also retain positive synthetic tests for `color red`, `color blue`, and
`color green`.

Completion criterion: the regression fails on the old behavior at the known
long-window cases and passes with the gate enabled.

### Step 5: calibrate against the actual cluster data

Do not assume local WAV paths, `.scratch` files, or this machine's `out/`
contents exist. On the cluster, begin with discovery:

```bash
git status --short
make -n optionb-train
rg -n "OPTIONB_MANIFEST|OPTIONB_OUT_DIR|optionb-train" Makefile
find out -maxdepth 4 -type f | sort | sed -n '1,200p'
```

The Makefile currently defaults `OPTIONB_MANIFEST` to
`out/conversions/v2/optionb/manifest.csv` and `OPTIONB_OUT_DIR` to
`out/vcm/optionb`, but environment overrides or cluster job wrappers may change
both. Use the expanded `make -n optionb-train` command, run metadata, and
checkpoint metadata to identify the manifest and dataset that actually trained
the checkpoint. Follow manifest-relative audio paths rather than inventing WAV
locations.

Do **not** run `optionb-train` for this task. Its resolved manifest is being
located to obtain representative evaluation audio, not to retrain the model.
Do not overwrite the training manifest or place generated probes into it.

Build an evaluation-only manifest or artifact under an appropriate `out/`
subdirectory. Preserve source split and speaker identity:

- use validation speakers/examples to select the margin;
- use a disjoint held-out split exactly once for the final comparison;
- never tune the margin on the held-out result;
- record source row identifiers and transformation metadata for reproducibility.

Prefer naturally recorded incomplete utterances if the cluster dataset has
them. If it only has complete commands, scalable synthetic incomplete probes
may be derived as follows:

1. Select a complete command whose transcript begins with a designated
   incomplete prefix.
2. Run the existing acoustic model and force-align the known complete
   transcript.
3. Map the last character of the prefix to its final aligned frame.
4. Crop near that word boundary, optionally varying a small grace period, and
   append several realistic trailing-silence durations.
5. Preserve the source row's split/speaker group.
6. Listen to or audit a stratified sample and discard failed alignments.
7. Save the generated probe path and exact crop/alignment metadata in the
   evaluation-only manifest.

This creates accented, microphone-realistic prefix speech while probing the
blank-duration bug. Because forced alignment can fail or choose a poor boundary,
quality checks are mandatory. Do not call a presumed WAV filename directly.

### Probe-manifest contract

Implemented by `vcm.optionb.incomplete_probes` (ticket 05); consumed by the
calibration tool (ticket 06). Fixed here so both can be built in parallel
against the same schema.

Probe WAVs and their manifest are written under an `out/vcm/...` directory
that is never the training manifest path (`OPTIONB_MANIFEST`); the tool
refuses to write there. Columns, one row per (source row, designated
prefix, trailing-silence bucket):

| Column | Meaning |
|---|---|
| `filename` | probe WAV filename |
| `path` | probe WAV path, relative to this manifest's own directory |
| `bucket` | always `"incomplete_prefix"` |
| `label` | always `""` -- these rows carry no accepted intent |
| `split` | `val` or `test`, inherited from the source row |
| `group_id` | source row's speaker id, for split/speaker-disjointness checks |
| `source_dataset` | always `"optionb_incomplete_probe"` |
| `source_filename` | the source Option B clip's `filename` |
| `source_path` | the source Option B clip's manifest `path` |
| `source_transcript` | the exact normalized text passed to `force_align` |
| `prefix` | the designated incomplete prefix this probe targets |
| `prefix_word_count` | `len(prefix.split())` |
| `char_overlap` | `bool`; always `False` for a written row -- recorded as an audit trail that the word-boundary check (not a bare substring check) gated selection |
| `grace_frames` | the grace value used for this crop |
| `crop_end_frame` | `min(last_char_frame + 1 + grace_frames, next_word_first_char_frame)` |
| `crop_end_sample` | `crop_end_frame * HOP_LENGTH` |
| `trailing_silence_s` | this probe's silence-bucket duration |
| `silence_source` | `"lead_in_room_tone"`, `"tail_room_tone"`, `"digital_zero"` (no candidate cleared the relative-quietness bar), or `"none"` (0.0s bucket) |
| `alignment_log_prob` | `ForcedAlignment.log_probability` for the source clip against `source_transcript` |
| `aligner_checkpoint` | path to the checkpoint used for forced alignment |
| `aligner_checkpoint_sha256` | sha256 of that checkpoint file |

Trailing-silence padding is **not** blind tiling of "samples before
`start_frame`", and existence of *a* candidate region is not sufficient
either (R2-1 then R2-5, tickets/05's Review Feedback and its re-review): CTC
spikes lag acoustic onset, so the frames immediately around `start_frame`/
`end_frame` are often already speech, and a same-level-as-speech "silence"
pad is an out-of-distribution shift on the whole utterance (`LogMelFeature
Extractor` normalizes per-utterance), not a neutral one. `find_quiet_window`
therefore searches **both** the lead-in (`[0, (start_frame - 20 frames) *
HOP_LENGTH)`) and the tail (`[(end_frame + 1 + 20 frames) * HOP_LENGTH,
len(waveform))`) for the minimum-RMS 100ms window, and accepts the winner
only if its RMS is at least 20dB below the aligned speech span's own RMS;
otherwise padding falls back to digital zero (`silence_source =
"digital_zero"`). Because most 1.5-2s Option B clips do not have a long
enough lead-in *or* tail to pass both the length and the 20dB bar, digital
zero remains a substantial share of real runs -- see ticket 05's Execution
Log for the measured `silence_source` distribution and ticket 06's
per-`silence_source` metrics breakdown for how residual digital-zero share
is surfaced in calibration reporting.

`generation_report.json` carries `failure_counts` keyed by exactly, and
always all four of: `force_align_error`, `no_gap` (no frame gap between the
prefix's last character and the next word's first character),
`crop_too_short` (crop shorter than 0.15s), and `missing_audio` --
zero-filled when a reason had no occurrences. `audit_sample.csv` uses the
same column schema, stratified to up to 3 distinct source rows per prefix,
each represented by its *largest* silence-bucket variant (crop plus padded
tail in one file) so a human listener hears the padding, not just the
cropped prefix.

For each margin candidate, report at least:

- incomplete-prefix false acceptance rate, overall and by prefix;
- accepted-command intent accuracy and slot exact match;
- false rejection rate for short commands and strict-prefix commands;
- results by trailing-silence-duration bucket;
- latency and peak memory versus baseline;
- count of alignment/probe-generation failures.

Completion criterion: the chosen value comes from validation data, removes or
materially reduces incomplete-prefix false accepts, and stays within an agreed
command-recall regression budget on the held-out data.

### Step 6: consider segment-local scoring only if residual failures remain

After the targeted gate is evaluated, inspect residual OOG and incomplete
false accepts. If `/ T` still creates unrelated acceptance errors, combine the
existing threshold with a calibrated segment-local diagnostic from
`segment_scorer.py`. Keep this as a separate experiment so its recall cost can
be measured independently.

Completion criterion: any additional gate has its own ablation, calibration,
and rollback switch. Do not bundle an unmeasured confidence-policy rewrite into
the targeted prefix fix.

## Required test matrix

At minimum, run or add coverage for:

| Area | Required check |
|---|---|
| Grammar | Derived prefixes are proper whole-word prefixes only |
| Grammar | Accepted phrase count and intent/slot mappings are unchanged |
| Grammar | Accepted strict prefixes (`time`, `pause`, and all others found programmatically) are never rejection competitors |
| Decoder | Synthetic `color` evidence cannot become `TIME` when the gate is enabled |
| Decoder | Result is stable under added leading/trailing blank frames |
| Decoder | Results are stable at beam widths 25, 50, and 1000 |
| Decoder | Disabling the gate preserves baseline behavior |
| Positive paths | Every accepted grammar phrase still resolves to its original intent and slots |
| Configuration | Evaluation and streaming receive the same explicit margin |
| Cluster evaluation | Validation tuning and held-out reporting use disjoint groups |
| Deployment | No checkpoint, ONNX model, alphabet, or feature contract changes |

Run focused tests first, then the repository's normal test/lint commands as
discovered from `pyproject.toml`, Makefile, and CI configuration. Do not assume
commands copied from an older session are still current.

## Decision log

- The issue is classified as decoder acceptance behavior, not CTC training
  behavior.
- The incomplete phrase remains invalid grammar; it is represented only as a
  rejection competitor.
- Raw same-window beam-score differences are preferred for the competitor gate
  because full-window mean confidence is sensitive to blank padding.
- Only designated whole-word prefixes participate; arbitrary nonterminal trie
  nodes do not.
- Existing accepted commands take precedence over their status as prefixes of
  longer commands.
- Cluster `out/` artifacts and the manifest resolved by `optionb-train` are the
  source for broad evaluation; local WAV availability is never assumed.
- Segment-local confidence remains a possible second-stage defense and must be
  independently calibrated before adoption.

## Handoff checklist

Before claiming completion, leave behind:

- code and unit tests for prefix derivation and decoder gating;
- the deterministic no-WAV regression fixture;
- the explicit runtime/evaluation margin and rejection diagnostics;
- an evaluation-only cluster manifest with provenance, outside the training
  manifest;
- baseline-versus-change metrics for validation and held-out sets;
- the selected margin and rationale;
- updated decoder contract documentation if `DecodeResult` changes;
- confirmation that training and exported model artifacts were untouched.
