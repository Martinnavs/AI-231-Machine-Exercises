# CTC + attention heads on the QuartzNet encoder

Status: **see [`CURRENT-MODEL.md`](CURRENT-MODEL.md) for the current model (the hybrid). This page is the heads design and the v2 results. v2 (rebuilt dataset, reproducible noise) scored on two seeds; the heads-A model of seed 1 is checked in as the
current best for streaming tests at the time (superseded by the hybrid); ticket 04 deferred, see `AI231-FIL50.md`.** The v1 sections further down (one seed, earlier dataset
version) are kept for the record and are superseded where they disagree.
Dates: 2026-10-02. Follow-up experiment (persona-padded training data): `docs/AI231-FIL50.md`. Design record: `feature-engineering/ctc-attention/tickets/00-RECAP.md` (tickets 01-04).

## v2 results (current): rebuilt dataset, noise and babble from the dataset itself

**What changed since v1.** The `airimonda/ai231-me2-voice-commands` dataset was rebuilt (train 10,733 / test 4,443 /
holdout 202 rows; commit `6947f130`) and gained a `synthetic_negatives` config (1,000 train + 250 test generated
out-of-scope clips: `noise_only`, `babble`, `reversed`, `truncated`, `near_silence`). `vcm.optionb.import_ai231` now
imports it: `noise_only` becomes the additive-noise pool (and the noisy-eval pool), `babble` the babble pool, all five
kinds train and test as rejection examples with empty transcripts. Training uses `--perturbation-plan table
--noise-source dataset --p-rir 0.7 --p-timestretch 0.25 --p-babble 0.15` (see "Reproducible training noise" below), so
nothing outside the public dataset is needed. The perturbed evaluation uses the test split's own 50 `noise_only` clips
plus a fixed-seed reverb. The v1 models and numbers are not comparable (different split, ESC-50 noise).

**Models** (`out/vcm/`): `v2-baseline` and `v2s1-baseline` (plain `quartznet5x3`), `v2-heads-A` and `v2s1-heads-A`
(`quartznet5x3-heads`, CTC + heads, weights 1.0 / 0.3 / 0.1). Seeds 0 and 1; seed 0 early-stopped with patience 10, seed 1
with patience 20 (baseline 55 / 64 epochs, heads-A 61 / 79). The OneCycle schedule is sized to `--max-epochs` (150), so
early stopping ends every run before the learning rate has annealed; this affects all models alike.

Test split, `exact` target rows (3,823), intent + slot accuracy %, CTC path unless stated. Perturbed = fixed-seed reverb +
dataset noise. Holdout = all 202 clips scored with the same models (186 commands, 16 out of scope); "real Filipino" is
the 84 clips of one speaker (202520785). False accepts are over the 226 speech-like test non-commands.

| model | test clean | test perturbed | holdout all | holdout real Filipino | false accepts |
|---|---|---|---|---|---|
| baseline CTC, seed 0 | 88.3 | 74.5 | 54.8 | 16.7 | 1.8% |
| baseline CTC, seed 1 | 87.3 | 73.2 | 53.8 | 7.1 | 2.2% |
| heads-A CTC, seed 0 | 84.4 | 66.7 | 52.2 | 16.7 | 0.9% |
| heads-A CTC, seed 1 | 87.5 | 76.1 | 52.2 | 6.0 | 0.4% |
| heads-A classifier, T=0.9, seed 0 | 93.6 | 82.9 | 64.5 | 27.4 | 4.4% |
| heads-A classifier, T=0.9, seed 1 | 95.4 | 88.9 | 58.1 | 8.3 | 4.4% |

Two-seed means: clean 87.8 (baseline CTC) / 86.0 (heads-A CTC) / 94.5 (heads-A classifier); perturbed 73.9 / 71.4 / 85.9.

- **CTC + heads is level with the baseline.** Seed 0 failed Gate A (clean 84.4 vs 88.3); seed 1 passes it (87.5 vs 87.3
  clean, 76.1 vs 73.2 perturbed, fewer false accepts). Seed-to-seed swings (about 3 points clean, up to 9 perturbed) are
  as large as the seed-0 gap, so two seeds support "no harm shown", not "no harm".
- **The classifier is the strongest model on the test split** (94.5% clean, 85.9% perturbed, two-seed mean) but
  accepts more non-commands (4.4% against 0.4-2.2% for CTC). Plain argmax (no rejection) is 96.8% clean on seed 0 with 45 of
  226 non-commands accepted. Operating curve, seed 0: T=0.8 95.0% / 12 false accepts, T=0.9 93.6% / 10, T=0.95 91.9% / 9,
  T=0.98 88.9% / 6, T=0.9878 86.7% / 4 (clean). The v1 threshold rule (match CTC's zero false accepts on a 121-clip val
  set) was too strict for this data.
- **Rejection of synthetic negatives (seed 0, 50 clips per kind, clean):** `noise_only`, `near_silence`, `reversed`
  0/50 for both models; `babble` 0/50; `truncated` 2/50 for both (the weak spot). Perturbed: `babble` 1/50, `truncated` 1/50
  baseline and 0/50 heads-A.
- **Accent gap.** Real Filipino speech is 5.6% of train command clips (590 of 10,463) and 4.3% of test (189 of 4,367),
  mostly one speaker per split (397+61 of 590 in train; 179 of 189 in test), and 45% of the holdout (84 of 186, one
  speaker). Accuracy on those holdout clips is 6-27% for every ai231-trained model and seed; synthetic holdout voices score
  83-96%. The `fil50` models trained on internal Filipino-accented synthetic data score 68-71% on the same 84 clips and
  97-98% on test, but their training data may overlap the test voices, so treat that as indicative only. Synthetic voices
  carry no accent label in the dataset, so Filipino-accented synthetic speech cannot be counted.
- **Not comparable:** `quartznet5x3-baseline-ai231-rir-135m` (v1, trained on the earlier split) scores 94.9% on the new test
  and 100% on the 100 synthetic holdout clips; its training voices probably overlap the new test and holdout.
- **Single speaker, 84 clips:** holdout real-Filipino accuracy is too noisy to rank models.

**Current best (checked in):** `v2s1-heads-A` (seed 1): CTC level with the baseline, best classifier (95.4% clean, 88.9%
perturbed at T=0.9, 58.1% holdout all), and the CTC path the streaming tests need. Chosen on one seed's test split, so
read it as "good enough to test streaming with", not as a proven winner. Streaming behaviour has not been measured.

## v1 result in one paragraph (superseded)

Adding an attention-pooled intent head and six slot heads to the stride-2 QuartzNet and training them jointly with CTC
does not hurt the CTC path: on unperturbed ai231 test audio the joint model's CTC decode gets 95.6% intent+slot against
95.0% for the plain CTC baseline trained the same way (a 19-clip difference, inside what one seed can produce). The new
heads themselves, read directly without any beam search, get 98.0% intent and 97.7% intent+slot, 2.7 points above that
CTC baseline, and the true intent is in their top 3 for 99.8% of clips. The cost is rejection: at the threshold implied by
the CTC path's false-accept budget the classifier accepts 8 of 47 babble clips where CTC accepts none, and it cannot match
CTC's zero false accepts without giving up about 13 points of accuracy. A classifier-only model (run B) is about as accurate as
the CTC baseline (95.5% / 94.9% against 95.0% / 95.0%), though with 6 of 47 babble clips falsely accepted. Perturbed audio is a separate story: models trained on
the ai231 data fall to 77-82% under the fixed-seed reverb and noise gate where the old production model keeps 95.8%.

## What was compared

| name | what | trained on | path |
|---|---|---|---|
| `old` | production QuartzNet stride 2 (`quartznet5x3-s2-fil50-ambient-rir-135m`) | old fil50-ambient data | CTC |
| `base` | plain `quartznet5x3`, **the Gate A baseline** | ai231 | CTC |
| `A` | `quartznet5x3-heads`, CTC + heads, loss weights ctc 1.0 / intent 0.3 / slot 0.1 | ai231 | CTC and classifier |
| `B` | same model, `--ctc-weight 0` (classifier only; its CTC head is untrained) | ai231 | classifier |

`old` is a cross-dataset reference only: it never saw ai231, and its earlier 3468/3416 figures were measured on a
different test set and are void. `base`, `A` and `B` share the recipe: ai231 manifest, seed 0, `--p-rir 0.7
--p-timestretch 0.25`, 135-minute cap with early stopping (patience 10). They stopped at 59, 76 and 41 epochs.

## Evaluation protocol

- **Clips:** ai231 `test` split, `exact` target rows (3,601; the Option B wording) and the 47 babble rows. Paraphrase
  (`close`) rows are trained on but never scored.
- **CTC path:** whole clip, grammar-constrained beam 50, **threshold -0.1, incomplete-prefix margin 4.0**, as fixed in
  `.scratch/ctc-attention-baseline-note.md`. Nothing is tuned on this data. A row counts as correct if the decoded intent
  equals the label (intent) and the decoded slot equals the manifest `slot_value`, case-insensitively (intent+slot).
- **Classifier:** accepts when the top-1 class is a real intent (not `unknown`/`silence`) and its softmax probability is
  at least the threshold. The threshold is the smallest one whose **clean val** false-accept rate over babble and
  ESC-50 noise rows is no higher than the CTC path's own clean-val rate for run A (17/339 = 5.0%). It is not re-swept on
  perturbed audio.
- **Unperturbed** audio is the headline. **Perturbed** is a separate robustness check: one deterministic RIR plus ESC-50
  noise per clip (seed 0, `vcm.noisy_eval`'s own perturbation, identical for every model).
- The ai231 manifest has no `background_noise` rows, which the perturbation needs as its noise pool. The eval manifest
  adds the ESC-50 val/test noise rows from the old manifest, so the 324 test noise rows are also scored as an extra
  "ESC-50 silence" false-accept probe. They are not part of the ai231 test.
- The scorer reproduces the independent figures in the baseline note for `old` (97.11% / 97.03%, 0/47).

## Results: unperturbed audio (headline)

| model | path | intent | intent+slot | babble FA | ESC-50 silence FA |
|---|---|---|---|---|---|
| old | CTC | 97.1% (3497/3601) | 97.0% (3494/3601) | 0/47 | 0/324 |
| **base** | CTC | **95.0%** (3422/3601) | **95.0%** (3422/3601) | 0/47 | 6/324 |
| A | CTC | 95.6% (3443/3601) | 95.6% (3441/3601) | 0/47 | 9/324 |
| A | classifier, thr 0.7943 | 98.0% (3528/3601) | 97.7% (3517/3601) | 8/47 | 14/324 |
| B | classifier, thr 0.7640 | 95.5% (3440/3601) | 94.9% (3419/3601) | 6/47 | 28/324 |

Classifier with no rejection (plain argmax): A 98.9% / 98.6%, B 97.4% / 96.7% (intent / intent+slot).

Top-k coverage and agreement (A): true intent is the classifier's top-1 for 98.9% of clips and in its top 3 for 99.8%.
Among the 3,443 clips A's CTC path gets right, the classifier's top 3 contains the right intent for **3,443 of 3,443**.
Where the CTC path accepts, the classifier's top-1 equals its intent 99.5% of the time (3,430/3,447). B: top-1 97.4%,
top-3 99.2%.

Confusable pairs, unperturbed (true intent -> where it went), classifier A: TIME 100 right, 1 other; TIMER 415 right,
7 rejected; PAUSE 106 right, 4 rejected; STOP 77 right, 3 other, 3 rejected; LIGHT_ON 82 right, 3 as LIGHT_OFF, 7 rejected.
The CTC path of A rejects 9 TIME and 18 TIMER clips where the classifier rejects 0 and 7. Full tables: `report.md`.

## Results: perturbed audio (robustness check)

| model | path | intent | intent+slot | babble FA | ESC-50 silence FA |
|---|---|---|---|---|---|
| old | CTC | 95.8% | 95.6% | 0/47 | 0/324 |
| base | CTC | 77.5% | 77.1% | 0/47 | 3/324 |
| A | CTC | 81.6% | 81.4% | 0/47 | 6/324 |
| A | classifier | 90.4% | 88.6% | 12/47 | 14/324 |
| B | classifier | 84.1% | 82.3% | 4/47 | 20/324 |

Classifier top-3 coverage (A) falls to 97.3% on perturbed audio, and the classifier's top 3 contains the intent for
99.8% of the clips the CTC path gets right.

## Pre-registered rules (00-RECAP) and outcomes

Gate A, re-expressed on ai231 because the original counts (3468 / 3416 / 3/255) belonged to the old test set. Tolerance:
0.5 pp of the 3,601 target clips (18 clips); the babble allowance is "3/255 scaled to 47 babble clips, rounded up, and never
worse than the baseline's own".

| check | baseline | joint (A) | requirement | result |
|---|---|---|---|---|
| clean intent+slot | 3422 | 3441 | >= 3403 | pass |
| perturbed intent+slot | 2777 | 2930 | >= 2758 | pass |
| perturbed babble FA | 0 | 0 | <= 1 | pass |

**Gate A: pass.** **Gate B** (classifier top-3 holds the true intent on >= 99.5% of clean test targets the CTC path
gets right): 3,443 of 3,443, **pass**, so ticket 04 is allowed to start. Competitiveness (reported, not gating) is the
table above: the classifier is more accurate at the budget-matched threshold but less selective, and a classifier-only
model matches the CTC baseline's accuracy but not its selectivity.

## What was and was not shown

Shown, on one seed and this test set:
- Joint training with the heads does not degrade the CTC path (Gate A), and the classifier reads the grammar's intent and
  slots accurately on unperturbed audio, with the right intent almost always in its top 3.
- A paired view of A's CTC path against `base` on the same clips: A is right and `base` wrong on 65 clips, `base` right
  and A wrong on 46. That is 19 net clips; it does not show that joint training improves CTC.

Not shown:
- **Rejection.** The classifier is much less selective than the CTC path. Fitting its threshold on the test split to
  zero babble/noise false accepts (informational, not a result) leaves 85.1% intent / 84.9% intent+slot; allowing 3 false
  accepts gives 94.8% / 94.5%. The 47 babble clips are also too few for tight intervals.
- **Streaming.** All numbers are whole-clip. The streaming policy decodes growing, unpadded windows and accepts only at
  an endpoint; nothing here covers that.
- **A second seed, or the holdout split** (167 exact clips, not scored).
- **Noise robustness of ai231-trained models.** The perturbed gate takes them from about 95% to 77-82% while `old` keeps
  95.8%. The ai231 manifest has no ESC-50 rows, so the augmenter's additive-noise pool was empty and no additive noise was
  applied online. That is a training-data difference, not a heads effect (`base` falls further than `A`).

## Transfer to the old (internal) test set

Question: if models trained only on the public ai231 data meet the project's own earlier data, do they hold up?
Test: the `test` split of `optionb-v3-vcmx-fil50` (the old gate's set): 3,494 target clips, of which 1,798 are synthetic TTS
(`optionb`), 1,567 synthetic Filipino-accented personas (`fil50_persona`) and **129 real recordings** (`vcm_balanced`),
plus 255 babble clips (211 of them real recordings) and 324 noise clips. This is a pure transfer test: the CTC setting is
the same (-0.1, margin 4.0) and the classifier thresholds are the ones already chosen on ai231 val (A 0.7943, B 0.7640);
nothing is tuned on this data. `old` was trained on this data's train split, so it is an in-distribution reference, not a
competitor. Intent+slot truth is parsed from each row's own transcript (0 rows unparseable).

| model | path | intent | intent+slot | babble FA | ESC-50 silence FA |
|---|---|---|---|---|---|
| old (in-distribution) | CTC | 97.4% (3402/3494) | 97.4% (3402/3494) | 0/255 | 0/324 |
| base (ai231) | CTC | 89.6% (3129/3494) | 89.5% (3126/3494) | 0/255 | 6/324 |
| A (ai231) | CTC | 89.9% (3140/3494) | 89.8% (3138/3494) | 0/255 | 9/324 |
| A | classifier | 96.1% (3358/3494) | 94.9% (3316/3494) | 46/255 | 14/324 |
| B | classifier | 91.1% (3182/3494) | 89.9% (3142/3494) | 40/255 | 28/324 |

Intent+slot by source, unperturbed:

| model | path | fil50_persona (1,567) | optionb (1,798) | vcm_balanced, real (129) |
|---|---|---|---|---|
| old | CTC | 97.5% | 97.3% | 96.9% |
| base | CTC | 82.5% | 95.5% | 89.9% |
| A | CTC | 82.7% | 95.8% | 92.2% |
| A | classifier | 90.2% | 99.0% | 94.6% |
| B | classifier | 81.8% | 97.3% | 86.0% |

Perturbed (seed 0) intent+slot: old 95.4%, base 70.3%, A CTC 73.2%, A classifier 83.8%, B 74.9%; on the 129 real
recordings: old 86.0%, base 42.6%, A CTC 47.3%, A classifier 52.7%.

What this says:
- The ai231-trained models do not match the in-distribution model on this data: about 90% for CTC against 97.4%. The
  gap is concentrated in the Filipino-accented synthetic personas (82.5-82.7% for CTC); the plain TTS clips stay near 96%.
- On the 129 real recordings the ai231 CTC models get 89.9% (base) and 92.2% (A). With only 129 clips the 95% intervals
  are about +/-5 points (base 83.5-94.0%, A 86.3-95.7%, A classifier 89.2-97.3%), so base and A cannot be told apart and
  this is a rough sense, not a measurement.
- Joint training does not hurt here either (A's CTC 89.8% vs base 89.5%). The classifier heads transfer better than the
  CTC decode on every source, but they accept 46 of 255 babble clips (18%), including real babble, where CTC accepts none.
- The perturbed numbers are poor for every ai231-trained model, and worst on real recordings (43-53%). The old model's
  training included ESC-50 noise; ai231 has none, so noise robustness is not available from the public data alone.

## Reproducible training noise (built; used by the v2 runs)

The ai231 manifest has no noise rows, and the ESC-50 pool the earlier models trained on cannot be rebuilt from this repo
(its fetch/chunk scripts are untracked and need a Kaggle login). Two options make training noise reproducible from the public
dataset and a seed alone:

- `--noise-source dataset`: the additive-noise pool is the manifest's own noise-only rows plus 240 deterministic synthetic
  clips (white / pink / brown noise from `wakeword.generate_silence.colored_noise`, half with a 1-4 Hz amplitude wobble;
  `vcm/noise_pool.py`). `--p-babble P` additionally mixes in one of the train split's own `babble` clips (people talking)
  at 12-25 dB SNR, quieter than the 5-25 dB used for noise so the target words stay intelligible. Pools only come from the
  train split.
- `--perturbation-plan table` (default `online` = the old per-clip coin flips): every training clip gets a recipe per
  epoch (stretch factor, reverb, noise clip + SNR, babble clip + SNR), a pure function of `(seed, epoch, class)`
  (`vcm/perturbation_plan.py`). It does not depend on DataLoader workers, shuffle order or thread count: two runs with the
  same seed, with 0 or 2 workers, write byte-identical plans. Recipes are stratified by intent+slot class (31 classes,
  plus `unknown` and `silence`): each class gets an exact quota of each perturbation (`p * n`, randomised rounding), the
  stretch factor and SNRs are spread evenly over the clips that receive them, and RIR / noise / babble clips are dealt from
  a shuffled deck so pool entries are used evenly. On the real train split: 9,310 clips, 33 classes, no quota violations,
  25.0% stretched / 70.0% reverb / 50.0% noise / 15.0% babble overall. `--dump-plan` writes every epoch's recipes
  (`<out>/metadata/perturbation_plan/epoch_NNN.csv.gz`) and a per-class summary.

**Grouping key changed (ai231-fil50 work).** The table originally grouped clips by intent+slot class (31 classes plus
`unknown` and `silence`). On the ai231-fil50 train split that left per-variation shares uneven (stretch 18-31%, reverb
63-79%, noise 43-58%, babble 9-21%), and persona clips could get a different share than real ones. `perturbation_plan.class_key`
now returns `label | slot_value | variation | source_dataset` for manifests that have a `variation` column (ai231 and later),
which gives 24-26% / 69-71% / 50-51% / 14-16% per variation with real and persona clips matching within a point. Manifests without a
`variation` column keep the intent+slot key, so runs on them reproduce as before. The earlier v2 runs (`v2*`, trained on
ai231 v2) used the intent+slot key; re-running them now yields a different (equally valid) table. Test:
`test_93_variations_two_sources_get_the_same_shares_real_and_persona`.

In table mode the RIR pool is built once from the seed (the online path builds it lazily per worker from a generator state
that depends on call order). SpecAugment is not part of the table: it still uses the augmenter's generator, so spectrogram
masks are reproducible only in distribution.

In v2 the pools come from the dataset's own `synthetic_negatives` (noise_only, babble) and the v2 perturbed evaluation uses
the test split's own `noise_only` clips, so v2 numbers are reproducible from the public data. The v1 perturbed numbers used
ESC-50 rows from the project's own manifests and are not.

## Defects found on the way (fixed; v1 runs A and B were not retrained)

- v2: `apply_noise` returned NaN when the noise window was digitally silent (some babble clips open with silence;
  torchaudio's `add_noise` divides by the noise power). About 6 batches per epoch were skipped in the first v2 runs; those
  runs were discarded and retrained after the fix (`apply_noise` now leaves the audio unchanged, regression test in
  `tests/test_vcm_augment.py`).
- v2: `resolve_transcript` now returns an empty target for `negative_*` sources (the synthetic negatives).

- The slot comparison was case-sensitive in two places: the scorer (manifest `Red` vs head `red`) and the training labels
  for paraphrase rows. The scorer is fixed. For training, 320 COLOR and CREATE_REMINDER paraphrase rows (of 614) had no
  slot label in runs A and B (294 instead of 614 labelled). Their test slot accuracy is still 99.3-99.7%, but a retrain
  would use all 614.
- `apply_timestretch` (ported for `--p-timestretch`) crashed on 1-D waveforms and took up to 5 s per clip at awkward
  factors; fixed with tests (ticket 02 log). The same fix belongs in the `me2-iteration3` worktree.

## Recommendation

Use `v2s1-heads-A` to test streaming. Its CTC path is level with the plain baseline and its classifier is the most accurate
model on the test split, but the classifier accepts more non-commands (4.4% at T=0.9), so it should not replace the CTC
path for rejection on its own. Ticket 04 (classifier as a pre-selector for the beam search) is the way to combine the two
and should be judged on streaming replay, not whole clips; it has not been started. Before drawing conclusions about
heads vs CTC, run more seeds (two are not enough) and, if accent coverage matters, add Filipino-accented training data
(the fil50 persona export; provenance of its 17 reference voices is still open).

## Run the checked-in model in streaming

`out/vcm/v2s1-heads-A/` is a normal run directory (`export/vcm_model.{fp32,int8}.onnx`, `metadata/eval_report.json` with the
checkpoint's preset and licence note), so the existing streaming CLI takes it as `--model`. The ONNX holds the CTC path only;
the intent and slot heads are not exported, so streaming uses the same grammar-constrained beam search as every other
QuartzNet model here. The settings below are the repo's current live ones (`make app-pipeline-live`): the `endpointed`
policy, threshold -0.1, incomplete-prefix margin 4.0, hold 200 ms, wake word gate.

```bash
# live microphone, wake word then command, answers when the command ends
uv run python -m me2_voicegen.vcm.streaming \
  --model out/vcm/v2s1-heads-A --backend onnx --onnx-variant int8 \
  --grammar optionb --threshold=-0.1 --required-command-margin 4.0 --beam-width 50 \
  --gate wakeword --policy endpointed --gate-period 3 --hold-ms 200 --stable-strides 1 \
  --wakeword-model out/wakeword-sesame-ambient-rir-45m --wakeword-backend onnx --log-periods \
  --source mic

# same through the app pipeline (pipes events to app.forward)
make app-pipeline-live APP_LIVE_MODEL=out/vcm/v2s1-heads-A

# deterministic file replay without the wake word (a clip that contains only the command)
uv run python -m me2_voicegen.vcm.streaming \
  --model out/vcm/v2s1-heads-A --backend onnx --onnx-variant int8 \
  --grammar optionb --threshold=-0.1 --required-command-margin 4.0 --beam-width 50 \
  --gate none --source path/to/command.wav
```

What has and has not been checked: the file-replay command (last one) was run on this model; 8 of 8 synthetic holdout
commands (one per intent) fired the right intent within 0.5-0.75 s, and a real-speaker TIMER clip fired nothing, in line with
the holdout results above. The wake word gate, the live microphone and the `make` target were **not** run for this model, and
latency, false actions on ambient audio and real-time factor on the target hardware are unmeasured. The threshold and margin
are the v1 production values, not re-tuned for this model. Use `--onnx-variant fp32` to rule out quantisation effects.

## Commands

```bash
D=out/conversions/v2/ai231-me2-voice-commands
# eval manifest (exact rows + babble + ESC-50 noise rows for the perturbation)
uv run python -m me2_voicegen.vcm.semantic_eval build-manifest --ai231 $D/manifest.csv \
  --noise out/conversions/v2/optionb-v3-vcmx-fil50/manifest.csv --out $D/manifest.eval-exact.csv

# per model, per condition: score in shards (CPU; ~3 min per condition with 4 shards)
for k in 0 1 2 3; do OMP_NUM_THREADS=1 uv run python -m me2_voicegen.vcm.semantic_eval score \
  --manifest $D/manifest.eval-exact.csv --checkpoint <run>/checkpoints/checkpoint.pt --out-dir <out>/<name> \
  --split test [--noisy-seed 0] --shard $k --n-shards 4 [--no-ctc] & done; wait   # also --split val (clean) for A and B

uv run python -m me2_voicegen.vcm.semantic_eval report --run old=<out>/old --run base=<out>/base \
  --run A=<out>/A --run B=<out>/B --budget-run A --baseline base --joint A --out <out>/report

# transfer to the old test set: same scoring on out/conversions/v2/optionb-v3-vcmx-fil50/manifest.csv (no val, no
# eval-manifest build; its own background_noise rows serve the perturbation), thresholds fixed from the ai231 report
uv run python -m me2_voicegen.vcm.semantic_eval report --run old=<oldtest>/old --run base=<oldtest>/base \
  --run A=<oldtest>/A --run B=<oldtest>/B --cls-threshold A=0.7943272590637208 --cls-threshold B=0.7640023827552797 \
  --baseline base --joint A --out <oldtest>/report
```

Runs: `out/vcm/quartznet5x3-baseline-ai231-rir-135m` (base), `quartznet5x3-heads-ai231-rir-135m` (A),
`quartznet5x3-heads-only-ai231-rir-135m` (B). Training: `vcm.train --manifest $D/manifest.csv --preset <preset>
--max-minutes 135 --seed 0 --p-rir 0.7 --p-timestretch 0.25` (B adds `--ctc-weight 0`). Outputs:
`out/vcm/quartznet-ctc-compare/ctc-attention/{report.md,report.json,<name>/*.jsonl}` and, for the transfer test,
`out/vcm/quartznet-ctc-compare/ctc-attention-oldtest/`.

v2 commands (all from the public dataset; `$R` = the downloaded dataset folder with `data/` and `synthetic_negatives/`):

```bash
uv run python -m me2_voicegen.vcm.optionb.import_ai231 --src $R --out out/conversions/v2/ai231-v2
uv run python -m me2_voicegen.vcm.train --manifest out/conversions/v2/ai231-v2/manifest.csv --preset quartznet5x3-heads \
  --seed 1 --patience 20 --max-minutes 135 --p-rir 0.7 --p-timestretch 0.25 --noise-source dataset --p-babble 0.15 \
  --perturbation-plan table --dump-plan          # baseline: --preset quartznet5x3
uv run python -m me2_voicegen.vcm.semantic_eval build-manifest --ai231 $D/manifest.csv --out $D/manifest.eval-exact.csv   # no --noise
# then score / report exactly as above; holdout: --split holdout on the full manifest.csv
```
